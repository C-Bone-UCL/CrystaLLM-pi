"""
flexible prompt construction for conditional or nonconditional models, supporting both automatic extraction from existing CIF data and manual 
specification of compositions and target properties.

You can do:
- Automatic prompt generation with 4 conditioning levels (minimal uncondinitional to spacegroup info)
- Manual prompt construction with custom compositions and property conditions
- Also supports making prompts for the raw method

Examples:

Automatic generation from DataFrame:
    python make_prompts.py --input_df data.parquet --automatic --level level_2 \\
        --condition_columns bandgap density --output_parquet prompts.parquet

From Hugging Face dataset:
    python make_prompts.py --HF_dataset "c-bone/mpdb-2prop_clean" --split test \\
        --automatic --level level_3 --output_parquet prompts.parquet

Manual construction:
    python make_prompts.py --manual --compositions "K4Sc4P8O28,Li2O" \\
        --condition_lists "1.2,2.5,3.1" --output_parquet prompts.parquet

Raw conditioning format:
    python make_prompts.py --manual --compositions "K4Sc4P8O28" \\
        --condition_lists "1.2,2.5" --raw --output_parquet prompts.parquet
"""

import argparse
import re
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
import commentjson
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import load_api_keys

# Configuration
API_KEY_PATH = 'API_keys.jsonc'

def load_hf_dataset(dataset_name, split):
    """Load dataset from Hugging Face with authentication."""

    data = load_api_keys(API_KEY_PATH)
    hf_key = str(data['HF_key'])
    
    login(hf_key)
    ds = load_dataset(dataset_name)
    
    if split == "all":
        available_splits = [ds[name] for name in ["train", "validation", "test"] if name in ds]
        if not available_splits:
            raise ValueError("No splits found to concatenate in this dataset.")
        combined_dataset = available_splits[0]
        for other_split in available_splits[1:]:
            combined_dataset = combined_dataset.concatenate(other_split)
        return combined_dataset.to_pandas()
    else:
        if split not in ds:
            raise ValueError(f"Split '{split}' not found in the dataset.")
        return ds[split].to_pandas()

def extract_composition_from_cif(cif_content):
    """Extract composition from CIF data_ line."""
    if pd.isna(cif_content):
        return None
    match = re.search(r'data_\[([^\]]+)\]', cif_content)
    return match.group(1) if match else None

def create_automatic_prompts(df, cif_column, level, condition_columns=None):
    """Generate prompts automatically from CIF data based on specified level."""
    df = df.copy()
    
    # Extract condition vector if specified
    if condition_columns:
        if 'Condition Vector' or 'condition_vector' in condition_columns:
            # take the string, and make a vector by removing the brackets and splitting by comma
            def parse_condition_vector(row):
                vec_str = row.get('Condition Vector') or row.get('condition_vector')
                # remove brackets
                if pd.isna(vec_str):
                    return "-100.0"
                vec_str = str(vec_str).replace('[', '').replace(']', '')
                return vec_str
            condition_vectors = df.apply(parse_condition_vector, axis=1)
            # remove Condition Vector from df
            df = df.drop(columns=['Condition Vector'], errors='ignore')
        else:
            def get_condition_value(row, col):
                if col not in df.columns or pd.isna(row[col]):
                    print(f"Warning: Column '{col}' not found or contains NaN. Using -100.0 as default.")
                    return -100.0
                try:
                    return float(row[col])
                except (ValueError, TypeError):
                    print(f"Warning: Non-numeric value in column '{col}'. Using -100.0 as default.")
                    return -100.0
            condition_vectors = [
                ", ".join(str(get_condition_value(row, col)) for col in condition_columns)
                for _, row in df.iterrows()
            ]
        df['condition_vector'] = condition_vectors
    
    # Define extraction functions based on level
    if level == "level_1": # minimal, unconditional
        def extract_prompt(cif_content):
            if pd.isna(cif_content):
                return ""
            return "<bos>\ndata_["
    
    elif level == "level_2": # composition only
        def extract_prompt(cif_content):
            if pd.isna(cif_content):
                return ""
            comp = extract_composition_from_cif(cif_content)
            if comp:
                return f"<bos>\ndata_[{comp}]\n"
            else:
                return "<bos>\ndata_["
    
    elif level == "level_3": # composition and atomic props
        def extract_prompt(cif_content):
            if pd.isna(cif_content):
                return ""
            parts = cif_content.split('\n_symmetry_space_group_name_H-M')
            prompt = parts[0] + '\n_symmetry_space_group_name_H-M' if len(parts) > 1 else parts[0]
            return "<bos>\n" + prompt
    
    elif level == "level_4": # up to spacegroup info
        def extract_prompt(cif_content):
            if pd.isna(cif_content):
                return ""
            parts = cif_content.split('\n_cell_length_a')
            return "<bos>\n" + parts[0]
    
    else:
        raise ValueError(f"Invalid level: {level}. Must be one of level_1, level_2, level_3, level_4")
    
    df['Prompt'] = df[cif_column].apply(extract_prompt)
    return df

def create_manual_prompts(compositions, condition_lists, raw_mode=False):
    """Generate prompts manually from compositions and condition lists."""
    # Handle compositions
    if not compositions or compositions == [None]:
        compositions = [None]
    # Build condition vectors from all condition lists
    condition_vector = []
    if condition_lists:
        # Get all combinations of conditions across lists
        def get_combinations(lists):
            if not lists:
                return ["None"]
            if len(lists) == 1:
                return lists[0]
            result = []
            for item in lists[0]:
                for combo in get_combinations(lists[1:]):
                    if combo == "None":
                        result.append(item)
                    else:
                        result.append(f"{item}, {combo}")
            return result
        condition_vector = get_combinations(condition_lists)
    else:
        condition_vector = ["None"]
    prompts = []
    conds = []
    
    for comp in compositions:
        for cond in condition_vector:
            if raw_mode:
                cond_str = str(cond).replace("'", "").replace(",", " ")
                if comp is None:
                    prompt = f"<bos>\n[{cond_str}]\ndata_["
                else:
                    prompt = f"<bos>\n[{cond_str}]\ndata_[{comp}]\n"
            else:
                if comp is None:
                    prompt = "<bos>\ndata_["
                else:
                    prompt = f"<bos>\ndata_[{comp}]\n"       
            prompts.append(prompt)
            conds.append(cond)
    
    return pd.DataFrame({'Prompt': prompts, 'condition_vector': conds})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct prompts from CIF data - supports both automatic and manual modes.")
    
    # Input source arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--input_df", type=str, help="Path to input DataFrame (parquet file)")
    input_group.add_argument("--HF_dataset", type=str, help="Hugging Face dataset identifier")
    parser.add_argument("--split", type=str, help="Dataset split to use (required for HF datasets)")
    
    # Mode selection (need to specify)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--automatic", action="store_true", help="Use automatic prompt generation from CIF data")
    mode_group.add_argument("--manual", action="store_true", help="Use manual prompt generation from compositions and conditions")
    
    # For both
    parser.add_argument("--output_parquet", required=True, help="Path to output parquet file")
    parser.add_argument("--raw", action="store_true", help="Use raw conditioning format")
    
    # Automatic mode arguments
    parser.add_argument("--cif_column", type=str, default="CIF", help="Column name containing CIF data")
    parser.add_argument("--level", type=str, choices=["level_1", "level_2", "level_3", "level_4"], 
                        help="Prompt level for automatic mode")
    parser.add_argument("--condition_columns", nargs='+', help="Columns to extract for condition vector, for example 2 columns: --condition_columns 'norm_pf_at_700K' 'gap'")
    parser.add_argument("--remove_ref_columns", action="store_true", help="Keep only Prompt and condition_vector columns in output")
    
    # Manual mode arguments
    parser.add_argument("--compositions", type=str, help="Comma-separated list of compositions")
    parser.add_argument("--condition_lists", nargs='+', help="Space-separated condition lists (each list is comma-separated, will make for ex list_1 times list_2 amount of combinations of conditions for each prompts)")
    
    args = parser.parse_args()
    
    
    # Generate prompts
    if args.automatic:
        # basic errors
        if args.HF_dataset and not args.split:
            parser.error("--split is required when using --HF_dataset")
        if not args.level:
            parser.error("--level is required for automatic mode")
        
        # Load data
        if args.input_df:
            df = pd.read_parquet(args.input_df)
            if args.cif_column not in df.columns:
                raise ValueError(f"Column '{args.cif_column}' not found in dataset")
            if args.split:
                # check if split column exists
                if 'split' not in df.columns:
                    print(f"Warning: 'split' column not found in {args.input_df}. Ignoring --split argument.")
                else:
                    df = df[df['split'] == args.split]
            print(f"Loaded {len(df)} records from {args.input_df}")
        else:
            df = load_hf_dataset(args.HF_dataset, args.split)
            print(f"Loaded {len(df)} records from {args.HF_dataset} ({args.split} split)")
        
        result_df = create_automatic_prompts(df, args.cif_column, args.level, args.condition_columns)
        print(f"\nGenerated automatic prompts at {args.level}")
        print(f"Using condition columns: {args.condition_columns}" if args.condition_columns else "No condition columns used")

        # If column name is 'Condition Vector', change it to condition_vector
        # result_df = result_df.rename(columns={'Condition Vector': 'condition_vector'})

        
    else:  # manual
        # Parse compositions
        if args.compositions:
            comp_str = re.sub(r'^\{|\}$', '', args.compositions.strip())
            compositions = [c.strip() for c in comp_str.split(',') if c.strip()]
        else:
            compositions = [None]
    
        # Parse condition lists
        condition_lists = []
        if args.condition_lists:
            for condition_list in args.condition_lists:
                conditions = [c.strip() for c in condition_list.split(',') if c.strip()]
                if conditions:
                    condition_lists.append(conditions)
        
        result_df = create_manual_prompts(compositions, condition_lists, args.raw)
        print(f"\nGenerated manual prompts for {len(compositions)} compositions and {len(condition_lists)} condition lists")

    # Remove reference columns if requested
    if args.remove_ref_columns:
        columns_to_keep = ['Prompt']
        if 'condition_vector' in result_df.columns:
            columns_to_keep.append('condition_vector')
        if 'Condition Vector' in result_df.columns:
            columns_to_keep.append('Condition Vector')
        if 'Material ID' in result_df.columns:
            columns_to_keep.append('Material ID')
        elif 'Material_ID' in result_df.columns:
            columns_to_keep.append('Material_ID')
        result_df = result_df[columns_to_keep]
        print(f"\nRemoved reference columns, keeping only: {columns_to_keep}")

    # # if condition_vector column exists, remove the brackets and spaces
    # if 'condition_vector' in result_df.columns:
    #     result_df['condition_vector'] = result_df['condition_vector'].apply(lambda x: str(x).replace('[', '').replace(']', ''))


    # print an example of the generated prompts
    print("\nFirst 3 rows:")
    print(result_df.head(3).to_string(index=False))
    # Save results
    # if there is a directory to create, create it
    if os.path.dirname(args.output_parquet):
        os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)

    result_df.to_parquet(args.output_parquet, index=False)
    print(f"\nSaved {len(result_df)} prompts to {args.output_parquet}")
