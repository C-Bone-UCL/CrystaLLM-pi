"""
Deduplicate CIF files or DataFrame tables by keeping the entry with the smallest volume per formula unit for each unique (formula, space group) pair. Supports filtering out entries with 'N/A', zero, or negative values in specified columns before deduplication.
"""

import argparse
import os
import pandas as pd
import pickle
import gzip
import warnings
import ast
from typing import Dict, Tuple, List, Optional, Union
import sys

from tqdm import tqdm
warnings.filterwarnings("ignore")

# import from one level above
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from _utils import extract_formula_nonreduced, extract_space_group_symbol, extract_volume, extract_formula_units

def process_cif_entry(idx: int, cif: str) -> Optional[Tuple[Tuple[str, str], int, float]]:
    """Extract formula, space group, and volume per formula unit from a CIF entry."""
    try:
        formula = extract_formula_nonreduced(cif)
        space_group = extract_space_group_symbol(cif)
        formula_units = extract_formula_units(cif)
        if formula_units == 0:
            formula_units = 1
        vpfu = extract_volume(cif) / formula_units
        return (formula, space_group), idx, vpfu
    except Exception:
        return None


def deduplicate_table(df: pd.DataFrame, num_workers: Optional[int] = None) -> pd.DataFrame:
    """Keep the entry with smallest volume per formula unit for each (formula, space_group) pair."""
    print("Deduplicating table CIF data...")
    
    lowest_vpfu: Dict[Tuple[str, str], Tuple[int, float]] = {}
    
    # Process CIF entries sequentially
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing CIF entries"):
        result = process_cif_entry(idx, row['CIF'])
        if result is not None:
            key, idx, vpfu = result
            
            if key not in lowest_vpfu:
                lowest_vpfu[key] = (idx, vpfu)
            else:
                existing_idx, existing_vpfu = lowest_vpfu[key]
                if vpfu < existing_vpfu:
                    lowest_vpfu[key] = (idx, vpfu)
    
    # Extract the indices to keep
    selected_indices = [idx for idx, _ in lowest_vpfu.values()]
    deduplicated_df = df.loc[selected_indices].reset_index(drop=True)
    return deduplicated_df

def load_data(file_path: str) -> pd.DataFrame:
    """Load DataFrame from .pkl, .pkl.gz, or .parquet file."""
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_path.endswith('.pkl.gz'):
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_path.endswith('.parquet'):
        data = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format for {file_path}. Supported formats are .pkl, .pkl.gz, and .parquet.")
    
    return data

def parse_list_argument(arg_value: str, arg_name: str) -> List[str]:
    """Parse and validate a list argument from command line."""
    try:
        parsed_list = ast.literal_eval(arg_value)
        if not isinstance(parsed_list, list):
            raise ValueError
        return parsed_list
    except Exception:
        raise ValueError(f"{arg_name} must be a valid list format")


def apply_filters(df: pd.DataFrame, filter_args: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Apply N/A, zero, and negative value filters to DataFrame based on specified columns."""
    print(f"\nAmount of entries before filtering: {df.shape[0]:,}")
    
    # Filter N/A values
    if filter_args['na_columns']:
        na_columns = parse_list_argument(filter_args['na_columns'], "filter_na_columns")
        print(f"Filtering out entries with 'N/A' or NaN values in columns: {na_columns}")
        for col in na_columns:
            if col in df.columns:
                df = df[df[col] != 'N/A']
                df = df[df[col] != 'n/a']
                df = df[df[col].notna()]
    
    # Filter zero values
    if filter_args['zero_columns']:
        zero_columns = parse_list_argument(filter_args['zero_columns'], "filter_zero_columns")
        print(f"Filtering out entries with 0 values in columns: {zero_columns}")
        for col in zero_columns:
            if col in df.columns:
                df = df[df[col] != 0]
    
    # Filter negative values
    if filter_args['negative_columns']:
        negative_columns = parse_list_argument(filter_args['negative_columns'], "filter_negative_columns")
        print(f"Filtering out entries with negative values in columns: {negative_columns}")
        for col in negative_columns:
            if col in df.columns:
                df = df[df[col] >= 0]
    
    print(f"Filtered down to: {df.shape[0]:,}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate CIF files or DataFrame tables.")
    parser.add_argument("--input_parquet", type=str,
                        help="Path to the input file. Supported formats: .pkl, .pkl.gz, .parquet.")
    parser.add_argument("--output_parquet", "-o", type=str, required=True,
                        help="Path to the output file. Output will be in .parquet format with zstd compression.")
    parser.add_argument("--property_columns", type=str, default=None,
                        help="List of property columns to keep in the output table. If multiple, separate by commas.")
    parser.add_argument("--filter_na_columns", type=str, default=None,
                        help="Comma-separated list of columns from which to filter out 'N/A' values.")
    parser.add_argument("--filter_zero_columns", type=str, default=None,
                        help="Comma-separated list of columns from which to filter out 0 values.")
    parser.add_argument("--filter_negative_columns", type=str, default=None,
                        help="Comma-separated list of columns from which to filter out negative values.")

    args = parser.parse_args()

    input_fname = args.input_parquet
    out_fname = args.output_parquet

    # Parse property columns if provided
    property_columns = None
    if args.property_columns is not None:
        property_columns = parse_list_argument(args.property_columns, "property_columns")
        print(f"Using property columns: {property_columns}")

    print(f"Loading data from {input_fname}...")
    if not os.path.exists(input_fname):
        raise FileNotFoundError(f"Input file {input_fname} does not exist.")


    df = load_data(input_fname)


    required_columns = ['Reduced Formula', 'CIF']
    # if there is the 'Material ID' column, we keep it as well
    if 'Material ID' in df.columns:
        required_columns.append('Material ID')
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Input table must contain the column '{col}'.")
        
    # Only keep required columns and property columns, if specified
    if property_columns:
        columns_to_keep = required_columns.copy()
        for col in property_columns:
            if col in df.columns:
                columns_to_keep.append(col)
        print(f"Keeping columns: {columns_to_keep}")
        df = df[columns_to_keep]

    # Apply filters using the helper function
    filter_args = {
        'na_columns': args.filter_na_columns,
        'zero_columns': args.filter_zero_columns,
        'negative_columns': args.filter_negative_columns
    }
    df = apply_filters(df, filter_args)

    print(f"\nStarting deduplication...")
    deduplicated_df = deduplicate_table(df)

    print(f"Number of entries after deduplication: {deduplicated_df.shape[0]:,}")

    print(f"\nSaving deduplicated data to {out_fname}...")
    
    # Ensure output directory exists (only if there's actually a directory path)
    output_dir = os.path.dirname(out_fname)
    if output_dir:  # Only create directory if it's not empty string
        os.makedirs(output_dir, exist_ok=True)
    
    deduplicated_df.to_parquet(out_fname)
    print("Process completed successfully.")