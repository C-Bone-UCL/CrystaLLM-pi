"""
Post-processing tool for cleaning and validating CIF strings in parquet datasets.
Outputs are fully standard-compliant CIF strings
"""

import pandas as pd
import argparse
import re
import os
import sys
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial 

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import (
    extract_space_group_symbol,
    replace_symmetry_operators,
    remove_atom_props_block
)


def validate_cif_numerics(cif: str) -> str:
    """Fix common numeric formatting issues in CIF strings"""
    # Fix multiple decimals in numbers like "3.6.69"
    cif = re.sub(r'(\d+\.\d+)\.(\d+)', r'\1\2', cif)
    # Fix missing/incomplete numeric values
    cif = re.sub(r'\s(\d+\.)\s', r' \g<1>0 ', cif)  # 0. -> 0.0
    cif = re.sub(r'\s(\.\d+)\s', r' 0\g<1> ', cif)  # .123 -> 0.123
    return cif


def postprocess(cif: str) -> str:
    """Process CIF string with enhanced validation"""
    original_cif = cif
    try:
        # Remove bracket characters from CIF content
        cif = cif.replace('[', '').replace(']', '')

        # Pre-clean numerical values
        cif = validate_cif_numerics(cif)
        
        # Handle multiple space group declarations
        space_group_symbol = extract_space_group_symbol(cif)
        
        # Only process if we found a valid space group
        if space_group_symbol:
            if space_group_symbol != "P 1":
                # Use more robust symmetry replacement
                cif = replace_symmetry_operators(cif, space_group_symbol)
            
            # Clean atom props with stricter regex
            cif = remove_atom_props_block(cif)
            
        # Post-clean formatting issues while preserving newlines
        # Replace multiple spaces with single space, but preserve newlines
        cif = re.sub(r'(?<=\S)[ \t]+(?=\S)', ' ', cif)
        
        # Reduce excess consecutive newlines (3+) to double newlines
        cif = re.sub(r'\n\s*\n+', '\n', cif)

    except Exception as e:
        cif = f"# WARNING: Processing failed for cif: {str(e)}\n" + original_cif
        print(f"Critical error processing cif: {str(e)}")

    return cif


def _process_generated(record: dict, column_name: str) -> str:
    """
    Top-level function for multiprocessing. 
    It must be defined at module scope so it can be pickled.
    """
    return postprocess(record[column_name])


def process_dataframe(df: pd.DataFrame, num_workers: int, column_name: str) -> pd.DataFrame:
    """
    Process CIF columns with validation, possibly in parallel, and display a progress bar.
    """
    # Check if column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    print(f"Processing {len(df)} records using {num_workers} worker(s)")
    
    if num_workers == 1:
        tqdm.pandas()  # Enable 'progress_apply'
        df[column_name] = df.progress_apply(
            lambda row: postprocess(row[column_name]), axis=1
        )
        print("Single-worker processing completed")
        return df
    else:
        # Multi-worker path using multiprocessing.Pool
        # Must pass a top-level function, not a closure or lambda.
        results_generated = []
        with Pool(num_workers) as pool:
            processor = partial(_process_generated, column_name=column_name)
            # .to_dict('records') produces a list of dictionaries
            # Each dictionary is {col1: val1, col2: val2, ...}
            for processed in tqdm(
                pool.imap(processor, df.to_dict('records')),
                total=len(df),
                desc=f"Processing {column_name}"
            ):
                results_generated.append(processed)
        df[column_name] = results_generated
        print(f"Multi-worker processing completed: {len(results_generated)} records")
        return df


def main():
    parser = argparse.ArgumentParser(
        description="Post-process CIFs in parquet DataFrame",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_parquet", type=str, 
                       help="Input parquet file path")
    parser.add_argument("--output_parquet", type=str,
                       help="Output parquet file path")
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of worker processes to use")
    parser.add_argument("--column_name", type=str, default="Generated CIF",
                       help="Column name containing CIF strings")
    
    args = parser.parse_args()

    # Read with pyarrow for better type preservation
    df = pd.read_parquet(args.input_parquet)
    
    # Process DataFrame (potentially in parallel)
    processed_df = process_dataframe(df, args.num_workers, column_name=args.column_name)
    
    # Write output preserving original schema
    processed_df.to_parquet(args.output_parquet, index=False)


if __name__ == "__main__":
    main()
