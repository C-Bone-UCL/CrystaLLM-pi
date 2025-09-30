"""
Converts pandas DataFrames to Hugging Face datasets with train/val/test splits.
Supports material-based splitting to prevent data leakage.

Example for 100k dataset:
- test_size=0.2, valid_size=0.2: Train=60k, Val=20k, Test=20k
- test_size=0.0, valid_size=0.2: Train=80k, Val=20k, Test=0k
- test_size=0.0, valid_size=0.0: Train=100k (all data in train set)
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys

from datasets import Dataset, DatasetDict
from huggingface_hub import login

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import load_api_keys

# Default configuration
DEFAULT_TEST_SIZE = 0.0
DEFAULT_VALID_SIZE = 0.0
HF_KEY_PATH = 'API_keys.jsonc'

def create_dataset_splits(df, test_size, valid_size, duplicates_mode=False):
    """Create train/val/test splits from DataFrame."""
    if duplicates_mode:
        if 'Material ID' not in df.columns:
            raise ValueError("When using duplicates mode, the 'Material ID' column must be present")
        
        print("Duplicates mode enabled: ensuring materials are not split across train/val/test sets")
        unique_materials = df['Material ID'].unique()
        np.random.seed(1)
        np.random.shuffle(unique_materials)
        
        n_materials = len(unique_materials)
        
        # Handle case where all data goes to train set
        if test_size == 0.0 and valid_size == 0.0:
            df_train = df.reset_index(drop=True)
            return DatasetDict({
                'train': Dataset.from_pandas(df_train)
            })
        
        if test_size > 0:
            # Calculate splits based on total materials
            n_test = int(n_materials * test_size)
            n_valid = int(n_materials * valid_size)
            n_train = n_materials - n_test - n_valid
        else:
            n_test = 0
            n_valid = int(n_materials * valid_size)
            n_train = n_materials - n_valid
        
        train_materials = unique_materials[:n_train]
        valid_materials = unique_materials[n_train:n_train + n_valid]
        test_materials = unique_materials[n_train + n_valid:] if test_size > 0 else []
        
        df_train = df[df['Material ID'].isin(train_materials)].reset_index(drop=True)
        df_valid = df[df['Material ID'].isin(valid_materials)].reset_index(drop=True)
        
        if test_size > 0:
            df_test = df[df['Material ID'].isin(test_materials)].reset_index(drop=True)
            return DatasetDict({
                'train': Dataset.from_pandas(df_train),
                'validation': Dataset.from_pandas(df_valid),
                'test': Dataset.from_pandas(df_test)
            })
        else:
            return DatasetDict({
                'train': Dataset.from_pandas(df_train),
                'validation': Dataset.from_pandas(df_valid)
            })
    
    elif 'Split' in df.columns:
        print("Splitting dataset according to the 'Split' column")
        
        # Check which splits are actually present
        available_splits = df['Split'].unique()
        
        result_dict = {}
        
        if 'train' in available_splits:
            df_train = df[df['Split'] == 'train'].drop(columns=['Split']).reset_index(drop=True)
            result_dict['train'] = Dataset.from_pandas(df_train)
            print('Train columns:', df_train.columns.tolist())
        
        if 'val' in available_splits:
            df_valid = df[df['Split'] == 'val'].drop(columns=['Split']).reset_index(drop=True)
            result_dict['validation'] = Dataset.from_pandas(df_valid)
        
        if 'test' in available_splits:
            df_test = df[df['Split'] == 'test'].drop(columns=['Split']).reset_index(drop=True)
            result_dict['test'] = Dataset.from_pandas(df_test)
        
        return DatasetDict(result_dict)
    
    else:
        # Random splitting
        dataset = Dataset.from_pandas(df)
        
        # Handle case where all data goes to train set
        if test_size == 0.0 and valid_size == 0.0:
            return DatasetDict({
                'train': dataset
            })
        
        # Handle case with test set but no validation set
        if valid_size == 0.0 and test_size > 0:
            dataset = dataset.train_test_split(test_size=test_size, seed=1)
            return DatasetDict({
                'train': dataset['train'],
                'test': dataset['test']
            })
        
        # Handle case with validation set but no test set
        if test_size == 0.0 and valid_size > 0:
            dataset = dataset.train_test_split(test_size=valid_size, seed=1)
            return DatasetDict({
                'train': dataset['train'],
                'validation': dataset['test']
            })
        
        # Handle case with both validation and test sets
        if test_size > 0 and valid_size > 0:
            dataset = dataset.train_test_split(test_size=valid_size + test_size, seed=1)
            test_valid = dataset['test'].train_test_split(test_size=test_size/(test_size + valid_size), seed=1)
            return DatasetDict({
                'train': dataset['train'],
                'validation': test_valid['train'],
                'test': test_valid['test']
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a pandas DataFrame to a Hugging Face dataset.")
    parser.add_argument("--input_parquet", type=str, help="Path to the input parquet file")
    parser.add_argument("--output_parquet", "-o", type=str, required=True, help="Path to the output dataset (local dir, the last part is used as dataset name on HF Hub)")
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE, help="Fraction for test split")
    parser.add_argument("--valid_size", type=float, default=DEFAULT_VALID_SIZE, help="Fraction for validation split")
    parser.add_argument("--save_local", action="store_true", help="Save the dataset locally")
    parser.add_argument("--save_hub", action="store_true", help="Save the dataset to Hugging Face Hub")
    parser.add_argument("--HF_username", type=str, default='c-bone', help="Hugging Face username")
    parser.add_argument("--duplicates", action="store_true", help="Prevent data leakage by splitting on Material ID (was done for thermo dataset as there were 2 entries per material)")

    args = parser.parse_args()

    print(f"Loading Hugging Face API key from {HF_KEY_PATH}")
    api_keys = load_api_keys(HF_KEY_PATH)
    hf_token = api_keys.get("HF_key")

    print(f"Loading data from {args.input_parquet} as Parquet with zstd compression")
    df = pd.read_parquet(args.input_parquet)

    required_columns = {'CIF'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The input dataframe must contain the columns: {required_columns}")
    
    # Convert DataFrame to strings for HF compatibility
    df = df.astype(str)

    # Create dataset splits
    dataset = create_dataset_splits(df, args.test_size, args.valid_size, args.duplicates)

    # Save locally if requested
    if args.save_local:
        # make sure the directory exists (wandle if no directory)
        if '/' in args.output_parquet:
            import os
            os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)

        dataset.save_to_disk(args.output_parquet)
        print(f"Dataset saved to {args.output_parquet}")

    # Save to Hub if requested
    if args.save_hub:
        login(hf_token)
        dataset_name = args.output_parquet.split('/')[-1]
        # remove any extension from dataset_name
        if '.' in dataset_name:
            dataset_name = dataset_name.split('.')[0]
        hf_repo_name = f"{args.HF_username}/{dataset_name}"
        dataset.push_to_hub(hf_repo_name)
        print(f"Dataset saved to Hugging Face Hub as {hf_repo_name}")