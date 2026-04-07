"""
Converts pandas DataFrames to Hugging Face datasets with train/val/test splits.
Supports material-based splitting to prevent data leakage.

Example for 100k dataset:
- test_size=0.2, valid_size=0.2: Train=60k, Val=20k, Test=20k
- test_size=0.0, valid_size=0.2: Train=80k, Val=20k, Test=0k
- test_size=0.0, valid_size=0.0: Train=100k (all data in train set)
"""

import argparse
import os
import sys
import pandas as pd

from datasets import Dataset, DatasetDict
from huggingface_hub import login

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import load_api_keys, assign_split_labels

# Default configuration
DEFAULT_TEST_SIZE = 0.0
DEFAULT_VALID_SIZE = 0.0
HF_KEY_PATH = 'API_keys.jsonc'


def _normalize_object_columns_for_hf(df):
    """Keep list-like columns intact and normalize numpy array cells to Python lists."""
    object_columns = df.select_dtypes(include=["object"]).columns
    for col in object_columns:
        sample = df[col].dropna().head(32)
        if sample.empty:
            continue

        if sample.map(lambda x: hasattr(x, "tolist") and not isinstance(x, list)).any():
            df[col] = df[col].map(lambda x: x.tolist() if hasattr(x, "tolist") and not isinstance(x, list) else x)

    return df


def _create_datasetdict_from_split_column(df):
    """Build DatasetDict from an existing Split column."""
    print("Splitting dataset according to the 'Split' column")

    split_series = df["Split"].astype(str).str.lower()
    available_splits = set(split_series.unique())
    splits = {}

    if "train" in available_splits:
        df_train = df[split_series == "train"].drop(columns=["Split"]).reset_index(drop=True)
        splits["train"] = Dataset.from_pandas(df_train)

    if "val" in available_splits:
        df_valid = df[split_series == "val"].drop(columns=["Split"]).reset_index(drop=True)
        splits["validation"] = Dataset.from_pandas(df_valid)

    if "test" in available_splits:
        df_test = df[split_series == "test"].drop(columns=["Split"]).reset_index(drop=True)
        splits["test"] = Dataset.from_pandas(df_test)

    if "train" not in splits:
        raise ValueError("Split column must contain at least one 'train' row")

    if len(splits) > 1:
        train_features = splits["train"].features
        for split_name in list(splits):
            if split_name != "train":
                splits[split_name] = splits[split_name].cast(train_features)

    return DatasetDict(splits)

def create_dataset_splits(df, test_size, valid_size, duplicates_mode=False, seed=1):
    """Create train/val/test splits from DataFrame."""
    if 'Split' in df.columns:
        return _create_datasetdict_from_split_column(df)

    if test_size == 0.0 and valid_size == 0.0:
        return DatasetDict({'train': Dataset.from_pandas(df.reset_index(drop=True))})

    df_with_split = df.copy()
    df_with_split['Split'] = assign_split_labels(
        dataframe=df_with_split,
        test_size=test_size,
        valid_size=valid_size,
        duplicates_mode=duplicates_mode,
        seed=seed,
    )
    return _create_datasetdict_from_split_column(df_with_split)


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
    parser.add_argument("--split_seed", type=int, default=1, help="Seed for deterministic split assignment")

    args = parser.parse_args()

    print(f"Loading Hugging Face API key from {HF_KEY_PATH}")
    api_keys = load_api_keys(HF_KEY_PATH)
    hf_token = api_keys.get("HF_key")

    print(f"Loading data from {args.input_parquet} as Parquet with zstd compression")
    df = pd.read_parquet(args.input_parquet)

    required_columns = {'CIF'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The input dataframe must contain the columns: {required_columns}")
    
    # Minimal dtype normalization for Arrow conversion while preserving list-array columns.
    df = _normalize_object_columns_for_hf(df)

    # Create dataset splits
    dataset = create_dataset_splits(df, args.test_size, args.valid_size, args.duplicates, args.split_seed)

    # Save locally if requested
    if args.save_local:
        # Create directory if it doesn't exist
        if '/' in args.output_parquet:
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