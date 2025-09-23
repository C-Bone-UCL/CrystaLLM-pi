"""
Calculate Validity, Uniqueness, and Novelty (VUN) metrics for generated CIFs.

"""

import argparse
import os
import sys
import warnings
from typing import Dict, List
import pandas as pd

# Add project root to path for internal imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import (
    extract_generated_formulas,
    build_generated_structures,
    load_and_process_generated_data,
    get_valid,
    get_unique,
    get_novelty,
    load_and_filter_training_data,
)

# Constants for structure matching
LTOL = 0.2
STOL = 0.3
ANGLE_TOL = 5.0

warnings.filterwarnings("ignore")

def compute_vun_metrics(df: pd.DataFrame) -> Dict[str, int]:
    """Compute Validity, Uniqueness, and Novelty counts from a dataframe."""
    return {
        "total": len(df),
        "valid": int(df['is_valid'].sum()),
        "unique": int(df['is_unique'].sum()),
        "novel": int(df['is_novel'].sum()),
    }

def save_vun_metrics(df, output_csv, condition_column_name, sort_metrics_by):
    """Save VUN metrics to a CSV file."""
    if not output_csv:
        return

    metrics_data = []
    
    # Helper to format a metrics row
    def get_metrics_row(label, metrics_dict):
        return {
            "CV": label,
            "Total gens": metrics_dict['total'],
            "Valid gens": metrics_dict['valid'],
            "V+unique": metrics_dict['unique'],
            "V+U+Novel": metrics_dict['novel'],
        }

    if sort_metrics_by in ["all", "both"]:
        print(f"Calculating overall VUN metrics...")
        metrics = compute_vun_metrics(df)
        metrics_data.append(get_metrics_row("all CVs", metrics))
    
    if sort_metrics_by in ["Condition Vector", "both"]:
        print(f"Calculating VUN metrics by Condition Vector...")
        for cond_vec, subdf in df.groupby(condition_column_name):
            cond_vec_str = str(cond_vec).replace(',', ';')
            metrics = compute_vun_metrics(subdf)
            metrics_data.append(get_metrics_row(cond_vec_str, metrics))

    # Save to CSV
    metrics_df = pd.DataFrame(metrics_data)
    print(f"Saving VUN metrics to {output_csv}...")
    metrics_df.to_csv(output_csv, index=False)


def print_vun_metrics(df, condition_column_name, sort_metrics_by):
    """Print VUN metrics to the console."""
    
    def _print_summary(label, metrics, total_df_len):
        print(f"\n--- {label} ---")
        print(f"Total Generated: {metrics['total']} ({metrics['total']/total_df_len*100:.2f}%)")
        print(f"Valid: {metrics['valid']} ({metrics['valid']/metrics['total']*100:.2f}%)")
        print(f"V+Unique: {metrics['unique']} ({metrics['unique']/metrics['total']*100:.2f}%)")
        print(f"V+U+Novel: {metrics['novel']} ({metrics['novel']/metrics['total']*100:.2f}%)")

    if sort_metrics_by in ["all", "both"]:
        metrics = compute_vun_metrics(df)
        _print_summary("Overall VUN Metrics", metrics, len(df))

    if sort_metrics_by in ["Condition Vector", "both"]:
        print("\n--- VUN Metrics by Condition Vector ---")
        for cond_vec, subdf in df.groupby(condition_column_name):
            metrics = compute_vun_metrics(subdf)
            _print_summary(f"Condition Vector: {cond_vec}", metrics, len(df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Valid, Unique, and Novel metrics for generated CIFs.")
    parser.add_argument("--input_parquet", required=True, help="Path to parquet file with generated CIFs.")
    parser.add_argument("--huggingface_dataset", required=True, help="Hugging Face dataset path for novelty check.")
    parser.add_argument("--output_parquet", default=None, help="Path to save processed parquet with VUN metric columns.")
    parser.add_argument("--output_csv", default=None, help="Path to save VUN metrics summary CSV.")
    parser.add_argument("--sort_metrics_by", default="all", choices=["all", "Condition Vector", "both"], help="How to group results for metrics.")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of parallel workers.")
    parser.add_argument("--load_processed_data", type=str, default=None, help="Path to a pre-processed training dataset to speed up novelty.")
    
    args = parser.parse_args()
    
    print(f"Starting VUN metrics calculation with {args.num_workers} workers.")
    
    # Load and process generated data
    gen_df_proc = load_and_process_generated_data(args.input_parquet, args.num_workers)
    
    # Compute Validity
    gen_df_proc = get_valid(gen_df_proc, args.num_workers)

    # Compute Uniqueness
    gen_df_proc = get_unique(gen_df_proc, args.num_workers)
    
    # For Novelty:
    # 1. Build structures and extract formulas from our generated set
    gen_structures = build_generated_structures(gen_df_proc)
    gen_formulas = extract_generated_formulas(gen_structures)
    
    # 2. Load training data, but only for compositions that appeared in our generated set
    base_comps = load_and_filter_training_data(
        args.huggingface_dataset, args.load_processed_data, args.num_workers, gen_formulas
    )
    
    # 3. Compute Novelty against the filtered training data
    gen_df_proc = get_novelty(
        df_gen=gen_df_proc, base_comps=base_comps,
        ltol=LTOL, stol=STOL, angle_tol=ANGLE_TOL,
        structures=gen_structures, workers=args.num_workers,
    )

    # Report and save results
    condition_column = 'condition_vector' if 'condition_vector' in gen_df_proc.columns else 'Condition Vector'
    
    print_vun_metrics(gen_df_proc, condition_column, args.sort_metrics_by)
    save_vun_metrics(gen_df_proc, args.output_csv, condition_column, args.sort_metrics_by)

    if args.output_parquet:
        print(f"\nSaving processed dataframe with VUN metrics to {args.output_parquet}...")
        gen_df_proc.to_parquet(args.output_parquet)
    else:
        print("\nNot saving processed dataframe.")

    print("\nVUN metrics calculation completed.")