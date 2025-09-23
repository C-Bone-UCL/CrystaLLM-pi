"""
Calculate property metrics for CIF structures.
Only really relevant for bandgap and density at the moment. But could be adapted for other properties.
"""

import argparse
import ast
import sys
import warnings
import os

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _utils import (
    predict_properties,
)

warnings.filterwarnings("ignore")

# Normalization utility functions
def inverse_power_log_normalization(x, xmax, beta=0.8):
    """Inverse power-log normalization for property values."""
    return np.exp((x ** (1 / beta)) * np.log(1 + xmax)) - 1

def linear_normalization(normed, x_min, x_max):
    """Linear denormalization for property values."""
    return normed * (x_max - x_min) + x_min

def inverse_signed_log_normalization(normed, x_min, x_max, beta=0.8):
    """Inverse signed-log normalization for property values."""
    def signed_log(x):
        return np.sign(x) * np.log1p(np.abs(x))
    
    signed_min = signed_log(x_min)
    signed_max = signed_log(x_max)
    unpow = normed ** (1 / beta)
    signed_val = unpow * (signed_max - signed_min) + signed_min
    return np.sign(signed_val) * (np.expm1(np.abs(signed_val)))

def _get_property_mae(df, prop, target_col):
    """Calculate MAE for a specific property."""
    if 'bandgap' in prop.lower() or 'bg' in prop.lower():
        pred_col = 'ALIGNN_bg (eV)'
    elif 'density' in prop.lower():
        pred_col = 'gen_density (g/cm3)'
    else:
        return None
    
    if pred_col in df.columns and target_col in df.columns:
        pred_data = df[pred_col].dropna()
        target_data = df[target_col].dropna()
        if not pred_data.empty and not target_data.empty:
            return df[pred_col].sub(df[target_col]).abs().mean()
    return None

def compute_property_mae(df: pd.DataFrame, property_targets):
    """Compute Mean Absolute Error for property predictions."""
    mae_results = {}
    
    for prop in property_targets:
        target_col = f"target_{prop}"
        mae_value = _get_property_mae(df, prop, target_col)
        mae_results[f'MAE_{prop}'] = mae_value
    
    return mae_results

def _denormalize_property_value(val, norm_method, x_min, x_max):
    """Denormalize a single property value based on the normalization method."""
    if val == -100 or pd.isna(val):
        return np.nan
    
    x = float(val)
    if norm_method == "power_log":
        return inverse_power_log_normalization(x, x_max)
    elif norm_method == "linear":
        return linear_normalization(x, x_min, x_max)
    elif norm_method == "signed_log":
        return inverse_signed_log_normalization(x, x_min, x_max)
    else:
        return x

def process_property_targets(gen_df_proc, property_targets, norm_methods, max_values, min_values, num_workers):
    """Process property targets and add denormalized target columns."""
    condition_column_name = 'condition_vector' if 'condition_vector' in gen_df_proc.columns else 'Condition Vector'
    
    for i in gen_df_proc.index:
        cond_val = gen_df_proc.at[i, condition_column_name]
        
        # Parse condition values (handle both string and list formats)
        try:
            cond_vals = ast.literal_eval(cond_val) if isinstance(cond_val, str) else cond_val
        except Exception:
            cond_vals = cond_val
        if not isinstance(cond_vals, (list, tuple)): 
            cond_vals = [cond_vals]

        # Process each property target
        for j, prop in enumerate(property_targets):
            target_col = f"target_{prop}"
            val = cond_vals[j] if j < len(cond_vals) else -100
            
            # Denormalize the property value
            denorm_val = _denormalize_property_value(val, norm_methods[j], min_values[j], max_values[j])
            gen_df_proc.at[i, target_col] = denorm_val
    
    return gen_df_proc

def _get_metric_header(property_targets):
    """Generate CSV header for metrics based on property types."""
    header_parts = ["CV"]
    for prop in property_targets:
        if 'bandgap' in prop.lower() or 'bg' in prop.lower():
            header_parts.append("MAE_BG")
        elif 'density' in prop.lower():
            header_parts.append("MAE_density")
    return ",".join(header_parts) + "\n"

def _get_metric_row(cond_vec_str, mae_results, property_targets):
    """Generate CSV row for metrics based on property types."""
    row_parts = [cond_vec_str]
    for prop in property_targets:
        mae_key = f'MAE_{prop}'
        row_parts.append(str(mae_results.get(mae_key, "")))
    return ",".join(row_parts) + "\n"

def save_property_metrics(df, property_targets, metrics_out, condition_column_name, sort_metrics_by):
    """Save property metrics to CSV file."""
    if not metrics_out:
        return
    
    mode = "w"
    
    # Save overall metrics if requested
    if sort_metrics_by in ["all", "both"]:
        print(f"Saving property metrics to {metrics_out}...")
        mae_results = compute_property_mae(df, property_targets)
        
        with open(metrics_out, mode) as f:
            f.write(_get_metric_header(property_targets))
            f.write(_get_metric_row("all_CVs", mae_results, property_targets))
        mode = "a"
    
    # Save metrics by condition vector if requested
    if sort_metrics_by in ["Condition Vector", "both"]:
        print(f"Saving property metrics by Condition Vector to {metrics_out}...")
        with open(metrics_out, mode) as f:
            if mode == "w":
                f.write(_get_metric_header(property_targets))
            
            # Group by condition vector and save metrics for each group
            groups = df.groupby(condition_column_name)
            for cond_vec, subdf in groups:
                cond_vec_str = str(cond_vec).replace(',', ';')
                mae_results = compute_property_mae(subdf, property_targets)
                f.write(_get_metric_row(cond_vec_str, mae_results, property_targets))

def _print_mae_results(mae_results, property_targets):
    """Print MAE results for given property targets."""
    for prop in property_targets:
        mae_key = f'MAE_{prop}'
        if mae_key in mae_results and mae_results[mae_key] is not None:
            if 'bandgap' in prop.lower() or 'bg' in prop.lower():
                print(f"Mean Absolute Error in Band Gap Prediction: {mae_results[mae_key]}")
            elif 'density' in prop.lower():
                print(f"Mean Absolute Error in Density Prediction: {mae_results[mae_key]}")

def print_property_metrics(df, property_targets, condition_column_name, sort_metrics_by):
    """Print property metrics to console."""
    # Print overall metrics if requested
    if sort_metrics_by in ["all", "both"]:
        print("\n============ Overall Property Metrics =============")
        mae_results = compute_property_mae(df, property_targets)
        _print_mae_results(mae_results, property_targets)

    # Print metrics by condition vector if requested
    if sort_metrics_by in ["Condition Vector", "both"]:
        print("\n======= Property Metrics by Condition Vector =====")
        groups = df.groupby(condition_column_name)
        for cond_vec, subdf in groups:
            print(f"\n=====Condition Vector: {cond_vec}=====")
            mae_results = compute_property_mae(subdf, property_targets)
            _print_mae_results(mae_results, property_targets)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate property metrics for CIF structures.")
    parser.add_argument("--post_parquet", required=True, help="Path to processed parquet file with VUN metrics.")
    parser.add_argument("--parquet_out", default=None, help="Path to save updated parquet with property metrics.")
    parser.add_argument("--metrics_out", default=None, help="Path to save property metrics CSV.")
    parser.add_argument("--sort_metrics_by", default="all", choices=["all", "Condition Vector", "both"], help="How to group results for metrics.")
    parser.add_argument("--num_workers", required=False, default=8, type=int, help="Number of parallel workers.")
    parser.add_argument("--property_targets", type=str, default="[]", help='List of property columns (unnormalised), in the same order as in the condition vector. E.g. \'["Bandgap (eV)", "Energy Above Hull (eV)"]\'.')
    
    # Normalization parameters for up to 3 properties
    for i in range(1, 4):
        parser.add_argument(f"--property{i}_normaliser", type=str, choices=["power_log", "linear", "None"], default="None", help=f"Normalization method for the {['first', 'second', 'third'][i-1]} property column.")
        parser.add_argument(f"--max_property{i}", type=float, default=17.89, help=f"Maximum value for the {['first', 'second', 'third'][i-1]} property column.")
        parser.add_argument(f"--min_property{i}", type=float, default=0.0, help=f"Minimum value for the {['first', 'second', 'third'][i-1]} property column.")
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Load processed data
    print(f"Loading processed data from {args.post_parquet}...")
    gen_df_proc = pd.read_parquet(args.post_parquet)
    
    # Parse property targets
    try:
        property_targets = ast.literal_eval(args.property_targets)
        if not isinstance(property_targets, list): 
            property_targets = [property_targets]
    except Exception:
        property_targets = []
        print("No valid property targets provided; skipping property calculation.")
    
    if not property_targets:
        print("No property targets specified. Exiting.")
        sys.exit(0)
    
    print(f"Processing property targets: {property_targets}")
    
    # Set up normalization parameters
    norm_methods = [getattr(args, f'property{i}_normaliser') for i in range(1, len(property_targets) + 1)]
    max_values = [getattr(args, f'max_property{i}') for i in range(1, len(property_targets) + 1)]
    min_values = [getattr(args, f'min_property{i}') for i in range(1, len(property_targets) + 1)]
    
    # Process property targets and denormalize condition vectors
    print("Processing property targets and denormalizing condition vectors...")
    gen_df_proc = process_property_targets(gen_df_proc, property_targets, norm_methods, max_values, min_values, args.num_workers)
    
    # Predict properties
    print("Predicting properties...")
    gen_df_proc = predict_properties(gen_df_proc, property_targets, args.num_workers)
    
    # Determine condition column name
    condition_column_name = 'condition_vector' if 'condition_vector' in gen_df_proc.columns else 'Condition Vector'
    
    # Print and save metrics
    print_property_metrics(gen_df_proc, property_targets, condition_column_name, args.sort_metrics_by)
    save_property_metrics(gen_df_proc, property_targets, args.metrics_out, condition_column_name, args.sort_metrics_by)
    
    # Save updated dataframe
    if args.parquet_out:
        print(f"Saving updated dataframe with property metrics to {args.parquet_out}...")
        gen_df_proc.to_parquet(args.parquet_out)
    
    print("\nProperty metrics calculation completed.")

if __name__ == "__main__":
    main()