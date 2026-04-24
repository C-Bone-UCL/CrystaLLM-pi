"""Extracts structural properties from generated CIFs and evaluates reconstruction accuracy.

This script parses lattice parameters, unit cell volumes, and calculates system types 
for generated materials. It also provides plotting utilities to visualize the 
accuracy of the generative models against ground truth structures.

Methodology Mapping Notes:
# Validation Metrics: The MAE/R^2 calculations for lattice parameters supplement 
  the Validity criteria in Section 1.0.4.
# Matching Proxy: The plotting function uses an 'RMS-d' column to classify structural 
  matches, acting as a rapid proxy for the Pymatgen StructureMatcher defined in Section 1.0.2.
# Model Naming Conventions for final paper: PKV is referred to as "Prefix", and 
  Slider is referred to as "Residual".
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local application imports
import __init__
from _utils import extract_numeric_property, get_unit_cell_volume, extract_formula_nonreduced
from ._shared_utils import get_stratified_metrics_xrd, _extract_atom_counts_worker


# Global plotting styling constants (paper-ready defaults)
# FIGSIZE = (12, 12)
# TITLE_FONTSIZE = 20
# LABEL_FONTSIZE = 18
# TICKS_FONTSIZE = 16
# LEGEND_FONTSIZE = 16
# ANNOT_FONTSIZE = 16
# MIN_MARKER_SIZE = 5 
# SIZE_MULTIPLIER = 5.0  
# SCATTER_EDGE_WIDTH = 0.3
# ALPHA_MATCHED = 0.9
# ALPHA_UNMATCHED = 0.5
# DIAG_LINE_WIDTH = 1.0
# AXES_LINEWIDTH = 1.0

PLOT_PROPERTIES = [
    {"t_key": "True a", "g_key": "Gen a", "title": "Lattice const. a", "xlab": "Target a [Å]", "ylab": "Pred. a [Å]"},
    {"t_key": "True b", "g_key": "Gen b", "title": "Lattice const. b", "xlab": "Target b [Å]", "ylab": "Pred. b [Å]"},
    {"t_key": "True c", "g_key": "Gen c", "title": "Lattice const. c", "xlab": "Target c [Å]", "ylab": "Pred. c [Å]"},
    {"t_key": "True volume", "g_key": "Gen volume", "title": "Volume", "xlab": "Target vol. [Å³]", "ylab": "Pred. vol. [Å³]"}
]


def extract_lattice_params_and_volume(cif):
    """Extracts lattice parameters (a, b, c) and calculates implied unit cell volume.
    
    This function parses the CIF directly. If formatting is invalid or critical keys 
    are missing, it fails gracefully by returning Nones.
    """
    try:
        a = extract_numeric_property(cif, "_cell_length_a")
        b = extract_numeric_property(cif, "_cell_length_b")
        c = extract_numeric_property(cif, "_cell_length_c")
        alpha = extract_numeric_property(cif, "_cell_angle_alpha")
        beta = extract_numeric_property(cif, "_cell_angle_beta")
        gamma = extract_numeric_property(cif, "_cell_angle_gamma")
        
        implied_vol = get_unit_cell_volume(a, b, c, alpha, beta, gamma)
        return a, b, c, implied_vol
        
    except Exception:
        # We silently swallow exceptions here because malformed CIFs from the model 
        # are expected during evaluation, and spamming stdout breaks the logging flow.
        return None, None, None, None


def get_system_type(cif):
    """Determines the number of unique elements (system type) from a CIF string.
    
    Note: While counting unique elements isn't strictly defined in the paper's 
    diversity metrics, it serves as a highly useful proxy for stratifying structural 
    complexity during intermediate analysis.
    """
    try:
        formula = extract_formula_nonreduced(cif)
        elements = re.findall(r'[A-Z][a-z]?', formula)
        return len(set(elements))
    except Exception:
        return None


def calculate_lattice_metrics(x_array, y_array):
    """Calculates Mean Absolute Error and R^2 for continuous property arrays."""
    if len(x_array) < 2:
        return np.nan, np.nan
        
    mae = np.nanmean(np.abs(x_array - y_array))
    r2 = np.corrcoef(x_array, y_array)[0, 1] ** 2
    
    return mae, r2


import concurrent.futures
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_true_vs_gen(df, 
                    figsize=(14, 14),
                    title_fontsize=26,
                    label_fontsize=24,
                    ticks_fontsize=22,
                    legend_fontsize=18,
                    annot_fontsize=22,
                    min_marker_size=3,
                    size_multiplier=6.0,
                    scatter_edge_width=0.5,
                    alpha_matched=0.6,
                    alpha_unmatched=0.6,
                    diag_line_width=1.0,
                    axes_linewidth=2.0,
                    savepath=None, 
                    show_match_legend=True, 
                    show_size_legend=True,
                    num_workers=32
                    ):
    """Generates a scatter plot grid of true vs generated structural properties.
    
    If 'conv_count' is missing from the DataFrame, it automatically triggers 
    parallelized atom counting using the CIF strings in 'Gen Struct'.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # 1. Handle Missing Atom Counts via Parallel Fallback
    if 'conv_count' not in df.columns:
        print(f"'conv_count' missing. Calculating atom counts with {num_workers} workers...")
        cifs = df["Gen Struct"].tolist()
        
        # Use a chunksize for efficiency as per parallel best practices
        chunksize = max(1, len(cifs) // (num_workers * 4))

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Reusing the worker logic from shared utils
            results = list(tqdm(
                executor.map(_extract_atom_counts_worker, cifs, chunksize=chunksize),
                total=len(cifs),
                desc="Parsing fallback geometries",
            ))
        
        # Update DataFrame with results
        df["conv_count"] = pd.to_numeric([r[0] for r in results], errors="coerce")
        print(f"Successfully parsed {df['conv_count'].notna().sum()} structures.")

    rms_valid_mask = df['RMS-d'].notna()
    rms_invalid_mask = df['RMS-d'].isna()
    
    invalid_rows = len(df[rms_invalid_mask])
    percent_valid = ((len(df) - invalid_rows) / len(df)) * 100
    
    print(f"\nNumber of rows with None RMS-d values: {invalid_rows}")
    print(f"Percentage of matches: {percent_valid:.2f}%\n")

    for ax, prop_dict in zip(axes, PLOT_PROPERTIES):
        x = df[prop_dict["t_key"]].to_numpy(dtype=float)
        y = df[prop_dict["g_key"]].to_numpy(dtype=float)

        # Ensure atom counts are valid for sizing
        valid_data_mask = ~np.isnan(x) & ~np.isnan(y) & df['conv_count'].notna()
        sizes = min_marker_size + (df['conv_count'] * size_multiplier)

        # Plotting logic remains consistent with previous iteration
        orange_mask = valid_data_mask & rms_invalid_mask
        if orange_mask.any():
            ax.scatter(
                x[orange_mask], y[orange_mask], 
                s=sizes[orange_mask], 
                alpha=alpha_unmatched, color="orange", marker='o', 
                edgecolor="black", linewidth=scatter_edge_width,
                label='No Structure Match', zorder=2
            )

        green_mask = valid_data_mask & rms_valid_mask
        if green_mask.any():
            ax.scatter(
                x[green_mask], y[green_mask], 
                s=sizes[green_mask], 
                alpha=alpha_matched, color="green", marker='o', 
                edgecolor="black", linewidth=scatter_edge_width,
                label='Structure Matched', zorder=3
            )
            
        if valid_data_mask.any():
            lims = [min(x[valid_data_mask].min(), y[valid_data_mask].min()), 
                    max(x[valid_data_mask].max(), y[valid_data_mask].max())]
            ax.plot(lims, lims, ls="--", lw=diag_line_width, color="black", zorder=1)

        ax.set_xlabel(prop_dict["xlab"], fontsize=label_fontsize)
        ax.set_ylabel(prop_dict["ylab"], fontsize=label_fontsize)
        ax.set_title(prop_dict["title"], fontsize=title_fontsize, pad=4)
        
        ax.tick_params(direction='in', width=axes_linewidth, labelsize=ticks_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(axes_linewidth)

        if show_match_legend and prop_dict["title"] == "Lattice const. c":
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='lower right', 
                      fontsize=legend_fontsize, frameon=False)

        if show_size_legend and prop_dict["title"] == "Volume":
            size_handles = []
            for n in [10, 40, 100]:
                marker_size = np.sqrt(min_marker_size + n * size_multiplier)
                size_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                               markersize=marker_size, markeredgecolor='black', 
                               markeredgewidth=0.5, label=f'{n} atoms')
                )
            ax.legend(handles=size_handles, loc='lower right', 
                      fontsize=legend_fontsize, frameon=False, title="Structure Size", 
                      title_fontsize=legend_fontsize)
        
        total_mae, total_r2 = calculate_lattice_metrics(x[valid_data_mask], y[valid_data_mask])
        unit = "Å³" if "vol" in prop_dict["ylab"].lower() else "Å"
        
        ax.text(
            0.04, 0.96, f"$R^2$: {total_r2:.2f}\nMAE: {total_mae:.2f} {unit}",
            transform=ax.transAxes, fontsize=annot_fontsize + 2,
            va="top", ha="left", zorder=4
        )

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()

def load_and_process_xrd_metrics(file_paths):
    """Loads and processes XRD metrics from a list of parquet files."""
    # Load and stack all parquet files
    dfs = [pd.read_parquet(path) for path in file_paths]
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"Loaded {len(df):,} total rows.")

    # Extract properties
    df[['Gen a', 'Gen b', 'Gen c', 'Gen volume']] = df['Gen Struct'].apply(
        lambda x: pd.Series(extract_lattice_params_and_volume(x))
    )
    df[['True a', 'True b', 'True c', 'True volume']] = df['True Struct'].apply(
        lambda x: pd.Series(extract_lattice_params_and_volume(x))
    )
    
    # Calculate System type
    df['System'] = df['True Struct'].apply(get_system_type)
    
    # Print metrics
    n_none_true_a = df['True a'].isna().sum()
    print(f"Number of rows with None in True a: {n_none_true_a}")
    print(f"Max value in 'System' column: {df['System'].max()}")
    
    mean_rmsd = df['RMS-d'].mean()
    print(f"Mean RMS-d (excluding NaNs): {mean_rmsd:.3f}\n")
    
    return df


__all__ = ["get_stratified_metrics_xrd", "extract_lattice_params_and_volume", "get_system_type", "plot_true_vs_gen", "load_and_process_xrd_metrics"]