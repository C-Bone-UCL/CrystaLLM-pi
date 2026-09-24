"""Provide density, accuracy, and parity-plot helpers for conditional generation.

The module computes densities from CIF strings, calculates accuracy metrics,
and compares generated structures with ground-truth targets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from _utils import get_density

NAME_MAPPING = {
    "PKV": "Prefix",
    "Slider": "Residual"
}

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    # Standard metrics for regression tasks
    abs_errors = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_errors))
    std_dev = float(np.std(abs_errors))
    r_val, _ = pearsonr(y_true, y_pred)
    return mae, std_dev, r_val

def get_processed_density_df(input_parquet: str) -> pd.DataFrame:
    # Load and compute densities, checking for different common CIF column names
    df = pd.read_parquet(input_parquet)
    cif_col = "CIF" if "CIF" in df.columns else "Generated CIF"

    # We apply the density utility directly to the CIF strings
    df["density_g/cm3"] = df[cif_col].apply(get_density)
    return df


def select_one_per_material(df: pd.DataFrame, score: str | None) -> pd.DataFrame:
    """Reduce a generation frame to one structure per input material.

    Generation appends `_1`, `_2`, ... to each prompt's Material ID and returns
    every candidate it produced, so the frame holds several rows per material.
    Rows whose CIF has no computable density are dropped first, which is what
    makes the remaining choice a choice between usable structures.

    `score` picks the winner: "min" for perplexity, where lower is better, and None to keep the first
    valid structure in generation order.
    """
    usable = df.dropna(subset=["density_g/cm3"]).copy()
    usable["Material ID"] = usable["Material ID"].str.replace(r"_\d+$", "", regex=True)

    if score is None or "score" not in usable.columns:
        return usable.groupby("Material ID", as_index=False).first()

    ascending = score == "min"
    ordered = usable.sort_values("score", ascending=ascending, kind="mergesort")
    return ordered.groupby("Material ID", as_index=False).first()

def plot_density_results(true_parquet: str, gen_parquets: dict[str, tuple[str, str | None]],
                         save_path: str | None = None):
    # Main plotting routine that handles data alignment and visualization.
    # Each entry maps a label to (parquet path, score direction), where the
    # direction is "min" for perplexity, or None for unranked.
    df_true = get_processed_density_df(true_parquet)
    total_true = len(df_true)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(gen_parquets)))

    # Track global limits for the parity line
    all_y_true = []
    all_y_pred = []

    for idx, (label, (path, score)) in enumerate(gen_parquets.items()):
        df_gen = select_one_per_material(get_processed_density_df(path), score)

        # Align datasets by Material ID to compare target vs actual generation
        merged = pd.merge(df_true, df_gen, on="Material ID", suffixes=("_true", "_gen"))
        valid_data = merged.dropna(subset=["density_g/cm3_true", "density_g/cm3_gen"])
        
        y_true = valid_data["density_g/cm3_true"].values
        y_pred = valid_data["density_g/cm3_gen"].values
        
        # Apply paper naming conventions if they exist in the label
        display_name = label
        for internal, paper in NAME_MAPPING.items():
            display_name = display_name.replace(internal, paper)
            
        mae, std, r = calculate_metrics(y_true, y_pred)
        failed = total_true - len(valid_data)
        
        # Build the legend label with structural stats and metrics
        legend_label = (
            f"{display_name}\n"
            f"N={len(valid_data)} (Failed: {failed})\n"
            f"MAE: {mae:.3f} ± {std:.3f}, r: {r:.3f}"
        )
        
        ax.scatter(y_true, y_pred, alpha=0.5, color=colors[idx], label=legend_label, edgecolors='none')
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    # Draw a 1:1 parity line based on the extent of the actual data
    if all_y_true:
        lims = [
            min(min(all_y_true), min(all_y_pred)),
            max(max(all_y_true), max(all_y_pred))
        ]
        ax.plot(lims, lims, 'k--', alpha=0.2, zorder=0)

    ax.set_xlabel("True Density (g/cm³)")
    ax.set_ylabel("Generated Density (g/cm³)")
    ax.set_title("CrystaLLM-pi Density Accuracy")
    
    # Legend is moved outside to avoid overlapping with high-density clusters
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved {save_path}")
    plt.show()

__all__ = [
    "calculate_metrics",
    "get_processed_density_df",
    "plot_density_results",
    "select_one_per_material",
]