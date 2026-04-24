"""Helpers for B2_Dataset_size_study.ipynb to evaluate structural and property metrics.

This module provides functions to calculate performance metrics (validity, uniqueness, 
novelty, and stability) and generate summary and parity plots for crystalline 
structure generation models across various dataset sizes.
"""

import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, ttest_rel, gaussian_kde
from sklearn.metrics import r2_score

from ._shared_utils import _sem

from pathlib import Path

# Constants & Layout Config
# Map internal codebase names to the official paper nomenclature
INTERNAL_METH_MAP = {"Prefix": "PKV", "Residual": "Slider", "Prepend": "Prepend"}
PAPER_METH_MAP = {"PKV": "Prefix", "Slider": "Residual", "Prepend": "Prepend"}

OKABE_ITO_DENSITY = {"Prefix": "#990099", "Residual": "#E69F00", "Prepend": "#56B4E9"}
DENSITY_METHODS = ["Prefix", "Residual", "Prepend"]
DENSITY_SIZE_ORDER = ["1k", "10k", "100k", "full"]
DENSITY_SIZE_ORDER_MAP = {"1k": 0, "10k": 1, "100k": 2, "full": 3}
DENSITY_SIZE_TITLES = {"1k": "1k", "10k": "10k", "100k": "100k", "full": "Full"}

LEGEND_DARK_PURPLE = "#990099"
LEGEND_PURPLE = "#CF81CF"
REFERENCE_FILL = "#888888"
CLUSTER_GUIDE = "#000000"


def plot_density_histogram(
    dfs_dict,
    column_name="Density (g/cm^3)",
    output_path="plots/density/density_histogram.png",
    figsize=(10, 8),
    label_fontsize=18,
    ticks_fontsize=16,
    legend_fontsize=16,
    line_width=2.5,
    axes_linewidth=1.5,
    num_xticks=4,
    num_yticks=4,
    bins=30,
    alpha_val=0.4
):
    """Plots a stacked histogram of density distributions across different dataset sizes."""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Pre-define colors for consistency
    colors = ['blue', 'orange', 'green', 'red']
    
    # Iterate through the dictionary to plot each dataset
    for (label, df), color in zip(dfs_dict.items(), colors):
        if column_name in df.columns:
            ax.hist(
                df[column_name], 
                bins=bins, 
                label=label, 
                color=color, 
                density=False, 
                alpha=alpha_val, 
                linewidth=line_width
            )

    # Set y-axis to log scale
    ax.set_yscale('log')
    ax.set_ylim(0.5, 10**5)  

    # Formatting
    ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=num_xticks))
    ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)

    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    for sp in ("right", "top"):
        ax.spines[sp].set_visible(False)

    ax.set_xlabel('Density (g/cm$^3$)', fontsize=label_fontsize)
    ax.set_ylabel('Frequency', fontsize=label_fontsize)
    
    ax.legend(loc="upper right", frameon=False, fontsize=legend_fontsize)

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        
    return fig

# Data Preprocessing & Core Helpers

def preprocess_density_data(df_dict, stability_threshold=0.157):
    """Standardizes dictionary keys to paper nomenclature and pre-calculates structural masks."""
    processed_dict = {}
    
    for key, df in df_dict.items():
        if "_paper_method" in df.columns:
            processed_dict[key] = df  # Skip if already processed
            continue
            
        df = df.copy()
        lowered = key.lower()
        
        # Determine model
        internal_model = "Prepend"
        if "pkv" in lowered:
            internal_model = "PKV"
        elif "slider" in lowered:
            internal_model = "Slider"
            
        # Determine dataset size
        size = "full"
        for s in DENSITY_SIZE_ORDER:
            if f"-{s}" in lowered or f"_{s}" in lowered or s in lowered:
                size = s
                break
                
        paper_method = PAPER_METH_MAP[internal_model]
        new_key = f"{paper_method}_{size}"
        
        # Pre-calculate boolean masks for downstream efficiency
        is_valid = df.get("is_valid", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        is_unique = df.get("is_unique", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        is_novel = df.get("is_novel", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        
        ehull = pd.to_numeric(df.get("ehull_mace_mp", np.nan), errors="coerce")
        is_stable = ehull.le(stability_threshold).fillna(False)
        
        df["valid_mask"] = is_valid
        df["is_stable_mace"] = is_stable
        df["q_vsun_mask"] = is_valid & is_unique & is_novel & is_stable
        
        # Store metadata in df for easy access
        df["_paper_method"] = paper_method
        df["_size"] = size
        
        processed_dict[new_key] = df
        
    return processed_dict


def _calculate_reference_kde(train_df, col_name, max_val, num_points=400):
    """Calculates the normalized KDE for a given reference dataset column."""
    if train_df is None or col_name not in train_df.columns or len(train_df) <= 50:
        return None, None
        
    values = pd.to_numeric(train_df[col_name], errors="coerce").dropna()
    if values.empty:
        return None, None
        
    kde = gaussian_kde(values, bw_method="scott")
    xs = np.linspace(0, max_val, num_points)
    dens_norm = kde(xs) / kde(xs).max()
    return xs, dens_norm


def _density_count_series_per_bin(predicted_values, subset_mask, bins):
    """Uses np.histogram to efficiently bin and count masked density values."""
    predicted = pd.to_numeric(predicted_values, errors="coerce")
    finite_mask = predicted.notna()
    
    mask = finite_mask
    if subset_mask is not None:
        if isinstance(subset_mask, pd.Series):
            mask = subset_mask.reindex(predicted.index, fill_value=False)
        else:
            mask = pd.Series(subset_mask, index=predicted.index)
        mask = mask.fillna(False).astype(bool) & finite_mask
        
    filtered_vals = predicted[mask].values
    counts, _ = np.histogram(filtered_vals, bins=bins)
    return counts.astype(float)


def _density_group_offsets(item_count, bin_width):
    """Calculates x-axis offsets for grouped bar charts."""
    if item_count <= 0:
        return np.array([0.0]), bin_width

    raw_width = bin_width / item_count
    offsets = (np.arange(item_count) - (item_count - 1) / 2.0) * raw_width
    return offsets, raw_width


def _format_tick_label(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _cluster_ticks(bins, target_spacing):
    """Generates centered tick positions and labels for histogram bins."""
    bin_width = np.diff(bins)[0] if len(bins) > 1 else 1.0
    bin_centers = bins[:-1] + np.diff(bins) / 2
    stride = max(1, int(round(target_spacing / bin_width)))
    
    tick_indices = np.arange(0, len(bin_centers), stride)
    tick_positions = bin_centers[tick_indices]
    tick_labels = [_format_tick_label(bin_centers[index]) for index in tick_indices]
    
    return bin_centers, tick_positions, tick_labels


def _get_frame_metadata(frame, column_name, default=None):
    """Returns a scalar metadata value stored as a constant dataframe column."""
    value = frame.get(column_name, default)
    if isinstance(value, pd.DataFrame):
        if value.empty or value.shape[1] == 0:
            return default
        first_column = value.iloc[:, 0]
        return first_column.iloc[0] if not first_column.empty else default
    if isinstance(value, pd.Series):
        return value.iloc[0] if not value.empty else default
    return value


def _density_sort_sizes(df_dict):
    """Returns sorted unique sizes present in the preprocessed dictionary."""
    unique_sizes = {_get_frame_metadata(df, "_size", "full") for df in df_dict.values()}
    return sorted(unique_sizes, key=lambda size: DENSITY_SIZE_ORDER_MAP.get(size, 99))


# Metrics & Calculation Functions

def get_metrics_dataset_size_study(
    dfs_dict,
    train_df,
    *,
    train_col="Density (g/cm^3)",
    target_col="target_Density (g/cm^3)",
    gen_col="gen_density (g/cm3)",
    targets=(1.2747, 15.30, 22.94),
):
    """Calculates core statistical metrics (MAE, std, Pearson) for the dataset size study."""
    proc_dict = preprocess_density_data(dfs_dict)
    
    def _subset(frame: pd.DataFrame, tgt_col: str, tgt: float, pred_col: str):
        mask = np.isclose(frame[tgt_col], tgt, atol=1e-2) & frame["valid_mask"] & frame[pred_col].notna()
        return pd.to_numeric(frame.loc[mask, pred_col], errors="coerce").dropna()

    metrics_list = []
    for target in targets:
        for key, frame in proc_dict.items():
            method = _get_frame_metadata(frame, "_paper_method")
            size = _get_frame_metadata(frame, "_size")
            values = _subset(frame, target_col, target, gen_col)
            
            if values.empty:
                continue

            metrics_list.append({
                "method": method,
                "size": size,
                "target": target,
                "count": len(values),
                "mean": float(np.mean(values.values)),
                "mae": np.mean(np.abs(values.values - target)),
                "std": np.std(values.values, ddof=0),
            })

    met_df = pd.DataFrame(metrics_list)
    if "size" in met_df.columns:
        met_df["size"] = met_df["size"].astype(str).str.strip()

    size_map = {"full": len(train_df), "100k": 1e5, "10k": 1e4, "1k": 1e3}
    met_df["size_numeric"] = met_df["size"].map(size_map)

    metrics = {}

    for method in met_df.method.unique():
        subset = met_df[met_df.method == method]
        if len(subset) >= 2:
            corr, p_value = pearsonr(subset["count"], subset["size_numeric"])
            metrics[f"{method}_count_vs_size_correlation"] = corr
            metrics[f"{method}_count_vs_size_pvalue"] = p_value

    for method in met_df.method.unique():
        mae_values = met_df[met_df.method == method]["mae"].values
        if len(mae_values) > 0:
            metrics[f"{method}_avg_mae"] = mae_values.mean()
            metrics[f"{method}_sem_mae"] = _sem(mae_values)

        std_values = met_df[met_df.method == method]["std"].values
        if len(std_values) > 0:
            metrics[f"{method}_avg_std"] = std_values.mean()
            metrics[f"{method}_sem_std"] = _sem(std_values)

    df_1k = met_df[met_df["size"] == "1k"]
    for method in DENSITY_METHODS:
        method_counts = df_1k[df_1k["method"] == method]["count"]
        if not method_counts.empty:
            metrics[f"{method}_1k_avg_valid_count"] = method_counts.mean()
            metrics[f"{method}_1k_sem_valid_count"] = _sem(method_counts.values, single_value=0.0)

    for method in met_df.method.unique():
        for size in DENSITY_SIZE_ORDER:
            subset = met_df[(met_df.method == method) & (met_df.size == size)]
            if not subset.empty:
                values = subset["mae"].values
                metrics[f"{method}_{size}_mae"] = values.mean()
                metrics[f"{method}_{size}_mae_sem"] = _sem(values)

    df_wide = met_df.pivot_table(index=["target", "size"], columns="method", values="mae").dropna()

    if {"Residual", "Prefix"}.issubset(df_wide.columns):
        t_stat, p_value = ttest_rel(df_wide["Residual"], df_wide["Prefix"])
        metrics["ttest_residual_vs_prefix_t"] = t_stat
        metrics["ttest_residual_vs_prefix_p"] = p_value

    if {"Prepend", "Prefix"}.issubset(df_wide.columns):
        t_stat, p_value = ttest_rel(df_wide["Prepend"], df_wide["Prefix"])
        metrics["ttest_prepend_vs_prefix_t"] = t_stat
        metrics["ttest_prepend_vs_prefix_p"] = p_value

    # Print nicely formatted summary
    for method in met_df.method.unique():
        if f"{method}_count_vs_size_correlation" in metrics:
            print(f"{method}: Pearson r (valid count vs size) = {metrics[f'{method}_count_vs_size_correlation']:.3f}")
    
    print("\n")
    for label, mean_suffix, sem_suffix in (("Avg MAE", "avg_mae", "sem_mae"), ("Avg std", "avg_std", "sem_std")):
        for method in met_df.method.unique():
            if f"{method}_{mean_suffix}" in metrics:
                print(f"{method}: {label} = {metrics[f'{method}_{mean_suffix}']:.3f} +/- {metrics[f'{method}_{sem_suffix}']:.3f}")
    
    print("\n")
    for method in DENSITY_METHODS:
        if f"{method}_1k_avg_valid_count" in metrics:
            print(f"{method} (1k): Avg valid count = {metrics[f'{method}_1k_avg_valid_count']:.1f} +/- {metrics[f'{method}_1k_sem_valid_count']:.1f}")

    print("\n")
    if "ttest_residual_vs_prefix_t" in metrics:
        print(f"t-test Residual vs Prefix MAE: t={metrics['ttest_residual_vs_prefix_t']:.3f}, p={metrics['ttest_residual_vs_prefix_p']:.3f}")

    if "ttest_prepend_vs_prefix_t" in metrics:
        print(f"t-test Prepend vs Prefix MAE: t={metrics['ttest_prepend_vs_prefix_t']:.3f}, p={metrics['ttest_prepend_vs_prefix_p']:.3f}")

    metrics["raw_dataframe"] = met_df
    return metrics


def calculate_vsun_parity_metrics(
    processed_dict, 
    target_col="target_Density (g/cm^3)", 
    pred_col="gen_density (g/cm3)", 
    hit_tol=1.0
):
    """Extracts R2, MAE, and Yield percentages for the Parity plot data."""
    metrics_data = []

    for key, data_frame in processed_dict.items():
        if target_col not in data_frame.columns or pred_col not in data_frame.columns:
            continue

        method = _get_frame_metadata(data_frame, "_paper_method")
        size = _get_frame_metadata(data_frame, "_size")
        
        valid_mask = data_frame["valid_mask"]
        q_vsun_mask = data_frame["q_vsun_mask"]
        data_frame_vsun = data_frame[q_vsun_mask]

        target_all = pd.to_numeric(data_frame[target_col], errors="coerce")
        pred_all = pd.to_numeric(data_frame[pred_col], errors="coerce")
        is_hit = (pred_all - target_all).abs().le(hit_tol)
        
        total_len = len(data_frame)
        valid_yield_pct = ((valid_mask & is_hit).sum() / total_len) * 100 if total_len > 0 else 0.0
        q_vsun_yield_pct = ((q_vsun_mask & is_hit).sum() / total_len) * 100 if total_len > 0 else 0.0

        target_vsun = pd.to_numeric(data_frame_vsun[target_col], errors="coerce")
        pred_vsun = pd.to_numeric(data_frame_vsun[pred_col], errors="coerce")

        valid_idx_all = target_vsun.notna() & pred_vsun.notna()
        abs_err = (pred_vsun[valid_idx_all] - target_vsun[valid_idx_all]).abs()
        r2 = r2_score(target_vsun[valid_idx_all], pred_vsun[valid_idx_all]) if valid_idx_all.sum() > 1 else np.nan

        metrics_data.append({
            "Method": method,
            "Training Set": DENSITY_SIZE_TITLES.get(size, size),
            "$R^2$": r2,
            "MAE [g/cm$^3$]": abs_err.mean(),
            "SE [g/cm$^3$]": abs_err.sem(),
            "N_Valid": int(valid_mask.sum()),
            "Valid Target Yield [%]": valid_yield_pct,
            "Q_VSUN Target Yield [%]": q_vsun_yield_pct,
        })

    return pd.DataFrame(
        metrics_data,
        columns=["Method", "Training Set", "$R^2$", "MAE [g/cm$^3$]", "SE [g/cm$^3$]", "N_Valid", "Valid Target Yield [%]", "Q_VSUN Target Yield [%]"],
    )


def format_density_metrics_table(metrics_df):
    """Cleans up the parity metrics dataframe for direct printing or LaTeX export."""
    table = metrics_df.copy()
    if table.empty:
        return table

    method_order = {"Prefix": 0, "Residual": 1, "Prepend": 2}
    table["size_order"] = table["Training Set"].map(DENSITY_SIZE_ORDER_MAP)
    table["method_order"] = table["Method"].map(method_order)
    table = table.sort_values(["size_order", "method_order"]).drop(columns=["size_order", "method_order"])

    table["$R^2$"] = table["$R^2$"].round(2)
    table["MAE (+/- SE)"] = table.apply(
        lambda row: f"{row['MAE [g/cm$^3$]']:.2f} ({row['SE [g/cm$^3$]']:.3f})" if pd.notna(row["MAE [g/cm$^3$]"]) else "--",
        axis=1,
    )
    table["N_Valid"] = table["N_Valid"].fillna(0).astype(int)
    table["Valid Target Yield (%)"] = table["Valid Target Yield [%]"].round(2)
    table["Q_VSUN Target Yield (%)"] = table["Q_VSUN Target Yield [%]"].round(2)

    return table[["Training Set", "Method", "$R^2$", "MAE (+/- SE)", "N_Valid", "Valid Target Yield (%)", "Q_VSUN Target Yield (%)"]]


# Plotting Functions

def _draw_density_summary_legend(ax, legend_fontsize=14, title_fontsize=16):
    """Renders the standard legends for the output space density summary."""
    ax.axis("off")

    method_handles = [
        Patch(facecolor=OKABE_ITO_DENSITY[method], edgecolor=OKABE_ITO_DENSITY[method], alpha=0.9)
        for method in DENSITY_METHODS
    ]
    method_legend = ax.legend(
        method_handles, DENSITY_METHODS, title="Model", loc="center left",
        bbox_to_anchor=(0.00, 0.52), frameon=False, fontsize=legend_fontsize,
        ncol=3, columnspacing=1.2, handlelength=1.4, borderaxespad=0.0,
    )
    method_legend.get_title().set_fontsize(title_fontsize)
    method_legend.get_title().set_fontweight("bold")
    ax.add_artist(method_legend)

    encoding_handles = [
        Patch(facecolor=LEGEND_DARK_PURPLE, edgecolor=LEGEND_DARK_PURPLE, alpha=0.95),
        Patch(facecolor=LEGEND_PURPLE, edgecolor=LEGEND_PURPLE, alpha=0.95),
    ]
    encoding_legend = ax.legend(
        encoding_handles, ["$\mathregular{Q_{VSUN}}$", "Valid subset"],
        title="Bar", loc="center", bbox_to_anchor=(0.53, 0.52), frameon=False,
        fontsize=legend_fontsize, ncol=2, columnspacing=1.3, handlelength=1.8, borderaxespad=0.0,
    )
    encoding_legend.get_title().set_fontsize(title_fontsize)
    encoding_legend.get_title().set_fontweight("bold")
    ax.add_artist(encoding_legend)

    reference_handles = [
        Patch(facecolor=REFERENCE_FILL, alpha=0.22, edgecolor="none"),
        Line2D([], [], color=CLUSTER_GUIDE, alpha=0.28, linestyle="--", linewidth=1.0),
    ]
    reference_legend = ax.legend(
        reference_handles, ["Training-set density", "Requested target"],
        title="Reference", loc="center right", bbox_to_anchor=(1.00, 0.52),
        frameon=False, fontsize=legend_fontsize, handlelength=1.9, borderaxespad=0.0,
    )
    reference_legend.get_title().set_fontsize(title_fontsize)
    reference_legend.get_title().set_fontweight("bold")
    ax.add_artist(reference_legend)


def plot_density_output_space_summary(
    df_dict,
    train_df=None,
    train_den_col="Density (g/cm^3)",
    target_den_col="target_Density (g/cm^3)",
    pred_den_col="gen_density (g/cm3)",
    max_density=25.0,
    bin_width=1.0,
    min_bin_count=10,
    figsize=(24, 6.6),
    title_fontsize=18,
    label_fontsize=16,
    ticks_fontsize=14,
    axes_linewidth=1.5,
    num_yticks=5,
    num_yticks_ref=3,
    wspace=0.025,
    hspace=0.05,
):
    """Plots grouped stacked count histograms across generated density space."""
    proc_dict = preprocess_density_data(df_dict)
    
    bins = np.arange(0, max_density + bin_width, bin_width)
    actual_bin_width = np.diff(bins)[0] if len(bins) > 1 else bin_width
    bin_centers, x_tick_values, x_tick_labels = _cluster_ticks(bins, target_spacing=5.0 if max_density > 12 else 2.0)

    requested_targets = set()
    target_columns = (target_den_col, "target_Density (g/cm3)")
    for data_frame in proc_dict.values():
        for target_column in target_columns:
            if target_column in data_frame.columns:
                numeric_targets = pd.to_numeric(data_frame[target_column], errors="coerce").dropna()
                requested_targets.update(map(float, numeric_targets.unique()))
                
    requested_target_positions = np.asarray(sorted(target for target in requested_targets if 0.0 <= target <= max_density), dtype=float)
    plot_sizes = [
        size for size in DENSITY_SIZE_ORDER
        if any(_get_frame_metadata(df, "_size", "full") == size for df in proc_dict.values())
    ]
    ordered_models = [
        model for model in DENSITY_METHODS
        if any(_get_frame_metadata(df, "_paper_method") == model for df in proc_dict.values())
    ]
    offsets, bar_width = _density_group_offsets(len(ordered_models), actual_bin_width)

    xs, dens_norm = _calculate_reference_kde(train_df, train_den_col, max_density)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(2, len(plot_sizes), height_ratios=[1.0, 0.25], hspace=hspace, wspace=wspace)

    axes = np.empty(len(plot_sizes), dtype=object)
    if len(plot_sizes) == 0:
        raise ValueError("No density datasets were found in df_dict.")

    axes[0] = fig.add_subplot(grid[0, 0])
    for col in range(1, len(plot_sizes)):
        axes[col] = fig.add_subplot(grid[0, col], sharex=axes[0], sharey=axes[0])
    legend_ax = fig.add_subplot(grid[1, :])

    y_max = 0.0
    for col, size_label in enumerate(plot_sizes):
        ax = axes[col]
        for label, data_frame in proc_dict.items():
            if _get_frame_metadata(data_frame, "_size", "full") != size_label or pred_den_col not in data_frame.columns:
                continue

            method = _get_frame_metadata(data_frame, "_paper_method")
            if method not in ordered_models:
                continue

            predicted = pd.to_numeric(data_frame[pred_den_col], errors="coerce")
            
            total_counts = _density_count_series_per_bin(predicted, None, bins)
            valid_counts = _density_count_series_per_bin(predicted, data_frame["valid_mask"], bins)
            q_counts = _density_count_series_per_bin(predicted, data_frame["q_vsun_mask"], bins)

            sufficient_counts = total_counts >= min_bin_count
            if not np.any(sufficient_counts):
                continue

            method_index = ordered_models.index(method)
            x_plot = bin_centers[sufficient_counts] + offsets[method_index]
            valid_plot = valid_counts[sufficient_counts]
            q_plot = np.minimum(q_counts[sufficient_counts], valid_plot)
            valid_only_plot = np.clip(valid_plot - q_plot, 0, None)

            color = OKABE_ITO_DENSITY[method]
            # Plot Q_VSUN (inner dark bar)
            ax.bar(
                x_plot, q_plot, width=bar_width, color=color, alpha=0.95, 
                edgecolor=color, linewidth=axes_linewidth * 0.8, zorder=3, align="center"
            )
            # Plot Valid Only (outer light bar)
            ax.bar(
                x_plot, valid_only_plot, bottom=q_plot, width=bar_width, color=color, alpha=0.28, 
                edgecolor=color, linewidth=axes_linewidth * 0.8, zorder=3, align="center"
            )

            if np.any(np.isfinite(valid_plot)):
                y_max = max(y_max, float(np.nanmax(valid_plot)))

    panel_labels = [f"({string.ascii_lowercase[index]})" for index in range(len(plot_sizes))]
    column_titles = [DENSITY_SIZE_TITLES[size] for size in plot_sizes]
    y_limit = (0, max(1.0, y_max * 1.12))

    for col, ax in enumerate(axes):
        if dens_norm is not None:
            density_ax = ax.twinx()
            density_ax.fill_between(xs, dens_norm, color=REFERENCE_FILL, alpha=0.22, zorder=0)
            density_ax.set_ylim(0, 1)
            density_ax.spines["top"].set_visible(False)
            density_ax.spines["left"].set_visible(False)
            density_ax.spines["bottom"].set_visible(False)
            ax.set_zorder(density_ax.get_zorder() + 1)
            ax.patch.set_visible(False)

            if col == len(plot_sizes) - 1:
                density_ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks_ref, prune="upper"))
                density_ax.spines["right"].set_visible(True)
                density_ax.spines["right"].set_linewidth(axes_linewidth)
                density_ax.spines["right"].set_color("gray")
                density_ax.spines["right"].set_position(("outward", 5))
                density_ax.tick_params(axis="y", which="major", labelsize=ticks_fontsize, colors="gray", right=True, labelright=True)
                density_ax.set_ylabel("Reference density", fontsize=label_fontsize, color="gray")
            else:
                density_ax.spines["right"].set_visible(False)
                density_ax.tick_params(axis="y", which="both", right=False, labelright=False)

        ax.set_axisbelow(True)
        ax.set_xlim(0, max_density)
        for target_value in requested_target_positions:
            ax.axvline(target_value, color=CLUSTER_GUIDE, alpha=1.0, linestyle="--", linewidth=1.5, zorder=0)
            
        ax.set_xticks(x_tick_values)
        ax.set_xticklabels(x_tick_labels)
        ax.set_ylim(*y_limit)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks, integer=True, min_n_ticks=3))
        ax.tick_params(axis="both", which="major", labelsize=ticks_fontsize)
        ax.text(-0.0, 1.05, panel_labels[col], transform=ax.transAxes, fontsize=title_fontsize, fontweight="bold", va="bottom", ha="right")

        ax.spines["bottom"].set_linewidth(axes_linewidth)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col == 0:
            ax.spines["left"].set_linewidth(axes_linewidth)
            ax.set_ylabel("Valid generations [count]", fontsize=label_fontsize)
        else:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)

        ax.set_title(column_titles[col], fontsize=title_fontsize)
        ax.set_xlabel("Generated density [g/cm$^{3}$]", fontsize=label_fontsize)

    _draw_density_summary_legend(legend_ax, legend_fontsize=ticks_fontsize, title_fontsize=title_fontsize)
    return fig


def plot_vsun_parity_grid_density(
    df_dict,
    train_df=None,
    train_den_col="Density (g/cm^3)",
    target_col="target_Density (g/cm^3)",
    pred_col="gen_density (g/cm3)",
    hit_tol=1.0,
    figsize=(22, 16),
    title_fontsize=18,
    label_fontsize=16,
    ticks_fontsize=14,
    line_width=2.5,
    scatter_size=100,
    axes_linewidth=1.5,
    num_yticks=5,
    num_yticks_ref=3,
):
    """Creates a parity grid mapping generated densities against requested targets.
    
    Returns:
        tuple: (Matplotlib figure, DataFrame of calculated metrics)
    """
    proc_dict = preprocess_density_data(df_dict)
    metrics_df = calculate_vsun_parity_metrics(proc_dict, target_col, pred_col, hit_tol)
    
    plot_sizes = _density_sort_sizes(proc_dict)
    if not plot_sizes:
        raise ValueError("No density datasets were found in df_dict.")

    fig, axes = plt.subplots(
        len(DENSITY_METHODS), len(plot_sizes), figsize=figsize, sharex=True, sharey=True, constrained_layout=True
    )
    if len(DENSITY_METHODS) == 1 and len(plot_sizes) == 1: axes = np.asarray([[axes]], dtype=object)
    elif len(DENSITY_METHODS) == 1: axes = np.expand_dims(axes, axis=0)
    elif len(plot_sizes) == 1: axes = np.expand_dims(axes, axis=1)

    # Reference KDE Logic
    train_max = pd.to_numeric(train_df[train_den_col], errors="coerce").max() if train_df is not None else 25.0
    xs, dens_norm = _calculate_reference_kde(train_df, train_den_col, train_max)

    all_target_values = set()
    for data_frame in proc_dict.values():
        if target_col in data_frame.columns:
            numeric_targets = pd.to_numeric(data_frame[target_col], errors="coerce").dropna()
            all_target_values.update(map(float, numeric_targets.unique()))

    common_targets = sorted(all_target_values)
    target_positions = np.arange(len(common_targets), dtype=float)
    target_to_position = {target: position for position, target in zip(target_positions, common_targets)}
    target_tick_labels = [f"{target:.1f}".rstrip("0").rstrip(".") for target in common_targets]

    def _map_to_target_axis(values):
        """Maps continuous values to the categorical target axis spacing."""
        values = np.asarray(values, dtype=float)
        mapped = np.full(values.shape, np.nan, dtype=float)
        finite_mask = np.isfinite(values)
        if not finite_mask.any(): return mapped

        finite_values = values[finite_mask]
        if len(common_targets) == 0:
            mapped[finite_mask] = finite_values
            return mapped
        if len(common_targets) == 1:
            mapped[finite_mask] = target_positions[0]
            return mapped

        mapped_values = np.interp(finite_values, common_targets, target_positions)

        # Extrapolate values outside the bounds
        left_mask = finite_values < common_targets[0]
        if np.any(left_mask):
            left_scale = (target_positions[1] - target_positions[0]) / (common_targets[1] - common_targets[0])
            mapped_values[left_mask] = target_positions[0] + (finite_values[left_mask] - common_targets[0]) * left_scale

        right_mask = finite_values > common_targets[-1]
        if np.any(right_mask):
            right_scale = (target_positions[-1] - target_positions[-2]) / (common_targets[-1] - common_targets[-2])
            mapped_values[right_mask] = target_positions[-1] + (finite_values[right_mask] - common_targets[-1]) * right_scale

        mapped[finite_mask] = mapped_values
        return mapped

    x_limits = (-0.45, max(len(common_targets) - 0.55, 0.55))
    y_plot_limits = x_limits
    flier_marker_size = max(4.0, np.sqrt(scatter_size) / 3.0)
    line_positions = np.asarray([target_to_position[target] for target in common_targets], dtype=float) if common_targets else np.array([])

    for row, target_method in enumerate(DENSITY_METHODS):
        for col, current_size in enumerate(plot_sizes):
            ax = axes[row, col]
            subplot_letter = string.ascii_lowercase[row * len(plot_sizes) + col]
            
            # Find matching preprocessed dataset
            dict_key = f"{target_method}_{current_size}"
            data_frame = proc_dict.get(dict_key)

            if row == 0:
                ax.set_title(f"Trained on: {DENSITY_SIZE_TITLES[current_size]}", fontsize=title_fontsize, weight="bold")
            if col == 0:
                ax.set_ylabel(f"{target_method}\nGenerated density [g/cm$^{{3}}$]", fontsize=label_fontsize)
            if row == len(DENSITY_METHODS) - 1:
                ax.set_xlabel("Target density [g/cm$^{3}$]", fontsize=label_fontsize)

            ax.text(-0.0, 1.05, f"({subplot_letter})", transform=ax.transAxes, fontsize=title_fontsize, fontweight="bold", va="bottom", ha="right")

            if data_frame is None or target_col not in data_frame.columns or pred_col not in data_frame.columns:
                ax.set_visible(False)
                continue

            data_frame_vsun = data_frame[data_frame["q_vsun_mask"]]
            target_vsun = pd.to_numeric(data_frame_vsun[target_col], errors="coerce")
            pred_vsun = pd.to_numeric(data_frame_vsun[pred_col], errors="coerce")
            unique_targets = sorted(target_vsun.dropna().unique())

            plot_positions = []
            plot_data = []
            for target_value in unique_targets:
                values = pred_vsun[target_vsun == target_value].dropna()
                if values.empty or float(target_value) not in target_to_position:
                    continue
                plot_positions.append(target_to_position[float(target_value)])
                plot_data.append(_map_to_target_axis(values.to_numpy(dtype=float)))

            color = OKABE_ITO_DENSITY.get(target_method, "#000000")
            if plot_data:
                ax.boxplot(
                    plot_data, positions=np.asarray(plot_positions, dtype=float), widths=0.48, patch_artist=True, manage_ticks=False,
                    boxprops=dict(facecolor=color, color=color, alpha=0.6, linewidth=axes_linewidth),
                    capprops=dict(color=color, lw=axes_linewidth),
                    whiskerprops=dict(color=color, lw=axes_linewidth),
                    flierprops=dict(marker="o", markerfacecolor=color, markeredgecolor="none", alpha=0.3, markersize=flier_marker_size),
                    medianprops=dict(color="black", linewidth=line_width),
                )
            if len(line_positions) > 0:
                ax.plot(line_positions, line_positions, "k--", lw=line_width * 0.8, alpha=0.8, zorder=0)

            if dens_norm is not None and len(common_targets) > 1:
                density_ax = ax.twinx()
                mapped_xs = _map_to_target_axis(xs)
                density_ax.fill_between(mapped_xs, dens_norm, color="gray", alpha=0.15, zorder=0)
                density_ax.set_ylim(0, 1)
                density_ax.spines["top"].set_visible(False)
                density_ax.spines["left"].set_visible(False)
                density_ax.spines["bottom"].set_visible(False)
                ax.set_zorder(density_ax.get_zorder() + 1)
                ax.patch.set_visible(False)

                if col == len(plot_sizes) - 1:
                    density_ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks_ref, prune="upper"))
                    density_ax.spines["right"].set_visible(True)
                    density_ax.spines["right"].set_linewidth(axes_linewidth)
                    density_ax.spines["right"].set_color("gray")
                    density_ax.tick_params(axis="y", which="major", labelsize=ticks_fontsize, colors="gray", right=True, labelright=True)
                    if row == 1:
                        density_ax.set_ylabel("Reference density", fontsize=label_fontsize, color="gray")
                else:
                    density_ax.spines["right"].set_visible(False)
                    density_ax.tick_params(axis="y", which="both", right=False, labelright=False)

            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_plot_limits)
            
            if len(target_positions) > 0:
                ax.set_xticks(target_positions)
                if row == len(DENSITY_METHODS) - 1:
                    ax.set_xticklabels(target_tick_labels, rotation=0)
                else:
                    ax.set_xticklabels([])
                ax.set_yticks(target_positions)
                ax.set_yticklabels(target_tick_labels)
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks, prune="upper"))
                
            ax.tick_params(axis="both", which="major", labelsize=ticks_fontsize)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(axes_linewidth)
            ax.spines["bottom"].set_linewidth(axes_linewidth)

            if col > 0:
                ax.tick_params(axis="y", labelleft=False)

    return fig, metrics_df


__all__ = [
    "plot_density_histogram",
    "preprocess_density_data",
    "get_metrics_dataset_size_study", 
    "calculate_vsun_parity_metrics",
    "plot_density_output_space_summary", 
    "plot_vsun_parity_grid_density", 
    "format_density_metrics_table"
]