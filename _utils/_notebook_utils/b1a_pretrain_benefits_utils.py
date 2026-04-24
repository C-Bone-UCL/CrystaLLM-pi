"""Helper functions for evaluating generative model performance on bandgap targets.

This module processes conditional generation outputs, tracks raw structural validity,
evaluates the full $Q_{VSUN}$ subset, computes target hit-rates, and builds the
band-gap plots used in the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score
import string
from matplotlib.lines import Line2D

from ._shared_utils import _fit_line_and_corr, _sem

# Global Configurations & Standards
DEFAULT_HIT_TOL_EV = 0.5
STABILITY_THRESHOLD_EV = 0.157

OKABE_ITO_PRETRAIN = {"PKV": "#990099", "Slider": "#E69F00", "Prepend": "#56B4E9", "Raw": "#000000"}
INTERNAL_METH_MAP = {"Prefix": "PKV", "Residual": "Slider", "Prepend": "Prepend", "Raw": "Raw"}
TABLE_METHODS = ["Prefix", "Residual", "Prepend", "Raw"]
COL_TITLES = ["Scratch", "Pre-trained"]


# Core Helper Functions 

def get_model_and_pretrain_status(label: str):
    """Parses a descriptive label into a standardized architecture name and pretraining status."""
    label_lower = label.lower()
    if "slider" in label_lower or "residual" in label_lower:
        method = "Slider"
    elif "pkv" in label_lower or "prefix" in label_lower:
        method = "PKV"
    elif "prepend" in label_lower:
        method = "Prepend"
    else:
        method = "Raw"
        
    is_pretrained = "scratch" not in label_lower
    return method, is_pretrained


def calculate_vsun_masks(data_frame, stability_threshold=STABILITY_THRESHOLD_EV):
    """Calculates raw-valid and full $Q_{VSUN}$ masks for generated structures."""
    if data_frame.empty:
        return pd.Series(False, index=data_frame.index), pd.Series(False, index=data_frame.index)

    ehull = pd.to_numeric(
        data_frame.get("ehull_mace_mp", pd.Series(np.nan, index=data_frame.index)),
        errors="coerce",
    )
    is_stable = ehull.le(stability_threshold).fillna(False)

    # Use .get() with fallback to handle missing columns gracefully
    is_valid = data_frame.get("is_valid", pd.Series(False, index=data_frame.index)).fillna(False).astype(bool)
    is_unique = data_frame.get("is_unique", pd.Series(False, index=data_frame.index)).fillna(False).astype(bool)
    is_novel = data_frame.get("is_novel", pd.Series(False, index=data_frame.index)).fillna(False).astype(bool)

    valid_mask = is_valid
    vsun_mask = is_valid & is_unique & is_novel & is_stable

    return valid_mask, vsun_mask


def get_training_density(train_df, col_name, min_samples=50, num_points=400, max_val=None):
    """Calculates a normalized Gaussian KDE from training data to act as a reference density."""
    if train_df is None or col_name not in train_df.columns or len(train_df) <= min_samples:
        return None, None

    values = pd.to_numeric(train_df[col_name], errors="coerce").dropna()
    if values.empty:
        return None, None

    kde = gaussian_kde(values, bw_method="scott")
    val_min = 0.0 if max_val else values.min()
    val_max = max_val if max_val else values.max()
    
    xs = np.linspace(val_min, val_max, num_points)
    dens_norm = kde(xs) / kde(xs).max()

    return xs, dens_norm


# Analysis & Plotting Functions 

def get_metrics_ptnd_vs_scratch(
    df_dict,
    *,
    train_df=None,
    train_bg_col="Bandgap (eV)",
    pred_bg_col="ALIGNN_bg (eV)",
    target_bg_col="target_Bandgap (eV)",
    cond_col="target_Bandgap (eV)",
    hit_tol_eV=DEFAULT_HIT_TOL_EV,
):
    """Computes the paper-facing summary metrics across target conditions."""
    conds = set()
    for frame in df_dict.values():
        if cond_col in frame.columns:
            conds.update(frame[cond_col].dropna().unique())
    conds = sorted(map(float, conds))

    rows = []
    for label, frame in df_dict.items():
        method, is_pretrained = get_model_and_pretrain_status(label)
        if cond_col not in frame.columns:
            frame = frame.assign(**{cond_col: [np.nan] * len(frame)})

        valid_mask, vsun_mask = calculate_vsun_masks(frame)

        for condition in conds or [np.nan]:
            subset = frame[frame[cond_col] == condition].copy() if not pd.isna(condition) else frame.copy()
            count = len(subset)
            if count == 0:
                rows.append({"m": method, "p": is_pretrained, "x": condition, "n": 0, "hit": np.nan, "valid": np.nan, "q": np.nan})
                continue

            hit = np.nan
            if {pred_bg_col, target_bg_col}.issubset(subset.columns):
                pred = np.asarray(subset[pred_bg_col], dtype=float)
                true = np.asarray(subset[target_bg_col], dtype=float)
                # Hit rate is naturally restricted to N items with this specific target
                hit = (np.abs(pred - true) <= hit_tol_eV).mean()

            valid = valid_mask.loc[subset.index].mean()
            quality = vsun_mask.loc[subset.index].mean()
                
            rows.append({"m": method, "p": is_pretrained, "x": condition, "n": count, "hit": hit, "valid": valid, "q": quality})

    met = pd.DataFrame(rows)

    xs, dens_norm = get_training_density(train_df, train_bg_col)
    if xs is not None:
        met["density"] = np.interp(met["x"].astype(float), xs, dens_norm)

    met = met[met["n"] > 0].copy()
    met["hit_rate"] = met["hit"]
    metrics = {}

    if "density" in met.columns:
        density_metrics = {
            "hit_rate": "hit_rate_density_correlation",
            "valid": "validity_density_correlation",
            "q": "quality_density_correlation",
        }
        for column_name, metric_name in density_metrics.items():
            _, _, corr = _fit_line_and_corr(met["density"], met[column_name])
            metrics[metric_name] = corr

    # Calculate average deltas (Pretrained - Scratch)
    for metric, prefix in zip(["valid", "q", "hit_rate"], ["validity", "quality", "hit_rate"]):
        diffs = []
        for method in met.m.unique():
            group = met[met.m == method]
            val_pre = group[group.p][metric].mean()
            val_scr = group[~group.p][metric].mean()
            if not np.isnan(val_pre) and not np.isnan(val_scr):
                diffs.append(val_pre - val_scr)
        if diffs:
            metrics[f"avg_delta_{prefix}"] = np.mean(diffs)
            metrics[f"sem_delta_{prefix}"] = _sem(diffs)

    # Print nicely formatted summary
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    return metrics


def plot_pretraining_benefits(
    df_dict, *,
    train_df=None,
    train_bg_col="Bandgap (eV)",
    pred_bg_col="ALIGNN_bg (eV)",
    target_bg_col="target_Bandgap (eV)",
    cond_col="target_Bandgap (eV)",
    hit_tol_eV=DEFAULT_HIT_TOL_EV,
    xlim=None, ylim_hit=None, ylim_valid=None, ylim_q=None,
    savepath=None,
    label_fontsize=16, title_fontsize=18, ticks_fontsize=14,
    line_width=2.2, marker_size=7, axes_thickness=1.5,
    figsize=(18, 5.5),
):
    """Generates the target-space summary line plot for the pre-training benefits study."""
    rows = []
    conds = set()
    for frame in df_dict.values():
        if cond_col in frame.columns:
            conds.update(frame[cond_col].dropna().unique())
    conds = sorted(map(float, conds))

    for label, frame in df_dict.items():
        method, is_pretrained = get_model_and_pretrain_status(label)
        valid_mask, vsun_mask = calculate_vsun_masks(frame)
        
        for condition_value in conds or [np.nan]:
            subset = frame[frame[cond_col] == condition_value] if not pd.isna(condition_value) else frame
            if len(subset) == 0:
                continue

            hit = np.nan
            if {pred_bg_col, target_bg_col}.issubset(subset.columns):
                pred = pd.to_numeric(subset[pred_bg_col], errors="coerce")
                targ = pd.to_numeric(subset[target_bg_col], errors="coerce")
                hit = (pred - targ).abs().le(hit_tol_eV).mean()

            valid = valid_mask[subset.index].mean()
            q_frac = vsun_mask[subset.index].mean()
            rows.append({"m": method, "p": is_pretrained, "x": condition_value, "hit": hit, "valid": valid, "q": q_frac})

    metrics = pd.DataFrame(rows)
    xs, dens_norm = get_training_density(train_df, train_bg_col)

    # Plotting Logic
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    ax_hit, ax_valid, ax_q = axes
    ymax = {"hit": 0.0, "valid": 0.0, "q": 0.0}
    
    for method, color in OKABE_ITO_PRETRAIN.items():
        for is_pretrained, linestyle, marker in ((True, "-", "o"), (False, "--", "^")):
            subset = metrics[(metrics.m == method) & (metrics.p == is_pretrained)]
            for ax, met in zip(axes, ["hit", "valid", "q"]):
                if subset[met].notna().any():
                    ax.plot(subset.x, subset[met], linestyle, color=color, marker=marker, ms=marker_size, lw=line_width)
                    ymax[met] = max(ymax[met], subset[met].max())

    x_max = max(conds) if conds else 0.0
    default_xlim = (0, x_max + 0.4)
    subplot_labels = ["(a)", "(b)", "(c)"]

    for index, (ax, metric_name) in enumerate(zip(axes, ("hit", "valid", "q"))):
        ax.set_xlim(*(xlim if xlim else default_xlim))
        curr_ylim = locals()[f"ylim_{metric_name}"]
        ax.set_ylim(*(curr_ylim if curr_ylim else (0, ymax[metric_name] * 1.15)))
        ax.tick_params(axis="both", which="major", labelsize=ticks_fontsize)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
        
        ax.spines["left"].set_linewidth(axes_thickness)
        ax.spines["bottom"].set_linewidth(axes_thickness)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.text(-0.0, 1.15, subplot_labels[index], transform=ax.transAxes, fontsize=title_fontsize, fontweight="bold", va="top", ha="right")

        if dens_norm is not None:
            density_ax = ax.twinx()
            density_ax.fill_between(xs, dens_norm, color="gray", alpha=0.2, zorder=0)
            density_ax.set_ylim(0, 1)
            density_ax.axis('off')

            if index == len(axes) - 1:
                density_ax.spines["right"].set_visible(True)
                density_ax.spines["right"].set_linewidth(axes_thickness)
                density_ax.spines["right"].set_color("gray")
                density_ax.tick_params(axis="y", which="major", labelsize=ticks_fontsize, colors="gray", right=True, labelright=True)
                density_ax.set_ylabel("Reference density", fontsize=label_fontsize, color="gray")

    ax_hit.set_title("Target hits vs. target band-gap", fontsize=title_fontsize)
    ax_hit.set_ylabel("Fraction target hits", fontsize=label_fontsize)
    ax_valid.set_title("Fraction structurally valid generations", fontsize=title_fontsize)
    ax_valid.set_ylabel("Fraction structurally valid", fontsize=label_fontsize)
    ax_q.set_title("Generation quality $Q_{\\mathrm{VSUN}}$", fontsize=title_fontsize)
    ax_q.set_ylabel("fraction $Q_{\\mathrm{VSUN}}$", fontsize=label_fontsize)
    for ax in axes:
        ax.set_xlabel("Target band-gap [eV]", fontsize=label_fontsize)

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig


def plot_vsun_parity_grid_bandgap(
    df_dict,
    train_df=None,
    train_bg_col="Bandgap (eV)",
    target_bg_col="target_Bandgap (eV)",
    pred_bg_col="ALIGNN_bg (eV)",
    include_raw_row=True,
    hit_tol_eV=DEFAULT_HIT_TOL_EV,
    figsize=(18, 5.5),
    title_fontsize=18, label_fontsize=16, ticks_fontsize=14,
    line_width=2.5, scatter_size=100, axes_linewidth=1.5,
    num_yticks=5, num_yticks_ref=3,
):
    """
    Creates a parity grid mapping generated band-gaps against requested targets.
    Calculates yields strictly out of the population that had an explicit target assigned.
    """
    plot_methods = ["Prefix", "Residual", "Prepend"] + (["Raw"] if include_raw_row else [])
    
    # Identify unique targets across all models
    common_targets = set()
    for df in df_dict.values():
        if target_bg_col in df.columns:
            common_targets.update(pd.to_numeric(df[target_bg_col], errors="coerce").dropna().unique())
    common_targets = sorted(map(float, common_targets))
    
    target_positions = np.arange(len(common_targets), dtype=float)
    target_to_pos = dict(zip(common_targets, target_positions))
    target_tick_labels = [f"{t:.1f}".rstrip("0").rstrip(".") for t in common_targets]

    def _map_to_target_axis(values):
        """Helper to map continuous values to discrete ordinal boxplot positions."""
        values = np.asarray(values, dtype=float)
        mapped = np.full(values.shape, np.nan, dtype=float)
        finite_mask = np.isfinite(values)
        if not finite_mask.any() or len(common_targets) == 0:
            return mapped

        finite_vals = values[finite_mask]
        if len(common_targets) == 1:
            mapped[finite_mask] = target_positions[0]
            return mapped

        mapped_vals = np.interp(finite_vals, common_targets, target_positions)
        
        # Extrapolate slightly out of bounds
        left_mask = finite_vals < common_targets[0]
        if np.any(left_mask):
            scale = (target_positions[1] - target_positions[0]) / (common_targets[1] - common_targets[0])
            mapped_vals[left_mask] = target_positions[0] + (finite_vals[left_mask] - common_targets[0]) * scale

        right_mask = finite_vals > common_targets[-1]
        if np.any(right_mask):
            scale = (target_positions[-1] - target_positions[-2]) / (common_targets[-1] - common_targets[-2])
            mapped_vals[right_mask] = target_positions[-1] + (finite_vals[right_mask] - common_targets[-1]) * scale

        mapped[finite_mask] = mapped_vals
        return mapped

    fig, axes = plt.subplots(len(plot_methods), 2, figsize=figsize)
    if len(plot_methods) == 1:
        axes = np.expand_dims(axes, axis=0)

    xs, dens_norm = get_training_density(train_df, train_bg_col)
    metrics_data = []

    def _style_parity_axis(ax, row, col, show_outer_x):
        if dens_norm is not None and len(common_targets) > 1:
            density_ax = ax.twinx()
            density_ax.fill_between(_map_to_target_axis(xs), dens_norm, color="gray", alpha=0.15, zorder=0)
            density_ax.set_ylim(0, 1)
            density_ax.axis('off')
            ax.patch.set_visible(False)

            if col == 1:
                density_ax.spines["right"].set_visible(True)
                density_ax.spines["right"].set_linewidth(axes_linewidth)
                density_ax.spines["right"].set_color("gray")
                density_ax.tick_params(axis="y", which="major", labelsize=ticks_fontsize, colors="gray", right=True, labelright=True)
                density_ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks_ref, prune="upper"))
                if row == 1:
                    density_ax.set_ylabel("Reference density", fontsize=label_fontsize, color="gray")

        ax.set_xlim(-0.45, max(len(common_targets) - 0.55, 0.55))
        ax.set_ylim(-0.45, max(len(common_targets) - 0.55, 0.55))

        if len(target_positions) > 0:
            ax.set_xticks(target_positions)
            ax.set_xticklabels(target_tick_labels if show_outer_x else [], rotation=20 if show_outer_x else 0, ha="right")
            ax.set_yticks(target_positions)
            ax.set_yticklabels(target_tick_labels)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks, prune="upper"))

        ax.tick_params(axis="both", which="major", labelsize=ticks_fontsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(axes_linewidth)
        ax.spines["bottom"].set_linewidth(axes_linewidth)
        if col == 1:
            ax.tick_params(axis="y", labelleft=False)

    for row, target_method in enumerate(plot_methods):
        for col, is_pretrained in enumerate([False, True]):
            ax = axes[row, col]
            show_outer_x = row == len(plot_methods) - 1

            # Find matching data slice
            internal_name = INTERNAL_METH_MAP[target_method]
            data_frame = None
            for key, frame in df_dict.items():
                m, p = get_model_and_pretrain_status(key)
                if m == internal_name and p == is_pretrained:
                    data_frame = frame
                    break

            # Styling setup
            if row == 0: ax.set_title(COL_TITLES[col], fontsize=title_fontsize, weight="bold")
            if col == 0: ax.set_ylabel("Generated band-gap [eV]", fontsize=label_fontsize)
            if show_outer_x: ax.set_xlabel("Target band-gap [eV]", fontsize=label_fontsize)
            ax.text(0, 1.05, f"({string.ascii_lowercase[row * 2 + col]})", transform=ax.transAxes, fontsize=title_fontsize, weight="bold", ha="right")
            if col == 0:
                ax.text(
                    0.03,
                    0.97,
                    target_method,
                    transform=ax.transAxes,
                    fontsize=title_fontsize - 1,
                    fontweight="bold",
                    color=OKABE_ITO_PRETRAIN[internal_name],
                    ha="left",
                    va="top",
                )

            # Early exit for empty panels
            if data_frame is None:
                ax.axis('off')
                continue

            # Compute Data Metrics
            valid_mask, vsun_mask = calculate_vsun_masks(data_frame)
            df_vsun = data_frame[vsun_mask]

            target_all = pd.to_numeric(data_frame[target_bg_col], errors="coerce")
            pred_all = pd.to_numeric(data_frame[pred_bg_col], errors="coerce")
            has_target = target_all.notna()
            n_targeted = has_target.sum()
            
            is_hit = (pred_all - target_all).abs().le(hit_tol_eV)
            
            # Hit yield percentages strictly out of targeted pop
            valid_yield = ((valid_mask & is_hit & has_target).sum() / n_targeted * 100) if n_targeted > 0 else 0.0
            vsun_yield = ((vsun_mask & is_hit & has_target).sum() / n_targeted * 100) if n_targeted > 0 else 0.0

            t_vsun = pd.to_numeric(df_vsun[target_bg_col], errors="coerce")
            p_vsun = pd.to_numeric(df_vsun[pred_bg_col], errors="coerce")
            
            valid_pairs = t_vsun.notna() & p_vsun.notna()
            abs_err = (p_vsun[valid_pairs] - t_vsun[valid_pairs]).abs()
            r2 = r2_score(t_vsun[valid_pairs], p_vsun[valid_pairs]) if valid_pairs.sum() > 1 else np.nan

            metrics_data.append({
                "Method": target_method, "Regime": COL_TITLES[col], "$R^2$": r2,
                "MAE [eV]": abs_err.mean(), "SE [eV]": abs_err.sem(),
                "N_Valid": int(valid_mask.sum()),
                "Valid Target Yield [%]": valid_yield, "VSUN Target Yield [%]": vsun_yield,
            })

            # Prepare plotting data
            plot_pos, plot_data = [], []
            for t_val in sorted(t_vsun.dropna().unique()):
                vals = p_vsun[t_vsun == t_val].dropna()
                if not vals.empty and float(t_val) in target_to_pos:
                    plot_pos.append(target_to_pos[float(t_val)])
                    plot_data.append(_map_to_target_axis(vals.to_numpy()))

            color = OKABE_ITO_PRETRAIN.get(internal_name, "#000000")
            if len(target_positions) > 0:
                ax.plot(target_positions, target_positions, "k--", lw=line_width * 0.8, alpha=0.8, zorder=0)

            if plot_data:
                ax.boxplot(
                    plot_data, positions=plot_pos, widths=0.48, patch_artist=True, manage_ticks=False,
                    boxprops=dict(facecolor=color, color=color, alpha=0.6, lw=axes_linewidth),
                    capprops=dict(color=color, lw=axes_linewidth), whiskerprops=dict(color=color, lw=axes_linewidth),
                    flierprops=dict(marker="o", mfc=color, mec="none", alpha=0.3, ms=max(4.0, np.sqrt(scatter_size) / 3.0)),
                    medianprops=dict(color="black", lw=line_width),
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No $Q_{\\mathrm{VSUN}}$ samples",
                    transform=ax.transAxes,
                    fontsize=label_fontsize,
                    color="#444444",
                    ha="center",
                    va="center",
                )

            _style_parity_axis(ax, row, col, show_outer_x)

    return fig, pd.DataFrame(metrics_data)


def format_bandgap_metrics_table(metrics_df):
    """Cleans up the metrics dataframe for direct printing or LaTeX export."""
    if metrics_df.empty:
        return metrics_df

    regime_order = {"Scratch": 0, "Pre-trained": 1}
    method_order = {"Prefix": 0, "Residual": 1, "Prepend": 2, "Raw": 3}

    table = metrics_df.copy()
    table["regime_order"] = table["Regime"].map(regime_order)
    table["method_order"] = table["Method"].map(method_order)
    table = table.sort_values(["regime_order", "method_order"]).drop(columns=["regime_order", "method_order"])

    table["$R^2$"] = table["$R^2$"].map(lambda value: "--" if pd.isna(value) else f"{value:.2f}")
    table["MAE (+/- SE)"] = table.apply(lambda r: f"{r['MAE [eV]']:.2f} ({r['SE [eV]']:.3f})" if pd.notna(r["MAE [eV]"]) else "--", axis=1)
    table["N Structurally Valid"] = table["N_Valid"].fillna(0).astype(int)
    table["Structurally Valid Target Yield (%)"] = table["Valid Target Yield [%]"].map(lambda value: f"{value:.2f}")
    table["VSUN Target Yield (%)"] = table["VSUN Target Yield [%]"].map(lambda value: f"{value:.2f}")

    return table.rename(columns={"Method": "Model"})[["Regime", "Model", "$R^2$", "MAE (+/- SE)", "N Structurally Valid", "Structurally Valid Target Yield (%)", "VSUN Target Yield (%)"]]


# Output Histogram Plotting Logic 

def _count_series_per_bin(predicted_values, subset_mask, bins):
    """Safely count dataframe subsets into predefined bins."""
    predicted = pd.to_numeric(predicted_values, errors="coerce")
    finite_mask = predicted.notna()

    if subset_mask is None:
        mask = finite_mask
    else:
        mask = subset_mask.reindex(predicted.index, fill_value=False).astype(bool) & finite_mask

    bin_ids = pd.cut(predicted, bins=bins, labels=False, include_lowest=True, right=False)
    counts = np.zeros(len(bins) - 1, dtype=float)
    for bin_index in range(len(bins) - 1):
        counts[bin_index] = float((mask & (bin_ids == bin_index)).sum())
    return counts


def plot_bandgap_output_space_summary(
    df_dict,
    train_df=None,
    train_bg_col="Bandgap (eV)",
    target_bg_col="target_Bandgap (eV)",
    pred_bg_col="ALIGNN_bg (eV)",
    max_bg=8.5, min_bin_count=10,
    figsize=(22, 6.6), title_fontsize=18, label_fontsize=16, ticks_fontsize=14,
    axes_linewidth=1.5, num_yticks=5, num_yticks_ref=3, wspace=0.02, hspace=0.05,
):
    """
    Plots grouped stacked count histograms in generated band-gap space.
    Evaluates all N generated structures.
    """
    bins = np.arange(0, max_bg + 0.5, 0.5)
    bin_width = 0.5
    bin_centers = bins[:-1] + bin_width / 2
    
    stride = max(1, int(round((1.0 if max_bg > 6 else 0.5) / bin_width)))
    x_ticks = bin_centers[::stride]
    x_tick_labels = [f"{v:.2f}".rstrip("0").rstrip(".") for v in x_ticks]

    # Gather requested vertical guide lines
    targets = set()
    for df in df_dict.values():
        if target_bg_col in df.columns:
            targets.update(pd.to_numeric(df[target_bg_col], errors="coerce").dropna().unique())
    requested_targets = [t for t in sorted(map(float, targets)) if 0.0 <= t <= max_bg]

    xs, dens_norm = get_training_density(train_df, train_bg_col, max_val=max_bg)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.25], hspace=hspace, wspace=wspace)

    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    legend_ax = fig.add_subplot(grid[1, :])

    panel_order = {
        False: [m for m in ["PKV", "Slider", "Prepend", "Raw"] if any(get_model_and_pretrain_status(lbl) == (m, False) for lbl in df_dict)],
        True: [m for m in ["PKV", "Slider", "Prepend", "Raw"] if any(get_model_and_pretrain_status(lbl) == (m, True) for lbl in df_dict)],
    }

    y_max = 0.0
    for label, data_frame in df_dict.items():
        if pred_bg_col not in data_frame.columns:
            continue

        method, is_pretrained = get_model_and_pretrain_status(label)
        methods_in_panel = panel_order[is_pretrained]
        if method not in methods_in_panel:
            continue

        valid_mask, vsun_mask = calculate_vsun_masks(data_frame)
        pred = data_frame[pred_bg_col]

        tot_counts = _count_series_per_bin(pred, None, bins)
        val_counts = _count_series_per_bin(pred, valid_mask, bins)
        q_counts = _count_series_per_bin(pred, vsun_mask, bins)

        suff_counts = tot_counts >= min_bin_count
        if not np.any(suff_counts):
            continue

        # Calculate offsets for side-by-side bars
        n_methods = len(methods_in_panel)
        bar_w = bin_width / n_methods
        offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_w
        
        idx = methods_in_panel.index(method)
        x_plot = bin_centers[suff_counts] + offsets[idx]
        
        v_plot = val_counts[suff_counts]
        q_plot = np.minimum(q_counts[suff_counts], v_plot)
        v_only = np.clip(v_plot - q_plot, 0, None)
        color = OKABE_ITO_PRETRAIN[method]

        ax = axes[1 if is_pretrained else 0]
        ax.bar(x_plot, q_plot, width=bar_w, color=color, alpha=0.95, ec=color, lw=axes_linewidth * 0.8, zorder=3, align="center")
        ax.bar(x_plot, v_only, bottom=q_plot, width=bar_w, color=color, alpha=0.28, ec=color, lw=axes_linewidth * 0.8, zorder=3, align="center")

        if np.any(np.isfinite(v_plot)):
            y_max = max(y_max, float(np.nanmax(v_plot)))

    # Apply global styling
    for col, ax in enumerate(axes):
        if dens_norm is not None:
            den_ax = ax.twinx()
            den_ax.fill_between(xs, dens_norm, color="#888888", alpha=0.22, zorder=0)
            den_ax.set_ylim(0, 1)
            den_ax.axis('off')
            ax.patch.set_visible(False)
            
            if col == 1:
                den_ax.spines["right"].set_visible(True)
                den_ax.spines["right"].set_linewidth(axes_linewidth)
                den_ax.spines["right"].set_color("gray")
                den_ax.spines["right"].set_position(("outward", 5))
                den_ax.tick_params(axis="y", which="major", labelsize=ticks_fontsize, colors="gray", right=True, labelright=True)
                den_ax.set_ylabel("Reference density", fontsize=label_fontsize, color="gray")

        ax.set_axisbelow(True)
        ax.set_xlim(0, max_bg)
        ax.set_ylim(0, max(1.0, y_max * 1.12))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks, integer=True, min_n_ticks=3))
        ax.tick_params(axis="both", which="major", labelsize=ticks_fontsize)

        for t in requested_targets:
            ax.axvline(t, color="#000000", alpha=1.0, ls="--", lw=1.5, zorder=0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(axes_linewidth)
        ax.text(-0.0, 1.05, ["(a)", "(b)"][col], transform=ax.transAxes, fontsize=title_fontsize, weight="bold", ha="right")
        ax.set_title(COL_TITLES[col], fontsize=title_fontsize)
        ax.set_xlabel("Generated band-gap [eV]", fontsize=label_fontsize)

        if col == 0:
            ax.spines["left"].set_linewidth(axes_linewidth)
            ax.set_ylabel("Structurally valid generations [count]", fontsize=label_fontsize)
        else:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)

    # Build the massive legend on bottom panel
    legend_ax.axis("off")
    m_handles = [Patch(fc=OKABE_ITO_PRETRAIN[m], ec=OKABE_ITO_PRETRAIN[m], alpha=0.9) for m in ("PKV", "Slider", "Prepend", "Raw")]
    m_leg = legend_ax.legend(m_handles, ["Prefix", "Residual", "Prepend", "Raw"], title="Model", loc="center left", bbox_to_anchor=(0.0, 0.52), frameon=False, fontsize=ticks_fontsize, ncol=4)
    m_leg.get_title().set_fontproperties({'size':title_fontsize, 'weight':'bold'})
    legend_ax.add_artist(m_leg)

    e_handles = [Patch(fc="#990099", ec="#990099", alpha=0.95), Patch(fc="#CF81CF", ec="#CF81CF", alpha=0.95)]
    e_leg = legend_ax.legend(e_handles, ["$\mathregular{Q_{VSUN}}$ subset", "Structurally valid subset"], title="Bar", loc="center", bbox_to_anchor=(0.53, 0.52), frameon=False, fontsize=ticks_fontsize, ncol=2)
    e_leg.get_title().set_fontproperties({'size':title_fontsize, 'weight':'bold'})
    legend_ax.add_artist(e_leg)

    r_handles = [Patch(fc="#888888", alpha=0.22, ec="none"), Line2D([], [], color="#000000", alpha=0.28, ls="--", lw=1.0)]
    r_leg = legend_ax.legend(r_handles, ["Training-set density", "Requested target"], title="Reference", loc="center right", bbox_to_anchor=(1.00, 0.52), frameon=False, fontsize=ticks_fontsize)
    r_leg.get_title().set_fontproperties({'size':title_fontsize, 'weight':'bold'})
    legend_ax.add_artist(r_leg)

    return fig


__all__ = [
    "get_metrics_ptnd_vs_scratch", 
    "plot_pretraining_benefits", 
    "plot_vsun_parity_grid_bandgap", 
    "format_bandgap_metrics_table", 
    "plot_bandgap_output_space_summary"
]