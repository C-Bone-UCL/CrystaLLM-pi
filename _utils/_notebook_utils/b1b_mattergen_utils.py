"""Helpers for B1b_Mattergen.ipynb to analyze structure generation metrics.

This script provides utilities for processing generated CIF files, computing
atom count distributions, annotating structural features (spacegroups, system types),
and plotting comparative parity grids and output space summaries for
CrystaLLM-pi and Mattergen models.
"""

import re
import string
import warnings
import concurrent.futures
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from tqdm import tqdm

from ._shared_utils import compute_atom_counts, load_count_datasets
from .b1a_pretrain_benefits_utils import (
    calculate_vsun_masks,
    get_training_density,
    _count_series_per_bin,
)

# Constants for model comparison mapping
MG_COLORS = {"PKV": "#990099", "Mattergen": "#FF4400"}
MG_COL_TITLES = ["CrystaLLM-Prefix Scratch", "Mattergen Scratch"]

CRYSTAL_SYSTEM_ORDER = (
    "p1",
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
)
CRYSTAL_SYSTEM_COLORS = {
    "p1": "#222222",
    "triclinic": "#4C78A8",
    "monoclinic": "#72B7B2",
    "orthorhombic": "#54A24B",
    "tetragonal": "#EECA3B",
    "trigonal": "#F58518",
    "hexagonal": "#E45756",
    "cubic": "#B279A2",
}
SPACEGROUP_TO_CRYSTAL_SYSTEM = {
    spacegroup_number: SpaceGroup.from_int_number(spacegroup_number).crystal_system
    for spacegroup_number in range(1, 231)
}

_DEFAULT_HIT_TOL_EV = 0.5


def _add_reference_density_axis(ax, x_values, density_values, show_right_axis=False, ticks_fontsize=14, label_fontsize=16, axes_linewidth=1.5, num_yticks_ref=3):
    """Overlay a normalized reference density distribution as a background shaded region."""
    if x_values is None or density_values is None or len(x_values) == 0:
        return

    density_ax = ax.twinx()
    density_ax.fill_between(x_values, density_values, color="#888888", alpha=0.18, zorder=0)
    density_ax.set_ylim(0, 1)
    density_ax.axis("off")
    ax.patch.set_visible(False)

    if show_right_axis:
        density_ax.spines["right"].set_visible(True)
        density_ax.spines["right"].set_linewidth(axes_linewidth)
        density_ax.spines["right"].set_color("gray")
        density_ax.tick_params(
            axis="y",
            which="major",
            labelsize=ticks_fontsize,
            colors="gray",
            right=True,
            labelright=True,
        )
        density_ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks_ref, prune="upper"))
        density_ax.set_ylabel("Reference density", fontsize=label_fontsize, color="gray")

    return density_ax


def _get_common_targets(dataframes, target_col):
    """Extract and sort a unique list of target bandgaps across multiple dataframes."""
    common_targets = set()
    for df in dataframes:
        if df is not None and target_col in df.columns:
            common_targets.update(pd.to_numeric(df[target_col], errors="coerce").dropna().unique())
    return sorted(map(float, common_targets))


def compute_atom_count_distribution(df):
    """Calculate the normalized probability distribution of primitive atom counts."""
    count_series = df['prim_count']
    value_counts = count_series.value_counts(normalize=True).sort_index()
    return value_counts.to_dict()


def _get_spacegroup_number(cif_string):
    """Extract the spacegroup IT number from a CIF string via regex to bypass full parsing."""
    if not isinstance(cif_string, str):
        return 0
    pattern = r"(?:_space_group_IT_number|_symmetry_Int_Tables_number)\s+['\"]?(\d+)['\"]?"
    match = re.search(pattern, cif_string)
    return int(match.group(1)) if match else 0


def _get_system_type_worker(cif_text):
    """Parse a CIF string to count the number of unique elements present."""
    if not isinstance(cif_text, str):
        return 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            struct = Structure.from_str(cif_text, fmt="cif")
        return len(struct.composition.elements)
    except Exception:
        return 0


def annotate_structural_features(df, cif_col="Generated CIF", num_workers=32):
    """Append parsed system types and regex-extracted spacegroup numbers to the dataset."""
    df = df.copy()

    if "spacegroup_number" not in df.columns:
        df["spacegroup_number"] = df[cif_col].apply(_get_spacegroup_number)
    else:
        print("spacegroup_number already cached, skipping.")

    if "system_type" not in df.columns:
        cifs = df[cif_col].tolist()
        chunksize = max(1, len(cifs) // (num_workers * 4))
        print(f"Extracting system types for {len(cifs):,} CIFs ({num_workers} workers)...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(_get_system_type_worker, cifs, chunksize=chunksize),
                total=len(cifs),
                desc="System types",
            ))
        df["system_type"] = results
        n_ok = (pd.Series(results) > 0).sum()
        print(f"  {n_ok:,} / {len(cifs):,} parsed successfully")
    else:
        print("system_type already cached, skipping.")

    return df


def _get_spacegroups_by_symprec_worker(cif_text, symprecs):
    """Compute structural spacegroups across varying symmetry precision tolerances."""
    if not isinstance(cif_text, str):
        return tuple(0 for _ in symprecs)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            struct = Structure.from_str(cif_text, fmt="cif")

        values = []
        for symprec in symprecs:
            try:
                analyzer = SpacegroupAnalyzer(struct, symprec=float(symprec))
                values.append(int(analyzer.get_space_group_number()))
            except Exception:
                values.append(0)
        return tuple(values)
    except Exception:
        return tuple(0 for _ in symprecs)


def annotate_spacegroups_by_symprec(
    df,
    cif_col="Generated CIF",
    symprecs=(0.01, 0.1, 0.2),
    num_workers=32,
):
    """Append spacegroup classifications derived from pymatgen symmetry analysis."""
    df = df.copy()
    symprecs = tuple(float(v) for v in symprecs)
    col_names = [f"spacegroup_symprec_{str(v).replace('.', 'p')}" for v in symprecs]
    missing_cols = [col for col in col_names if col not in df.columns]

    if not missing_cols:
        print("Symmetry-derived spacegroup columns already cached, skipping.")
        return df

    cifs = df[cif_col].tolist()
    chunksize = max(1, len(cifs) // (num_workers * 4))
    print(f"Extracting symmetry-derived spacegroups for {len(cifs):,} CIFs ({num_workers} workers, symprecs={symprecs})...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        rows = list(
            tqdm(
                executor.map(
                    _get_spacegroups_by_symprec_worker,
                    cifs,
                    itertools.repeat(symprecs),
                    chunksize=chunksize,
                ),
                total=len(cifs),
                desc="Spacegroups",
            )
        )

    values_arr = np.asarray(rows, dtype=int)
    for idx, col_name in enumerate(col_names):
        if col_name not in df.columns:
            df[col_name] = values_arr[:, idx]

    return df


def plot_mg_vs_pkv_parity_grid(
    df_pkv,
    df_mg,
    train_df=None,
    train_bg_col="Bandgap (eV)",
    target_bg_col="target_Bandgap (eV)",
    pred_bg_col="ALIGNN_bg (eV)",
    hit_tol_eV=_DEFAULT_HIT_TOL_EV,
    figsize=(18, 6),
    title_fontsize=18, label_fontsize=16, ticks_fontsize=14,
    line_width=2.5, scatter_size=100, axes_linewidth=1.5,
    num_yticks=5, num_yticks_ref=3,
):
    """Visualize generation target adherence using side-by-side parity boxplots."""
    common_targets = _get_common_targets([df_pkv, df_mg], target_bg_col)
    target_positions = np.arange(len(common_targets), dtype=float)
    target_to_pos = dict(zip(common_targets, target_positions))
    target_tick_labels = [f"{t:.1f}".rstrip("0").rstrip(".") for t in common_targets]

    def _map_to_target_axis(values):
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

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    xs, dens_norm = get_training_density(train_df, train_bg_col)
    metrics_data = []
    model_entries = [("PKV", df_pkv), ("Mattergen", df_mg)]

    def _style_parity_axis(ax, col):
        if dens_norm is not None and len(common_targets) > 1:
            _add_reference_density_axis(
                ax, 
                _map_to_target_axis(xs), 
                dens_norm, 
                show_right_axis=(col == 1), 
                ticks_fontsize=ticks_fontsize, 
                label_fontsize=label_fontsize, 
                axes_linewidth=axes_linewidth,
                num_yticks_ref=num_yticks_ref
            )

        # set x and y limits to encompass all target positions with some padding, ensuring visibility of all points # pad max target
        ax.set_xlim(-0.45, max(len(common_targets), 0.55))
        ax.set_ylim(-0.45, max(len(common_targets), 0.55))
        if len(target_positions) > 0:
            ax.set_xticks(target_positions)
            ax.set_xticklabels(target_tick_labels, rotation=20, ha="right")
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

        # set y and x upper limit to the max of target_positions + 0.5 to ensure all points are visible

    for col, (model_key, data_frame) in enumerate(model_entries):
        ax = axes[col]
        color = MG_COLORS[model_key]
        display_title = MG_COL_TITLES[col]

        ax.set_title(display_title, fontsize=title_fontsize, weight="bold")
        ax.set_xlabel("Target band-gap [eV]", fontsize=label_fontsize)
        ax.text(0, 1.05, f"({string.ascii_lowercase[col]})", transform=ax.transAxes, fontsize=title_fontsize, weight="bold", ha="right")
        if col == 0:
            ax.set_ylabel("Generated band-gap [eV]", fontsize=label_fontsize)

        if data_frame is None or data_frame.empty:
            ax.axis("off")
            continue

        valid_mask, vsun_mask = calculate_vsun_masks(data_frame)
        df_vsun = data_frame[vsun_mask]

        target_all = pd.to_numeric(data_frame[target_bg_col], errors="coerce") if target_bg_col in data_frame.columns else pd.Series(np.nan, index=data_frame.index)
        pred_all = pd.to_numeric(data_frame[pred_bg_col], errors="coerce") if pred_bg_col in data_frame.columns else pd.Series(np.nan, index=data_frame.index)
        has_target = target_all.notna()
        total_len = len(data_frame)

        is_hit = (pred_all - target_all).abs().le(hit_tol_eV)
        valid_yield = ((valid_mask & is_hit & has_target).sum() / total_len * 100) if total_len > 0 else 0.0
        vsun_yield = ((vsun_mask & is_hit & has_target).sum() / total_len * 100) if total_len > 0 else 0.0

        t_vsun = pd.to_numeric(df_vsun[target_bg_col], errors="coerce") if target_bg_col in df_vsun.columns else pd.Series(dtype=float)
        p_vsun = pd.to_numeric(df_vsun[pred_bg_col], errors="coerce") if pred_bg_col in df_vsun.columns else pd.Series(dtype=float)
        valid_pairs = t_vsun.notna() & p_vsun.notna()
        abs_err = (p_vsun[valid_pairs] - t_vsun[valid_pairs]).abs()
        r2 = r2_score(t_vsun[valid_pairs], p_vsun[valid_pairs]) if valid_pairs.sum() > 1 else np.nan

        metrics_data.append({
            "Model": display_title,
            "$R^2$": r2,
            "MAE [eV]": abs_err.mean() if len(abs_err) > 0 else np.nan,
            "SE [eV]": abs_err.sem() if len(abs_err) > 0 else np.nan,
            "MSE [eV^2]": (abs_err ** 2).mean() if len(abs_err) > 0 else np.nan,
            "MSE (SE^2) [eV^2]": (abs_err ** 2).sem() if len(abs_err) > 0 else np.nan,
            "N_Valid": int(valid_mask.sum()),
            "Valid Target Yield [%]": valid_yield,
            "VSUN Target Yield [%]": vsun_yield,
        })

        plot_pos, plot_data = [], []
        for t_val in sorted(t_vsun.dropna().unique()):
            vals = p_vsun[t_vsun == t_val].dropna()
            if not vals.empty and float(t_val) in target_to_pos:
                plot_pos.append(target_to_pos[float(t_val)])
                plot_data.append(_map_to_target_axis(vals.to_numpy()))

        if len(target_positions) > 0:
            ax.plot(target_positions, target_positions, "k--", lw=line_width * 0.8, alpha=0.8, zorder=0)

        if plot_data:
            ax.boxplot(
                plot_data, positions=plot_pos, widths=0.48, patch_artist=True, manage_ticks=False,
                boxprops=dict(facecolor=color, color=color, alpha=0.6, lw=axes_linewidth),
                capprops=dict(color=color, lw=axes_linewidth),
                whiskerprops=dict(color=color, lw=axes_linewidth),
                flierprops=dict(marker="o", mfc=color, mec="none", alpha=0.3, ms=max(4.0, np.sqrt(scatter_size) / 3.0)),
                medianprops=dict(color="black", lw=line_width),
            )

        else:
            ax.text(0.5, 0.5, "No $Q_{\\mathrm{VSUN}}$ samples", transform=ax.transAxes,
                    fontsize=label_fontsize, color="#444444", ha="center", va="center")

        _style_parity_axis(ax, col)

    return fig, pd.DataFrame(metrics_data)


def format_mg_vs_pkv_metrics_table(metrics_df):
    """Prepare raw metrics dataframe for final tabular display or LaTeX rendering."""
    if metrics_df.empty:
        return metrics_df

    table = metrics_df.copy()
    table["$R^2$"] = table["$R^2$"].map(lambda v: "--" if pd.isna(v) else f"{v:.2f}")
    table["MAE (+/- SE)"] = table.apply(
        lambda r: f"{r['MAE [eV]']:.2f} ({r['SE [eV]']:.3f})" if pd.notna(r["MAE [eV]"]) else "--", axis=1
    )
    table["MSE (+/- SE^2)"] = table.apply(
        lambda r: f"{r['MSE [eV^2]']:.3f} ({r['MSE (SE^2) [eV^2]']:.3f})" if pd.notna(r["MSE [eV^2]"]) else "--", axis=1
    )
    table["N Structurally Valid"] = table["N_Valid"].fillna(0).astype(int)
    table["Valid Target Yield (%)"] = table["Valid Target Yield [%]"].map(lambda v: f"{v:.2f}")
    table["VSUN Target Yield (%)"] = table["VSUN Target Yield [%]"].map(lambda v: f"{v:.2f}")
    return table[["Model", "$R^2$", "MAE (+/- SE)", "MSE (+/- SE^2)", "N Structurally Valid",
                  "Valid Target Yield (%)", "VSUN Target Yield (%)"]]


def plot_mg_vs_pkv_output_space_summary(
    df_pkv,
    df_mg,
    train_df=None,
    train_bg_col="Bandgap (eV)",
    target_bg_col="target_Bandgap (eV)",
    pred_bg_col="ALIGNN_bg (eV)",
    max_bg=8.5, min_bin_count=10,
    figsize=(12, 6.6), title_fontsize=18, label_fontsize=16, ticks_fontsize=14,
    axes_linewidth=1.5, num_yticks=5, num_yticks_ref=3,
):
    """Compare the aggregate predicted bandgap distributions against the training set."""
    bins = np.arange(0, max_bg + 0.5, 0.5)
    bin_width = 0.5
    bar_width = 0.22 
    bin_centers = bins[:-1] + bin_width / 2

    stride = max(1, int(round((1.0 if max_bg > 6 else 0.5) / bin_width)))
    x_ticks = bin_centers[::stride]
    x_tick_labels = [f"{v:.2f}".rstrip("0").rstrip(".") for v in x_ticks]

    common_targets = _get_common_targets([df_pkv, df_mg], target_bg_col)
    requested_targets = [t for t in common_targets if 0.0 <= t <= max_bg]

    xs, dens_norm = get_training_density(train_df, train_bg_col, max_val=max_bg)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    model_entries = [("PKV", df_pkv), ("Mattergen", df_mg)]
    y_max = 0.0

    for col, (model_key, data_frame) in enumerate(model_entries):
        if data_frame is None or pred_bg_col not in data_frame.columns:
            continue
        color = MG_COLORS[model_key]
        valid_mask, vsun_mask = calculate_vsun_masks(data_frame)
        pred = data_frame[pred_bg_col]

        tot_counts = _count_series_per_bin(pred, None, bins)
        val_counts = _count_series_per_bin(pred, valid_mask, bins)
        q_counts = _count_series_per_bin(pred, vsun_mask, bins)

        suff_counts = tot_counts >= min_bin_count
        if not np.any(suff_counts):
            continue

        x_plot = bin_centers[suff_counts]
        offset = -bar_width / 2 if col == 0 else bar_width / 2
        x_plot_offset = x_plot + offset

        v_plot = val_counts[suff_counts]
        q_plot = np.minimum(q_counts[suff_counts], v_plot)
        v_only = np.clip(v_plot - q_plot, 0, None)

        ax.bar(x_plot_offset, q_plot, width=bar_width, color=color, alpha=0.95, ec=color, lw=axes_linewidth * 0.8, zorder=3, align="center")
        ax.bar(x_plot_offset, v_only, bottom=q_plot, width=bar_width, color=color, alpha=0.28, ec=color, lw=axes_linewidth * 0.8, zorder=3, align="center")

        if np.any(np.isfinite(v_plot)):
            y_max = max(y_max, float(np.nanmax(v_plot)))

    if dens_norm is not None:
        den_ax = _add_reference_density_axis(
            ax, xs, dens_norm, 
            show_right_axis=True, 
            ticks_fontsize=ticks_fontsize, 
            label_fontsize=label_fontsize, 
            axes_linewidth=axes_linewidth
        )
        if den_ax:
            den_ax.spines["right"].set_position(("outward", 5))

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
    ax.spines["left"].set_linewidth(axes_linewidth)
    
    ax.set_title("CrystaLLM-Prefix vs Mattergen Output Space", fontsize=title_fontsize)
    ax.set_xlabel("Generated band-gap [eV]", fontsize=label_fontsize)
    ax.set_ylabel("Structurally valid generations [count]", fontsize=label_fontsize)

    m_handles = [Patch(fc=MG_COLORS["PKV"], ec=MG_COLORS["PKV"], alpha=0.9), Patch(fc=MG_COLORS["Mattergen"], ec=MG_COLORS["Mattergen"], alpha=0.9)]
    m_leg = ax.legend(m_handles, MG_COL_TITLES, title="Model", loc="upper left", bbox_to_anchor=(1.08, 1.0), frameon=False, fontsize=ticks_fontsize)
    m_leg.get_title().set_fontproperties({'size': title_fontsize, 'weight': 'bold'})
    ax.add_artist(m_leg)

    e_handles = [Patch(fc="#888888", alpha=0.95, ec="none"), Patch(fc="#888888", alpha=0.28, ec="none")]
    e_leg = ax.legend(e_handles, ["$\\mathregular{Q_{VSUN}}$", "Valid"], title="Bar", loc="upper left", bbox_to_anchor=(1.08, 0.70), frameon=False, fontsize=ticks_fontsize)
    e_leg.get_title().set_fontproperties({'size': title_fontsize, 'weight': 'bold'})
    ax.add_artist(e_leg)

    r_handles = [Patch(fc="#888888", alpha=0.22, ec="none"), Line2D([], [], color="#000000", alpha=0.99, ls="--", lw=1.0)]
    r_leg = ax.legend(r_handles, ["Training-set density", "Requested target"], title="Reference", loc="upper left", bbox_to_anchor=(1.08, 0.40), frameon=False, fontsize=ticks_fontsize)
    r_leg.get_title().set_fontproperties({'size': title_fontsize, 'weight': 'bold'})
    ax.add_artist(r_leg)

    return fig

def _add_reference_density_axis(ax, x_values, density_values, show_right_axis=False, ticks_fontsize=14, label_fontsize=16, axes_linewidth=1.5, num_yticks_ref=3, fill_alpha=0.18):
    """Overlay a normalized reference density distribution as a background shaded region."""
    if x_values is None or density_values is None or len(x_values) == 0:
        return

    density_ax = ax.twinx()
    density_ax.fill_between(x_values, density_values, color="#888888", alpha=fill_alpha, zorder=0)
    density_ax.set_ylim(0, 1)
    density_ax.axis("off")
    ax.patch.set_visible(False)

    if show_right_axis:
        density_ax.spines["right"].set_visible(True)
        density_ax.spines["right"].set_linewidth(axes_linewidth)
        density_ax.spines["right"].set_color("gray")
        density_ax.tick_params(
            axis="y",
            which="major",
            labelsize=ticks_fontsize,
            colors="gray",
            right=True,
            labelright=True,
        )
        density_ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks_ref, prune="upper"))
        density_ax.set_ylabel("Reference density", fontsize=label_fontsize, color="gray")

    return density_ax


def _add_reference_density_axis(ax, x_values, density_values, show_right_axis=False, ticks_fontsize=14, label_fontsize=16, axes_linewidth=1.5, num_yticks_ref=3, fill_alpha=0.18):
    """Overlay a normalized reference density distribution as a background shaded region for continuous data."""
    if x_values is None or density_values is None or len(x_values) == 0:
        return

    density_ax = ax.twinx()
    density_ax.fill_between(x_values, density_values, color="#888888", alpha=fill_alpha, zorder=0)
    density_ax.set_ylim(0, 1)
    density_ax.axis("off")
    ax.patch.set_visible(False)

    if show_right_axis:
        density_ax.spines["right"].set_visible(True)
        density_ax.spines["right"].set_linewidth(axes_linewidth)
        density_ax.spines["right"].set_color("gray")
        density_ax.tick_params(
            axis="y",
            which="major",
            labelsize=ticks_fontsize,
            colors="gray",
            right=True,
            labelright=True,
        )
        density_ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks_ref, prune="upper"))
        density_ax.set_ylabel("Reference density", fontsize=label_fontsize, color="gray")

    return density_ax


def _compute_crystal_system_distribution(spacegroup_values):
    """Group spacegroup numbers into crystal-system bins with SG1 isolated as P1."""
    if spacegroup_values is None:
        spacegroups = pd.Series(dtype=int)
    else:
        spacegroups = pd.to_numeric(pd.Series(spacegroup_values), errors="coerce").dropna().astype(int)
    spacegroups = spacegroups[(spacegroups >= 1) & (spacegroups <= 230)]

    density = pd.Series(0.0, index=CRYSTAL_SYSTEM_ORDER, dtype=float)
    if spacegroups.empty:
        return density, np.nan

    crystal_systems = spacegroups.map(
        lambda spacegroup_number: "p1"
        if int(spacegroup_number) == 1
        else SPACEGROUP_TO_CRYSTAL_SYSTEM[int(spacegroup_number)]
    )
    counts = crystal_systems.value_counts().reindex(CRYSTAL_SYSTEM_ORDER, fill_value=0).astype(float)
    density = counts / counts.sum()
    p1_pct = float((spacegroups == 1).mean() * 100)
    return density, p1_pct


def _plot_crystal_system_pie(ax, spacegroup_values, title, title_fontsize=18, ticks_fontsize=14):
    """Plot a single crystal-system pie chart and include the P1 fraction in the title."""
    density, p1_pct = _compute_crystal_system_distribution(spacegroup_values)
    nonzero_density = density[density > 0]
    p1_text = "P1: --" if pd.isna(p1_pct) else f"P1: {p1_pct:.1f}%"

    if nonzero_density.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=ticks_fontsize)
    else:
        ax.pie(
            nonzero_density.values,
            colors=[CRYSTAL_SYSTEM_COLORS[name] for name in nonzero_density.index],
            startangle=90,
            counterclock=False,
            radius=0.9,
            wedgeprops=dict(edgecolor="white", linewidth=1.0),
        )

    ax.set_title(f"{title}\n{p1_text}", fontsize=title_fontsize)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_mg_structural_distributions(
    df_pkv,
    df_mg,
    train_df=None,
    conv_count_col="conv_count",
    system_type_col="system_type",
    spacegroup_col="spacegroup_number",
    spacegroup_symprec_cols=("spacegroup_symprec_0p01", "spacegroup_symprec_0p1", "spacegroup_symprec_0p2"),
    train_spacegroup_ref_col="spacegroup_symprec_0p1",
    atom_bin_width=4,
    sg_bin_width=10,
    max_system_type=6,
    figsize=(18, 16),  # Increased height to accommodate 4 total rows
    title_fontsize=18, label_fontsize=16, ticks_fontsize=14,
    axes_linewidth=1.5, bar_alpha=0.65,
):
    """Display top-level structural distributions and crystal-system pie summaries."""
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=(1.0, 1.05, 1.05), width_ratios=(1.05, 1.0, 1.0, 1.0))

    ax_atom = fig.add_subplot(gs[0, :2])
    ax_sys = fig.add_subplot(gs[0, 2:])
    ax_ref = fig.add_subplot(gs[1:, 0])
    axes_pkv = [fig.add_subplot(gs[1, col]) for col in range(1, 4)]
    axes_mg = [fig.add_subplot(gs[2, col]) for col in range(1, 4)]

    axes_top = [ax_atom, ax_sys]
    axes_pies = [ax_ref, *axes_pkv, *axes_mg]
    all_axes = axes_top + axes_pies

    model_entries = [("PKV", df_pkv), ("Mattergen", df_mg)]
    panel_labels = [f"({string.ascii_lowercase[idx]})" for idx in range(len(all_axes))]

    train_atom = None
    train_sys = None
    train_sg = None
    if train_df is not None:
        if conv_count_col in train_df.columns:
            train_atom = pd.to_numeric(train_df[conv_count_col], errors="coerce").dropna()
        if system_type_col in train_df.columns:
            train_sys = pd.to_numeric(train_df[system_type_col], errors="coerce").dropna()
        if train_spacegroup_ref_col in train_df.columns:
            train_sg = pd.to_numeric(train_df[train_spacegroup_ref_col], errors="coerce").dropna()
        elif spacegroup_col in train_df.columns:
            train_sg = pd.to_numeric(train_df[spacegroup_col], errors="coerce").dropna()

    # (a) Atom count
    ax = ax_atom
    all_counts = pd.concat([
        pd.to_numeric(df[conv_count_col], errors="coerce").dropna()
        for _, df in model_entries
        if df is not None and conv_count_col in df.columns
    ])
    if not all_counts.empty:
        max_count = int(all_counts.max())
        bins_atom = np.arange(0, max_count + atom_bin_width + 1, atom_bin_width)
        for i, (model_key, df) in enumerate(model_entries):
            if df is None or conv_count_col not in df.columns:
                continue
            vals = pd.to_numeric(df[conv_count_col], errors="coerce").dropna()
            color = MG_COLORS[model_key]
            display_title = MG_COL_TITLES[i]
            
            ax.hist(vals, bins=bins_atom, density=True, color=color,
                    histtype='step', linewidth=2.5, alpha=0.9, 
                    label=display_title)
                    
        if train_atom is not None and not train_atom.empty:
            t_counts, _ = np.histogram(train_atom, bins=bins_atom, density=True)
            if np.nanmax(t_counts) > 0:
                _add_reference_density_axis(ax, bins_atom[:-1] + atom_bin_width / 2, t_counts / np.nanmax(t_counts), fill_alpha=0.35)
                
    ax.set_xlabel("Atoms in unit cell", fontsize=label_fontsize)
    ax.set_ylabel("Density", fontsize=label_fontsize)
    ax.set_title("Atom count", fontsize=title_fontsize)
    ax.legend(frameon=False, fontsize=ticks_fontsize, loc="upper right")

    # (b) System type
    ax = ax_sys
    sys_types = list(range(1, max_system_type + 2))
    bar_w = 0.35
    tick_labels = [str(t) for t in range(1, max_system_type + 1)] + [f"{max_system_type + 1}+"]
    for i, (model_key, df) in enumerate(model_entries):
        if df is None or system_type_col not in df.columns:
            continue
        n_total = len(df)
        st = df[system_type_col].dropna().astype(int)
        st = st.clip(upper=max_system_type + 1)
        st = st[st > 0]
        counts = st.value_counts().reindex(sys_types, fill_value=0)
        fracs = counts.values / max(n_total, 1)
        offset = (i - 0.5) * bar_w
        color = MG_COLORS[model_key]
        display_title = MG_COL_TITLES[i]
        
        ax.bar(np.array(sys_types) + offset, fracs, width=bar_w,
               color=color, alpha=bar_alpha, label=display_title,
               ec=color, lw=0.5)
               
    if train_sys is not None and not train_sys.empty:
        t_sys = train_sys.astype(int).clip(lower=1, upper=max_system_type + 1)
        t_sys = t_sys[t_sys > 0]
        t_counts = t_sys.value_counts().reindex(sys_types, fill_value=0).values.astype(float)
        if np.nanmax(t_counts) > 0:
            _add_reference_density_axis(ax, np.array(sys_types), t_counts / np.nanmax(t_counts), fill_alpha=0.35)
            
    ax.set_xticks(sys_types)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Unique elements in structure", fontsize=label_fontsize)
    ax.set_ylabel("Fraction of total generated", fontsize=label_fontsize)
    ax.set_title("System type", fontsize=title_fontsize)

    ref_label = train_spacegroup_ref_col.split("_")[-1].replace("p", ".") if train_spacegroup_ref_col else None
    ref_title = "Reference" if not ref_label else f"Reference\nsymprec={ref_label}"
    _plot_crystal_system_pie(ax_ref, train_sg, ref_title, title_fontsize=title_fontsize, ticks_fontsize=ticks_fontsize)

    for idx, col in enumerate(spacegroup_symprec_cols):
        symprec_val = col.split("_")[-1].replace("p", ".")
        pkv_values = None if df_pkv is None or col not in df_pkv.columns else df_pkv[col]
        mg_values = None if df_mg is None or col not in df_mg.columns else df_mg[col]

        _plot_crystal_system_pie(
            axes_pkv[idx],
            pkv_values,
            f"Prefix\nsymprec={symprec_val}",
            title_fontsize=title_fontsize,
            ticks_fontsize=ticks_fontsize,
        )
        _plot_crystal_system_pie(
            axes_mg[idx],
            mg_values,
            f"Mattergen\nsymprec={symprec_val}",
            title_fontsize=title_fontsize,
            ticks_fontsize=ticks_fontsize,
        )

    crystal_system_legend = fig.legend(
        [Patch(fc=CRYSTAL_SYSTEM_COLORS[name], ec="white") for name in CRYSTAL_SYSTEM_ORDER],
        [name.capitalize() for name in CRYSTAL_SYSTEM_ORDER],
        title="Crystal system",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        frameon=False,
        fontsize=title_fontsize,
    )
    crystal_system_legend.get_title().set_fontproperties({'size': title_fontsize + 2, 'weight': 'bold'})

    # Apply consistent styling to all panels
    for i, ax_panel in enumerate(all_axes):
        ax_panel.text(0, 1.05, panel_labels[i], transform=ax_panel.transAxes,
                fontsize=title_fontsize, fontweight="bold", ha="right", va="bottom")

    for ax_panel in axes_top:
        ax_panel.spines["top"].set_visible(False)
        ax_panel.spines["right"].set_visible(False)
        ax_panel.spines["left"].set_linewidth(axes_linewidth)
        ax_panel.spines["bottom"].set_linewidth(axes_linewidth)
        ax_panel.tick_params(axis="both", which="major", labelsize=ticks_fontsize)

    for ax_panel in axes_pies:
        for spine in ax_panel.spines.values():
            spine.set_visible(False)

    return fig


__all__ = [
    "load_count_datasets",
    "compute_atom_counts",
    "compute_atom_count_distribution",
    "annotate_structural_features",
    "annotate_spacegroups_by_symprec",
    "apply_system_type_filter",
    "plot_mg_vs_pkv_parity_grid",
    "format_mg_vs_pkv_metrics_table",
    "plot_mg_vs_pkv_output_space_summary",
    "plot_mg_structural_distributions",
]