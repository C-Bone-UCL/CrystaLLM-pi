"""Helpers for notebooks/Y_Dataset_stats.ipynb."""

import os

import numpy as np
import pandas as pd

from ._shared_utils import compute_atom_counts, load_count_datasets

def _draw_hist_series(ax, series, *, bins):
    for item in sorted(series, key=lambda entry: len(entry[0]), reverse=True):
        if len(item) == 3:
            data, color, alpha = item
            label = None
        else:
            data, color, alpha, label = item

        if len(data):
            hist_kwargs = {
                "bins": bins,
                "color": color,
                "edgecolor": "none",
                "alpha": alpha,
            }
            if label is not None:
                hist_kwargs["label"] = label
            ax.hist(data, **hist_kwargs)


def plot_dataset_stats(loaded: list[tuple], atom_density_line: int = 20, save_dir: str = "plots/") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    c_conv_within = "#E69F00"
    c_prim_within = "#009E73"
    c_conv_above = "#CC4125"
    c_prim_above = "#FF9980"

    axes_lw, label_fs, tick_fs, annot_fs = 1.2, 13, 11, 10

    def _style_ax(ax):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(axes_lw)
        ax.tick_params(axis="both", which="major", labelsize=tick_fs)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))

    os.makedirs(save_dir, exist_ok=True)

    for name, df, ctx in loaded:
        if df is None:
            print(f"[SKIP] {name}")
            continue

        token_counts = df["token_count"]
        n_over = int((token_counts > ctx).sum())
        pct_over = 100.0 * n_over / len(token_counts)

        fig, (ax_tok, ax_atom) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
        fig.suptitle(name, fontsize=label_fs + 1, fontweight="bold")

        tok_bins = np.histogram_bin_edges(token_counts, bins=80)
        tok_series = [
            (token_counts[token_counts <= ctx], c_prim_within, 0.85),
            (token_counts[token_counts > ctx], c_conv_above, 0.85),
        ]
        _draw_hist_series(ax_tok, tok_series, bins=tok_bins)

        ax_tok.axvline(ctx, color="black", linestyle="--", linewidth=1.4)
        xmin, xmax = ax_tok.get_xlim()
        _, ymax = ax_tok.get_ylim()
        ax_tok.text(
            ctx - (xmax - xmin) * 0.02,
            ymax * 0.96,
            f"ctx = {ctx:,}",
            color="black",
            fontsize=annot_fs,
            ha="right",
            va="top",
        )
        ax_tok.text(
            ctx + (xmax - xmin) * 0.02,
            ymax * 0.96,
            f"{pct_over:.1f}% beyond context  (n={n_over:,})",
            color=c_conv_above,
            fontsize=annot_fs,
            ha="left",
            va="top",
        )
        ax_tok.set_xlabel("Token count", fontsize=label_fs)
        ax_tok.set_ylabel("Structures", fontsize=label_fs)
        _style_ax(ax_tok)

        if "conv_count" not in df.columns or "prim_count" not in df.columns:
            ax_atom.text(
                0.5,
                0.5,
                "Run compute_atom_counts() first",
                ha="center",
                va="center",
                transform=ax_atom.transAxes,
                fontsize=annot_fs,
                color="gray",
            )
        else:
            valid = df["conv_count"].notna()
            within = df["token_count"] <= ctx
            conv_within = df.loc[valid & within, "conv_count"].astype(int)
            conv_above = df.loc[valid & ~within, "conv_count"].astype(int)
            prim_within = df.loc[valid & within, "prim_count"].astype(int)
            prim_above = df.loc[valid & ~within, "prim_count"].astype(int)

            all_counts = pd.concat([conv_within, conv_above, prim_within, prim_above])
            if not all_counts.empty:
                max_bin = min(int(all_counts.max()), 300)
                atom_series = [
                    (conv_within.clip(upper=max_bin), c_conv_within, 0.75, f"Conventional <=ctx  (n={len(conv_within):,})"),
                    (prim_within.clip(upper=max_bin), c_prim_within, 0.80, f"Primitive <=ctx  (n={len(prim_within):,})"),
                    (conv_above.clip(upper=max_bin), c_conv_above, 0.85, f"Conventional >ctx  (n={len(conv_above):,})"),
                    (prim_above.clip(upper=max_bin), c_prim_above, 0.85, f"Primitive >ctx  (n={len(prim_above):,})"),
                ]
                _draw_hist_series(ax_atom, atom_series, bins=80)

                ax_atom.axvline(atom_density_line, color="black", linestyle="--", linewidth=1.2, label=f"{atom_density_line} atoms")
                ax_atom.legend(fontsize=annot_fs - 1, frameon=False, loc="upper right")
                cap_label = (
                    f"Atoms per unit cell  (capped at {max_bin})"
                    if int(all_counts.max()) > max_bin
                    else "Atoms per unit cell"
                )
                ax_atom.set_xlabel(cap_label, fontsize=label_fs)
            else:
                ax_atom.text(
                    0.5,
                    0.5,
                    "No parseable geometries",
                    ha="center",
                    va="center",
                    transform=ax_atom.transAxes,
                    fontsize=annot_fs,
                    color="gray",
                )

        ax_atom.set_ylabel("Structures", fontsize=label_fs)
        ax_atom.set_title(f"Atom count per unit cell  ({len(df):,} structures total)", fontsize=label_fs)
        _style_ax(ax_atom)

        save_path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
        plt.show()
        plt.close(fig)


__all__ = [
    "load_count_datasets",
    "compute_atom_counts",
    "plot_dataset_stats",
]