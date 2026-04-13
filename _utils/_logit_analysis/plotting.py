"""
Plot digit probability heatmaps for CIF numeric fields.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from typing import Optional

from .logit_extraction import DIGIT_TOKENS


ROW_LABELS = DIGIT_TOKENS + ["other"]


def plot_digit_heatmap_grid(
    fields: list,
    tags_to_plot: Optional[list] = None,
    ncols: int = 3,
    figsize_per_panel: tuple = (3.2, 2.4),
    cmap: str = "YlOrRd",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 200,
):
    """
    Plot a grid of digit probability heatmaps.

    Each panel shows one CIF field. Y-axis is digit value (0-9, '.', 'other'),
    X-axis is position within the number. Color = softmax probability.
    The actually generated digit gets a black border.
    """
    if tags_to_plot is not None:
        fields = [f for f in fields if f["tag"] in tags_to_plot]

    n_panels = len(fields)
    if n_panels == 0:
        print("No fields to plot.")
        return None, None

    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    norm = mcolors.Normalize(vmin=0, vmax=1)

    for panel_idx, field in enumerate(fields):
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row, col]

        probs = field["probs"]  # shape (n_digits, 12)
        n_positions = probs.shape[0]
        generated = field["digit_tokens"]

        # Transpose so rows = token values, cols = positions
        heatmap_data = probs.T  # (12, n_positions)

        im = ax.imshow(
            heatmap_data, aspect="auto", cmap=cmap, norm=norm,
            interpolation="nearest",
        )

        # Mark the generated digit with a black rectangle
        for pos_idx, tok in enumerate(generated):
            if tok in ROW_LABELS:
                tok_row = ROW_LABELS.index(tok)
                rect = Rectangle(
                    (pos_idx - 0.5, tok_row - 0.5), 1, 1,
                    linewidth=1.8, edgecolor="black", facecolor="none"
                )
                ax.add_patch(rect)

        ax.set_yticks(range(len(ROW_LABELS)))
        ax.set_yticklabels(ROW_LABELS, fontsize=7)
        ax.set_xticks(range(n_positions))
        ax.set_xticklabels(
            [f"d{i+1}" for i in range(n_positions)], fontsize=7
        )
        ax.set_xlabel("digit position", fontsize=8)

        tag_display = _format_tag(field["tag"])
        value_str = "".join(generated)
        ax.set_title(f"{tag_display} = {value_str}", fontsize=8, fontweight="bold")

    # Hide unused axes
    for panel_idx in range(n_panels, nrows * ncols):
        row = panel_idx // ncols
        col = panel_idx % ncols
        axes[row, col].set_visible(False)

    # Single colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax, label="P(token)"
    )

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)

    fig.subplots_adjust(
        hspace=0.5, wspace=0.35, right=0.91, top=0.92, bottom=0.08
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig, axes


def _format_tag(tag: str) -> str:
    """Shorten tag for display. E.g. '_cell_length_a' → 'cell_a'."""
    replacements = {
        "_cell_length_a": "cell_a",
        "_cell_length_b": "cell_b",
        "_cell_length_c": "cell_c",
        "_cell_angle_alpha": "angle_α",
        "_cell_angle_beta": "angle_β",
        "_cell_angle_gamma": "angle_γ",
        "_cell_volume": "volume",
        "_cell_formula_units_Z": "Z",
    }
    # Handle coord tags with atom index suffix
    for coord_tag in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"):
        if tag.startswith(coord_tag):
            suffix = tag[len(coord_tag):]
            axis = coord_tag[-1]
            return f"frac_{axis}{suffix}"

    return replacements.get(tag, tag.lstrip("_"))
