"""
This script takes in probability data for CIF generation and creates a grid of 
heatmaps, allowing developers to visually inspect how confident the model was 
for each digit at each position.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from typing import Optional

# Assuming this is imported from your local module
from .logit_extraction import DIGIT_TOKENS

ROW_LABELS = DIGIT_TOKENS + ["other"]
DEFAULT_CMAP = "magma_r"
DEFAULT_DPI = 300
FIGSIZE_PER_PANEL = (3.5, 2.8)

# Typography & Layout
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICKS_FONTSIZE = 10
AXES_LINEWIDTH = 1.5
BOX_LINEWIDTH = 2.0
BOX_COLOR = "black"


def plot_digit_heatmap_grid(
    fields: list,
    tags_to_plot: Optional[list] = None,
    ncols: int = 3,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plots a grid of digit probability heatmaps for visual inspection.
    
    Creates a multi-panel figure where each panel shows the softmax probability 
    distribution for a specific CIF field. The actually generated token is highlighted.
    """
    if tags_to_plot is not None:
        fields = [f for f in fields if f["tag"] in tags_to_plot]

    n_panels = len(fields)
    if n_panels == 0:
        print("No fields to plot.")
        return None, None

    nrows = int(np.ceil(n_panels / ncols))
    
    # constrained_layout natively handles spacing and dynamic colorbar injection
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(FIGSIZE_PER_PANEL[0] * ncols, FIGSIZE_PER_PANEL[1] * nrows),
        squeeze=False,
        constrained_layout=True 
    )

    norm = mcolors.Normalize(vmin=0, vmax=1)
    im = None  # Track the mappable for the colorbar

    # Map each field directly to its corresponding axis
    for field, ax in zip(fields, axes.flat):
        probs = field["probs"]  # shape (n_digits, 12)
        n_positions = probs.shape[0]
        generated = field["digit_tokens"]
        
        # Transpose so rows = token values, cols = positions
        heatmap_data = probs.T  

        im = ax.imshow(
            heatmap_data, aspect="auto", cmap=DEFAULT_CMAP, norm=norm,
            interpolation="nearest"
        )

        # Mark the generated digit with a highlighted rectangle
        for pos_idx, tok in enumerate(generated):
            if tok in ROW_LABELS:
                tok_row = ROW_LABELS.index(tok)
                rect = Rectangle(
                    (pos_idx - 0.5, tok_row - 0.5), 1, 1,
                    linewidth=BOX_LINEWIDTH, edgecolor=BOX_COLOR, facecolor="none"
                )
                ax.add_patch(rect)

        # Typographic and axis styling
        ax.set_yticks(range(len(ROW_LABELS)))
        ax.set_yticklabels(ROW_LABELS, fontsize=TICKS_FONTSIZE)
        ax.set_xticks(range(n_positions))
        ax.set_xticklabels([f"d{i+1}" for i in range(n_positions)], fontsize=TICKS_FONTSIZE)
        
        # Remove physical tick lines but keep labels for a cleaner "heatmap" look
        ax.tick_params(axis='both', which='both', length=0)
        
        # Thicken the bounding box of the heatmap
        for spine in ax.spines.values():
            spine.set_linewidth(AXES_LINEWIDTH)
            spine.set_color("black")

        ax.set_xlabel("Digit Position", fontsize=LABEL_FONTSIZE)

        tag_display = _format_tag(field["tag"])
        value_str = "".join(generated)
        ax.set_title(f"{tag_display} = {value_str}", fontsize=TITLE_FONTSIZE, fontweight="bold")

    # Hide any unused axes in the grid if panels don't perfectly fill the rows
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)

    # Attach a single global colorbar dynamically
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, aspect=30)
        cbar.set_label("P(token)", fontsize=LABEL_FONTSIZE, weight='bold')
        cbar.ax.tick_params(labelsize=TICKS_FONTSIZE)
        cbar.outline.set_linewidth(AXES_LINEWIDTH)

    if title:
        # Pad title slightly so it doesn't crowd the top row
        fig.suptitle(title, fontsize=TITLE_FONTSIZE + 2, fontweight="bold", y=1.02)

    if save_path:
        try:
            fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
            print(f"Saved to {save_path}")
        except Exception as e:
            print(f"Failed to save figure to {save_path}. Error: {e}")

    return fig, axes


def _format_tag(tag: str) -> str:
    """
    Shortens CIF tags for display purposes to prevent title overlapping. 
    E.g., '_cell_length_a' becomes 'cell_a'.
    """
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