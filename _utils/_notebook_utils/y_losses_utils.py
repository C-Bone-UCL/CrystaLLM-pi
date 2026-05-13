import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from brokenaxes import brokenaxes

__all__ = ["plot_losses", "plot_component_losses"]

# Okabe-Ito color palette for accessibility
# Mapping repo names to paper names: Slider -> Residual, PKV -> Prefix
STYLE_MAP = {
    "Slider": {"color": "#E69F00", "label": "Residual", "style": (0, (3, 1, 1, 1))},
    "PKV": {"color": "#990099", "label": "Prefix", "style": "-."},
    "Prepend": {"color": "#56B4E9", "label": "Prepend", "style": ":"},
    "Raw": {"color": "#000000", "label": "Raw", "style": (0, (5, 2))}
}

def _load_loss_json(path):
    """Load and parse the losses.json file into numpy arrays."""
    with open(path, 'r') as f:
        d = json.load(f)
    
    return (
        np.array(d.get("training_steps", []), float),
        np.array(d.get("training_losses", []), float),
        np.array(d.get("lm_losses", []), float),
        np.array(d.get("format_losses", []), float)
    )

def _set_axis_style(ax, axes_linewidth, num_xticks, num_yticks, ticks_fontsize):
    """Apply consistent styling to plot axes."""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=num_yticks))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=num_xticks))
    ax.tick_params(axis='both', which='major', labelsize=ticks_fontsize)
    
    for spine in ax.spines.values():
        spine.set_linewidth(axes_linewidth)
    
    if hasattr(ax, 'spines'):
        if 'right' in ax.spines: ax.spines['right'].set_visible(False)
        if 'top' in ax.spines: ax.spines['top'].set_visible(False)

def plot_component_losses(
    losses_path,
    output_path,
    ymax=None,
    ymin=0.0,
    figsize=(12, 8),
    label_fontsize=16,
    ticks_fontsize=14,
    legend_fontsize=14,
    line_width=2.5,
    axes_linewidth=1.5,
    num_xticks=6,
    num_yticks=6,
):
    """Plot LM and Format loss components from a training run."""
    steps, train_loss, lm_loss, format_loss = _load_loss_json(losses_path)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(steps[:train_loss.size], train_loss, color="#000000", 
            linewidth=line_width, label="Total Train Loss")
    
    if lm_loss.size:
        ax.plot(steps[:lm_loss.size], lm_loss, color="#E69F00", 
                linewidth=line_width, linestyle=":", label="LM Loss")
    
    if format_loss.size:
        ax.plot(steps[:format_loss.size], format_loss, color="#990099", 
                linewidth=line_width, linestyle="-.", label="Format Loss")

    if ymax is None:
        valid_losses = [l for l in [train_loss, lm_loss, format_loss] if l.size > 0]
        ymax = max([l.max() for l in valid_losses]) * 1.05 if valid_losses else 1.0

    ax.set_ylim(ymin, ymax)
    _set_axis_style(ax, axes_linewidth, num_xticks, num_yticks, ticks_fontsize)

    ax.set_xlabel("Trainer Step", fontsize=label_fontsize)
    ax.set_ylabel("Loss", fontsize=label_fontsize)
    ax.legend(loc="upper right", frameon=False, fontsize=legend_fontsize)

    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_losses(
    pretrained_model_losses, 
    finetuned_models_losses, 
    output_path, 
    window=5000, 
    ymax=0.85, 
    ymin=0.2,
    figsize=(12, 8),
    label_fontsize=16,
    ticks_fontsize=14,
    legend_fontsize=14,
    line_width=2.5,
    axes_linewidth=1.5,
    num_xticks=6,
    num_yticks=6
):
    """Compare multiple fine-tuned models (train loss only) against base pre-training."""
    p_steps, p_train, _, _ = _load_loss_json(pretrained_model_losses)
    p_max = p_steps.max()

    fine_data = []
    max_step = 0
    
    for pth in finetuned_models_losses:
        s, t, _, _ = _load_loss_json(pth)
        s_shift = s + p_max
        
        found_key = "Raw"
        for key in STYLE_MAP.keys():
            if key.lower() in str(pth).lower():
                found_key = key
                break
        
        fine_data.append({
            "key": found_key,
            "steps": s_shift,
            "train": t
        })
        max_step = max(max_step, s_shift.max())

    fig = plt.figure(figsize=figsize)
    bax = brokenaxes(
        xlims=((0, window), (p_max - window, max_step - 15000)),
        hspace=0.2, tilt=90, d=0, fig=fig, wspace=0.2,
    )
    
    for ax in bax.axs:
        _set_axis_style(ax, axes_linewidth, num_xticks, num_yticks, ticks_fontsize)

    # Base pre-training lines
    pre_mask_start = p_steps <= window
    pre_mask_end = p_steps >= p_max - window
    
    bax.plot(p_steps[pre_mask_start], p_train[pre_mask_start], color='gray', 
             label="Uncond. Pre-training", linewidth=line_width)
    bax.plot(p_steps[pre_mask_end], p_train[pre_mask_end], color='gray', linewidth=line_width)

    # Fine-tuned models (Train losses only)
    for data in fine_data:
        style = STYLE_MAP[data["key"]]
        bax.plot(
            data["steps"], data["train"],
            color=style["color"],
            linewidth=line_width,
            linestyle=style["style"],
            label=style["label"]
        )
            
    bax.set_ylim(ymin, ymax)
    
    # Manual label positioning for brokenaxes
    fig.text(0.5, 0.07, "Trainer Step", ha="center", va="center", fontsize=label_fontsize)
    fig.text(0.02, 0.5, "LLM Train Loss", ha="center", va="center", rotation="vertical", fontsize=label_fontsize)

    bax.legend(loc="upper right", frameon=False, fontsize=legend_fontsize)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)