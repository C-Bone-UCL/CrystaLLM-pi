"""Helpers for notebooks/Y_Logits.ipynb."""

from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from _models.PKV_model import PKVGPT
from _tokenizer import CustomCIFTokenizer
from _utils._processing_utils import add_variable_brackets_to_cif, remove_comments

__all__ = [
    "CELL_PARAM_TAGS",
    "CELL_META_TAGS",
    "ATOM_COORD_TAGS",
    "DIGIT_TOKENS",
    "ROW_LABELS",
    "parse_cif_numeric_fields",
    "reconstruct_bracketed_cif",
    "load_model_for_logits",
    "extract_digit_logits",
    "plot_digit_heatmap_grid",
]


# CIF tags that contain numeric values we care about
CELL_PARAM_TAGS = [
    "_cell_length_a",
    "_cell_length_b",
    "_cell_length_c",
    "_cell_angle_alpha",
    "_cell_angle_beta",
    "_cell_angle_gamma",
]

CELL_META_TAGS = [
    "_cell_volume",
    "_cell_formula_units_Z",
]

ATOM_COORD_TAGS = [
    "_atom_site_fract_x",
    "_atom_site_fract_y",
    "_atom_site_fract_z",
]

DIGIT_TOKENS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]

ROW_LABELS = DIGIT_TOKENS + ["other"]
DEFAULT_CMAP = "magma_r"
DEFAULT_DPI = 300
FIGSIZE_PER_PANEL = (3.5, 2.8)

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICKS_FONTSIZE = 10
AXES_LINEWIDTH = 1.5
BOX_LINEWIDTH = 2.0
BOX_COLOR = "black"


def parse_cif_numeric_fields(tokens, id_to_token):
    """Find token positions of numeric CIF fields in a token id sequence."""
    if not id_to_token:
        return []

    first_value = next(iter(id_to_token.values()))
    id_to_str = (
        {tid: tok for tok, tid in id_to_token.items()}
        if isinstance(first_value, int)
        else id_to_token
    )

    token_strs = [id_to_str.get(t, "<?>") for t in tokens]
    digit_chars = set("0123456789.")
    fields = []

    cell_tag_names = set(CELL_PARAM_TAGS + CELL_META_TAGS)
    coord_tag_names = set(ATOM_COORD_TAGS)

    # Pass 1: cell parameters (tag followed by whitespace then digits)
    for i, token_str in enumerate(token_strs):
        if token_str not in cell_tag_names:
            continue

        digit_positions = []
        digit_tokens = []
        j = i + 1
        while j < len(token_strs) and token_strs[j] in (" ", "["):
            j += 1
        while j < len(token_strs) and token_strs[j] in digit_chars:
            digit_positions.append(j)
            digit_tokens.append(token_strs[j])
            j += 1

        if digit_positions:
            fields.append(
                {
                    "tag": token_str,
                    "digit_positions": digit_positions,
                    "digit_tokens": digit_tokens,
                }
            )

    # Pass 2: atom-site coordinates inside a loop_ table
    loop_header_tags = []
    in_loop_header = False
    loop_body_start = None

    for i, token_str in enumerate(token_strs):
        if token_str == "loop_":
            j = i + 1
            while j < len(token_strs) and token_strs[j] in ("\n", " "):
                j += 1
            if j < len(token_strs) and token_strs[j].startswith("_atom_site_"):
                in_loop_header = True
                loop_header_tags = []
        elif in_loop_header:
            if token_str.startswith("_atom_site_"):
                loop_header_tags.append(token_str)
            elif token_str in ("\n", " "):
                continue
            else:
                in_loop_header = False
                loop_body_start = i
                break

    if not loop_header_tags or loop_body_start is None:
        return fields

    coord_col_indices = {
        col_idx: tag
        for col_idx, tag in enumerate(loop_header_tags)
        if tag in coord_tag_names
    }
    if not coord_col_indices:
        return fields

    i = loop_body_start
    atom_idx = 0
    while i < len(token_strs):
        while i < len(token_strs) and token_strs[i] in ("\n", " ", "["):
            i += 1
        if i >= len(token_strs) or token_strs[i] in ("loop_", "<eos>", "]"):
            break

        col = 0
        while col < len(loop_header_tags) and i < len(token_strs):
            value_positions = []
            value_tokens = []
            while i < len(token_strs) and token_strs[i] not in ("\n", " ", "]"):
                value_positions.append(i)
                value_tokens.append(token_strs[i])
                i += 1
            while i < len(token_strs) and token_strs[i] == " ":
                i += 1

            if col in coord_col_indices and value_tokens:
                digit_pos = [
                    p for p, t in zip(value_positions, value_tokens) if t in digit_chars
                ]
                digit_tok = [t for t in value_tokens if t in digit_chars]
                if digit_pos:
                    fields.append(
                        {
                            "tag": f"{coord_col_indices[col]}_{atom_idx}",
                            "digit_positions": digit_pos,
                            "digit_tokens": digit_tok,
                        }
                    )
            col += 1

        while i < len(token_strs) and token_strs[i] == "\n":
            i += 1
        atom_idx += 1

    return fields


def reconstruct_bracketed_cif(decoded_cif: str) -> str:
    """Restore the bracketed training format from decoded generated CIF text."""
    cleaned_cif = decoded_cif.strip()
    cleaned_cif = cleaned_cif.removeprefix("<bos>").removesuffix("<eos>").strip()
    cleaned_cif = remove_comments(cleaned_cif)
    return add_variable_brackets_to_cif(cleaned_cif)


def _get_condition_dtype(model):
    """Use the conditioning module dtype when available."""
    if hasattr(model, "conditioning"):
        try:
            return next(model.conditioning.parameters()).dtype
        except StopIteration:
            pass

    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def load_model_for_logits(checkpoint_dir, device="cuda"):
    """Load PKV model and tokenizer from a checkpoint, ready for inference."""
    tokenizer = CustomCIFTokenizer.from_pretrained(
        pretrained_dir=checkpoint_dir, pad_token="<pad>"
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if device == "cpu":
        dtype = torch.float32
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    try:
        model = PKVGPT.from_pretrained(
            checkpoint_dir,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).eval().to(device)
    except Exception:
        model = PKVGPT.from_pretrained(
            checkpoint_dir,
            torch_dtype=dtype,
        ).eval().to(device)

    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def extract_digit_logits(
    model,
    tokenizer,
    cif_text: str,
    condition_value: float,
    device: str = "cuda",
    include_coords: bool = True,
    max_coord_atoms: Optional[int] = None,
):
    """Run a teacher-forced pass and return per-field digit probabilities."""
    full_text = f"{tokenizer.bos_token}\n{cif_text}\n{tokenizer.eos_token}"
    token_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)

    condition_tensor = torch.tensor(
        [[condition_value]],
        device=token_ids.device,
        dtype=_get_condition_dtype(model),
    )

    with torch.inference_mode():
        attention_mask = torch.ones_like(token_ids)
        outputs = model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            condition_values=condition_tensor,
        )

    logits = outputs.logits[0].float()
    probs = torch.softmax(logits, dim=-1)
    digit_ids = [tokenizer.token_to_id[d] for d in DIGIT_TOKENS]

    token_id_list = token_ids[0].cpu().tolist()
    fields = parse_cif_numeric_fields(token_id_list, tokenizer.token_to_id)

    if not include_coords:
        fields = [field for field in fields if "fract" not in field["tag"]]

    if max_coord_atoms is not None:
        seen_atoms = set()
        filtered_fields = []
        for field in fields:
            if "fract" in field["tag"]:
                atom_suffix = field["tag"].rsplit("_", 1)[-1]
                if atom_suffix in seen_atoms and len(seen_atoms) >= max_coord_atoms:
                    continue
                seen_atoms.add(atom_suffix)
            filtered_fields.append(field)
        fields = filtered_fields

    for field in fields:
        positions = field["digit_positions"]
        field_probs = np.zeros((len(positions), len(DIGIT_TOKENS) + 1))
        for idx, pos in enumerate(positions):
            if pos == 0:
                continue
            logit_row = probs[pos - 1]
            for d_idx, d_id in enumerate(digit_ids):
                field_probs[idx, d_idx] = logit_row[d_id].item()
            field_probs[idx, -1] = max(0, 1.0 - field_probs[idx, :-1].sum())
        field["probs"] = field_probs

    return {
        "fields": fields,
        "token_ids": token_id_list,
        "all_logits": logits.cpu(),
    }


def plot_digit_heatmap_grid(
    fields: list,
    tags_to_plot: Optional[list] = None,
    ncols: int = 3,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Plot a grid of digit probability heatmaps for CIF numeric fields."""
    if tags_to_plot is not None:
        fields = [field for field in fields if field["tag"] in tags_to_plot]

    n_panels = len(fields)
    if n_panels == 0:
        print("No fields to plot.")
        return None, None

    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIGSIZE_PER_PANEL[0] * ncols, FIGSIZE_PER_PANEL[1] * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    norm = mcolors.Normalize(vmin=0, vmax=1)
    im = None

    for field, ax in zip(fields, axes.flat):
        probs = field["probs"]
        n_positions = probs.shape[0]
        generated = field["digit_tokens"]
        heatmap_data = probs.T

        im = ax.imshow(
            heatmap_data,
            aspect="auto",
            cmap=DEFAULT_CMAP,
            norm=norm,
            interpolation="nearest",
        )

        for pos_idx, tok in enumerate(generated):
            if tok in ROW_LABELS:
                tok_row = ROW_LABELS.index(tok)
                rect = Rectangle(
                    (pos_idx - 0.5, tok_row - 0.5),
                    1,
                    1,
                    linewidth=BOX_LINEWIDTH,
                    edgecolor=BOX_COLOR,
                    facecolor="none",
                )
                ax.add_patch(rect)

        ax.set_yticks(range(len(ROW_LABELS)))
        ax.set_yticklabels(ROW_LABELS, fontsize=TICKS_FONTSIZE)
        ax.set_xticks(range(n_positions))
        ax.set_xticklabels([f"d{i + 1}" for i in range(n_positions)], fontsize=TICKS_FONTSIZE)
        ax.tick_params(axis="both", which="both", length=0)

        for spine in ax.spines.values():
            spine.set_linewidth(AXES_LINEWIDTH)
            spine.set_color("black")

        ax.set_xlabel("Digit Position", fontsize=LABEL_FONTSIZE)
        tag_display = _format_tag(field["tag"])
        value_str = "".join(generated)
        ax.set_title(
            f"{tag_display} = {value_str}",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
        )

    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, aspect=30)
        cbar.set_label("P(token)", fontsize=LABEL_FONTSIZE, weight="bold")
        cbar.ax.tick_params(labelsize=TICKS_FONTSIZE)
        cbar.outline.set_linewidth(AXES_LINEWIDTH)

    if title:
        fig.suptitle(title, fontsize=TITLE_FONTSIZE + 2, fontweight="bold", y=1.02)

    if save_path:
        try:
            fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
            print(f"Saved to {save_path}")
        except Exception as error:
            print(f"Failed to save figure to {save_path}. Error: {error}")

    return fig, axes


def _format_tag(tag: str) -> str:
    """Shorten CIF tags for compact panel titles."""
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

    for coord_tag in ATOM_COORD_TAGS:
        if tag.startswith(coord_tag):
            suffix = tag[len(coord_tag):]
            axis = coord_tag[-1]
            return f"frac_{axis}{suffix}"

    return replacements.get(tag, tag.lstrip("_"))