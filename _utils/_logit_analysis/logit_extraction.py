"""
Load a model checkpoint and extract per-position logits via teacher-forced
forward pass. Returns softmax probabilities for digit tokens at each numeric
field position in a CIF.
"""

import torch
import numpy as np
from typing import Optional

from _tokenizer import CustomCIFTokenizer
from _models.PKV_model import PKVGPT
from .cif_parser import parse_cif_numeric_fields


DIGIT_TOKENS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]


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
            checkpoint_dir, torch_dtype=dtype, attn_implementation="sdpa"
        ).eval().to(device)
    except Exception:
        model = PKVGPT.from_pretrained(
            checkpoint_dir, torch_dtype=dtype
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
    """
    Teacher-forced forward pass on a CIF string. Returns per-field digit
    probability distributions.

    The CIF should be in the TRAINING format (with [...] brackets). If you have
    a decoded CIF (no brackets), use reconstruct_bracketed_cif() first.

    Returns dict with:
        - fields: list of {tag, digit_positions, digit_tokens, probs}
          where probs is shape (n_digits, 12) — 10 digits + '.' + 'other'
        - token_ids: full token ID sequence
        - all_logits: raw logits tensor (seq_len, vocab_size)
    """
    # Wrap CIF in bos/eos as training does
    full_text = f"{tokenizer.bos_token}\n{cif_text}\n{tokenizer.eos_token}"
    token_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)

    # Build condition tensor with the dtype expected by the conditioning path.
    condition_dtype = _get_condition_dtype(model)
    condition_tensor = torch.tensor(
        [[condition_value]],
        device=token_ids.device,
        dtype=condition_dtype,
    )

    # Forward pass
    with torch.inference_mode():
        attention_mask = torch.ones_like(token_ids)
        outputs = model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            condition_values=condition_tensor,
        )
    # logits shape: (1, seq_len, vocab_size)
    logits = outputs.logits[0].float()  # cast to fp32 for stable softmax
    probs = torch.softmax(logits, dim=-1)

    # Map digit token strings to their vocab IDs
    digit_ids = [tokenizer.token_to_id[d] for d in DIGIT_TOKENS]

    # Parse the token sequence to find numeric fields
    token_id_list = token_ids[0].cpu().tolist()
    fields = parse_cif_numeric_fields(token_id_list, tokenizer.token_to_id)

    if not include_coords:
        fields = [f for f in fields if "fract" not in f["tag"]]

    if max_coord_atoms is not None:
        seen_atoms = set()
        filtered = []
        for f in fields:
            if "fract" in f["tag"]:
                atom_suffix = f["tag"].rsplit("_", 1)[-1]
                if atom_suffix in seen_atoms and len(seen_atoms) >= max_coord_atoms:
                    continue
                seen_atoms.add(atom_suffix)
            filtered.append(f)
        fields = filtered

    # Extract probabilities for each field
    for field in fields:
        positions = field["digit_positions"]
        # For autoregressive logits: the logit at position i predicts token at
        # position i+1. So to get P(token at pos j), we look at logits[j-1].
        field_probs = np.zeros((len(positions), len(DIGIT_TOKENS) + 1))
        for idx, pos in enumerate(positions):
            if pos == 0:
                continue
            logit_row = probs[pos - 1]  # logits at prev position predict this token
            for d_idx, d_id in enumerate(digit_ids):
                field_probs[idx, d_idx] = logit_row[d_id].item()
            # "other" column: 1 - sum of digit probs
            field_probs[idx, -1] = max(0, 1.0 - field_probs[idx, :-1].sum())
        field["probs"] = field_probs

    return {
        "fields": fields,
        "token_ids": token_id_list,
        "all_logits": logits.cpu(),
    }
