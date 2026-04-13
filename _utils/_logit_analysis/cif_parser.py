"""
Parse raw CIF text to identify numeric fields and their token positions.
Restore the bracketed training format from decoded generated CIFs.
"""

from _utils._processing_utils import add_variable_brackets_to_cif, remove_comments


# CIF tags that contain numeric values we care about
CELL_PARAM_TAGS = [
    "_cell_length_a", "_cell_length_b", "_cell_length_c",
    "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
]

CELL_META_TAGS = [
    "_cell_volume", "_cell_formula_units_Z",
]

ATOM_COORD_TAGS = [
    "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
]

def parse_cif_numeric_fields(tokens, id_to_token):
    """
    Given a list of token IDs, find all cell parameter and coordinate numeric
    fields. Returns a list of dicts: {tag, digit_positions, digit_tokens}.

    Each digit_positions entry is the index into `tokens` where the digit
    (or '.') appears. digit_tokens is the corresponding token string.
    """
    id_to_str = {tid: tok for tok, tid in id_to_token.items()} if isinstance(
        next(iter(id_to_token.values())), int) else id_to_token

    token_strs = [id_to_str.get(t, "<?>") for t in tokens]
    digit_chars = set("0123456789.")

    fields = []

    cell_tag_names = set(CELL_PARAM_TAGS + CELL_META_TAGS)
    coord_tag_names = {"_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"}

    # Pass 1: cell parameters (tag followed by space then digits)
    for i, ts in enumerate(token_strs):
        if ts in cell_tag_names:
            digit_positions = []
            digit_tokens = []
            j = i + 1
            # skip space / bracket tokens
            while j < len(token_strs) and token_strs[j] in (" ", "["):
                j += 1
            while j < len(token_strs) and token_strs[j] in digit_chars:
                digit_positions.append(j)
                digit_tokens.append(token_strs[j])
                j += 1
            if digit_positions:
                fields.append({
                    "tag": ts,
                    "digit_positions": digit_positions,
                    "digit_tokens": digit_tokens,
                })

    # Pass 2: atom site coordinates (inside loop_ table rows)
    # Find the column order from the loop_ header
    loop_header_tags = []
    in_loop_header = False
    loop_body_start = None

    for i, ts in enumerate(token_strs):
        if ts == "loop_":
            # Check if next tokens are atom_site header tags
            j = i + 1
            while j < len(token_strs) and token_strs[j] in ("\n", " "):
                j += 1
            if j < len(token_strs) and token_strs[j].startswith("_atom_site_"):
                in_loop_header = True
                loop_header_tags = []
        elif in_loop_header:
            if ts.startswith("_atom_site_"):
                loop_header_tags.append(ts)
            elif ts in ("\n", " "):
                continue
            else:
                in_loop_header = False
                loop_body_start = i
                break

    if not loop_header_tags or loop_body_start is None:
        return fields

    # Find column indices for coordinate tags
    coord_col_indices = {}
    for col_idx, tag in enumerate(loop_header_tags):
        if tag in coord_tag_names:
            coord_col_indices[col_idx] = tag

    if not coord_col_indices:
        return fields

    # Parse table rows: each row has values for each column
    i = loop_body_start
    atom_idx = 0
    while i < len(token_strs):
        # Skip whitespace / newlines
        while i < len(token_strs) and token_strs[i] in ("\n", " ", "["):
            i += 1
        if i >= len(token_strs) or token_strs[i] in ("loop_", "<eos>", "]"):
            break

        # Read one row: one value per column
        col = 0
        while col < len(loop_header_tags) and i < len(token_strs):
            # Collect one value (sequence of non-whitespace tokens)
            value_positions = []
            value_tokens = []
            while i < len(token_strs) and token_strs[i] not in ("\n", " ", "]"):
                value_positions.append(i)
                value_tokens.append(token_strs[i])
                i += 1
            # Skip trailing spaces within the row
            while i < len(token_strs) and token_strs[i] == " ":
                i += 1

            if col in coord_col_indices and value_tokens:
                # Filter to just digit/dot tokens
                digit_pos = [p for p, t in zip(value_positions, value_tokens) if t in digit_chars]
                digit_tok = [t for t in value_tokens if t in digit_chars]
                if digit_pos:
                    fields.append({
                        "tag": f"{coord_col_indices[col]}_{atom_idx}",
                        "digit_positions": digit_pos,
                        "digit_tokens": digit_tok,
                    })
            col += 1

        # Skip newline at end of row
        while i < len(token_strs) and token_strs[i] == "\n":
            i += 1
        atom_idx += 1

    return fields


def reconstruct_bracketed_cif(decoded_cif: str) -> str:
    """
    Restore the exact bracket placement used by the training pipeline for a
    generated CIF decoded with `skip_special_tokens=True`.

    This is intentionally narrow: it re-inserts the tokenizer's bracket tokens
    without changing the generated content. It assumes the CIF already contains
    the atomic-properties block used during generation.
    """
    cleaned_cif = decoded_cif.strip()
    cleaned_cif = cleaned_cif.removeprefix("<bos>").removesuffix("<eos>").strip()
    cleaned_cif = remove_comments(cleaned_cif)
    return add_variable_brackets_to_cif(cleaned_cif)
