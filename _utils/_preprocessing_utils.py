"""Shared helpers for CIF preprocessing and augmentation pipelines."""

import hashlib
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

from _tokenizer import CustomCIFTokenizer
from _utils._processing_utils import (
    add_atomic_props_block,
    add_variable_brackets_to_cif,
    extract_formula_units,
    remove_comments,
    replace_data_formula_with_nonreduced_formula,
    round_numbers,
    semisymmetrize_cif,
)

# Budget-based supercell candidates. Budget N = N increments distributed across 3 axes (each starting at 1).
BUDGET_CANDIDATES = {
    1: [(2, 1, 1), (1, 2, 1), (1, 1, 2)],
    2: [(3, 1, 1), (1, 3, 1), (1, 1, 3), (2, 2, 1), (2, 1, 2), (1, 2, 2)],
    3: [
        (4, 1, 1), (1, 4, 1), (1, 1, 4),
        (3, 2, 1), (3, 1, 2), (2, 3, 1),
        (1, 3, 2), (2, 1, 3), (1, 2, 3),
        (2, 2, 2),
    ],
}
MAX_BUDGET = 3

_WORKER_TOKENIZER = None


def run_parallel_chunks(pool, worker_fn, chunks, total_items):
    """Execute chunk worker in parallel with low-overhead progress display."""
    all_results = []
    with tqdm(total=total_items) as progress:
        for chunk_results, processed_count in pool.imap_unordered(worker_fn, chunks):
            all_results.extend(chunk_results)
            progress.update(processed_count)
    return all_results


def get_train_augmentation_mask(dataframe):
    """Return boolean mask for rows eligible for augmentation."""
    if "Split" in dataframe.columns:
        return dataframe["Split"].astype(str).str.lower().eq("train")
    return pd.Series(np.ones(len(dataframe), dtype=bool), index=dataframe.index)


def strip_variable_brackets(cif_text):
    return str(cif_text).replace("[", "").replace("]", "")


def parse_structure(cif_text):
    """Parse structure from CIF text; supports bracketed training format."""
    return Structure.from_str(strip_variable_brackets(cif_text), fmt="cif")


def _normalize_generated_cif_text(generated_cif, oxi, decimal_places=4):
    """Normalize generated CIF text to training format."""
    if generated_cif is None:
        return None

    formula_units = extract_formula_units(generated_cif)
    if formula_units == 0:
        return None

    cleaned = replace_data_formula_with_nonreduced_formula(generated_cif)
    cleaned = semisymmetrize_cif(cleaned)
    cleaned = add_atomic_props_block(cleaned, oxi)
    cleaned = round_numbers(cleaned, decimal_places=decimal_places)
    cleaned = remove_comments(cleaned)
    return add_variable_brackets_to_cif(cleaned)


def structure_to_augmented_cif(structure, oxi, decimal_places=4, symmetrize=True):
    """Convert structure to CIF, optionally symmetrizing, and normalize to training text format.
    
    Setting symmetrize=False forces CifWriter to use symprec=None (P1 symmetry), 
    preventing pymatgen from collapsing supercells or Niggli cells back to conventional forms.
    """
    try:
        if symmetrize:
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            structure_to_write = sga.get_symmetrized_structure()
            generated = str(CifWriter(structure_to_write, symprec=0.1))
        else:
            # Bypass SpacegroupAnalyzer to preserve exact lattice bounds (writes as P1)
            generated = str(CifWriter(structure, symprec=None))
    except Exception:
        generated = structure.to(fmt="cif")
        
    return _normalize_generated_cif_text(generated, oxi=oxi, decimal_places=decimal_places)


def pick_supercell_params(row_key, seed, budget, exclude=None):
    """Return deterministically selected supercell params for the given budget, or None if unavailable."""
    candidates = [c for c in BUDGET_CANDIDATES[budget] if c != exclude]
    if not candidates:
        return None
    digest = hashlib.sha1(f"{row_key}|{seed}|{budget}".encode("utf-8")).hexdigest()
    return candidates[int(digest[:8], 16) % len(candidates)]


def get_supercell_candidate_sequence(row_key, seed):
    """Yield one supercell candidate per budget level from largest to smallest."""
    for budget in range(MAX_BUDGET, 0, -1):
        params = pick_supercell_params(row_key, seed, budget)
        if params is not None:
            yield params


def _is_equivalent_cell(struct1, struct2):
    """Return True if two structures represent the same unit cell (same atom count and volume)."""
    if len(struct1) != len(struct2):
        return False
    if struct2.volume == 0:
        return False
    return abs(struct1.volume / struct2.volume - 1.0) < 1e-3


def apply_variant_dedup_and_thresholds(
    base_cif,
    token_counts,
    max_augmented_tokens,
    *,
    supercell_1_cif=None,
    supercell_2_cif=None,
    niggli_cif=None,
    primitive_cif=None,
    supercell_cif=None,
):
    """Blank duplicate or over-threshold augmented variants and keep token counts aligned.

    Supports two modes determined by which kwargs are passed:
    - supercell_1_cif / supercell_2_cif  (_cleaning.py, returns 2 variants + counts)
    - niggli_cif / primitive_cif / supercell_cif  (_cleaning_all.py, returns 3 variants + counts)
    """
    base_text = "" if base_cif is None else str(base_cif)
    counts = [int(v) for v in token_counts]

    if niggli_cif is not None or primitive_cif is not None or supercell_cif is not None:
        raw = [niggli_cif or "", primitive_cif or "", supercell_cif or ""]
    else:
        raw = [supercell_1_cif or "", supercell_2_cif or ""]

    normalized = []
    for i, text in enumerate(raw):
        text = "" if text is None else str(text)
        should_blank = (
            text.strip() == ""
            or text == base_text
            or (max_augmented_tokens is not None and counts[i + 1] > max_augmented_tokens)
        )
        if should_blank:
            normalized.append("")
            counts[i + 1] = 0
        else:
            normalized.append(text)

    return (*normalized, counts)


def prepare_variant_row_for_counting(base_cif, *augmented_cifs):
    """Normalize augmented variants before counting to skip duplicate/base-equivalent tokenization."""
    base_text = "" if base_cif is None else str(base_cif)

    def normalize_augmented(value):
        text = "" if value is None else str(value)
        return "" if text.strip() == "" or text == base_text else text

    return (base_text, *(normalize_augmented(v) for v in augmented_cifs))


def init_tokenizer_worker(tokenizer_dir):
    """Load tokenizer once per worker process for augmentation token counting."""
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = CustomCIFTokenizer.from_pretrained(tokenizer_dir, pad_token="<pad>")


def quick_token_count(cif_text):
    """Return token count for a single CIF string, or None if tokenizer is not initialized."""
    if _WORKER_TOKENIZER is None:
        return None
    wrapped = f"{_WORKER_TOKENIZER.bos_token}\n{cif_text}\n{_WORKER_TOKENIZER.eos_token}"
    encoded = _WORKER_TOKENIZER([wrapped], truncation=False)
    return len(encoded["input_ids"][0])


def count_chunk_token_lengths(cif_rows, batch_size):
    """Count token lengths for [base, supercell_1, supercell_2] CIF rows in one chunk."""
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Tokenizer worker is not initialized")

    n_cols = len(cif_rows[0]) if cif_rows else 3
    counts = np.zeros((len(cif_rows), n_cols), dtype=np.int32)
    batch_texts = []
    batch_positions = []

    def flush_batch():
        if not batch_texts:
            return
        encoded = _WORKER_TOKENIZER(batch_texts, truncation=False)
        for pos_idx, token_ids in enumerate(encoded["input_ids"]):
            row_i, col_i = batch_positions[pos_idx]
            counts[row_i, col_i] = len(token_ids)
        batch_texts.clear()
        batch_positions.clear()

    for row_i, variants in enumerate(cif_rows):
        for col_i, cif_text in enumerate(variants):
            if cif_text is None or str(cif_text).strip() == "":
                continue

            wrapped = f"{_WORKER_TOKENIZER.bos_token}\n{cif_text}\n{_WORKER_TOKENIZER.eos_token}"
            batch_texts.append(wrapped)
            batch_positions.append((row_i, col_i))

            if len(batch_texts) >= batch_size:
                flush_batch()

    flush_batch()
    return counts