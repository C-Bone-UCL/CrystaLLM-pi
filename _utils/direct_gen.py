"""Provide helpers for direct generation entry points.

The module contains model metadata, normalisation and XRD parsing utilities,
reduced-formula search helpers, and final row-selection logic.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition
from transformers import AutoConfig

from _utils import is_valid, normalize_values_with_method
from _utils._generating.generate_cifs import DEFAULT_MAX_LENGTH
from _utils._preprocessing.process_exp_xrd_inputs import process_and_convert

# XRD normalization constants
XRD_TOP_K_PEAKS = 20
XRD_THETA_MIN, XRD_THETA_MAX = 0.0, 90.0
XRD_PADDING_VALUE = -100

def _load_custom_model_registry(path: str) -> dict[str, object]:
    """Load model metadata supplied outside the package."""
    with open(path, encoding="utf-8") as file:
        registry = json.load(file)
    if not isinstance(registry, dict):
        raise ValueError("Model registry must be a JSON object keyed by Hugging Face path")
    return registry


# Default registry of published hub models, --model_registry overlays extra entries on top.
_DEFAULT_REGISTRY_PATH = Path(__file__).with_name("model_registry.json")
MODEL_INFO: dict[str, dict] = _load_custom_model_registry(str(_DEFAULT_REGISTRY_PATH))


def _as_list(value: object, fallback: list) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return [fallback]
    return [value]

XRD_FORMATS = {"xrd_top20"}


def get_condition_format(model_path: str) -> str | None:
    """Return the conditioning format a model expects, from the registry.

    One of "scalar", "xrd_top20", or None for unconditional models. Registry
    entries without an explicit format fall back to treating Slider models as top-20 XRD, matching
    the released legacy checkpoints.
    """
    info = MODEL_INFO.get(model_path) or {}
    fmt = info.get("condition_format")
    if fmt is None and info.get("model_type") == "Slider":
        fmt = "xrd_top20"
    return fmt


def is_xrd_model(model_path: str) -> bool:
    """Return True when the model takes XRD conditioning.
    """
    # Path substring kept only for overlay registry entries missing condition_format.
    return get_condition_format(model_path) in XRD_FORMATS or "xrd" in model_path.lower()

def parse_xrd_file_to_condition_vector(file_path: str, wavelength: float = 1.54056) -> list[float]:
    """Parse a raw powder scan into the legacy 40-value condition vector.

    Reads a two-column scan, converts 2theta to Q with `wavelength` (CuKα1 1.54056 Å by default),
    picks the top peaks and formats them for the Slider-family XRD models.
    """
    try:
        processed_peaks = process_and_convert(file_path, xrd_wavelength=wavelength)
    except Exception as err:
        raise ValueError(f"Failed to process XRD file '{file_path}': {err}") from err

    thetas = [p[0] for p in processed_peaks]
    intensities = [p[1] for p in processed_peaks]

    # Pad vectors to expected 20 max peaks
    pad_len = XRD_TOP_K_PEAKS - len(thetas)
    thetas += [XRD_PADDING_VALUE] * pad_len
    intensities += [XRD_PADDING_VALUE] * pad_len

    scaled_thetas = [
        round((t - XRD_THETA_MIN) / (XRD_THETA_MAX - XRD_THETA_MIN), 3) if t != XRD_PADDING_VALUE else t 
        for t in thetas
    ]
    
    scaled_intensities = [
        round(i / 100.0, 3) if i != XRD_PADDING_VALUE else i 
        for i in intensities
    ]

    for i in range(1, len(scaled_intensities)):
        if scaled_intensities[i] > scaled_intensities[i - 1]:
            raise ValueError(f"Intensity values are not in descending order at index {i}: {scaled_intensities[i]} > {scaled_intensities[i - 1]}")

    return scaled_thetas + scaled_intensities

def validate_model_conditions(model_path: str, condition_lists: list[list[float]] | None) -> None:
    """Check that the supplied condition vectors match what the model expects.

    Raises before any generation starts, so a width mismatch surfaces immediately instead of after a
    long run.
    """
    model_info = MODEL_INFO.get(model_path)
    if not model_info: return

    expected = model_info["conditions"]
    provided = len(condition_lists) if condition_lists else 0

    if expected > 0 and expected != provided:
        examples = model_info.get("example_conditions") or []
        example_str = ", ".join(examples) if isinstance(examples, list) else str(examples)
        raise ValueError(
            f"Model {model_path} expects {expected} condition value(s) per formula, got {provided}.\n"
            f'Pass one comma-separated string per formula, e.g.: --condition_lists "{example_str}"\n'
            f"({model_info['description']})"
        )

@lru_cache(maxsize=None)
def get_hf_model_max_length(hf_model_path: str, model_type: str | None = None) -> int:
    """Fetch usable text context length from HF config, falling back to default.

    Cached per (model, type): the Z-search loop asks repeatedly and AutoConfig re-reads the HF cache
    on every call.
    """
    try:
        cfg = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                if model_type == "Prefix":
                    # Prefix families extend wpe by n_prefix_tokens, so the text budget
                    # excludes them. PKV is deliberately NOT subtracted so legacy hub models
                    # generate identically.
                    return max(val - int(getattr(cfg, "n_prefix_tokens", 0) or 0), 1)
                return val
    except Exception:
        pass
    return DEFAULT_MAX_LENGTH

def get_visible_gpu_count() -> int:
    """Return visible CUDA device count for current process."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def resolve_multi_gpu_workers(args: argparse.Namespace, n_prompts: int) -> int:
    """Resolve effective GPU worker count for generation.

    --num_workers_gpu is the single arg: unset uses all visible GPUs, an integer caps the worker
    count, and 1 forces the single-process path.
    """
    gpu_count = get_visible_gpu_count()
    if gpu_count < 2 or n_prompts < 1:
        return 0

    requested_workers = args.num_workers_gpu if args.num_workers_gpu else gpu_count
    worker_count = min(requested_workers, gpu_count) if n_prompts == 1 else min(requested_workers, gpu_count, n_prompts)
    return worker_count if worker_count >= 2 else 0

def parse_reduced_formula_list_arg(reduced_formula_list: str) -> list[str]:
    """Parse comma-separated reduced formulas from CLI input."""
    return [item.strip() for item in str(reduced_formula_list).split(",") if item.strip()]

def canonicalize_reduced_formulas(formulas: Sequence[str]) -> list[str]:
    """Canonicalize formulas using pymatgen reduced formula representation, preserving duplicates."""
    tokens = [str(f).strip() for f in formulas]
    if not tokens:
        raise ValueError("No valid reduced formulas were provided")
    # "X" is the level-1 placeholder and must bypass pymatgen parsing.
    return [t if t == "X" else Composition(t).reduced_formula for t in tokens]

def _parse_formula_tokens(reduced_formula: str) -> list[tuple[str, float]]:
    """Extract chemical symbols and their amounts, maintaining string order."""
    comp = Composition(reduced_formula).as_dict()
    pattern = re.compile(r"([A-Z][a-z]?)")
    
    parsed_order = list(dict.fromkeys(pattern.findall(reduced_formula)))
            
    parsed_order.extend([sym for sym in comp if sym not in parsed_order])

    return [(sym, comp[sym]) for sym in parsed_order]

def _format_atom_count(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-8:
        return str(int(rounded))
    return f"{value:.6f}".rstrip("0").rstrip(".")

def reduced_formula_to_explicit_formula(reduced_formula: str, z_value: int) -> str:
    """Expand reduced formula with explicit stoichiometry for a target Z value."""
    if z_value < 1:
        raise ValueError("z_value must be >= 1")
    tokens = _parse_formula_tokens(reduced_formula)
    return "".join([f"{sym}{_format_atom_count(amount * z_value)}" for sym, amount in tokens])

def build_reduced_formula_specs(
    formulas: list[str],
    z_values: list[int],
    properties: list[dict],
    xrd_format: str | None = None,
    xrd_wavelength: float | None = None
) -> list[dict]:
    """Build one prompt spec per formula row using strictly parallel lists.

    Each index i in formulas/z_values/properties corresponds to one output spec. properties[i] must
    be a dict with keys: xrd (file path or None), sg (str or None), cond (condition-vector string or
    None). With an XRD format set, each spec's condition_vector comes from its raw scan file
    instead: "xrd_top20" yields the legacy 40-value string.
    """
    specs = []
    for prompt_order, (formula, z_val, prop) in enumerate(zip(formulas, z_values, properties), start=1):
        xrd_source = prop.get("xrd") if xrd_format else None
        cond_str = prop.get("cond")

        if xrd_source and xrd_format == "xrd_top20":
            legacy_wavelength = xrd_wavelength if xrd_wavelength is not None else 1.54056
            cond_str = ", ".join(
                str(v) for v in parse_xrd_file_to_condition_vector(xrd_source, legacy_wavelength)
            )

        sg = prop.get("sg")

        if formula == "X":
            mat_id = "Level1"
            comp_expanded = "X"
        else:
            base_formula = formula.replace(" ", "")
            mat_id = f"{base_formula}_Z{z_val}"
            comp_expanded = reduced_formula_to_explicit_formula(formula, z_val)

        specs.append({
            "reduced_formula_target": formula,
            "Z_search": z_val,
            "composition_expanded": comp_expanded,
            "prompt_order": prompt_order,
            "condition_vector": cond_str,
            "spacegroup": sg,
            "Material ID": mat_id,
        })

    return specs

def build_formula_condition_map(formulas: list[str], condition_lists_arg: list[str] | None, model_path: str) -> list[str | None]:
    """Return one normalized condition-vector string per formula, positionally.

    `--condition_lists` accepts either a single string broadcast to every formula or exactly one per
    formula. The result lines up index by index with the formula list, letting downstream code zip
    the two without re-checking lengths.
    """
    if not condition_lists_arg:
        return [None] * len(formulas)

    raw_condition_vectors = parse_condition_list_args(condition_lists_arg)
    # zip truncates to the shortest vector, so bad input would silently
    # drop condition values, so we reject it before any data can be lost.
    lengths = {len(vec) for vec in raw_condition_vectors}
    if len(lengths) > 1:
        raise ValueError(
            "Each --condition_lists string must contain the same number of "
            f"comma-separated values; got lengths {sorted(lengths)}."
        )
    transposed = [list(x) for x in zip(*raw_condition_vectors)]
    validate_model_conditions(model_path, transposed)

    if not is_xrd_model(model_path):
        model_info = MODEL_INFO.get(model_path)
        if model_info and model_info.get("normalization"):
            norm_methods = _as_list(model_info.get("normalization"), "linear")
            min_vals = _as_list(model_info.get("min"), 0.0)
            max_vals = _as_list(model_info.get("max"), 1.0)
            transposed = [
                normalize_values_with_method(
                    values,
                    norm_methods[idx] if idx < len(norm_methods) else "linear",
                    min_vals[idx] if idx < len(min_vals) else 0.0,
                    max_vals[idx] if idx < len(max_vals) else 1.0,
                )
                for idx, values in enumerate(transposed)
            ]

    normalized_vectors = [list(x) for x in zip(*transposed)]
    as_str = [", ".join(str(v) for v in vec) for vec in normalized_vectors]

    if len(as_str) == 1:
        return [as_str[0]] * len(formulas)
    if len(as_str) == len(formulas):
        return list(as_str)

    raise ValueError(f"Need either 1 condition vector or one per formula. Got {len(as_str)}.")

def parse_condition_list_args(condition_lists_arg: list[str] | None) -> list[list[float]]:
    """Parse CLI condition list strings into vectors of floats."""
    if not condition_lists_arg: return []
    return [[float(x.strip()) for x in cond_str.split(",")] for cond_str in condition_lists_arg]

def attach_prompt_metadata(df_prompts: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    """Attach reduced-formula metadata columns to generated prompt dataframe."""
    specs_df = pd.DataFrame(specs)
    if len(df_prompts) != len(specs_df):
        raise ValueError(f"Prompt/spec mismatch: got {len(df_prompts)} prompts but {len(specs_df)} specs")

    out = df_prompts.copy().reset_index(drop=True)
    for col in ["reduced_formula_target", "Z_search", "prompt_order", "Material ID", "condition_vector"]:
        if col in specs_df.columns:
            out[col] = specs_df[col].values
        else:
            out[col] = None
    return out

def reduce_rows_for_reduced_formula_search(
    df_generated: pd.DataFrame,
    df_prompts: pd.DataFrame,
    formulas_in_order: list[str],
    scoring_mode: str
) -> pd.DataFrame:
    """Select the single best generated row for each reduced formula.

    Scored rows sort by ascending perplexity. With scoring off, rows keep generation order and the
    first valid one wins. Ties break on prompt order then generation order,
    so a run is reproducible.
    """
    if df_generated.empty:
        return pd.DataFrame()

    prompts_meta = df_prompts[["Material ID", "reduced_formula_target", "Z_search", "prompt_order"]].drop_duplicates()
    prompts_meta = prompts_meta.rename(columns={"Material ID": "base_Material ID"})

    # assign() keeps the join key local: the caller's dataframe must not gain columns.
    generated = (
        df_generated
        .assign(**{"base_Material ID": df_generated["Material ID"].str.rsplit('_', n=1).str[0]})
        .merge(prompts_meta, on="base_Material ID", how="left")
        .reset_index(drop=True)
    )
    generated["_generation_order"] = generated.index
    
    if "is_consistent" in generated.columns:
        generated["is_valid"] = generated["is_consistent"].fillna(False)
    elif "is_valid" not in generated.columns:
        generated["is_valid"] = generated["Generated CIF"].apply(lambda cif: bool(is_valid(cif, bond_length_acceptability_cutoff=1.0, debug=False)) if cif and isinstance(cif, str) else False)

    valid_subset = generated[generated["is_valid"]].copy()

    if valid_subset.empty:
        return pd.DataFrame()

    if scoring_mode == "logp":
        valid_subset["score"] = pd.to_numeric(valid_subset.get("score"), errors="coerce")
        valid_subset = valid_subset[np.isfinite(valid_subset["score"])]
        # logp (perplexity) is lower-better.
        sorted_subset = valid_subset.sort_values(["score", "prompt_order", "_generation_order"])
    else:
        sorted_subset = valid_subset.sort_values(["prompt_order", "_generation_order"])

    best_rows = sorted_subset.drop_duplicates(subset=["reduced_formula_target"], keep="first")
    best_rows.set_index("reduced_formula_target", inplace=True)

    valid_formulas = [f for f in formulas_in_order if f in best_rows.index]
    out = best_rows.loc[valid_formulas].reset_index() if valid_formulas else pd.DataFrame()
    
    return out.drop(columns=["_generation_order", "Z_search", "prompt_order", "is_valid", "base_Material ID"], errors="ignore")
