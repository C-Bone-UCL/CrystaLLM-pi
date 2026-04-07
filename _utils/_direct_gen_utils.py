"""Helpers for direct generation entrypoints.

Includes model metadata, normalization helpers, XRD parsing,
reduced-formula search utilities and final row selection logic.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition
from transformers import AutoConfig

from _utils import is_valid, normalize_values_with_method
from _utils._generating.generate_CIFs import DEFAULT_MAX_LENGTH
from _utils._preprocessing._process_exp_XRD_inputs import process_and_convert

# XRD normalization constants
XRD_TOP_K_PEAKS = 20
XRD_THETA_MIN, XRD_THETA_MAX = 0.0, 90.0
XRD_PADDING_VALUE = -100

MODEL_INFO = {
    "c-bone/CrystaLLM-pi_base": {
        "description": "Unconditional generation",
        "conditions": 0,
        "example_conditions": None,
        "max": None,
        "min": None,
        "normalization": None,
        "model_type": "Base",
    },
    "c-bone/CrystaLLM-pi_SLME": {
        "description": "Solar cell efficiency (SLME) conditioning",
        "conditions": 1,
        "example_conditions": ["25.0"],
        "max": 33.192,
        "min": 0.0,
        "normalization": "linear",
        "model_type": "PKV",
    },
    "c-bone/CrystaLLM-pi_bandgap": {
        "description": "Bandgap + stability conditioning",
        "conditions": 2,
        "example_conditions": ["1.1", "0.0"],
        "max": [17.891, 5.418],
        "min": [0.0, 0.0],
        "normalization": ["power_log", "linear"],
        "model_type": "PKV",
    },
    "c-bone/CrystaLLM-pi_density": {
        "description": "Density + stability conditioning",
        "conditions": 2,
        "example_conditions": ["3.0", "0.0"],
        "max": [25.494, 0.1],
        "min": [0.0, 0.0],
        "normalization": ["linear", "linear"],
        "model_type": "PKV",
    },
    "c-bone/CrystaLLM-pi_COD-XRD": {
        "description": "Experimental XRD conditioning",
        "conditions": 40,
        "example_conditions": "See tests/fixtures/test_rutile_processed.csv",
        "max": [90.0, 100.0],
        "min": [0.0, 0.0],
        "normalization": ["linear", "linear"],
        "model_type": "Slider",
    },
    "c-bone/CrystaLLM-pi_Mattergen-XRD": {
        "description": "Theoretical XRD conditioning",
        "conditions": 40,
        "example_conditions": "See tests/fixtures/test_rutile_processed.csv",
        "max": [90.0, 100.0],
        "min": [0.0, 0.0],
        "normalization": ["linear", "linear"],
        "model_type": "Slider",
    },
}


def _as_list(value, fallback):
    if isinstance(value, list):
        return value
    if value is None:
        return [fallback]
    return [value]

def is_xrd_model(model_path: str) -> bool:
    """Return True when the model path indicates XRD conditioning."""
    return "xrd" in model_path.lower()

def parse_xrd_file_to_condition_vector(file_path: str, wavelength: float = 1.54056) -> List[float]:
    """Parse and process raw XRD file to 40-value condition vector using dynamic processing."""
    try:
        processed_peaks = process_and_convert(file_path, xrd_wavelength=wavelength)
    except Exception as err:
        raise ValueError(f"Failed to process XRD file '{file_path}': {err}")

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

def normalize_property_values(raw_values: List[List[float]], model_path: str) -> List[List[float]]:
    """Normalize property vectors according to model-specific normalization settings."""
    if not raw_values or is_xrd_model(model_path):
        return raw_values

    model_info = MODEL_INFO.get(model_path)
    if not model_info or not model_info.get("normalization"):
        return raw_values

    norm_methods = _as_list(model_info.get("normalization"), "linear")
    min_vals = _as_list(model_info.get("min"), 0.0)
    max_vals = _as_list(model_info.get("max"), 1.0)

    normalized_values = []
    for idx, values in enumerate(raw_values):
        method = norm_methods[idx] if idx < len(norm_methods) else "linear"
        min_val = min_vals[idx] if idx < len(min_vals) else 0.0
        max_val = max_vals[idx] if idx < len(max_vals) else 1.0
        normalized_values.append(normalize_values_with_method(values, method, min_val, max_val))

    return normalized_values

def validate_model_conditions(model_path: str, condition_lists: Optional[List[List[float]]], is_xrd: bool = False) -> None:
    """Validate that provided condition dimensions match model expectations."""
    if is_xrd: return
    model_info = MODEL_INFO.get(model_path)
    if not model_info: return

    expected = model_info["conditions"]
    provided = len(condition_lists) if condition_lists else 0

    if expected > 0 and expected != provided:
        examples = model_info.get("example_conditions", [])
        raise ValueError(
            f"Model {model_path} needs {expected} condition list(s), got {provided}.\n"
            f"Try: {examples}\n({model_info['description']})"
        )

def get_hf_model_max_length(hf_model_path: str) -> int:
    """Fetch max context length from HF config, falling back to default."""
    try:
        cfg = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
        for attr in ("n_positions", "max_position_embeddings", "n_ctx"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    except Exception:
        pass
    return DEFAULT_MAX_LENGTH

def get_visible_gpu_count() -> int:
    """Return visible CUDA device count for current process."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def resolve_multi_gpu_workers(args, n_prompts: int) -> int:
    """Resolve effective GPU worker count for generation."""
    if getattr(args, "output_cif_dir", None):
        return 0

    gpu_count = get_visible_gpu_count()
    if gpu_count < 2 or args.multi_gpu == "false" or n_prompts < 1:
        return 0

    requested_workers = args.num_workers_gpu if args.num_workers_gpu else gpu_count
    worker_count = min(requested_workers, gpu_count) if n_prompts == 1 else min(requested_workers, gpu_count, n_prompts)
    return worker_count if worker_count >= 2 else 0

def parse_reduced_formula_list_arg(reduced_formula_list: str) -> List[str]:
    """Parse comma-separated reduced formulas from CLI input."""
    return [item.strip() for item in str(reduced_formula_list).split(",") if item.strip()]

def canonicalize_reduced_formulas(formulas: Sequence[str]) -> List[str]:
    """Canonicalize formulas using pymatgen reduced formula representation."""
    ordered = OrderedDict()
    for raw_formula in formulas:
        token = str(raw_formula).strip()
        if token == "X":
            ordered["X"] = True
            continue
        ordered[Composition(token).reduced_formula] = True
    if not ordered:
        raise ValueError("No valid reduced formulas were provided")
    return list(ordered.keys())

def _parse_formula_tokens(reduced_formula: str) -> List[Tuple[str, float]]:
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
    formulas: List[str],
    z_mapping: Dict[str, List[int]],
    property_map: Dict[str, dict],
    is_xrd: bool = False,
    xrd_wavelength: float = 1.54056
) -> List[dict]:
    """Build reduced-formula prompt specs using strictly mapped Z values and properties."""
    specs = []
    prompt_order = 0
    
    for formula in formulas:
        zs = z_mapping.get(formula, [1])
        cond_str = None
        
        if is_xrd:
            xrd_file = property_map.get(formula, {}).get("xrd")
            if xrd_file:
                cond_str = ", ".join(str(v) for v in parse_xrd_file_to_condition_vector(xrd_file, xrd_wavelength))
        else:
            cond_str = property_map.get(formula, {}).get("cond")
            
        sg = property_map.get(formula, {}).get("sg")

        for z_val in zs:
            prompt_order += 1
            
            if formula == "X":
                mat_id = "Level1"
                comp_expanded = "X"
            else:
                base_formula = formula.replace(' ', '')
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

def build_formula_condition_map(formulas: List[str], condition_lists_arg: Optional[List[str]], model_path: str) -> Dict[str, Optional[str]]:
    """Map reduced formulas to normalized condition-vector strings."""
    if not condition_lists_arg:
        return {formula: None for formula in formulas}

    raw_condition_vectors = parse_condition_list_args(condition_lists_arg)
    transposed = [list(x) for x in zip(*raw_condition_vectors)]
    validate_model_conditions(model_path, transposed, is_xrd=False)
    
    normalized_transposed = normalize_property_values(transposed, model_path)
    normalized_vectors = [list(x) for x in zip(*normalized_transposed)]
    as_str = [", ".join(str(v) for v in vec) for vec in normalized_vectors]

    if len(as_str) == 1:
        return {formula: as_str[0] for formula in formulas}
    if len(as_str) == len(formulas):
        return {formula: cond for formula, cond in zip(formulas, as_str)}

    raise ValueError(f"Need either 1 condition vector or one per formula. Got {len(as_str)}.")

def parse_condition_list_args(condition_lists_arg: Optional[List[str]]) -> List[List[float]]:
    """Parse CLI condition list strings into vectors of floats."""
    if not condition_lists_arg: return []
    return [[float(x.strip()) for x in cond_str.split(",")] for cond_str in condition_lists_arg]

def attach_prompt_metadata(df_prompts: pd.DataFrame, specs: List[dict]) -> pd.DataFrame:
    """Attach reduced-formula metadata columns to generated prompt dataframe."""
    specs_df = pd.DataFrame(specs)
    if len(df_prompts) != len(specs_df):
        raise ValueError(f"Prompt/spec mismatch: got {len(df_prompts)} prompts but {len(specs_df)} specs")
    
    out = df_prompts.copy().reset_index(drop=True)
    # The fix + robust `.get` assignment
    for col in ["reduced_formula_target", "Z_search", "prompt_order", "Material ID", "condition_vector"]:
        if col in specs_df.columns:
            out[col] = specs_df[col].values
        else:
            out[col] = None
    return out

def _validity_worker(cif_str: str, debug: bool = True) -> bool:
    """Validate one CIF with optional debug traces from `is_valid`."""
    if not cif_str or not isinstance(cif_str, str): return False
    try:
        return bool(is_valid(cif_str, bond_length_acceptability_cutoff=1.0, debug=debug))
    except Exception as err:
        if debug: print(f"Validity worker exception: {err}")
        return False
        
def reduce_rows_for_reduced_formula_search(
    df_generated: pd.DataFrame, 
    df_prompts: pd.DataFrame, 
    formulas_in_order: List[str], 
    scoring_mode: str
) -> pd.DataFrame:
    """Select one best row per reduced formula according to scoring mode."""
    if df_generated.empty:
        return pd.DataFrame()

    prompts_meta = df_prompts[["Material ID", "reduced_formula_target", "Z_search", "prompt_order"]].drop_duplicates()
    
    df_generated["base_Material ID"] = df_generated["Material ID"].str.rsplit('_', n=1).str[0]
    prompts_meta = prompts_meta.rename(columns={"Material ID": "base_Material ID"})
    
    generated = df_generated.merge(prompts_meta, on="base_Material ID", how="left").reset_index(drop=True)
    generated["_generation_order"] = generated.index
    
    if "is_consistent" in generated.columns:
        generated["is_valid"] = generated["is_consistent"].fillna(False)
    elif "is_valid" not in generated.columns:
        generated["is_valid"] = generated["Generated CIF"].apply(lambda cif: _validity_worker(cif, debug=False))

    valid_subset = generated[generated["is_valid"]].copy()

    if valid_subset.empty:
        return pd.DataFrame()

    if scoring_mode == "logp":
        valid_subset["score"] = pd.to_numeric(valid_subset.get("score"), errors="coerce")
        valid_subset = valid_subset[np.isfinite(valid_subset["score"])]
        sorted_subset = valid_subset.sort_values(["score", "prompt_order", "_generation_order"])
    else:
        sorted_subset = valid_subset.sort_values(["prompt_order", "_generation_order"])

    best_rows = sorted_subset.drop_duplicates(subset=["reduced_formula_target"], keep="first")
    best_rows.set_index("reduced_formula_target", inplace=True)

    valid_formulas = [f for f in formulas_in_order if f in best_rows.index]
    out = best_rows.loc[valid_formulas].reset_index() if valid_formulas else pd.DataFrame()
    
    return out.drop(columns=["_generation_order", "Z_search", "prompt_order", "is_valid", "base_Material ID"], errors="ignore")