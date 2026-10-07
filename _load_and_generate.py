#!/usr/bin/env python3
r"""Load a released CrystaLLM-pi model and generate CIF structures.

Prompts can come from a parquet dataset or reduced formulas. Formula mode can search over Z values
and either stop at the first valid structure or select the highest-scoring candidate. Generation
uses every visible GPU.

`--scoring_mode` supports `LOGP` and `None`. `LOGP` requires `--target_valid_cifs` to be greater
than zero.

Usage:
    ```bash
    python _load_and_generate.py --hf_model_path c-bone/CrystaLLM-pi_ft_alex_mp_20-text \
        --reduced_formula_list "TiO2,SiO2" --z_list "2,4" --output_cif_dir outputs/cifs
    ```
"""

import argparse
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _tokenizer import CustomCIFTokenizer
from _utils._generating.make_prompts import create_manual_prompts
from _utils._generating.generate_cifs import (
    init_tokenizer,
    build_generation_kwargs,
    run_generation_pool,
    _normalize_scoring_mode,
)
from _utils._generating.postprocess import process_dataframe
from _utils import extract_formula_nonreduced
from _args import str_to_bool
from _utils.direct_gen import (
    MODEL_INFO,
    XRD_FORMATS,
    _load_custom_model_registry,
    get_condition_format,
    get_hf_model_max_length,
    resolve_multi_gpu_workers,
    parse_reduced_formula_list_arg,
    canonicalize_reduced_formulas,
    build_reduced_formula_specs,
    build_formula_condition_map,
    parse_condition_list_args,
    attach_prompt_metadata,
    reduce_rows_for_reduced_formula_search
)

TOKENIZER_DIR = "HF-cif-tokenizer"

# Global Generation Constants
DO_SAMPLE = True
TOP_K = 15
TOP_P = 0.95
DEFAULT_Z_LIST = [1, 2, 3, 4, 6]
SIDECAR_SAMPLING_KEYS = (
    "do_sample", "temperature", "top_k", "top_p", "gen_max_length", "num_return_sequences",
    "max_return_attempts", "target_valid_cifs", "scoring_mode", "screening_profile", "level",
)


@lru_cache(maxsize=1)
def _tokenizer() -> CustomCIFTokenizer:
    """Load the CIF tokenizer once per process so the early-stop Z loop can reuse it."""
    return init_tokenizer(TOKENIZER_DIR)


def _broadcast(values: list, n_formulas: int, what: str, parser: argparse.ArgumentParser) -> list:
    """Expand a 1-or-N per-formula CLI list to exactly N entries, erroring on any other length."""
    if len(values) == 1:
        return list(values) * n_formulas
    if len(values) == n_formulas:
        return list(values)
    parser.error(f"Expected 1 or {n_formulas} {what}, got {len(values)}.")


def _postprocess_non_empty_cifs(df: pd.DataFrame, num_workers: int, column_name: str = "Generated CIF") -> pd.DataFrame:
    """Postprocess only non-empty CIF rows and preserve original row order."""
    if df.empty or column_name not in df.columns:
        return df

    non_empty_mask = df[column_name].astype(str).str.strip().ne("")
    if not non_empty_mask.any():
        return df

    df_non_empty = process_dataframe(df[non_empty_mask].copy(), num_workers, column_name)
    df_empty = df[~non_empty_mask].copy()
    return pd.concat([df_non_empty, df_empty], ignore_index=False).sort_index().reset_index(drop=True)


def generate_prompts_from_specs(specs: list, args: argparse.Namespace) -> pd.DataFrame:
    """Build the prompt DataFrame from expanded formula specs.

    Each spec carries an expanded composition, an optional spacegroup, and an optional condition
    vector. Nested condition vectors (the continuous-XRD profiles) are passed through untouched
    rather than parsed from a string, because round-tripping them through `str()` would lose
    precision. Specs pair 1:1 with prompts. The caller owns the combinatorics, not this function.

    Returns the prompt frame with the per-row metadata from `attach_prompt_metadata` already on it.
    """
    compositions = [s["composition_expanded"] for s in specs]
    sgs = [s.get("spacegroup") for s in specs]
    if all(sg is None for sg in sgs):
        sgs = None

    condition_lists = []
    for s in specs:
        cond_value = s.get("condition_vector")
        if isinstance(cond_value, (list, np.ndarray)) or cond_value in (None, "None"):
            # Nested condition vectors (continuous XRD) bypass string parsing:
            # don't round trip through str() and back.
            condition_lists.append([None])
        else:
            condition_lists.append(parse_condition_list_args([str(cond_value)])[0])

    df = create_manual_prompts(
        compositions=compositions,
        condition_lists=condition_lists,
        level=args.level,
        spacegroups=sgs,
        mode="paired",
    )
    return attach_prompt_metadata(df, specs)


def generate_cifs_with_hf_model(df_prompts: pd.DataFrame, hf_model_path: str, args: argparse.Namespace, worker_count: int = 1) -> pd.DataFrame:
    """Generate CIFs for a prompt frame with one Hub model.

    Looks the model up in the registry, pins the generation length to that model's context window,
    and hands the work to `run_generation_pool`. `worker_count` above 1 fans generation across that
    many GPU workers. The seed is fixed at 1, since the CLI exposes no seed flag and every run keeps
    the historical default.

    Returns a DataFrame of generated rows, one per returned sequence.
    """
    tokenizer = _tokenizer()
    info = MODEL_INFO[hf_model_path]
    max_length = get_hf_model_max_length(hf_model_path, info["model_type"])  # lru_cached in utils
    # build_generation_kwargs reads gen_max_length off args, so pin it to the model context.
    args.gen_max_length = max_length
    generation_kwargs = build_generation_kwargs(args, tokenizer, max_length)
    # There is no --seed CLI knob, so every run keeps the historical default.
    base_seed = 1

    if worker_count >= 2:
        print(f"Multi-GPU generation active with {worker_count} workers.")

    generated_rows = run_generation_pool(
        df_prompts=df_prompts,
        generation_kwargs=generation_kwargs,
        activate_conditionality=info["model_type"],
        # run_generation_pool normalizes scoring_mode itself, so the raw value passes through.
        scoring_mode=args.scoring_mode,
        target_valid_cifs=args.target_valid_cifs,
        max_return_attempts=args.max_return_attempts,
        base_seed=base_seed,
        worker_count=worker_count,
        initargs_override=(hf_model_path, TOKENIZER_DIR, info["model_type"], base_seed, "hf",
                           info.get("config_overrides")),
        screening_profile=args.screening_profile,
    )

    return pd.DataFrame(generated_rows)


def _generate_and_score(df_prompts: pd.DataFrame, args: argparse.Namespace, scoring_mode: str) -> pd.DataFrame:
    """Resolve GPU workers and run one generation pass."""
    worker_count = resolve_multi_gpu_workers(args, len(df_prompts))
    return generate_cifs_with_hf_model(df_prompts, args.hf_model_path, args, worker_count)


def run_parquet_mode(args: argparse.Namespace, scoring_mode: str) -> pd.DataFrame:
    """Generate from a parquet of prebuilt prompts.

    The simpler of the two input paths. Prompts are already built, so this only reads them, applies
    `--max_samples` if given, and generates. Z search belongs to formula mode and does not apply
    here.
    """
    print(f"\nLoading Prompts\nSource: {args.input_parquet}")
    df_prompts = pd.read_parquet(args.input_parquet)
    if args.max_samples:
        df_prompts = df_prompts.head(args.max_samples)

    print("\nStarting CIF Generation")
    return _generate_and_score(df_prompts, args, scoring_mode)


def _run_early_stop_search(args: argparse.Namespace, canonical_formulas: list, row_properties: list, xrd_format: str | None) -> pd.DataFrame:
    """Iterate DEFAULT_Z_LIST, dropping each formula after its first valid structure."""
    print(f"\nExecuting Early-Stopping Z_search over Z={DEFAULT_Z_LIST}")
    active_rows = list(zip(canonical_formulas, row_properties))
    completed_dfs = []

    for z in DEFAULT_Z_LIST:
        if not active_rows:
            break
        active_fs = [f for f, _ in active_rows]
        active_ps = [p for _, p in active_rows]
        print(f"\nSearching Z={z} for {len(active_rows)} formulas...")
        specs = build_reduced_formula_specs(active_fs, [z] * len(active_rows), active_ps, xrd_format, args.xrd_wavelength)
        df_prompts = generate_prompts_from_specs(specs, args)
        # Early stop implies scoring "none" (enforced by the dispatch condition).
        df_gen = _generate_and_score(df_prompts, args, "none")

        if not df_gen.empty:
            # The pool pre-validates rows in this mode (target_valid_cifs > 0), so the shared
            # reducer only merges each row's formula target back and keeps the first hit per formula.
            best_valid = reduce_rows_for_reduced_formula_search(df_gen, df_prompts, active_fs, "none")
            if not best_valid.empty:
                completed_dfs.append(best_valid)
                found = set(best_valid["reduced_formula_target"])
                active_rows = [(f, p) for f, p in active_rows if f not in found]
                print(f"  Found valid structures for {len(found)} formulas.")

    remaining = [f for f, _ in active_rows]
    if remaining:
        print(f"\nFailed to find valid structures for: {', '.join(remaining)}")
    return pd.concat(completed_dfs, ignore_index=True) if completed_dfs else pd.DataFrame()


def _run_batch_generation(args: argparse.Namespace, canonical_formulas: list, row_properties: list, z_list: list[int] | None, xrd_format: str | None, scoring_mode: str) -> pd.DataFrame:
    """Single generation pass: explicit Z values, or the full formula x DEFAULT_Z_LIST grid."""
    print("\nExecuting Batch Generation")
    if args.search_zs:
        # Expand each formula x DEFAULT_Z_LIST. The reducer below keeps one best row per formula.
        formulas = [f for f in canonical_formulas for _ in DEFAULT_Z_LIST]
        z_values = [z for _ in canonical_formulas for z in DEFAULT_Z_LIST]
        properties = [p for p in row_properties for _ in DEFAULT_Z_LIST]
    else:
        formulas, properties = canonical_formulas, row_properties
        z_values = z_list if z_list else [1] * len(canonical_formulas)

    specs = build_reduced_formula_specs(formulas, z_values, properties, xrd_format, args.xrd_wavelength)
    df_prompts = generate_prompts_from_specs(specs, args)
    df_gen = _generate_and_score(df_prompts, args, scoring_mode)

    if args.search_zs and args.target_valid_cifs > 0:
        return reduce_rows_for_reduced_formula_search(df_gen, df_prompts, canonical_formulas, scoring_mode)
    return df_gen


def run_formula_mode(args: argparse.Namespace, parser: argparse.ArgumentParser, scoring_mode: str, xrd_format: str | None) -> pd.DataFrame:
    """Generate from a list of reduced formulas, expanding each over Z.

    Validates the per-formula CLI lists, which each accept either one value broadcast to every
    formula or exactly one value per formula, then assembles the per-row conditioning and dispatches
    to one of two strategies. An unscored `--search_zs` run walks Z values and drops each formula
    once it yields a valid structure. Everything else generates the full formula-by-Z grid, because
    ranking needs every candidate present before it can choose.

    Rejects duplicate formulas under `--search_zs`, where completion is tracked by formula name.
    """
    raw_formulas = parse_reduced_formula_list_arg(args.reduced_formula_list)
    canonical_formulas = canonicalize_reduced_formulas(raw_formulas)
    n_formulas = len(canonical_formulas)

    # --search_zs iterates Z values internally and uses formula-name dedup to track
    # completion, so duplicate formulas would produce incorrect early-stop behaviour.
    if args.search_zs and len(set(canonical_formulas)) != n_formulas:
        parser.error("--search_zs requires unique formulas. Repeat detection is name-based.")

    # Per-formula CLI lists accept either one value broadcast to all formulas or exactly one each.
    xrd_files = _broadcast(args.xrd_files, n_formulas, "XRD files", parser) if args.xrd_files else [None] * n_formulas

    sg_list = []
    if args.spacegroups:
        if args.level in ("level_1", "level_2"):
            print("\nWarning: --spacegroups provided but prompt level is not level_3 or level_4. Spacegroups will be ignored.")
        sg_list = _broadcast([s.strip() for s in args.spacegroups.split(",")], n_formulas, "spacegroups", parser)

    z_list = None
    if args.z_list:
        z_list = [int(z.strip()) for z in args.z_list.split(",")]
        if len(z_list) != n_formulas:
            parser.error(f"Expected {n_formulas} Z integers, got {len(z_list)}.")

    # The registry tags each model "scalar", "xrd_top20" or None.
    # XRD_FORMATS membership keeps scalar PKV models out of the XRD input path.
    is_xrd = xrd_format in XRD_FORMATS

    if is_xrd and not args.xrd_files:
        print("\nWarning: Slider XRD model selected without --xrd_files. "
              "Generation will run with missing conditioning values.")
    if args.xrd_files and args.xrd_wavelength is None:
        print("\nWarning: --xrd_wavelength not given, assuming CuKa1 1.54056 A. "
              "Specify it explicitly for non-CuKa data.")
        args.xrd_wavelength = 1.54056  # set once here so the parsing module doesn't warn again per file

    # XRD models take conditioning from scan files, so scalar condition lists only apply otherwise.
    if args.condition_lists and not is_xrd:
        cond_list = build_formula_condition_map(canonical_formulas, args.condition_lists, args.hf_model_path)
    else:
        cond_list = [None] * n_formulas

    row_properties = [
        {"xrd": xrd_files[i], "sg": sg_list[i] if sg_list else None, "cond": cond_list[i]}
        for i in range(n_formulas)
    ]

    # Early stopping only makes sense unscored: with a scoring mode active, every Z must be
    # generated and ranked, so those runs fall through to the batch grid instead.
    if args.search_zs and scoring_mode == "none" and args.target_valid_cifs > 0:
        return _run_early_stop_search(args, canonical_formulas, row_properties, xrd_format)
    return _run_batch_generation(args, canonical_formulas, row_properties, z_list, xrd_format, scoring_mode)


def write_outputs(df_final: pd.DataFrame, args: argparse.Namespace) -> None:
    """Write the finished frame as CIF files or as a single parquet.

    `--output_cif_dir` writes one file per structure, named by full non-reduced formula plus
    Material ID so runs stay distinguishable, skipping rows whose generation came back empty.
    `--output_parquet` writes everything to one file instead, creating the parent directory if
    needed.
    """
    if args.output_cif_dir:
        os.makedirs(args.output_cif_dir, exist_ok=True)
        for idx, row in df_final.iterrows():
            cif_txt = row["Generated CIF"]
            if not isinstance(cif_txt, str) or not cif_txt.strip():
                continue

            # Name files by full (non-reduced) formula plus Material ID so runs stay distinguishable.
            formula_nonreduced = extract_formula_nonreduced(cif_txt).replace(" ", "_")
            mid = row.get("Material ID", f"Generated_{idx + 1}")
            filename = f"{formula_nonreduced}_{mid}" if formula_nonreduced else str(mid)

            with open(os.path.join(args.output_cif_dir, f"{filename}.cif"), 'w') as f:
                f.write(cif_txt)

        print(f"\nProcess Complete\nSaved {len(df_final)} CIF files to: {args.output_cif_dir}")
    else:
        output_dir = os.path.dirname(args.output_parquet)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df_final.to_parquet(args.output_parquet, index=False)
        print(f"\nProcess Complete\nSaved {len(df_final)} structures to: {args.output_parquet}")


def write_run_metadata(args: argparse.Namespace) -> None:
    """Write the model commit and sampling settings to a JSON file next to the outputs."""
    # The Hub commit pins the exact weights, None for a local checkpoint or offline
    try:
        revision = HfApi().model_info(args.hf_model_path).sha
    except Exception:
        revision = None

    meta = {
        "model": args.hf_model_path,
        "revision": revision,
        "sampling": {k: getattr(args, k, None) for k in SIDECAR_SAMPLING_KEYS},
        "seed": getattr(args, "seed", 1),  # generate_cifs falls back to 1, each GPU worker adds its id
        "num_workers_gpu": args.num_workers_gpu,
    }

    # out.parquet gets out.json, a CIF directory gets <dir>.json beside it
    path = Path(args.output_parquet or args.output_cif_dir.rstrip("/")).with_suffix(".json")
    path.write_text(json.dumps(meta, indent=2))
    print(f"Run metadata saved to: {path}")


def main() -> None:
    """Parse arguments and run the generation pipeline."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_model_path", required=True, help="HuggingFace model path")
    parser.add_argument("--model_registry", help="Optional JSON registry containing metadata for custom Hugging Face models")

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output_parquet", default=None, help="Output parquet file")
    output_group.add_argument("--output_cif_dir", default=None, help="Output directory for individual CIF files")

    prompt_group = parser.add_mutually_exclusive_group(required=False)
    prompt_group.add_argument("--input_parquet", help="Input parquet with prompts")
    prompt_group.add_argument("--reduced_formula_list", type=str, help="Comma-separated reduced formulas")

    parser.add_argument("--max_samples", type=int, default=None, help="Max prompts to process from input parquet (for testing)")

    z_group = parser.add_mutually_exclusive_group()
    z_group.add_argument("--search_zs", action="store_true", help="Search through Z=1,2,3,4,6 to find valid structures")
    z_group.add_argument("--z_list", type=str, help="Comma-separated explicit Z integers mapping 1:1 to formulas")

    parser.add_argument("--condition_lists", nargs='+', help="One string per formula, or a single string broadcast to all. Each string holds that formula's comma-separated condition values, e.g. --condition_lists \"2.16, 0.0\"")
    parser.add_argument("--level", choices=["level_1", "level_2", "level_3", "level_4"], default="level_2")
    parser.add_argument("--spacegroups", help="Comma-separated spacegroups mapped to formulas")

    parser.add_argument("--xrd_files", nargs='+', help="Raw XRD scan files (.csv, .xy, .dat, .txt) mapped to formulas")
    parser.add_argument("--xrd_wavelength", type=float, default=None, help="Wavelength in Angstrom of the provided XRD data (default: CuKa1 1.54056, assumed with a warning)")

    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Sequences per sample")
    parser.add_argument("--max_return_attempts", type=int, default=1, help="Generation attempts per sample")
    parser.add_argument("--target_valid_cifs", type=int, default=1, help="Number of valid CIFs to target per prompt. If no scoring mode, we can also specify 0 to return all generated CIFs regardless of validity.")

    parser.add_argument("--scoring_mode", type=str, default="None", help="Scoring: 'LOGP' (model perplexity) or 'None'.")

    parser.add_argument("--screening_profile", type=str, default="application", choices=("benchmark", "application"),
                        help="How hard to screen generated CIFs. 'application' runs the bond-length check and ranks over the whole generated batch. 'benchmark' skips the bond-length check and ranks a pool truncated at target_valid_cifs, reproducing the screening behind the published MP-20 and CHILI-100K numbers.")

    parser.add_argument("--num_workers", type=int, default=4, help="CPU Post-processing workers")
    parser.add_argument("--num_workers_gpu", type=int, default=None, help="GPU workers for inference (defaults to all visible GPUs). Set to 1 for a single GPU.")
    parser.add_argument("--skip_postprocess", action="store_true", help="Skip CIF validation")

    args = parser.parse_args()

    # Fixed sampling settings for every CLI run. build_generation_kwargs reads these off args.
    args.do_sample = DO_SAMPLE
    args.top_k = TOP_K
    args.top_p = TOP_P

    if args.model_registry:
        try:
            # The CLI is one-shot, so sharing this overlay keeps existing metadata consumers unchanged.
            MODEL_INFO.update(_load_custom_model_registry(args.model_registry))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

    model_info = MODEL_INFO.get(args.hf_model_path)
    if not model_info:
        parser.error(
            f"Model '{args.hf_model_path}' was not found in the built-in registry "
            "or --model_registry"
        )

    print(f"\nModel Configuration\nPath: {args.hf_model_path}\nType: {model_info['model_type']}\nTask: {model_info['description']}")

    xrd_format = get_condition_format(args.hf_model_path)

    scoring_mode = _normalize_scoring_mode(args.scoring_mode)
    if scoring_mode == "logp" and args.target_valid_cifs == 0:
        parser.error(f"scoring_mode={args.scoring_mode} requires --target_valid_cifs > 0.")

    # Level 1 is unconditional generation, so a placeholder formula stands in for real input.
    if not args.input_parquet and not args.reduced_formula_list:
        if args.level == "level_1":
            args.reduced_formula_list = "X"
        else:
            parser.error("Must provide --input_parquet or --reduced_formula_list for level 2+.")

    if args.input_parquet:
        df_work = run_parquet_mode(args, scoring_mode)
    else:
        df_work = run_formula_mode(args, parser, scoring_mode, xrd_format)

    if not args.skip_postprocess:
        print("\nConverting CIFs to standard CIF format")
        df_final = _postprocess_non_empty_cifs(df_work, args.num_workers, "Generated CIF")
    else:
        df_final = df_work

    if df_final.empty:
        print("\nPipeline filtered out all structures.")
        return

    # Internal bookkeeping columns never ship in user outputs.
    df_final = df_final.drop(
        columns=["is_consistent", "rank", "Z_search", "prompt_order", "is_valid", "selection_status", "Base Material ID"],
        errors="ignore"
    )

    write_outputs(df_final, args)
    write_run_metadata(args)


if __name__ == "__main__":
    main()
