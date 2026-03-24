#!/usr/bin/env python3
"""Main entry point for loading and generating CIF structures from HuggingFace CrystaLLM models.

This script processes input data (either from parquet or mapped reduced formula strings), builds
the necessary conditioning prompts, and manages generation across single or multi-GPU environments.
Features an early-stopping iteration loop for efficient Z-value discovery.
"""

import os
import sys
import argparse
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _utils._generating.make_prompts import create_manual_prompts
from _utils._generating.generate_CIFs import (
    init_tokenizer,
    build_generation_kwargs,
    run_generation_pool,
    _normalize_scoring_mode,
)
from _utils._generating.postprocess import process_dataframe
from _utils import extract_formula_nonreduced
from _utils._direct_gen_utils import (
    MODEL_INFO,
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
DO_SAMPLE = "True"
TOP_K = 15
TOP_P = 0.95
GEN_MAX_LENGTH = 1024
DEFAULT_Z_LIST = [1, 2, 3, 4, 6]

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


def generate_prompts_from_specs(specs: list, args) -> pd.DataFrame:
    """Construct prompt dataframe natively mapped from specs."""
    compositions = [s["composition_expanded"] for s in specs]
    sgs = [s.get("spacegroup") for s in specs]
    if all(sg is None for sg in sgs):
        sgs = None

    condition_lists = []
    for s in specs:
        cond_str = s.get("condition_vector")
        if cond_str in (None, "None"):
            condition_lists.append([None])
        else:
            condition_lists.append(parse_condition_list_args([str(cond_str)])[0])

    df = create_manual_prompts(
        compositions=compositions,
        condition_lists=condition_lists,
        raw_mode=False,
        level=args.level,
        spacegroups=sgs,
        mode="paired",
    )
    return attach_prompt_metadata(df, specs)


def generate_cifs_with_hf_model(df_prompts: pd.DataFrame, hf_model_path: str, args, worker_count: int = 1) -> pd.DataFrame:
    """Generate CIFs using HF model, optionally across multiple GPUs."""
    tokenizer = init_tokenizer(TOKENIZER_DIR)
    max_length = get_hf_model_max_length(hf_model_path)
    generation_kwargs = build_generation_kwargs(args, tokenizer, max_length)
    scoring_mode = _normalize_scoring_mode(args.scoring_mode)
    base_seed = getattr(args, "seed", 1)
    
    model_type = MODEL_INFO[hf_model_path]["model_type"]

    if worker_count >= 2:
        print(f"Multi-GPU generation active with {worker_count} workers.")

    generated_rows = run_generation_pool(
        df_prompts=df_prompts,
        generation_kwargs=generation_kwargs,
        activate_conditionality=model_type,
        scoring_mode=scoring_mode,
        target_valid_cifs=args.target_valid_cifs,
        max_return_attempts=args.max_return_attempts,
        base_seed=base_seed,
        worker_count=worker_count,
        initargs_override=(hf_model_path, TOKENIZER_DIR, model_type, base_seed, "hf"),
    )
    
    return pd.DataFrame(generated_rows)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--hf_model_path", required=True, help="HuggingFace model path")

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output_parquet", default=None, help="Output parquet file")
    output_group.add_argument("--output_cif_dir", default=None, help="Output directory for individual CIF files")
    
    prompt_group = parser.add_mutually_exclusive_group(required=False)
    prompt_group.add_argument("--input_parquet", help="Input parquet with prompts")
    prompt_group.add_argument("--reduced_formula_list", type=str, help="Comma-separated reduced formulas")

    parser.add_argument("--max_samples", type=int, default=None, help="Max prompts to process from input parquet (for testing)")

    z_group = parser.add_mutually_exclusive_group()
    z_group.add_argument("--search_zs", action="store_true", help="Search through Z=1 to Z=4 to find valid structures")
    z_group.add_argument("--z_list", type=str, help="Comma-separated explicit Z integers mapping 1:1 to formulas")
    
    parser.add_argument("--condition_lists", nargs='+', help="Real property values strings.")
    parser.add_argument("--level", choices=["level_1", "level_2", "level_3", "level_4"], default="level_2")
    parser.add_argument("--spacegroups", help="Comma-separated spacegroups mapped to formulas")
    
    parser.add_argument("--xrd_files", nargs='+', help="Files with XRD peaks (.csv, .xy, .dat, .txt) mapped to formulas")
    parser.add_argument("--xrd_wavelength", type=float, default=1.54056, help="Wavelength of the provided XRD data (default: CuKa 1.54056)")

    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Sequences per sample")
    parser.add_argument("--max_return_attempts", type=int, default=1, help="Generation attempts per sample")
    parser.add_argument("--target_valid_cifs", type=int, default=1, help="Number of valid CIFs to target per prompt. If no scoring mode, we can also specify 0 to return all generated CIFs regardless of validity.")

    parser.add_argument("--scoring_mode", type=str, default="None", help="Scoring: 'LOGP' or 'None'")
    
    parser.add_argument("--num_workers", type=int, default=4, help="CPU Post-processing workers")
    parser.add_argument("--num_workers_gpu", type=int, default=None, help="Max GPU workers for inference")
    parser.add_argument("--multi_gpu", choices=["auto", "true", "false"], default="auto", help="Multi-GPU mode")
    parser.add_argument("--skip_postprocess", action="store_true", help="Skip CIF validation")
    
    args = parser.parse_args()
    
    args.do_sample = DO_SAMPLE
    args.top_k = TOP_K
    args.top_p = TOP_P
    args.gen_max_length = GEN_MAX_LENGTH

    normalized_scoring_mode = _normalize_scoring_mode(args.scoring_mode)
    if normalized_scoring_mode == "logp" and args.target_valid_cifs == 0:
        parser.error("scoring_mode=LOGP requires --target_valid_cifs > 0.")
    
    model_info = MODEL_INFO.get(args.hf_model_path)
    if not model_info:
        raise ValueError(f"Model {args.hf_model_path} not found in MODEL_INFO dict.")
        
    print(f"\nModel Configuration\nPath: {args.hf_model_path}\nType: {model_info['model_type']}\nTask: {model_info['description']}")

    # Apply Level 1 bypass logic
    if not args.input_parquet and not args.reduced_formula_list:
        if args.level == "level_1":
            args.reduced_formula_list = "X"
        else:
            parser.error("Must provide --input_parquet or --reduced_formula_list for level 2+.")
    
    if args.input_parquet:
        print(f"\nLoading Prompts\nSource: {args.input_parquet}")
        df_prompts = pd.read_parquet(args.input_parquet)
        if args.max_samples:
            df_prompts = df_prompts.head(args.max_samples)
            
        worker_count = resolve_multi_gpu_workers(args, len(df_prompts))
        print("\nStarting CIF Generation")
        df_work = generate_cifs_with_hf_model(df_prompts, args.hf_model_path, args, worker_count)
        
    elif args.reduced_formula_list:
        raw_formulas = parse_reduced_formula_list_arg(args.reduced_formula_list)
        canonical_formulas = canonicalize_reduced_formulas(raw_formulas)
        
        if len(canonical_formulas) != len(raw_formulas):
            raise ValueError("Duplicate formulas detected. Please provide strictly unique reduced formulas.")
            
        n_formulas = len(canonical_formulas)
        property_map = {}
        
        if args.xrd_files and len(args.xrd_files) != n_formulas:
            raise ValueError(f"Expected {n_formulas} XRD files, got {len(args.xrd_files)}.")

        sg_list = []
        if args.spacegroups:
            raw_sg = [s.strip() for s in args.spacegroups.split(",")]
            if len(raw_sg) == 1:
                sg_list = raw_sg * n_formulas
            elif len(raw_sg) == n_formulas:
                sg_list = raw_sg
            else:
                raise ValueError(f"Expected 1 or {n_formulas} spacegroups, got {len(raw_sg)}.")
                
        if args.z_list:
            z_list = [int(z.strip()) for z in args.z_list.split(",")]
            if len(z_list) != n_formulas:
                raise ValueError(f"Expected {n_formulas} Z integers, got {len(z_list)}.")
        
        is_slider_model = model_info["model_type"] == "Slider"
        if is_slider_model and not args.xrd_files:
            print(
                "\nWarning: Slider model selected without --xrd_files. "
                "Generation will run with missing conditioning values."
            )

        if args.condition_lists and not is_slider_model:
            formula_cond_map = build_formula_condition_map(canonical_formulas, args.condition_lists, args.hf_model_path)
        else:
            formula_cond_map = {}

        property_map = {
            formula: {
                "xrd": args.xrd_files[i] if args.xrd_files else None,
                "sg": sg_list[i] if sg_list else None,
                "cond": formula_cond_map.get(formula),
            }
            for i, formula in enumerate(canonical_formulas)
        }
            
        is_early_stop = args.search_zs and normalized_scoring_mode == "none" and args.target_valid_cifs > 0
        
        if is_early_stop:
            print(f"\nExecuting Early-Stopping Z_search ({DEFAULT_Z_LIST[0]} to {DEFAULT_Z_LIST[-1]})")
            active_formulas = canonical_formulas.copy()
            completed_dfs = []
            
            for z in DEFAULT_Z_LIST:
                if not active_formulas: break
                print(f"\nSearching Z={z} for {len(active_formulas)} formulas...")
                z_mapping = {f: [z] for f in active_formulas}
                specs = build_reduced_formula_specs(active_formulas, z_mapping, property_map, is_slider_model, args.xrd_wavelength)
                df_prompts = generate_prompts_from_specs(specs, args)
                
                worker_count = resolve_multi_gpu_workers(args, len(df_prompts))
                df_gen = generate_cifs_with_hf_model(df_prompts, args.hf_model_path, args, worker_count)
                
                if not df_gen.empty:
                    df_gen["Base Material ID"] = df_gen["Material ID"].str.rsplit('_', n=1).str[0]
                    df_gen = df_gen.merge(
                        df_prompts[["Material ID", "reduced_formula_target"]]
                        .rename(columns={"Material ID": "Base Material ID"})
                        .drop_duplicates(),
                        on="Base Material ID", 
                        how="left"
                    )
                    best_valid = df_gen.drop_duplicates(subset=["reduced_formula_target"], keep="first")
                    completed_dfs.append(best_valid)
                    found = best_valid["reduced_formula_target"].dropna().tolist()
                    active_formulas = [f for f in active_formulas if f not in found]
                    print(f"  Found valid structures for {len(found)} formulas.")
            
            if active_formulas:
                print(f"\nFailed to find valid structures for: {', '.join(active_formulas)}")
            df_work = pd.concat(completed_dfs, ignore_index=True) if completed_dfs else pd.DataFrame()

        else:
            print("\nExecuting Batch Generation")
            if args.search_zs:
                z_mapping = {f: DEFAULT_Z_LIST for f in canonical_formulas}
            else:
                z_mapping = {f: [z] for f, z in zip(canonical_formulas, z_list)} if args.z_list else {f: [1] for f in canonical_formulas}

            specs = build_reduced_formula_specs(canonical_formulas, z_mapping, property_map, is_slider_model, args.xrd_wavelength)
            
            df_prompts = generate_prompts_from_specs(specs, args)

            worker_count = resolve_multi_gpu_workers(args, len(df_prompts))
            df_gen = generate_cifs_with_hf_model(df_prompts, args.hf_model_path, args, worker_count)
            
            if args.search_zs and args.target_valid_cifs > 0:
                df_work = reduce_rows_for_reduced_formula_search(
                    df_gen, df_prompts, canonical_formulas, normalized_scoring_mode
                )
            else:
                df_work = df_gen

    if not args.skip_postprocess:
        print("\nConverting CIFs to standard CIF format")
        df_final = _postprocess_non_empty_cifs(df_work, args.num_workers, "Generated CIF")
    else:
        df_final = df_work
        
    if df_final.empty:
        print("\nPipeline filtered out all structures.")
        return

    # Drop intermediary columns prior to finalizing outputs
    df_final = df_final.drop(
        columns=["is_consistent", "rank", "Z_search", "prompt_order", "is_valid", "selection_status", "Base Material ID"], 
        errors="ignore"
    )
    
    if args.output_cif_dir:
        os.makedirs(args.output_cif_dir, exist_ok=True)
        for idx, row in df_final.iterrows():
            cif_txt = row["Generated CIF"]
            if not isinstance(cif_txt, str) or not cif_txt.strip():
                continue
            
            formula_nonreduced = extract_formula_nonreduced(cif_txt).replace(" ", "_")
            mid = row.get("Material ID", f"Generated_{idx + 1}")
            filename = f"{formula_nonreduced}_{mid}" if formula_nonreduced else str(mid)
            
            with open(os.path.join(args.output_cif_dir, f"{filename}.cif"), 'w') as f:
                f.write(cif_txt)
                
        print(f"\nProcess Complete\nSaved {len(df_final)} CIF files to: {args.output_cif_dir}")
    else:
        output_dir = os.path.dirname(args.output_parquet)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        df_final.to_parquet(args.output_parquet, index=False)
        print(f"\nProcess Complete\nSaved {len(df_final)} structures to: {args.output_parquet}")

if __name__ == "__main__":
    main()