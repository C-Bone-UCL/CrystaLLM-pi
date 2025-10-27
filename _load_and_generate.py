#!/usr/bin/env python3
"""
Load and generate CIF structures from HuggingFace CrystaLLM models.

Works with all the CrystaLLM-2.0 models on Huggingface. Give real property values for manual generation and it handles the normalization automatically.

Models:
- c-bone/CrystaLLM-2.0_base (unconditional)
- c-bone/CrystaLLM-2.0_SLME (solar efficiency)  
- c-bone/CrystaLLM-2.0_bandgap (bandgap + ehull)
- c-bone/CrystaLLM-2.0_density (density + ehull)
- c-bone/CrystaLLM-2.0_COD-XRD (XRD patterns, for this one you need a prepared df first)

Usage examples:

# Basic unconditional generation
python _load_and_generate.py --hf_model_path c-bone/CrystaLLM-2.0_base \
    --manual --compositions "LiFePO4,TiO2" --output_parquet results.parquet

# Solar efficiency model - just give real SLME values (0-33 range is what was seen during training)  
python _load_and_generate.py --hf_model_path c-bone/CrystaLLM-2.0_SLME \
    --manual --compositions "CsPbI3" --condition_lists "25.0" --output_parquet results.parquet

# Bandgap model - bandgap in eV, ehull in eV/atom (set ehull=0 for stable, max bandgap ~17.9 eV was seen during training)
python _load_and_generate.py --hf_model_path c-bone/CrystaLLM-2.0_bandgap \
    --manual --compositions "Si" --condition_lists "1.1" "0.0" --output_parquet results.parquet

# Density model - density in g/cm3, ehull in eV/atom (set ehull=0 for stable, max density ~25.5 g/cm3 was seen during training)
python _load_and_generate.py --hf_model_path c-bone/CrystaLLM-2.0_density \
    --manual --compositions "Os" --condition_lists "22.0" "0.0" --output_parquet results.parquet
"""


import os
import sys
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _utils._generating.make_prompts import create_manual_prompts
from _utils._generating.generate_CIFs import (
    init_tokenizer, get_model_class, build_generation_kwargs, setup_device,
    parse_condition_vector, remove_conditionality, check_cif, score_output_logp
)
from _utils._generating.postprocess import process_dataframe
from _utils import normalize_property_column, normalize_values_with_method

TOKENIZER_DIR = "HF-cif-tokenizer"

# Model specs - used for auto-normalization
MODEL_INFO = {
    "c-bone/CrystaLLM-2.0_base": {
        "description": "Unconditional generation",
        "conditions": 0,
        "example_conditions": None, 
        "max": None,
        "min": None,
        "normalization": None
    },
    "c-bone/CrystaLLM-2.0_SLME": {
        "description": "Solar cell efficiency (SLME) conditioning", 
        "conditions": 1,
        "example_conditions": ["25.0"],
        "max": 33.192,
        "min": 0.0,
        "normalization": "linear"
    },
    "c-bone/CrystaLLM-2.0_bandgap": {
        "description": "Bandgap + stability conditioning",
        "conditions": 2, 
        "example_conditions": ["1.1", "0.0"],
        "max": [17.891, 5.418],
        "min": [0.0, 0.0],
        "normalization": ["power_log", "linear"]
    },
    "c-bone/CrystaLLM-2.0_density": {
        "description": "Density + stability conditioning",
        "conditions": 2,
        "example_conditions": ["15.0", "0.0"],
        "max": [25.494, 0.1],
        "min": [0.0, 0.0],
        "normalization": ["linear", "linear"]
    },
    "c-bone/CrystaLLM-2.0_COD-XRD": {
        "description": "Experimental XRD conditioning",
        "conditions": 40,
        "example_conditions": ["-100"] * 20 + ["-100"] * 20,
        "max": [90.0, 100.0],
        "min": [0.0, 0.0],
        "normalization": ["linear", "linear"]
    },
}


def normalize_property_values(raw_values: List[List[float]], model_path: str) -> List[List[float]]:
    """Normalize raw property values to [0-1] using model settings."""
    if not raw_values:
        return raw_values
        
    model_info = MODEL_INFO.get(model_path)
    if not model_info or not model_info.get("normalization"):
        print("No normalization info - assuming values already normalized")
        return raw_values

    norm_methods = model_info["normalization"]
    min_vals = model_info["min"] 
    max_vals = model_info["max"]
    
    # make everything lists for easier processing
    if not isinstance(norm_methods, list):
        norm_methods = [norm_methods] * len(raw_values)
    if not isinstance(min_vals, list):
        min_vals = [min_vals] * len(raw_values) if min_vals is not None else [0.0] * len(raw_values)
    if not isinstance(max_vals, list):
        max_vals = [max_vals] * len(raw_values) if max_vals is not None else [1.0] * len(raw_values)

    normalized_values = []
    
    for i, values in enumerate(raw_values):
        method = norm_methods[i] if i < len(norm_methods) else "linear"
        min_val = min_vals[i] if i < len(min_vals) else 0.0
        max_val = max_vals[i] if i < len(max_vals) else 1.0
        
        # use the direct normalization function
        norm_vals = normalize_values_with_method(values, method, min_val, max_val)
        normalized_values.append(norm_vals)
    
    return normalized_values


def validate_model_conditions(model_path: str, condition_lists: Optional[List[List[float]]]) -> None:
    """Check that condition count matches what the model expects."""
    model_info = MODEL_INFO.get(model_path)
    if not model_info:
        print(f"Warning: Unknown model {model_path}")
        return
        
    expected = model_info["conditions"]
    provided = len(condition_lists) if condition_lists else 0
    
    if expected == 0 and condition_lists:
        print("Base model doesn't need conditions - ignoring them")
    elif expected > 0 and expected != provided:
        examples = model_info.get("example_conditions", [])
        raise ValueError(
            f"Model {model_path} needs {expected} condition list(s), got {provided}.\n"
            f"Try: {examples}\n"
            f"({model_info['description']})"
        )


def load_hf_model(hf_model_path: str, model_type: str = "None"):
    """Load model from HuggingFace."""
    print(f"Loading from HF: {hf_model_path}")
    
    model_class = get_model_class(model_type)
    print(f"Using {model_class.__name__}")
    
    try:
        print("Downloading... (might take a few mins first time)")
        model = model_class.from_pretrained(hf_model_path, trust_remote_code=True)
        
        print(f"Loaded {model_type} model")
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"~{param_count:.1f}M parameters")
        return model
    except Exception as e:
        print(f"Error: {e}")
        print("Check internet connection and model path")
        available = list(MODEL_INFO.keys())
        print(f"Available: {', '.join(available)}")
        return None


def get_material_id(row: Dict[str, Any], count: int, offset: int = 0) -> str:
    """Get material ID from row or make one up."""
    if "Material ID" in row:
        return row["Material ID"]
    elif "Formula" in row:
        return row["Formula"]
    else:
        return f"Generated_{count + offset + 1}"


def generate_prompts_from_args(args) -> pd.DataFrame:
    """Make prompts from args."""    
    if args.input_parquet:
        print(f"Loading prompts from: {args.input_parquet}")
        print("Note: should already have normalized [0-1] values")
        return pd.read_parquet(args.input_parquet)
    
    elif args.manual:
        print("Making prompts from compositions and conditions")
        
        compositions = [comp.strip() for comp in args.compositions.split(',')]

        # parse condition lists
        raw_condition_lists = []
        if args.condition_lists:
            for cond_list in args.condition_lists:
                values = [float(x.strip()) for x in cond_list.split(',')]
                raw_condition_lists.append(values)

        validate_model_conditions(args.hf_model_path, raw_condition_lists)
        
        # normalise using the utility
        normalized_condition_lists = normalize_property_values(raw_condition_lists, args.hf_model_path)
        
        spacegroups = None
        if args.spacegroups:
            spacegroups = [sg.strip() for sg in args.spacegroups.split(',')]
        
        df = create_manual_prompts(
            compositions=compositions,
            condition_lists=normalized_condition_lists,
            raw_mode=args.raw,
            level=args.level,
            spacegroups=spacegroups
        )
        
        print(f"Made {len(df)} prompts at {args.level}")
        return df
    
    else:
        raise ValueError("Need either --input_parquet or --manual")


def setup_generation_args(args):
    """Package up generation settings."""
    class GenerationArgs:
        def __init__(self):
            self.gen_max_length = args.gen_max_length
            self.do_sample = args.do_sample
            self.num_return_sequences = args.num_return_sequences
            self.top_k = args.top_k
            self.top_p = args.top_p
            self.temperature = args.temperature
            self.scoring_mode = args.scoring_mode
            self.max_samples = args.max_samples
            self.target_valid_cifs = args.target_valid_cifs
            self.max_return_attempts = args.max_return_attempts
            self.activate_conditionality = args.model_type
    
    return GenerationArgs()


def generate_cifs_with_hf_model(df_prompts, hf_model_path, gen_args):
    """Generate CIFs using HF model - uses same logic as generate_on_gpu."""
    print(f"Generating with: {hf_model_path}")
    
    tokenizer = init_tokenizer(TOKENIZER_DIR)
    
    model = load_hf_model(hf_model_path, gen_args.activate_conditionality)
    if model is None:
        raise RuntimeError("Model loading failed")
    
    model.eval()
    model.resize_token_embeddings(len(tokenizer))
    
    device = setup_device(0)
    model = model.to(device)
    
    generation_kwargs = build_generation_kwargs(gen_args, tokenizer, model.config.n_positions)
    print(f"Generation settings: {generation_kwargs}")
    
    all_results = []
    
    for idx, row in df_prompts.iterrows():        
        input_ids = tokenizer.encode(row["Prompt"], return_tensors="pt").to(device)
        
        valid_cifs = []
        attempts = 0
        
        scoring_mode = gen_args.scoring_mode
        max_attempts = gen_args.max_return_attempts
        
        if scoring_mode == "None":
            target_gens = generation_kwargs.get("num_return_sequences", 1) * max_attempts
            max_tries = max_attempts
        else:
            target_gens = gen_args.target_valid_cifs
            max_tries = max_attempts
        
        while len(valid_cifs) < target_gens and attempts < max_tries:
            attempts += 1
            
            try:
                if gen_args.activate_conditionality in ["PKV", "Prepend", "Slider"]:
                    # conditional models need condition tensor
                    condition_tensor = None
                    if row["condition_vector"] not in (None, "None"):
                        values = parse_condition_vector(row["condition_vector"])
                        if values:
                            condition_tensor = torch.tensor([values], device=device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=input_ids,
                            condition_values=condition_tensor,
                            return_dict_in_generate=True,
                            output_scores=True,
                            **generation_kwargs,
                        )
                else:
                    # Raw or unconditional
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=input_ids,
                            return_dict_in_generate=True,
                            output_scores=True,
                            **generation_kwargs,
                        )
                
                # process each sequence
                for seq_idx, output_seq in enumerate(outputs.sequences):
                    if torch.isnan(output_seq).any() or torch.isinf(output_seq).any():
                        continue
                    
                    # find EOS and truncate
                    eos_idx = (output_seq == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if eos_idx.numel() > 0:
                        full_sequence = output_seq[:int(eos_idx[0])]
                    else:
                        full_sequence = output_seq
                    
                    cif_txt = tokenizer.decode(full_sequence, skip_special_tokens=True).replace("\n\n", "\n")
                    
                    if gen_args.activate_conditionality == "Raw":
                        cif_txt = remove_conditionality(cif_txt)
                    
                    if scoring_mode == "None":
                        # just collect everything
                        mid = get_material_id(row, len(valid_cifs), 0)
                        
                        valid_cifs.append({
                            "Material ID": mid,
                            "Prompt": row["Prompt"],
                            "Generated CIF": cif_txt,
                            "condition_vector": row.get("condition_vector", "None"),
                        })
                    else:
                        # validate first
                        is_valid = check_cif(cif_txt)
                        
                        if is_valid:
                            if scoring_mode.upper() == "LOGP":
                                score = score_output_logp(model, outputs.scores, outputs.sequences, seq_idx, input_ids.shape[1], tokenizer.eos_token_id)
                            else:
                                score = 0.0
                            
                            mid = get_material_id(row, len(valid_cifs), 0)
                            
                            valid_cifs.append({
                                "Material ID": mid,
                                "Prompt": row["Prompt"],
                                "Generated CIF": cif_txt,
                                "is_consistent": True,
                                "score": score,
                                "condition_vector": row.get("condition_vector", "None"),
                            })
                    
                    if len(valid_cifs) >= target_gens:
                        break
                        
            except Exception as e:
                print(f"Error on prompt {idx + 1}: {e}")
                continue
        
        # handle results based on scoring
        if valid_cifs:
            if scoring_mode == "None":
                all_results.extend(valid_cifs)
                print(f"Got {len(valid_cifs)} CIFs (no validation)")
            else:
                # rank by score if using LOGP
                if scoring_mode.upper() == "LOGP":
                    ranked = sorted(valid_cifs, key=lambda x: x["score"] if not np.isnan(x["score"]) and not np.isinf(x["score"]) else float('inf'), reverse=False)
                else:
                    ranked = valid_cifs
                
                for rank, cif_data in enumerate(ranked, 1):
                    cif_data["rank"] = rank
                
                all_results.extend(ranked)
                if ranked:
                    best = ranked[0]["score"]
                    worst = ranked[-1]["score"]
                    print(f"Got {len(valid_cifs)} valid CIFs, scores: {best:.4f} to {worst:.4f}")
        else:
            print(f"Failed after {attempts} attempts")
    
    return pd.DataFrame(all_results)


def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    
    # Required args
    parser.add_argument("--hf_model_path", required=True,
                       help="HuggingFace model path")
    parser.add_argument("--output_parquet", required=True,
                       help="Output parquet file")
    
    # Input source
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--input_parquet", 
                             help="Input parquet with prompts")
    prompt_group.add_argument("--manual", action="store_true",
                             help="Make prompts from compositions/conditions")
    
    # Manual prompt options
    parser.add_argument("--compositions", 
                       help="Comma-separated compositions (e.g., 'LiFePO4,Si,TiO2')")
    parser.add_argument("--condition_lists", nargs='+',
                       help="Real property values. Base: none, SLME: 1 value (0-33), bandgap/density: 2 values")
    parser.add_argument("--level", choices=["level_1", "level_2", "level_3", "level_4"],
                       default="level_2", help="Prompt detail level")
    parser.add_argument("--spacegroups",
                       help="Comma-separated spacegroups (level_4 only)")
    
    # Generation settings
    parser.add_argument("--do_sample", type=str, default="True", 
                       help="Sampling mode (True/False/beam)")
    parser.add_argument("--top_k", type=int, default=15,
                       help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p sampling")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--gen_max_length", type=int, default=1024,
                       help="Max generation length")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                       help="Sequences per sample")
    parser.add_argument("--max_return_attempts", type=int, default=1,
                       help="Generation attempts per sample")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples to process")
    parser.add_argument("--scoring_mode", type=str, default="None",
                       help="Scoring: 'LOGP' or 'None'")
    parser.add_argument("--target_valid_cifs", type=int, default=20,
                       help="Target valid CIFs per prompt")
    
    # Model and processing
    parser.add_argument("--model_type", choices=["PKV", "Prepend", "Slider", "Raw", "Base"], default="PKV",
                       help="Model architecture")
    parser.add_argument("--raw", action="store_true",
                       help="Raw conditioning format")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Post-processing workers")
    parser.add_argument("--skip_postprocess", action="store_true",
                       help="Skip CIF validation")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # show model info
    print(f"Model: {args.hf_model_path}")
    model_info = MODEL_INFO.get(args.hf_model_path)
    if model_info:
        print(f"Type: {model_info['description']}")
        if model_info['conditions'] > 0:
            print(f"Needs {model_info['conditions']} conditions")
            print(f"Example: {model_info.get('example_conditions', 'N/A')}")
    else:
        print("Warning: Unknown model")
    
    # validate manual input
    if args.manual:
        if not args.compositions and not args.level.startswith("level_1"):
            parser.error("Need --compositions for level_2+")
        
        print(f"Compositions: {args.compositions}")
        if args.condition_lists:
            print(f"Conditions: {args.condition_lists}")
    else:
        print(f"Input: {args.input_parquet}")
    
    # make prompts
    print("\nGenerating Prompts ")
    df_prompts = generate_prompts_from_args(args)
    print(f"Got {len(df_prompts)} prompts")

    if args.verbose:
        print(f"\nExample:\n{df_prompts.iloc[0]['Prompt']}\n")
    
    if args.max_samples:
        df_prompts = df_prompts.head(args.max_samples)
        print(f"Limited to {len(df_prompts)} samples")
    
    gen_args = setup_generation_args(args)
    
    # generate CIFs
    print("\nGenerating CIFs ")
    df_generated = generate_cifs_with_hf_model(df_prompts, args.hf_model_path, gen_args)
    
    # post-process
    if not args.skip_postprocess:
        print("\nPost-processing ")
        df_final = process_dataframe(df_generated, args.num_workers, "Generated CIF")
    else:
        df_final = df_generated
    
    # save
    output_dir = os.path.dirname(args.output_parquet)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    df_final.to_parquet(args.output_parquet, index=False)
    
    print("\nDone ")
    print(f"Total: {len(df_final)} structures")

    print(f"Saved to: {args.output_parquet}")


if __name__ == "__main__":
    main()