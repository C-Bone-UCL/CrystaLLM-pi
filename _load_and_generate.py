#!/usr/bin/env python3
"""
Generate crystal structures using CrystaLLM-2.0 models from Hugging Face Hub.

Simple interface for generating CIF structures with different CrystaLLM models.
Handles prompt creation, conditional generation, and post-processing automatically.

AVAILABLE MODELS:
- c-bone/CrystaLLM-2.0_base     : Unconditional generation (no conditions)
- c-bone/CrystaLLM-2.0_SLME     : Solar efficiency conditioning (1 condition)
- c-bone/CrystaLLM-2.0_bandgap  : Bandgap + stability conditioning (2 conditions)
- c-bone/CrystaLLM-2.0_density  : Density + stability conditioning (2 conditions)

CONDITIONING:
All values must be normalized [0-1]. For bandgap/density models, if you only care
about one property, set the second condition (ehull) to 0.0 for stable materials.

EXAMPLES:
# Unconditional
python generate_from_hf.py --hf_model_path c-bone/CrystaLLM-2.0_base \\
    --manual --compositions "LiFePO4,TiO2" --output_parquet out.parquet

# SLME model (solar efficiency)
python generate_from_hf.py --hf_model_path c-bone/CrystaLLM-2.0_SLME \\
    --manual --compositions "CsPbI3" --condition_lists "0.8" --output_parquet out.parquet

# Bandgap model 
python generate_from_hf.py --hf_model_path c-bone/CrystaLLM-2.0_bandgap \\
    --manual --compositions "Si" --condition_lists "0.3" "0.0" --output_parquet out.parquet

# From existing prompts
python generate_from_hf.py --hf_model_path c-bone/CrystaLLM-2.0_bandgap \\
    --input_parquet prompts.parquet --output_parquet out.parquet
"""


import os
import sys
import argparse
import pandas as pd
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from _utils._generating.make_prompts import create_manual_prompts
from _utils._generating.generate_CIFs import (
    init_tokenizer, get_model_class, build_generation_kwargs, setup_device,
    parse_condition_vector, remove_conditionality, check_cif, score_output_logp
)
from _utils._generating.postprocess import process_dataframe

TOKENIZER_DIR = "HF-cif-tokenizer"

MODEL_INFO = {
    "c-bone/CrystaLLM-2.0_base": {
        "description": "Unconditional generation",
        "conditions": 0,
        "example_conditions": None
    },
    "c-bone/CrystaLLM-2.0_SLME": {
        "description": "Solar cell efficiency (SLME) conditioning", 
        "conditions": 1,
        "example_conditions": ["0.8"]
    },
    "c-bone/CrystaLLM-2.0_bandgap": {
        "description": "Bandgap + stability conditioning",
        "conditions": 2, 
        "example_conditions": ["0.3", "0.0"]
    },
    "c-bone/CrystaLLM-2.0_density": {
        "description": "Density + stability conditioning",
        "conditions": 2,
        "example_conditions": ["0.9", "0.0"] 
    }
}


def load_hf_model(hf_model_path, model_type="None"):
    """Load CrystaLLM model from Hugging Face Hub."""
    print(f"Loading model from Hugging Face: {hf_model_path}")
    
    # Get appropriate model class
    model_class = get_model_class(model_type)
    print(f"Using model class: {model_class.__name__}")
    
    try:
        # Load model with trust_remote_code for custom CrystaLLM architectures
        print("Downloading model (this may take a few minutes for first download)...")
        model = model_class.from_pretrained(hf_model_path, trust_remote_code=True)
        
        print(f"Successfully loaded {model_type} model")
        print(f"  Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have internet connection and the model path is correct.")
        print("Available models: c-bone/CrystaLLM-2.0_base, c-bone/CrystaLLM-2.0_SLME, c-bone/CrystaLLM-2.0_bandgap, c-bone/CrystaLLM-2.0_density")
        return None


def get_material_id(row, count, offset=0):
    """Get material ID from row data or generate one."""
    if "Material ID" in row:
        return row["Material ID"]
    elif "Formula" in row:
        return row["Formula"]
    else:
        return f"Generated_{count + offset + 1}"


def generate_prompts_from_args(args):
    """Generate prompts from command line arguments."""
    
    if args.input_parquet:
        print(f"Loading existing prompts from: {args.input_parquet}")
        return pd.read_parquet(args.input_parquet)
    
    elif args.manual:
        print("Creating prompts from compositions and conditions...")
        
        # Parse compositions
        compositions = [comp.strip() for comp in args.compositions.split(',')]
        
        # Parse condition lists
        condition_lists = []
        if args.condition_lists:
            for cond_list in args.condition_lists:
                values = [float(x.strip()) for x in cond_list.split(',')]
                if any(v < 0 or v > 1 for v in values):
                    print(f"WARNING: Condition values should be normalized [0-1]. Found: {values}")
                condition_lists.append(values)
        
        # Parse spacegroups (level_4 only)
        spacegroups = [sg.strip() for sg in args.spacegroups.split(',')] if args.spacegroups else None
        
        df = create_manual_prompts(
            compositions=compositions,
            condition_lists=condition_lists,
            raw_mode=args.raw,
            level=args.level,
            spacegroups=spacegroups
        )
        
        print(f"Created {len(df)} prompts at {args.level}")
        return df
    
    else:
        raise ValueError("Must specify either --input_parquet or --manual")


def setup_generation_args(args):
    """Create generation arguments object for the generation pipeline."""
    class GenerationArgs:
        def __init__(self):
            # Standard evaluation arguments
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
            
            # Model type
            self.activate_conditionality = args.model_type
    
    return GenerationArgs()


def generate_cifs_with_hf_model(df_prompts, hf_model_path, gen_args):
    """Generate CIFs using a Hugging Face model with proper generate_on_gpu logic.
    
    This function now uses the same generation logic as generate_on_gpu to ensure:
    - Proper conditional model handling (PKV, Prepend, Slider)
    - Correct EOS token truncation and sequence processing  
    - Optional CIF validation and scoring
    """
    print(f"Generating CIFs with model: {hf_model_path}")
    
    # Initialize tokenizer
    tokenizer_dir = TOKENIZER_DIR
    tokenizer = init_tokenizer(tokenizer_dir)
    
    # Load model
    model = load_hf_model(hf_model_path, gen_args.activate_conditionality)
    if model is None:
        raise RuntimeError("Failed to load model. Please check the model path and try again.")
    
    model.eval()
    model.resize_token_embeddings(len(tokenizer))
    
    # Setup device
    device = setup_device(0)
    model = model.to(device)
    
    # Build generation kwargs
    generation_kwargs = build_generation_kwargs(gen_args, tokenizer, model.config.n_positions)
    print(f"Generation settings: {generation_kwargs}")
    
    # Use generate_on_gpu logic for proper CIF generation
    all_results = []
    
    for idx, row in df_prompts.iterrows():        
        # Tokenize prompt
        input_ids = tokenizer.encode(row["Prompt"], return_tensors="pt").to(device)
        
        valid_cifs = []
        generation_attempts = 0
        
        # Use scoring mode logic similar to generate_on_gpu
        scoring_mode = gen_args.scoring_mode
        max_return_attempts = gen_args.max_return_attempts
        
        if scoring_mode == "None":
            target_generations = generation_kwargs.get("num_return_sequences", 1) * max_return_attempts
            max_attempts = max_return_attempts
        else:
            target_generations = gen_args.target_valid_cifs
            max_attempts = max_return_attempts
        
        while len(valid_cifs) < target_generations and generation_attempts < max_attempts:
            generation_attempts += 1
            
            try:
                # Handle different conditionality types like generate_on_gpu
                if gen_args.activate_conditionality in ["PKV", "Prepend", "Slider"]:
                    # Parse condition vector for conditional models
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
                    # Handle Raw or unconditional generation
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=input_ids,
                            return_dict_in_generate=True,
                            output_scores=True,
                            **generation_kwargs,
                        )
                
                # Process each generated sequence using generate_on_gpu logic
                for seq_idx, output_seq in enumerate(outputs.sequences):
                    if torch.isnan(output_seq).any() or torch.isinf(output_seq).any():
                        continue
                    
                    # Find EOS token in the full sequence and truncate there
                    eos_idx = (output_seq == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if eos_idx.numel() > 0:
                        # Include everything from start to EOS (but not EOS itself)
                        full_sequence = output_seq[:int(eos_idx[0])]
                    else:
                        # No EOS found, use full sequence
                        full_sequence = output_seq
                    
                    # Decode full sequence (input + generated) and clean up CIF
                    cif_txt = tokenizer.decode(full_sequence, skip_special_tokens=True).replace("\n\n", "\n")
                    
                    # Apply remove_conditionality for Raw conditioning
                    if gen_args.activate_conditionality == "Raw":
                        cif_txt = remove_conditionality(cif_txt)
                    
                    if scoring_mode == "None":
                        # No validation or scoring - just collect all CIFs
                        mid = get_material_id(row, len(valid_cifs), 0)
                        
                        valid_cifs.append({
                            "Material ID": mid,
                            "Prompt": row["Prompt"],
                            "Generated CIF": cif_txt,
                            "condition_vector": row.get("condition_vector", "None"),
                        })
                    else:
                        # Validate CIF
                        is_consistent = check_cif(cif_txt)
                        
                        if is_consistent:
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
                    
                    # Break if we've reached our target
                    if len(valid_cifs) >= target_generations:
                        break
                        
            except Exception as e:
                print(f"Error generating for prompt {idx + 1}: {e}")
                continue
        
        # Process results based on scoring mode
        if valid_cifs:
            if scoring_mode == "None":
                # No ranking for unscored mode
                all_results.extend(valid_cifs)
                print(f"Generated {len(valid_cifs)} CIFs (no validation/scoring)")
            else:
                # Rank all CIFs for this prompt by score 
                if scoring_mode.upper() == "LOGP":
                    ranked_cifs = sorted(valid_cifs, key=lambda x: x["score"] if not np.isnan(x["score"]) and not np.isinf(x["score"]) else float('inf'), reverse=False)
                else:
                    ranked_cifs = valid_cifs
                
                for rank, cif_data in enumerate(ranked_cifs, 1):
                    cif_data["rank"] = rank
                
                all_results.extend(ranked_cifs)
                if ranked_cifs:
                    best_score = ranked_cifs[0]["score"]
                    worst_score = ranked_cifs[-1]["score"]
                    print(f"Generated {len(valid_cifs)} valid CIFs, {scoring_mode} scores: {best_score:.4f} to {worst_score:.4f}")
        else:
            print(f"Failed to generate any CIFs after {generation_attempts} attempts")
    
    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument("--hf_model_path", required=True,
                       help="HuggingFace model path (c-bone/CrystaLLM-2.0_base, c-bone/CrystaLLM-2.0_SLME, c-bone/CrystaLLM-2.0_bandgap, c-bone/CrystaLLM-2.0_density, c-bone/CrystaLLM-2.0_COD-XRD)")
    parser.add_argument("--output_parquet", required=True,
                       help="Output parquet file path for generated CIF structures")
    
    # Prompt source (mutually exclusive)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--input_parquet", 
                             help="Input parquet file with existing prompts")
    prompt_group.add_argument("--manual", action="store_true",
                             help="Create prompts manually from compositions and conditions")
    
    # Manual prompt arguments
    parser.add_argument("--compositions", 
                       help="Comma-separated chemical compositions (e.g., 'LiFePO4,Si,TiO2')")
    parser.add_argument("--condition_lists", nargs='+',
                       help="Normalized condition values [0-1]. Base: none, SLME: 1 list, bandgap/density: 2 lists")
    parser.add_argument("--level", choices=["level_1", "level_2", "level_3", "level_4"],
                       default="level_2", help="Prompt detail level")
    parser.add_argument("--spacegroups",
                       help="Comma-separated spacegroups (only for level_4)")
    
    # Generation arguments (aligned with evaluation standards)
    # add a gen_config arg
    parser.add_argument("--do_sample", type=str, default="True", 
                       help="Enable sampling for generation (True/False/beam)")
    parser.add_argument("--top_k", type=int, default=15,
                       help="Number of highest probability tokens to consider for top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Cumulative probability threshold for nucleus (top-p) sampling")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for controlling randomness in generation")
    parser.add_argument("--gen_max_length", type=int, default=1024,
                       help="Maximum length of generated CIF sequences")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                       help="Number of CIF sequences to generate per sample")
    parser.add_argument("--max_return_attempts", type=int, default=1,
                       help="Number of generation runs per sample (total = max_return_attempts * num_return_sequences) if using logp scoring itll do until it hits either max_return attempts or the target of valid cifs.")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process from input (if a df_prompts is used)")
    parser.add_argument("--scoring_mode", type=str, default="None",
                       help="Scoring mode for generated structures: 'LOGP', 'None'")
    parser.add_argument("--target_valid_cifs", type=int, default=20,
                       help="Target number of valid CIFs to generate per prompt")
    
    # Model and processing arguments
    parser.add_argument("--model_type", choices=["PKV", "Prepend", "Slider", "Raw", "Base"], default="PKV",
                       help="Model architecture type")
    parser.add_argument("--raw", action="store_true",
                       help="Use raw conditioning format")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for post-processing")
    parser.add_argument("--skip_postprocess", action="store_true",
                       help="Skip CIF cleaning and validation")
    
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    
    # Basic validation and info display
    print(f"Model: {args.hf_model_path}")
    model_info = MODEL_INFO.get(args.hf_model_path)
    if model_info:
        print(f"Type: {model_info['description']}")
    
    if args.manual:
        if not args.compositions:
            if not args.level.startswith("level_1"):
                parser.error("Manual mode requires --compositions for level_2, level_3, and level_4")

        
        # Simple condition validation
        if model_info:
            expected = model_info['conditions']
            provided = len(args.condition_lists) if args.condition_lists else 0
            
            if expected == 0 and args.condition_lists:
                print("Base model is unconditional - ignoring conditions")
            elif expected > 0 and expected != provided:
                example = model_info.get('example_conditions', [])
                parser.error(f"Model needs {expected} condition list(s), got {provided}. "
                           f"Example: --condition_lists {' '.join(example)}")
        
        print(f"Compositions: {args.compositions}")
        if args.condition_lists:
            print(f"Conditions: {args.condition_lists}")
    else:
        print(f"Input: {args.input_parquet}")
    
    
    # Generate prompts
    print("\n****** Generating Prompts *******")
    df_prompts = generate_prompts_from_args(args)
    print(f"Got {len(df_prompts)} prompts")

    if args.verbose:
        print(f"\nExample prompt:\n{df_prompts.iloc[0]['Prompt']}\n")
    
    # Limit samples if specified
    if args.max_samples:
        df_prompts = df_prompts.head(args.max_samples)
        print(f"Limited to {len(df_prompts)} samples")
    
    # Setup generation arguments
    gen_args = setup_generation_args(args)
    
    # Generate CIFs
    print("\n****** Generating CIFs ******")
    df_generated = generate_cifs_with_hf_model(df_prompts, args.hf_model_path, gen_args)
    
    # Post-process CIFs
    if not args.skip_postprocess:
        print("\n****** Post-processing CIFs ******")
        df_final = process_dataframe(df_generated, args.num_workers, "Generated CIF")
    else:
        df_final = df_generated
    
    # Save results
    output_dir = os.path.dirname(args.output_parquet)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    df_final.to_parquet(args.output_parquet, index=False)

    
    print("\n****** Generation Complete ******")
    print(f"Total structures: {len(df_final)}")

    print(f"\nSaved to: {args.output_parquet}")


if __name__ == "__main__":
    main()