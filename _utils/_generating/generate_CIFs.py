"""

"""

import os
import multiprocessing as mp
import sys
import json
import time
import re
import warnings
import gc
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel

# Global constants
DEFAULT_MAX_LENGTH = 1024
TOKENIZER_PAD_TOKEN = "<pad>"
DEFAULT_TOKENIZER_DIR = "HF-cif-tokenizer"

# Global warning filters
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tokenizer import CustomCIFTokenizer
from _models import PKVGPT, PrependGPT, SliderGPT
from _args import parse_args
from _utils import find_checkpoint_from_dir, is_sensible, is_formula_consistent, is_space_group_consistent, extract_space_group_symbol, replace_symmetry_operators

global_model = None
global_tokenizer = None

def check_cif(cif_str):
    """Check if CIF string is self-consistent.""" 
    try:
        space_group_symbol = extract_space_group_symbol(cif_str)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            cif_str = replace_symmetry_operators(cif_str, space_group_symbol)

        if not is_sensible(cif_str):
            return False
        if not is_formula_consistent(cif_str):
            return False
        if not is_space_group_consistent(cif_str):
            return False
        
        return True
    except Exception as e:
        return False
    
def score_output_logp(model, scores, full_sequences, sequence_idx, input_length, eos_token_id=None):
    """Score output based on perplexity of generated tokens up to EOS if present."""
    
    if scores is None or len(scores) == 0:
        return float('inf')
    
    # Compute transition scores for all sequences
    transition_scores = model.compute_transition_scores(
        full_sequences, scores, normalize_logits=True
    )
    
    # Extract scores for the specific sequence
    if transition_scores.dim() == 2:
        generated_scores = transition_scores[sequence_idx]
    else:
        generated_scores = transition_scores[0] if transition_scores.dim() > 1 else transition_scores
    
    # Find EOS token position in the original sequence to determine scoring length
    original_sequence = full_sequences[sequence_idx]
    scoring_length = len(original_sequence)
    
    if eos_token_id is not None:
        eos_positions = (original_sequence == eos_token_id).nonzero(as_tuple=True)[0]
        if eos_positions.numel() > 0:
            # Score up to (but not including) the first EOS token
            eos_pos = int(eos_positions[0])
            scoring_length = eos_pos
    
    # Adjust transition scores to match the scoring length
    # transition_scores has length = original_sequence_length - 1 (no score for first token)
    max_score_idx = min(scoring_length - 1, len(generated_scores) - 1)
    if max_score_idx < 0:
        return float('inf')
    
    # Get scores only for generated tokens (skip input) up to scoring length
    generated_only_scores = generated_scores[input_length:max_score_idx + 1]
    
    if len(generated_only_scores) == 0:
        return float('inf')
    
    # Filter out any NaN/inf values if they exist
    if torch.isnan(generated_only_scores).any() or torch.isinf(generated_only_scores).any():
        valid_scores = generated_only_scores[~(torch.isnan(generated_only_scores) | torch.isinf(generated_only_scores))]
        if len(valid_scores) == 0:
            return float('inf')
        generated_only_scores = valid_scores
    
    # Calculate perplexity: exp(-mean_log_probability)
    mean_log_prob = torch.mean(generated_only_scores).item()
    perplexity = np.exp(-mean_log_prob)
    
    return perplexity
        

def init_tokenizer(pretrained_tokenizer_dir):
    """Initialize tokenizer with standard config."""
    tokenizer = CustomCIFTokenizer.from_pretrained(
        pretrained_dir=pretrained_tokenizer_dir,
        pad_token=TOKENIZER_PAD_TOKEN
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer

def get_model_class(conditionality_type):
    """Return the appropriate model class based on the conditionality type."""
    if conditionality_type == "PKV":
        return PKVGPT
    elif conditionality_type == "Prepend":
        return PrependGPT
    elif conditionality_type == "Slider":
        return SliderGPT
    else:
        # Default to GPT2LMHeadModel for unconditional or raw generation
        return GPT2LMHeadModel

def get_model_max_length(model_ckpt_dir, activate_conditionality):
    """Get model's max length from config by loading temp model."""
    model_class = get_model_class(activate_conditionality)
    
    try:
        temp_model = model_class.from_pretrained(model_ckpt_dir, low_cpu_mem_usage=True)
        n_positions = temp_model.config.n_positions
        del temp_model
        return n_positions
    except Exception:
        return DEFAULT_MAX_LENGTH


def build_generation_kwargs(args, tokenizer, max_length):
    """Build generation kwargs based on args and tokenizer."""
    base_kwargs = {
        "max_length": min(args.gen_max_length, max_length),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "renormalize_logits": True,
        "remove_invalid_values": True,
    }
    
    do_sample_str = str(args.do_sample).lower()
    
    if "true" in do_sample_str:
        base_kwargs.update({
            "num_return_sequences": args.num_return_sequences,
            "do_sample": True,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        })
    elif "false" in do_sample_str:
        base_kwargs.update({
            "num_return_sequences": 1,
            "do_sample": False,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 1.0,
        })
    elif args.do_sample == "beam":
        base_kwargs.update({
            "num_return_sequences": args.num_return_sequences,
            "do_sample": False,
            "num_beams": args.num_return_sequences,
            "top_k": 0,
            "top_p": 1.0,
            "temperature": 1.0,
        })
    
    return base_kwargs

def remove_conditionality(cif_str: str) -> str:
    """Remove comments preceding the 'data_' block in CIF string for Raw conditioning."""
    match = re.search(r'(data_.*)', cif_str, re.DOTALL)
    return match.group(1) if match else cif_str

def setup_device(gpu_id):
    """Setup device and return appropriate torch device."""
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device

def parse_condition_vector(condition_vector):
    """Parse condition vector string into tensor values."""
    if condition_vector in (None, "None"):
        return None
    
    if "," in str(condition_vector):
        return [float(x.strip()) for x in str(condition_vector).split(",")]
    else:
        return [float(condition_vector)]

def determine_material_id(row, generated_count, global_offset=0):
    """Determine material ID from row data."""
    if "Material ID" in row:
        return row["Material ID"]
    elif "Formula" in row:
        return row["Formula"]
    else:
        return f"Generated_{generated_count + global_offset + 1}"

def init_worker(model_ckpt_dir, pretrained_tokenizer_dir, activate_conditionality):
    global global_model, global_tokenizer
    
    global_tokenizer = init_tokenizer(pretrained_tokenizer_dir)
    
    # Load model based on conditionality type
    model_class = get_model_class(activate_conditionality)
    global_model = model_class.from_pretrained(model_ckpt_dir).eval()
    global_model.resize_token_embeddings(len(global_tokenizer))

def progress_listener(queue, total):
    pbar = tqdm(total=total, desc="Generating CIFs...")
    while True:
        message = queue.get()
        if message == "kill":
            break
        pbar.update(message)
    pbar.close()

def generate_on_gpu(
    gpu_id,
    prompts,
    generation_kwargs,
    queue,
    start_idx,
    end_idx,
    model_ckpt_dir,
    activate_conditionality,
    num_repeats,
    global_offset=0,
    scoring_mode="None",
    target_valid_cifs=20,
    max_return_attempts=2,
):
    global global_model, global_tokenizer
    
    device = setup_device(gpu_id)
    model = global_model.to(device)
    tokenizer = global_tokenizer
    all_results = []  # Store all ranked CIFs per prompt
    
    torch.cuda.manual_seed_all(int(time.time()) + os.getpid())
    
    # Process each prompt individually
    for idx in range(start_idx, end_idx):
        row = prompts.iloc[idx]
        input_ids = tokenizer.encode(row["Prompt"], return_tensors="pt").to(device)
        
        valid_cifs = []
        generation_attempts = 0
        
        # For 'None' scoring mode, generate num_return_sequences * MAX_GENERATION_ATTEMPTS without validation
        if scoring_mode == "None":
            target_generations = generation_kwargs.get("num_return_sequences", 1) * max_return_attempts
            max_attempts = max_return_attempts
        else:
            # Generate until we have TARGET_VALID_CIFS valid ones or hit max attempts
            target_generations = target_valid_cifs
            max_attempts = max_return_attempts
        
        while len(valid_cifs) < target_generations and generation_attempts < max_attempts:
            generation_attempts += 1
            
            try:
                # Handle different conditionality types
                if activate_conditionality in ["PKV", "Prepend", "Slider"]:
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
                
                # Process each generated sequence
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
                    if activate_conditionality == "Raw":
                        cif_txt = remove_conditionality(cif_txt)
                    
                    if scoring_mode == "None":
                        # No validation or scoring - just collect all CIFs
                        mid = determine_material_id(row, len(valid_cifs), global_offset)
                        
                        valid_cifs.append({
                            "Material ID": mid,
                            "Prompt": row["Prompt"],
                            "Generated CIF": cif_txt,
                            "condition_vector": row.get("condition_vector", "None"),
                        })
                        
                        queue.put(1)
                    else:
                        # Validate CIF
                        is_consistent = check_cif(cif_txt)
                        
                        if is_consistent:
                            if scoring_mode.upper() == "LOGP":
                                score = score_output_logp(model, outputs.scores, outputs.sequences, seq_idx, input_ids.shape[1], tokenizer.eos_token_id)
                            else:
                                score = 0.0
                            
                            mid = determine_material_id(row, len(valid_cifs), global_offset)
                            
                            valid_cifs.append({
                                "Material ID": mid,
                                "Prompt": row["Prompt"],
                                "Generated CIF": cif_txt,
                                "is_consistent": True,
                                "score": score,
                                "condition_vector": row.get("condition_vector", "None"),
                            })
                            
                            queue.put(1)
                    
                    # Break if we've reached our target
                    if len(valid_cifs) >= target_generations:
                        break
                
                # For 'None' scoring mode, break after first successful generation attempt
                # since we don't need validation - just collect all sequences from the generation
                if scoring_mode == "None" and valid_cifs:
                    break
                        
            except Exception:
                continue
        
        # Process results based on scoring mode
        if valid_cifs:
            if scoring_mode == "None":
                # No ranking for unscored mode
                all_results.extend(valid_cifs)
                print(f"Prompt {idx}: Generated {len(valid_cifs)} CIFs (no validation/scoring)")
            else:
                # Rank all CIFs for this prompt by score 
                # For LOGP (perplexity): lower perplexity = better (reverse=False)
                if scoring_mode.upper() == "LOGP":
                    ranked_cifs = sorted(valid_cifs, key=lambda x: x["score"] if not np.isnan(x["score"]) and not np.isinf(x["score"]) else float('inf'), reverse=False)
                
                for rank, cif_data in enumerate(ranked_cifs, 1):
                    cif_data["rank"] = rank
                
                all_results.extend(ranked_cifs)
                best_score = ranked_cifs[0]["score"]
                worst_score = ranked_cifs[-1]["score"]
                print(f"Prompt {idx}: Generated {len(valid_cifs)} valid CIFs, {scoring_mode} scores: {best_score:.4f} to {worst_score:.4f}")
        else:
            print(f"Prompt {idx}: Failed to generate any CIFs after {generation_attempts} attempts")
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return all_results

def create_output_dataframe(generated_data, args, df_prompts):
    """Create and format the output DataFrame."""
    df = pd.DataFrame(generated_data)
    if args.input_parquet and 'True CIF' in df_prompts.columns:
        df_prompts['Material ID'] = df_prompts['Material ID'].astype(str)
        df['Material ID'] = df['Material ID'].astype(str)
        df = df.merge(df_prompts[['Material ID', 'True CIF']], on='Material ID', how='left')
    return df

def main():
    args = parse_args()
    
    # Minimum required arguments
    required_args = ['model_ckpt_dir', 'input_parquet', 'output_parquet']
    for req_arg in required_args:
        if not getattr(args, req_arg):
            sys.exit(f"ERROR: {req_arg} argument must be set.")
    
    # Set defaults
    args.num_repeats = getattr(args, 'num_repeats', 1)

    # set to GPU 1, dont use GPU 0 which is used by other processes
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    print("Environment info")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\nGeneration settings")
    print(f"Total sequences per prompt-condition pair: {args.num_return_sequences * args.num_repeats}")
    print(f"Will save generated CIFs to {args.output_parquet}")
    

    # Find checkpoint if needed
    if 'checkpoint' not in args.model_ckpt_dir:
        args.model_ckpt_dir = find_checkpoint_from_dir(args.model_ckpt_dir)
    
    # Display model checkpoint info
    losses_file = os.path.join(args.model_ckpt_dir, "losses.json")
    if os.path.exists(losses_file):
        with open(losses_file, "r") as f:
            losses = json.load(f)
        print("\nModel checkpoint info")
        print(f"Most Recent Train Loss: {losses['training_losses'][-1]:.4f}")
        print(f"Most Recent Validation Loss: {losses['validation_losses'][-1]:.4f}")
    

    # Get model's max length from config
    n_positions = get_model_max_length(args.model_ckpt_dir, args.activate_conditionality)
    print(f"Model's max_length: {n_positions}")
    if n_positions < args.gen_max_length:
        print(f"WARNING: The model's max_length is {n_positions}, adjusting generation max_length")
    

    # Initialize tokenizer and build generation kwargs
    tokenizer = init_tokenizer(DEFAULT_TOKENIZER_DIR)
    generation_kwargs = build_generation_kwargs(args, tokenizer, n_positions)
    print(f"Generation kwargs: {generation_kwargs}")
    

    # Load and prepare data
    df_prompts = pd.read_parquet(args.input_parquet)
    if args.max_samples:
        df_prompts = df_prompts.sample(n=int(args.max_samples), random_state=1)
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    total_samples = len(df_prompts)
    
    print(f"\nGeneration Strategy")
    print(f"Number of condition-prompt pairs: {total_samples}")
    if args.scoring_mode == "None":
        target_per_prompt = generation_kwargs.get("num_return_sequences", 1) * args.max_return_attempts
        print(f"Target CIFs per prompt: {target_per_prompt} (no validation/scoring)")
        print(f"Will save all generated CIFs without validation or ranking")
    else:
        print(f"Target valid CIFs per prompt: {args.target_valid_cifs}")
        print(f"Will save all CIFs ranked by {args.scoring_mode} score (up to {args.target_valid_cifs} per prompt)")
    

    # Setup multiprocessing
    manager = mp.Manager()
    queue = manager.Queue()
    # Progress tracking: calculate expected generations based on scoring mode
    if args.scoring_mode == "None":
        target_per_prompt = generation_kwargs.get("num_return_sequences", 1) * args.max_return_attempts
        total_expected_generations = total_samples * target_per_prompt
    else:
        total_expected_generations = total_samples * args.target_valid_cifs
    samples_per_gpu = total_samples // num_gpus
    
    results = []
    # Use spawn context for CUDA safety
    ctx = mp.get_context('spawn')
    with ctx.Pool(num_gpus + 1,
                 initializer=init_worker,
                 initargs=(args.model_ckpt_dir, DEFAULT_TOKENIZER_DIR, args.activate_conditionality)) as pool:
        
        pool.apply_async(progress_listener, (queue, total_expected_generations))
        
        try:
            # Handle case with 0 GPUs
            if num_gpus == 0:
                results.append(pool.apply_async(
                    generate_on_gpu,
                    (0, df_prompts, generation_kwargs, queue, 0, total_samples,
                    args.model_ckpt_dir, args.activate_conditionality, args.num_repeats, 0, args.scoring_mode, args.target_valid_cifs, args.max_return_attempts)
                ))
            else:
                for gpu_id in range(num_gpus):
                    start = gpu_id * samples_per_gpu
                    end = (gpu_id + 1) * samples_per_gpu if gpu_id != num_gpus - 1 else total_samples
                    global_offset = start  # Simple offset based on start index
                    results.append(pool.apply_async(
                        generate_on_gpu,
                        (gpu_id, df_prompts, generation_kwargs, queue, start, end,
                        args.model_ckpt_dir, args.activate_conditionality, args.num_repeats, global_offset, args.scoring_mode, args.target_valid_cifs, args.max_return_attempts)
                    ))
        except Exception as e:
            print(f"Generation error (check activate_conditionality setting): {e}")
            pool.terminate()
            manager.shutdown()
            sys.exit(1)
        
        # Collect results
        generated_data = []
        for res in results:
            generated_data.extend(res.get())
        queue.put("kill")
    
    # Create and save output
    df = create_output_dataframe(generated_data, args, df_prompts)

    # if there is a directory in the output path, create it
    output_dir = os.path.dirname(args.output_parquet)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df.to_parquet(args.output_parquet, index=False)
    print(f"\nSaved {len(df)} CIFs to {args.output_parquet}")
    if args.scoring_mode == "None":
        print(f"Results include all generated CIFs (no validation or scoring applied).")
    else:
        print(f"Results include all generated CIFs ranked by {args.scoring_mode} score per prompt.")
    

    # Cleanup
    manager.shutdown()
    gc.collect()

if __name__ == "__main__":
    # Set start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()
