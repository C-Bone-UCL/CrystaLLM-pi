"""
Generate CIF structures using trained conditional/unconditional models.
Supports parallel generation across multiple GPUs with validation and scoring options.


IMPORTANT: This is the script at time of paper publication. For the latest version, please check the GitHub repository main branch.

- main changes in this version:
-- we dont check bond length reasonableness for validity because we say in paper we just look at basic consistency within the cif
-- also during perplexity ranking, when we hit target valid CIFs we stop processing within a generated batch to calculate perplexities (eg if you do target valid CIFs = 1 but generate 20 in a batch, we stop as soon as we get 1 valid CIF and calculate perplexity only for that one, instead of calculating perplexity for all 20 and then ranking)
"""

import os
import multiprocessing as mp
import sys
import json
import re
import warnings
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

# Enable TF32 and cudnn optimizations when CUDA is available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tokenizer import CustomCIFTokenizer
from _models import PKVGPT, PrependGPT, SliderGPT
from _args import parse_args
from _utils import find_checkpoint_from_dir, is_sensible, is_formula_consistent, is_space_group_consistent, extract_space_group_symbol, replace_symmetry_operators

model = None
tokenizer = None

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
    except Exception:
        return False
    
def _score_transition_slice(generated_scores, original_sequence, input_length, eos_token_id=None):
    """Score one generated sequence from a precomputed transition-score row."""
    scoring_length = len(original_sequence)

    if eos_token_id is not None:
        eos_positions = (original_sequence == eos_token_id).nonzero(as_tuple=True)[0]
        if eos_positions.numel() > 0:
            scoring_length = int(eos_positions[0])

    max_score_idx = min(scoring_length - 1, len(generated_scores) - 1)
    if max_score_idx < 0:
        return float('inf')

    generated_only_scores = generated_scores[input_length:max_score_idx + 1]

    if len(generated_only_scores) == 0:
        return float('inf')

    if torch.isnan(generated_only_scores).any() or torch.isinf(generated_only_scores).any():
        valid_scores = generated_only_scores[~(torch.isnan(generated_only_scores) | torch.isinf(generated_only_scores))]
        if len(valid_scores) == 0:
            return float('inf')
        generated_only_scores = valid_scores

    mean_log_prob = torch.mean(generated_only_scores).item()
    return np.exp(-mean_log_prob)


def score_output_logp(model, scores, full_sequences, sequence_idx, input_length, eos_token_id=None):
    """Score one generated output based on transition scores up to EOS if present."""
    batch_scores = score_outputs_logp(
        model=model,
        scores=scores,
        full_sequences=full_sequences,
        input_length=input_length,
        eos_token_id=eos_token_id,
    )
    return batch_scores[sequence_idx]


def score_outputs_logp(model, scores, full_sequences, input_length, eos_token_id=None):
    """Score every generated output in a batch with one transition-score pass."""
    if scores is None or len(scores) == 0:
        if full_sequences is None:
            return [float('inf')]
        n_sequences = full_sequences.shape[0] if hasattr(full_sequences, "shape") else len(full_sequences)
        return [float('inf')] * n_sequences

    transition_scores = model.compute_transition_scores(
        full_sequences, scores, normalize_logits=True
    )

    if transition_scores.dim() == 1:
        transition_scores = transition_scores.unsqueeze(0)

    scored_outputs = []
    for sequence_idx, original_sequence in enumerate(full_sequences):
        scored_outputs.append(
            _score_transition_slice(
                generated_scores=transition_scores[sequence_idx],
                original_sequence=original_sequence,
                input_length=input_length,
                eos_token_id=eos_token_id,
            )
        )

    return scored_outputs
        

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
    """Get model's max length from config.json without loading the full model."""
    config_path = os.path.join(model_ckpt_dir, "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("n_positions", DEFAULT_MAX_LENGTH)
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

def get_material_id(row, count, offset=0):
    """Get material ID from row data or generate one, appending a unique counter."""
    base_id = row.get("Material ID") or row.get("Formula") or "Generated"
    return f"{base_id}_{count + offset + 1}"


def _normalize_scoring_mode(mode):
    """Normalize scoring mode to 'none' or 'logp'."""
    if mode is None or str(mode).lower() in ("none", "null", ""):
        return "none"
    return str(mode).lower()


def resolve_generation_plan(scoring_mode, target_valid_cifs, max_return_attempts, num_return_sequences, total_samples):
    """Resolve validation and stopping behavior for a generation run."""
    normalized_scoring_mode = _normalize_scoring_mode(scoring_mode)
    need_scores = (normalized_scoring_mode == "logp")
    check_validity = need_scores or (target_valid_cifs > 0)

    if check_validity:
        target_per_prompt = target_valid_cifs
    else:
        target_per_prompt = num_return_sequences * max_return_attempts

    return {
        "normalized_scoring_mode": normalized_scoring_mode,
        "need_scores": need_scores,
        "check_validity": check_validity,
        "target_per_prompt": target_per_prompt,
        "total_expected_generations": total_samples * target_per_prompt,
    }


def _load_worker_model(model_class, model_source_path, model_source, dtype):
    """Load worker model from local checkpoint or HuggingFace source."""
    if model_source == "hf":
        try:
            return model_class.from_pretrained(
                model_source_path,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                trust_remote_code=True,
            ).eval()
        except Exception:
            return model_class.from_pretrained(
                model_source_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).eval()

    try:
        return model_class.from_pretrained(
            model_source_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).eval()
    except Exception:
        return model_class.from_pretrained(model_source_path, torch_dtype=dtype).eval()


def init_worker(
    model_ckpt_dir,
    pretrained_tokenizer_dir,
    activate_conditionality,
    base_seed=1,
    model_source="checkpoint",
):
    global model, tokenizer
    
    tokenizer = init_tokenizer(pretrained_tokenizer_dir)
    model_class = get_model_class(activate_conditionality)
    
    # Determine dtype: prefer bfloat16, fall back to float16 if unsupported
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    model = _load_worker_model(model_class, model_ckpt_dir, model_source, dtype)
    model.resize_token_embeddings(len(tokenizer))
    
    # Deterministic seeding per worker
    gpu_id = int(os.environ.get("LOCAL_RANK", 0))
    worker_seed = base_seed + gpu_id
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)

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
    activate_conditionality,
    global_offset=0,
    scoring_mode="none",
    target_valid_cifs=0,
    max_return_attempts=2,
    base_seed=1,
):
    global model, tokenizer
    
    device = setup_device(gpu_id)
    model = model.to(device)
    results = []
    
    # Deterministic seeding per GPU
    worker_seed = base_seed + gpu_id
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)
    
    # Normalize scoring mode
    scoring_mode = _normalize_scoring_mode(scoring_mode)
    need_scores = (scoring_mode == "logp")
    check_validity = need_scores or (target_valid_cifs > 0)
    
    # Process each prompt individually
    for idx in range(start_idx, end_idx):
        row = prompts.iloc[idx]
        input_ids = tokenizer.encode(row["Prompt"], return_tensors="pt").to(device)
        
        valid_cifs = []
        generation_attempts = 0
        progress_made = 0
        
        # For 'none' scoring mode without a validity target, collect all without validation
        if not check_validity:
            target_generations = generation_kwargs.get("num_return_sequences", 1) * max_return_attempts
            max_attempts = max_return_attempts
        else:
            # Generate until we have target_valid_cifs valid ones or hit max attempts
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
                    
                    with torch.inference_mode():
                        outputs = model.generate(
                            input_ids=input_ids,
                            condition_values=condition_tensor,
                            return_dict_in_generate=True,
                            output_scores=need_scores,
                            **generation_kwargs,
                        )
                else:
                    # Handle Raw or unconditional generation
                    with torch.inference_mode():
                        outputs = model.generate(
                            input_ids=input_ids,
                            return_dict_in_generate=True,
                            output_scores=need_scores,
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
                    
                    if not check_validity:
                        # No validation or scoring - just collect all CIFs
                        mid = get_material_id(row, len(valid_cifs), global_offset)
                        valid_cifs.append({
                            "Material ID": mid,
                            "Prompt": row["Prompt"],
                            "Generated CIF": cif_txt,
                            "condition_vector": row.get("condition_vector", "None"),
                        })
                        if progress_made < target_generations:
                            queue.put(1)
                            progress_made += 1
                    else:
                        # Validate CIF
                        is_consistent = check_cif(cif_txt)
                        
                        if is_consistent:
                            if need_scores:
                                score = score_output_logp(model, outputs.scores, outputs.sequences, seq_idx, input_ids.shape[1], tokenizer.eos_token_id)
                            else:
                                score = -100
                            
                            mid = get_material_id(row, len(valid_cifs), global_offset)
                            valid_cifs.append({
                                "Material ID": mid,
                                "Prompt": row["Prompt"],
                                "Generated CIF": cif_txt,
                                "is_consistent": True,
                                "score": score,
                                "condition_vector": row.get("condition_vector", "None"),
                            })
                            if progress_made < target_generations:
                                queue.put(1)
                                progress_made += 1
                    
                    # Break if we've reached our target
                    if len(valid_cifs) >= target_generations:
                        break
                        
            except Exception:
                continue
        
        # Process results based on scoring mode
        if valid_cifs:
            if not check_validity:
                results.extend(valid_cifs[:target_generations])
            else:
                if need_scores:
                    ranked_cifs = sorted(
                        valid_cifs,
                        key=lambda x: x["score"] if not np.isnan(x["score"]) and not np.isinf(x["score"]) else float('inf'),
                        reverse=False,
                    )
                else:
                    ranked_cifs = valid_cifs
                
                best_cifs = ranked_cifs[:target_generations]
                for rank, cif_data in enumerate(best_cifs, 1):
                    cif_data["rank"] = rank
                results.extend(best_cifs)
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return results

def build_output_df(data, args, df_prompts):
    """Build the final output dataframe."""
    df = pd.DataFrame(data)
    if args.input_parquet and 'True CIF' in df_prompts.columns:
        df_prompts['Material ID'] = df_prompts['Material ID'].astype(str)
        df['Material ID'] = df['Material ID'].astype(str)
        df = df.merge(df_prompts[['Material ID', 'True CIF']], on='Material ID', how='left')
    return df

def main():
    args = parse_args()
    
    # Check required arguments
    if not args.model_ckpt_dir:
        sys.exit("ERROR: model_ckpt_dir is required")
    if not args.input_parquet:
        sys.exit("ERROR: input_parquet is required")
    if not args.output_parquet:
        sys.exit("ERROR: output_parquet is required")
    
    print("Environment info")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print()
    print("Generation settings")
    print(f"Max raw generations per prompt-condition pair: {args.num_return_sequences * args.max_return_attempts}")
    print(f"Will save generated CIFs to {args.output_parquet}")
    

    # Find checkpoint if needed
    if 'checkpoint' not in args.model_ckpt_dir:
        args.model_ckpt_dir = find_checkpoint_from_dir(args.model_ckpt_dir)
    
    # Display model checkpoint info
    losses_file = os.path.join(args.model_ckpt_dir, "losses.json")
    if os.path.exists(losses_file):
        with open(losses_file, "r") as f:
            losses = json.load(f)
        print()
        print("Model checkpoint info")
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
    
    # Normalize scoring mode and set base seed
    base_seed = getattr(args, 'seed', 1)
    plan = resolve_generation_plan(
        scoring_mode=args.scoring_mode,
        target_valid_cifs=args.target_valid_cifs,
        max_return_attempts=args.max_return_attempts,
        num_return_sequences=generation_kwargs.get("num_return_sequences", 1),
        total_samples=total_samples,
    )
    scoring_mode = plan["normalized_scoring_mode"]

    if not plan["check_validity"]:
        print(f"Target CIFs per prompt: {plan['target_per_prompt']} (no validation/scoring)")
        print("Will save all generated CIFs without validation or ranking")
    elif plan["need_scores"]:
        print(f"Target valid CIFs per prompt: {args.target_valid_cifs}")
        print(f"Will save all CIFs ranked by {scoring_mode} score (up to {args.target_valid_cifs} per prompt)")
    else:
        print(f"Target valid CIFs per prompt: {args.target_valid_cifs}")
        print("Will save valid CIFs only, without ranking")
    

    # Setup multiprocessing
    manager = mp.Manager()
    queue = manager.Queue()
    total_expected_generations = plan["total_expected_generations"]
    
    
    # Determine splitting strategy
    is_single_prompt = (total_samples == 1) and (num_gpus > 1)

    target_valid_cifs = getattr(args, 'target_valid_cifs', 20)

    results = []
    # Use spawn context for CUDA safety
    ctx = mp.get_context('spawn')
    with ctx.Pool(num_gpus + 1,
                  initializer=init_worker,
                  initargs=(args.model_ckpt_dir, DEFAULT_TOKENIZER_DIR, args.activate_conditionality, base_seed)) as pool:
        
        pool.apply_async(progress_listener, (queue, total_expected_generations))
        
        try:
            # Handle case with 0 GPUs
            if num_gpus == 0:
                results.append(pool.apply_async(
                    generate_on_gpu,
                    (0, df_prompts, generation_kwargs, queue, 0, total_samples,
                    args.activate_conditionality, 0, scoring_mode, target_valid_cifs, args.max_return_attempts, base_seed)
                ))
            
            elif is_single_prompt:
                # SINGLE PROMPT MODE: Split workload (attempts/targets) across GPUs
                print("\nSingle prompt detected. Distributing workload across GPUs.")
                
                # Split total work among GPUs
                base_target = target_valid_cifs
                base_attempts = args.max_return_attempts
                
                # Calculate distribution
                targets_per_gpu = [base_target // num_gpus] * num_gpus
                attempts_per_gpu = [base_attempts // num_gpus] * num_gpus
                
                # Distribute remainders
                for i in range(base_target % num_gpus):
                    targets_per_gpu[i] += 1
                for i in range(base_attempts % num_gpus):
                    attempts_per_gpu[i] += 1
                    
                current_offset = 0
                
                for gpu_id in range(num_gpus):
                    # Ensure at least 1 attempt if there's work to do
                    local_attempts = max(1, attempts_per_gpu[gpu_id]) if base_attempts > 0 else 0
                    local_target = max(1, targets_per_gpu[gpu_id]) if base_target > 0 else 0
                    
                    if local_attempts == 0 and local_target == 0:
                        continue
                    
                    # Pass the full dataframe (1 row) to everyone, same index (0 to 1)
                    results.append(pool.apply_async(
                        generate_on_gpu,
                        (gpu_id, df_prompts, generation_kwargs, queue, 0, 1,
                         args.activate_conditionality, current_offset, scoring_mode, local_target, local_attempts, base_seed)
                    ))
                    
                    # Estimate offset increment for next worker to avoid ID collision
                    if not plan["check_validity"]:
                        expected_gen = local_attempts * generation_kwargs.get("num_return_sequences", 1)
                        current_offset += expected_gen
                    else:
                        current_offset += local_target

            else:
                # STANDARD MODE: Split prompts across GPUs
                samples_per_gpu = total_samples // num_gpus
                for gpu_id in range(num_gpus):
                    start = gpu_id * samples_per_gpu
                    end = (gpu_id + 1) * samples_per_gpu if gpu_id != num_gpus - 1 else total_samples
                    global_offset = start  # Simple offset based on start index
                    results.append(pool.apply_async(
                        generate_on_gpu,
                        (gpu_id, df_prompts, generation_kwargs, queue, start, end,
                        args.activate_conditionality, global_offset, scoring_mode, target_valid_cifs, args.max_return_attempts, base_seed)
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
    df = build_output_df(generated_data, args, df_prompts)

    # if there is a directory in the output path, create it
    output_dir = os.path.dirname(args.output_parquet)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:    
        df.to_parquet(args.output_parquet, index=False)
        print(f"\nSaved {len(df)} CIFs to {args.output_parquet}")
        if not plan["check_validity"]:
            print("Results include all generated CIFs (no validation or scoring applied).")
        elif plan["need_scores"]:
            print(f"Results include all generated CIFs ranked by {scoring_mode} score per prompt.")
        else:
            print("Results include valid CIFs only (no scoring applied).")
    except Exception as e:
        # do a fallback to saving a 'fallback.parquet' file in the current directory
        fallback_path = "fallback.parquet"
        df.to_parquet(fallback_path, index=False)
        print(f"\nERROR: Could not save to {args.output_parquet} due to: {e}")
        print(f"Saved output to fallback file {fallback_path} instead.")
    

    # Cleanup
    manager.shutdown()


def run_generation_pool(
    df_prompts,
    generation_kwargs,
    activate_conditionality,
    scoring_mode,
    target_valid_cifs,
    max_return_attempts,
    base_seed=1,
    worker_count=None,
    initargs_override=None,
):
    """Run generation across workers and return generated row dicts."""
    if initargs_override is None:
        initargs_override = ()

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    worker_count = worker_count or num_gpus
    worker_count = max(1, int(worker_count))
    worker_count = min(worker_count, max(1, num_gpus))

    total_samples = len(df_prompts)
    plan = resolve_generation_plan(
        scoring_mode=scoring_mode,
        target_valid_cifs=target_valid_cifs,
        max_return_attempts=max_return_attempts,
        num_return_sequences=generation_kwargs.get("num_return_sequences", 1),
        total_samples=total_samples,
    )
    normalized_scoring_mode = plan["normalized_scoring_mode"]
    check_validity = plan["check_validity"]
    total_expected_generations = plan["total_expected_generations"]

    is_single_prompt = (total_samples == 1) and (worker_count > 1)

    generated_data = []
    results = []
    ctx = mp.get_context('spawn')
    manager = mp.Manager()
    queue = manager.Queue()
    with ctx.Pool(worker_count + 1, initializer=init_worker, initargs=initargs_override) as pool:
        listener = pool.apply_async(progress_listener, (queue, total_expected_generations))

        if worker_count == 1:
            results.append(pool.apply_async(
                generate_on_gpu,
                (0, df_prompts, generation_kwargs, queue, 0, total_samples,
                 activate_conditionality, 0, normalized_scoring_mode, target_valid_cifs, max_return_attempts, base_seed)
            ))

        elif is_single_prompt:
            base_goal = target_valid_cifs if check_validity else max_return_attempts
            goal_per_gpu = [base_goal // worker_count] * worker_count
            for i in range(base_goal % worker_count):
                goal_per_gpu[i] += 1

            current_offset = 0
            for gpu_id in range(worker_count):
                local_goal = goal_per_gpu[gpu_id]
                if local_goal == 0:
                    continue
                l_target = local_goal if check_validity else 0
                l_attempts = local_goal
                results.append(pool.apply_async(
                    generate_on_gpu,
                    (gpu_id, df_prompts, generation_kwargs, queue, 0, 1,
                     activate_conditionality, current_offset, normalized_scoring_mode,
                     l_target, l_attempts, base_seed)
                ))
                if not check_validity:
                    current_offset += local_goal * generation_kwargs.get("num_return_sequences", 1)
                else:
                    current_offset += local_goal

        else:
            samples_per_gpu = max(1, total_samples // worker_count)
            for gpu_id in range(worker_count):
                start = gpu_id * samples_per_gpu
                end = (gpu_id + 1) * samples_per_gpu if gpu_id != worker_count - 1 else total_samples
                if start >= total_samples:
                    continue
                results.append(pool.apply_async(
                    generate_on_gpu,
                    (gpu_id, df_prompts, generation_kwargs, queue, start, end,
                     activate_conditionality, start, normalized_scoring_mode, target_valid_cifs, max_return_attempts, base_seed)
                ))

        for res in results:
            generated_data.extend(res.get())

        queue.put("kill")
        listener.get()

    manager.shutdown()

    if is_single_prompt and generated_data:
        base_id = generated_data[0]["Material ID"].rsplit("_", 1)[0]
        for i, row in enumerate(generated_data):
            row["Material ID"] = f"{base_id}_{i + 1}"

    return generated_data


if __name__ == "__main__":
    # Set start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    main()