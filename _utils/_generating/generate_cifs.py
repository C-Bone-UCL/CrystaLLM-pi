r"""Generate CIF structures across GPU workers with validation and optional ranking.

Candidates are validated and surviving structures can be ranked by model perplexity or XRD fit
before selection. The workflow accepts already-built prompts and can resolve a run directory to its
newest checkpoint.

Usage:
    ```bash
    python _utils/_generating/generate_cifs.py \
        --config _config_files/generation/conditional/slme/slme-PKV-opt_eval.jsonc
    ```
"""

import argparse
import ast
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

# # set CUDA visible to "1"
# os.environ["CUDA_AVAILABLE_DEVICES"] = "1"
# # only do on GPU 1 (not 0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _tokenizer import CustomCIFTokenizer
from _models import PKVGPT, SliderGPT
from _utils.model import MODEL_REGISTRY
from _args import parse_args
from _utils import find_checkpoint_from_dir, is_sensible, is_formula_consistent, is_space_group_consistent, extract_space_group_symbol, replace_symmetry_operators, bond_length_reasonableness_score

model = None
tokenizer = None

def check_cif(cif_str: str, check_bond_length: bool=True) -> bool:
    """Check if CIF string is structurally and chemically self-consistent.

    Callers set `check_bond_length` from the screening profile. The published numbers had it off.
    """
    if not cif_str:
        return False
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
        if check_bond_length:
            # None means disordered, so not checked.
            bond_length_score = bond_length_reasonableness_score(cif_str)
            if bond_length_score is not None and bond_length_score < 1.0:
                return False

        return True
    except Exception:
        return False
        






def init_tokenizer(pretrained_tokenizer_dir: str) -> CustomCIFTokenizer:
    """Load the CIF tokenizer with the padding configuration required for generation.

    If the tokenizer has no pad token, the EOS token is used as the pad token for
    batched generation.
    """
    tokenizer = CustomCIFTokenizer.from_pretrained(
        pretrained_dir=pretrained_tokenizer_dir,
        pad_token=TOKENIZER_PAD_TOKEN
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer

def get_model_class(conditionality_type: str | None) -> type:
    """Strict lookup: unknown names raise instead of silently loading GPT2."""
    if conditionality_type in (None, "None", "Base"):
        return GPT2LMHeadModel
    if conditionality_type in MODEL_REGISTRY:
        return MODEL_REGISTRY[conditionality_type][1]
    raise ValueError(f"Unknown model type {conditionality_type!r}. "
                     f"Valid: Base, {', '.join(sorted(k for k in MODEL_REGISTRY if k))}")

def get_model_max_length(model_ckpt_dir: str, activate_conditionality: str | None) -> int:
    """Get the usable text length from config.json without loading the full model."""
    config_path = os.path.join(model_ckpt_dir, "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        n_positions = config.get("n_positions", DEFAULT_MAX_LENGTH)
        if activate_conditionality in ("Prefix", "PrefixXRD"):
            # Prefix families extend wpe by n_prefix_tokens, so the text budget excludes them.
            # PKV is deliberately NOT subtracted so legacy hub models generate identically.
            return max(n_positions - config.get("n_prefix_tokens", 0), 1)
        return n_positions
    except Exception:
        return DEFAULT_MAX_LENGTH


def build_generation_kwargs(args: argparse.Namespace, tokenizer: CustomCIFTokenizer, max_length: int) -> dict:
    """Assemble the keyword arguments passed to `model.generate`.

    Clamps `max_length` to the model's own context window, so a longer request cannot overrun the
    positional embeddings, and sets the sampling knobs from `args`.
    """
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


def parse_condition_vector(condition_vector: object) -> list | None:
    """Normalize a condition vector from any of the formats a parquet round trip can produce.

    Accepts a scalar, a comma-separated string, a repr string, a list, or a numpy array, and returns
    a list of floats. Nested sequences stay nested, which keeps a `(1000, 2)` continuous-XRD profile
    as `[Q, I]` pairs instead of flattening it.

    Returns None for a genuine absence: None itself, the string "None", or NaN.
    """
    if condition_vector is None:
        return None
    if isinstance(condition_vector, str) and condition_vector.strip().lower() == "none":
        return None
    if isinstance(condition_vector, (float, np.floating)) and np.isnan(condition_vector):
        return None

    # str -> try literal_eval first, then comma split
    if isinstance(condition_vector, str):
        try:
            condition_vector = ast.literal_eval(condition_vector)
        except (ValueError, SyntaxError):
            if "," in condition_vector:
                return [float(x.strip()) for x in condition_vector.split(",")]
            return [float(condition_vector)]

    if isinstance(condition_vector, (list, tuple, np.ndarray)):
        return [
            [float(v) for v in item] if isinstance(item, (list, tuple, np.ndarray)) else float(item)
            for item in condition_vector
        ]

    return [float(condition_vector)]

def get_material_id(row: dict, count: int, offset: int=0) -> str:
    """Get material ID from row data or generate one, appending a unique counter."""
    base_id = row.get("Material ID") or row.get("Formula") or "Generated"
    return f"{base_id}_{count + offset + 1}"


def _normalize_scoring_mode(mode: str | None) -> str:
    """Normalize scoring mode to 'none' or 'logp'."""
    if mode is None or str(mode).lower() in ("none", "null", ""):
        return "none"
    return str(mode).lower()


def resolve_generation_plan(scoring_mode: str | None, target_valid_cifs: int, max_return_attempts: int, num_return_sequences: int, total_samples: int) -> dict:
    """Work out how many candidates to generate per prompt and whether to validate them.

    Scoring implies validation, since an invalid CIF cannot be meaningfully ranked, and so does any
    non-zero `target_valid_cifs`. When neither applies, the target becomes the full
    `num_return_sequences * max_return_attempts` grid and everything generated is returned.

    Returns a dict with the normalized scoring mode, the `need_scores` and `check_validity` flags,
    the per-prompt target, and the total expected generations used for the progress bar.
    """
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







def build_output_df(data: list[dict], args: argparse.Namespace, df_prompts: pd.DataFrame) -> pd.DataFrame:
    """Build the final output dataframe."""
    df = pd.DataFrame(data)
    if args.input_parquet and 'True CIF' in df_prompts.columns:
        df_prompts['Material ID'] = df_prompts['Material ID'].astype(str)
        df['Material ID'] = df['Material ID'].astype(str)
        df = df.merge(df_prompts[['Material ID', 'True CIF']], on='Material ID', how='left')
    return df

def main() -> None:
    """Validate the required arguments and run the generation pipeline."""
    # Imported here rather than at module scope: workers.py imports this module's
    # helpers, so a top-level import either way would be circular.
    from _utils._generating.workers import generate_on_gpu, init_worker, progress_listener
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
    print(f"Screening profile: {args.screening_profile}")
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
                  initargs=(args.model_ckpt_dir, DEFAULT_TOKENIZER_DIR, args.activate_conditionality, base_seed, "checkpoint", None)) as pool:
        
        pool.apply_async(progress_listener, (queue, total_expected_generations))
        
        try:
            # Handle case with 0 GPUs
            if num_gpus == 0:
                results.append(pool.apply_async(
                    generate_on_gpu,
                    (0, df_prompts, generation_kwargs, queue, 0, total_samples,
                    args.activate_conditionality, 0, scoring_mode, target_valid_cifs, args.max_return_attempts, base_seed,
                    args.screening_profile)
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
                    # Do not skip a non-empty workload when the attempt count is zero.
                    local_attempts = max(1, attempts_per_gpu[gpu_id]) if base_attempts > 0 else 0
                    local_target = max(1, targets_per_gpu[gpu_id]) if base_target > 0 else 0
                    
                    if local_attempts == 0 and local_target == 0:
                        continue
                    
                    # Pass the full dataframe (1 row) to everyone, same index (0 to 1)
                    results.append(pool.apply_async(
                        generate_on_gpu,
                        (gpu_id, df_prompts, generation_kwargs, queue, 0, 1,
                         args.activate_conditionality, current_offset, scoring_mode, local_target, local_attempts, base_seed,
                         args.screening_profile)
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
                        args.activate_conditionality, global_offset, scoring_mode, target_valid_cifs, args.max_return_attempts, base_seed,
                        args.screening_profile)
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
    df_prompts: pd.DataFrame,
    generation_kwargs: dict,
    activate_conditionality: str | None,
    scoring_mode: str | None,
    target_valid_cifs: int,
    max_return_attempts: int,
    base_seed: int=1,
    worker_count: int | None=None,
    initargs_override: tuple | None=None,
    screening_profile: str="application",
) -> list[dict]:
    """Generate CIFs across worker processes and return the collected rows.

    Args:
        df_prompts: one row per prompt, with prompt text and any condition columns
        generation_kwargs: sampling args from build_generation_kwargs, passed to model.generate
        activate_conditionality: registry key selecting the model class, None for the base model
        scoring_mode: raw mode string, normalized internally
        target_valid_cifs: valid CIFs wanted per prompt (0 returns everything and needs scoring off)
        max_return_attempts: generation rounds per prompt before giving up
        base_seed: worker N seeds with base_seed + N so GPUs do not duplicate samples
        worker_count: GPU workers, clamped to visible devices (None uses all)
        initargs_override: replaces the defaults passed to init_worker, used to load from the Hub
                           and carry config_overrides such as skip_xrd_convert_model
        screening_profile: 'benchmark' or 'application', see the --screening_profile help in
                           _args.py

    Returns:
        list of row dicts with prompt metadata, generated CIF text, and the ranking score when
        scoring is on
    """
    # Imported here rather than at module scope: workers.py imports this module's
    # helpers, so a top-level import either way would be circular.
    from _utils._generating.workers import generate_on_gpu, init_worker, progress_listener
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
                 activate_conditionality, 0, normalized_scoring_mode, target_valid_cifs, max_return_attempts, base_seed,
                 screening_profile)
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
                     l_target, l_attempts, base_seed, screening_profile)
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
                     activate_conditionality, start, normalized_scoring_mode, target_valid_cifs, max_return_attempts, base_seed,
                     screening_profile)
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