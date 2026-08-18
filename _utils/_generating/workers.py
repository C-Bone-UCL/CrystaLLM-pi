"""Worker-process side of multi-GPU generation.

Everything here runs inside a `multiprocessing` pool worker. `init_worker` loads the model and
tokenizer into module globals that `generate_on_gpu` then reads, which is why the two live in the
same module: a pool cannot pass a loaded model through the task queue, so the handoff has to go
through module state.

The driver side (`run_generation_pool`, the CLI) stays in `generate_cifs.py`.
"""

import os

import numpy as np
import torch
from tqdm import tqdm

from _utils._generating.generate_cifs import (
    check_cif,
    get_material_id,
    get_model_class,
    init_tokenizer,
    parse_condition_vector,
    _normalize_scoring_mode,
)
from _utils._generating.scoring_methods import score_outputs_logp

model = None
tokenizer = None


def setup_device(gpu_id: int) -> torch.device:
    """Setup device and return appropriate torch device."""
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _load_worker_model(model_class: type, model_source_path: str, model_source: str, dtype: torch.dtype, config_overrides: dict | None=None) -> torch.nn.Module:
    """Load worker model from local checkpoint or HuggingFace source."""
    extra_kwargs = {"trust_remote_code": True} if model_source == "hf" else {}
    if config_overrides:
        # Registry-supplied config overrides (e.g. skip_xrd_convert_model) go straight into
        # from_pretrained so the model's own config class applies them. Do NOT pre-fetch an
        # AutoConfig here, it resolves plain GPT2Config and bypasses conditional defaults.
        extra_kwargs.update(config_overrides)

    try:
        return model_class.from_pretrained(
            model_source_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            **extra_kwargs,
        ).eval()
    except Exception:
        return model_class.from_pretrained(
            model_source_path,
            torch_dtype=dtype,
            **extra_kwargs,
        ).eval()


def init_worker(
    model_ckpt_dir: str,
    pretrained_tokenizer_dir: str,
    activate_conditionality: str | None,
    base_seed: int=1,
    model_source: str="checkpoint",
    config_overrides: dict | None=None,
) -> None:
    """Load the model and tokenizer once per worker process.

    Populates the module-level `model` and `tokenizer` globals that `generate_on_gpu` reads, since a `multiprocessing` pool cannot pass a loaded model through the task queue. Picks bfloat16 where the GPU supports it, float16 on older GPUs, float32 on CPU. Each worker seeds with `base_seed + LOCAL_RANK`, which keeps runs reproducible while stopping every GPU from drawing the same samples.
    """
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

    model = _load_worker_model(model_class, model_ckpt_dir, model_source, dtype, config_overrides)
    model.resize_token_embeddings(len(tokenizer))
    
    # Deterministic seeding per worker
    gpu_id = int(os.environ.get("LOCAL_RANK", 0))
    worker_seed = base_seed + gpu_id
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)


def generate_on_gpu(
    gpu_id: int,
    prompts: list,
    generation_kwargs: dict,
    queue: object,
    start_idx: int,
    end_idx: int,
    activate_conditionality: str | None,
    global_offset: int=0,
    scoring_mode: str="none",
    target_valid_cifs: int=0,
    max_return_attempts: int=2,
    base_seed: int=1,
) -> list[dict]:
    """Generate CIFs for one slice of the prompt list inside a worker process.

    Runs on the device `gpu_id` selects, reading the model and tokenizer that `init_worker` placed in module globals. Handles `prompts[start_idx:end_idx]`, retrying up to `max_return_attempts` rounds until it has `target_valid_cifs` valid structures per prompt, and reports progress through `queue`. `global_offset` keeps material IDs unique when several workers write into one run.
    """
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

    # XRD fit scoring ranks after generation in _load_and_generate, so workers hand
    # back every valid candidate. Truncating to target_valid_cifs here would rank on
    # first-come order and starve the XRD comparison of its candidate pool.
    keep_all_valid = scoring_mode == "pearson"
    
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
                if activate_conditionality in ["PKV", "Slider", "Prefix", "PrefixXRD", "Residual"]:
                    # Parse first: a nested (1000, 2) profile becomes a (1, 1000, 2) tensor,
                    # a flat PKV list stays (1, n), unchanged legacy behavior.
                    condition_tensor = None
                    values = parse_condition_vector(row.get("condition_vector"))
                    if values is not None:
                        condition_tensor = torch.tensor([values], device=device, dtype=model.dtype)
                    
                    with torch.inference_mode():
                        outputs = model.generate(
                            input_ids=input_ids,
                            condition_values=condition_tensor,
                            return_dict_in_generate=True,
                            output_scores=need_scores,
                            **generation_kwargs,
                        )
                else:
                    # Handle unconditional generation
                    with torch.inference_mode():
                        outputs = model.generate(
                            input_ids=input_ids,
                            return_dict_in_generate=True,
                            output_scores=need_scores,
                            **generation_kwargs,
                        )

                batch_scores = None
                if need_scores:
                    batch_scores = score_outputs_logp(
                        model=model,
                        scores=outputs.scores,
                        full_sequences=outputs.sequences,
                        input_length=input_ids.shape[1],
                        eos_token_id=tokenizer.eos_token_id,
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
                                score = batch_scores[seq_idx]
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
                    
                    # Validation-only mode can stop mid-batch. LOGP and XRD fit scoring
                    # must keep the full batch for ranking.
                    if len(valid_cifs) >= target_generations and not need_scores and not keep_all_valid:
                        break
                        
            except Exception as e:
                print(f"Error during generation/validation for prompt index {idx} on GPU {gpu_id}, attempt {generation_attempts}")
                print(f"Error details: {e}")
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

                best_cifs = ranked_cifs if keep_all_valid else ranked_cifs[:target_generations]
                for rank, cif_data in enumerate(best_cifs, 1):
                    cif_data["rank"] = rank
                results.extend(best_cifs)
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return results


def progress_listener(queue: object, total: int) -> None:
    pbar = tqdm(total=total, desc="Generating CIFs...")
    while True:
        message = queue.get()
        if message == "kill":
            break
        pbar.update(message)
    pbar.close()
