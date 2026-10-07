"""Train or finetune a CrystaLLM-pi model, conditionally or unconditionally.

Conditional runs require `condition_columns` and an `activate_conditionality` model family.
Unconditional runs use plain GPT-2. Under `torchrun`, rank 0 handles logging and checkpoint writes.

Usage:
    ```bash
    python _train.py --config _config_files/training/conditional/density-example/mpdb-density-finetune_example.jsonc
    ```
"""

import math
import os
import atexit

import torch
import torch.distributed as dist
import numpy as np
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from huggingface_hub import HfApi, login, constants as hf_constants

from _args import parse_args
from _dataloader import load_data
from _tokenizer import CustomCIFTokenizer, checkpoint_vocab_size
from _utils import (
    LossTrack_EarlyStop_Callback,
    TrainingArgsCallback,
    CIFFormattingTrainer,
    DualLRLogger,
    resolve_data_mode,
    ContextExtensionWarmupCallback,
    has_context_extension_wpe,
    tokenizer_ID_check, 
    start_codecarbon_tracker, 
    find_checkpoint_from_dir, 
    params_stats_check,
    load_pretrained_model,
    build_model, 
    setup_scheduler,
    load_api_keys,
    acquire_port,
)

API_KEY_PATH = "API_keys.jsonc" # Path to API keys file
VERBOSE = False

torch.set_printoptions(profile="full", linewidth=200)
np.set_printoptions(threshold=np.inf)

process_socket = None
process_port = None

def cleanup() -> None:
    """Clean up distributed processes if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()

cleanup()
atexit.register(cleanup)

# Enable TF32 and cudnn optimizations when CUDA is available
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

def main() -> None:
    """Parse the config and run the training job."""
    global process_socket, process_port

    # Setting up environment
    ## Parse first: argparse serves --help here, and a fresh clone has no API_keys.jsonc.
    args = parse_args()

    ## Keep Muon and AdamW learning rates proportional during sweeps.
    if args.muon_lr_factor is not None:
        args.muon_lr = args.learning_rate * args.muon_lr_factor
        print(f"muon_lr = {args.muon_lr} from learning_rate {args.learning_rate} x muon_lr_factor {args.muon_lr_factor}")

    ## Load API keys
    data = load_api_keys(API_KEY_PATH)
    hf_key_json = str(data['HF_key'])
    wandb_key = str(data['wandb_key'])

    ## Acquire and hold an unused port
    process_socket, process_port = acquire_port()

    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print()

    ## Setup wandb and HF login
    import wandb  # local: --help must work without the optional train extra installed
    if not hf_constants.HF_HUB_OFFLINE:  # login() raises on offline compute nodes
        login(token=hf_key_json)
    if os.environ.get("WANDB_MODE") not in ("offline", "disabled"):
        wandb.login(key=wandb_key)
    if args.wandb_project_folder and args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project_folder

    ## CodeCarbon runs on rank 0 only, to avoid duplicate emissions logs.
    track_carbon = args.codecarbon and int(os.environ.get("RANK", 0)) == 0
    if track_carbon:
        tracker = start_codecarbon_tracker(args)
        print("CodeCarbon tracker started")

    # Multi-GPU training uses DDP so Muon has full weight matrices on every GPU.
    ## Reject plain python with multiple GPUs to avoid DataParallel.
    n_ranks = int(os.environ.get("WORLD_SIZE", 1))
    if "LOCAL_RANK" not in os.environ and torch.cuda.device_count() > 1:
        raise RuntimeError("Several GPUs are visible without torchrun. Use torchrun --nproc_per_node=N or select one GPU with CUDA_VISIBLE_DEVICES.")
    print(f"Using {n_ranks} GPUs with DDP" if n_ranks > 1 else "Using single GPU")

    ## Global batches keep the batch size independent of GPU count.
    train_parts = n_ranks * args.gradient_accumulation_steps
    if args.train_batch_size % train_parts or args.eval_batch_size % n_ranks:
        raise ValueError(
            f"train_batch_size {args.train_batch_size} must divide by GPUs x gradient_accumulation_steps ({train_parts}), "
            f"and eval_batch_size {args.eval_batch_size} by GPUs ({n_ranks})"
        )
    args.per_device_train_batch_size = args.train_batch_size // train_parts
    args.per_device_eval_batch_size = args.eval_batch_size // n_ranks
    print(f"Global batch {args.train_batch_size} = {n_ranks} GPUs x {args.per_device_train_batch_size} per GPU x {args.gradient_accumulation_steps} accumulation steps")

    # Dataloading and tokenization
    ## HF_DATASETS_CACHE lets runs share a dataset cache.
    cache_dir = os.environ.get("HF_DATASETS_CACHE", os.path.join(args.output_dir, "..", ".cache"))
    print(f"Cache directory: {cache_dir}")
    dataset = load_dataset(args.dataset_HF, revision=args.dataset_revision, cache_dir=cache_dir)

    ## Sweep runs share a parent folder and cache. Separate output folders prevent resuming another run.
    if "WANDB_SWEEP_ID" in os.environ:
        args.output_dir = os.path.join(args.output_dir, os.environ["WANDB_SWEEP_ID"], os.environ["WANDB_RUN_ID"])
        print(f"Sweep run output_dir: {args.output_dir}")

    ## Record the dataset commit in each checkpoint (None for offline or local data).
    try:
        args.dataset_commit = HfApi().dataset_info(args.dataset_HF, revision=args.dataset_revision).sha
    except Exception:
        args.dataset_commit = None
    print(f"Dataset commit: {args.dataset_commit}")

    ## Older checkpoints keep their original vocabulary.
    print("Tokenizing dataset")
    if args.pretrained_model_dir and 'checkpoint' not in args.pretrained_model_dir:
        args.pretrained_model_dir = find_checkpoint_from_dir(args.pretrained_model_dir)
    max_vocab_size = checkpoint_vocab_size(args.pretrained_model_dir) if args.pretrained_model_dir else None
    tokenizer = CustomCIFTokenizer.from_pretrained(
        pretrained_dir=args.pretrained_tokenizer_dir,
        pad_token="<pad>",
        max_vocab_size=max_vocab_size,
    )
    print(f"Tokenizer vocabulary: {len(tokenizer)} tokens")

    ## Fetch data_collator and tokenized dataset
    data_mode = resolve_data_mode(args.activate_conditionality)
    if data_mode == "conditional":
        print("\n**CONDITIONALITY ACTIVATED**")
        print(f"Condition type: {args.activate_conditionality}")
        # Load data and data collator
        tokenized_dataset, data_collator = load_data(
            tokenizer=tokenizer,
            dataset=dataset,
            context_length=args.context_length,
            mode="conditional",
            condition_columns=args.condition_columns,
            remove_CIFs_above_context=args.remove_CIFs_above_context,
            remove_CIFs_with_unk=args.remove_CIFs_with_unk,
            show_token_stats=VERBOSE,
            validate_conditions=VERBOSE
        )
    else:
        print("\n**CONDITIONALITY DEACTIVATED**")
        # Load data and data collator
        tokenized_dataset, data_collator = load_data(
            tokenizer=tokenizer,
            dataset=dataset,
            context_length=args.context_length,
            mode="unconditional",
            remove_CIFs_above_context=args.remove_CIFs_above_context,
            remove_CIFs_with_unk=args.remove_CIFs_with_unk,
            show_token_stats=VERBOSE,
            validate_conditions=VERBOSE
        )

    if VERBOSE:
        # Check if the dataset and tokenizer dont have any mismatched IDs
        tokenizer_ID_check(args, tokenized_dataset, tokenizer)

    ## Convert epochs to steps after loading the data, for warmup and the learning-rate schedule.
    if args.num_train_epochs is not None:
        args.max_steps = math.ceil(args.num_train_epochs * len(tokenized_dataset["train"]) / args.train_batch_size)
        print(f"{args.num_train_epochs} epochs = {args.max_steps} steps at global batch {args.train_batch_size}")

    # Build or Load a model
    ## If conditional model chosen, we assume finetuning, and so we always want to eval on start
    eval_on_start = data_mode == "conditional" and args.eval_strategy != "no"
    ## Build base model (works for all)
    model = build_model(args, tokenizer)
    
    ## Load pretrained weights if specified
    if args.pretrained_model_dir:
        loaded_model = load_pretrained_model(args, tokenizer)
        if loaded_model is not None:
            model = loaded_model
            print("Loaded model from pretrained model directory successfully")
        else:
            raise ValueError("Failed to load model from pretrained model directory even though the path was provided")
    else:
        print("Successfully built model from scratch")
    
    ## Print parameter details for the model 
    ## eg how many trainable, total parameters, how many dedicated to conditioning...
    if VERBOSE:
        params_stats_check(model)
    

    # Training setup and launch
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # W&B uses the last output_dir folder as the run name.
        run_name=os.path.basename(os.path.normpath(args.output_dir)),
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=args.lr_scheduler_kwargs,
        warmup_steps=args.warmup_steps if args.warmup_steps is not None else int(args.warmup_ratio * args.max_steps),
        max_grad_norm=args.grad_clip,
        seed=args.seed,
        data_seed=args.data_seed,
        load_best_model_at_end=args.load_best_model_at_end,
        torch_compile=args.torch_compile,
        save_strategy=args.save_strategy,
        max_steps=args.max_steps,
        remove_unused_columns=False,
        eval_on_start=eval_on_start,
        gradient_checkpointing=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=True,
        ddp_find_unused_parameters=False,
    )

    # If finetune with conditioning, setup the dual LR optimizer
    optimizer, lr_scheduler = setup_scheduler(args, model)

    # Capture arguments here so runtime adjustments are recorded in each checkpoint.
    callbacks = [DualLRLogger(), TrainingArgsCallback(vars(args).copy())]
    if has_context_extension_wpe(model) and args.context_extension_warmup_steps > 0:
        callbacks.append(
            ContextExtensionWarmupCallback(
                context_extension_warmup_steps=args.context_extension_warmup_steps
            )
        )

    # Only add early stopping if evaluation is enabled
    if args.eval_strategy != "no" and args.early_stopping_patience:
        callbacks.append(
            LossTrack_EarlyStop_Callback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold
            )
        )

    trainer = CIFFormattingTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        optimizers=(optimizer, lr_scheduler)
    )

    # Resume from the newest checkpoint if a previous job was interrupted.
    last_checkpoint = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last_checkpoint:
        print(f"Resuming from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Final evaluation
    if args.eval_strategy != "no":
        eval_results = trainer.evaluate()
        print("Evaluation Results:", eval_results)
    
    print("Saving model to:", args.output_dir)

    # Cleanup
    if track_carbon:
        tracker.stop()
        print("CodeCarbon tracker stopped")
    if process_socket:
        process_socket.close()
        print(f"Process socket on port {process_port} closed")

if __name__ == "__main__":
    main()
