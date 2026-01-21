"""
Main training script for CrystaLLM_pi conditional or unconditional crystal structure generation.
"""

import os
import atexit

import torch
import torch.distributed as dist
import numpy as np
import wandb
from transformers import TrainingArguments
from datasets import load_dataset
from huggingface_hub import login

from _args import parse_args
from _dataloader import load_data
from _tokenizer import CustomCIFTokenizer
from _utils import (
    LossTrack_EarlyStop_Callback, 
    CIFFormattingTrainer, 
    DualLRLogger,
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

def cleanup():
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

def main():
    """Main training function for CrystaLLM_pi."""
    global process_socket, process_port

    # Setting up environment
    ## Load API keys
    data = load_api_keys(API_KEY_PATH)
    hf_key_json = str(data['HF_key'])
    wandb_key = str(data['wandb_key'])

    ## Acquire and hold an unused port
    process_socket, process_port = acquire_port()

    ## Parse arguments
    args = parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print()

    ## Setup wandb and HF login
    login(token=hf_key_json)
    wandb.login(key=wandb_key)
    if args.wandb_project_folder and args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project_folder

    ## CodeCarbon tracker
    if args.codecarbon:
        tracker = start_codecarbon_tracker(args)
        print("CodeCarbon tracker started")

    # ## Handle multi-GPU and deepspeed settings
    # # check if the file exists
    # if args.deepspeed_config is None:
    #     if torch.cuda.device_count() > 1:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #         print(f"You have {torch.cuda.device_count()} GPUs")
    #         print("No deepspeed config provided. Setting to use only 1 GPU")
    # elif not os.path.isfile(args.deepspeed_config):
    #     raise ValueError(f"Deepspeed config file {args.deepspeed_config} not found")
    # else:
    #     if torch.cuda.device_count() == 1:
    #         args.deepspeed_config = None
    #         print("Deepspeed config provided, but only 1 GPU detected. Disabling deepspeed.")

    if args.deepspeed_config is None:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"Using {n_gpus} GPUs with DDP (no DeepSpeed)")
            # DDP will be handled by HuggingFace Trainer automatically
        else:
            print("Using single GPU")
    elif not os.path.isfile(args.deepspeed_config):
        raise ValueError(f"Deepspeed config file {args.deepspeed_config} not found")
    else:
        if torch.cuda.device_count() == 1:
            args.deepspeed_config = None
            print("Deepspeed config provided, but only 1 GPU detected. Disabling deepspeed.")

    ## If raw conditionality is activated, we need to adjust the context length
    if args.activate_conditionality == "Raw":
        # For fair comparison with PKV/Prepend/Slider conditioning
        additional_tokens = int(5 * args.n_prefix_tokens + args.n_prefix_tokens + 2)
        args.context_length = args.context_length + additional_tokens


    # Dataloading and tokenization
    ## Load dataset with the specified cache directory
    cache_dir = os.path.join(args.output_dir, "..", ".cache")
    print(f"Cache directory: {cache_dir}")
    dataset = load_dataset(args.dataset_HF, cache_dir=cache_dir)

    ## Load custom tokenizer
    print("Tokenizing dataset")
    tokenizer = CustomCIFTokenizer.from_pretrained(
        pretrained_dir=args.pretrained_tokenizer_dir,
        pad_token="<pad>"
    )

    ## Fetch data_collator and tokenized dataset
    if args.activate_conditionality in ["PKV", "Prepend", "Slider"]:
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
    elif args.activate_conditionality == "Raw":
        print("\n**RAW CONDITIONALITY ACTIVATED**")
        print(f"Condition type: {args.activate_conditionality}")
        # Load data and data collator
        tokenized_dataset, data_collator = load_data(
            tokenizer=tokenizer,
            dataset=dataset,
            context_length=args.context_length,
            mode="raw",
            condition_columns=args.condition_columns,
            remove_CIFs_above_context=args.remove_CIFs_above_context,
            remove_CIFs_with_unk=args.remove_CIFs_with_unk,
            show_token_stats=VERBOSE,
            validate_conditions=VERBOSE
        )
        print(f"Context length for raw conditionality (to account for new condition tokens): {args.context_length}")
    elif args.activate_conditionality == "None" or args.activate_conditionality is None:
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


    # 3. Build or Load a model
    ## If conditional model chosen, we assume finetuning, and so we always want to eval on start
    eval_on_start = args.activate_conditionality in ["PKV", "Prepend", "Slider", "Raw"] and args.eval_strategy != "no"
    ## Build base model (works for all)
    model = build_model(args, tokenizer)
    
    ## Load pretrained weights if specified
    if args.pretrained_model_dir:
        if 'checkpoint' not in args.pretrained_model_dir:
            args.pretrained_model_dir = find_checkpoint_from_dir(args.pretrained_model_dir)
        
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
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
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
        ddp_find_unused_parameters=False,
    )

    # Integrate deepspeed config if provided
    if args.deepspeed_config:
        training_args.deepspeed = args.deepspeed_config

    # If finetune with conditioning, setup the dual LR optimizer
    optimizer, lr_scheduler = setup_scheduler(args, model)

    # Setup callbacks based on evaluation strategy
    callbacks = [DualLRLogger()]
    
    # Only add early stopping if evaluation is enabled
    if args.eval_strategy != "no" and hasattr(args, 'early_stopping_patience'):
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
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        optimizers=(optimizer, lr_scheduler)
    )

    # Start training
    trainer.train()

    # Final evaluation
    if args.eval_strategy != "no":
        eval_results = trainer.evaluate()
        print("Evaluation Results:", eval_results)
    
    print("Saving model to:", args.output_dir)

    # Cleanup
    if args.codecarbon:
        tracker.stop()
        print("CodeCarbon tracker stopped")
    if process_socket:
        process_socket.close()
        print(f"Process socket on port {process_port} closed")

if __name__ == "__main__":
    main()
