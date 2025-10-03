
"""
Hyperparameter search script for conditional crystal structure generation fine-tuning.
Optimizes parameters using Optuna for improved conditional generation performance.
"""

import os
import json
import wandb
import atexit
import torch.distributed as dist
import torch
import numpy as np
from huggingface_hub import login
from datasets import load_dataset
from transformers import TrainingArguments
import optuna
import copy

from _args import parse_args
from _dataloader import load_data
from _tokenizer import CustomCIFTokenizer
from _utils import (
    CIFFormattingTrainer, 
    LossTrack_EarlyStop_Callback,
    DualLRLogger,
    start_codecarbon_tracker, 
    find_checkpoint_from_dir, 
    params_stats_check, 
    load_pretrained_model, 
    build_model, 
    setup_scheduler,
    load_api_keys,
)

API_KEY_PATH = "API_keys.jsonc"
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

def optuna_hp_space(trial):
    """Define hyperparameter search space focused on conditioning parameters for small dataset."""
    hp = {}
    
    # Architecture - More conservative for small dataset
    # hp["n_prefix_tokens"] = trial.suggest_categorical("n_prefix_tokens", [1, 2, 3, 4, 5]) 
    # hp["n_hidden_cond"] = trial.suggest_categorical("n_hidden_cond", [512, 768, 1024])
    factor = trial.suggest_float("cond_lr_factor", 10.0, 100.0, log=True)     
    hp["cond_dropout"] = trial.suggest_float("cond_dropout", 0.0, 0.2, step=0.02)
    
    base_lr = trial.suggest_float("learning_rate", 5e-7, 5e-5, log=True)
    hp["learning_rate"] = base_lr
    hp["cond_lr"] = base_lr * factor
    
    hp["weight_decay"] = trial.suggest_float("weight_decay", 0.01, 0.2, step=0.01)
    hp["cond_wd"] = trial.suggest_float("cond_wd", 0.01, 0.3, step=0.01)
    
    hp["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.01, 0.1, step=0.01)
    
    return hp

def model_init(trial=None):
    """Initialize model with hyperparameters from the trial."""
    args = parse_args()
    tokenizer = CustomCIFTokenizer.from_pretrained(
        pretrained_dir=args.pretrained_tokenizer_dir,
        pad_token="<pad>"
    )

    new_args = copy.deepcopy(args)
    
    # Apply hyperparameters from trial if provided
    if trial is not None:
        hp = optuna_hp_space(trial)
        for key, value in hp.items():
            setattr(new_args, key, value)

    # Build model using unified function
    model = build_model(new_args, tokenizer)
    
    # Load pretrained weights if specified
    if new_args.pretrained_model_dir:
        if 'checkpoint' not in new_args.pretrained_model_dir:
            new_args.pretrained_model_dir = find_checkpoint_from_dir(new_args.pretrained_model_dir)
        
        loaded_model = load_pretrained_model(new_args, tokenizer)
        if loaded_model is not None:
            model = loaded_model

    return model

class OOMPruningTrainer(CIFFormattingTrainer):
    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        except RuntimeError as e:
            # Check if this is an OOM error
            if "out of memory" in str(e).lower():
                print("OOM error detected. Pruning trial...")
                raise optuna.exceptions.TrialPruned("Pruning trial due to OOM error.")
            else:
                # If it's not an OOM error go to next trial but it should not be pruned, just failed
                print("Non-OOM error detected. Proceeding to next trial...")
                raise optuna.exceptions.TrialPruned("Non-OOM error detected. Proceeding to next trial...")

def main():
    """Main hyperparameter search function for fine-tuning optimization."""
    global process_socket, process_port
    
    # Load API keys
    data = load_api_keys(API_KEY_PATH)
    hf_key_json = str(data['HF_key'])
    wandb_key = str(data['wandb_key'])
    
    # Parse arguments
    args = parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print()
    
    # Setup wandb and HF login
    login(token=hf_key_json)
    wandb.login(key=wandb_key)
    os.environ["WANDB_PROJECT"] = args.wandb_project_folder

    # CodeCarbon tracker
    if args.codecarbon:
        tracker = start_codecarbon_tracker(args)
        print("CodeCarbon tracker started")

    # Force single GPU for hyperparameter search
    if torch.cuda.device_count() > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print(f"Multiple GPUs detected. Using only GPU 1 for hyperparameter search")

    # Load dataset
    cache_dir = os.path.join(args.output_dir, "..", ".cache")
    print(f"Cache directory: {cache_dir}")
    dataset = load_dataset(args.dataset_HF, cache_dir=cache_dir)

    # Load tokenizer
    print("Tokenizing dataset")
    tokenizer = CustomCIFTokenizer.from_pretrained(
        pretrained_dir=args.pretrained_tokenizer_dir,
        pad_token="<pad>"
    )

    # Load data for conditioning
    if args.activate_conditionality == "PKV" or args.activate_conditionality == "Slider":

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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy=args.eval_strategy,
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
        eval_steps=args.eval_steps,
        max_steps=args.max_steps,
        remove_unused_columns=False,
        eval_on_start=True,
        gradient_checkpointing=False,
    )

    # Hyperparameter search trainer
    trainer = OOMPruningTrainer(
        model=None,
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            LossTrack_EarlyStop_Callback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold
            ),
            DualLRLogger()
        ],
    )

    def compute_objective(metrics):
        return metrics["eval_loss"]
    
    # Run hyperparameter search
    best_run = trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        n_trials=50,
        compute_objective=compute_objective,
        hp_space=optuna_hp_space,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2000
            ),
        study_name=f"{args.wandb_project_folder}",
        storage=f"sqlite:///optuna_hpsearch_{args.wandb_project_folder}.db",
        load_if_exists=True,
        n_jobs=1,
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
    )

    print("Best run hyperparameters:", best_run.hyperparameters)
    print("Best run objective:", best_run.objective)

    # Save results
    with open(os.path.join(args.output_dir, "best_run_hyperparameters.json"), "w") as f:
        json.dump(best_run.hyperparameters, f, indent=4)

    # Cleanup
    if args.codecarbon:
        tracker.stop()
        print("CodeCarbon tracker stopped")

if __name__ == "__main__":
    main()
