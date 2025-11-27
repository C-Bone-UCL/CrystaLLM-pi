"""
Argument parsing for CrystaLLM_pi training and generation scripts.

Handles configuration for Transformer-based crystalline structure generation
with support for conditional models (PKV, Prepend, Slider, Raw architectures).
"""

import argparse
import commentjson

def parse_args():
    """Parse command-line arguments, supporting JSON config files with comments."""

    parser = argparse.ArgumentParser(description="CrystaLLM_pi Training Script")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file with comments (commentjson).")


    # Data Arguments
    #######################
    parser.add_argument("--dataset_HF", type=str, default="HF-databases/mp-db_test", help="Path to Hugging Face dataset containing crystalline structures and optionally properties. Check README for expected format.")
    parser.add_argument("--pretrained_tokenizer_dir", type=str, default="HF-cif-tokenizer", help="Directory containing pretrained CIF tokenizer for crystal structure parsing.")
    parser.add_argument("--context_length", type=int, default=1024, help="Maximum sequence length for CIF token sequences (default: 1024 tokens which is about ~20 atoms per cell).")
    # Filters
    parser.add_argument("--remove_CIFs_above_context", action="store_true", help="Filter out CIF entries that exceed the context length limit.")
    parser.add_argument("--remove_CIFs_with_unk", action="store_true", help="Remove CIF entries containing unknown tokens not in the tokenizer vocabulary.")


    # Conditional Arguments
    #######################
    parser.add_argument("--condition_columns", type=str, default=None, help="Comma-separated dataset column names to condition on (e.g., 'bandgap,density'). Must match exact column names in dataset. Values should be pre-normalized.")
    parser.add_argument("--n_prefix_tokens", type=int, default=None, help="Number of learned prefix tokens or ghost tokens prefixed to input sequence (Prepend-GPT and PKV-GPT only).")
    parser.add_argument("--n_hidden_cond", type=int, default=None, help="Hidden dimension for property embedding projections (PKV and Slider).")
    parser.add_argument("--cond_dropout", type=float, default=None, help="Dropout rate applied to conditional embeddings during training (PKV an Slider)).")
    parser.add_argument("--share_layers", type=bool, default=None, help="Share conditional key-value projections across all layers (PKV-GPT only). Reduces parameters but may limit expressivity.")
    parser.add_argument("--n_heads_sharing_slider", type=int, default=None, help="Number of attention heads that use shared conditioning weights (Slider-GPT only). Must be ≤ n_head.")
    parser.add_argument("--cond_lr", type=float, default=None, help="Learning rate for conditional parameters (all models except for Raw). Separate from main model learning rate.") 
    parser.add_argument("--cond_wd", type=float, default=None, help="Weight decay for conditional parameters (all models except for Raw).")

    # Model Arguments
    #######################
    # Model Depth
    parser.add_argument("--activate_conditionality", type=str, default=None, help="Select conditioning architecture: 'PKV', 'Prepend', 'Slider', 'Raw', or None for unconditional model. Default None loads base unconditional model.")
    # parser.add_argument("--n_positions", type=int, default=1024, help="Model context size")
    parser.add_argument("--n_embd", type=int, default=256, help="Transformer embedding dimension size.")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of Transformer layers in the model.")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads per Transformer layer.")
    # Dropout
    parser.add_argument("--residual_dropout", type=float, default=0.1, help="Dropout probability applied to residual connections.")
    parser.add_argument("--embedding_dropout", type=float, default=0.1, help="Dropout probability applied to token embeddings.")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Dropout probability applied within attention mechanism.")


    # Trainer Arguments
    #######################
    
    # Batching
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size for crystalline structure generation.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size for model validation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps before parameter update. Effective batch size = train_batch_size * gradient_accumulation_steps.")

    # Learning Rate and Optimizer
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Base learning rate for transformer training.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type (linear, cosine, constant).")
    parser.add_argument("--lr_scheduler_kwargs", type=dict, default={}, help="Additional keyword arguments for learning rate scheduler.")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Number of warmup steps, if specified overrides warmup_ratio.")
    parser.add_argument("--warmup_ratio", type=float, default=0.02, help="Warmup ratio as percentage of total training steps.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam optimizer beta1 parameter for momentum.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam optimizer beta2 parameter for squared gradient averaging.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold to prevent exploding gradients.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay regularization for transformer parameters.")
    
    # Logging
    parser.add_argument("--output_dir", type=str, default="model_ckpts/test-model", help="Output directory for model checkpoints and logs.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Frequency of logging training metrics")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--report_to", type=str, default="none", help="Reporting service (e.g. 'wandb', 'tensorboard', 'none').")
    parser.add_argument("--wandb_project_folder", type=str, default="CrystaLLM_pi", help="Project name for Weights & Biases logging.")
    parser.add_argument("--pretrained_model_dir", type=str, default=None, help="Directory containing pretrained model checkpoint. Use for loading base unconditional model or another pass of conditional training.")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy during training.")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Checkpoint saving strategy.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Number of steps between evaluations & save points.")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum number of training steps.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of evaluation steps without improvement before stopping training.")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.01, help="Minimum improvement required to reset early stopping patience.")
    
    # Utils
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible training.")
    parser.add_argument("--data_seed", type=int, default=1, help="Data shuffling seed for reproducible data ordering.")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model checkpoint at end of training. Required for early stopping and best model selection.")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="Metric to use for best model selection.")
    parser.add_argument("--greater_is_better", action="store_true", help="Whether higher values are better for the best model metric.")
    parser.add_argument("--torch_compile", action="store_true", help="Enable PyTorch 2.0+ compilation for faster training (requires PyTorch ≥2.0).")
    parser.add_argument("--fp16", action="store_true", help="Use 16-bit floating point precision to reduce memory usage.")
    
    # https://huggingface.co/docs/transformers/en/deepspeed
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed configuration file for distributed training. Required for multi-GPU training.")


    # Evaluation arguments
    #######################
    parser.add_argument("--model_ckpt_dir", type=str, default="model_ckpts/cif-gpt2-small/checkpoint-400", help="Path to the trained model checkpoint for CIF generation.")
    parser.add_argument("--do_sample", type=str, default="True", help="Sampling mode: 'True' for stochastic sampling, 'False' for greedy decoding, 'beam' for beam search.")
    parser.add_argument("--top_k", type=int, default=15, help="Number of highest probability tokens to consider for top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Cumulative probability threshold for nucleus (top-p) sampling.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for controlling randomness in generation (higher = more random).")
    parser.add_argument("--gen_max_length", type=int, default=1024, help="Maximum length of generated CIF sequences (max can be set to context length of your model).")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of CIF sequences to generate per sample (adjust for GPU memory).")
    parser.add_argument("--input_parquet", type=str, default=None, help="Input parquet file containing Prompts for generation, and also 'condition_vector' (normalised accoridng to pre-processing) if conditioning output.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of prompts to process from the input parquet file.")
    parser.add_argument("--output_parquet", type=str, default=None, help="Output parquet file to save generated CIF structures.")
    parser.add_argument("--max_return_attempts", type=int, default=1, help="Number of generation batches per prompt. For 'logp' mode: stops when target_valid_cifs reached or max_return_attempts hit. For 'None' mode: runs this many batches of num_return_sequences.")
    parser.add_argument("--scoring_mode", type=str, default="None", help="Scoring mode for ranking generated structures: 'logp' for perplexity-based ranking (validates and ranks CIFs), or 'None' for no validation/scoring.")

    # If scoring_mode is 'logp', the model will compute log-perplexity scores for 'target_valid_cifs' amount of valid generated CIFs to rank them.
    parser.add_argument("--target_valid_cifs", type=int, default=20, help="Target number of valid CIFs per prompt (only used with scoring_mode='logp'). Generation continues until this target or max_return_attempts is reached.")


    # CodeCarbon arguments
    #######################
    parser.add_argument("--codecarbon", action="store_true", help="Enable CodeCarbon tracking for carbon emission monitoring during training (requires codecarbon package).")
    parser.add_argument("--tracker_project", type=str, default="CrystaLLM_pi", help="CodeCarbon project name for emission tracking.")


    # Parse arguments
    args = parser.parse_args()
    # Load JSON config if provided
    if args.config:
        with open(args.config, "r") as f:
            config_data = commentjson.load(f)
        for key, value in config_data.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args