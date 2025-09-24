"""
Argument parsing for CrystaLLMv2 training and generation scripts.

Handles configuration for Transformer-based crystalline structure generation
with support for conditional models (PKV, Prepend, Slider, Raw architectures).
"""

import argparse
import commentjson

def parse_args():
    """Parse command-line arguments, supporting JSON config files with comments."""

    parser = argparse.ArgumentParser(description="CrystaLLMv2 Training Script")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file with comments (commentjson).")

    # Not used currently
    # # Hyperparameter search
    # #######################
    # parser.add_argument("--hyperparameter_search", type=bool, default=False, help="Enable hyperparameter search for model optimization.")


    # Data Arguments
    #######################
    parser.add_argument("--dataset_HF", type=str, default="HF-databases/mp-db_test", help="Path to Hugging Face dataset containing crystalline structures and properties.")
    parser.add_argument("--pretrained_tokenizer_dir", type=str, default="HF-cif-tokenizer", help="Directory containing pretrained CIF tokenizer for crystal structure parsing.")
    parser.add_argument("--context_length", type=int, default=1024, help="Maximum sequence length for CIF tokenization and model input.")
    # Filters
    parser.add_argument("--remove_CIFs_above_context", type=bool, default=False, help="Filter out CIF entries that exceed the context length limit.")
    parser.add_argument("--remove_CIFs_with_unk", type=bool, default=False, help="Remove CIF entries containing unknown tokens not in the tokenizer vocabulary.")


    # Conditional Arguments
    #######################
    parser.add_argument("--condition_columns", type=str, default=None, help="Dataset columns to condition on (e.g., 'bandgap,density' for property-guided generation).")
    parser.add_argument("--n_prefix_tokens", type=int, default=None, help="Number of virtual prefix tokens for Prepend-GPT conditioning.")
    parser.add_argument("--n_hidden_cond", type=int, default=None, help="Hidden layer size for conditional projection layers in PKV/Prepend architectures.")
    parser.add_argument("--cond_dropout", type=float, default=None, help="Dropout rate applied to conditional embeddings during training.")
    parser.add_argument("--share_layers", type=bool, default=None, help="Whether to share key-value pairs across transformer layers in PKV-GPT.")
    parser.add_argument("--n_heads_sharing_slider", type=int, default=None, help="Number of attention heads that share conditioning weights in Slider-GPT.")
    parser.add_argument("--cond_lr", type=float, default=None, help="Learning rate for conditional parameters (separate from main model learning rate).") 
    parser.add_argument("--cond_wd", type=float, default=None, help="Weight decay for conditional parameters in Slider-GPT architecture.")
    

    # Model Arguments
    #######################
    # Model Depth
    parser.add_argument("--activate_conditionality", type=str, default=None, help="Activate conditionality: 'PKV' or 'Prepend' or 'Slider', or 'Raw', or 'None' supported.")
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps before parameter update.")

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
    parser.add_argument("--logging_steps", type=int, default=50, help="Frequency of logging training metrics.")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--report_to", type=str, default="none", help="Reporting service (e.g. 'wandb', 'tensorboard', 'none').")
    parser.add_argument("--wandb_project_folder", type=str, default="CrystaLLMv2", help="Project name for Weights & Biases logging.")
    parser.add_argument("--pretrained_model_dir", type=str, default=None, help="Directory containing pretrained model for fine-tuning.")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy during training.")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Checkpoint saving strategy.")
    parser.add_argument("--eval_steps", type=int, default=50, help="Number of steps between evaluations.")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum number of training steps.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of evaluations to wait before early stopping.")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.01, help="Minimum improvement required to reset early stopping patience.")
    
    # Utils
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible training.")
    parser.add_argument("--data_seed", type=int, default=1, help="Data shuffling seed for reproducible data ordering.")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model checkpoint at end of training.")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="Metric to use for best model selection.")
    parser.add_argument("--greater_is_better", action="store_true", help="Whether higher values are better for the best model metric.")
    parser.add_argument("--torch_compile", action="store_true", help="Enable PyTorch 2.0 compilation for faster training.")
    parser.add_argument("--fp16", action="store_true", help="Use 16-bit floating point precision to reduce memory usage.")
    
    # https://huggingface.co/docs/transformers/en/deepspeed
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed configuration file for distributed training.")


    # Evaluation arguments
    #######################
    parser.add_argument("--model_ckpt_dir", type=str, default="model_ckpts/cif-gpt2-small/checkpoint-400",
                        help="Path to the trained model checkpoint for CIF generation.")
    parser.add_argument("--do_sample", type=str, default="True", help="Enable sampling for generation (True/False/beam).")
    parser.add_argument("--top_k", type=int, default=15, help="Number of highest probability tokens to consider for top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Cumulative probability threshold for nucleus (top-p) sampling.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for controlling randomness in generation (higher = more random).")
    parser.add_argument("--gen_max_length", type=int, default=1024, help="Maximum length of generated CIF sequences.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of CIF sequences to generate per sample (adjust for GPU memory).")
    parser.add_argument("--input_parquet", type=str, default=None, help="Input parquet file containing test data for generation.")
    parser.add_argument("--output_parquet", type=str, default=None, help="Output parquet file to save generated CIF structures.")
    parser.add_argument("--num_repeats", type=int, default=1, help="Number of generation runs per sample (total = num_repeats * num_return_sequences).")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process from the input parquet file.")
    []

    # CodeCarbon arguments
    #######################
    parser.add_argument("--codecarbon", action="store_true", help="Enable CodeCarbon tracking for carbon emission monitoring during training.")
    parser.add_argument("--tracker_project", type=str, default="CrystaLLMv2", help="CodeCarbon project name for emission tracking.")


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