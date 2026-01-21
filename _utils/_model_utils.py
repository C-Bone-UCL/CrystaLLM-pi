"""
Model utilities for loading and building CrystaLLM conditional and standard GPT models.
"""

import math
import ast
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from _models import (
    PKVGPT, 
    PrependGPT,
    SliderGPT,
    PKVGPT2Config,
    PrependGPT2Config,
    SliderGPT2Config,
)

# Registry mapping conditionality types to (config_class, model_class)
MODEL_REGISTRY = {
    "PKV": (PKVGPT2Config, PKVGPT),
    "Prepend": (PrependGPT2Config, PrependGPT),
    "Slider": (SliderGPT2Config, SliderGPT),
    "Raw": (GPT2Config, GPT2LMHeadModel),
    None: (GPT2Config, GPT2LMHeadModel),
}


def _parse_condition_columns(args):
    """Extract condition vector size from args.condition_columns."""
    try:
        condition_list = ast.literal_eval(str(args.condition_columns))
        return len(condition_list)
    except (ValueError, SyntaxError):
        raise ValueError(f"Invalid condition_columns format: {args.condition_columns}")


def _load_with_sdpa_fallback(model_class, pretrained_path, config, **kwargs):
    """Load a model with SDPA attention, falling back to default if unavailable."""
    try:
        return model_class.from_pretrained(
            pretrained_path, config=config, attn_implementation="sdpa", **kwargs
        )
    except Exception:
        return model_class.from_pretrained(pretrained_path, config=config, **kwargs)


def _get_n_positions(args, conditionality):
    """Calculate target n_positions based on conditionality type."""
    if conditionality in ("PKV", "Prepend"):
        return args.context_length + args.n_prefix_tokens
    return args.context_length


def _get_base_config(args, tokenizer):
    """Build base config dict shared across all model types."""
    return dict(
        vocab_size=len(tokenizer),
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        resid_pdrop=args.residual_dropout,
        embd_pdrop=args.embedding_dropout,
        attn_pdrop=args.attention_dropout,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

def resize_positional_embeddings(model, new_n_positions):
    """Extend positional embeddings using sinusoidal patterns for conditional models."""
    old_n_positions = model.config.n_positions
    if new_n_positions == old_n_positions:
        return model
        
    old_wpe = model.transformer.wpe.weight.data.clone()
    new_wpe = torch.nn.Parameter(torch.zeros(new_n_positions, old_wpe.size(1), device=old_wpe.device))
    new_wpe.data[:old_n_positions, :] = old_wpe
    
    # Generate position-aware embeddings for new positions using sinusoidal functions
    # This gives the model distinct positional understanding for prefix tokens
    print("\nResizing positional embeddings with sinusoidal function for position awareness")
    dim = old_wpe.size(1)
    for pos in range(old_n_positions, new_n_positions):
        for i in range(0, dim, 2):
            new_wpe.data[pos, i] = math.sin(pos / (10000 ** (i / dim)))
            if i+1 < dim:
                new_wpe.data[pos, i+1] = math.cos(pos / (10000 ** (i / dim)))
    
    model.transformer.wpe = torch.nn.Embedding(new_n_positions, old_wpe.size(1))
    model.transformer.wpe.weight = new_wpe
    model.config.n_positions = new_n_positions
    print(f"Resized positional embeddings from {old_n_positions} to {new_n_positions}")
    return model


def load_pretrained_model(args, tokenizer):
    """Load pretrained models with auto-detection of conditional architectures."""
    print(f"Loading model weights from {args.pretrained_model_dir}")
    
    vocab_size = len(tokenizer)
    conditionality = getattr(args, 'activate_conditionality', None)
    target_n_positions = _get_n_positions(args, conditionality)
    
    config_class, model_class = MODEL_REGISTRY.get(conditionality, MODEL_REGISTRY[None])
    
    # Build config based on conditionality type
    if conditionality == "PKV":
        config = config_class.from_pretrained(
            args.pretrained_model_dir,
            n_input_vector=_parse_condition_columns(args),
            n_prefix_tokens=args.n_prefix_tokens,
            n_hidden_cond=args.n_hidden_cond,
            share_layers=getattr(args, 'share_layers', False),
        )
    elif conditionality == "Prepend":
        config = config_class.from_pretrained(
            args.pretrained_model_dir,
            n_input_vector=_parse_condition_columns(args),
            n_prefix_tokens=args.n_prefix_tokens,
            n_hidden_cond=args.n_hidden_cond,
            share_layers=getattr(args, 'share_layers', False),
        )
    elif conditionality == "Slider":
        config = config_class.from_pretrained(
            args.pretrained_model_dir,
            n_positions=target_n_positions,
            vocab_size=vocab_size,
            slider_on=True,
            slider_n_variables=args.n_prefix_tokens,
            slider_n_hidden=args.n_hidden_cond,
            slider_n_heads_sharing_slider=args.n_heads_sharing_slider,
            slider_dropout=args.cond_dropout,
        )
    elif conditionality == "Raw":
        config = config_class.from_pretrained(
            args.pretrained_model_dir,
            n_positions=target_n_positions,
            vocab_size=vocab_size,
        )
    else:
        config = config_class.from_pretrained(args.pretrained_model_dir)
        config.n_positions = args.context_length
    
    model = _load_with_sdpa_fallback(model_class, args.pretrained_model_dir, config, ignore_mismatched_sizes=True)
    print(f"Loaded as {conditionality or 'GPT2'} with n_positions={target_n_positions}")

    model.resize_token_embeddings(vocab_size)
    if model.config.n_positions != target_n_positions:
        model = resize_positional_embeddings(model, target_n_positions)

    return model

def build_model(args, tokenizer):
    """Build fresh models with auto-detection of conditional architectures."""
    vocab_size = len(tokenizer)
    conditionality = getattr(args, 'activate_conditionality', None)
    target_n_positions = _get_n_positions(args, conditionality)
    base_config = _get_base_config(args, tokenizer)
    
    config_class, model_class = MODEL_REGISTRY.get(conditionality, MODEL_REGISTRY[None])
    
    # Build config based on conditionality type
    if conditionality == "PKV":
        config = config_class(
            n_input_vector=_parse_condition_columns(args),
            n_prefix_tokens=args.n_prefix_tokens,
            n_hidden_cond=args.n_hidden_cond,
            share_layers=args.share_layers,
            dropout=args.cond_dropout,
            n_positions=target_n_positions,
            **base_config
        )
    elif conditionality == "Prepend":
        config = config_class(
            n_input_vector=_parse_condition_columns(args),
            n_prefix_tokens=args.n_prefix_tokens,
            n_hidden_cond=args.n_hidden_cond,
            dropout=args.cond_dropout,
            n_positions=target_n_positions,
            **base_config
        )
    elif conditionality == "Slider":
        config = config_class(
            slider_on=True,
            slider_n_variables=args.n_prefix_tokens,
            slider_n_hidden=args.n_hidden_cond,
            slider_n_heads_sharing_slider=args.n_heads_sharing_slider,
            slider_dropout=args.cond_dropout,
            n_positions=target_n_positions,
            **base_config
        )
    else:
        # Raw or unconditional
        config = config_class(n_positions=target_n_positions, **base_config)

    model = model_class(config)
    print(f"Built {conditionality or 'GPT2'} model with n_positions={target_n_positions}")
    
    model.resize_token_embeddings(vocab_size)
    return model
