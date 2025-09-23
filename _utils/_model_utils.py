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

def _parse_condition_columns(args):
    """Extract condition vector size from args.condition_columns."""
    try:
        condition_list = ast.literal_eval(str(args.condition_columns))
        return len(condition_list)
    except (ValueError, SyntaxError):
        raise ValueError(f"Invalid condition_columns format: {args.condition_columns}")

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
    
    if not conditionality:
        config = GPT2Config.from_pretrained(args.pretrained_model_dir)
        config.n_positions = args.context_length
        model = GPT2LMHeadModel.from_pretrained(
            args.pretrained_model_dir, config=config, ignore_mismatched_sizes=True
        )
        target_n_positions = args.context_length
        print(f"Loading as standard GPT2 with n_positions={target_n_positions}")
        
    elif conditionality == "PKV":
        target_n_positions = args.context_length + args.n_prefix_tokens
        config = PKVGPT2Config.from_pretrained(
            args.pretrained_model_dir,
            n_input_vector=_parse_condition_columns(args),
            n_prefix_tokens=args.n_prefix_tokens,
            n_hidden_cond=args.n_hidden_cond,
            share_layers=getattr(args, 'share_layers', False),
        )
        model = PKVGPT.from_pretrained(args.pretrained_model_dir, config=config, ignore_mismatched_sizes=True)
        print(f"\nLoading as PKV with n_positions={target_n_positions}")
        
    elif conditionality == "Prepend":
        target_n_positions = args.context_length + args.n_prefix_tokens
        config = PrependGPT2Config.from_pretrained(
            args.pretrained_model_dir,
            n_input_vector=_parse_condition_columns(args),
            n_prefix_tokens=args.n_prefix_tokens,
            n_hidden_cond=args.n_hidden_cond,
            share_layers=getattr(args, 'share_layers', False),
        )
        model = PrependGPT.from_pretrained(args.pretrained_model_dir, config=config, ignore_mismatched_sizes=True)
        print(f"\nLoading as PrependGPT with n_positions={target_n_positions}")
        
    elif conditionality == "Slider":
        target_n_positions = args.context_length
        config = SliderGPT2Config.from_pretrained(
            args.pretrained_model_dir,
            n_positions=target_n_positions,
            vocab_size=vocab_size,
            slider_on=True,
            slider_n_variables=args.n_prefix_tokens,
            slider_n_hidden=args.n_hidden_cond,
            slider_n_heads_sharing_slider=args.n_heads_sharing_slider,
            slider_dropout=args.cond_dropout,
        )
        model = SliderGPT.from_pretrained(args.pretrained_model_dir, config=config, ignore_mismatched_sizes=True)
        print(f"\nLoading as SliderGPT with n_positions={target_n_positions}")
        
    elif conditionality == "Raw":
        target_n_positions = args.context_length
        config = GPT2Config.from_pretrained(
            args.pretrained_model_dir,
            n_positions=target_n_positions,
            vocab_size=vocab_size,
        )
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_dir, config=config, ignore_mismatched_sizes=True)
        print(f"\nLoading as Raw GPT2 with n_positions={target_n_positions}")
        
    else:
        raise ValueError(f"Unknown conditionality type: {conditionality}")

    model.resize_token_embeddings(vocab_size)
    if model.config.n_positions != target_n_positions:
        model = resize_positional_embeddings(model, target_n_positions)

    return model

def build_model(args, tokenizer):
    """Build fresh models with auto-detection of conditional architectures."""
    vocab_size = len(tokenizer)
    conditionality = getattr(args, 'activate_conditionality', None)
    
    # Base config shared across all model types
    base_config = dict(
        vocab_size=vocab_size,
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
    
    if not conditionality:
        target_n_positions = args.context_length
        config = GPT2Config(n_positions=target_n_positions, **base_config)
        model = GPT2LMHeadModel(config=config)
        print(f"\nBuilding standard GPT2 with n_positions={target_n_positions}")
        
    elif conditionality == "PKV":
        # PKV needs extra positions for injected prefix tokens
        target_n_positions = args.context_length + args.n_prefix_tokens
        config = PKVGPT2Config(
            n_input_vector=_parse_condition_columns(args),
            n_prefix_tokens=args.n_prefix_tokens,
            n_hidden_cond=args.n_hidden_cond,
            share_layers=args.share_layers,
            dropout=args.cond_dropout,
            n_positions=target_n_positions,
            **base_config
        )
        model = PKVGPT(config)
        print(f"\nBuilding PKV model with n_positions={target_n_positions}, prefix_tokens={args.n_prefix_tokens}")
        
    elif conditionality == "Prepend":
        target_n_positions = args.context_length + args.n_prefix_tokens
        config = PrependGPT2Config(
            n_input_vector=_parse_condition_columns(args),
            n_prefix_tokens=args.n_prefix_tokens,
            n_hidden_cond=args.n_hidden_cond,
            dropout=args.cond_dropout,
            n_positions=target_n_positions,
            **base_config
        )
        model = PrependGPT(config)
        print(f"\nBuilding PrependGPT with n_positions={target_n_positions}, prefix_tokens={args.n_prefix_tokens}")
        
    elif conditionality == "Slider":
        # Slider uses dual attention, no extra positions needed
        target_n_positions = args.context_length
        config = SliderGPT2Config(
            slider_on=True,
            slider_n_variables=args.n_prefix_tokens,
            slider_n_hidden=args.n_hidden_cond,
            slider_n_heads_sharing_slider=args.n_heads_sharing_slider,
            slider_dropout=args.cond_dropout,
            n_positions=target_n_positions,
            **base_config
        )
        model = SliderGPT(config)
        print(f"\nBuilding SliderGPT with n_positions={target_n_positions}")
        
    elif conditionality == "Raw":
        # Raw conditioning uses text tokens, no architecture changes
        target_n_positions = args.context_length
        config = GPT2Config(n_positions=target_n_positions, **base_config)
        model = GPT2LMHeadModel(config=config)
        print(f"\nBuilding Raw GPT2 with n_positions={target_n_positions}")
        
    else:
        raise ValueError(f"Unknown conditionality type: {conditionality}")

    model.resize_token_embeddings(vocab_size)
    return model
