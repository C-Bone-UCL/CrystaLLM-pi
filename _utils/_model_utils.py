"""
Model utilities for loading and building CrystaLLM conditional and standard GPT models.

PKV and Slider are legacy load-and-generate-only families kept for released checkpoints;
training builds them nowhere — new work uses Prefix (PKV successor), PrefixXRD or
Residual (Slider successor), ported from CrystaLLM-graph.
"""

import ast
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from _models import (
    PKVGPT,
    SliderGPT,
    PKVGPT2Config,
    SliderGPT2Config,
    PrefixGPT,
    PrefixGPT2Config,
    PrefixXRDGPT,
    PrefixXRDGPT2Config,
    ResidualGPT,
    ResidualGPT2Config,
)

# Registry mapping conditionality types to (config_class, model_class)
MODEL_REGISTRY = {
    # Legacy families: byte-identical classes, kept for released checkpoints (deprecated for new work)
    "PKV": (PKVGPT2Config, PKVGPT),
    "Slider": (SliderGPT2Config, SliderGPT),
    # New generation, ported from CrystaLLM-graph
    "Prefix": (PrefixGPT2Config, PrefixGPT),
    "PrefixXRD": (PrefixXRDGPT2Config, PrefixXRDGPT),
    "Residual": (ResidualGPT2Config, ResidualGPT),
    None: (GPT2Config, GPT2LMHeadModel),
}

# PrefixPerceiverGPT stays in CrystaLLM-graph but its checkpoints are still detectable sources.
PREFIX_ARCHITECTURE_NAMES = {"PrefixGPT", "PrefixXRDGPT", "PrefixPerceiverGPT"}


def _resolve_model_entry(conditionality):
    """Strict registry lookup for training/finetuning, with the legacy-training rail."""
    if conditionality in ("PKV", "Slider"):
        raise ValueError(
            f"{conditionality} is a legacy load-and-generate-only family (kept for released checkpoints). "
            "Train new models with 'Prefix' (PKV successor) or 'Residual' (Slider successor)."
        )
    if conditionality not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown activate_conditionality {conditionality!r}. "
            f"Valid: {sorted(k for k in MODEL_REGISTRY if k)} or None"
        )
    return MODEL_REGISTRY[conditionality]


def configure_runtime_model_flags(model, args):
    """Apply runtime-only debug flags after model creation or checkpoint load."""
    if getattr(args, 'activate_conditionality', None) == "PrefixXRD" and hasattr(model, "set_debug"):
        model.set_debug(getattr(args, 'xrd_debug', False))
    return model


def _parse_condition_columns(args):
    """Extract condition vector size from args.condition_columns."""
    try:
        condition_list = ast.literal_eval(str(args.condition_columns))
        return len(condition_list)
    except (ValueError, SyntaxError):
        raise ValueError(f"Invalid condition_columns format: {args.condition_columns}")


def _validate_residual_condition_width(args):
    """Ensure Residual uses one slider variable per scalar condition."""
    if getattr(args, "activate_conditionality", None) != "Residual":
        return

    condition_width = _parse_condition_columns(args)
    if args.n_prefix_tokens != condition_width:
        raise ValueError(
            "Residual requires n_prefix_tokens to match the number of scalar "
            f"condition columns: got n_prefix_tokens={args.n_prefix_tokens}, "
            f"condition width={condition_width}, "
            f"condition_columns={args.condition_columns}"
        )


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
    # Prefix families prepend n_prefix_tokens as past_key_values, so GPT-2 position
    # IDs are offset and wpe must cover the extended length. Residual (like Slider)
    # injects conditioning inside attention and needs no extra positions.
    if conditionality in ("PKV", "Prefix", "PrefixXRD"):
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


def _source_checkpoint_is_prefix_family(config_class, pretrained_model_dir):
    """Return True when checkpoint metadata shows a Prefix-family source model."""
    try:
        raw_config, _ = config_class.get_config_dict(pretrained_model_dir)
    except Exception:
        return False

    if not isinstance(raw_config, dict):
        return False

    if "n_prefix_tokens" in raw_config:
        return True

    architectures = raw_config.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    if any(arch in PREFIX_ARCHITECTURE_NAMES for arch in architectures):
        return True

    metadata_fields = (
        raw_config.get("model_type"),
        raw_config.get("_class_name"),
        raw_config.get("auto_map"),
    )
    metadata_blob = " ".join(str(field) for field in metadata_fields if field is not None)
    return any(name in metadata_blob for name in PREFIX_ARCHITECTURE_NAMES)


def resize_positional_embeddings(model, new_n_positions, shift_right_by=0):
    """Resize GPT-2 positional embeddings, optionally shifting old rows right."""
    old_n_positions = model.config.n_positions
    if new_n_positions == old_n_positions:
        return model
    if shift_right_by < 0:
        raise ValueError("shift_right_by must be non-negative")
    if shift_right_by and new_n_positions < old_n_positions + shift_right_by:
        raise ValueError(
            "new_n_positions must cover the original rows plus the requested right shift"
        )

    old_wpe = model.transformer.wpe.weight.data

    new_embedding = torch.nn.Embedding(new_n_positions, old_wpe.size(1)).to(
        device=old_wpe.device, dtype=old_wpe.dtype
    )
    new_wpe = new_embedding.weight.data

    if shift_right_by:
        # Reserve the prefix rows at the front and move the old rows right.
        new_wpe[:shift_right_by, :] = _match_weight_stats(
            old_wpe, shift_right_by, old_wpe.device
        )
        new_wpe[shift_right_by:shift_right_by + old_n_positions, :] = old_wpe
        tail_start = shift_right_by + old_n_positions
        if tail_start < new_n_positions:
            new_wpe[tail_start:, :] = _match_weight_stats(
                old_wpe, new_n_positions - tail_start, old_wpe.device
            )
    else:
        new_wpe[:old_n_positions, :] = old_wpe
        new_wpe[old_n_positions:, :] = _match_weight_stats(
            old_wpe, new_n_positions - old_n_positions, old_wpe.device
        )

    model.transformer.wpe = new_embedding
    model.config.n_positions = new_n_positions
    copied_start = shift_right_by
    copied_end = shift_right_by + old_n_positions
    model.context_extension_metadata = {
        "is_context_extension": new_n_positions > copied_end,
        "protected_wpe_row_ranges": [(copied_start, copied_end)],
        "source_n_positions": old_n_positions,
        "target_n_positions": new_n_positions,
        "shift_right_by": shift_right_by,
    }

    if shift_right_by:
        print(
            f"Resized wpe: {old_n_positions} -> {new_n_positions} "
            f"(shifted right by {shift_right_by})"
        )
    else:
        print(f"Resized wpe: {old_n_positions} -> {new_n_positions} (distribution-matched init)")
    return model


def _match_weight_stats(reference_weights, num_rows, device):
    """Sample rows that match the mean/std of an existing embedding table."""
    if num_rows <= 0:
        return torch.zeros(0, reference_weights.size(1), device=device)

    mean = reference_weights.mean(dim=0)
    std = reference_weights.std(dim=0)
    normal_noise = torch.randn(
        num_rows,
        reference_weights.size(1),
        device=device,
        dtype=reference_weights.dtype,
    )
    return normal_noise * std + mean


def load_pretrained_model(args, tokenizer):
    """Load pretrained models with strict conditional-architecture selection."""
    print(f"Loading model weights from {args.pretrained_model_dir}")

    vocab_size = len(tokenizer)
    conditionality = getattr(args, 'activate_conditionality', None)
    config_class, model_class = _resolve_model_entry(conditionality)
    target_n_positions = _get_n_positions(args, conditionality)

    # Build config based on conditionality type
    if conditionality is None:
        config = config_class.from_pretrained(args.pretrained_model_dir)
        config.n_positions = args.context_length

    elif conditionality == "Prefix":
        config = config_class.from_pretrained(
            args.pretrained_model_dir,
            n_input_vector=_parse_condition_columns(args),
            n_prefix_tokens=args.n_prefix_tokens,
            n_hidden_cond=args.n_hidden_cond,
        )

    elif conditionality == "PrefixXRD":
        config = config_class.from_pretrained(
            args.pretrained_model_dir,
            n_prefix_tokens=args.n_prefix_tokens,
            n_hidden_cond=args.n_hidden_cond,
            perceiver_depth=args.perceiver_depth,
            perceiver_heads=args.perceiver_n_heads,
            perceiver_dim_head=args.perceiver_dim_head,
            perceiver_ff_mult=args.perceiver_ff_mult,
            skip_xrd_convert_model=getattr(args, 'skip_xrd_convert_model', False),
        )

    elif conditionality == "Residual":
        _validate_residual_condition_width(args)
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

    model, info = _load_with_sdpa_fallback(
        model_class, args.pretrained_model_dir, config,
        ignore_mismatched_sizes=True, output_loading_info=True,
    )
    # ignore_mismatched_sizes silently re-initializes any weight whose shape disagrees with
    # the checkpoint. Outside the legitimate resize surface (positions, vocab, tied head)
    # that means the checkpoint does not belong to this architecture — fail loudly instead
    # of finetuning partially random weights.
    benign = ("transformer.wpe", "transformer.wte", "lm_head")
    bad = [k for k, *_ in info["mismatched_keys"] if not k.startswith(benign)]
    if bad:
        raise ValueError(
            f"Checkpoint does not fit {model_class.__name__}: re-initialized weights {bad[:5]} "
            "(wrong activate_conditionality for this checkpoint?)"
        )
    print(f"Loaded as {conditionality or 'GPT2'} with n_positions={target_n_positions}")

    model.resize_token_embeddings(vocab_size)
    if model.config.n_positions != target_n_positions:
        # Converting a non-prefix checkpoint into a Prefix-family model must shift the
        # pretrained position rows right so text tokens keep their learned embeddings
        # behind the new prefix slots.
        target_is_prefix_family = conditionality in ("Prefix", "PrefixXRD")
        source_is_prefix_family = _source_checkpoint_is_prefix_family(
            config_class, args.pretrained_model_dir
        )
        shift_right_by = (
            args.n_prefix_tokens if target_is_prefix_family and not source_is_prefix_family else 0
        )
        model = resize_positional_embeddings(
            model, target_n_positions, shift_right_by=shift_right_by
        )

    return configure_runtime_model_flags(model, args)


def build_model(args, tokenizer):
    """Build fresh models with strict conditional-architecture selection."""
    vocab_size = len(tokenizer)
    conditionality = getattr(args, 'activate_conditionality', None)
    config_class, model_class = _resolve_model_entry(conditionality)
    target_n_positions = _get_n_positions(args, conditionality)
    base_config = _get_base_config(args, tokenizer)

    # Build config based on conditionality type
    if conditionality is None:
        # unconditional
        config = config_class(n_positions=target_n_positions, **base_config)

    elif conditionality == "Prefix":
        config = config_class(
            n_input_vector=_parse_condition_columns(args),
            n_prefix_tokens=args.n_prefix_tokens,
            n_hidden_cond=args.n_hidden_cond,
            dropout=args.cond_dropout,
            n_positions=target_n_positions,
            **base_config
        )

    elif conditionality == "PrefixXRD":
        config = config_class(
            n_prefix_tokens=args.n_prefix_tokens,
            n_hidden_cond=args.n_hidden_cond,
            perceiver_depth=args.perceiver_depth,
            perceiver_heads=args.perceiver_n_heads,
            perceiver_dim_head=args.perceiver_dim_head,
            perceiver_ff_mult=args.perceiver_ff_mult,
            skip_xrd_convert_model=getattr(args, 'skip_xrd_convert_model', False),
            dropout=args.cond_dropout,
            n_positions=target_n_positions,
            **base_config
        )

    elif conditionality == "Residual":
        _validate_residual_condition_width(args)
        config = config_class(
            slider_on=True,
            slider_n_variables=args.n_prefix_tokens,
            slider_n_hidden=args.n_hidden_cond,
            slider_n_heads_sharing_slider=args.n_heads_sharing_slider,
            slider_dropout=args.cond_dropout,
            n_positions=target_n_positions,
            **base_config
        )

    model = model_class(config)
    print(f"Built {conditionality or 'GPT2'} model with n_positions={target_n_positions}")

    model.resize_token_embeddings(vocab_size)
    return configure_runtime_model_flags(model, args)
