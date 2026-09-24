"""Provide utilities for loading and building CrystaLLM GPT models.

PKV and Slider are legacy load-and-generate-only families retained for released
checkpoints. New training uses Prefix or Residual, with Prefix and
Residual serving as successors to PKV and Slider respectively.
"""

import argparse
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
    ResidualGPT,
    ResidualGPT2Config,
)

# Registry
MODEL_REGISTRY = {
    # Legacy families
    "PKV": (PKVGPT2Config, PKVGPT),
    "Slider": (SliderGPT2Config, SliderGPT),
    # New generation
    "Prefix": (PrefixGPT2Config, PrefixGPT),
    "Residual": (ResidualGPT2Config, ResidualGPT),
    None: (GPT2Config, GPT2LMHeadModel),
}

PREFIX_ARCHITECTURE_NAMES = {"PrefixGPT"}

LEGACY_FAMILIES = ("PKV", "Slider")

# Derived from the registry so adding a family cannot leave the training dispatch behind.
TRAINABLE_CONDITIONAL_FAMILIES = tuple(
    name for name in MODEL_REGISTRY if name and name not in LEGACY_FAMILIES
)


def resolve_data_mode(conditionality: str | None) -> str:
    """Map ``activate_conditionality`` to the corresponding dataloader mode.

    Conditioning families return ``"conditional"``. ``None`` returns
    ``"unconditional"``.
    """
    if conditionality in TRAINABLE_CONDITIONAL_FAMILIES:
        return "conditional"
    if conditionality in ("None", None):
        return "unconditional"
    _resolve_model_entry(conditionality)
    raise ValueError(f"No dataloader mode registered for {conditionality!r}")


def _resolve_model_entry(conditionality: str | None) -> dict:
    """Strict registry lookup for training/finetuning, with the legacy-training rail."""
    if conditionality in LEGACY_FAMILIES:
        raise ValueError(
            f"{conditionality} is a legacy family (kept for older model ckpts). "
            "Train new models with 'Prefix' (same logic as PKV) or 'Residual' (same for slider)."
        )
    if conditionality not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown activate_conditionality {conditionality!r}. "
            f"Valid: {sorted(k for k in MODEL_REGISTRY if k)} or None"
        )
    return MODEL_REGISTRY[conditionality]


def _parse_condition_columns(args: argparse.Namespace) -> list[str]:
    """Extract condition vector size from args.condition_columns."""
    try:
        condition_list = ast.literal_eval(str(args.condition_columns))
        return len(condition_list)
    except (ValueError, SyntaxError):
        raise ValueError(f"Invalid condition_columns format: {args.condition_columns}")


def _validate_residual_condition_width(args: argparse.Namespace) -> None:
    """Validate that Residual uses one slider variable per scalar condition."""
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


def _load_with_sdpa_fallback(model_class: type, pretrained_path: str, config: object, **kwargs) -> tuple:
    """Load a model with SDPA attention, falling back to default if unavailable."""
    try:
        return model_class.from_pretrained(
            pretrained_path, config=config, attn_implementation="sdpa", **kwargs
        )
    except Exception:
        return model_class.from_pretrained(pretrained_path, config=config, **kwargs)


def _get_n_positions(args: argparse.Namespace, conditionality: str | None) -> int:
    """Calculate target n_positions based on conditionality type."""
    # Prefix families prepend n_prefix_tokens as past_key_values, so GPT-2 position
    # IDs are offset and wpe must cover the extended length. Residual (like Slider)
    # injects conditioning inside attention and needs no extra positions.
    if conditionality in ("PKV", "Prefix"):
        return args.context_length + args.n_prefix_tokens
    return args.context_length


def _get_base_config(args: argparse.Namespace, tokenizer: "CustomCIFTokenizer") -> object:
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


def _source_checkpoint_is_prefix_family(config_class: type, pretrained_model_dir: str) -> bool:
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


def resize_positional_embeddings(model: torch.nn.Module, new_n_positions: int, shift_right_by: int=0) -> torch.nn.Module:
    """Resize GPT-2 position embeddings, optionally reserving rows at the front.

    Copies the old rows into the new embedding and fills anything new with values drawn to match the
    old rows' statistics, rather than the default initialization, so the added positions start on
    the same scale as the trained ones.

    `shift_right_by` reserves that many rows at the front and moves the old rows behind them, which
    a Prefix-family conversion needs because the prefix occupies the first positions and text tokens
    must keep their learned embeddings. Records what happened in `model.context_extension_metadata`,
    including the protected row range, so a later context extension can tell trained rows from fresh
    ones.

    Raises ValueError if `shift_right_by` is negative, or if the new size cannot hold the old rows
    plus the requested shift.
    """
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


def _match_weight_stats(reference_weights: torch.Tensor, num_rows: int, device: torch.device) -> torch.Tensor:
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


def load_pretrained_model(args: argparse.Namespace, tokenizer: "CustomCIFTokenizer") -> torch.nn.Module:
    """Load a checkpoint into the model class selected by `activate_conditionality`.

    The model class is resolved through `MODEL_REGISTRY`. Unknown conditionality values raise an
    error. Only the intended resize surface, consisting of position embeddings, token embeddings,
    and the tied head, may have checkpoint shape mismatches. Other mismatches raise.

    When the target context differs, positions are resized. Converting a non-prefix checkpoint to a
    Prefix family shifts pretrained position rows by `n_prefix_tokens` so text tokens retain their
    original positional embeddings.
    """
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
    # that means the checkpoint does not belong to this architecture, so fail loudly instead
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
        target_is_prefix_family = conditionality == "Prefix"
        source_is_prefix_family = _source_checkpoint_is_prefix_family(
            config_class, args.pretrained_model_dir
        )
        shift_right_by = (
            args.n_prefix_tokens if target_is_prefix_family and not source_is_prefix_family else 0
        )
        model = resize_positional_embeddings(
            model, target_n_positions, shift_right_by=shift_right_by
        )

    return model


def build_model(args: argparse.Namespace, tokenizer: "CustomCIFTokenizer") -> torch.nn.Module:
    """Build a fresh model using the class selected by `activate_conditionality`.

    Model selection follows `MODEL_REGISTRY`, and unknown conditionality values raise an error.
    """
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
    return model
