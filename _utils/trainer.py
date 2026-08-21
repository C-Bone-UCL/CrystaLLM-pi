"""Training utilities for CrystaLLM-pi including custom trainers, callbacks, and optimizer setup.
"""

import json
import logging
import os
import socket
from importlib import reload
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AdamW,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    get_scheduler,
)
import torch.distributed as dist

import _tokenizer
reload(_tokenizer)

logger = logging.getLogger(__name__)

FACTOR_1 = 1.0  # Weight for the sequence loss
FACTOR_2 = 1.0  # Weight for the formatting loss

class CIFFormattingTrainer(Trainer):
    """Trainer that adds a fixed-formatting penalty to the model loss.

    The penalty discourages errors in invariant CIF formatting tokens such as
    ``data_``, ``loop_``, and ``cell_length_a``.
    """

    def compute_loss(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        return_outputs=False,
        num_items_in_batch=None
    ):
        """Compute language-model loss with an optional CIF formatting penalty.

        Args:
            model: Model used to compute the language-model loss.
            inputs: Input tensors, optionally including ``fixed_mask`` for the
                formatting penalty.
            return_outputs: Whether to return model outputs together with the loss.
            num_items_in_batch: Number of items in the batch. Unused.

        Returns:
            The loss tensor, or ``(loss, outputs)`` when ``return_outputs=True``.
        """
        # Keep only what the model expects for the CE loss
        model_inputs = {
            k: v for k, v in inputs.items()
            if k not in ["fixed_mask", "special_tokens_mask"]
        }

        outputs = model(**model_inputs)

        # For Finetuning we use original loss
        lm_loss = outputs.loss

        # Early return if no formatting loss needed
        if "fixed_mask" not in inputs:
            return (lm_loss, outputs) if return_outputs else lm_loss

        # Compute formatting loss
        input_ids = inputs["input_ids"]
        fixed_mask = inputs["fixed_mask"]
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Also shift the attention mask if present
        shift_attention_mask = inputs["attention_mask"][..., 1:].contiguous().float()

        # Ensure fixed_mask is a tensor
        if not isinstance(fixed_mask, torch.Tensor):
            fixed_mask_tensor = torch.tensor(fixed_mask, device=shift_labels.device)
        else:
            fixed_mask_tensor = fixed_mask
        shift_fixed_mask = fixed_mask_tensor[..., 1:].contiguous().float()

        # Compute CE loss
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none"
        )

        # For format loss: multiply by shift_fixed_mask (so only 'fixed' tokens) 
        # and shift_attention_mask (ignore any pad).
        ce_loss = ce_loss * shift_fixed_mask.view(-1) * shift_attention_mask.view(-1)
        format_loss = ce_loss.sum() / (shift_fixed_mask.view(-1).sum() + 1e-8)

        # Add format penalty
        # if loss is a tensor, take the mean (seems to be necessary)
        if isinstance(lm_loss, torch.Tensor):
            lm_loss = lm_loss.mean()

        final_loss = FACTOR_1 * lm_loss + FACTOR_2 * format_loss

        # Log breakdown at certain steps
        should_log = model.training and self.state.global_step % self.args.logging_steps == 0
        if should_log:
            loss_breakdown = {"lm_loss": lm_loss.mean().item(), "format_loss": format_loss.item()}
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, loss_breakdown)

        return (final_loss, outputs) if return_outputs else final_loss


class TrainingArgsCallback(TrainerCallback):
    """Write the effective CrystaLLM arguments into every checkpoint."""

    def __init__(self, training_spec: dict[str, object]) -> None:
        self.training_spec = training_spec

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> TrainerControl:
        # Trainer invokes callbacks on every rank. Only one process should write.
        if not state.is_world_process_zero:
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        args_path = checkpoint_dir / "training_args.json"
        args_path.write_text(
            json.dumps(self.training_spec, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Saved training arguments to {args_path}")
        return control


class LossTrack_EarlyStop_Callback(TrainerCallback):
    """Callback to track training and validation losses, and implement early stopping."""
    
    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: float = 0.0):
        super().__init__()
        self.training_losses = []      # Stores training losses
        self.validation_losses = []    # Stores validation losses
        self.training_steps = []       # Tracks steps where training losses are logged
        self.validation_steps = []     # Tracks steps where validation losses are logged
        self.format_losses = []        # Stores formatting losses
        self.lm_losses = []            # Stores language model losses

        # Early stopping attributes
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.patience_counter = 0

    def on_train_begin(self, args, state, control, **kwargs):
        """Perform early stopping setup checks."""
        assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert args.metric_for_best_model is not None, "EarlyStoppingCallback requires metric_for_best_model to be defined"
        assert args.evaluation_strategy != "no", "EarlyStoppingCallback requires EvaluationStrategy of steps or epoch"
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Captures training and validation losses and logs the corresponding step."""
        if logs is None:
            return control
            
        # Map of log keys to their storage lists and step tracking
        loss_mappings = {
            'eval_loss': (self.validation_losses, self.validation_steps),
            'loss': (self.training_losses, self.training_steps),
            'format_loss': (self.format_losses, None),
            'lm_loss': (self.lm_losses, None)
        }
        
        for log_key, (loss_list, step_list) in loss_mappings.items():
            if log_key in logs:
                loss_list.append(logs[log_key])
                if step_list is not None:
                    step_list.append(state.global_step)
        
        return control

    def on_save(self, args, state, control, **kwargs):
        """Saves training and validation losses to a JSON file."""
        output_dir = args.output_dir
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{state.global_step}")

        # Prepare data to save
        losses = {
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "training_steps": self.training_steps,
            "validation_steps": self.validation_steps,
            "format_losses": self.format_losses,
            "lm_losses": self.lm_losses
        }

        # Save to JSON
        loss_file_path = os.path.join(checkpoint_path, "losses.json")
        os.makedirs(checkpoint_path, exist_ok=True)
        with open(loss_file_path, "w") as f:
            json.dump(losses, f, indent=4)
        print(f"Saved losses to {loss_file_path}")
        return control

    def check_metric_value(self, args, state, control, metric_value):
        """Checks if the metric value has improved based on the specified threshold."""
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Evaluates the metric and decides whether to stop training early."""
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"Early stopping requires metric_for_best_model, but did not find {metric_to_check}; early stopping is disabled"
            )
            return control

        self.check_metric_value(args, state, control, metric_value)
        if self.patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
        return control


class DualLRLogger(TrainerCallback):
    """Log learning rates for the base and conditioning parameter groups."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
            
        trainer = kwargs.get("trainer")
        if trainer is None:
            return control
            
        opt = trainer.optimizer
        if opt is None or len(opt.param_groups) < 2:
            return control
            
        # Log both base (group 0) and conditioning (group 1) learning rates
        logs["base_lr"] = opt.param_groups[0]["lr"]
        logs["cond_lr"] = opt.param_groups[1]["lr"]

        return control


def _context_extension_metadata(model: object) -> dict | None:
    """Return resize metadata attached during checkpoint context extension."""
    unwrapped_model = getattr(model, "module", model)
    metadata = getattr(unwrapped_model, "context_extension_metadata", None)
    return metadata if isinstance(metadata, dict) else None


def has_context_extension_wpe(model: object) -> bool:
    """Return whether the model has true context-extension positional rows."""
    metadata = _context_extension_metadata(model)
    if metadata is None:
        return False
    protected_ranges = metadata.get("protected_wpe_row_ranges")
    return bool(metadata.get("is_context_extension") and protected_ranges)


def _context_extension_wpe_weight(model: object) -> torch.nn.Parameter | None:
    """Return the positional embedding weight when context extension is active."""
    if not has_context_extension_wpe(model):
        return None
    unwrapped_model = getattr(model, "module", model)
    transformer = getattr(unwrapped_model, "transformer", None)
    wpe = getattr(transformer, "wpe", None)
    return getattr(wpe, "weight", None)


class ContextExtensionWarmupCallback(TrainerCallback):
    """Mask gradients for copied positional-embedding rows during context-extension warmup.

    A gradient hook is used instead of freezing the rows because freezing would
    require splitting the parameters into separate optimiser groups.
    """

    def __init__(self, context_extension_warmup_steps: int) -> None:
        super().__init__()
        self.context_extension_warmup_steps = int(context_extension_warmup_steps)
        self._state: object | None = None
        self._hook_handle: torch.utils.hooks.RemovableHandle | None = None
        self._protected_row_ranges: list[tuple[int, int]] = []

    def on_train_begin(self, args, state, control, **kwargs):
        """Register a parameter hook before the first backward pass."""
        model = kwargs.get("model")
        if model is None or self.context_extension_warmup_steps <= 0:
            return control

        metadata = _context_extension_metadata(model)
        if not has_context_extension_wpe(model) or metadata is None:
            return control

        wpe_weight = _context_extension_wpe_weight(model)
        if wpe_weight is None:
            return control

        if self._hook_handle is not None:
            self._hook_handle.remove()

        # ref to the trainers live state
        self._state = state
        self._protected_row_ranges = [
            (int(start), int(end)) for start, end in metadata["protected_wpe_row_ranges"]
        ]
        self._hook_handle = wpe_weight.register_hook(self._zero_protected_rows)
        print(
            "Context-extension warmup active: "
            f"protecting transformer.wpe rows {self._protected_row_ranges} "
            f"for {self.context_extension_warmup_steps} optimizer steps"
        )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Remove the hook once training exits."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        return control

    def _zero_protected_rows(self, gradient: torch.Tensor) -> torch.Tensor:
        """Return a gradient tensor with checkpoint-copied rows masked."""
        if self._state is None:
            return gradient
        global_step = int(getattr(self._state, "global_step", 0))
        if global_step >= self.context_extension_warmup_steps:
            return gradient

        # this does the cloning
        masked_gradient = gradient.clone()
        for start_idx, end_idx in self._protected_row_ranges:
            masked_gradient[start_idx:end_idx].zero_()
        return masked_gradient


def start_codecarbon_tracker(args):
    """Start the CodeCarbon tracker to log emissions during training."""
    # Imported here, not at module scope: codecarbon is a training extra, and _utils/__init__
    # star-imports this module, so a top-level import would make it mandatory for generation.
    from codecarbon import OfflineEmissionsTracker

    output_directory = os.path.join('__comp_metrics', args.output_dir)
    os.makedirs(output_directory, exist_ok=True)

    tracker = OfflineEmissionsTracker(
        output_dir=output_directory,
        project_name=f'{args.tracker_project}',
        country_iso_code="GBR",
        log_level="error",
        allow_multiple_runs=True,
    )
    tracker.stop()
    tracker.start()
    return tracker

def find_checkpoint_from_dir(checkpoint_dir):
    """Find the earliest checkpoint in a directory and return the full path."""
    model_files = os.listdir(checkpoint_dir)
    steps = [int(file.split('-')[-1]) for file in model_files if 'checkpoint' in file]
    
    if not steps:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    best_step = min(steps)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint-{best_step}')
    print(f"Using model checkpoint: {checkpoint_path}")
    return checkpoint_path

def params_stats_check(model):
    """Print number of frozen params and trainable params."""
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage_trainable = (trainable_params / (frozen_params + trainable_params)) * 100

    cond_params = 0
    transformer_params = 0
    for name, param in model.named_parameters():
        if "conditioning" in name:
            cond_params += param.numel()
        elif "slider" in name.lower():
            cond_params += param.numel()
        if "transformer" in name and "slider" not in name.lower():
            transformer_params += param.numel()
    percentage_condition = (cond_params / (cond_params + transformer_params)) * 100

    print("\n##Trainable Params##")
    print(f"Model parameters: {model.num_parameters()}")
    print(f"Number of frozen parameters: {frozen_params}")
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Percentage of trainable parameters: {percentage_trainable:.2f}%")
    
    print("\n##Conditioning Params##")
    print(f"Number of conditioning parameters: {cond_params}")
    print(f"Number of base transformer parameters: {transformer_params}")
    print(f"Percentage of conditioning parameters: {percentage_condition:.2f}%\n")

def setup_scheduler(args, model):
    """Set up the optimiser and learning-rate scheduler for training.

    During fine-tuning, conditioning and base parameters use separate learning
    rates. The base model is updated more conservatively while conditioning
    parameters use the higher learning rate.
    """
    extra_sched_kwargs = args.lr_scheduler_kwargs or {}
    num_training_steps = args.max_steps
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(args.warmup_ratio * num_training_steps)

    if args.optimizer == "muon":
        # Training extra, imported lazily for the same reason as codecarbon above.
        from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

        hidden_matrix_params = [] # 2D weights in transformer blocks (Muon)
        adamw_params = [] # Embeddings, lm_head, 1D params (AdamW)
        cond_params = [] # Conditioning module params (AdamW with cond_lr)

        # Keywords that identify conditioning module parameters
        cond_keywords = ["slider", "conditioning", "prefix_embedding", "perceiver"]
        use_cond_separation = args.activate_conditionality in ["Prefix", "PrefixXRD", "Residual"]
        context_extension_wpe_weight = _context_extension_wpe_weight(model)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            name_lower = name.lower()

            # First check if this is a conditioning parameter (separate from Muon entirely)
            if use_cond_separation and any(k in name_lower for k in cond_keywords):
                cond_params.append(param)
            # Context-extension loads protect existing rows and train newly initialized
            # rows, so keep the whole positional table on AdamW.
            elif param is context_extension_wpe_weight:
                adamw_params.append(param)
            # wte/wpe are 2D and match nothing here, so embeddings fall through to Muon at
            # muon_lr. Deliberate, all released models were trained that way and work well.
            # TODO: check perf diff if we add wpe and wte here
            elif "lm_head" in name_lower:
                adamw_params.append(param)
            elif param.ndim < 2:
                # Biases, layer norms, RMSNorm weights (1D)
                adamw_params.append(param)
            else:
                # 2D weights in transformer layers -> Muon
                # This includes q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
                hidden_matrix_params.append(param)

        print(f"Muon params: {len(hidden_matrix_params)}, AdamW params: {len(adamw_params)}, Conditioning params: {len(cond_params)}")
        if context_extension_wpe_weight is not None:
            print("Routing transformer.wpe.weight through AdamW for context-extension load")
        
        is_distributed = dist.is_initialized() and dist.get_world_size() > 1
        
        # Build parameter groups: Muon for 2D transformer weights, AdamW for rest
        param_groups = [
            dict(params=hidden_matrix_params, lr=args.muon_lr, momentum=args.muon_momentum,
                 weight_decay=args.weight_decay, use_muon=True),
            dict(params=adamw_params, lr=args.learning_rate,
                 betas=(args.adam_beta1, args.adam_beta2), eps=1e-8,
                 weight_decay=args.weight_decay, use_muon=False),
        ]
        
        # Add conditioning params as separate AdamW group with cond_lr
        if cond_params:
            cond_lr = args.cond_lr if args.cond_lr is not None else args.learning_rate
            cond_wd = args.cond_wd if args.cond_wd is not None else args.weight_decay
            param_groups.append(
                dict(params=cond_params, lr=cond_lr,
                     betas=(args.adam_beta1, args.adam_beta2), eps=1e-8,
                     weight_decay=cond_wd, use_muon=False)
            )
            print(f"Conditioning params use AdamW with lr={cond_lr}, wd={cond_wd}")
        
        if is_distributed:
            optimizer = MuonWithAuxAdam(param_groups)
            print(f"Using distributed MuonWithAuxAdam (muon_lr={args.muon_lr}, adam_lr={args.learning_rate})")
        else:
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            print(f"Using SingleDeviceMuonWithAuxAdam (muon_lr={args.muon_lr}, adam_lr={args.learning_rate})")
        
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=extra_sched_kwargs,
        )
        return optimizer, lr_scheduler

    if args.activate_conditionality in ["Prefix", "PrefixXRD", "Residual"]:
        base_params, cond_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n.lower() for k in ["slider", "conditioning", "perceiver"]):
                cond_params.append(p)
            else:
                base_params.append(p)

        print(f"Base params: {len(base_params)}, Conditioning params: {len(cond_params)}")

        optimizer = AdamW(
            [
                {"params": base_params,  "lr": args.learning_rate,
                "weight_decay": args.weight_decay},
                {"params": cond_params, "lr": args.cond_lr,
                "weight_decay": args.cond_wd},
            ],
            betas=(args.adam_beta1, args.adam_beta2),
            eps=1e-8
        )

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=extra_sched_kwargs,
        )
        print(f"Using different learning rates for base and conditioning params: {args.learning_rate} and {args.cond_lr}")

    else:
        print("Using same learning rate for all params")
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=1e-8,
            weight_decay=args.weight_decay
        )

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=extra_sched_kwargs,
        )
    return optimizer, lr_scheduler

def acquire_port():
    """Acquire an available port for distributed training."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) 
    pid = os.getpid()
    print(f"\nProcess/rank {rank} (PID {pid}) acquired port {port}")
    return s, port
