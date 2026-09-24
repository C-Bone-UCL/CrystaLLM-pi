"""Provide ranking strategies for generated CIF candidates.

Log-probability ranking measures model output likelihood and applies to all model families.
"""

import torch


def forward_pass_logp(model: torch.nn.Module, full_sequences: torch.Tensor, input_length: int, eos_token_id: int | None=None, condition_tensor: torch.Tensor | None=None) -> list[float]:
    """Score generated sequences by perplexity under the model, in one forward pass.

    Uses a fresh teacher-forced pass over the full vocabulary at temperature 1. Scoring
    `outputs.scores` instead measures the sampling distribution, whose per-step renormaliser
    varies by context and so moves the ranking. Excludes tokens before `input_length` and
    from EOS onward.
    """
    if full_sequences is None or full_sequences.numel() == 0:
        return []

    forward_kwargs = {}
    if condition_tensor is not None:
        if condition_tensor.shape[0] != full_sequences.shape[0]:
            repeat_shape = [full_sequences.shape[0]] + [1] * (condition_tensor.dim() - 1)
            condition_tensor = condition_tensor.repeat(*repeat_shape)
        forward_kwargs["condition_values"] = condition_tensor

    with torch.inference_mode():
        logits = model(input_ids=full_sequences, **forward_kwargs).logits

    shifted_logits = logits[:, :-1, :]
    shifted_labels = full_sequences[:, 1:]
    token_log_probs = torch.log_softmax(shifted_logits, dim=-1)
    gathered_log_probs = token_log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)

    start_idx = max(int(input_length) - 1, 0)
    positions = torch.arange(shifted_labels.shape[1], device=full_sequences.device).unsqueeze(0)
    valid_mask = positions >= start_idx

    if eos_token_id is not None:
        eos_mask = full_sequences.eq(eos_token_id)
        has_eos = eos_mask.any(dim=1)
        first_eos = torch.where(
            has_eos,
            eos_mask.float().argmax(dim=1),
            torch.full((full_sequences.shape[0],), full_sequences.shape[1], device=full_sequences.device, dtype=torch.long),
        )
        valid_mask = valid_mask & (positions < (first_eos.unsqueeze(1) - 1))

    token_counts = valid_mask.sum(dim=1)
    safe_counts = torch.clamp(token_counts, min=1)
    mean_log_probs = (gathered_log_probs * valid_mask).sum(dim=1) / safe_counts
    perplexities = torch.exp(-mean_log_probs)

    return [
        float("inf") if token_counts[i].item() <= 0 else perplexities[i].item()
        for i in range(full_sequences.shape[0])
    ]
