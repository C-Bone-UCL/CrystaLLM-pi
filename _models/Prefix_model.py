"""Prefix-conditioned GPT-2 models using learned past-key-value tokens."""

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from torch import nn


def reshape_prefix_kv_to_past_key_values(kv_tensor: torch.Tensor, batch_size: int, n_tokens: int, config: "PrefixGPT2Config") -> tuple:
    """Reshape a flat prefix projection into GPT-2 past_key_values tuples.

    The layer axis is permuted ahead of the token axis so each block can take its own slice without extra indexing. Legacy `PKV_model` keeps this layer-major, which is half of why a PKV checkpoint loads into `PrefixGPT` without error and behaves differently.

    Args:
        kv_tensor: [B, n_tokens * n_layer * 2 * hidden_size] - flat encoder output
        batch_size: rows in the batch
        n_tokens: prefix tokens per layer
        config: supplies n_layer, n_head, hidden_size

    Returns:
        n_layer tuples of (key, value), each [B, n_head, n_tokens, head_dim]
    """
    head_dim = config.hidden_size // config.n_head
    kv_tensor = kv_tensor.view(batch_size, n_tokens, config.n_layer, -1)

    # Move the layer axis ahead of the token axis so each GPT-2 block can
    # recover its own prefix slice without extra indexing logic.
    kv_tensor = kv_tensor.permute(0, 2, 1, 3)
    k, v = kv_tensor.chunk(2, dim=-1)

    # Split each layer's prefix projection into per-head K and V tensors.
    k = k.view(batch_size, config.n_layer, n_tokens, config.n_head, head_dim).permute(0, 1, 3, 2, 4)

    v = v.view(batch_size, config.n_layer, n_tokens, config.n_head, head_dim).permute(0, 1, 3, 2, 4)

    return tuple((k[:, i], v[:, i]) for i in range(config.n_layer))


def prepend_prefix_attention_mask(attention_mask: torch.Tensor, batch_size: int, n_prefix_tokens: int) -> torch.Tensor | None:
    """Extend attention_mask to cover prefix tokens with ones."""
    if attention_mask is None:
        return None
    prefix_attention = torch.ones(
        batch_size,
        n_prefix_tokens,
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return torch.cat([prefix_attention, attention_mask], dim=1)

class PrefixGPT2Config(GPT2Config):
    """Configuration for prefix-conditioned GPT-2.

    `n_input_vector` specifies the number of scalar properties, `n_prefix_tokens` the number of virtual tokens emitted per layer, and `n_hidden_cond` the conditioning encoder width. Cross-attention is disabled because conditioning enters through cached key-values.
    """

    def __init__(
        self,
        n_input_vector: int = 2,
        n_prefix_tokens: int = 2,
        n_hidden_cond: int = 128,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.add_cross_attention = False

        self.n_input_vector = n_input_vector
        self.n_prefix_tokens = n_prefix_tokens
        self.n_hidden_cond = n_hidden_cond
        self.dropout = dropout

class PrefixEncoder(nn.Module):
    """Project a scalar conditioning vector into per-layer prefix key-value tensors.

    The MLP emits one flat tensor per batch item, which is later reshaped into layers, heads, and prefix tokens. This encoder uses GELU, whereas the legacy `PKV_model` uses ReLU, so the two checkpoint families are not interchangeable.
    """

    def __init__(self, config: PrefixGPT2Config) -> None:
        super().__init__()
        self.config = config

        # Calculate KV size for every layer.
        self.kv_size = (
            config.n_prefix_tokens
            * config.n_head
            * (config.hidden_size // config.n_head)
            * 2
            * config.n_layer
        )

        self.processor = nn.Sequential(
            nn.Linear(config.n_input_vector, config.n_hidden_cond),
            nn.LayerNorm(config.n_hidden_cond),
            # nn.ReLU(),
            # now GELU
            nn.GELU(),
            nn.Linear(config.n_hidden_cond, config.n_hidden_cond * 2)
        )

        self.to_kv = nn.Linear(config.n_hidden_cond * 2, self.kv_size)
        self.dropout = nn.Dropout(config.dropout) # Use conditioning dropout

    def forward(self, x: torch.Tensor) -> tuple:
        """Encode scalar conditioning values into GPT-2 ``past_key_values``.

        Args:
            x: Scalar conditioning values with shape ``[B, n_input_vector]``.

        Returns:
            One ``(key, value)`` tuple per layer, each with shape
            ``[B, n_head, n_prefix_tokens, head_dim]``.
        """
        batch_size = x.shape[0]

        # The encoder emits one flat prefix blob per batch item. Reshape happens
        # only after the final linear layer so the MLP can stay token-agnostic.
        x = self.processor(x)
        x = self.to_kv(x)
        x = self.dropout(x)
        return reshape_prefix_kv_to_past_key_values(
            x, batch_size, self.config.n_prefix_tokens, self.config
        )

class PrefixGPT(GPT2LMHeadModel):
    """GPT-2 conditioned on scalar properties through learned prefix key-values.

    The encoder output occupies `config.n_prefix_tokens` cached positions before the text sequence. Generation must therefore account for the prefix when computing the effective text length and position embeddings.
    """
    config_class = PrefixGPT2Config

    def __init__(self, config: PrefixGPT2Config) -> None:
        super().__init__(config)
        self.conditioning = PrefixEncoder(config)

    def forward(
        self,
        input_ids: torch.Tensor | None=None,
        attention_mask: torch.Tensor | None=None,
        condition_values: torch.Tensor | None=None,
        labels: torch.Tensor | None=None,
        **kwargs: object
    ) -> "CausalLMOutputWithCrossAttentions":
        # Check if cached past_key_values exist in kwargs
        """Run GPT-2 with scalar conditioning represented as prefix key-values.

        A ``ValueError`` is raised when neither ``condition_values`` nor cached
        ``past_key_values`` is supplied, because no prefix can then be constructed.

        Args:
            input_ids: Token IDs with shape ``[B, T]``.
            attention_mask: Text attention mask with shape ``[B, T]``. Prefix positions
                are added internally.
            condition_values: Scalar conditioning with shape ``[B, n_input_vector]``.
                Values may include ``MISSING_CONDITION_VALUE``.
            labels: Optional targets with shape ``[B, T]``.

        Returns:
            ``CausalLMOutputWithCrossAttentions`` containing logits with shape
            ``[B, T, vocab_size]``.
        """
        if "past_key_values" in kwargs:
            past_key_values = kwargs.pop("past_key_values")
        else:
            if condition_values is not None:
                # Prefix tokens are prepended to the attention mask so cached
                # positions remain visible to every decoder layer.
                past_key_values = self.conditioning(condition_values)
                attention_mask = prepend_prefix_attention_mask(
                    attention_mask, condition_values.shape[0], self.config.n_prefix_tokens
                )
            else:
                past_key_values = None

        if past_key_values is None:
            raise ValueError(
                "WARNING: Prefix Condition values activated but not passed correctly."
            )

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            **kwargs
        )

        return output

