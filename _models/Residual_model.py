"""Residual-slider GPT-2 conditioning for scalar non-graph features."""

import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2Block as HFGPT2Block,
    GPT2MLP,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)
from transformers.pytorch_utils import Conv1D
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask_for_sdpa,
    _prepare_4d_attention_mask_for_sdpa,
)
from transformers import GPT2Config, GPT2PreTrainedModel, GenerationMixin
from transformers.cache_utils import Cache, DynamicCache
from torch.nn import CrossEntropyLoss
import math
from typing import Callable
import warnings
from transformers.utils import logging
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast


logger = logging.get_logger(__name__)

# Define the marker for missing condition values
MISSING_CONDITION_VALUE = -100.0

class ResidualGPT2Config(GPT2Config):
    """Configuration for GPT-2 with residual scalar conditioning.

    `slider_on` enables the conditioning path. `slider_n_variables` specifies the number of scalar
    properties, `slider_n_hidden` the encoder MLP width, and `slider_n_heads_sharing_slider` the
    number of attention heads sharing each conditioning projection. The sharing count must divide
    the number of attention heads.
    """

    def __init__(
        self,
        slider_on: bool=False,
        slider_n_variables: int=1,
        slider_n_hidden: int=768,
        slider_n_heads_sharing_slider: int=1,
        slider_dropout: float=0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.slider_on = slider_on
        self.slider_n_variables = slider_n_variables
        self.slider_n_hidden = slider_n_hidden
        self.slider_n_heads_sharing_slider = slider_n_heads_sharing_slider
        self.slider_dropout = slider_dropout

class ResidualEncoder(nn.Module):
    """Encode scalar conditioning values into per-head key-value tensors.

    Each variable has its own encode, upscaling, and downscaling projections. A projection is shared
    across each group of `slider_n_heads_sharing_slider` attention heads. The learned mixing weight
    is stored as `attention_factor`.
    """

    def __init__(self, config: ResidualGPT2Config) -> None:
        super().__init__()

        self.n_variables = config.slider_n_variables
        self.n_hidden = config.slider_n_hidden
        self.n_heads_sharing_slider = config.slider_n_heads_sharing_slider
        self.n_base_heads = config.num_attention_heads

        if config.hidden_size % self.n_base_heads != 0:
            raise ValueError(f"hidden_size {config.hidden_size} must be divisible by heads {self.n_base_heads}")

        self.n_token_dim = config.hidden_size // self.n_base_heads  # head_dim

        self.register_buffer('dummy', torch.empty(0))

        if self.n_base_heads % self.n_heads_sharing_slider != 0:
            raise ValueError(
                f"n_base_heads ({self.n_base_heads}) must be divisible by "
                f"n_heads_sharing_slider ({self.n_heads_sharing_slider})."
            )

        self.n_slider_heads = self.n_base_heads // self.n_heads_sharing_slider
        self.kv_size = 2 * self.n_token_dim * self.n_slider_heads

        self.encode_linear = nn.Linear(1, self.n_variables * self.kv_size)
        self.upscale_linear = nn.Linear(self.kv_size, self.n_variables * self.n_hidden)
        self.downscale_linear = nn.Linear(self.n_hidden, self.n_variables * self.kv_size)

        self.attention_factor = nn.Parameter(torch.tensor(0.0))

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.slider_dropout)

    def forward(self, prefix: torch.Tensor, hidden_states: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode scalar conditions into per-head key-values and a mixing weight.

        Args:
            prefix: Scalar conditioning with shape ``[B, n_variables]``.
                ``MISSING_CONDITION_VALUE`` and values outside ``(-100.1, 100.1)``
                are treated as absent.
            hidden_states: Optional tensor used to determine device and dtype.

        Returns:
            ``slider_keys`` and ``slider_values`` with shape
            ``[B, n_base_heads, n_variables, head_dim]``, an ``attention_factor``
            parameter, and a ``condition_mask`` with shape ``[B, n_variables]`` that
            is true for present conditions.
        """
        device = hidden_states.device if hidden_states is not None else self.dummy.device
        dtype = hidden_states.dtype if hidden_states is not None else self.dummy.dtype
        prefix = prefix.to(device=device, dtype=dtype)

        # Create mask for present conditions
        # The sentinel marks missing values. The extra numeric bounds protect
        # against accidental extreme inputs that should not contribute.
        condition_mask = (prefix != MISSING_CONDITION_VALUE) & (prefix > -100.1) & (prefix < 100.1)

        # Replace marker with 0.0 for MLP processing
        prefix_processed = torch.where(condition_mask, prefix, torch.zeros_like(prefix))
        prefix_processed = prefix_processed.unsqueeze(-1) # [B, N_vars, 1]

        # MLP Processing (Encode -> Up -> Down)
        # 1. Encode
        encode_w = self.encode_linear.weight.view(self.n_variables, self.kv_size, 1)
        encode_b = self.encode_linear.bias.view(1, self.n_variables, self.kv_size)
        slider_kv = torch.einsum("VKI,BVI->BVK", encode_w, prefix_processed) + encode_b
        slider_kv = self.tanh(slider_kv)
        slider_kv = self.dropout(slider_kv)

        # 2. Upscale
        upscale_w = self.upscale_linear.weight.view(self.n_variables, self.n_hidden, self.kv_size)
        upscale_b = self.upscale_linear.bias.view(1, self.n_variables, self.n_hidden)
        slider_kv = torch.einsum("VHK,BVK->BVH", upscale_w, slider_kv) + upscale_b
        slider_kv = self.tanh(slider_kv)
        slider_kv = self.dropout(slider_kv)

        # 3. Downscale
        downscale_w = self.downscale_linear.weight.view(self.n_variables, self.kv_size, self.n_hidden)
        downscale_b = self.downscale_linear.bias.view(1, self.n_variables, self.kv_size)
        slider_kv = torch.einsum("VKH,BVH->BVK", downscale_w, slider_kv) + downscale_b

        # Reshape and Expand Heads
        slider_kv = slider_kv.view(prefix.shape[0], self.n_variables, 2, self.n_slider_heads, self.n_token_dim)
        # Share each slider head across a small bundle of GPT heads so the
        # conditioning signal is broad without needing one projection per head.
        slider_kv = slider_kv.repeat_interleave(self.n_heads_sharing_slider, dim=3)

        # Permute to [batch, heads, vars, dim, 2]
        slider_kv = slider_kv.permute(0, 3, 1, 4, 2)
        slider_keys, slider_values = slider_kv[..., 0], slider_kv[..., 1]

        return slider_keys, slider_values, self.attention_factor, condition_mask

class ResidualAttention(GPT2Attention):
    """GPT-2 attention with optional residual conditioning contributions."""

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.config = config
        self.layer_idx = layer_idx if layer_idx is not None else "Unknown"

        if getattr(config, "slider_on", False):
            self.slider_out_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.slider_dropout = nn.Dropout(config.slider_dropout)
        else:
            self.slider_out_proj = None
            self.slider_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: tuple[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        slider_key_value_factor_mask: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:

        # Standard GPT2 Self-Attention
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

        present = (key_states, value_states) if use_cache else None
        is_causal = attention_mask is None and query_states.shape[-2] > 1

        # Use SDPA or Eager
        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                using_eager = True
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # Slider Logic
        slider_contribution = 0.0

        if getattr(self.config, "slider_on", False) and slider_key_value_factor_mask is not None:
            slider_key, slider_value, slider_factor, condition_mask = slider_key_value_factor_mask

            bsz, num_heads, seq_len_q, head_dim = query_states.shape

            # 1. Attention Scores: Q * K_slider
            # [B, H, N, Z] @ [B, H, M, Z]^T -> [B, H, N, M]
            slider_scores = torch.einsum("BHNZ,BHMZ->BHNM", query_states, slider_key)

            # 2. Masking
            mask_expanded = condition_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, M]
            slider_scores = torch.where(
                mask_expanded,
                slider_scores,
                torch.full_like(slider_scores, torch.finfo(slider_scores.dtype).min)
            )

            # 3. Softmax
            slider_weights = torch.softmax(slider_scores / math.sqrt(head_dim), dim=-1)
            slider_weights = torch.nan_to_num(slider_weights, nan=0.0)

            # 4. Weighted Sum: Weights * V_slider
            # [B, H, N, M] @ [B, H, M, Z] -> [B, H, N, Z]
            slider_output = torch.einsum("BHNM,BHMZ->BHNZ", slider_weights, slider_value)

            # 5. Reshape to [B, N, H*Z]
            slider_output = slider_output.transpose(1, 2).contiguous()
            slider_contribution = slider_output.reshape(bsz, seq_len_q, num_heads * head_dim)

            # 6. Global Zeroing (if batch item has NO conditions)
            all_masked = ~condition_mask.any(dim=1) # [B]
            if all_masked.any():
                zero_mask = all_masked.view(bsz, 1, 1)
                slider_contribution = torch.where(zero_mask, torch.zeros_like(slider_contribution), slider_contribution)

            # 7. Output Projection (Head Mixing)
            slider_contribution = self.slider_out_proj(slider_contribution)

            # 8. Scaling Factor
            slider_contribution = slider_contribution * slider_factor

            # 9. Dropout
            slider_contribution = self.slider_dropout(slider_contribution)

            # 10. Add to residual stream
            attn_output = attn_output + slider_contribution

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class ResidualGPT2Block(HFGPT2Block):
    """GPT-2 block with residual scalar conditioning."""
    def __init__(self, config, layer_idx=None):
        super(HFGPT2Block, self).__init__()

        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.attn = ResidualAttention(config, layer_idx=layer_idx)

        if getattr(config, "slider_on", False):
            if not hasattr(config, 'slider_n_variables') or config.slider_n_variables <= 0:
                raise ValueError("config.slider_n_variables must be set and positive")
            self.slider = ResidualEncoder(config)
        else:
            self.slider = None

    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        layer_past: tuple[torch.Tensor] | None = None,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        condition_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        # Compute Slider K/V/Factor
        slider_info = None
        if self.slider is not None:
            if condition_values is None:
                raise ValueError("condition_values is None but slider is on")
            slider_info = self.slider(condition_values, hidden_states=hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            slider_key_value_factor_mask=slider_info,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class ResidualGPT2PreTrainedModel(GPT2PreTrainedModel):
    """Base class with residual-slider weight initialization."""
    config_class = ResidualGPT2Config
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["ResidualGPT2Block"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, ResidualEncoder):
            module.attention_factor.data.zero_()

        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                if isinstance(module, (GPT2Attention, ResidualAttention)):
                    p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


class ResidualGPT2Model(ResidualGPT2PreTrainedModel):
    """Transformer body for GPT-2 with residual scalar conditioning."""
    _supports_param_buffer_assignment = False

    def __init__(self, config: ResidualGPT2Config):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([ResidualGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation
        self.post_init()

    def parallelize(self, device_map=None):
        warnings.warn("`parallelize` is deprecated.", FutureWarning)
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        for k, v in self.device_map.items():
             for block_idx in v:
                self.h[block_idx] = self.h[block_idx].to("cuda:" + str(k))
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        warnings.warn("`deparallelize` is deprecated.", FutureWarning)
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        condition_values: torch.Tensor | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Process condition values
        processed_condition_values = None
        if getattr(self.config, "slider_on", False):
            if condition_values is None:
                logger.warning_once("slider_on=True but condition_values=None. Using default missing values.")
                processed_condition_values = torch.full(
                    (batch_size, self.config.slider_n_variables),
                    MISSING_CONDITION_VALUE,
                    dtype=torch.float32,
                    device=device
                )
            else:
                if condition_values.dim() != 2 or condition_values.shape[0] != batch_size:
                     raise ValueError(f"condition_values mismatch: expected ({batch_size}, n_vars), got {condition_values.shape}")
                processed_condition_values = condition_values.to(device=device)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        # generate() passes a Cache object, the blocks below still run on legacy (k, v) tuples
        if isinstance(past_key_values, Cache):
            past_key_values = (
                tuple((layer.keys, layer.values) for layer in past_key_values.layers)
                if past_key_values.get_seq_length() > 0 else None
            )
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size must be > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            if self._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self._attn_implementation == "sdpa" and not output_attentions:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, input_shape[-1]),
                    inputs_embeds if inputs_embeds is not None else self.wte(input_ids),
                    past_length
                )
            else:
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.to(dtype=self.dtype)
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids.view(-1, input_shape[-1]))
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` incompatible with gradient checkpointing. Setting to False.")
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values is not None else None

            if self.model_parallel:
                 block_device = block.ln_1.weight.device
                 hidden_states = hidden_states.to(block_device)
                 if layer_past is not None:
                     layer_past = tuple(past_state.to(block_device) for past_state in layer_past)
                 if attention_mask is not None:
                     attention_mask = attention_mask.to(block_device)
                 if isinstance(head_mask, torch.Tensor):
                     layer_head_mask = head_mask[i].to(block_device)
                 else:
                     layer_head_mask = None
            else:
                 layer_head_mask = head_mask[i] if head_mask is not None else None

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def custom_forward(hs, am, hm):
                    return block(
                        hs, layer_past=None, attention_mask=am, head_mask=hm,
                        use_cache=False, output_attentions=output_attentions,
                        condition_values=processed_condition_values,
                    )

                outputs = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    use_reentrant=False,
                )
                hidden_states = outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (outputs[1],)
            else:
                 outputs = block(
                    hidden_states, layer_past=layer_past, attention_mask=attention_mask,
                    head_mask=layer_head_mask, use_cache=use_cache, output_attentions=output_attentions,
                    condition_values=processed_condition_values
                )
                 hidden_states = outputs[0]
                 if use_cache:
                    presents = presents + (outputs[1],)
                 if output_attentions:
                    attention_output_idx = 2 if use_cache else 1
                    all_self_attentions = all_self_attentions + (outputs[attention_output_idx],)

            if self.model_parallel:
                 for k, v in self.device_map.items():
                     if i == v[-1] and "cuda:" + str(k) != self.last_device:
                         next_device_idx = k + 1
                         while next_device_idx not in self.device_map and next_device_idx <= int(self.last_device.split(":")[1]):
                              next_device_idx += 1
                         if next_device_idx in self.device_map:
                              hidden_states = hidden_states.to("cuda:" + str(next_device_idx))
                         else:
                              hidden_states = hidden_states.to(self.last_device)

        if self.model_parallel:
             hidden_states = hidden_states.to(self.last_device)
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=DynamicCache(presents) if presents is not None else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ResidualGPT(ResidualGPT2PreTrainedModel, GenerationMixin):
    """GPT-2 with scalar conditioning injected into each attention block.

    Conditioning is mixed into attention per layer rather than represented as prefix tokens, so it
    does not change the text sequence length. Missing properties are handled through the encoder's
    condition mask.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: ResidualGPT2Config):
        super().__init__(config)
        self.transformer = ResidualGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.model_parallel = False
        self.device_map = None
        self.post_init()

    def parallelize(self, device_map=None):
        warnings.warn("`parallelize` is deprecated.", FutureWarning)
        self.device_map = (
             get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
             if device_map is None else device_map
         )
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.ln_f.weight.device)
        self.model_parallel = True

    def deparallelize(self):
        warnings.warn("`deparallelize` is deprecated.", FutureWarning)
        self.transformer.deparallelize()
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # generate() pre-builds an empty Cache that is truthy, which would crop the prompt on step one
        if isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0:
            past_key_values = None
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # generate() may supply full-length position_ids, keep only the rows being fed
        if position_ids is not None:
            position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        condition_values = kwargs.get("condition_values", None)
        if condition_values is not None:
             if not getattr(self.config, "slider_on", False):
                  logger.warning_once("condition_values provided but slider_on=False.")
             else:
                  model_inputs["condition_values"] = condition_values

        model_inputs.update({
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
        })
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor]] | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        condition_values: torch.Tensor | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        """Run GPT-2 with conditioning mixed into each attention block.

        Conditioning does not occupy a cached position. Consequently,
        ``past_key_values`` contains text positions only and the text sequence length
        is unchanged.

        Args:
            input_ids: Token IDs with shape ``[B, T]``.
            attention_mask: Attention mask for the text sequence.
            condition_values: Scalar conditions with shape ``[B, slider_n_variables]``.
                ``MISSING_CONDITION_VALUE`` is masked independently for each row and
                variable.
            labels: Optional targets with shape ``[B, T]``.
            past_key_values: Standard GPT-2 cache containing text positions only.

        Returns:
            ``CausalLMOutputWithPast`` containing logits with shape
            ``[B, T, vocab_size]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            condition_values=condition_values,
        )
        hidden_states = transformer_outputs[0]

        if self.model_parallel:
             lm_head_device = self.lm_head.weight.device
             if hidden_states.device != lm_head_device:
                  hidden_states = hidden_states.to(lm_head_device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if shift_logits.shape[1] != shift_labels.shape[1]:
                 min_len = min(shift_logits.shape[1], shift_labels.shape[1])
                 shift_logits = shift_logits[:, :min_len, :]
                 shift_labels = shift_labels[:, :min_len]

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: tuple[tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> tuple[tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
