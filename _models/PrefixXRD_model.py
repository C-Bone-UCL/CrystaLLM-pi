"""Prefix-conditioned GPT-2 model for XRD spectra.

Ported from CrystaLLM-graph `_models/Prefix_perceiver_model.py`; the MACE graph-conditioned classes in that module stay in CrystaLLM-graph.

XRD Processing inspired by: Inspired by: https://github.com/FrederikLizakJohansen/deCIFer/tree/main
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from .Prefix_model import (
    PrefixGPT2Config,
    prepend_prefix_attention_mask,
    reshape_prefix_kv_to_past_key_values,
)
from .perceiver import PerceiverResampler
from .xrd_utils import (
    QMIN, QMAX, QSTEP, NUM_Q_POINTS, FWHM_RANGE, ETA_RANGE,
    discrete_to_continuous_xrd, xrd_debug_check,
)

VERBOSE = False

# padding value
MISSING_CONDITION_VALUE = -100.0

def _is_torch_compiling() -> bool:
    """Return True when TorchDynamo is compiling the current graph."""
    compiler = getattr(torch, "compiler", None)
    if compiler is None or not hasattr(compiler, "is_compiling"):
        return False
    return compiler.is_compiling()


def _is_primary_rank() -> bool:
    """Return True on rank 0 or when distributed training is inactive."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


class PrefixXRDGPT2Config(PrefixGPT2Config):
    """Configuration for XRD-conditioned prefix GPT-2.

    Extends `PrefixGPT2Config` with the Perceiver resampler's shape. `perceiver_heads * perceiver_dim_head` must equal `n_hidden_cond`, and the constructor asserts it. `skip_xrd_convert_model` selects the condition input mode: False expects discrete peaks and broadens them internally, True expects a dense profile already on the canonical Q grid. `n_input_vector` is pinned to 2 and kept only for backwards compatibility with older checkpoints.
    """

    def __init__(
        self,
        n_prefix_tokens: int = 16,
        n_hidden_cond: int = 256,
        perceiver_depth: int = 2,
        perceiver_heads: int = 8,
        perceiver_dim_head: int = 32,
        perceiver_ff_mult: int = 2,
        skip_xrd_convert_model: bool = False,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        assert perceiver_heads * perceiver_dim_head == n_hidden_cond, (
            "PrefixXRD expects perceiver_heads * perceiver_dim_head == n_hidden_cond"
        )
        # Filter keys we pass explicitly so from_pretrained does not resend them.
        _explicit = {'n_input_vector', 'n_prefix_tokens', 'n_hidden_cond', 'dropout'}
        super().__init__(
            n_input_vector=2,  # Kept for backwards compatibility.
            n_prefix_tokens=n_prefix_tokens,
            n_hidden_cond=n_hidden_cond,
            dropout=dropout,
            **{k: v for k, v in kwargs.items() if k not in _explicit},
        )
        self.perceiver_depth = perceiver_depth
        self.perceiver_heads = perceiver_heads
        self.perceiver_dim_head = perceiver_dim_head
        self.perceiver_ff_mult = perceiver_ff_mult
        self.skip_xrd_convert_model = skip_xrd_convert_model


class XRDPerceiverEncoder(nn.Module):
    """Encode an XRD trace into per-layer prefix key-value tensors.

    A point-wise MLP lifts each `[I, Q]` pair, then the Perceiver resampler cross-attends a fixed number of latents over the trace. That bottleneck is what makes conditioning cost the same compute whatever the trace length.
    """

    def __init__(self, config: PrefixXRDGPT2Config) -> None:
        super().__init__()
        self.config = config
        self.debug = VERBOSE
        self._logged_shapes = False
        head_dim = config.hidden_size // config.n_head

        # Internal format stays [I, Q] for checkpoint compatibility.
        self.point_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, config.n_hidden_cond),
            nn.LayerNorm(config.n_hidden_cond),
        )

        # Compress 1000 projected points into n_prefix_tokens via cross-attention.
        self.resampler = PerceiverResampler(
            dim=config.n_hidden_cond,
            depth=config.perceiver_depth,
            dim_head=config.perceiver_dim_head,
            heads=config.perceiver_heads,
            num_latents=config.n_prefix_tokens,
            ff_mult=config.perceiver_ff_mult,
        )

        # Per-token projection to (K, V) space across all GPT-2 layers.
        kv_per_token = config.n_head * head_dim * 2 * config.n_layer
        self.to_pkv = nn.Linear(config.n_hidden_cond, kv_per_token)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> tuple:
        # x: (B, 1000, 2) with [intensity, Q] point pairs.
        """Encode an XRD point cloud into GPT-2 past_key_values.

        Note the pair order is swapped relative to the model's condition_values, which are [Q, I]. `_build_continuous_points` does the swap.

        Args:
            x: [B, 1000, 2] - XRD points ordered [I, Q]

        Returns:
            n_layer tuples of (key, value), each [B, n_head, n_prefix_tokens, head_dim]
        """
        batch_size = x.shape[0]
        n = self.config.n_prefix_tokens

        # (B, 1000, 2) -> (B, 1000, n_hidden_cond)
        h = self.point_mlp(x)

        if self.debug and not self._logged_shapes and not _is_torch_compiling() and _is_primary_rank():
            print(f"  [XRDEncoder] input      : {tuple(x.shape)}")
            print(f"  [XRDEncoder] point_mlp  : {tuple(h.shape)}")

        # Cross-attention is the bottleneck here: it turns a fixed-length XRD
        # trace into a smaller set of latents that the GPT decoder can consume.
        latents = self.resampler(h)    # (B, n, n_hidden_cond)

        if self.debug and not self._logged_shapes and not _is_torch_compiling() and _is_primary_rank():
            print(f"  [XRDEncoder] resampler  : {tuple(latents.shape)}")

        latents = self.dropout(latents)

        # Project each latent to its KV slice.
        kv_out = self.to_pkv(latents)  # (B, n, kv_per_token)

        if self.debug and not self._logged_shapes and not _is_torch_compiling() and _is_primary_rank():
            print(f"  [XRDEncoder] to_pkv     : {tuple(kv_out.shape)}")
            self._logged_shapes = True

        return reshape_prefix_kv_to_past_key_values(kv_out, batch_size, n, self.config)


class PrefixXRDGPT(GPT2LMHeadModel):
    """GPT-2 conditioned on an XRD pattern through a Perceiver prefix encoder.

    Takes conditioning in one of two shapes depending on `config.skip_xrd_convert_model`, and unlike the scalar families it has no missing-condition mask, so `condition_values` is required at every forward pass.
    """

    config_class = PrefixXRDGPT2Config

    def __init__(self, config: PrefixXRDGPT2Config) -> None:
        super().__init__(config)
        self.conditioning = XRDPerceiverEncoder(config)
        # Toggle model.debug = True to enable shape checks and PNG dumps.
        self.debug = VERBOSE
        self.debug_save_dir = "debug_xrd"
        self._debug_call_count = 0

    def _split_continuous_profile(self, condition_values: torch.Tensor) -> tuple:
        q_cont = condition_values[:, :, 0]
        iq_cont = condition_values[:, :, 1]
        expected_q = torch.arange(QMIN, QMAX, QSTEP, device=condition_values.device, dtype=condition_values.dtype)
        expected_q = expected_q.unsqueeze(0).expand_as(q_cont)
        if not torch.allclose(q_cont, expected_q, atol=1e-4, rtol=0.0):
            raise ValueError("Continuous PrefixXRD input has an unexpected Q grid. Expected local 0.00..9.99 grid.")
        return q_cont, iq_cont

    def _looks_like_continuous_profile(self, condition_values: torch.Tensor) -> bool:
        return condition_values.dim() == 3 and condition_values.shape[-2:] == (NUM_Q_POINTS, 2)

    def _build_continuous_points(self, condition_values: torch.Tensor) -> tuple:
        if getattr(self.config, "skip_xrd_convert_model", False):
            # Some checkpoints already provide a dense profile, so only validate
            # the grid and skip the discrete-peak broadening path.
            if not self._looks_like_continuous_profile(condition_values):
                raise ValueError(
                    "PrefixXRD with skip_xrd_convert_model=True expects condition_values shaped (B, 1000, 2) as [Q, I] pairs."
                )

            q_cont, iq_cont = self._split_continuous_profile(condition_values)
            xrd_points = torch.stack([iq_cont, q_cont], dim=-1)
            return xrd_points, q_cont[0], iq_cont, None, None

        if condition_values.dim() != 3 or condition_values.shape[-1] != 2:
            raise ValueError(
                "PrefixXRD with skip_xrd_convert_model=False expects condition_values shaped (B, N_peaks, 2) as discrete [Q, I] peaks."
            )

        # Discrete PrefixXRD padding uses negative sentinel rows such as [-100, -100].
        # We clamp those padded rows to [0, 0] here, and discrete_to_continuous_xrd()
        # ignores peaks with Q == 0, so they do not contribute to the broadened spectrum.
        cond = condition_values.clamp(min=0.0)
        batch_q = cond[:, :, 0]
        batch_iq = cond[:, :, 1]
        xrd_out = discrete_to_continuous_xrd(batch_q, batch_iq)
        iq_cont = xrd_out['iq']
        q_cont = xrd_out['q']
        q_exp = q_cont.unsqueeze(0).expand(batch_q.shape[0], -1)
        xrd_points = torch.stack([iq_cont, q_exp], dim=-1)
        return xrd_points, q_cont, iq_cont, batch_q, batch_iq

    def set_debug(self, enabled: bool, save_dir: str | None = None) -> None:
        """Keep PrefixXRD debug state in sync across the model and encoder."""
        self.debug = bool(enabled)
        self.conditioning.debug = bool(enabled)
        if save_dir is not None:
            self.debug_save_dir = save_dir

    def forward(
        self,
        input_ids: torch.Tensor | None=None,
        attention_mask: torch.Tensor | None=None,
        condition_values: torch.Tensor | None=None,
        labels: torch.Tensor | None=None,
        **kwargs: object,
    ) -> "CausalLMOutputWithCrossAttentions":
        """Forward pass with XRD conditioning prepended as prefix key-values.

        condition_values takes one of two shapes, chosen by config.skip_xrd_convert_model. False expects discrete peaks and broadens them here; padded rows use a negative sentinel and are clamped to zero, which the broadening ignores. True expects a dense profile whose Q column must match the canonical 0.00-9.99 grid. Raises ValueError on a missing or mis-shaped condition, or an unexpected grid.

        Args:
            input_ids: [B, T] - token ids
            attention_mask: [B, T] - text mask only, prefix positions prepended internally
            condition_values: [B, N_peaks, 2] discrete or [B, 1000, 2] continuous, both [Q, I]
            labels: [B, T] - optional targets

        Returns:
            CausalLMOutputWithCrossAttentions, logits [B, T, vocab_size]
        """
        if "past_key_values" in kwargs:
            past_key_values = kwargs.pop("past_key_values")
        else:
            if condition_values is not None:
                xrd_points, q_cont, iq_cont, batch_q, batch_iq = self._build_continuous_points(condition_values)

                past_key_values = self.conditioning(xrd_points)

                if self.debug and not _is_torch_compiling() and _is_primary_rank():
                    xrd_debug_check(
                        batch_q=batch_q,
                        batch_iq=batch_iq,
                        q_cont=q_cont,
                        iq_cont=iq_cont,
                        past_key_values=past_key_values,
                        config=self.config,
                        step=self._debug_call_count,
                        save_dir=self.debug_save_dir,
                    )
                    self._debug_call_count += 1

                # Extend the attention mask to cover the prefix tokens.
                attention_mask = prepend_prefix_attention_mask(
                    attention_mask, xrd_points.shape[0], self.config.n_prefix_tokens
                )

                if self.debug and not _is_torch_compiling() and _is_primary_rank():
                    if attention_mask is not None:
                        print(f"  [PrefixXRD] attention_mask after prefix extension: {tuple(attention_mask.shape)}")
            else:
                raise ValueError("PrefixXRD requires condition_values at every forward pass.")

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            **kwargs,
        )
