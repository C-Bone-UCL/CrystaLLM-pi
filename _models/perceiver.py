"""
Perceiver Resampler module for variable-length to fixed-length sequence compression.

Implements the Flamingo-style Perceiver Resampler which compresses variable-length
inputs (e.g., MACE atom embeddings/XRD) into fixed-size latent representations via
learned queries and cross-attention.

Components:
- PerceiverFeedForward: Standard FFN for Perceiver layers
- PerceiverAttention: Cross-attention where latent queries attend to media
- PerceiverResampler: Full resampler with multiple layers

Used by graph-conditioned models (FlamGraphGPT, ResidualGraphGPT) to handle
variable atom counts while maintaining fixed compute budget.

Inspired by: https://github.com/lucidrains/flamingo-pytorch/blob/main/flamingo_pytorch/flamingo_pytorch.py

"""

import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
from typing import Optional


class PerceiverFeedForward(nn.Module):
    """Feed-forward network for Perceiver layers."""
    
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PerceiverAttention(nn.Module):
    """
    Perceiver cross-attention: latent queries attend to media (e.g. MACE embeddings).
    
    Following Flamingo, latents are concatenated to key/value so they can also
    attend to themselves during the cross-attention operation.
    """
    
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self, 
        media: torch.Tensor, 
        latents: torch.Tensor,
        media_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            media: [B, N_items, dim] - input embeddings (variable length)
            latents: [B, num_latents, dim] - learnable queries
            media_mask: [B, N_items] - True where item is valid, False for padding
            
        Returns:
            Updated latents [B, num_latents, dim]
        """
        b, n_media, _ = media.shape
        h = self.heads

        media = self.norm_media(media)
        latents = self.norm_latents(latents)

        # Queries from latents
        q = self.to_q(latents)

        # Keys and values from concatenated [media, latents]
        # Allows latents to attend to themselves too
        kv_input = torch.cat([media, latents], dim=1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        # Scaled dot-product attention
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # Apply media mask if provided
        if media_mask is not None:
            num_latents = latents.shape[1]
            latent_mask = torch.ones(b, num_latents, dtype=torch.bool, device=media_mask.device)
            full_mask = torch.cat([media_mask, latent_mask], dim=1)
            full_mask = rearrange(full_mask, 'b j -> b 1 1 j')
            
            # Mask out invalid positions -> gives 0 attention weights after softmax
            sim = sim.masked_fill(~full_mask, -1e9)

        # Numerical stability: subtract max to prevent exponential explosion
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        
        attn = sim.softmax(dim=-1).type(q.dtype)

        # Weighted sum
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler: compresses variable-length inputs to fixed latents.
    
    Uses learnable latent queries that cross-attend to input embeddings,
    producing a fixed-size representation regardless of input length.
    
    Args:
        dim: Hidden dimension
        depth: Number of Perceiver attention layers
        dim_head: Dimension per attention head (should divide dim)
        heads: Number of attention heads
        num_latents: Number of output latent vectors
        ff_mult: FFN expansion multiplier
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        dim_head: int = 64,
        heads: int = 8,
        num_latents: int = 32,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                PerceiverFeedForward(dim=dim, mult=ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(
        self, 
        media: torch.Tensor,
        media_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            media: [B, N_items, dim] - input embeddings
            media_mask: [B, N_items] - True where item is valid
            
        Returns:
            latents: [B, num_latents, dim] - compressed representation
        """
        b = media.shape[0]
        latents = repeat(self.latents, 'n d -> b n d', b=b)

        for attn, ff in self.layers:
            latents = attn(media, latents, media_mask=media_mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)