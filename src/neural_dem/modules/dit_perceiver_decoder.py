from typing import Any

import einops
import torch
from torch import nn

from .continuous_sincos_embed import ContinuousSincosEmbed
from .dit_perceiver_block import DitPerceiverBlock


class DitPerceiverDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ndim: int,
        output_dim: int,
        cond_dim: int,
        flatten_output: bool = True,
        use_math_attention: bool = False,
        use_time_embedding: bool = False,
        tau_max: float = 0.1,
        eps=1e-6,
    ):
        super().__init__()
        self.flatten_output = flatten_output
        self.use_time_embedding = use_time_embedding
        self.tau_max = float(tau_max)
        if self.tau_max <= 0.0:
            raise ValueError("tau_max must be positive")
        # create query
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.time_embed = (
            ContinuousSincosEmbed(dim=dim, ndim=1)
            if use_time_embedding else None
        )
        if self.time_embed is not None:
            # Frequencies are deterministic constants. Keeping them out of the
            # state dict lets every pre-Stage-6 checkpoint load strictly.
            self.time_embed._non_persistent_buffers_set.add("omega")
        self.query = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # perceiver
        self.perc = DitPerceiverBlock(
            dim=dim,
            num_heads=num_heads,
            cond_dim=cond_dim,
            use_math_attention=use_math_attention,
            eps=eps,
        )
        # to final output
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pred = nn.Linear(dim, output_dim)

    def normalize_time(self, query_time: torch.Tensor) -> torch.Tensor:
        """Map physical relative time in seconds to the decoder time coordinate."""
        return query_time / self.tau_max

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        block_kwargs: dict[str, Any] | None = None,
        query_time: torch.Tensor | None = None,
        flatten_output: bool | None = None,
        include_time: bool | None = None,
    ):
        if pos is None:
            return None
        assert x.ndim == 3
        assert pos.ndim == 3

        # Spatial and relative-time coordinates are embedded independently,
        # then fused pointwise. There is no query-query self-attention.
        query_embedding = self.pos_embed(pos)
        should_include_time = self.use_time_embedding if include_time is None else include_time
        if should_include_time:
            if not self.use_time_embedding:
                raise ValueError("decoder was constructed without a time embedding")
            if query_time is None:
                raise ValueError("query_time is required when use_time_embedding=True")
            if query_time.shape != (*pos.shape[:-1], 1):
                raise ValueError(
                    f"query_time shape {query_time.shape} must match pos {pos.shape[:-1]} + (1,)")
            query_embedding = query_embedding + self.time_embed(
                self.normalize_time(query_time))
        elif query_time is not None:
            raise ValueError("decoder was constructed without time embedding")
        query = self.query(query_embedding)

        # perceiver
        x = self.perc(q=query, kv=x, **(block_kwargs or {}))

        # predict value
        x = self.norm(x)
        x = self.pred(x)

        should_flatten = self.flatten_output if flatten_output is None else flatten_output
        if should_flatten:
            # Particle outputs retain the original sparse representation.
            x = einops.rearrange(x, "batch_size max_num_points dim -> (batch_size max_num_points) dim")
        return x
