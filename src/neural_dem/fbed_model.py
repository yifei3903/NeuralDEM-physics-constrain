import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neural_dem.modules import (
    ContinuousSincosEmbed,
    DitPerceiverDecoder,
    MultimodalDitBlock,
    ScalarConditioner,
    SupernodePooling,
)
from neural_dem.utils import sampling_utils


class FbedModel(nn.Module):
    """Two-main-branch NeuralDEM with pointwise compressible-fluid decoding."""

    fluid_channels = ("U_G_x", "U_G_y", "U_G_z", "EP_G", "P_G_normalized")

    def __init__(
        self,
        dim: int = 156,
        num_heads: int = 4,
        radius: int | float = 0.003,
        patch_size: tuple[int, int, int] = (4, 4, 3),
        resolution: tuple[int, int, int] = (12, 24, 3),
        depth: int = 4,
        normalized_domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
        normalized_domain_max: tuple[float, float, float] = (0.044, 0.12, 0.01),
        normalized_particle_radius: float = 0.0006,
        tau_max: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.radius = radius
        self.patch_size = patch_size
        self.resolution = resolution
        self.depth = depth
        self.tau_max = float(tau_max)

        domain_min = torch.tensor(normalized_domain_min, dtype=torch.float32)
        domain_max = torch.tensor(normalized_domain_max, dtype=torch.float32)
        self.register_buffer("domain_min", domain_min, persistent=True)
        self.register_buffer("domain_max", domain_max, persistent=True)

        self.conditioner = ScalarConditioner(dim)
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=3)

        self.supernode_pooling = SupernodePooling(radius=radius, input_dim=3, ndim=3, hidden_dim=dim)
        self.to_particle_token = nn.Linear(dim, dim)

        patch_volume = int(np.prod(patch_size))
        self.to_fluid_token = nn.Linear(patch_volume * len(self.fluid_channels), dim)

        self.blocks = nn.ModuleList([
            MultimodalDitBlock(
                dim=dim,
                num_heads=num_heads,
                num_main_modalities=2,
                num_aux_modalities=0,
                cond_dim=self.conditioner.cond_dim,
            )
            for _ in range(depth)
        ])

        self.particle_decoder = DitPerceiverDecoder(
            dim=dim, num_heads=num_heads, ndim=3, output_dim=3,
            cond_dim=self.conditioner.cond_dim,
            use_time_embedding=True, tau_max=self.tau_max,
        )
        self.fluid_decoder = DitPerceiverDecoder(
            dim=dim, num_heads=num_heads, ndim=3, output_dim=len(self.fluid_channels),
            cond_dim=self.conditioner.cond_dim, flatten_output=False,
            use_math_attention=True,
            use_time_embedding=True, tau_max=self.tau_max,
        )

        self.register_buffer(
            "normalized_domain_boundary",
            torch.stack([domain_min, domain_max]),
            persistent=False,
        )
        self.register_buffer(
            "padded_normalized_domain_boundary",
            torch.stack([domain_min + normalized_particle_radius, domain_max - normalized_particle_radius]),
            persistent=False,
        )
        self.reset_weights()

    def reset_weights(self) -> None:
        # The old implementation tested Parameter objects with isinstance(...,
        # nn.Linear), so its intended DiT zero initialization never ran.
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "modulation" in name:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _normalize_fluid_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize physical coordinates inside the Torch graph to [0,1]."""
        return (coords - self.domain_min) / (self.domain_max - self.domain_min)

    def encode_state(
        self,
        inlet_velocity: torch.Tensor,
        input_particle_coords: torch.Tensor,
        input_particle_displacement: torch.Tensor,
        supernode_idx: torch.Tensor,
        input_fluid_coords: torch.Tensor,
        input_fluid_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode one current state exactly once, independently of query tau."""
        condition = self.conditioner(inlet_velocity)

        particle_tokens = self.supernode_pooling(
            input_feat=input_particle_displacement,
            input_pos=input_particle_coords,
            supernode_idx=supernode_idx,
        )
        particle_tokens = self.to_particle_token(particle_tokens)

        fluid_tokens = self.to_fluid_token(input_fluid_state)
        fluid_tokens = fluid_tokens + self.pos_embed(self._normalize_fluid_coords(input_fluid_coords))

        x = [particle_tokens, fluid_tokens]
        normalization_boundaries = {depth_idx for depth_idx in (self.depth // 3, 2 * self.depth // 3)
                                    if 0 < depth_idx < self.depth}
        for i, block in enumerate(self.blocks):
            if i in normalization_boundaries:
                x = [F.layer_norm(branch, [self.dim], eps=1e-6) for branch in x]
            x = block(x=x, cond=(condition, condition))
        particle_tokens, fluid_tokens = x
        return {
            "condition": condition,
            "particle_tokens": particle_tokens,
            "fluid_tokens": fluid_tokens,
            "particle_query_coords": (
                input_particle_coords.unsqueeze(0)
                if input_particle_coords.ndim == 2 else input_particle_coords),
        }

    @staticmethod
    def _expand_latent_for_queries(value: torch.Tensor, num_queries: int) -> torch.Tensor:
        return value[:, None].expand(-1, num_queries, *value.shape[1:]).reshape(
            value.shape[0] * num_queries, *value.shape[1:])

    def _canonicalize_spacetime_queries(
        self,
        output_fluid_coords: torch.Tensor,
        query_tau: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
        """Return [B,K,N,*] coordinates/time and legacy-output marker."""
        legacy = query_tau is None
        if output_fluid_coords.ndim not in (3, 4):
            raise ValueError("output_fluid_coords must have shape [B,N,3] or [B,K,N,3]")
        if output_fluid_coords.ndim == 3:
            batch_size, num_cells, _ = output_fluid_coords.shape
        else:
            batch_size, _, num_cells, _ = output_fluid_coords.shape

        if query_tau is None:
            query_tau = output_fluid_coords.new_full(
                (batch_size, 1, num_cells, 1), self.tau_max)
        elif query_tau.ndim == 2:
            query_tau = query_tau[:, :, None, None].expand(-1, -1, num_cells, 1).clone()
        elif query_tau.ndim == 3 and query_tau.shape[-1] == 1:
            query_tau = query_tau[:, :, None, :].expand(-1, -1, num_cells, -1).clone()
        elif query_tau.ndim != 4:
            raise ValueError("query_tau must have shape [B,K], [B,K,1], or [B,K,N,1]")

        if query_tau.shape[0] != batch_size or query_tau.shape[2:] != (num_cells, 1):
            raise ValueError(
                f"query_tau shape {query_tau.shape} is incompatible with {batch_size=} {num_cells=}")
        num_queries = query_tau.shape[1]
        if output_fluid_coords.ndim == 3:
            output_fluid_coords = output_fluid_coords[:, None].expand(
                -1, num_queries, -1, -1)
        elif output_fluid_coords.shape[1] != num_queries:
            raise ValueError("coordinate K dimension does not match query_tau")
        return output_fluid_coords, query_tau, num_queries, legacy

    def decode_queries(
        self,
        encoded: dict[str, torch.Tensor],
        output_fluid_coords: torch.Tensor,
        query_tau: torch.Tensor | None,
        fluid_voidfraction_min: torch.Tensor,
        num_output_particles: torch.Tensor | int,
    ):
        coords_bk, tau_bk, num_queries, legacy = self._canonicalize_spacetime_queries(
            output_fluid_coords, query_tau)
        batch_size, _, num_cells, _ = coords_bk.shape
        if batch_size != 1:
            raise ValueError("variable-size particle decoding currently requires batch_size=1")

        condition = self._expand_latent_for_queries(encoded["condition"], num_queries)
        fluid_tokens = self._expand_latent_for_queries(encoded["fluid_tokens"], num_queries)
        flat_coords = coords_bk.reshape(batch_size * num_queries, num_cells, 3)
        flat_tau = tau_bk.reshape(batch_size * num_queries, num_cells, 1)

        # Queries are cell centers. DitPerceiverDecoder has cross-attention only:
        # query i never mixes with query j, which makes diagonal space/time AD valid.
        fluid_output = self.fluid_decoder(
            x=fluid_tokens,
            pos=self._normalize_fluid_coords(flat_coords),
            block_kwargs=dict(cond=condition),
            query_time=flat_tau,
        )
        fluid_output = fluid_output.reshape(
            batch_size, num_queries, num_cells, len(self.fluid_channels))

        # Cell order is canonical flattened (x,y,z), matching sampling_utils.
        voidfraction = fluid_output[..., 3:4]
        solidfraction = (1.0 - voidfraction).clamp(min=0.0, max=1.0)
        max_solidfraction = (1.0 - fluid_voidfraction_min).reshape(-1, 1, 1)
        solidfraction = solidfraction / max_solidfraction
        if isinstance(num_output_particles, torch.Tensor):
            num_particles = int(num_output_particles.reshape(-1)[0].item())
        else:
            num_particles = int(num_output_particles)
        if legacy:
            # Preserve the original stochastic output-particle query path.
            solidfraction_legacy = solidfraction[:, 0].reshape(*self.resolution, 1)
            sampled = sampling_utils.sample_particles_from_solidfraction(
                solidfraction=solidfraction_legacy,
                num_particles=num_particles,
                bbox=self.normalized_domain_boundary,
                sampling_bbox=self.padded_normalized_domain_boundary,
                volume_shape=self.resolution,
            )
            output_particle_coords = sampled.unsqueeze(0).unsqueeze(0)
            num_particles = sampled.shape[0]
        else:
            # Continuous-time displacement is queried for the same current
            # particles at every tau, giving a stable [B,K,Np,3] tensor.
            base_particle_coords = encoded["particle_query_coords"][:, :num_particles]
            num_particles = base_particle_coords.shape[1]
            output_particle_coords = base_particle_coords[:, None].expand(
                -1, num_queries, -1, -1)

        particle_tokens = self._expand_latent_for_queries(
            encoded["particle_tokens"], num_queries)
        flat_particle_coords = output_particle_coords.reshape(
            batch_size * num_queries, num_particles, 3)
        particle_tau = tau_bk[:, :, :1, :].expand(
            -1, -1, num_particles, -1).reshape(
                batch_size * num_queries, num_particles, 1)
        particle_displacement = self.particle_decoder(
            x=particle_tokens,
            pos=flat_particle_coords,
            block_kwargs=dict(cond=condition),
            query_time=particle_tau,
            flatten_output=False,
        )
        particle_displacement = particle_displacement.reshape(
            batch_size, num_queries, num_particles, 3)

        if legacy:
            return {
                "particle_displacement": particle_displacement[:, 0].reshape(-1, 3),
                "particle_coords": output_particle_coords[:, 0],
                "fluid": fluid_output[:, 0],
            }
        return {
            "particle_displacement": particle_displacement,
            "particle_coords": output_particle_coords,
            "fluid": fluid_output,
        }

    def decode_fluid_only(
        self,
        encoded: dict[str, torch.Tensor],
        output_fluid_coords: torch.Tensor,
        query_tau: torch.Tensor,
    ) -> torch.Tensor:
        """Decode spacetime fluid queries without invoking particle sampling."""
        coords_bk, tau_bk, num_queries, _ = self._canonicalize_spacetime_queries(
            output_fluid_coords, query_tau)
        batch_size, _, num_cells, _ = coords_bk.shape
        condition = self._expand_latent_for_queries(encoded["condition"], num_queries)
        fluid_tokens = self._expand_latent_for_queries(encoded["fluid_tokens"], num_queries)
        output = self.fluid_decoder(
            x=fluid_tokens,
            pos=self._normalize_fluid_coords(
                coords_bk.reshape(batch_size * num_queries, num_cells, 3)),
            block_kwargs=dict(cond=condition),
            query_time=tau_bk.reshape(batch_size * num_queries, num_cells, 1),
        )
        return output.reshape(
            batch_size, num_queries, num_cells, len(self.fluid_channels))

    def forward(
        self,
        inlet_velocity: torch.Tensor,
        input_particle_coords: torch.Tensor,
        input_particle_displacement: torch.Tensor,
        supernode_idx: torch.Tensor,
        input_fluid_coords: torch.Tensor,
        input_fluid_state: torch.Tensor,
        output_fluid_coords: torch.Tensor,
        fluid_voidfraction_min: torch.Tensor,
        num_output_particles: torch.Tensor | int,
        query_tau: torch.Tensor | None = None,
    ):
        encoded = self.encode_state(
            inlet_velocity=inlet_velocity,
            input_particle_coords=input_particle_coords,
            input_particle_displacement=input_particle_displacement,
            supernode_idx=supernode_idx,
            input_fluid_coords=input_fluid_coords,
            input_fluid_state=input_fluid_state,
        )
        return self.decode_queries(
            encoded=encoded,
            output_fluid_coords=output_fluid_coords,
            query_tau=query_tau,
            fluid_voidfraction_min=fluid_voidfraction_min,
            num_output_particles=num_output_particles,
        )
