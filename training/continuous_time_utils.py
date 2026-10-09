"""Shared helpers for Stage-6 continuous relative-time NeuralDEM."""

import torch
import torch.nn.functional as F


FLUID_CHANNELS = ("u", "v", "w", "epsilon", "pressure")


def make_pointwise_tau(
    tau_values: torch.Tensor,
    num_cells: int,
    device,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Create a leaf [B,K,N,1] tensor with one variable per query point."""
    values = tau_values.to(device=device, dtype=torch.float32)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError("tau_values must have shape [K] or [B,K]")
    query = values[:, :, None, None].expand(-1, -1, num_cells, 1).clone().detach()
    query.requires_grad_(requires_grad)
    return query


def make_spacetime_coords(
    spatial_coords: torch.Tensor,
    num_queries: int,
    device,
    requires_grad: bool = False,
) -> torch.Tensor:
    coords = spatial_coords.to(device=device, dtype=torch.float32)
    if coords.ndim != 3:
        raise ValueError("spatial_coords must have shape [B,N,3]")
    coords = coords[:, None].expand(-1, num_queries, -1, -1).clone().detach()
    coords.requires_grad_(requires_grad)
    return coords


def continuous_model_inputs(batch, device, tau_query, output_coords=None):
    return {
        "inlet_velocity": batch["inlet_velocity"].to(device),
        "input_particle_coords": batch["input_particle_coords"].to(device),
        "input_particle_displacement": batch["input_particle_displacement"].to(device),
        "supernode_idx": batch["supernode_idx"].to(device),
        "input_fluid_coords": batch["input_fluid_coords"].to(device),
        "input_fluid_state": batch["input_fluid_state"].to(device),
        "output_fluid_coords": (
            batch["output_fluid_coords"].to(device)
            if output_coords is None else output_coords
        ),
        "fluid_voidfraction_min": batch["fluid_voidfraction_min"].to(device),
        "num_output_particles": batch["num_output_particles"].to(device),
        "query_tau": tau_query,
    }


def compute_multi_horizon_losses(prediction, batch, device, train_cfg):
    """Preserve the existing per-horizon objective and average over K."""
    target_fluid = batch["gt_fluid"].to(device)
    target_particle = batch["gt_particle_displacement"].to(device)
    input_particle_coords = batch["input_particle_coords"].to(device)
    if prediction["fluid"].shape != target_fluid.shape:
        raise ValueError(
            f"fluid prediction/target mismatch {prediction['fluid'].shape} vs {target_fluid.shape}")
    num_queries = prediction["fluid"].shape[1]
    particle_losses = []
    channel_losses = [[] for _ in FLUID_CHANNELS]
    for query_index in range(num_queries):
        pred_coords = prediction["particle_coords"][0, query_index]
        pred_displacement = prediction["particle_displacement"][0, query_index]
        if len(pred_coords) and len(input_particle_coords):
            if pred_coords.shape == input_particle_coords.shape and torch.equal(
                    pred_coords, input_particle_coords):
                matched_target = target_particle[0, query_index]
            else:
                nearest = torch.cdist(
                    pred_coords.float(), input_particle_coords.float()).argmin(dim=1)
                matched_target = target_particle[0, query_index, nearest]
            particle_losses.append(F.mse_loss(pred_displacement, matched_target))
        else:
            particle_losses.append(prediction["fluid"].new_zeros(()))
        for channel_index in range(len(FLUID_CHANNELS)):
            channel_losses[channel_index].append(F.mse_loss(
                prediction["fluid"][:, query_index, :, channel_index:channel_index + 1],
                target_fluid[:, query_index, :, channel_index:channel_index + 1],
            ))

    particle_loss = torch.stack(particle_losses).mean()
    channel_mean = [torch.stack(losses).mean() for losses in channel_losses]
    velocity_loss = sum(channel_mean[:3]) / 3.0
    voidfraction_loss = channel_mean[3]
    pressure_loss = channel_mean[4]
    fluid_loss = (
        train_cfg.loss_velocity_weight * velocity_loss
        + train_cfg.loss_voidfraction_weight * voidfraction_loss
        + train_cfg.loss_pressure_weight * pressure_loss
    )
    total_loss = (
        train_cfg.loss_particle_weight * particle_loss
        + train_cfg.loss_fluid_weight * fluid_loss
    )
    return {
        "total_data_loss": total_loss,
        "particle_loss": particle_loss,
        "fluid_total_loss": fluid_loss,
        "velocity_loss": velocity_loss,
        "voidfraction_loss": voidfraction_loss,
        "pressure_loss": pressure_loss,
        **{f"{name}_mse": loss for name, loss in zip(FLUID_CHANNELS, channel_mean)},
    }


def load_legacy_initial_state(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected legacy state mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}")
    return incompatible
