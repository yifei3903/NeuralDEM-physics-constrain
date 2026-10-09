"""Stage-7 local dynamics state conversion, Euler integration, and losses."""

import torch
import torch.nn.functional as F


RATE_CHANNELS = ("du_dt", "dv_dt", "dw_dt", "dmstar_dt", "dpstar_dt")
FLUID_CHANNELS = ("u", "v", "w", "epsilon", "pressure")


def reference_density(data_cfg) -> float:
    return (
        data_cfg.pressure_ref * data_cfg.molecular_weight
        / (data_cfg.gas_constant * data_cfg.gas_temperature)
    )


def density_from_pressure_star(pressure_star: torch.Tensor, data_cfg) -> torch.Tensor:
    pressure = pressure_star * data_cfg.pressure_scale + data_cfg.pressure_ref
    return pressure * data_cfg.molecular_weight / (
        data_cfg.gas_constant * data_cfg.gas_temperature)


def external_to_dynamics_state(fluid: torch.Tensor, data_cfg) -> torch.Tensor:
    """[u,v,w,epsilon,P*] -> [u,v,w,m*,P*]."""
    rho = density_from_pressure_star(fluid[..., 4:5], data_cfg)
    mass_star = fluid[..., 3:4] * rho / reference_density(data_cfg)
    return torch.cat([fluid[..., :3], mass_star, fluid[..., 4:5]], dim=-1)


def dynamics_to_external_state(state: torch.Tensor, data_cfg) -> torch.Tensor:
    """[u,v,w,m*,P*] -> [u,v,w,epsilon,P*], without clamping."""
    rho = density_from_pressure_star(state[..., 4:5], data_cfg)
    epsilon = reference_density(data_cfg) * state[..., 3:4] / rho
    return torch.cat([state[..., :3], epsilon, state[..., 4:5]], dim=-1)


def euler_integrate_fluid(
    current_external: torch.Tensor,
    predicted_rate: torch.Tensor,
    dt: float,
    data_cfg,
    return_internal: bool = False,
):
    if current_external.shape != predicted_rate.shape:
        raise ValueError("current fluid and rate must have identical shapes")
    current_internal = external_to_dynamics_state(current_external, data_cfg)
    next_internal = current_internal + float(dt) * predicted_rate
    next_external = dynamics_to_external_state(next_internal, data_cfg)
    if return_internal:
        return next_external, current_internal, next_internal
    return next_external


def dynamics_model_inputs(batch, device):
    return {
        key: batch[key].to(device)
        for key in (
            "inlet_velocity", "input_particle_coords", "input_particle_displacement",
            "supernode_idx", "input_fluid_coords", "input_fluid_state",
            "output_fluid_coords",
        )
    }


def compute_dynamics_losses(prediction, batch, predicted_fluid, device, train_cfg):
    pred_coords = prediction["particle_coords"].squeeze(0)
    input_coords = batch["input_particle_coords"].to(device)
    target_particle = batch["gt_particle_displacement"].to(device)
    if pred_coords.shape == input_coords.shape and torch.equal(pred_coords, input_coords):
        matched_target = target_particle
    else:
        nearest = torch.cdist(pred_coords.float(), input_coords.float()).argmin(dim=1)
        matched_target = target_particle[nearest]
    particle_loss = F.mse_loss(prediction["particle_displacement"], matched_target)

    target = batch["gt_fluid"].to(device)
    channel_losses = [
        F.mse_loss(predicted_fluid[..., index:index + 1], target[..., index:index + 1])
        for index in range(5)
    ]
    velocity_loss = sum(channel_losses[:3]) / 3.0
    voidfraction_loss = channel_losses[3]
    pressure_loss = channel_losses[4]
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
        **{f"{name}_mse": loss for name, loss in zip(FLUID_CHANNELS, channel_losses)},
    }


def invalid_state_counts(fluid: torch.Tensor, data_cfg) -> dict:
    pressure = fluid[..., 4:5] * data_cfg.pressure_scale + data_cfg.pressure_ref
    epsilon = fluid[..., 3:4]
    return {
        "epsilon_below_zero": int((epsilon < 0.0).sum().item()),
        "epsilon_above_one": int((epsilon > 1.0).sum().item()),
        "pressure_nonpositive": int((pressure <= 0.0).sum().item()),
        "nan": int(torch.isnan(fluid).sum().item()),
        "inf": int(torch.isinf(fluid).sum().item()),
        "epsilon_min": float(epsilon.min().item()),
        "epsilon_max": float(epsilon.max().item()),
        "pressure_min_pa": float(pressure.min().item()),
        "pressure_max_pa": float(pressure.max().item()),
    }


def load_shared_initial_state(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"initial-state mismatch missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}")
    return {
        "loaded_keys": len(state),
        "newly_initialized_keys": [],
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
