"""Stage-7 local dynamics and Euler backward smoke test."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, DataConfig, DynamicsConfig, ModelConfig, TrainConfig
from dataset import LocalDynamicsDataset
from dynamics_utils import (
    compute_dynamics_losses,
    density_from_pressure_star,
    dynamics_model_inputs,
    euler_integrate_fluid,
    external_to_dynamics_state,
    invalid_state_counts,
    load_shared_initial_state,
)
from inference import build_model
from train import set_seed


def group_gradient(model, prefix):
    gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.grad is not None
    ]
    return {
        "parameters_with_gradient": len(gradients),
        "norm": float(torch.sqrt(sum(gradient.detach().square().sum()
                                     for gradient in gradients)).item()) if gradients else 0.0,
    }


def run_backward(model, batch, data_cfg, dynamics_cfg, train_cfg, device):
    prediction = model.predict_dynamics(
        **dynamics_model_inputs(batch, device),
        particle_query_tau=dynamics_cfg.dt_dyn)
    current = batch["input_fluid_cell_state"].to(device)
    predicted_fluid, current_internal, next_internal = euler_integrate_fluid(
        current, prediction["fluid_rate"], dynamics_cfg.dt_dyn,
        data_cfg, return_internal=True)
    losses = compute_dynamics_losses(
        prediction, batch, predicted_fluid, device, train_cfg)
    model.zero_grad(set_to_none=True)
    losses["total_data_loss"].backward()
    gradients = {
        "fluid_decoder": group_gradient(model, "fluid_decoder"),
        "transformer": group_gradient(model, "blocks"),
        "fluid_encoder": group_gradient(model, "to_fluid_token"),
    }
    return prediction, predicted_fluid, current_internal, next_internal, losses, gradients


def main():
    data_cfg, model_cfg = DataConfig(), ModelConfig(model_scale="small")
    train_cfg = TrainConfig(use_mass_constraint=False, lambda_mass=0.0)
    dynamics_cfg = DynamicsConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)
    dataset = LocalDynamicsDataset(
        data_cfg, model_cfg, DATA_SOURCES[0],
        dynamics_cfg.num_blocks, dynamics_cfg.block_size_frames,
        train_cfg.seed)
    batch = dataset.get_transition(0, dynamics_cfg.evaluation_offset)
    model = build_model(model_cfg, device)
    initial_path = os.path.join(train_cfg.debug_output_dir, "shared_initial_state.pt")
    load_report = load_shared_initial_state(model, initial_path, device)

    first = run_backward(
        model, batch, data_cfg, dynamics_cfg, train_cfg, device)
    prediction, predicted_fluid, current_internal, next_internal, losses, gradients_initial = first
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay)
    optimizer.step()
    second = run_backward(
        model, batch, data_cfg, dynamics_cfg, train_cfg, device)
    _, _, _, _, _, gradients_after_step = second

    current = batch["input_fluid_cell_state"].to(device)
    zero_dt, zero_internal, zero_next_internal = euler_integrate_fluid(
        current, prediction["fluid_rate"], 0.0, data_cfg, return_internal=True)
    rho = density_from_pressure_star(predicted_fluid[..., 4:5], data_cfg)
    checks = {
        "device": str(device),
        "rate_shape": list(prediction["fluid_rate"].shape),
        "particle_shape": list(prediction["particle_displacement"].shape),
        "rate_finite": bool(torch.isfinite(prediction["fluid_rate"]).all()),
        "predicted_fluid_finite": bool(torch.isfinite(predicted_fluid).all()),
        "mass_star_finite": bool(torch.isfinite(next_internal[..., 3:4]).all()),
        "reconstructed_epsilon_finite": bool(torch.isfinite(predicted_fluid[..., 3:4]).all()),
        "pressure_density_finite": bool(
            torch.isfinite(predicted_fluid[..., 4:5]).all() and torch.isfinite(rho).all()),
        "dt_zero_external_max_error": float((zero_dt - current).abs().max().item()),
        "dt_zero_internal_max_error": float((zero_next_internal - zero_internal).abs().max().item()),
        "euler_internal_identity_max_error": float((
            (next_internal - current_internal)
            - dynamics_cfg.dt_dyn * prediction["fluid_rate"]).abs().max().item()),
        "loss": losses["total_data_loss"].detach().item(),
        "gradients_initial": gradients_initial,
        "gradients_after_one_optimizer_step": gradients_after_step,
        "invalid_state": invalid_state_counts(predicted_fluid, data_cfg),
        "load_report": load_report,
    }
    checks["nan_or_inf"] = not all((
        checks["rate_finite"], checks["predicted_fluid_finite"],
        checks["mass_star_finite"], checks["pressure_density_finite"],
    ))
    if checks["rate_shape"] != [1, data_cfg.n_cells, 5]:
        raise AssertionError("unexpected local-rate shape")
    if checks["nan_or_inf"]:
        raise FloatingPointError("non-finite dynamics smoke result")
    if checks["dt_zero_external_max_error"] > 1e-6:
        raise AssertionError("dt=0 does not reconstruct the current external state")
    if checks["euler_internal_identity_max_error"] > 1e-6:
        raise AssertionError("Euler internal-state identity failed")
    for group in ("fluid_decoder", "transformer", "fluid_encoder"):
        if checks["gradients_after_one_optimizer_step"][group]["parameters_with_gradient"] == 0:
            raise AssertionError(f"no gradient tensors reached {group}")
    os.makedirs(dynamics_cfg.output_dir, exist_ok=True)
    with open(os.path.join(dynamics_cfg.output_dir, "smoke_test.json"),
              "w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
