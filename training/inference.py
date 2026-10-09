"""Autoregressive rollout for the five-channel compressible fluid state."""

import os
import sys
from contextlib import nullcontext

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, DataConfig, ModelConfig
from dataset import (
    build_fluid_coords,
    build_patch_coords,
    denormalize_pressure,
    normalize_pressure,
    patchify_fluid,
    read_fluid_file,
    read_particle_file,
    select_supernodes,
)
from neural_dem import FbedModel
from physics_losses import compute_mass_residual


def build_model(model_cfg, device):
    return FbedModel(
        dim=model_cfg.dim,
        num_heads=model_cfg.num_heads,
        radius=model_cfg.radius,
        patch_size=model_cfg.patch_size,
        resolution=model_cfg.resolution,
        depth=model_cfg.depth,
        normalized_domain_min=model_cfg.domain_min,
        normalized_domain_max=model_cfg.domain_max,
        normalized_particle_radius=model_cfg.particle_radius,
        tau_max=model_cfg.tau_max,
    ).to(device)


def load_model(checkpoint_path, model_cfg, device="cpu"):
    model = build_model(model_cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


def make_fluid_state(velocity, voidfraction, pressure, data_cfg):
    return np.concatenate([
        velocity,
        voidfraction,
        normalize_pressure(pressure, data_cfg),
    ], axis=-1).astype(np.float32)


def run_autoregressive(model, model_cfg, data_cfg, source, start_frame=0,
                       num_steps=20, device="cpu", compute_mass_metrics=True,
                       tau_test=None):
    """Roll out using only predictions after the initial CFD-DEM state."""
    fluid_dir, particle_dir = source["fluid_dir"], source["particle_dir"]
    vel_f, eps_f, pressure_f = read_fluid_file(
        os.path.join(fluid_dir, f"fluid_{start_frame}.txt"), data_cfg)
    vel_p, particle_coords, num_particles = read_particle_file(
        os.path.join(particle_dir, f"particles_{start_frame}.txt"), data_cfg.max_particles)

    fluid_state = torch.from_numpy(make_fluid_state(vel_f, eps_f, pressure_f, data_cfg)).unsqueeze(0).to(device)
    particle_coords = torch.from_numpy(particle_coords).to(device)
    particle_displacement = torch.from_numpy(vel_p * data_cfg.ml_dt).to(device)
    fluid_coords_np = build_fluid_coords(data_cfg)
    patch_coords = torch.from_numpy(build_patch_coords(
        fluid_coords_np, data_cfg, model_cfg.patch_size)).unsqueeze(0).to(device)
    fluid_coords_base = torch.from_numpy(fluid_coords_np).unsqueeze(0).to(device)
    voidfraction_min = torch.tensor([float(eps_f.min())], device=device)

    outputs = []
    model.eval()
    for step in range(1, num_steps + 1):
        supernode_idx = select_supernodes(
            particle_coords.detach().cpu().numpy(), model_cfg.num_supernodes,
            np.random.RandomState(step),
        )
        patched = patchify_fluid(
            fluid_state.squeeze(0).detach().cpu().numpy(), data_cfg, model_cfg.patch_size)
        query_coords = fluid_coords_base.detach().clone()
        query_coords.requires_grad_(compute_mass_metrics)

        context = nullcontext() if compute_mass_metrics else torch.no_grad()
        query_tau = None
        if tau_test is not None:
            query_tau = torch.full(
                (1, 1, data_cfg.n_cells, 1), float(tau_test), device=device)
        with context:
            prediction = model(
                inlet_velocity=torch.tensor([[source["inlet_velocity"]]], device=device),
                input_particle_coords=particle_coords,
                input_particle_displacement=particle_displacement,
                supernode_idx=torch.from_numpy(supernode_idx).to(device),
                input_fluid_coords=patch_coords,
                input_fluid_state=torch.from_numpy(patched).unsqueeze(0).to(device),
                output_fluid_coords=query_coords,
                fluid_voidfraction_min=voidfraction_min,
                num_output_particles=num_particles,
                query_tau=query_tau,
            )

            if tau_test is None:
                predicted_fluid = prediction["fluid"]
                predicted_particle_coords = prediction["particle_coords"]
                predicted_particle_displacement = prediction["particle_displacement"]
                step_dt = data_cfg.ml_dt
            else:
                predicted_fluid = prediction["fluid"][:, 0]
                predicted_particle_displacement = prediction["particle_displacement"][:, 0].squeeze(0)
                predicted_particle_coords = (
                    particle_coords + predicted_particle_displacement).unsqueeze(0)
                step_dt = float(tau_test)

            if compute_mass_metrics:
                residual = compute_mass_residual(
                    predicted_fluid, fluid_state, query_coords, step_dt,
                    data_cfg.pressure_ref, data_cfg.pressure_scale,
                    data_cfg.molecular_weight, data_cfg.gas_constant,
                    data_cfg.gas_temperature, create_graph=False,
                )
                mass_mae = residual.abs().mean().item()
                mass_rms = residual.square().mean().sqrt().item()
            else:
                mass_mae = mass_rms = float("nan")

        fluid_state = predicted_fluid.detach()
        particle_coords = predicted_particle_coords.squeeze(0).detach()
        actual_particle_displacement = predicted_particle_displacement.detach()
        # The encoder's current-state particle feature retains the historical
        # 0.1-s scale independently of the chosen rollout query interval.
        particle_displacement = (
            actual_particle_displacement / step_dt * data_cfg.ml_dt).detach()
        num_particles = len(particle_coords)

        state_np = fluid_state.squeeze(0).cpu().numpy()
        outputs.append({
            "step": step,
            "particle_coords": particle_coords.cpu().numpy(),
            "particle_displacement": actual_particle_displacement.cpu().numpy(),
            "fluid_velocity": state_np[:, :3],
            "fluid_voidfraction": state_np[:, 3:4],
            "fluid_pressure": denormalize_pressure(state_np[:, 4:5], data_cfg),
            "mass_residual_mae": mass_mae,
            "mass_residual_rms": mass_rms,
        })
    return outputs


def evaluate_rollout(predictions, source, start_frame, data_cfg,
                     steps=(1, 5, 10, 20, 50), tau_test=None):
    metrics = {}
    for step in steps:
        if step > len(predictions):
            continue
        pred = predictions[step - 1]
        frame_stride = (
            data_cfg.frame_interval if tau_test is None
            else int(round(float(tau_test) / data_cfg.original_dt)))
        frame = start_frame + step * frame_stride
        true_vel, true_eps, true_pressure = read_fluid_file(
            os.path.join(source["fluid_dir"], f"fluid_{frame}.txt"), data_cfg)
        metrics[step] = {
            "velocity_mae": float(np.abs(pred["fluid_velocity"] - true_vel).mean()),
            "voidfraction_mae": float(np.abs(pred["fluid_voidfraction"] - true_eps).mean()),
            "pressure_mae_pa": float(np.abs(pred["fluid_pressure"] - true_pressure).mean()),
            "mass_residual_mae": pred["mass_residual_mae"],
            "mass_residual_rms": pred["mass_residual_rms"],
        }
    return metrics


if __name__ == "__main__":
    data_cfg, model_cfg = DataConfig(), ModelConfig()
    source = DATA_SOURCES[0]
    checkpoint_path = os.path.join(
        r"D:\NeuralDEM\NeuralDEM-main\training\checkpoints_reduced", "best_model.th")
    model = load_model(checkpoint_path, model_cfg, device="cpu")
    predictions = run_autoregressive(
        model, model_cfg, data_cfg, source, start_frame=5000, num_steps=20,
        device="cpu", compute_mass_metrics=True)
    output_dir = r"D:\NeuralDEM\NeuralDEM-main\training\predictions_reduced"
    os.makedirs(output_dir, exist_ok=True)
    for prediction in predictions:
        np.savez(
            os.path.join(output_dir, f"step_{prediction['step'] - 1:04d}.npz"),
            particle_coords=prediction["particle_coords"],
            particle_disp=prediction["particle_displacement"],
            fluid_vel=prediction["fluid_velocity"],
            fluid_voidfraction=prediction["fluid_voidfraction"],
            fluid_pressure=prediction["fluid_pressure"],
            mass_residual_mae=prediction["mass_residual_mae"],
            mass_residual_rms=prediction["mass_residual_rms"],
        )
    print(evaluate_rollout(predictions, source, 5000, data_cfg))
