"""True autoregressive 1-ms Euler rollout for the Stage-7 generator."""

import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, DataConfig, DynamicsConfig, ModelConfig, TrainConfig
from dataset import (
    build_fluid_coords, build_patch_coords, denormalize_pressure, normalize_pressure,
    patchify_fluid, read_fluid_file, read_particle_file, select_supernodes,
)
from dynamics_utils import (
    dynamics_to_external_state, euler_integrate_fluid, external_to_dynamics_state,
    invalid_state_counts,
)
from evaluate_continuous_time import metrics
from evaluate_dynamics import load_dynamics_model
from train import set_seed


FIELDS = ("u", "v", "w", "epsilon", "pressure", "mstar")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_fluid_state(velocity, epsilon, pressure, data_cfg):
    return np.concatenate([
        velocity, epsilon, normalize_pressure(pressure, data_cfg),
    ], axis=-1).astype(np.float32)


def true_external_state(source, frame, data_cfg):
    velocity, epsilon, pressure = read_fluid_file(
        os.path.join(source["fluid_dir"], f"fluid_{frame}.txt"), data_cfg)
    return make_fluid_state(velocity, epsilon, pressure, data_cfg)


def main():
    data_cfg, model_cfg = DataConfig(), ModelConfig(model_scale="small")
    train_cfg, dynamics_cfg = TrainConfig(), DynamicsConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)
    source = DATA_SOURCES[0]
    model = load_dynamics_model(
        os.path.join(dynamics_cfg.output_dir, "checkpoints", "epoch_500.pt"),
        model_cfg, device)
    fluid_coords_np = build_fluid_coords(data_cfg)
    fluid_coords = torch.from_numpy(fluid_coords_np).unsqueeze(0).to(device)
    patch_coords = torch.from_numpy(build_patch_coords(
        fluid_coords_np, data_cfg, model_cfg.patch_size)).unsqueeze(0).to(device)
    collected_pred = {
        (step, field): [] for step in dynamics_cfg.rollout_steps
        for field in FIELDS[:5]
    }
    collected_true = {
        (step, field): [] for step in dynamics_cfg.rollout_steps
        for field in FIELDS[:5]
    }
    particle_pred = {step: [] for step in dynamics_cfg.rollout_steps}
    particle_true = {step: [] for step in dynamics_cfg.rollout_steps}
    trajectory_rows = []
    invalid_counts = {name: 0 for name in (
        "epsilon_below_zero", "epsilon_above_one", "pressure_nonpositive", "nan", "inf")}
    invalid_extrema = {
        "epsilon_min": float("inf"), "epsilon_max": -float("inf"),
        "pressure_min_pa": float("inf"), "pressure_max_pa": -float("inf"),
    }

    for anchor in dynamics_cfg.rollout_anchor_frames:
        fluid_state_np = true_external_state(source, anchor, data_cfg)
        velocity_p, particle_coords_np, num_particles = read_particle_file(
            os.path.join(source["particle_dir"], f"particles_{anchor}.txt"),
            data_cfg.max_particles)
        fluid_state = torch.from_numpy(fluid_state_np).unsqueeze(0).to(device)
        particle_coords = torch.from_numpy(particle_coords_np).to(device)
        particle_feature = torch.from_numpy(velocity_p * data_cfg.ml_dt).to(device)

        # Include the exact initial condition in trajectory output.
        true_internal_zero = external_to_dynamics_state(fluid_state, data_cfg)
        for cell in dynamics_cfg.trajectory_cells:
            values = torch.cat([fluid_state[..., :5], true_internal_zero[..., 3:4]], dim=-1)[0, cell]
            for channel, field in enumerate(FIELDS):
                value = values[channel].item()
                if field == "pressure":
                    value = float(denormalize_pressure(value, data_cfg))
                trajectory_rows.append({
                    "anchor_frame": anchor, "rollout_step": 0,
                    "physical_time_offset": 0.0, "cell_index": cell,
                    "channel": field, "prediction": value, "cfd_target": value,
                })

        for step in range(1, max(dynamics_cfg.rollout_steps) + 1):
            patched = patchify_fluid(
                fluid_state.squeeze(0).detach().cpu().numpy(),
                data_cfg, model_cfg.patch_size)
            supernode_idx = select_supernodes(
                particle_coords.detach().cpu().numpy(), model_cfg.num_supernodes,
                np.random.RandomState(step))
            with torch.no_grad():
                output = model.predict_dynamics(
                    inlet_velocity=torch.tensor([[source["inlet_velocity"]]], device=device),
                    input_particle_coords=particle_coords,
                    input_particle_displacement=particle_feature,
                    supernode_idx=torch.from_numpy(supernode_idx).to(device),
                    input_fluid_coords=patch_coords,
                    input_fluid_state=torch.from_numpy(patched).unsqueeze(0).to(device),
                    output_fluid_coords=fluid_coords,
                    particle_query_tau=dynamics_cfg.dt_dyn,
                )
                fluid_state = euler_integrate_fluid(
                    fluid_state, output["fluid_rate"], dynamics_cfg.dt_dyn, data_cfg)
                displacement = output["particle_displacement"]
                particle_coords = particle_coords + displacement
                particle_feature = displacement / dynamics_cfg.dt_dyn * data_cfg.ml_dt

            invalid = invalid_state_counts(fluid_state, data_cfg)
            for name in invalid_counts:
                invalid_counts[name] += invalid[name]
            for name in invalid_extrema:
                if name.endswith("min") or "_min_" in name:
                    invalid_extrema[name] = min(invalid_extrema[name], invalid[name])
                else:
                    invalid_extrema[name] = max(invalid_extrema[name], invalid[name])

            true_np = true_external_state(source, anchor + step, data_cfg)
            pred_np = fluid_state.squeeze(0).cpu().numpy()
            pred_internal = external_to_dynamics_state(fluid_state, data_cfg).squeeze(0).cpu().numpy()
            true_internal = external_to_dynamics_state(
                torch.from_numpy(true_np).unsqueeze(0).to(device), data_cfg).squeeze(0).cpu().numpy()
            for cell in dynamics_cfg.trajectory_cells:
                for channel, field in enumerate(FIELDS):
                    if field == "mstar":
                        pred_value, true_value = pred_internal[cell, 3], true_internal[cell, 3]
                    else:
                        pred_value, true_value = pred_np[cell, channel], true_np[cell, channel]
                        if field == "pressure":
                            pred_value = denormalize_pressure(pred_value, data_cfg)
                            true_value = denormalize_pressure(true_value, data_cfg)
                    trajectory_rows.append({
                        "anchor_frame": anchor, "rollout_step": step,
                        "physical_time_offset": step * dynamics_cfg.dt_dyn,
                        "cell_index": cell, "channel": field,
                        "prediction": float(pred_value), "cfd_target": float(true_value),
                    })

            if step in dynamics_cfg.rollout_steps:
                for channel, field in enumerate(FIELDS[:5]):
                    pred_values, true_values = pred_np[:, channel], true_np[:, channel]
                    if field == "pressure":
                        pred_values = denormalize_pressure(pred_values, data_cfg)
                        true_values = denormalize_pressure(true_values, data_cfg)
                    collected_pred[(step, field)].append(pred_values)
                    collected_true[(step, field)].append(true_values)
                _, true_particle_coords, _ = read_particle_file(
                    os.path.join(source["particle_dir"], f"particles_{anchor + step}.txt"),
                    data_cfg.max_particles)
                particle_pred[step].append(particle_coords.cpu().numpy())
                particle_true[step].append(true_particle_coords[:num_particles])

    accuracy_rows = []
    for step in dynamics_cfg.rollout_steps:
        for field in FIELDS[:5]:
            row = {
                "steps": step,
                "horizon_seconds": step * dynamics_cfg.dt_dyn,
                "field": field,
            }
            row.update(metrics(
                np.concatenate(collected_pred[(step, field)]),
                np.concatenate(collected_true[(step, field)])))
            accuracy_rows.append(row)
        row = {
            "steps": step,
            "horizon_seconds": step * dynamics_cfg.dt_dyn,
            "field": "particle_position",
        }
        row.update(metrics(
            np.concatenate(particle_pred[step]), np.concatenate(particle_true[step])))
        accuracy_rows.append(row)

    write_csv(os.path.join(dynamics_cfg.output_dir, "rollout_accuracy.csv"), accuracy_rows)
    write_csv(os.path.join(dynamics_cfg.output_dir, "dynamics_trajectory_samples.csv"), trajectory_rows)
    invalid_path = os.path.join(dynamics_cfg.output_dir, "invalid_state_statistics.json")
    with open(invalid_path, "r", encoding="utf-8") as handle:
        invalid_summary = json.load(handle)
    invalid_summary["autoregressive_rollout"] = {
        "anchors": list(dynamics_cfg.rollout_anchor_frames),
        "steps_per_anchor": max(dynamics_cfg.rollout_steps),
        "counts": invalid_counts,
        "extrema": invalid_extrema,
    }
    with open(invalid_path, "w", encoding="utf-8") as handle:
        json.dump(invalid_summary, handle, indent=2)
    print(json.dumps({
        "rollout_accuracy": accuracy_rows,
        "trajectory_rows": len(trajectory_rows),
        "invalid": invalid_summary["autoregressive_rollout"],
    }, indent=2))


if __name__ == "__main__":
    main()
