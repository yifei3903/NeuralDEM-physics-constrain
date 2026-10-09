"""Multi-horizon accuracy, trajectories, and rollout diagnostics for Stage 6."""

import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, ContinuousTimeConfig, DataConfig, ModelConfig, TrainConfig
from continuous_time_utils import continuous_model_inputs, make_pointwise_tau
from dataset import ContinuousTimeCFDDEMDataset, denormalize_pressure
from inference import build_model, evaluate_rollout, run_autoregressive
from train import set_seed


CHANNELS = ("u", "v", "w", "epsilon", "pressure")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    difference = prediction - target
    pred_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    denominator = np.sqrt(np.sum(pred_centered ** 2) * np.sum(target_centered ** 2))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(target), 1e-30)),
        "correlation": float(np.sum(pred_centered * target_centered) / max(denominator, 1e-30)),
    }


def load_stage6_model(path, model_cfg, device):
    model = build_model(model_cfg, device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def evaluate_horizons(model, dataset, data_cfg, ct_cfg, device):
    offsets = [int(round(tau / data_cfg.original_dt)) for tau in ct_cfg.evaluation_taus]
    collected_pred = {(offset, channel): [] for offset in offsets for channel in range(5)}
    collected_true = {(offset, channel): [] for offset in offsets for channel in range(5)}
    particle_pred = {offset: [] for offset in offsets}
    particle_true = {offset: [] for offset in offsets}
    with torch.no_grad():
        for anchor_index in range(len(dataset)):
            batch = dataset.get_with_offsets(anchor_index, offsets)
            tau_query = make_pointwise_tau(
                batch["query_tau_values"], data_cfg.n_cells, device)
            prediction = model(**continuous_model_inputs(batch, device, tau_query))
            pred_fluid = prediction["fluid"].cpu().numpy()[0]
            true_fluid = batch["gt_fluid"].numpy()[0]
            pred_particle = prediction["particle_displacement"].cpu().numpy()[0]
            true_particle = batch["gt_particle_displacement"].numpy()[0]
            for query_index, offset in enumerate(offsets):
                for channel in range(5):
                    pred_values = pred_fluid[query_index, :, channel]
                    true_values = true_fluid[query_index, :, channel]
                    if channel == 4:
                        pred_values = denormalize_pressure(pred_values, data_cfg)
                        true_values = denormalize_pressure(true_values, data_cfg)
                    collected_pred[(offset, channel)].append(pred_values)
                    collected_true[(offset, channel)].append(true_values)
                particle_pred[offset].append(pred_particle[query_index])
                particle_true[offset].append(true_particle[query_index])

    rows = []
    for offset in offsets:
        tau = offset * data_cfg.original_dt
        for channel, name in enumerate(CHANNELS):
            row = {"tau_seconds": tau, "field": name}
            row.update(metrics(
                np.concatenate(collected_pred[(offset, channel)]),
                np.concatenate(collected_true[(offset, channel)])))
            rows.append(row)
        row = {"tau_seconds": tau, "field": "particle_displacement"}
        row.update(metrics(
            np.concatenate(particle_pred[offset]),
            np.concatenate(particle_true[offset])))
        rows.append(row)
    return rows


def temporal_trajectories(model, dataset, data_cfg, device):
    anchor_indices = (0, 5, 10, 15)
    cells = (235, 469, 703)  # deterministic interior cells
    offsets = np.arange(1, 101, dtype=np.int64)
    tau_values = torch.from_numpy(
        offsets.astype(np.float32) * np.float32(data_cfg.original_dt)).unsqueeze(0)
    rows = []
    with torch.no_grad():
        for anchor_index in anchor_indices:
            batch = dataset.get_with_offsets(anchor_index, [1])
            encoded = model.encode_state(**{
                key: batch[key].to(device) for key in (
                    "inlet_velocity", "input_particle_coords", "input_particle_displacement",
                    "supernode_idx", "input_fluid_coords", "input_fluid_state")})
            tau_query = make_pointwise_tau(tau_values, data_cfg.n_cells, device)
            prediction = model.decode_fluid_only(
                encoded, batch["output_fluid_coords"].to(device), tau_query)
            prediction = prediction.cpu().numpy()[0]
            for query_index, offset in enumerate(offsets):
                target = dataset._fluid_state(batch["anchor_frame"] + int(offset))
                for cell in cells:
                    for channel, name in enumerate(CHANNELS):
                        pred_value = prediction[query_index, cell, channel]
                        true_value = target[cell, channel]
                        if channel == 4:
                            pred_value = denormalize_pressure(pred_value, data_cfg)
                            true_value = denormalize_pressure(true_value, data_cfg)
                        rows.append({
                            "anchor_frame": batch["anchor_frame"],
                            "cell_index": cell,
                            "channel": name,
                            "tau_seconds": float(offset * data_cfg.original_dt),
                            "prediction": float(pred_value),
                            "cfd_target": float(true_value),
                        })
    return rows


def rollout_smoke(model, model_cfg, data_cfg, source, device):
    output = {}
    for tau, steps in ((0.1, 5), (0.05, 2)):
        predictions = run_autoregressive(
            model, model_cfg, data_cfg, source, start_frame=0,
            num_steps=steps, device=device, compute_mass_metrics=False,
            tau_test=tau)
        evaluated = evaluate_rollout(
            predictions, source, 0, data_cfg,
            steps=tuple(range(1, steps + 1)), tau_test=tau)
        # This interface smoke intentionally disables the legacy discrete-time
        # residual; continuous spacetime diagnostics are written separately.
        for step_metrics in evaluated.values():
            step_metrics.pop("mass_residual_mae", None)
            step_metrics.pop("mass_residual_rms", None)
        finite = all(
            np.isfinite(value)
            for step_metrics in evaluated.values() for value in step_metrics.values()
        )
        output[str(tau)] = {
            "steps": steps,
            "success": True,
            "finite": bool(finite),
            "metrics": evaluated,
        }
    return output


def main():
    data_cfg, model_cfg = DataConfig(), ModelConfig(model_scale="small")
    train_cfg, ct_cfg = TrainConfig(), ContinuousTimeConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)
    checkpoint = os.path.join(
        ct_cfg.output_dir, "checkpoints", "continuous_time_epoch_100.pt")
    model = load_stage6_model(checkpoint, model_cfg, device)
    dataset = ContinuousTimeCFDDEMDataset(
        data_cfg, model_cfg, DATA_SOURCES[0],
        num_anchor_states=ct_cfg.num_anchor_states,
        queries_per_anchor=ct_cfg.queries_per_anchor,
        min_offset=ct_cfg.min_offset, max_offset=ct_cfg.max_offset,
        seed=train_cfg.seed)
    horizon_rows = evaluate_horizons(model, dataset, data_cfg, ct_cfg, device)
    trajectory_rows = temporal_trajectories(model, dataset, data_cfg, device)
    rollout = rollout_smoke(model, model_cfg, data_cfg, DATA_SOURCES[0], device)
    write_csv(os.path.join(ct_cfg.output_dir, "horizon_accuracy.csv"), horizon_rows)
    write_csv(os.path.join(ct_cfg.output_dir, "temporal_trajectory_samples.csv"), trajectory_rows)
    with open(os.path.join(ct_cfg.output_dir, "rollout_smoke.json"), "w", encoding="utf-8") as handle:
        json.dump(rollout, handle, indent=2)
    print(json.dumps({"horizon_rows": len(horizon_rows),
                      "trajectory_rows": len(trajectory_rows),
                      "rollout": rollout}, indent=2))


if __name__ == "__main__":
    main()
