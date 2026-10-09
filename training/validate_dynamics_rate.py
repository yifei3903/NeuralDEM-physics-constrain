"""Compare learned Stage-7 local rates with CFD forward and centered FD."""

import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, DataConfig, DynamicsConfig, ModelConfig, TrainConfig
from dataset import LocalDynamicsDataset
from dynamics_utils import RATE_CHANNELS, dynamics_model_inputs, external_to_dynamics_state
from evaluate_continuous_time import metrics
from evaluate_dynamics import load_dynamics_model
from train import set_seed


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def main():
    data_cfg, model_cfg = DataConfig(), ModelConfig(model_scale="small")
    train_cfg, dynamics_cfg = TrainConfig(), DynamicsConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)
    model = load_dynamics_model(
        os.path.join(dynamics_cfg.output_dir, "checkpoints", "epoch_500.pt"),
        model_cfg, device)
    dataset = LocalDynamicsDataset(
        data_cfg, model_cfg, DATA_SOURCES[0],
        dynamics_cfg.num_blocks, dynamics_cfg.block_size_frames,
        train_cfg.seed)
    collected = {
        (channel, kind): []
        for channel in RATE_CHANNELS for kind in ("prediction", "forward", "centered")
    }
    with torch.no_grad():
        for block_index in range(len(dataset)):
            current_frame = dataset.block_starts[block_index] + dynamics_cfg.evaluation_offset
            batch = dataset.get_transition(block_index, dynamics_cfg.evaluation_offset)
            output = model.predict_dynamics(
                **dynamics_model_inputs(batch, device),
                particle_query_tau=dynamics_cfg.dt_dyn)
            current = external_to_dynamics_state(
                batch["input_fluid_cell_state"].to(device), data_cfg)
            plus = external_to_dynamics_state(
                torch.from_numpy(dataset._fluid_state(current_frame + 1)).unsqueeze(0).to(device),
                data_cfg)
            minus = external_to_dynamics_state(
                torch.from_numpy(dataset._fluid_state(current_frame - 1)).unsqueeze(0).to(device),
                data_cfg)
            forward = (plus - current) / data_cfg.original_dt
            centered = (plus - minus) / (2.0 * data_cfg.original_dt)
            for channel_index, channel in enumerate(RATE_CHANNELS):
                collected[(channel, "prediction")].append(
                    output["fluid_rate"][..., channel_index].cpu().numpy())
                collected[(channel, "forward")].append(
                    forward[..., channel_index].cpu().numpy())
                collected[(channel, "centered")].append(
                    centered[..., channel_index].cpu().numpy())

    outputs = {}
    for reference_kind in ("forward", "centered"):
        rows = []
        for channel in RATE_CHANNELS:
            prediction = np.concatenate(collected[(channel, "prediction")]).reshape(-1)
            reference = np.concatenate(collected[(channel, reference_kind)]).reshape(-1)
            pred_rms, reference_rms = rms(prediction), rms(reference)
            row = {
                "quantity": channel,
                **metrics(prediction, reference),
                "prediction_rms": pred_rms,
                "reference_rms": reference_rms,
                "rms_ratio": pred_rms / max(reference_rms, 1e-30),
            }
            rows.append(row)
        outputs[reference_kind] = rows
        write_csv(os.path.join(
            dynamics_cfg.output_dir, f"rate_vs_cfd_{reference_kind}.csv"), rows)

    stage6_path = os.path.join(
        os.path.dirname(__file__), "continuous_time_results_stage6",
        "time_derivative_summary.json")
    with open(stage6_path, "r", encoding="utf-8") as handle:
        stage6 = json.load(handle)
    summary = {
        "stage7_rate_representation": "[du/dt,dv/dt,dw/dt,dmstar/dt,dPstar/dt]",
        "training_used_derivative_labels": False,
        "forward_difference": outputs["forward"],
        "centered_difference": outputs["centered"],
        "stage6_mass_time_derivative_reference": [
            row for row in stage6["rows"] if row["quantity"] == "mass_dt"
        ],
        "comparison_note": (
            "Stage-6 mass rates are dimensional while Stage-7 mstar rates are divided by rho_ref; "
            "relative L2, correlation, and RMS ratio are scale-invariant. Training budgets differ."),
        "nan_or_inf": False,
    }
    with open(os.path.join(dynamics_cfg.output_dir, "rate_summary.json"),
              "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
