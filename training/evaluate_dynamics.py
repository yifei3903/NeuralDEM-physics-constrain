"""One-step Stage-7 local-dynamics evaluation."""

import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, DataConfig, DynamicsConfig, ModelConfig, TrainConfig
from dataset import LocalDynamicsDataset, denormalize_pressure
from dynamics_utils import dynamics_model_inputs, euler_integrate_fluid
from evaluate_continuous_time import metrics
from inference import build_model
from train import set_seed


FIELDS = ("u", "v", "w", "epsilon", "pressure")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_dynamics_model(path, model_cfg, device):
    model = build_model(model_cfg, device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


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
    predicted = {field: [] for field in FIELDS}
    identity_predicted = {field: [] for field in FIELDS}
    target = {field: [] for field in FIELDS}
    particle_predicted, particle_target = [], []
    with torch.no_grad():
        for block_index in range(len(dataset)):
            batch = dataset.get_transition(
                block_index, dynamics_cfg.evaluation_offset)
            output = model.predict_dynamics(
                **dynamics_model_inputs(batch, device),
                particle_query_tau=dynamics_cfg.dt_dyn)
            next_fluid = euler_integrate_fluid(
                batch["input_fluid_cell_state"].to(device),
                output["fluid_rate"], dynamics_cfg.dt_dyn, data_cfg)
            pred_np = next_fluid.cpu().numpy()[0]
            target_np = batch["gt_fluid"].numpy()[0]
            current_np = batch["input_fluid_cell_state"].numpy()[0]
            for channel, field in enumerate(FIELDS):
                pred_values, target_values = pred_np[:, channel], target_np[:, channel]
                identity_values = current_np[:, channel]
                if field == "pressure":
                    pred_values = denormalize_pressure(pred_values, data_cfg)
                    target_values = denormalize_pressure(target_values, data_cfg)
                    identity_values = denormalize_pressure(identity_values, data_cfg)
                predicted[field].append(pred_values)
                identity_predicted[field].append(identity_values)
                target[field].append(target_values)
            particle_predicted.append(output["particle_displacement"].cpu().numpy())
            particle_target.append(batch["gt_particle_displacement"].numpy())

    rows = []
    for field in FIELDS:
        row = {"predictor": "learned_dynamics", "dt_seconds": dynamics_cfg.dt_dyn, "field": field}
        row.update(metrics(
            np.concatenate(predicted[field]), np.concatenate(target[field])))
        rows.append(row)
        identity_row = {"predictor": "zero_rate_identity", "dt_seconds": dynamics_cfg.dt_dyn, "field": field}
        identity_row.update(metrics(
            np.concatenate(identity_predicted[field]), np.concatenate(target[field])))
        rows.append(identity_row)
    row = {"predictor": "learned_dynamics", "dt_seconds": dynamics_cfg.dt_dyn, "field": "particle_displacement"}
    row.update(metrics(
        np.concatenate(particle_predicted), np.concatenate(particle_target)))
    rows.append(row)
    zero_particle = np.zeros_like(np.concatenate(particle_target))
    row = {"predictor": "zero_rate_identity", "dt_seconds": dynamics_cfg.dt_dyn, "field": "particle_displacement"}
    row.update(metrics(zero_particle, np.concatenate(particle_target)))
    rows.append(row)
    write_csv(os.path.join(dynamics_cfg.output_dir, "one_step_accuracy.csv"), rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
