"""Stage-7 learned local dynamics baseline training."""

import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, DataConfig, DynamicsConfig, ModelConfig, TrainConfig
from dataset import LocalDynamicsDataset
from dynamics_utils import (
    FLUID_CHANNELS, RATE_CHANNELS,
    compute_dynamics_losses, dynamics_model_inputs, euler_integrate_fluid,
    invalid_state_counts, load_shared_initial_state,
)
from inference import build_model
from train import set_seed


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, model, optimizer, epoch, data_cfg, model_cfg, train_cfg, dynamics_cfg):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "data_config": asdict(data_cfg),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "dynamics_config": asdict(dynamics_cfg),
    }, path)


def main():
    data_cfg, model_cfg = DataConfig(), ModelConfig(model_scale="small")
    train_cfg = TrainConfig(use_mass_constraint=False, lambda_mass=0.0)
    dynamics_cfg = DynamicsConfig()
    if train_cfg.lambda_mass != 0.0 or train_cfg.use_mass_constraint:
        raise RuntimeError("Stage 7 must remain a data-only learned-dynamics baseline")
    if abs(dynamics_cfg.dt_dyn - data_cfg.original_dt) > 1e-12:
        raise RuntimeError("Stage-7 Euler dt must equal raw CFD output spacing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)
    dataset = LocalDynamicsDataset(
        data_cfg, model_cfg, DATA_SOURCES[0],
        dynamics_cfg.num_blocks, dynamics_cfg.block_size_frames,
        train_cfg.seed)
    model = build_model(model_cfg, device)
    initial_path = os.path.join(train_cfg.debug_output_dir, "shared_initial_state.pt")
    load_report = load_shared_initial_state(model, initial_path, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay)
    output_dir = dynamics_cfg.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    history = []
    order_rng = np.random.RandomState(train_cfg.seed)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.time()
    total_invalid = {name: 0 for name in (
        "epsilon_below_zero", "epsilon_above_one", "pressure_nonpositive", "nan", "inf")}
    extrema = {
        "epsilon_min": float("inf"), "epsilon_max": -float("inf"),
        "pressure_min_pa": float("inf"), "pressure_max_pa": -float("inf"),
    }
    print(
        f"device={device} blocks={len(dataset)} dt={dynamics_cfg.dt_dyn} "
        f"parameters={parameter_count:,} lambda_mass=0", flush=True)

    for epoch in range(1, dynamics_cfg.train_epochs + 1):
        model.train()
        epoch_started = time.time()
        totals = {}
        rate_squares = {name: 0.0 for name in RATE_CHANNELS}
        sampled_frames = []
        for index in order_rng.permutation(len(dataset)):
            batch = dataset[int(index)]
            prediction = model.predict_dynamics(
                **dynamics_model_inputs(batch, device),
                particle_query_tau=dynamics_cfg.dt_dyn)
            current = batch["input_fluid_cell_state"].to(device)
            predicted_fluid = euler_integrate_fluid(
                current, prediction["fluid_rate"], dynamics_cfg.dt_dyn, data_cfg)
            losses = compute_dynamics_losses(
                prediction, batch, predicted_fluid, device, train_cfg)
            if not torch.isfinite(losses["total_data_loss"]):
                raise FloatingPointError("non-finite Stage-7 training loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total_data_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item()
            for channel, name in enumerate(RATE_CHANNELS):
                rate_squares[name] += prediction["fluid_rate"][..., channel].detach().square().mean().item()
            invalid = invalid_state_counts(predicted_fluid.detach(), data_cfg)
            for name in total_invalid:
                total_invalid[name] += invalid[name]
            extrema["epsilon_min"] = min(extrema["epsilon_min"], invalid["epsilon_min"])
            extrema["epsilon_max"] = max(extrema["epsilon_max"], invalid["epsilon_max"])
            extrema["pressure_min_pa"] = min(extrema["pressure_min_pa"], invalid["pressure_min_pa"])
            extrema["pressure_max_pa"] = max(extrema["pressure_max_pa"], invalid["pressure_max_pa"])
            sampled_frames.append(int(batch["current_frame"]))

        count = len(dataset)
        means = {name: value / count for name, value in totals.items()}
        row = {
            "epoch": epoch,
            "epoch_seconds": time.time() - epoch_started,
            **means,
            **{f"{name}_rmse": math.sqrt(means[f"{name}_mse"])
               for name in FLUID_CHANNELS},
            **{f"{name}_rms": math.sqrt(value / count)
               for name, value in rate_squares.items()},
            "sampled_frame_min": min(sampled_frames),
            "sampled_frame_max": max(sampled_frames),
        }
        history.append(row)
        write_csv(os.path.join(output_dir, "training_curve.csv"), history)
        if epoch % 10 == 0 or epoch <= 5:
            print(
                f"epoch={epoch} sec={row['epoch_seconds']:.2f} "
                f"total={row['total_data_loss']:.6g} "
                + " ".join(f"{name}={row[name + '_rmse']:.6g}" for name in FLUID_CHANNELS)
                + f" dmstar_dt={row['dmstar_dt_rms']:.6g}",
                flush=True)
        if epoch in dynamics_cfg.checkpoint_epochs:
            save_checkpoint(
                os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"),
                model, optimizer, epoch, data_cfg, model_cfg, train_cfg, dynamics_cfg)

    invalid_summary = {
        "scope": "all Stage-7 one-step training predictions",
        "counts": total_invalid,
        "extrema": extrema,
    }
    with open(os.path.join(output_dir, "invalid_state_statistics.json"),
              "w", encoding="utf-8") as handle:
        json.dump(invalid_summary, handle, indent=2)
    summary = {
        "description": "Stage-7 learned local dynamics, data-only Euler supervision",
        "source": DATA_SOURCES[0]["name"],
        "block_starts": dataset.block_starts,
        "dt_dyn_seconds": dynamics_cfg.dt_dyn,
        "epochs": dynamics_cfg.train_epochs,
        "optimizer_updates": dynamics_cfg.train_epochs * len(dataset),
        "seed": train_cfg.seed,
        "optimizer": "AdamW",
        "learning_rate": train_cfg.learning_rate,
        "weight_decay": train_cfg.weight_decay,
        "lambda_mass": 0.0,
        "parameter_count": parameter_count,
        "initial_state": initial_path,
        "load_report": load_report,
        "elapsed_seconds": time.time() - started,
        "nan_or_inf": False,
        "final_training_metrics": history[-1],
    }
    with open(os.path.join(output_dir, "training_summary.json"),
              "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("FINAL", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
