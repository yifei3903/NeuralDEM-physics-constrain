"""Stage-6 data-only continuous relative-time baseline training."""

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

from config import DATA_SOURCES, ContinuousTimeConfig, DataConfig, ModelConfig, TrainConfig
from continuous_time_utils import (
    FLUID_CHANNELS,
    compute_multi_horizon_losses,
    continuous_model_inputs,
    load_legacy_initial_state,
    make_pointwise_tau,
)
from dataset import ContinuousTimeCFDDEMDataset
from inference import build_model
from train import set_seed


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, model, optimizer, epoch, data_cfg, model_cfg, train_cfg, ct_cfg):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "data_config": asdict(data_cfg),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "continuous_time_config": asdict(ct_cfg),
    }, path)


def main():
    data_cfg, model_cfg = DataConfig(), ModelConfig(model_scale="small")
    train_cfg = TrainConfig(use_mass_constraint=False, lambda_mass=0.0)
    ct_cfg = ContinuousTimeConfig()
    if train_cfg.lambda_mass != 0.0 or train_cfg.use_mass_constraint:
        raise RuntimeError("Stage 6 baseline must remain data-only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)
    dataset = ContinuousTimeCFDDEMDataset(
        data_cfg, model_cfg, DATA_SOURCES[0],
        num_anchor_states=ct_cfg.num_anchor_states,
        queries_per_anchor=ct_cfg.queries_per_anchor,
        min_offset=ct_cfg.min_offset, max_offset=ct_cfg.max_offset,
        seed=train_cfg.seed,
    )
    model = build_model(model_cfg, device)
    initial_path = os.path.join(train_cfg.debug_output_dir, "shared_initial_state.pt")
    load_legacy_initial_state(model, initial_path, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay)

    output_dir = ct_cfg.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    history = []
    order_rng = np.random.RandomState(train_cfg.seed)
    started = time.time()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} anchors={len(dataset)} K={ct_cfg.queries_per_anchor} "
        f"parameters={parameter_count:,} lambda_mass=0", flush=True)

    for epoch in range(1, ct_cfg.train_epochs + 1):
        model.train()
        epoch_started = time.time()
        totals = {}
        sampled_taus = []
        for index in order_rng.permutation(len(dataset)):
            batch = dataset[int(index)]
            tau_query = make_pointwise_tau(
                batch["query_tau_values"], data_cfg.n_cells, device)
            prediction = model(**continuous_model_inputs(batch, device, tau_query))
            losses = compute_multi_horizon_losses(
                prediction, batch, device, train_cfg)
            if not torch.isfinite(losses["total_data_loss"]):
                raise FloatingPointError("non-finite continuous-time data loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total_data_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item()
            sampled_taus.extend(batch["query_tau_values"].reshape(-1).tolist())

        count = len(dataset)
        means = {name: value / count for name, value in totals.items()}
        row = {
            "epoch": epoch,
            "epoch_seconds": time.time() - epoch_started,
            **means,
            **{f"{name}_rmse": math.sqrt(means[f"{name}_mse"])
               for name in FLUID_CHANNELS},
            "sampled_tau_min": min(sampled_taus),
            "sampled_tau_max": max(sampled_taus),
            "sampled_tau_mean": float(np.mean(sampled_taus)),
        }
        history.append(row)
        write_csv(os.path.join(output_dir, "training_curve.csv"), history)
        print(
            f"epoch={epoch} sec={row['epoch_seconds']:.2f} "
            f"total={row['total_data_loss']:.6g} "
            + " ".join(f"{name}={row[name + '_rmse']:.6g}" for name in FLUID_CHANNELS)
            + f" tau=[{row['sampled_tau_min']:.3f},{row['sampled_tau_max']:.3f}]",
            flush=True,
        )
        if epoch in (ct_cfg.smoke_epochs, ct_cfg.train_epochs):
            save_checkpoint(
                os.path.join(checkpoint_dir, f"continuous_time_epoch_{epoch:03d}.pt"),
                model, optimizer, epoch, data_cfg, model_cfg, train_cfg, ct_cfg)

    summary = {
        "description": "Stage-6 data-only continuous relative-time baseline",
        "source": DATA_SOURCES[0]["name"],
        "anchor_frames": dataset.anchor_frames,
        "tau_definition": "tau=(target_frame-anchor_frame)*0.001 s",
        "tau_range_seconds": [ct_cfg.min_offset * data_cfg.original_dt,
                              ct_cfg.max_offset * data_cfg.original_dt],
        "queries_per_anchor": ct_cfg.queries_per_anchor,
        "epochs": ct_cfg.train_epochs,
        "optimizer": "AdamW",
        "learning_rate": train_cfg.learning_rate,
        "weight_decay": train_cfg.weight_decay,
        "seed": train_cfg.seed,
        "lambda_mass": 0.0,
        "parameter_count": parameter_count,
        "initial_state": initial_path,
        "elapsed_seconds": time.time() - started,
        "nan_or_inf": False,
        "final_training_metrics": history[-1],
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("FINAL", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
