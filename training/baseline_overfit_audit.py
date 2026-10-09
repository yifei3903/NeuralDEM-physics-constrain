"""Stage 5: long pure-data overfit audit on the fixed 20-pair dataset."""

import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, DataConfig, ModelConfig, TrainConfig
from dataset import CFDDEMDataset
from evaluate_baseline_overfit import (
    FLUID_CHANNELS,
    channel_gradient_audit,
    evaluate_checkpoint,
    normalization_audit,
    write_csv,
)
from inference import build_model
from train import compute_supervised_losses, model_inputs_from_batch, set_seed


MILESTONES = (0, 10, 25, 50, 100, 200, 300)
FD_EPOCHS = (10, 50, 100, 200, 300, 500)
GRADIENT_EPOCHS = (0, 10, 100, 300, 500)


def train_epoch(model, samples, optimizer, device, train_cfg):
    model.train()
    totals = {
        "total_data_loss": 0.0,
        "particle_loss": 0.0,
        "fluid_total_loss": 0.0,
        **{f"{name}_mse": 0.0 for name in FLUID_CHANNELS},
    }
    for batch in samples:
        inputs, _ = model_inputs_from_batch(
            batch, device, require_coordinate_grad=False)
        prediction = model(**inputs)
        parts = compute_supervised_losses(
            prediction, batch, device, train_cfg)
        particle_loss, fluid_loss = parts[:2]
        target = batch["gt_fluid"].to(device)
        channel_losses = [
            F.mse_loss(prediction["fluid"][..., index:index + 1],
                       target[..., index:index + 1])
            for index in range(5)
        ]
        # Preserve the exact existing objective: velocity MSE averages u/v/w.
        reconstructed_fluid_loss = (
            sum(channel_losses[:3]) / 3.0
            + channel_losses[3] + channel_losses[4])
        if not torch.allclose(fluid_loss, reconstructed_fluid_loss,
                              rtol=1e-6, atol=1e-8):
            raise AssertionError("per-channel loss reconstruction mismatch")
        total_loss = (
            train_cfg.loss_particle_weight * particle_loss
            + train_cfg.loss_fluid_weight * fluid_loss)
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite loss {total_loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["total_data_loss"] += total_loss.detach().item()
        totals["particle_loss"] += particle_loss.detach().item()
        totals["fluid_total_loss"] += fluid_loss.detach().item()
        for name, loss in zip(FLUID_CHANNELS, channel_losses):
            totals[f"{name}_mse"] += loss.detach().item()
    count = len(samples)
    output = {name: total / count for name, total in totals.items()}
    for name in FLUID_CHANNELS:
        output[f"{name}_rmse"] = math.sqrt(output[f"{name}_mse"])
    return output


def save_model_checkpoint(path, model, epoch, data_cfg, model_cfg, train_cfg):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "data_config": vars(data_cfg),
        "model_config": vars(model_cfg),
        "train_config": vars(train_cfg),
    }, path)


def plateau_diagnostic(history, end_epoch, window=25, threshold=0.02):
    if end_epoch < 2 * window:
        return {"plateau": False, "reason": "insufficient history"}
    previous = history[-2 * window:-window]
    recent = history[-window:]
    improvements = {}
    for channel in FLUID_CHANNELS:
        key = f"{channel}_mse"
        old = float(np.mean([row[key] for row in previous]))
        new = float(np.mean([row[key] for row in recent]))
        improvements[channel] = (old - new) / max(abs(old), 1e-30)
    plateau = all(value < threshold for value in improvements.values())
    return {
        "plateau": plateau,
        "window_epochs": window,
        "relative_improvement_previous_vs_recent": improvements,
        "criterion": f"all five channel improvements < {threshold:.1%}",
    }


def main():
    data_cfg = DataConfig()
    model_cfg = ModelConfig(model_scale="small")
    train_cfg = TrainConfig(
        debug_overfit=True,
        use_mass_constraint=False,
        lambda_mass=0.0,
    )
    if train_cfg.lambda_mass != 0.0 or train_cfg.use_mass_constraint:
        raise RuntimeError("Stage 5 must remain a pure-data baseline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(os.path.dirname(__file__), "baseline_overfit_results")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = CFDDEMDataset(
        data_cfg, model_cfg,
        sources=[DATA_SOURCES[train_cfg.debug_source_index]])
    samples = [dataset[index] for index in range(train_cfg.debug_num_pairs)]
    pair_ids = [(int(batch["t0"]), int(batch["t1"])) for batch in samples]
    with open(os.path.join(output_dir, "normalization_audit.json"),
              "w", encoding="utf-8") as handle:
        json.dump(normalization_audit(samples, data_cfg), handle, indent=2)

    set_seed(train_cfg.seed)
    model = build_model(model_cfg, device)
    initial_state_path = os.path.join(
        train_cfg.debug_output_dir, "shared_initial_state.pt")
    initial_state = torch.load(
        initial_state_path, map_location=device, weights_only=True)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay)

    print(f"device={device} source={DATA_SOURCES[0]['name']} pairs={pair_ids[0]}..{pair_ids[-1]}", flush=True)
    print(f"initial_state={initial_state_path} lambda_mass=0 lr={train_cfg.learning_rate}", flush=True)

    history = []
    channel_rows = []
    field_rows, fluctuation_rows, flux_rows = [], [], []
    fd_rows, yz_rows = [], []
    gradient_audits = []

    def persist():
        write_csv(os.path.join(output_dir, "training_curve.csv"), history)
        write_csv(os.path.join(output_dir, "per_channel_loss.csv"), channel_rows)
        write_csv(os.path.join(output_dir, "field_accuracy_by_epoch.csv"), field_rows)
        write_csv(os.path.join(output_dir, "fluctuation_accuracy_by_epoch.csv"), fluctuation_rows)
        write_csv(os.path.join(output_dir, "flux_accuracy_by_epoch.csv"), flux_rows)
        write_csv(os.path.join(output_dir, "fd_gradient_accuracy_by_epoch.csv"), fd_rows)
        write_csv(os.path.join(output_dir, "yz_cancellation_by_epoch.csv"), yz_rows)
        with open(os.path.join(output_dir, "channel_gradient_audit.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(gradient_audits, handle, indent=2)

    def audit_checkpoint(epoch, final=False):
        checkpoint_path = os.path.join(
            checkpoint_dir, f"baseline_epoch_{epoch:03d}.pt")
        save_model_checkpoint(
            checkpoint_path, model, epoch, data_cfg, model_cfg, train_cfg)
        include_fd = epoch in FD_EPOCHS or final
        evaluated = evaluate_checkpoint(
            model, samples, epoch, data_cfg, device,
            include_fd_gradients=include_fd)
        fields, fluctuations, fluxes, gradients, cancellations = evaluated
        field_rows.extend(fields)
        fluctuation_rows.extend(fluctuations)
        flux_rows.extend(fluxes)
        fd_rows.extend(gradients)
        yz_rows.extend(cancellations)
        if epoch in GRADIENT_EPOCHS or final:
            gradient_audits.append(channel_gradient_audit(
                model, samples[0], epoch, device))
        persist()
        print(f"checkpoint/evaluation saved at epoch {epoch}", flush=True)

    audit_checkpoint(0)
    started = time.time()
    target_epoch = 300
    epoch = 0
    while epoch < target_epoch:
        epoch += 1
        epoch_started = time.time()
        metrics = train_epoch(model, samples, optimizer, device, train_cfg)
        row = {"epoch": epoch, "epoch_seconds": time.time() - epoch_started, **metrics}
        history.append(row)
        for channel in FLUID_CHANNELS:
            channel_rows.append({
                "epoch": epoch,
                "channel": channel,
                "normalized_mse": metrics[f"{channel}_mse"],
                "normalized_rmse": metrics[f"{channel}_rmse"],
            })
        if epoch % 10 == 0:
            print(
                f"epoch={epoch} sec={row['epoch_seconds']:.2f} total={metrics['total_data_loss']:.6g} "
                + " ".join(f"{name}={metrics[name + '_rmse']:.6g}" for name in FLUID_CHANNELS),
                flush=True)
            persist()
        if epoch in MILESTONES:
            audit_checkpoint(epoch)

    plateau = plateau_diagnostic(history, epoch)
    print("epoch300 plateau diagnostic", plateau, flush=True)
    if not plateau["plateau"]:
        target_epoch = 500
        print("main channel losses are still improving; extending to epoch 500", flush=True)
        while epoch < target_epoch:
            epoch += 1
            epoch_started = time.time()
            metrics = train_epoch(model, samples, optimizer, device, train_cfg)
            row = {"epoch": epoch, "epoch_seconds": time.time() - epoch_started, **metrics}
            history.append(row)
            for channel in FLUID_CHANNELS:
                channel_rows.append({
                    "epoch": epoch, "channel": channel,
                    "normalized_mse": metrics[f"{channel}_mse"],
                    "normalized_rmse": metrics[f"{channel}_rmse"],
                })
            if epoch % 10 == 0:
                print(
                    f"epoch={epoch} sec={row['epoch_seconds']:.2f} total={metrics['total_data_loss']:.6g} "
                    + " ".join(f"{name}={metrics[name + '_rmse']:.6g}" for name in FLUID_CHANNELS),
                    flush=True)
                persist()
            if epoch == 500:
                audit_checkpoint(epoch, final=True)
        plateau = plateau_diagnostic(history, epoch)
    else:
        # Re-run the final audit marker so epoch 300 is explicitly recognized
        # as final for gradient/FD bookkeeping without changing its weights.
        if not any(item["epoch"] == 300 for item in gradient_audits):
            gradient_audits.append(channel_gradient_audit(
                model, samples[0], 300, device))
            persist()

    final_epoch = epoch
    final_summary = {
        "description": "Pure-data long overfit; lambda_mass=0 throughout",
        "source": DATA_SOURCES[0]["name"],
        "pairs": pair_ids,
        "seed": train_cfg.seed,
        "optimizer": "AdamW",
        "learning_rate": train_cfg.learning_rate,
        "weight_decay": train_cfg.weight_decay,
        "initial_state": initial_state_path,
        "final_epoch": final_epoch,
        "elapsed_seconds": time.time() - started,
        "nan_or_inf": False,
        "plateau_diagnostic": plateau,
        "final_training_metrics": history[-1],
    }
    with open(os.path.join(output_dir, "final_summary.json"),
              "w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2)

    markdown = [
        "# Stage 5 baseline overfit audit", "",
        f"- Final epoch: {final_epoch}",
        f"- Elapsed seconds: {final_summary['elapsed_seconds']:.3f}",
        "- lambda_mass: 0", "- NaN/Inf: no", "",
        "Detailed conclusions are based on the adjacent CSV/JSON diagnostics.", "",
    ]
    with open(os.path.join(output_dir, "stage5_summary.md"),
              "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown))
    persist()
    print("FINAL", json.dumps(final_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
