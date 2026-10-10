"""Controlled Stage-6B continuous spacetime mass-conservation training."""

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

from config import (
    DATA_SOURCES, ContinuousTimeConfig, DataConfig, ModelConfig,
    Stage6BConfig, TrainConfig,
)
from continuous_time_utils import (
    compute_multi_horizon_losses, continuous_model_inputs,
    load_legacy_initial_state, make_pointwise_tau, make_spacetime_coords,
)
from dataset import ContinuousTimeCFDDEMDataset
from inference import build_model
from stage6b_utils import (
    TERM_NAMES, collect_detached, continuity_terms,
    detached_epoch_diagnostics,
)
from train import set_seed


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, model, optimizer, epoch, configs):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        **{name: asdict(value) for name, value in configs.items()},
    }, path)


def write_config(path, data_cfg, model_cfg, train_cfg, ct_cfg, physics_cfg,
                 initial_path):
    lines = [
        "stage: 6B",
        "description: continuous spacetime mass-conservation training",
        f"lambda_mass: {physics_cfg.lambda_mass}",
        f"epochs: {physics_cfg.train_epochs}",
        f"seed: {train_cfg.seed}",
        "optimizer: AdamW",
        f"learning_rate: {train_cfg.learning_rate}",
        f"weight_decay: {train_cfg.weight_decay}",
        f"queries_per_anchor: {ct_cfg.queries_per_anchor}",
        f"num_anchor_states: {ct_cfg.num_anchor_states}",
        f"tau_min_seconds: {ct_cfg.min_offset * data_cfg.original_dt}",
        f"tau_max_seconds: {ct_cfg.max_offset * data_cfg.original_dt}",
        f"model_dim: {model_cfg.dim}",
        f"model_depth: {model_cfg.depth}",
        f"model_heads: {model_cfg.num_heads}",
        f"initial_state: '{initial_path}'",
        "time_derivative: pointwise autograd with create_graph=true",
        "spatial_derivative: pointwise physical-coordinate autograd",
        "mass_equation: d(epsilon*rho)/dtau + div(epsilon*rho*u)",
        "pressure_denormalization: P=P_star*1000+101325 Pa",
        "derivative_label_loss: false",
        "momentum_loss: false",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    data_cfg = DataConfig()
    model_cfg = ModelConfig(model_scale="small")
    train_cfg = TrainConfig(use_mass_constraint=True, lambda_mass=1e-3)
    ct_cfg, physics_cfg = ContinuousTimeConfig(), Stage6BConfig()
    if physics_cfg.lambda_mass != 1e-3 or train_cfg.lambda_mass != 1e-3:
        raise RuntimeError("Stage-6B is fixed at lambda_mass=1e-3")
    if physics_cfg.train_epochs != ct_cfg.train_epochs:
        raise RuntimeError("Stage-6B must retain the Stage-6 100-epoch budget")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)
    dataset = ContinuousTimeCFDDEMDataset(
        data_cfg, model_cfg, DATA_SOURCES[0],
        num_anchor_states=ct_cfg.num_anchor_states,
        queries_per_anchor=ct_cfg.queries_per_anchor,
        min_offset=ct_cfg.min_offset, max_offset=ct_cfg.max_offset,
        seed=train_cfg.seed)
    model = build_model(model_cfg, device)
    initial_path = os.path.join(
        train_cfg.debug_output_dir, "shared_initial_state.pt")
    load_legacy_initial_state(model, initial_path, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay)

    output_dir = physics_cfg.output_dir
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    write_config(
        os.path.join(output_dir, "stage6b_config.yaml"),
        data_cfg, model_cfg, train_cfg, ct_cfg, physics_cfg, initial_path)

    history = []
    order_rng = np.random.RandomState(train_cfg.seed)
    started = time.time()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} anchors={len(dataset)} K={ct_cfg.queries_per_anchor} "
        f"parameters={parameter_count:,} lambda_mass={physics_cfg.lambda_mass}",
        flush=True)

    for epoch in range(1, physics_cfg.train_epochs + 1):
        model.train()
        epoch_started = time.time()
        totals = {}
        sampled_taus = []
        collected = {name: [] for name in TERM_NAMES}
        for index in order_rng.permutation(len(dataset)):
            batch = dataset[int(index)]
            tau_query = make_pointwise_tau(
                batch["query_tau_values"], data_cfg.n_cells, device,
                requires_grad=True)
            coords = make_spacetime_coords(
                batch["output_fluid_coords"], ct_cfg.queries_per_anchor,
                device, requires_grad=True)
            prediction = model(**continuous_model_inputs(
                batch, device, tau_query, output_coords=coords))
            data_losses = compute_multi_horizon_losses(
                prediction, batch, device, train_cfg)
            terms = continuity_terms(
                prediction["fluid"], coords, tau_query, data_cfg,
                create_graph=True)
            mass_loss = terms["R"].square().mean()
            weighted_mass = physics_cfg.lambda_mass * mass_loss
            total_loss = data_losses["total_data_loss"] + weighted_mass
            if not all(bool(torch.isfinite(value)) for value in (
                    data_losses["total_data_loss"], mass_loss,
                    weighted_mass, total_loss)):
                raise FloatingPointError("non-finite Stage-6B loss")

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                raise FloatingPointError("non-finite Stage-6B parameter gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_values = {
                "L_total": total_loss,
                "L_data": data_losses["total_data_loss"],
                "L_particle": data_losses["particle_loss"],
                "L_fluid": data_losses["fluid_total_loss"],
                "L_u": data_losses["u_mse"],
                "L_v": data_losses["v_mse"],
                "L_w": data_losses["w_mse"],
                "L_epsilon": data_losses["epsilon_mse"],
                "L_pressure": data_losses["pressure_mse"],
                "L_mass_raw": mass_loss,
                "L_mass_weighted": weighted_mass,
            }
            for name, value in batch_values.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item()
            collect_detached(collected, terms)
            sampled_taus.extend(
                batch["query_tau_values"].reshape(-1).tolist())

        count = len(dataset)
        means = {name: value / count for name, value in totals.items()}
        diagnostics = detached_epoch_diagnostics(collected)
        row = {
            "epoch": epoch,
            "epoch_seconds": time.time() - epoch_started,
            **means,
            **diagnostics,
            "weighted_mass_fraction_of_total": (
                means["L_mass_weighted"] / max(means["L_total"], 1e-30)),
            "sampled_tau_min": min(sampled_taus),
            "sampled_tau_max": max(sampled_taus),
            "sampled_tau_mean": float(np.mean(sampled_taus)),
        }
        history.append(row)
        write_csv(
            os.path.join(output_dir, "stage6b_training_history.csv"), history)
        print(
            f"epoch={epoch} sec={row['epoch_seconds']:.2f} "
            f"total={row['L_total']:.6g} data={row['L_data']:.6g} "
            f"mass={row['L_mass_raw']:.6g} weighted={row['L_mass_weighted']:.6g} "
            f"RMS_A={row['RMS_A']:.5g} RMS_B={row['RMS_B']:.5g} "
            f"RMS_R={row['RMS_R']:.5g}", flush=True)

        if epoch in physics_cfg.checkpoint_epochs:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(checkpoint_path, model, optimizer, epoch, {
                "data_config": data_cfg,
                "model_config": model_cfg,
                "train_config": train_cfg,
                "continuous_time_config": ct_cfg,
                "stage6b_config": physics_cfg,
            })

    final_path = os.path.join(checkpoint_dir, "final.pt")
    save_checkpoint(final_path, model, optimizer, physics_cfg.train_epochs, {
        "data_config": data_cfg,
        "model_config": model_cfg,
        "train_config": train_cfg,
        "continuous_time_config": ct_cfg,
        "stage6b_config": physics_cfg,
    })
    summary = {
        "description": "Stage-6B continuous spacetime mass-conservation training",
        "source": DATA_SOURCES[0]["name"],
        "anchor_frames": dataset.anchor_frames,
        "epochs": physics_cfg.train_epochs,
        "optimizer_updates": physics_cfg.train_epochs * len(dataset),
        "lambda_mass": physics_cfg.lambda_mass,
        "seed": train_cfg.seed,
        "optimizer": "AdamW",
        "learning_rate": train_cfg.learning_rate,
        "weight_decay": train_cfg.weight_decay,
        "queries_per_anchor": ct_cfg.queries_per_anchor,
        "parameter_count": parameter_count,
        "initial_state": initial_path,
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
