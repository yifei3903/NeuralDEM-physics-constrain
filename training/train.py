"""Training entry point for Reduced NeuralDEM with compressible mass loss."""

import os
import random
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, DataConfig, ModelConfig, TrainConfig
from dataset import CFDDEMDataset, collate_fn
from inference import build_model, evaluate_rollout, run_autoregressive
from physics_losses import compute_mass_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_inputs_from_batch(batch, device, require_coordinate_grad=False):
    output_coords = batch["output_fluid_coords"].to(device).detach().clone()
    output_coords.requires_grad_(require_coordinate_grad)
    return {
        "inlet_velocity": batch["inlet_velocity"].to(device),
        "input_particle_coords": batch["input_particle_coords"].to(device),
        "input_particle_displacement": batch["input_particle_displacement"].to(device),
        "supernode_idx": batch["supernode_idx"].to(device),
        "input_fluid_coords": batch["input_fluid_coords"].to(device),
        "input_fluid_state": batch["input_fluid_state"].to(device),
        "output_fluid_coords": output_coords,
        "fluid_voidfraction_min": batch["fluid_voidfraction_min"].to(device),
        "num_output_particles": batch["num_output_particles"].to(device),
    }, output_coords


def compute_supervised_losses(predictions, batch, device, train_cfg):
    pred_coords = predictions["particle_coords"].squeeze(0)
    gt_coords = batch["input_particle_coords"].to(device)
    gt_displacement = batch["gt_particle_displacement"].to(device)
    if len(pred_coords) and len(gt_coords):
        nearest = torch.cdist(pred_coords.float(), gt_coords.float()).argmin(dim=1)
        loss_particle = F.mse_loss(predictions["particle_displacement"], gt_displacement[nearest])
    else:
        loss_particle = predictions["fluid"].new_zeros(())

    target = batch["gt_fluid"].to(device)
    loss_velocity = F.mse_loss(predictions["fluid"][..., :3], target[..., :3])
    loss_voidfraction = F.mse_loss(predictions["fluid"][..., 3:4], target[..., 3:4])
    # Pressure MSE is deliberately evaluated in normalized P* units.
    loss_pressure = F.mse_loss(predictions["fluid"][..., 4:5], target[..., 4:5])
    loss_fluid = (
        train_cfg.loss_velocity_weight * loss_velocity
        + train_cfg.loss_voidfraction_weight * loss_voidfraction
        + train_cfg.loss_pressure_weight * loss_pressure
    )
    return loss_particle, loss_fluid, loss_velocity, loss_voidfraction, loss_pressure


def train_one_batch(model, batch, optimizer, device, data_cfg, train_cfg):
    physics_active = train_cfg.use_mass_constraint and train_cfg.lambda_mass != 0.0
    inputs, output_coords = model_inputs_from_batch(batch, device, physics_active)
    previous_fluid = batch["input_fluid_cell_state"].to(device)

    optimizer.zero_grad(set_to_none=True)
    predictions = model(**inputs)
    losses = compute_supervised_losses(predictions, batch, device, train_cfg)
    loss_particle, loss_fluid, loss_velocity, loss_voidfraction, loss_pressure = losses

    if physics_active:
        loss_mass, residual = compute_mass_loss(
            predictions["fluid"], previous_fluid, output_coords, data_cfg.ml_dt,
            data_cfg.pressure_ref, data_cfg.pressure_scale,
            data_cfg.molecular_weight, data_cfg.gas_constant,
            data_cfg.gas_temperature, create_graph=True,
        )
        residual_rms = residual.square().mean().sqrt()
    else:
        # This branch makes lambda_mass=0 exactly the supervised baseline path.
        loss_mass = predictions["fluid"].new_zeros(())
        residual_rms = predictions["fluid"].new_zeros(())

    loss_total = (
        train_cfg.loss_particle_weight * loss_particle
        + train_cfg.loss_fluid_weight * loss_fluid
        + train_cfg.lambda_mass * loss_mass
    )
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "total_loss": loss_total.detach().item(),
        "particle_loss": loss_particle.detach().item(),
        "fluid_loss": loss_fluid.detach().item(),
        "velocity_loss": loss_velocity.detach().item(),
        "voidfraction_loss": loss_voidfraction.detach().item(),
        "pressure_loss": loss_pressure.detach().item(),
        "mass_loss": loss_mass.detach().item(),
        "mass_residual_rms": residual_rms.detach().item(),
    }, predictions


def train_one_epoch(model, dataloader, optimizer, device, data_cfg, train_cfg):
    model.train()
    totals = {}
    for index, batch in enumerate(dataloader, start=1):
        metrics, _ = train_one_batch(model, batch, optimizer, device, data_cfg, train_cfg)
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
        if index % 20 == 0:
            print("  batch", index, " ".join(f"{k}={v:.6g}" for k, v in metrics.items()))
    return {name: value / len(dataloader) for name, value in totals.items()}


def main():
    data_cfg, model_cfg, train_cfg = DataConfig(), ModelConfig(), TrainConfig()
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")

    dataset = CFDDEMDataset(data_cfg, model_cfg)
    generator = torch.Generator().manual_seed(train_cfg.seed)
    loader = DataLoader(
        dataset, batch_size=train_cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, generator=generator,
    )
    model = build_model(model_cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.num_epochs)
    os.makedirs(train_cfg.save_dir, exist_ok=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} samples={len(dataset)} parameters={parameter_count:,}")
    print(f"model_scale={model_cfg.model_scale} dim={model_cfg.dim} depth={model_cfg.depth} heads={model_cfg.num_heads}")
    print(f"mass_constraint={train_cfg.use_mass_constraint} lambda_mass={train_cfg.lambda_mass}")
    print(f"raw CFD output dt={data_cfg.original_dt:.9g} s")
    print(f"frame interval={data_cfg.frame_interval} raw output files per ML step")
    print(f"actual ML dt used in continuity residual={data_cfg.ml_dt:.9g} s")

    for epoch in range(1, train_cfg.num_epochs + 1):
        started = time.time()
        metrics = train_one_epoch(model, loader, optimizer, device, data_cfg, train_cfg)
        scheduler.step()
        print(f"epoch={epoch} seconds={time.time() - started:.1f} "
              + " ".join(f"{key}={value:.6g}" for key, value in metrics.items()))

        if epoch % train_cfg.ar_test_every == 0:
            rollout = run_autoregressive(
                model, model_cfg, data_cfg, DATA_SOURCES[0], start_frame=5000,
                num_steps=train_cfg.ar_test_steps, device=device,
                compute_mass_metrics=True,
            )
            print("rollout", evaluate_rollout(
                rollout, DATA_SOURCES[0], 5000, data_cfg, train_cfg.metric_steps))
            model.train()

        if epoch % train_cfg.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "data_config": asdict(data_cfg),
                "model_config": asdict(model_cfg),
                "train_config": asdict(train_cfg),
            }, os.path.join(train_cfg.save_dir, f"checkpoint_epoch{epoch}.th"))


if __name__ == "__main__":
    main()
