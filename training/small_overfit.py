"""Deterministic 20-pair overfit/continuity scaling experiment.

This diagnostic intentionally does not change the NeuralDEM architecture.  It
reuses one saved initialization for the baseline and every lambda_mass run.
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, DataConfig, ModelConfig, TrainConfig
from dataset import CFDDEMDataset
from inference import build_model
from physics_losses import compute_mass_loss
from train import compute_supervised_losses, model_inputs_from_batch, set_seed


TERM_NAMES = ("temporal", "divergence", "dJx_dx", "dJy_dy", "dJz_dz")


class TensorStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_abs = 0.0
        self.total_square = 0.0
        self.max_abs = 0.0

    def update(self, value):
        value = value.detach().double()
        self.count += value.numel()
        self.total += value.sum().item()
        self.total_abs += value.abs().sum().item()
        self.total_square += value.square().sum().item()
        self.max_abs = max(self.max_abs, value.abs().max().item())

    def result(self):
        mean = self.total / self.count
        mean_square = self.total_square / self.count
        return {
            "mean": mean,
            "mean_abs": self.total_abs / self.count,
            "std": math.sqrt(max(mean_square - mean * mean, 0.0)),
            "rms": math.sqrt(mean_square),
            "max_abs": self.max_abs,
        }


def clone_state_dict(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def data_loss_from_parts(parts, cfg):
    particle, fluid, *_ = parts
    return cfg.loss_particle_weight * particle + cfg.loss_fluid_weight * fluid


def parameter_groups(model):
    return {
        "fluid_decoder": list(model.fluid_decoder.parameters()),
        "fluid_encoder": list(model.to_fluid_token.parameters()),
        "transformer": list(model.blocks.parameters()),
        "particle_decoder": list(model.particle_decoder.parameters()),
        "particle_encoder": list(model.supernode_pooling.parameters()) + list(model.to_particle_token.parameters()),
    }


def group_gradient_norm(loss, groups, retain_graph):
    flat = []
    spans = {}
    for group_name, parameters in groups.items():
        start = len(flat)
        flat.extend(parameters)
        spans[group_name] = (start, len(flat))
    gradients = torch.autograd.grad(
        loss, flat, retain_graph=retain_graph, allow_unused=True)
    output = {}
    for group_name, (start, stop) in spans.items():
        squared = sum(
            gradient.detach().double().square().sum().item()
            for gradient in gradients[start:stop] if gradient is not None
        )
        non_none = sum(gradient is not None for gradient in gradients[start:stop])
        output[group_name] = {"norm": math.sqrt(squared), "tensors_with_grad": non_none,
                              "parameter_tensors": stop - start}
    return output


def gradient_audit(model, batch, device, data_cfg, train_cfg):
    model.train()
    groups = parameter_groups(model)
    inputs, coords = model_inputs_from_batch(batch, device, require_coordinate_grad=True)
    prediction = model(**inputs)
    parts = compute_supervised_losses(prediction, batch, device, train_cfg)
    loss_data = data_loss_from_parts(parts, train_cfg)
    previous = batch["input_fluid_cell_state"].to(device)
    loss_mass, _, _ = compute_mass_loss(
        prediction["fluid"], previous, coords, data_cfg.ml_dt,
        data_cfg.pressure_ref, data_cfg.pressure_scale,
        data_cfg.molecular_weight, data_cfg.gas_constant,
        data_cfg.gas_temperature, create_graph=True, return_terms=True)
    data_norms = group_gradient_norm(loss_data, groups, retain_graph=True)
    mass_norms = group_gradient_norm(loss_mass, groups, retain_graph=False)
    return {
        "loss_data": loss_data.detach().item(),
        "loss_mass_raw": loss_mass.detach().item(),
        "data_gradient_norms": data_norms,
        "mass_gradient_norms": mass_norms,
    }


def evaluate(model, samples, device, data_cfg, train_cfg, lambda_mass):
    model.eval()
    scalar_names = ("particle_loss", "fluid_loss", "velocity_loss",
                    "voidfraction_loss", "pressure_loss", "data_loss")
    scalar_sums = {name: 0.0 for name in scalar_names}
    statistics = {"residual": TensorStats(), **{name: TensorStats() for name in TERM_NAMES}}
    temporal_divergence_product = 0.0

    # Coordinate AD needs grad mode, but create_graph=False avoids second-order
    # graphs and no optimizer/backward is invoked during evaluation.
    with torch.enable_grad():
        for batch in samples:
            inputs, coords = model_inputs_from_batch(batch, device, require_coordinate_grad=True)
            prediction = model(**inputs)
            parts = compute_supervised_losses(prediction, batch, device, train_cfg)
            particle, fluid, velocity, voidfraction, pressure = parts
            data_loss = data_loss_from_parts(parts, train_cfg)
            values = (particle, fluid, velocity, voidfraction, pressure, data_loss)
            for name, value in zip(scalar_names, values):
                scalar_sums[name] += value.detach().item()

            _, residual, terms = compute_mass_loss(
                prediction["fluid"], batch["input_fluid_cell_state"].to(device),
                coords, data_cfg.ml_dt, data_cfg.pressure_ref,
                data_cfg.pressure_scale, data_cfg.molecular_weight,
                data_cfg.gas_constant, data_cfg.gas_temperature,
                create_graph=False, return_terms=True)
            statistics["residual"].update(residual)
            for name in TERM_NAMES:
                statistics[name].update(terms[name])
            temporal_divergence_product += (
                terms["temporal"].detach().double()
                * terms["divergence"].detach().double()
            ).sum().item()

    count = len(samples)
    metrics = {name: total / count for name, total in scalar_sums.items()}
    for name, accumulator in statistics.items():
        for stat_name, value in accumulator.result().items():
            metrics[f"{name}_{stat_name}"] = value
    metrics["mass_loss_raw"] = metrics["residual_rms"] ** 2
    point_count = statistics["temporal"].count
    mean_product = temporal_divergence_product / point_count
    covariance = (mean_product
                  - metrics["temporal_mean"] * metrics["divergence_mean"])
    std_product = metrics["temporal_std"] * metrics["divergence_std"]
    metrics["temporal_divergence_correlation"] = (
        covariance / std_product if std_product > 0 else float("nan"))
    rms_product = metrics["temporal_rms"] * metrics["divergence_rms"]
    metrics["temporal_divergence_cosine"] = (
        mean_product / rms_product if rms_product > 0 else float("nan"))
    metrics["residual_cancellation_ratio"] = (
        metrics["residual_rms"]
        / (metrics["temporal_rms"] + metrics["divergence_rms"]))
    metrics["weighted_mass_loss"] = lambda_mass * metrics["mass_loss_raw"]
    metrics["total_loss"] = metrics["data_loss"] + metrics["weighted_mass_loss"]
    return metrics


def train_epoch(model, samples, optimizer, device, data_cfg, train_cfg, lambda_mass):
    model.train()
    for batch in samples:
        physics_active = lambda_mass != 0.0
        inputs, coords = model_inputs_from_batch(batch, device, physics_active)
        prediction = model(**inputs)
        parts = compute_supervised_losses(prediction, batch, device, train_cfg)
        loss_data = data_loss_from_parts(parts, train_cfg)
        if physics_active:
            loss_mass, _ = compute_mass_loss(
                prediction["fluid"], batch["input_fluid_cell_state"].to(device),
                coords, data_cfg.ml_dt, data_cfg.pressure_ref,
                data_cfg.pressure_scale, data_cfg.molecular_weight,
                data_cfg.gas_constant, data_cfg.gas_temperature,
                create_graph=True)
        else:
            loss_mass = loss_data.new_zeros(())
        total = loss_data + lambda_mass * loss_mass
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite training loss: {total.item()}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_epoch(run_name, row):
    print(
        f"{run_name} epoch={row['epoch']:02d} seconds={row['epoch_seconds']:.2f} "
        f"L_data={row['data_loss']:.6g} L_mass_raw={row['mass_loss_raw']:.6g} "
        f"lambda*L_mass={row['weighted_mass_loss']:.6g} total={row['total_loss']:.6g} "
        f"R_mae={row['residual_mean_abs']:.6g} R_rms={row['residual_rms']:.6g} "
        f"A_rms={row['temporal_rms']:.6g} B_rms={row['divergence_rms']:.6g}"
    )
    for name in TERM_NAMES:
        print(
            f"  {name}: mean={row[name + '_mean']:.6g} "
            f"std={row[name + '_std']:.6g} rms={row[name + '_rms']:.6g} "
            f"maxabs={row[name + '_max_abs']:.6g}"
        )


def run_one(name, lambda_mass, initial_state, samples, model_cfg, data_cfg,
            train_cfg, device, epochs, output_dir):
    set_seed(train_cfg.seed)
    model = build_model(model_cfg, device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay)
    history = []
    run_started = time.time()
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        train_epoch(model, samples, optimizer, device, data_cfg, train_cfg, lambda_mass)
        metrics = evaluate(model, samples, device, data_cfg, train_cfg, lambda_mass)
        row = {"epoch": epoch, "epoch_seconds": time.time() - epoch_started, **metrics}
        history.append(row)
        print_epoch(name, row)
        write_csv(os.path.join(output_dir, f"{name}_history.csv"), history)
    final_audit = gradient_audit(model, samples[0], device, data_cfg, train_cfg)
    with open(os.path.join(output_dir, f"{name}_gradient_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(final_audit, handle, indent=2)
    print(f"{name} FINAL GRADIENT AUDIT", json.dumps(final_audit, indent=2))
    torch.save({"model_state_dict": model.state_dict(), "lambda_mass": lambda_mass,
                "epochs": epochs}, os.path.join(output_dir, f"{name}_final.pt"))
    return history, time.time() - run_started, final_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-pairs", type=int, default=None)
    parser.add_argument("--lambdas", nargs="*", type=float,
                        default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1])
    args = parser.parse_args()

    data_cfg, model_cfg = DataConfig(), ModelConfig(model_scale="small")
    train_cfg = TrainConfig(debug_overfit=True)
    epochs = train_cfg.debug_epochs if args.epochs is None else args.epochs
    num_pairs = train_cfg.debug_num_pairs if args.num_pairs is None else args.num_pairs
    if not train_cfg.debug_overfit:
        raise RuntimeError("small_overfit.py requires debug_overfit=True")
    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(train_cfg.debug_output_dir, exist_ok=True)

    source = DATA_SOURCES[train_cfg.debug_source_index]
    dataset = CFDDEMDataset(data_cfg, model_cfg, sources=[source])
    if num_pairs > len(dataset):
        raise ValueError(f"requested {num_pairs} pairs from only {len(dataset)}")
    # Cache exactly the same consecutive pairs in RAM for every run.
    samples = [dataset[index] for index in range(num_pairs)]
    pair_ids = [(int(sample["t0"]), int(sample["t1"])) for sample in samples]

    set_seed(train_cfg.seed)
    initialization_model = build_model(model_cfg, device)
    initial_state = clone_state_dict(initialization_model)
    initial_path = os.path.join(train_cfg.debug_output_dir, "shared_initial_state.pt")
    torch.save(initial_state, initial_path)
    del initialization_model

    print(f"device={device} source={source['name']} pairs={pair_ids[0]}..{pair_ids[-1]}")
    print(f"seed={train_cfg.seed} epochs={epochs} lr={train_cfg.learning_rate}")
    print(f"shared_initial_state={initial_path}")
    print(f"raw_output_dt={data_cfg.original_dt} frame_interval={data_cfg.frame_interval} ml_dt={data_cfg.ml_dt}")

    audit_model = build_model(model_cfg, device)
    audit_model.load_state_dict(initial_state, strict=True)
    audit = gradient_audit(audit_model, samples[0], device, data_cfg, train_cfg)
    del audit_model
    print("GRADIENT AUDIT", json.dumps(audit, indent=2))
    with open(os.path.join(train_cfg.debug_output_dir, "gradient_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    summary = []
    for lambda_mass in args.lambdas:
        name = "baseline" if lambda_mass == 0 else f"physics_{lambda_mass:.0e}"
        history, elapsed, final_audit = run_one(
            name, lambda_mass, initial_state, samples, model_cfg, data_cfg,
            train_cfg, device, epochs, train_cfg.debug_output_dir)
        final = history[-1]
        summary.append({"model": name, "lambda_mass": lambda_mass,
                        "epochs": epochs, "training_seconds": elapsed,
                        "nan_or_inf": False, **final,
                        "final_gradient_audit_file": f"{name}_gradient_audit.json"})
        with open(os.path.join(train_cfg.debug_output_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    write_csv(os.path.join(train_cfg.debug_output_dir, "summary.csv"), summary)
    print("FINAL SUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
