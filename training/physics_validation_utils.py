"""Shared utilities for post-training spatial/continuity validation."""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from config import DATA_SOURCES, DataConfig, ModelConfig, TrainConfig
from dataset import CFDDEMDataset
from inference import build_model
from physics_losses import denormalize_pressure, gas_density_from_pressure
from train import model_inputs_from_batch, set_seed


CHECKPOINTS = {
    "baseline": ("baseline_final.pt", 0.0),
    "physics_1e-03": ("physics_1e-03_final.pt", 1e-3),
    "physics_1e-02": ("physics_1e-02_final.pt", 1e-2),
}


def validation_setup():
    data_cfg = DataConfig()
    model_cfg = ModelConfig(model_scale="small")
    train_cfg = TrainConfig(debug_overfit=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CFDDEMDataset(
        data_cfg, model_cfg,
        sources=[DATA_SOURCES[train_cfg.debug_source_index]],
    )
    samples = [dataset[index] for index in range(train_cfg.debug_num_pairs)]
    output_dir = os.path.join(os.path.dirname(__file__), "physics_validation_results")
    os.makedirs(output_dir, exist_ok=True)
    return data_cfg, model_cfg, train_cfg, device, samples, output_dir


def load_validation_model(name, model_cfg, train_cfg, device):
    filename, lambda_mass = CHECKPOINTS[name]
    checkpoint = torch.load(
        os.path.join(train_cfg.debug_output_dir, filename),
        map_location=device,
        weights_only=True,
    )
    model = build_model(model_cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, lambda_mass


def encode_fixed_latent(model, inputs):
    """Run the current encoder/approximator once and return fixed decoder KV.

    This mirrors FbedModel.forward up to (but excluding) both decoders. It is
    intentionally validation-only and does not alter the model implementation.
    """
    condition = model.conditioner(inputs["inlet_velocity"])
    particle_tokens = model.supernode_pooling(
        input_feat=inputs["input_particle_displacement"],
        input_pos=inputs["input_particle_coords"],
        supernode_idx=inputs["supernode_idx"],
    )
    particle_tokens = model.to_particle_token(particle_tokens)
    fluid_tokens = model.to_fluid_token(inputs["input_fluid_state"])
    fluid_tokens = fluid_tokens + model.pos_embed(
        model._normalize_fluid_coords(inputs["input_fluid_coords"]))
    branches = [particle_tokens, fluid_tokens]
    boundaries = {
        depth_index
        for depth_index in (model.depth // 3, 2 * model.depth // 3)
        if 0 < depth_index < model.depth
    }
    for index, block in enumerate(model.blocks):
        if index in boundaries:
            branches = [F.layer_norm(branch, [model.dim], eps=1e-6)
                        for branch in branches]
        branches = block(x=branches, cond=(condition, condition))
    return branches[1], condition


def decode_fluid_at_physical_coords(model, latent, condition, physical_coords):
    return model.fluid_decoder(
        x=latent,
        pos=model._normalize_fluid_coords(physical_coords),
        block_kwargs={"cond": condition},
    )


def fluid_flux(fluid, data_cfg):
    pressure = denormalize_pressure(
        fluid[..., 4:5], data_cfg.pressure_ref, data_cfg.pressure_scale)
    density = gas_density_from_pressure(
        pressure, data_cfg.molecular_weight, data_cfg.gas_constant,
        data_cfg.gas_temperature)
    mass = fluid[..., 3:4] * density
    return mass * fluid[..., :3], mass


def reshape_cells(values, data_cfg):
    """Canonical flattened x,y,z cells -> structured [nx,ny,nz,C]."""
    return values.reshape(data_cfg.nx, data_cfg.ny, data_cfg.nz, -1)


def finite_difference_structured(values, spacing, axis):
    """Second-order centered interior and second-order one-sided boundaries."""
    if values.shape[axis] < 3:
        raise ValueError("at least three cells are required for second-order FD")
    result = torch.empty_like(values)
    middle = [slice(None)] * values.ndim
    plus = [slice(None)] * values.ndim
    minus = [slice(None)] * values.ndim
    middle[axis] = slice(1, -1)
    plus[axis] = slice(2, None)
    minus[axis] = slice(None, -2)
    result[tuple(middle)] = (
        values[tuple(plus)] - values[tuple(minus)]) / (2.0 * spacing)

    first = [slice(None)] * values.ndim
    second = [slice(None)] * values.ndim
    third = [slice(None)] * values.ndim
    first[axis], second[axis], third[axis] = 0, 1, 2
    result[tuple(first)] = (
        -3.0 * values[tuple(first)] + 4.0 * values[tuple(second)]
        - values[tuple(third)]) / (2.0 * spacing)

    last = [slice(None)] * values.ndim
    penultimate = [slice(None)] * values.ndim
    antepenultimate = [slice(None)] * values.ndim
    last[axis], penultimate[axis], antepenultimate[axis] = -1, -2, -3
    result[tuple(last)] = (
        3.0 * values[tuple(last)] - 4.0 * values[tuple(penultimate)]
        + values[tuple(antepenultimate)]) / (2.0 * spacing)
    return result


def cfd_directional_divergence(fluid, data_cfg):
    """FD derivatives of the three matching flux components on the CFD grid."""
    flux, mass = fluid_flux(fluid, data_cfg)
    structured = reshape_cells(flux.squeeze(0), data_cfg)
    spacings = (data_cfg.x_len / data_cfg.nx,
                data_cfg.y_len / data_cfg.ny,
                data_cfg.z_len / data_cfg.nz)
    derivatives = []
    for axis, spacing in enumerate(spacings):
        component = structured[..., axis:axis + 1]
        derivatives.append(
            finite_difference_structured(component, spacing, axis).reshape(1, -1, 1))
    return tuple(derivatives), mass


def common_interior_mask(data_cfg, device="cpu"):
    mask = torch.zeros(
        data_cfg.nx, data_cfg.ny, data_cfg.nz, dtype=torch.bool,
        device=device)
    mask[1:-1, 1:-1, 1:-1] = True
    return mask.reshape(-1)


def axis_interior_mask(data_cfg, axis, h, device="cpu"):
    lengths = (data_cfg.x_len, data_cfg.y_len, data_cfg.z_len)
    counts = (data_cfg.nx, data_cfg.ny, data_cfg.nz)
    centers = [
        (torch.arange(count, device=device, dtype=torch.float32) + 0.5)
        * (length / count)
        for length, count in zip(lengths, counts)
    ]
    grid = torch.meshgrid(*centers, indexing="ij")
    coordinate = grid[axis].reshape(-1)
    return (coordinate - h >= 0.0) & (coordinate + h <= lengths[axis])


def descriptive_stats(values):
    values = values.detach().double().reshape(-1)
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "mae": values.abs().mean().item(),
        "rms": values.square().mean().sqrt().item(),
        "max_abs": values.abs().max().item(),
    }


def correlation(left, right):
    left = left.detach().double().reshape(-1)
    right = right.detach().double().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if denominator.item() == 0.0:
        return float("nan")
    return (left * right).sum().div(denominator).item()


def comparison_metrics(predicted, reference, epsilon=1e-12):
    predicted = predicted.detach().double().reshape(-1)
    reference = reference.detach().double().reshape(-1)
    difference = predicted - reference
    denominator = reference.square().sum().sqrt().item()
    return {
        "mae": difference.abs().mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "relative_l2": difference.square().sum().sqrt().item() / (denominator + epsilon),
        "correlation": correlation(predicted, reference),
        "max_abs_difference": difference.abs().max().item(),
    }


def cancellation_metrics(first, second):
    first_stats = descriptive_stats(first)
    second_stats = descriptive_stats(second)
    combined_stats = descriptive_stats(first + second)
    denominator = first_stats["rms"] + second_stats["rms"]
    return {
        "correlation": correlation(first, second),
        "cancellation_ratio": combined_stats["rms"] / denominator,
        "combined_rms": combined_stats["rms"],
    }


def concatenate(items, mask=None):
    selected = []
    for item in items:
        flattened = item.reshape(-1)
        selected.append(flattened if mask is None else flattened[mask])
    return torch.cat(selected)


def save_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)

