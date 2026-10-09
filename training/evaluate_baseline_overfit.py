"""Evaluation utilities for the Stage-5 pure-data baseline overfit audit."""

import csv
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from physics_losses import denormalize_pressure, gas_density_from_pressure
from physics_validation_utils import (
    cancellation_metrics,
    cfd_directional_divergence,
    common_interior_mask,
    comparison_metrics,
    concatenate,
    decode_fluid_at_physical_coords,
    descriptive_stats,
    encode_fixed_latent,
    fluid_flux,
)
from train import model_inputs_from_batch


FLUID_CHANNELS = ("u", "v", "w", "epsilon", "pressure")
PHYSICAL_FIELDS = ("u", "v", "w", "epsilon", "pressure", "rho")
FLUX_FIELDS = ("Jx", "Jy", "Jz")
DERIVATIVES = ("dJx_dx", "dJy_dy", "dJz_dz", "divergence")


def fluid_fields(fluid, data_cfg):
    pressure = denormalize_pressure(
        fluid[..., 4:5], data_cfg.pressure_ref, data_cfg.pressure_scale)
    rho = gas_density_from_pressure(
        pressure, data_cfg.molecular_weight, data_cfg.gas_constant,
        data_cfg.gas_temperature)
    flux, _ = fluid_flux(fluid, data_cfg)
    return {
        "u": fluid[..., 0:1],
        "v": fluid[..., 1:2],
        "w": fluid[..., 2:3],
        "epsilon": fluid[..., 3:4],
        "pressure": pressure,
        "rho": rho,
        "Jx": flux[..., 0:1],
        "Jy": flux[..., 1:2],
        "Jz": flux[..., 2:3],
    }


def predict_fluid(model, batch, device):
    inputs, coords = model_inputs_from_batch(
        batch, device, require_coordinate_grad=False)
    with torch.no_grad():
        latent, condition = encode_fixed_latent(model, inputs)
        return decode_fluid_at_physical_coords(
            model, latent, condition, coords).detach()


def normalization_audit(samples, data_cfg):
    normalized = torch.cat([batch["gt_fluid"].reshape(-1, 5) for batch in samples])
    physical = normalized.clone()
    physical[:, 4:5] = denormalize_pressure(
        normalized[:, 4:5], data_cfg.pressure_ref, data_cfg.pressure_scale)
    formulas = {
        "u": "identity: u*=u [m/s]",
        "v": "identity: v*=v [m/s]",
        "w": "identity: w*=w [m/s]",
        "epsilon": "identity: epsilon*=epsilon",
        "pressure": (
            f"P*=(P-{data_cfg.pressure_ref})/{data_cfg.pressure_scale}; "
            f"P=P*{data_cfg.pressure_scale}+{data_cfg.pressure_ref} Pa"
        ),
    }
    output = {
        "scope": "20 supervised target fields t_(n+1), 0.60 m/s",
        "channels": {},
        "fluid_loss_structure": {
            "formula": "L_fluid=mean(MSE_u,MSE_v,MSE_w)+MSE_epsilon+MSE_Pstar",
            "outer_fluid_weight": 3.0,
            "effective_channel_weights_in_total_data_loss": {
                "u": 1.0, "v": 1.0, "w": 1.0,
                "epsilon": 3.0, "pressure": 3.0,
            },
            "five_channels_equally_weighted": False,
        },
    }
    for index, name in enumerate(FLUID_CHANNELS):
        p = descriptive_stats(physical[:, index:index + 1])
        n = descriptive_stats(normalized[:, index:index + 1])
        output["channels"][name] = {
            "physical_min": physical[:, index].min().item(),
            "physical_max": physical[:, index].max().item(),
            "physical_mean": p["mean"],
            "physical_std": p["std"],
            "normalization_formula": formulas[name],
            "normalized_min": normalized[:, index].min().item(),
            "normalized_max": normalized[:, index].max().item(),
            "normalized_mean": n["mean"],
            "normalized_std": n["std"],
        }
    return output


def centered_per_sample(values):
    return values - values.mean(dim=1, keepdim=True)


def raw_accuracy_row(epoch, field, predicted, target):
    metrics = comparison_metrics(predicted, target)
    pred_stats = descriptive_stats(predicted)
    target_stats = descriptive_stats(target)
    return {
        "epoch": epoch,
        "field": field,
        **metrics,
        "prediction_mean": pred_stats["mean"],
        "target_mean": target_stats["mean"],
        "prediction_std": pred_stats["std"],
        "target_std": target_stats["std"],
        "std_ratio": pred_stats["std"] / (target_stats["std"] + 1e-12),
    }


def fluctuation_row(epoch, field, predicted_by_sample, target_by_sample):
    predicted = torch.cat([centered_per_sample(value).reshape(-1)
                           for value in predicted_by_sample])
    target = torch.cat([centered_per_sample(value).reshape(-1)
                        for value in target_by_sample])
    metrics = comparison_metrics(predicted, target)
    pred_stats = descriptive_stats(predicted)
    target_stats = descriptive_stats(target)
    return {
        "epoch": epoch,
        "field": field,
        "fluctuation_relative_l2": metrics["relative_l2"],
        "centered_correlation": metrics["correlation"],
        "prediction_fluctuation_rms": pred_stats["rms"],
        "target_fluctuation_rms": target_stats["rms"],
        "fluctuation_std_ratio": pred_stats["std"] / (target_stats["std"] + 1e-12),
        "fluctuation_mae": metrics["mae"],
        "fluctuation_rmse": metrics["rmse"],
    }


def derivative_dict(directional):
    return {
        "dJx_dx": directional[0],
        "dJy_dy": directional[1],
        "dJz_dz": directional[2],
        "divergence": sum(directional),
    }


def evaluate_checkpoint(model, samples, epoch, data_cfg, device,
                        include_fd_gradients=False):
    model.eval()
    predicted_fields = {name: [] for name in PHYSICAL_FIELDS + FLUX_FIELDS}
    target_fields = {name: [] for name in PHYSICAL_FIELDS + FLUX_FIELDS}
    nn_fd_records, cfd_fd_records = [], []

    for batch in samples:
        predicted = predict_fluid(model, batch, device)
        target = batch["gt_fluid"].to(device)
        predicted_dict = fluid_fields(predicted, data_cfg)
        target_dict = fluid_fields(target, data_cfg)
        for name in predicted_fields:
            predicted_fields[name].append(predicted_dict[name])
            target_fields[name].append(target_dict[name])
        if include_fd_gradients:
            nn_directional, _ = cfd_directional_divergence(predicted, data_cfg)
            cfd_directional, _ = cfd_directional_divergence(target, data_cfg)
            nn_fd_records.append(derivative_dict(nn_directional))
            cfd_fd_records.append(derivative_dict(cfd_directional))

    field_rows, fluctuation_rows, flux_rows = [], [], []
    for name in PHYSICAL_FIELDS:
        predicted = torch.cat([value.reshape(-1) for value in predicted_fields[name]])
        target = torch.cat([value.reshape(-1) for value in target_fields[name]])
        field_rows.append(raw_accuracy_row(epoch, name, predicted, target))
    for name in ("u", "v", "w") + FLUX_FIELDS:
        fluctuation_rows.append(fluctuation_row(
            epoch, name, predicted_fields[name], target_fields[name]))
    for name in FLUX_FIELDS:
        predicted = torch.cat([value.reshape(-1) for value in predicted_fields[name]])
        target = torch.cat([value.reshape(-1) for value in target_fields[name]])
        row = raw_accuracy_row(epoch, name, predicted, target)
        centered = next(item for item in fluctuation_rows if item["field"] == name)
        row.update({key: value for key, value in centered.items()
                    if key not in ("epoch", "field")})
        flux_rows.append(row)

    fd_rows, yz_rows = [], []
    if include_fd_gradients:
        interior_mask = common_interior_mask(data_cfg, device=device)
        for region, mask in (("all_cells", None),
                             ("interior_only", interior_mask)):
            for name in DERIVATIVES:
                predicted = concatenate([item[name] for item in nn_fd_records], mask)
                target = concatenate([item[name] for item in cfd_fd_records], mask)
                metrics = comparison_metrics(predicted, target)
                fd_rows.append({
                    "epoch": epoch, "region": region, "quantity": name,
                    **metrics,
                    "nn_rms": descriptive_stats(predicted)["rms"],
                    "cfd_rms": descriptive_stats(target)["rms"],
                })
            nn_y = concatenate([item["dJy_dy"] for item in nn_fd_records], mask)
            nn_z = concatenate([item["dJz_dz"] for item in nn_fd_records], mask)
            yz = cancellation_metrics(nn_y, nn_z)
            yz_rows.append({
                "epoch": epoch, "region": region,
                "corr_Dy_Dz": yz["correlation"],
                "C_yz": yz["cancellation_ratio"],
                "rms_Dy": descriptive_stats(nn_y)["rms"],
                "rms_Dz": descriptive_stats(nn_z)["rms"],
                "rms_Dy_plus_Dz": yz["combined_rms"],
            })
    return field_rows, fluctuation_rows, flux_rows, fd_rows, yz_rows


def gradient_groups(model):
    return {
        "fluid_decoder": list(model.fluid_decoder.parameters()),
        "fluid_encoder": list(model.to_fluid_token.parameters()),
        "transformer": list(model.blocks.parameters()),
    }


def channel_gradient_audit(model, batch, epoch, device):
    model.train()
    inputs, coords = model_inputs_from_batch(
        batch, device, require_coordinate_grad=False)
    latent, condition = encode_fixed_latent(model, inputs)
    prediction = decode_fluid_at_physical_coords(
        model, latent, condition, coords)
    target = batch["gt_fluid"].to(device)
    groups = gradient_groups(model)
    flat_parameters = []
    spans = {}
    for group_name, parameters in groups.items():
        start = len(flat_parameters)
        flat_parameters.extend(parameters)
        spans[group_name] = (start, len(flat_parameters))
    effective_weights = {"u": 1.0, "v": 1.0, "w": 1.0,
                         "epsilon": 3.0, "pressure": 3.0}
    output = {"epoch": epoch, "channels": {}}
    for channel_index, channel_name in enumerate(FLUID_CHANNELS):
        loss = F.mse_loss(
            prediction[..., channel_index:channel_index + 1],
            target[..., channel_index:channel_index + 1])
        gradients = torch.autograd.grad(
            loss, flat_parameters,
            retain_graph=channel_index < len(FLUID_CHANNELS) - 1,
            allow_unused=True)
        group_results = {}
        for group_name, (start, stop) in spans.items():
            squared = sum(
                gradient.detach().double().square().sum().item()
                for gradient in gradients[start:stop] if gradient is not None)
            raw_norm = math.sqrt(squared)
            group_results[group_name] = {
                "raw_gradient_norm": raw_norm,
                "effective_weight_in_total_loss": effective_weights[channel_name],
                "effective_gradient_norm": raw_norm * effective_weights[channel_name],
            }
        output["channels"][channel_name] = {
            "raw_channel_loss": loss.detach().item(),
            "groups": group_results,
        }
    return output


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(
        "Run baseline_overfit_audit.py; this module provides its evaluation utilities.")
