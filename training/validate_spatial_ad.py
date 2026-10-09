"""Part A: validate coordinate AD against central differences of the same NN field."""

import csv
import os

import torch

from physics_validation_utils import (
    CHECKPOINTS,
    axis_interior_mask,
    comparison_metrics,
    decode_fluid_at_physical_coords,
    encode_fixed_latent,
    fluid_flux,
    load_validation_model,
    save_json,
    validation_setup,
)
from train import model_inputs_from_batch, set_seed


# The five requested ratios are retained. Two coarser probes expose the
# truncation-error side of the curve when float32 round-off already dominates
# at h/L=1e-2 (observed especially for x and z).
H_RATIOS = (5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4)
AXIS_NAMES = ("x", "y", "z")


def ad_derivatives(model, latent, condition, physical_coords, data_cfg):
    coords = physical_coords.detach().clone().requires_grad_(True)
    fluid = decode_fluid_at_physical_coords(model, latent, condition, coords)
    flux, _ = fluid_flux(fluid, data_cfg)
    derivatives = []
    for axis in range(3):
        gradient = torch.autograd.grad(
            flux[..., axis:axis + 1].sum(), coords,
            retain_graph=True, create_graph=False)[0]
        derivatives.append(gradient[..., axis:axis + 1].detach())
    return derivatives


def cd_derivative(model, latent, condition, physical_coords, data_cfg, axis, h):
    plus = physical_coords.detach().clone()
    minus = physical_coords.detach().clone()
    plus[..., axis] += h
    minus[..., axis] -= h
    with torch.no_grad():
        plus_fluid = decode_fluid_at_physical_coords(model, latent, condition, plus)
        minus_fluid = decode_fluid_at_physical_coords(model, latent, condition, minus)
        plus_flux, _ = fluid_flux(plus_fluid, data_cfg)
        minus_flux, _ = fluid_flux(minus_fluid, data_cfg)
    return ((plus_flux[..., axis:axis + 1]
             - minus_flux[..., axis:axis + 1]) / (2.0 * h)).detach()


def query_independence(model, latent, condition, physical_coords):
    query = physical_coords[:, :4].detach().clone()
    with torch.no_grad():
        original = decode_fluid_at_physical_coords(model, latent, condition, query)
        changed_query = query.clone()
        changed_query[:, 0, 0] += 1e-4
        changed = decode_fluid_at_physical_coords(
            model, latent, condition, changed_query)
    return (original[:, 1:] - changed[:, 1:]).abs().max().item()


def main():
    data_cfg, model_cfg, train_cfg, device, samples, output_dir = validation_setup()
    set_seed(train_cfg.seed)
    batch = samples[0]
    domain_lengths = (data_cfg.x_len, data_cfg.y_len, data_cfg.z_len)
    rows = []
    summary = {
        "sample_pair": [int(batch["t0"]), int(batch["t1"])],
        "description": "AD and CD evaluate the same decoder field f(x; fixed latent)",
        "models": {},
    }

    for model_name in CHECKPOINTS:
        model, _ = load_validation_model(model_name, model_cfg, train_cfg, device)
        inputs, physical_coords = model_inputs_from_batch(
            batch, device, require_coordinate_grad=False)
        with torch.no_grad():
            latent, condition = encode_fixed_latent(model, inputs)
            latent = latent.detach()
            condition = condition.detach()
            fixed_latent_output = decode_fluid_at_physical_coords(
                model, latent, condition, physical_coords)
            full_forward_output = model(**inputs)["fluid"]
            reconstruction_error = (
                fixed_latent_output - full_forward_output).abs().max().item()
        independence_error = query_independence(
            model, latent, condition, physical_coords)
        ad_values = ad_derivatives(
            model, latent, condition, physical_coords, data_cfg)
        model_summary = {
            "query_independence_max_off_diagonal_change": independence_error,
            "fixed_latent_vs_full_forward_max_abs_difference": reconstruction_error,
            "axes": {},
        }

        for axis, axis_name in enumerate(AXIS_NAMES):
            axis_rows = []
            for ratio in H_RATIOS:
                h = ratio * domain_lengths[axis]
                mask = axis_interior_mask(data_cfg, axis, h, device=device)
                cd_value = cd_derivative(
                    model, latent, condition, physical_coords,
                    data_cfg, axis, h)
                metrics = comparison_metrics(
                    cd_value.reshape(-1)[mask],
                    ad_values[axis].reshape(-1)[mask])
                row = {
                    "model": model_name,
                    "sample_t0": int(batch["t0"]),
                    "sample_t1": int(batch["t1"]),
                    "axis": axis_name,
                    "h_over_L": ratio,
                    "h_m": h,
                    "valid_points": int(mask.sum().item()),
                    **metrics,
                }
                rows.append(row)
                axis_rows.append(row)
                print(model_name, axis_name, f"h/L={ratio:g}",
                      f"relative_l2={metrics['relative_l2']:.6g}",
                      f"correlation={metrics['correlation']:.9g}")
            best = min(axis_rows, key=lambda item: item["relative_l2"])
            model_summary["axes"][axis_name] = {
                "best_h_over_L": best["h_over_L"],
                "best_metrics": {
                    key: best[key] for key in
                    ("mae", "rmse", "relative_l2", "correlation",
                     "max_abs_difference")
                },
                "rows": axis_rows,
            }
        summary["models"][model_name] = model_summary

    csv_path = os.path.join(output_dir, "ad_cd_convergence.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(os.path.join(output_dir, "ad_cd_summary.json"), summary)
    print("saved", csv_path)


if __name__ == "__main__":
    main()
