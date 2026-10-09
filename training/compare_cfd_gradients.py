"""Part B: compare NeuralDEM AD flux gradients with CFD-grid FD references."""

import csv
import os

import torch

from physics_losses import compute_mass_residual
from physics_validation_utils import (
    CHECKPOINTS,
    cancellation_metrics,
    cfd_directional_divergence,
    common_interior_mask,
    comparison_metrics,
    concatenate,
    decode_fluid_at_physical_coords,
    descriptive_stats,
    encode_fixed_latent,
    load_validation_model,
    save_json,
    validation_setup,
)
from train import model_inputs_from_batch, set_seed


QUANTITIES = ("dJx_dx", "dJy_dy", "dJz_dz", "divergence")


def cfd_reference_for_samples(samples, data_cfg, device):
    output = []
    for batch in samples:
        fluid = batch["gt_fluid"].to(device)
        directional, _ = cfd_directional_divergence(fluid, data_cfg)
        output.append({
            "dJx_dx": directional[0].detach(),
            "dJy_dy": directional[1].detach(),
            "dJz_dz": directional[2].detach(),
            "divergence": sum(directional).detach(),
        })
    return output


def nn_gradients_for_samples(model, samples, data_cfg, device):
    output = []
    with torch.enable_grad():
        for batch in samples:
            inputs, coords = model_inputs_from_batch(
                batch, device, require_coordinate_grad=True)
            with torch.no_grad():
                latent, condition = encode_fixed_latent(model, inputs)
                latent = latent.detach()
                condition = condition.detach()
            predicted = decode_fluid_at_physical_coords(
                model, latent, condition, coords)
            residual, terms = compute_mass_residual(
                predicted, batch["input_fluid_cell_state"].to(device),
                coords, data_cfg.ml_dt, data_cfg.pressure_ref,
                data_cfg.pressure_scale, data_cfg.molecular_weight,
                data_cfg.gas_constant, data_cfg.gas_temperature,
                create_graph=False, return_terms=True)
            output.append({
                "dJx_dx": terms["dJx_dx"].detach(),
                "dJy_dy": terms["dJy_dy"].detach(),
                "dJz_dz": terms["dJz_dz"].detach(),
                "divergence": terms["divergence"].detach(),
                "temporal": terms["temporal"].detach(),
                "residual": residual.detach(),
            })
    return output


def region_mask(region, interior_mask):
    return None if region == "all_cells" else interior_mask


def main():
    data_cfg, model_cfg, train_cfg, device, samples, output_dir = validation_setup()
    set_seed(train_cfg.seed)
    interior_mask = common_interior_mask(data_cfg, device=device)
    references = cfd_reference_for_samples(samples, data_cfg, device)
    rows = []
    summary = {
        "pairs": [[int(batch["t0"]), int(batch["t1"])] for batch in samples],
        "grid": {
            "shape_xyz": [data_cfg.nx, data_cfg.ny, data_cfg.nz],
            "all_cells_per_pair": data_cfg.n_cells,
            "common_interior_cells_per_pair": int(interior_mask.sum().item()),
            "warning": "z has only 3 cells; CFD z derivative is a grid numerical reference, not an exact derivative",
        },
        "models": {},
        "cfd_yz_cancellation": {},
    }

    for region in ("all_cells", "interior_only"):
        mask = region_mask(region, interior_mask)
        cfd_y = concatenate([item["dJy_dy"] for item in references], mask)
        cfd_z = concatenate([item["dJz_dz"] for item in references], mask)
        summary["cfd_yz_cancellation"][region] = cancellation_metrics(cfd_y, cfd_z)

    for model_name in CHECKPOINTS:
        model, lambda_mass = load_validation_model(
            model_name, model_cfg, train_cfg, device)
        predictions = nn_gradients_for_samples(model, samples, data_cfg, device)
        model_summary = {"lambda_mass": lambda_mass, "regions": {}}
        for region in ("all_cells", "interior_only"):
            mask = region_mask(region, interior_mask)
            region_summary = {"gradient_comparison": {}}
            for quantity in QUANTITIES:
                predicted = concatenate([item[quantity] for item in predictions], mask)
                reference = concatenate([item[quantity] for item in references], mask)
                metrics = comparison_metrics(predicted, reference)
                row = {
                    "model": model_name,
                    "lambda_mass": lambda_mass,
                    "region": region,
                    "quantity": quantity,
                    "points": predicted.numel(),
                    **metrics,
                    "nn_rms": descriptive_stats(predicted)["rms"],
                    "cfd_rms": descriptive_stats(reference)["rms"],
                }
                rows.append(row)
                region_summary["gradient_comparison"][quantity] = row

            nn_y = concatenate([item["dJy_dy"] for item in predictions], mask)
            nn_z = concatenate([item["dJz_dz"] for item in predictions], mask)
            region_summary["nn_yz_cancellation"] = cancellation_metrics(nn_y, nn_z)
            temporal = concatenate([item["temporal"] for item in predictions], mask)
            divergence = concatenate([item["divergence"] for item in predictions], mask)
            residual = concatenate([item["residual"] for item in predictions], mask)
            region_summary["network_continuity"] = {
                "temporal": descriptive_stats(temporal),
                "divergence": descriptive_stats(divergence),
                "residual": descriptive_stats(residual),
                "temporal_divergence": cancellation_metrics(temporal, divergence),
            }
            model_summary["regions"][region] = region_summary
            print(model_name, region,
                  "R_rms=", region_summary["network_continuity"]["residual"]["rms"],
                  "yz=", region_summary["nn_yz_cancellation"])
        summary["models"][model_name] = model_summary

    csv_path = os.path.join(output_dir, "nn_vs_cfd_gradients.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(os.path.join(output_dir, "nn_vs_cfd_summary.json"), summary)
    print("saved", csv_path)


if __name__ == "__main__":
    main()
