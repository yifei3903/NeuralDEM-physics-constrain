"""Stage 4 Parts B-E: field/flux and NN-FD/NN-AD/CFD-FD diagnostics."""

import csv
import json
import os

import torch

from physics_losses import compute_mass_residual, denormalize_pressure, gas_density_from_pressure
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
    fluid_flux,
    load_validation_model,
    save_json,
    validation_setup,
)
from train import model_inputs_from_batch, set_seed


FIELD_NAMES = ("u", "v", "w", "epsilon", "pressure", "rho", "Jx", "Jy", "Jz")
DERIVATIVE_NAMES = ("dJx_dx", "dJy_dy", "dJz_dz", "divergence")
COMPARISONS = {
    "nn_fd_vs_cfd_fd": ("NN-FD", "CFD-FD"),
    "nn_ad_vs_nn_fd": ("NN-AD", "NN-FD"),
    "nn_ad_vs_cfd_fd": ("NN-AD", "CFD-FD"),
}


def field_values(fluid, data_cfg):
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


def derivative_dict(directional):
    return {
        "dJx_dx": directional[0],
        "dJy_dy": directional[1],
        "dJz_dz": directional[2],
        "divergence": sum(directional),
    }


def evaluate_model(model, samples, data_cfg, device):
    records = []
    with torch.enable_grad():
        for batch in samples:
            inputs, coords = model_inputs_from_batch(
                batch, device, require_coordinate_grad=True)
            with torch.no_grad():
                latent, condition = encode_fixed_latent(model, inputs)
                latent, condition = latent.detach(), condition.detach()
            predicted = decode_fluid_at_physical_coords(
                model, latent, condition, coords)
            cfd = batch["gt_fluid"].to(device)

            nn_fd_directional, _ = cfd_directional_divergence(predicted, data_cfg)
            cfd_fd_directional, _ = cfd_directional_divergence(cfd, data_cfg)
            _, ad_terms = compute_mass_residual(
                predicted, batch["input_fluid_cell_state"].to(device),
                coords, data_cfg.ml_dt, data_cfg.pressure_ref,
                data_cfg.pressure_scale, data_cfg.molecular_weight,
                data_cfg.gas_constant, data_cfg.gas_temperature,
                create_graph=False, return_terms=True)
            nn_ad = {
                name: ad_terms[name].detach()
                for name in DERIVATIVE_NAMES
            }
            records.append({
                "fields_nn": {key: value.detach() for key, value in field_values(predicted, data_cfg).items()},
                "fields_cfd": {key: value.detach() for key, value in field_values(cfd, data_cfg).items()},
                "NN-FD": {key: value.detach() for key, value in derivative_dict(nn_fd_directional).items()},
                "NN-AD": nn_ad,
                "CFD-FD": {key: value.detach() for key, value in derivative_dict(cfd_fd_directional).items()},
            })
    return records


def selected(records, group, name, mask):
    return concatenate([record[group][name] for record in records], mask)


def metrics_row(model_name, lambda_mass, region, quantity, left_name,
                right_name, left, right):
    metrics = comparison_metrics(left, right)
    return {
        "model": model_name,
        "lambda_mass": lambda_mass,
        "region": region,
        "quantity": quantity,
        "field_a": left_name,
        "field_b": right_name,
        "points": left.numel(),
        **metrics,
        "rms_field_a": descriptive_stats(left)["rms"],
        "rms_field_b": descriptive_stats(right)["rms"],
    }


def classify(interior_comparisons):
    """Transparent heuristic implementing the four requested diagnostic cases."""
    fd_cfd = interior_comparisons["nn_fd_vs_cfd_fd"]
    ad_fd = interior_comparisons["nn_ad_vs_nn_fd"]
    fd_cfd_close = all(
        fd_cfd[name]["relative_l2"] < 0.5 and fd_cfd[name]["correlation"] > 0.8
        for name in DERIVATIVE_NAMES)
    ad_fd_close = all(
        ad_fd[name]["relative_l2"] < 0.5 and ad_fd[name]["correlation"] > 0.8
        for name in DERIVATIVE_NAMES)
    if fd_cfd_close and not ad_fd_close:
        case = 1
        diagnosis = "cell field is reasonably correct; continuous interpolation differs"
    elif not fd_cfd_close:
        case = 2
        diagnosis = "predicted discrete spatial field is not accurate enough"
    elif ad_fd_close:
        case = 4
        diagnosis = "NN-FD, NN-AD and CFD-FD are broadly consistent"
    else:
        case = 3
        diagnosis = "neural field is internally consistent but represents the wrong spatial structure"
    return {
        "case": case,
        "diagnosis": diagnosis,
        "heuristic": "close means relative_l2<0.5 and correlation>0.8 for all Dx,Dy,Dz,B",
        "nn_fd_vs_cfd_fd_close": fd_cfd_close,
        "nn_ad_vs_nn_fd_close": ad_fd_close,
    }


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    data_cfg, model_cfg, train_cfg, device, samples, _ = validation_setup()
    output_dir = os.path.join(os.path.dirname(__file__), "physics_validation_results_stage4")
    os.makedirs(output_dir, exist_ok=True)
    set_seed(train_cfg.seed)
    interior_mask = common_interior_mask(data_cfg, device=device)
    field_rows = []
    derivative_rows = {name: [] for name in COMPARISONS}
    yz_rows = []
    summary = {
        "pairs": [[int(batch["t0"]), int(batch["t1"])] for batch in samples],
        "grid": {
            "shape_xyz": [data_cfg.nx, data_cfg.ny, data_cfg.nz],
            "interior_cells_per_pair": int(interior_mask.sum().item()),
            "warning": "z has only 3 cells; CFD-FD z derivative is a grid numerical reference",
        },
        "models": {},
        "classification_rule": "See each model classification. Thresholds are diagnostic, not acceptance criteria.",
    }

    cfd_yz_written = set()
    for model_name in CHECKPOINTS:
        model, lambda_mass = load_validation_model(
            model_name, model_cfg, train_cfg, device)
        records = evaluate_model(model, samples, data_cfg, device)
        model_summary = {"lambda_mass": lambda_mass, "regions": {}}

        for region, mask in (("all_cells", None),
                             ("interior_only", interior_mask)):
            region_summary = {"field_and_flux": {}, "derivative_comparisons": {},
                              "yz_cancellation": {}}
            for name in FIELD_NAMES:
                nn_value = selected(records, "fields_nn", name, mask)
                cfd_value = selected(records, "fields_cfd", name, mask)
                row = metrics_row(model_name, lambda_mass, region, name,
                                  "NN", "CFD", nn_value, cfd_value)
                field_rows.append(row)
                region_summary["field_and_flux"][name] = row

            for comparison_name, (left_group, right_group) in COMPARISONS.items():
                comparison_summary = {}
                for name in DERIVATIVE_NAMES:
                    left = selected(records, left_group, name, mask)
                    right = selected(records, right_group, name, mask)
                    row = metrics_row(
                        model_name, lambda_mass, region, name,
                        left_group, right_group, left, right)
                    derivative_rows[comparison_name].append(row)
                    comparison_summary[name] = row
                region_summary["derivative_comparisons"][comparison_name] = comparison_summary

            for representation in ("CFD-FD", "NN-FD", "NN-AD"):
                if representation == "CFD-FD" and region in cfd_yz_written:
                    continue
                y_value = selected(records, representation, "dJy_dy", mask)
                z_value = selected(records, representation, "dJz_dz", mask)
                yz = cancellation_metrics(y_value, z_value)
                row = {
                    "model": "CFD_reference" if representation == "CFD-FD" else model_name,
                    "lambda_mass": "" if representation == "CFD-FD" else lambda_mass,
                    "region": region,
                    "representation": representation,
                    "corr_Dy_Dz": yz["correlation"],
                    "C_yz": yz["cancellation_ratio"],
                    "rms_Dy": descriptive_stats(y_value)["rms"],
                    "rms_Dz": descriptive_stats(z_value)["rms"],
                    "rms_Dy_plus_Dz": yz["combined_rms"],
                }
                yz_rows.append(row)
                region_summary["yz_cancellation"][representation] = row
                if representation == "CFD-FD":
                    cfd_yz_written.add(region)
            model_summary["regions"][region] = region_summary

        model_summary["classification"] = classify(
            model_summary["regions"]["interior_only"]["derivative_comparisons"])
        summary["models"][model_name] = model_summary
        print(model_name, model_summary["classification"], flush=True)

    write_csv(os.path.join(output_dir, "field_and_flux_accuracy.csv"), field_rows)
    for comparison_name, rows in derivative_rows.items():
        write_csv(os.path.join(output_dir, comparison_name + ".csv"), rows)
    write_csv(os.path.join(output_dir, "yz_cancellation_three_way.csv"), yz_rows)
    save_json(os.path.join(output_dir, "field_and_flux_summary.json"), {
        "models": {
            name: {
                region: contents["field_and_flux"]
                for region, contents in model["regions"].items()
            }
            for name, model in summary["models"].items()
        }
    })
    save_json(os.path.join(output_dir, "derivative_three_way_summary.json"), summary)

    markdown = [
        "# Stage 4 diagnostic summary",
        "",
        "No model was trained or modified. Main scientific comparisons use the common interior mask.",
        "",
        "## Automatic diagnostic classification",
        "",
        "| model | case | diagnosis |",
        "|---|---:|---|",
    ]
    for model_name, model in summary["models"].items():
        classification = model["classification"]
        markdown.append(
            f"| {model_name} | {classification['case']} | {classification['diagnosis']} |")
    temporal_path = os.path.join(output_dir, "temporal_scheme_summary.json")
    if os.path.isfile(temporal_path):
        with open(temporal_path, "r", encoding="utf-8") as handle:
            temporal_summary = json.load(handle)
        markdown.extend([
            "", "## Temporal schemes: interior-only", "",
            "| scheme | residual MAE | residual RMS |",
            "|---|---:|---:|",
        ])
        for scheme, result in temporal_summary["regions"]["interior_only"]["schemes"].items():
            markdown.append(
                f"| {scheme} | {result['residual']['mae']:.6g} | {result['residual']['rms']:.6g} |")

    markdown.extend([
        "", "## Flux-value accuracy: interior-only", "",
        "| model | flux | relative L2 | correlation |",
        "|---|---|---:|---:|",
    ])
    for model_name, model in summary["models"].items():
        fields = model["regions"]["interior_only"]["field_and_flux"]
        for flux_name in ("Jx", "Jy", "Jz"):
            result = fields[flux_name]
            markdown.append(
                f"| {model_name} | {flux_name} | {result['relative_l2']:.6g} | {result['correlation']:.6g} |")

    markdown.extend([
        "", "## Derivative-source isolation: interior-only divergence", "",
        "| model | comparison | relative L2 | correlation |",
        "|---|---|---:|---:|",
    ])
    for model_name, model in summary["models"].items():
        comparisons = model["regions"]["interior_only"]["derivative_comparisons"]
        for comparison_name in COMPARISONS:
            result = comparisons[comparison_name]["divergence"]
            markdown.append(
                f"| {model_name} | {comparison_name} | {result['relative_l2']:.6g} | {result['correlation']:.6g} |")

    markdown.extend([
        "", "## y-z cancellation: interior-only", "",
        "| model | representation | corr(Dy,Dz) | C_yz |",
        "|---|---|---:|---:|",
    ])
    for row in yz_rows:
        if row["region"] == "interior_only":
            markdown.append(
                f"| {row['model']} | {row['representation']} | {row['corr_Dy_Dz']:.6g} | {row['C_yz']:.6g} |")
    markdown.extend(["", "Full raw metrics are stored in the adjacent CSV and JSON files.", ""])
    with open(os.path.join(output_dir, "stage4_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown))
    print("saved stage4 three-way diagnostics to", output_dir, flush=True)


if __name__ == "__main__":
    main()
