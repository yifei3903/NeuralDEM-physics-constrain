"""Part C: sampled-data CFD continuity reference and dt sensitivity."""

import csv
import os

import torch

from config import DATA_SOURCES
from dataset import read_fluid_file
from inference import make_fluid_state
from physics_validation_utils import (
    cancellation_metrics,
    cfd_directional_divergence,
    common_interior_mask,
    concatenate,
    descriptive_stats,
    fluid_flux,
    save_json,
    validation_setup,
)


def load_fluid_state(source, frame, data_cfg, device):
    velocity, voidfraction, pressure = read_fluid_file(
        os.path.join(source["fluid_dir"], f"fluid_{frame}.txt"), data_cfg)
    state = make_fluid_state(velocity, voidfraction, pressure, data_cfg)
    return torch.from_numpy(state).unsqueeze(0).to(device)


def cfd_terms(fluid_n, fluid_np1, dt, data_cfg):
    _, mass_n = fluid_flux(fluid_n, data_cfg)
    directional, mass_np1 = cfd_directional_divergence(fluid_np1, data_cfg)
    temporal = (mass_np1 - mass_n) / dt
    divergence = sum(directional)
    residual = temporal + divergence
    return {
        "temporal": temporal,
        "divergence": divergence,
        "residual": residual,
        "dJx_dx": directional[0],
        "dJy_dy": directional[1],
        "dJz_dz": directional[2],
    }


def summarize_terms(items, mask):
    temporal = concatenate([item["temporal"] for item in items], mask)
    divergence = concatenate([item["divergence"] for item in items], mask)
    residual = concatenate([item["residual"] for item in items], mask)
    y_derivative = concatenate([item["dJy_dy"] for item in items], mask)
    z_derivative = concatenate([item["dJz_dz"] for item in items], mask)
    return {
        "temporal": descriptive_stats(temporal),
        "divergence": descriptive_stats(divergence),
        "residual": descriptive_stats(residual),
        "temporal_divergence": cancellation_metrics(temporal, divergence),
        "y_z_cancellation": cancellation_metrics(y_derivative, z_derivative),
        "directional": {
            "dJx_dx": descriptive_stats(concatenate(
                [item["dJx_dx"] for item in items], mask)),
            "dJy_dy": descriptive_stats(y_derivative),
            "dJz_dz": descriptive_stats(z_derivative),
        },
    }


def flatten_pair_row(t0, t1, dt, region, summary):
    row = {"t0": t0, "t1": t1, "dt_s": dt, "region": region}
    for term in ("temporal", "divergence", "residual"):
        for statistic, value in summary[term].items():
            row[f"{term}_{statistic}"] = value
    row["corr_temporal_divergence"] = summary["temporal_divergence"]["correlation"]
    row["continuity_cancellation_ratio"] = summary["temporal_divergence"]["cancellation_ratio"]
    row["corr_dJy_dy_dJz_dz"] = summary["y_z_cancellation"]["correlation"]
    row["yz_cancellation_ratio"] = summary["y_z_cancellation"]["cancellation_ratio"]
    return row


def main():
    data_cfg, _, train_cfg, device, samples, output_dir = validation_setup()
    source = DATA_SOURCES[train_cfg.debug_source_index]
    interior_mask = common_interior_mask(data_cfg, device=device)
    pairs = [(int(batch["t0"]), int(batch["t1"])) for batch in samples]
    terms_by_pair = []
    pair_rows = []

    for t0, t1 in pairs:
        fluid_n = load_fluid_state(source, t0, data_cfg, device)
        fluid_np1 = load_fluid_state(source, t1, data_cfg, device)
        terms = cfd_terms(fluid_n, fluid_np1, data_cfg.ml_dt, data_cfg)
        terms_by_pair.append(terms)
        for region, mask in (("all_cells", None),
                             ("interior_only", interior_mask)):
            summary = summarize_terms([terms], mask)
            pair_rows.append(flatten_pair_row(
                t0, t1, data_cfg.ml_dt, region, summary))

    overall = {
        "description": (
            "ML-scale / sampled-data continuity reference; this is not the "
            "MFiX internal-dt discrete residual"
        ),
        "raw_output_dt_s": data_cfg.original_dt,
        "frame_interval": data_cfg.frame_interval,
        "ml_dt_s": data_cfg.ml_dt,
        "pairs": pairs,
        "grid_shape_xyz": [data_cfg.nx, data_cfg.ny, data_cfg.nz],
        "common_interior_cells_per_pair": int(interior_mask.sum().item()),
        "z_derivative_warning": (
            "Only 3 z cells: dJz/dz is a CFD-grid numerical reference, not an exact derivative"
        ),
        "regions": {
            "all_cells": summarize_terms(terms_by_pair, None),
            "interior_only": summarize_terms(terms_by_pair, interior_mask),
        },
    }

    pair_path = os.path.join(output_dir, "cfd_continuity_by_pair.csv")
    with open(pair_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    save_json(os.path.join(output_dir, "cfd_continuity_summary.json"), overall)

    # Use the same 20 starting frames for all intervals so the sensitivity
    # comparison is not confounded by selecting a different flow segment.
    sensitivity_rows = []
    for interval in (1, 10, 100):
        interval_terms = []
        dt = data_cfg.original_dt * interval
        for t0, _ in pairs:
            fluid_n = load_fluid_state(source, t0, data_cfg, device)
            fluid_np1 = load_fluid_state(source, t0 + interval, data_cfg, device)
            interval_terms.append(cfd_terms(fluid_n, fluid_np1, dt, data_cfg))
        for region, mask in (("all_cells", None),
                             ("interior_only", interior_mask)):
            summary = summarize_terms(interval_terms, mask)
            sensitivity_rows.append({
                "frame_interval": interval,
                "dt_s": dt,
                "region": region,
                "num_pairs": len(pairs),
                "temporal_rms": summary["temporal"]["rms"],
                "divergence_rms": summary["divergence"]["rms"],
                "residual_mae": summary["residual"]["mae"],
                "residual_rms": summary["residual"]["rms"],
                "residual_max_abs": summary["residual"]["max_abs"],
                "corr_temporal_divergence": summary["temporal_divergence"]["correlation"],
                "cancellation_ratio": summary["temporal_divergence"]["cancellation_ratio"],
            })
    sensitivity_path = os.path.join(output_dir, "cfd_dt_sensitivity.csv")
    with open(sensitivity_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)

    for region, result in overall["regions"].items():
        print(region, "CFD A_rms=", result["temporal"]["rms"],
              "B_rms=", result["divergence"]["rms"],
              "R_rms=", result["residual"]["rms"])
    print("saved", pair_path, sensitivity_path)


if __name__ == "__main__":
    main()
