"""Stage 4 Part A: sampled-data temporal discretization consistency."""

import csv
import os

import torch

from cfd_continuity_reference import load_fluid_state
from config import DATA_SOURCES
from physics_validation_utils import (
    cancellation_metrics,
    cfd_directional_divergence,
    common_interior_mask,
    comparison_metrics,
    concatenate,
    descriptive_stats,
    fluid_flux,
    save_json,
    validation_setup,
)


SCHEMES = ("FE", "BE_current", "endpoint_trapezoid", "midpoint", "full_time_average")


def summarize(a_values, b_values):
    residual = a_values + b_values
    return {
        "temporal": descriptive_stats(a_values),
        "divergence_used": descriptive_stats(b_values),
        "residual": descriptive_stats(residual),
        "temporal_divergence": cancellation_metrics(a_values, b_values),
    }


def flatten_row(t0, t1, region, scheme, result):
    row = {"t0": t0, "t1": t1, "dt_s": 0.1,
           "region": region, "scheme": scheme}
    for term in ("temporal", "divergence_used", "residual"):
        for statistic, value in result[term].items():
            row[f"{term}_{statistic}"] = value
    row["corr_A_B_used"] = result["temporal_divergence"]["correlation"]
    row["cancellation_ratio"] = result["temporal_divergence"]["cancellation_ratio"]
    return row


def main():
    data_cfg, _, train_cfg, device, samples, _ = validation_setup()
    source = DATA_SOURCES[train_cfg.debug_source_index]
    output_dir = os.path.join(os.path.dirname(__file__), "physics_validation_results_stage4")
    os.makedirs(output_dir, exist_ok=True)
    pairs = [(int(batch["t0"]), int(batch["t1"])) for batch in samples]
    first_frame = min(t0 for t0, _ in pairs)
    last_frame = max(t1 for _, t1 in pairs)

    # The 20 intervals are contiguous, so loading frames 0..2000 once avoids
    # reading shared endpoints and intermediate frames repeatedly.
    cache = {}
    for frame in range(first_frame, last_frame + 1):
        state = load_fluid_state(source, frame, data_cfg, device)
        directional, mass = cfd_directional_divergence(state, data_cfg)
        cache[frame] = {
            "mass": mass.detach().cpu(),
            "divergence": sum(directional).detach().cpu(),
        }
        if frame % 200 == 0:
            print(f"loaded CFD frame {frame}/{last_frame}", flush=True)

    interior_mask = common_interior_mask(data_cfg, device="cpu")
    pair_rows = []
    accumulated = {
        region: {scheme: {"A": [], "B": []} for scheme in SCHEMES}
        for region in ("all_cells", "interior_only")
    }
    endpoint_vs_average = {
        region: {"endpoint": [], "average": []}
        for region in ("all_cells", "interior_only")
    }

    for t0, t1 in pairs:
        temporal = (cache[t1]["mass"] - cache[t0]["mass"]) / data_cfg.ml_dt
        b_n = cache[t0]["divergence"]
        b_np1 = cache[t1]["divergence"]
        b_endpoint_trapezoid = 0.5 * (b_n + b_np1)
        b_midpoint = cache[(t0 + t1) // 2]["divergence"]
        # 101 samples and 100 raw intervals: integral/dt = weighted sum / 100.
        divergence_sum = 0.5 * b_n + 0.5 * b_np1
        for frame in range(t0 + 1, t1):
            divergence_sum = divergence_sum + cache[frame]["divergence"]
        b_time_average = divergence_sum / (t1 - t0)
        scheme_values = {
            "FE": b_n,
            "BE_current": b_np1,
            "endpoint_trapezoid": b_endpoint_trapezoid,
            "midpoint": b_midpoint,
            "full_time_average": b_time_average,
        }

        for region, mask in (("all_cells", None),
                             ("interior_only", interior_mask)):
            a_selected = temporal.reshape(-1) if mask is None else temporal.reshape(-1)[mask]
            for scheme, divergence in scheme_values.items():
                b_selected = (divergence.reshape(-1) if mask is None
                              else divergence.reshape(-1)[mask])
                result = summarize(a_selected, b_selected)
                pair_rows.append(flatten_row(t0, t1, region, scheme, result))
                accumulated[region][scheme]["A"].append(a_selected)
                accumulated[region][scheme]["B"].append(b_selected)
            endpoint_vs_average[region]["endpoint"].append(
                b_np1.reshape(-1) if mask is None else b_np1.reshape(-1)[mask])
            endpoint_vs_average[region]["average"].append(
                b_time_average.reshape(-1) if mask is None else b_time_average.reshape(-1)[mask])

    summary = {
        "description": (
            "Sampled-data temporal consistency diagnostic using cell-centered "
            "outputs and CFD-grid FD; not an MFiX solver conservation error"
        ),
        "raw_output_dt_s": data_cfg.original_dt,
        "ml_dt_s": data_cfg.ml_dt,
        "pairs": pairs,
        "time_average": {
            "points_per_interval": 101,
            "integration": "trapezoidal, including both endpoints",
        },
        "regions": {},
    }
    for region in ("all_cells", "interior_only"):
        region_result = {"schemes": {}}
        for scheme in SCHEMES:
            a_values = torch.cat(accumulated[region][scheme]["A"])
            b_values = torch.cat(accumulated[region][scheme]["B"])
            region_result["schemes"][scheme] = summarize(a_values, b_values)
        endpoint = torch.cat(endpoint_vs_average[region]["endpoint"])
        average = torch.cat(endpoint_vs_average[region]["average"])
        region_result["B_np1_vs_full_time_average"] = comparison_metrics(
            endpoint, average)
        summary["regions"][region] = region_result

    csv_path = os.path.join(output_dir, "temporal_scheme_by_pair.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    save_json(os.path.join(output_dir, "temporal_scheme_summary.json"), summary)
    for region, result in summary["regions"].items():
        print(region, flush=True)
        for scheme in SCHEMES:
            metrics = result["schemes"][scheme]["residual"]
            print(f"  {scheme}: MAE={metrics['mae']:.6g} RMS={metrics['rms']:.6g}", flush=True)
        print("  B_np1 vs B_avg", result["B_np1_vs_full_time_average"], flush=True)
    print("saved", csv_path, flush=True)


if __name__ == "__main__":
    main()
