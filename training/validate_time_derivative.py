"""Compare Stage-6 pointwise time AD with raw-CFD temporal finite differences."""

import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, ContinuousTimeConfig, DataConfig, ModelConfig, TrainConfig
from continuous_time_utils import make_pointwise_tau, make_spacetime_coords
from dataset import ContinuousTimeCFDDEMDataset, denormalize_pressure
from evaluate_continuous_time import load_stage6_model, metrics
from physics_losses import (
    gas_density_from_pressure,
    spacetime_continuity_residual,
)
from train import set_seed


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values ** 2)))


def correlation(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.sqrt(np.sum(first ** 2) * np.sum(second ** 2))
    return float(np.sum(first * second) / max(denominator, 1e-30))


def physical_mass(state, data_cfg):
    pressure = denormalize_pressure(state[..., 4:5], data_cfg)
    density = pressure * data_cfg.molecular_weight / (
        data_cfg.gas_constant * data_cfg.gas_temperature)
    return state[..., 3:4] * density


def main():
    data_cfg, model_cfg = DataConfig(), ModelConfig(model_scale="small")
    train_cfg, ct_cfg = TrainConfig(), ContinuousTimeConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)
    checkpoint = os.path.join(
        ct_cfg.output_dir, "checkpoints", "continuous_time_epoch_100.pt")
    model = load_stage6_model(checkpoint, model_cfg, device)
    dataset = ContinuousTimeCFDDEMDataset(
        data_cfg, model_cfg, DATA_SOURCES[0],
        num_anchor_states=ct_cfg.num_anchor_states,
        queries_per_anchor=ct_cfg.queries_per_anchor,
        min_offset=ct_cfg.min_offset, max_offset=ct_cfg.max_offset,
        seed=train_cfg.seed)
    offsets = [int(round(tau / data_cfg.original_dt)) for tau in ct_cfg.derivative_taus]
    derivative_values = {
        (offset, quantity, source): []
        for offset in offsets
        for quantity in ("epsilon_dt", "pressure_dt", "mass_dt")
        for source in ("ad", "cfd_fd")
    }
    continuity_values = {
        (offset, term): []
        for offset in offsets
        for term in ("A_dm_dt", "B_divergence", "R_mass", "dJx_dx", "dJy_dy", "dJz_dz")
    }

    for anchor_index in range(len(dataset)):
        batch = dataset.get_with_offsets(anchor_index, offsets)
        encoded = model.encode_state(**{
            key: batch[key].to(device) for key in (
                "inlet_velocity", "input_particle_coords", "input_particle_displacement",
                "supernode_idx", "input_fluid_coords", "input_fluid_state")})
        tau_query = make_pointwise_tau(
            batch["query_tau_values"], data_cfg.n_cells, device, requires_grad=True)
        coords = make_spacetime_coords(
            batch["output_fluid_coords"], len(offsets), device, requires_grad=True)
        prediction = model.decode_fluid_only(encoded, coords, tau_query)

        epsilon_dt = torch.autograd.grad(
            prediction[..., 3:4].sum(), tau_query,
            retain_graph=True, create_graph=False)[0]
        pressure_dt = torch.autograd.grad(
            prediction[..., 4:5].sum(), tau_query,
            retain_graph=True, create_graph=False)[0] * data_cfg.pressure_scale
        pressure = (
            prediction[..., 4:5] * data_cfg.pressure_scale + data_cfg.pressure_ref)
        density = gas_density_from_pressure(
            pressure, data_cfg.molecular_weight,
            data_cfg.gas_constant, data_cfg.gas_temperature)
        mass = prediction[..., 3:4] * density
        mass_dt = torch.autograd.grad(
            mass.sum(), tau_query, retain_graph=True, create_graph=False)[0]

        residual, terms = spacetime_continuity_residual(
            prediction, coords, tau_query,
            data_cfg.pressure_ref, data_cfg.pressure_scale,
            data_cfg.molecular_weight, data_cfg.gas_constant,
            data_cfg.gas_temperature, create_graph=False, return_terms=True)
        ad_quantities = {
            "epsilon_dt": epsilon_dt.detach().cpu().numpy()[0],
            "pressure_dt": pressure_dt.detach().cpu().numpy()[0],
            "mass_dt": mass_dt.detach().cpu().numpy()[0],
        }
        term_tensors = {
            "A_dm_dt": terms["dm_dt"],
            "B_divergence": terms["divergence"],
            "R_mass": residual,
            "dJx_dx": terms["dJx_dx"],
            "dJy_dy": terms["dJy_dy"],
            "dJz_dz": terms["dJz_dz"],
        }

        for query_index, offset in enumerate(offsets):
            state_minus = dataset._fluid_state(batch["anchor_frame"] + offset - 1)
            state_plus = dataset._fluid_state(batch["anchor_frame"] + offset + 1)
            cfd_quantities = {
                "epsilon_dt": (
                    state_plus[..., 3:4] - state_minus[..., 3:4])
                    / (2.0 * data_cfg.original_dt),
                "pressure_dt": (
                    denormalize_pressure(state_plus[..., 4:5], data_cfg)
                    - denormalize_pressure(state_minus[..., 4:5], data_cfg))
                    / (2.0 * data_cfg.original_dt),
                "mass_dt": (
                    physical_mass(state_plus, data_cfg)
                    - physical_mass(state_minus, data_cfg))
                    / (2.0 * data_cfg.original_dt),
            }
            for quantity in cfd_quantities:
                derivative_values[(offset, quantity, "ad")].append(
                    ad_quantities[quantity][query_index])
                derivative_values[(offset, quantity, "cfd_fd")].append(
                    cfd_quantities[quantity])
            for name, tensor in term_tensors.items():
                continuity_values[(offset, name)].append(
                    tensor.detach().cpu().numpy()[0, query_index])

    derivative_rows = []
    for offset in offsets:
        for quantity in ("epsilon_dt", "pressure_dt", "mass_dt"):
            ad = np.concatenate(derivative_values[(offset, quantity, "ad")]).reshape(-1)
            reference = np.concatenate(
                derivative_values[(offset, quantity, "cfd_fd")]).reshape(-1)
            row = {
                "tau_seconds": offset * data_cfg.original_dt,
                "quantity": quantity,
                **metrics(ad, reference),
                "ad_rms": rms(ad),
                "cfd_temporal_fd_rms": rms(reference),
                "ad_to_cfd_rms_ratio": rms(ad) / max(rms(reference), 1e-30),
            }
            derivative_rows.append(row)

    continuity_rows = []
    for offset in offsets:
        values = {
            term: np.concatenate(continuity_values[(offset, term)]).reshape(-1)
            for term in ("A_dm_dt", "B_divergence", "R_mass", "dJx_dx", "dJy_dy", "dJz_dz")
        }
        rms_a, rms_b, rms_r = rms(values["A_dm_dt"]), rms(values["B_divergence"]), rms(values["R_mass"])
        continuity_rows.append({
            "tau_seconds": offset * data_cfg.original_dt,
            "temporal_rms": rms_a,
            "divergence_rms": rms_b,
            "residual_rms": rms_r,
            "corr_temporal_divergence": correlation(
                values["A_dm_dt"], values["B_divergence"]),
            "cancellation_ratio": rms_r / max(rms_a + rms_b, 1e-30),
            "dJx_dx_rms": rms(values["dJx_dx"]),
            "dJy_dy_rms": rms(values["dJy_dy"]),
            "dJz_dz_rms": rms(values["dJz_dz"]),
        })

    derivative_summary = {
        "method": "pointwise AD versus centered raw-CFD FD with dt=0.001 s",
        "rows": derivative_rows,
        "nan_or_inf": not all(
            np.isfinite(value)
            for row in derivative_rows for key, value in row.items()
            if key not in ("quantity",)),
    }
    continuity_summary = {
        "method": "dm/dtau_AD + spatial_divergence_AD; data-only checkpoint",
        "rows": continuity_rows,
        "nan_or_inf": not all(
            np.isfinite(value) for row in continuity_rows for value in row.values()),
    }
    write_csv(os.path.join(ct_cfg.output_dir, "time_derivative_vs_cfd.csv"), derivative_rows)
    write_csv(os.path.join(ct_cfg.output_dir, "spacetime_continuity.csv"), continuity_rows)
    with open(os.path.join(ct_cfg.output_dir, "time_derivative_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(derivative_summary, handle, indent=2)
    with open(os.path.join(ct_cfg.output_dir, "spacetime_continuity_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(continuity_summary, handle, indent=2)
    print(json.dumps({
        "time_derivative": derivative_summary,
        "spacetime_continuity": continuity_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
