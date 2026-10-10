"""Controlled Stage-6 versus Stage-6B post-training diagnostics."""

import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_SOURCES, ContinuousTimeConfig, DataConfig, ModelConfig,
    Stage6BConfig, TrainConfig,
)
from continuous_time_utils import make_pointwise_tau, make_spacetime_coords
from dataset import (
    ContinuousTimeCFDDEMDataset, denormalize_pressure, read_particle_file,
)
from evaluate_continuous_time import evaluate_horizons, load_stage6_model
from inference import evaluate_rollout, run_autoregressive
from physics_losses import gas_density_from_pressure
from physics_validation_utils import cfd_directional_divergence, fluid_flux
from stage6b_utils import continuity_terms, numpy_metrics
from train import set_seed


DERIVATIVE_QUANTITIES = ("epsilon_dt", "pressure_dt", "mass_dt")
CONTINUITY_NAMES = ("A", "Dx", "Dy", "Dz", "B", "R")
FLUX_NAMES = ("Jx", "Jy", "Jz", "dJx_dx", "dJy_dy", "dJz_dz", "divergence")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlation(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(np.sum(left ** 2) * np.sum(right ** 2))
    return float(np.sum(left * right) / max(denominator, 1e-30))


def array_rms(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values ** 2)))


def physical_mass(state, data_cfg):
    pressure = denormalize_pressure(state[..., 4:5], data_cfg)
    density = pressure * data_cfg.molecular_weight / (
        data_cfg.gas_constant * data_cfg.gas_temperature)
    return state[..., 3:4] * density


def collect_derivative_physics_flux(model, dataset, data_cfg, ct_cfg,
                                    device, model_name, lambda_mass):
    offsets = [int(round(tau / data_cfg.original_dt))
               for tau in ct_cfg.derivative_taus]
    derivative = {
        (offset, quantity, source): []
        for offset in offsets for quantity in DERIVATIVE_QUANTITIES
        for source in ("nn_ad", "cfd_centered_fd")
    }
    continuity = {
        (offset, name): []
        for offset in offsets for name in CONTINUITY_NAMES
    }
    fluxes = {
        (offset, name, source): []
        for offset in offsets for name in FLUX_NAMES
        for source in ("nn", "cfd")
    }

    model.eval()
    for anchor_index in range(len(dataset)):
        batch = dataset.get_with_offsets(anchor_index, offsets)
        encoded = model.encode_state(**{
            key: batch[key].to(device) for key in (
                "inlet_velocity", "input_particle_coords",
                "input_particle_displacement", "supernode_idx",
                "input_fluid_coords", "input_fluid_state")})
        tau_query = make_pointwise_tau(
            batch["query_tau_values"], data_cfg.n_cells, device,
            requires_grad=True)
        coords = make_spacetime_coords(
            batch["output_fluid_coords"], len(offsets), device,
            requires_grad=True)
        prediction = model.decode_fluid_only(encoded, coords, tau_query)

        epsilon_dt = torch.autograd.grad(
            prediction[..., 3:4].sum(), tau_query,
            retain_graph=True, create_graph=False)[0]
        pressure_dt = torch.autograd.grad(
            prediction[..., 4:5].sum(), tau_query,
            retain_graph=True, create_graph=False)[0] * data_cfg.pressure_scale
        pressure = (
            prediction[..., 4:5] * data_cfg.pressure_scale
            + data_cfg.pressure_ref)
        density = gas_density_from_pressure(
            pressure, data_cfg.molecular_weight,
            data_cfg.gas_constant, data_cfg.gas_temperature)
        mass = prediction[..., 3:4] * density
        mass_dt = torch.autograd.grad(
            mass.sum(), tau_query, retain_graph=True,
            create_graph=False)[0]
        terms = continuity_terms(
            prediction, coords, tau_query, data_cfg, create_graph=False)
        pred_flux, _ = fluid_flux(prediction, data_cfg)

        ad_derivatives = {
            "epsilon_dt": epsilon_dt.detach().cpu().numpy()[0],
            "pressure_dt": pressure_dt.detach().cpu().numpy()[0],
            "mass_dt": mass_dt.detach().cpu().numpy()[0],
        }
        for query_index, offset in enumerate(offsets):
            anchor = batch["anchor_frame"]
            state_minus = dataset._fluid_state(anchor + offset - 1)
            state_plus = dataset._fluid_state(anchor + offset + 1)
            cfd_state_np = dataset._fluid_state(anchor + offset)
            cfd_derivatives = {
                "epsilon_dt": (
                    state_plus[..., 3:4] - state_minus[..., 3:4])
                    / (2 * data_cfg.original_dt),
                "pressure_dt": (
                    denormalize_pressure(state_plus[..., 4:5], data_cfg)
                    - denormalize_pressure(state_minus[..., 4:5], data_cfg))
                    / (2 * data_cfg.original_dt),
                "mass_dt": (
                    physical_mass(state_plus, data_cfg)
                    - physical_mass(state_minus, data_cfg))
                    / (2 * data_cfg.original_dt),
            }
            for quantity in DERIVATIVE_QUANTITIES:
                derivative[(offset, quantity, "nn_ad")].append(
                    ad_derivatives[quantity][query_index])
                derivative[(offset, quantity, "cfd_centered_fd")].append(
                    cfd_derivatives[quantity])

            for name in CONTINUITY_NAMES:
                continuity[(offset, name)].append(
                    terms[name].detach().cpu().numpy()[0, query_index])

            cfd_state = torch.from_numpy(cfd_state_np).unsqueeze(0).to(device)
            cfd_flux, _ = fluid_flux(cfd_state, data_cfg)
            cfd_directional, _ = cfd_directional_divergence(
                cfd_state, data_cfg)
            nn_flux_q = pred_flux[:, query_index]
            nn_flux_values = {
                "Jx": nn_flux_q[..., 0:1],
                "Jy": nn_flux_q[..., 1:2],
                "Jz": nn_flux_q[..., 2:3],
                "dJx_dx": terms["Dx"][:, query_index],
                "dJy_dy": terms["Dy"][:, query_index],
                "dJz_dz": terms["Dz"][:, query_index],
                "divergence": terms["B"][:, query_index],
            }
            cfd_flux_values = {
                "Jx": cfd_flux[..., 0:1],
                "Jy": cfd_flux[..., 1:2],
                "Jz": cfd_flux[..., 2:3],
                "dJx_dx": cfd_directional[0],
                "dJy_dy": cfd_directional[1],
                "dJz_dz": cfd_directional[2],
                "divergence": sum(cfd_directional),
            }
            for name in FLUX_NAMES:
                fluxes[(offset, name, "nn")].append(
                    nn_flux_values[name].detach().cpu().numpy())
                fluxes[(offset, name, "cfd")].append(
                    cfd_flux_values[name].detach().cpu().numpy())

    derivative_rows = []
    physics_rows = []
    flux_rows = []
    for offset in offsets:
        tau = offset * data_cfg.original_dt
        for quantity in DERIVATIVE_QUANTITIES:
            predicted = np.concatenate(
                derivative[(offset, quantity, "nn_ad")]).reshape(-1)
            reference = np.concatenate(
                derivative[(offset, quantity, "cfd_centered_fd")]).reshape(-1)
            derivative_rows.append({
                "model": model_name,
                "lambda_mass": lambda_mass,
                "tau_seconds": tau,
                "quantity": quantity,
                **numpy_metrics(predicted, reference),
            })

        values = {
            name: np.concatenate(continuity[(offset, name)]).reshape(-1)
            for name in CONTINUITY_NAMES
        }
        rms_values = {name: array_rms(value) for name, value in values.items()}
        physics_rows.append({
            "model": model_name,
            "lambda_mass": lambda_mass,
            "tau_seconds": tau,
            **{f"RMS_{name}": rms_values[name] for name in CONTINUITY_NAMES},
            "corr_A_B": correlation(values["A"], values["B"]),
            "corr_Dy_Dz": correlation(values["Dy"], values["Dz"]),
            "continuity_cancellation": (
                rms_values["R"] / max(rms_values["A"] + rms_values["B"], 1e-30)),
            "directional_cancellation_Dy_Dz": (
                array_rms(values["Dy"] + values["Dz"])
                / max(rms_values["Dy"] + rms_values["Dz"], 1e-30)),
        })
        for quantity in FLUX_NAMES:
            predicted = np.concatenate(
                fluxes[(offset, quantity, "nn")]).reshape(-1)
            reference = np.concatenate(
                fluxes[(offset, quantity, "cfd")]).reshape(-1)
            flux_rows.append({
                "model": model_name,
                "lambda_mass": lambda_mass,
                "tau_seconds": tau,
                "quantity": quantity,
                "prediction_source": "NN field / pointwise spatial AD",
                "reference_source": "sampled CFD field / grid FD",
                **numpy_metrics(predicted, reference),
            })
    return derivative_rows, physics_rows, flux_rows


def rollout_rows(model, model_name, lambda_mass, model_cfg, data_cfg,
                 source, device):
    rows = []
    for tau, steps in ((0.1, 5), (0.05, 2)):
        predictions = run_autoregressive(
            model, model_cfg, data_cfg, source, start_frame=0,
            num_steps=steps, device=device, compute_mass_metrics=False,
            tau_test=tau)
        field_metrics = evaluate_rollout(
            predictions, source, 0, data_cfg,
            steps=tuple(range(1, steps + 1)), tau_test=tau)
        stride = int(round(tau / data_cfg.original_dt))
        for step in range(1, steps + 1):
            prediction = predictions[step - 1]
            _, true_coords, true_count = read_particle_file(
                os.path.join(
                    source["particle_dir"],
                    f"particles_{step * stride}.txt"),
                data_cfg.max_particles)
            count = min(len(prediction["particle_coords"]), true_count)
            particle_difference = (
                prediction["particle_coords"][:count] - true_coords[:count])
            epsilon = prediction["fluid_voidfraction"]
            pressure = prediction["fluid_pressure"]
            all_values = np.concatenate([
                prediction["fluid_velocity"].reshape(-1),
                epsilon.reshape(-1), pressure.reshape(-1),
                prediction["particle_coords"].reshape(-1)])
            metrics = field_metrics[step]
            rows.append({
                "model": model_name,
                "lambda_mass": lambda_mass,
                "tau_test_seconds": tau,
                "step": step,
                "horizon_seconds": tau * step,
                "velocity_mae": metrics["velocity_mae"],
                "epsilon_mae": metrics["voidfraction_mae"],
                "pressure_mae_pa": metrics["pressure_mae_pa"],
                "particle_position_mae": float(
                    np.mean(np.abs(particle_difference))),
                "particle_position_rmse": float(
                    np.sqrt(np.mean(particle_difference ** 2))),
                "epsilon_below_zero": int(np.sum(epsilon < 0)),
                "epsilon_above_one": int(np.sum(epsilon > 1)),
                "pressure_nonpositive": int(np.sum(pressure <= 0)),
                "nan_count": int(np.sum(np.isnan(all_values))),
                "inf_count": int(np.sum(np.isinf(all_values))),
            })
    return rows


def comparison_rows(fixed_rows, physics_rows, derivative_rows, flux_rows,
                    rollout_metrics):
    rows = []
    sources = {
        "fixed_horizon": (fixed_rows, ("tau_seconds", "field"), "rmse"),
        "continuity": (physics_rows, ("tau_seconds",), "RMS_R"),
        "mass_time_derivative_ratio": (
            [row for row in derivative_rows if row["quantity"] == "mass_dt"],
            ("tau_seconds",), "rms_ratio"),
        "divergence_fidelity": (
            [row for row in flux_rows if row["quantity"] == "divergence"],
            ("tau_seconds",), "relative_l2"),
        "rollout_velocity": (
            rollout_metrics, ("tau_test_seconds", "step"), "velocity_mae"),
    }
    for category, (source, key_names, metric_name) in sources.items():
        by_key = {}
        for row in source:
            key = tuple(row[name] for name in key_names)
            by_key.setdefault(key, {})[row["model"]] = row[metric_name]
        for key, models in by_key.items():
            if "stage6" not in models or "stage6b" not in models:
                continue
            result = {
                "category": category,
                "metric": metric_name,
                "stage6": models["stage6"],
                "stage6b": models["stage6b"],
                "stage6b_minus_stage6": models["stage6b"] - models["stage6"],
                "relative_change": (
                    (models["stage6b"] - models["stage6"])
                    / max(abs(models["stage6"]), 1e-30)),
            }
            result.update({name: value for name, value in zip(key_names, key)})
            rows.append(result)
    # Normalize column order across heterogeneous keys.
    columns = (
        "category", "metric", "tau_seconds", "field",
        "tau_test_seconds", "step", "stage6", "stage6b",
        "stage6b_minus_stage6", "relative_change")
    return [{name: row.get(name, "") for name in columns} for row in rows]


def main():
    data_cfg = DataConfig()
    model_cfg = ModelConfig(model_scale="small")
    train_cfg = TrainConfig()
    ct_cfg, physics_cfg = ContinuousTimeConfig(), Stage6BConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)
    dataset = ContinuousTimeCFDDEMDataset(
        data_cfg, model_cfg, DATA_SOURCES[0],
        num_anchor_states=ct_cfg.num_anchor_states,
        queries_per_anchor=ct_cfg.queries_per_anchor,
        min_offset=ct_cfg.min_offset, max_offset=ct_cfg.max_offset,
        seed=train_cfg.seed)
    checkpoints = {
        "stage6": (
            os.path.join(ct_cfg.output_dir, "checkpoints",
                         "continuous_time_epoch_100.pt"), 0.0),
        "stage6b": (
            os.path.join(physics_cfg.output_dir, "checkpoints", "final.pt"),
            physics_cfg.lambda_mass),
    }

    fixed_rows, derivative_rows, physics_rows = [], [], []
    flux_rows, rollout_metrics = [], []
    for model_name, (checkpoint, lambda_mass) in checkpoints.items():
        model = load_stage6_model(checkpoint, model_cfg, device)
        for row in evaluate_horizons(model, dataset, data_cfg, ct_cfg, device):
            fixed_rows.append({
                "model": model_name,
                "lambda_mass": lambda_mass,
                **row,
            })
        derivative, physics, flux = collect_derivative_physics_flux(
            model, dataset, data_cfg, ct_cfg, device,
            model_name, lambda_mass)
        derivative_rows.extend(derivative)
        physics_rows.extend(physics)
        flux_rows.extend(flux)
        rollout_metrics.extend(rollout_rows(
            model, model_name, lambda_mass, model_cfg, data_cfg,
            DATA_SOURCES[0], device))

    comparison = comparison_rows(
        fixed_rows, physics_rows, derivative_rows, flux_rows,
        rollout_metrics)
    output_dir = physics_cfg.output_dir
    write_csv(os.path.join(output_dir, "fixed_horizon_metrics.csv"), fixed_rows)
    write_csv(os.path.join(output_dir, "physics_diagnostics.csv"), physics_rows)
    write_csv(os.path.join(output_dir, "time_derivative_metrics.csv"), derivative_rows)
    write_csv(os.path.join(output_dir, "flux_divergence_metrics.csv"), flux_rows)
    write_csv(os.path.join(output_dir, "rollout_metrics.csv"), rollout_metrics)
    write_csv(os.path.join(output_dir, "stage6_vs_stage6b.csv"), comparison)
    summary = {
        "models": list(checkpoints),
        "fixed_horizon_rows": len(fixed_rows),
        "physics_rows": len(physics_rows),
        "time_derivative_rows": len(derivative_rows),
        "flux_rows": len(flux_rows),
        "rollout_rows": len(rollout_metrics),
        "comparison_rows": len(comparison),
        "nan_or_inf": not all(
            np.isfinite(float(value))
            for collection in (fixed_rows, physics_rows, derivative_rows,
                               flux_rows, rollout_metrics)
            for row in collection
            for key, value in row.items()
            if key not in ("model", "field", "quantity",
                           "prediction_source", "reference_source")),
    }
    with open(os.path.join(output_dir, "evaluation_summary.json"),
              "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
