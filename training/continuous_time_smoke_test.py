"""Minimal Stage-6 pointwise relative-time and backward smoke test."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, ContinuousTimeConfig, DataConfig, ModelConfig, TrainConfig
from continuous_time_utils import (
    compute_multi_horizon_losses,
    continuous_model_inputs,
    load_legacy_initial_state,
    make_pointwise_tau,
)
from dataset import ContinuousTimeCFDDEMDataset
from inference import build_model
from physics_losses import (
    denormalize_pressure,
    gas_density_from_pressure,
    spacetime_continuity_residual,
)
from train import set_seed


def main():
    data_cfg, model_cfg = DataConfig(), ModelConfig(model_scale="small")
    train_cfg, ct_cfg = TrainConfig(use_mass_constraint=False, lambda_mass=0.0), ContinuousTimeConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)
    dataset = ContinuousTimeCFDDEMDataset(
        data_cfg, model_cfg, DATA_SOURCES[0], num_anchor_states=ct_cfg.num_anchor_states,
        queries_per_anchor=ct_cfg.queries_per_anchor, min_offset=ct_cfg.min_offset,
        max_offset=ct_cfg.max_offset, seed=train_cfg.seed,
    )
    batch = dataset[0]
    model = build_model(model_cfg, device)
    initial_path = os.path.join(
        train_cfg.debug_output_dir, "shared_initial_state.pt")
    load_legacy_initial_state(model, initial_path, device)

    tau_query = make_pointwise_tau(
        batch["query_tau_values"], data_cfg.n_cells, device, requires_grad=True)
    coords = batch["output_fluid_coords"].to(device)
    coords = coords[:, None].expand(-1, tau_query.shape[1], -1, -1).clone().detach()
    coords.requires_grad_(True)

    encode_calls = {"count": 0}
    hook = model.to_fluid_token.register_forward_hook(
        lambda *_: encode_calls.__setitem__("count", encode_calls["count"] + 1))
    prediction = model(**continuous_model_inputs(
        batch, device, tau_query, output_coords=coords))
    hook.remove()
    losses = compute_multi_horizon_losses(prediction, batch, device, train_cfg)
    losses["total_data_loss"].backward(retain_graph=True)
    gradient_norm = torch.sqrt(sum(
        parameter.grad.detach().square().sum()
        for parameter in model.parameters() if parameter.grad is not None)).item()

    pressure = denormalize_pressure(
        prediction["fluid"][..., 4:5], data_cfg.pressure_ref, data_cfg.pressure_scale)
    density = gas_density_from_pressure(
        pressure, data_cfg.molecular_weight, data_cfg.gas_constant, data_cfg.gas_temperature)
    mass = prediction["fluid"][..., 3:4] * density
    dm_dt = torch.autograd.grad(
        mass.sum(), tau_query, create_graph=False, retain_graph=True)[0]
    residual, terms = spacetime_continuity_residual(
        prediction["fluid"], coords, tau_query,
        data_cfg.pressure_ref, data_cfg.pressure_scale,
        data_cfg.molecular_weight, data_cfg.gas_constant, data_cfg.gas_temperature,
        create_graph=False, return_terms=True,
    )

    tau_probe = torch.tensor([[[0.037]]], device=device, requires_grad=True)
    normalized = model.fluid_decoder.normalize_time(tau_probe)
    normalization_derivative = torch.autograd.grad(normalized.sum(), tau_probe)[0]

    # A pointwise independence test with fixed sampled particle queries: decode
    # fluid only so particle rejection-sampling randomness cannot enter.
    model.eval()
    with torch.no_grad():
        encoded = model.encode_state(**{
            key: batch[key].to(device) for key in (
                "inlet_velocity", "input_particle_coords", "input_particle_displacement",
                "supernode_idx", "input_fluid_coords", "input_fluid_state")})
        condition = model._expand_latent_for_queries(encoded["condition"], tau_query.shape[1])
        latent = model._expand_latent_for_queries(encoded["fluid_tokens"], tau_query.shape[1])
        flat_coords = coords.reshape(-1, data_cfg.n_cells, 3)
        base_tau = tau_query.detach().reshape(-1, data_cfg.n_cells, 1)
        base = model.fluid_decoder(
            latent, model._normalize_fluid_coords(flat_coords),
            block_kwargs={"cond": condition}, query_time=base_tau)
        perturbed_tau = base_tau.clone()
        perturbed_tau[0, 17, 0] += 1.0e-4
        perturbed = model.fluid_decoder(
            latent, model._normalize_fluid_coords(flat_coords),
            block_kwargs={"cond": condition}, query_time=perturbed_tau)
        changed = (perturbed[0, 17] - base[0, 17]).abs().max().item()
        unchanged_mask = torch.ones_like(base[..., 0], dtype=torch.bool)
        unchanged_mask[0, 17] = False
        unchanged = (perturbed - base).abs()[unchanged_mask].max().item()

    checks = {
        "device": str(device),
        "tau_requires_grad": tau_query.requires_grad,
        "fluid_shape": list(prediction["fluid"].shape),
        "particle_shape": list(prediction["particle_displacement"].shape),
        "tau_gradient_shape": list(dm_dt.shape),
        "tau_gradient_finite": bool(torch.isfinite(dm_dt).all()),
        "residual_finite": bool(torch.isfinite(residual).all()),
        "total_loss": losses["total_data_loss"].detach().item(),
        "parameter_gradient_norm": gradient_norm,
        "tau_normalization_derivative": normalization_derivative.item(),
        "encoder_calls_for_k_queries": encode_calls["count"],
        "perturbed_query_change": changed,
        "other_query_max_change": unchanged,
        "dm_dt_rms": terms["dm_dt"].square().mean().sqrt().item(),
        "residual_rms": residual.square().mean().sqrt().item(),
        "nan_or_inf": not all((
            torch.isfinite(losses["total_data_loss"]),
            torch.isfinite(dm_dt).all(),
            torch.isfinite(residual).all(),
        )),
    }
    if checks["encoder_calls_for_k_queries"] != 1:
        raise AssertionError("state encoder ran more than once for K queries")
    if abs(checks["tau_normalization_derivative"] - 10.0) > 1e-6:
        raise AssertionError("tau normalization derivative is not 10 s^-1")
    if unchanged > 1e-7 or changed == 0.0:
        raise AssertionError("pointwise query-time independence failed")
    if checks["nan_or_inf"]:
        raise FloatingPointError("non-finite smoke-test result")
    os.makedirs(ct_cfg.output_dir, exist_ok=True)
    with open(os.path.join(ct_cfg.output_dir, "smoke_test.json"), "w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
