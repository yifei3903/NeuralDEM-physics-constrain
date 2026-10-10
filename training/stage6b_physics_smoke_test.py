"""Stage-6B higher-order spacetime-continuity smoke test."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_SOURCES, ContinuousTimeConfig, DataConfig, ModelConfig,
    Stage6BConfig, TrainConfig,
)
from continuous_time_utils import (
    compute_multi_horizon_losses, continuous_model_inputs,
    load_legacy_initial_state, make_pointwise_tau, make_spacetime_coords,
)
from dataset import ContinuousTimeCFDDEMDataset
from inference import build_model
from stage6b_utils import (
    TERM_NAMES, continuity_terms, correlation, model_parameter_groups,
    parameter_group_norm, rms,
)
from train import set_seed


def forward_with_physics(model, batch, data_cfg, device, encode_calls=None):
    tau_query = make_pointwise_tau(
        batch["query_tau_values"], data_cfg.n_cells, device,
        requires_grad=True)
    coords = make_spacetime_coords(
        batch["output_fluid_coords"], tau_query.shape[1], device,
        requires_grad=True)
    hook = None
    if encode_calls is not None:
        hook = model.to_fluid_token.register_forward_hook(
            lambda *_: encode_calls.__setitem__("count", encode_calls["count"] + 1))
    prediction = model(**continuous_model_inputs(
        batch, device, tau_query, output_coords=coords))
    if hook is not None:
        hook.remove()
    terms = continuity_terms(
        prediction["fluid"], coords, tau_query, data_cfg,
        create_graph=True)
    return prediction, terms, coords, tau_query


def main():
    data_cfg = DataConfig()
    model_cfg = ModelConfig(model_scale="small")
    train_cfg = TrainConfig(use_mass_constraint=True, lambda_mass=1e-3)
    ct_cfg, physics_cfg = ContinuousTimeConfig(), Stage6BConfig()
    if train_cfg.lambda_mass != physics_cfg.lambda_mass:
        raise RuntimeError("Stage-6B lambda mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(train_cfg.seed)
    dataset = ContinuousTimeCFDDEMDataset(
        data_cfg, model_cfg, DATA_SOURCES[0],
        num_anchor_states=ct_cfg.num_anchor_states,
        queries_per_anchor=ct_cfg.queries_per_anchor,
        min_offset=ct_cfg.min_offset, max_offset=ct_cfg.max_offset,
        seed=train_cfg.seed)
    batch = dataset[0]
    model = build_model(model_cfg, device)
    initial_path = os.path.join(
        train_cfg.debug_output_dir, "shared_initial_state.pt")
    load_legacy_initial_state(model, initial_path, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay)

    encode_calls = {"count": 0}
    prediction, terms, _, _ = forward_with_physics(
        model, batch, data_cfg, device, encode_calls)
    data_losses = compute_multi_horizon_losses(
        prediction, batch, device, train_cfg)
    mass_loss = terms["R"].square().mean()
    weighted_mass = physics_cfg.lambda_mass * mass_loss
    total_loss = data_losses["total_data_loss"] + weighted_mass
    groups = model_parameter_groups(model)
    initial_physics_gradients = {
        name: parameter_group_norm(mass_loss, parameters, retain_graph=True)
        for name, parameters in groups.items()
    }
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    total_gradient_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # The Stage-6 zero gates can make backbone gradients exactly zero at the
    # initial state. Re-evaluate physics-only gradients after one valid update.
    optimizer.zero_grad(set_to_none=True)
    prediction_after, terms_after, _, tau_after = forward_with_physics(
        model, batch, data_cfg, device)
    mass_loss_after = terms_after["R"].square().mean()
    groups_after = model_parameter_groups(model)
    post_step_physics_gradients = {
        name: parameter_group_norm(
            mass_loss_after, parameters, retain_graph=True)
        for name, parameters in groups_after.items()
    }

    # Pointwise time-query independence after the physics-enabled update.
    model.eval()
    with torch.no_grad():
        encoded = model.encode_state(**{
            key: batch[key].to(device) for key in (
                "inlet_velocity", "input_particle_coords",
                "input_particle_displacement", "supernode_idx",
                "input_fluid_coords", "input_fluid_state")})
        k = tau_after.shape[1]
        condition = model._expand_latent_for_queries(encoded["condition"], k)
        latent = model._expand_latent_for_queries(encoded["fluid_tokens"], k)
        coords = make_spacetime_coords(
            batch["output_fluid_coords"], k, device).reshape(
                -1, data_cfg.n_cells, 3)
        base_tau = tau_after.detach().reshape(-1, data_cfg.n_cells, 1)
        base = model.fluid_decoder(
            latent, model._normalize_fluid_coords(coords),
            block_kwargs={"cond": condition}, query_time=base_tau)
        perturbed_tau = base_tau.clone()
        perturbed_tau[0, 17, 0] += 1e-4
        perturbed = model.fluid_decoder(
            latent, model._normalize_fluid_coords(coords),
            block_kwargs={"cond": condition}, query_time=perturbed_tau)
        changed = (perturbed[0, 17] - base[0, 17]).abs().max().item()
        unchanged_mask = torch.ones_like(base[..., 0], dtype=torch.bool)
        unchanged_mask[0, 17] = False
        unchanged = (perturbed - base).abs()[unchanged_mask].max().item()

    term_stats = {f"RMS_{name}": rms(terms[name]).item()
                  for name in TERM_NAMES}
    checks = {
        "device": str(device),
        "lambda_mass": physics_cfg.lambda_mass,
        "L_data": data_losses["total_data_loss"].detach().item(),
        "L_mass_raw": mass_loss.detach().item(),
        "L_mass_weighted": weighted_mass.detach().item(),
        "L_total": total_loss.detach().item(),
        **term_stats,
        "corr_A_B": correlation(terms["A"], terms["B"]).item(),
        "corr_Dy_Dz": correlation(terms["Dy"], terms["Dz"]).item(),
        "forward_finite": bool(torch.isfinite(prediction["fluid"]).all()),
        "all_terms_finite": all(
            bool(torch.isfinite(terms[name]).all()) for name in TERM_NAMES),
        "losses_finite": all(bool(torch.isfinite(value)) for value in (
            data_losses["total_data_loss"], mass_loss, weighted_mass, total_loss)),
        "backward_gradients_finite": bool(total_gradient_finite),
        "optimizer_step_success": True,
        "initial_physics_gradients": initial_physics_gradients,
        "post_step_physics_gradients": post_step_physics_gradients,
        "encoder_calls_for_k_queries": encode_calls["count"],
        "queries_per_anchor": int(tau_after.shape[1]),
        "perturbed_query_change": changed,
        "other_query_max_change": unchanged,
        "has_stage7_dynamics_generator": hasattr(model, "predict_dynamics"),
        "nan_or_inf": False,
    }

    if checks["encoder_calls_for_k_queries"] != 1:
        raise AssertionError("K=4 did not reuse one encoder call")
    if unchanged > 1e-7 or changed == 0.0:
        raise AssertionError("pointwise query independence failed")
    if checks["has_stage7_dynamics_generator"]:
        raise AssertionError("Stage-7 dynamics generator remains in Stage-6B")
    if not (checks["forward_finite"] and checks["all_terms_finite"]
            and checks["losses_finite"] and checks["backward_gradients_finite"]):
        raise FloatingPointError("non-finite Stage-6B smoke result")
    for name in ("fluid_decoder", "transformer", "fluid_encoder"):
        result = post_step_physics_gradients[name]
        if result["parameters_with_gradient"] == 0 or result["norm"] <= 0 or not result["finite"]:
            raise AssertionError(f"physics gradient did not reach {name}")
    particle = post_step_physics_gradients["particle_decoder"]
    if particle["parameters_with_gradient"] != 0 or particle["norm"] != 0:
        raise AssertionError("gas continuity directly reached particle decoder")

    os.makedirs(physics_cfg.output_dir, exist_ok=True)
    path = os.path.join(physics_cfg.output_dir, "physics_smoke_test.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
