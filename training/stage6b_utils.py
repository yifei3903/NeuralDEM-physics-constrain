"""Shared Stage-6B continuous spacetime physics helpers."""

import math

import numpy as np
import torch

from physics_losses import spacetime_continuity_residual


TERM_NAMES = ("A", "Dx", "Dy", "Dz", "B", "R")


def continuity_terms(prediction, coords, tau_query, data_cfg, create_graph=True):
    residual, raw = spacetime_continuity_residual(
        prediction, coords, tau_query,
        data_cfg.pressure_ref, data_cfg.pressure_scale,
        data_cfg.molecular_weight, data_cfg.gas_constant,
        data_cfg.gas_temperature, create_graph=create_graph,
        return_terms=True,
    )
    return {
        "A": raw["dm_dt"],
        "Dx": raw["dJx_dx"],
        "Dy": raw["dJy_dy"],
        "Dz": raw["dJz_dz"],
        "B": raw["divergence"],
        "R": residual,
        "mass": raw["mass"],
        "rho": raw["rho"],
    }


def rms(tensor):
    return tensor.double().square().mean().sqrt()


def correlation(left, right):
    left = left.double().reshape(-1)
    right = right.double().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if denominator.item() == 0.0:
        return left.new_tensor(float("nan"))
    return (left * right).sum() / denominator


def detached_epoch_diagnostics(collected):
    values = {
        name: torch.cat(collected[name]).double()
        for name in TERM_NAMES
    }
    rms_values = {name: rms(value).item() for name, value in values.items()}
    return {
        **{f"RMS_{name}": value for name, value in rms_values.items()},
        "corr_A_B": correlation(values["A"], values["B"]).item(),
        "corr_Dy_Dz": correlation(values["Dy"], values["Dz"]).item(),
        "directional_cancellation_Dy_Dz": (
            rms(values["Dy"] + values["Dz"]).item()
            / max(rms_values["Dy"] + rms_values["Dz"], 1e-30)
        ),
        "continuity_cancellation": (
            rms_values["R"] / max(rms_values["A"] + rms_values["B"], 1e-30)
        ),
    }


def collect_detached(collected, terms):
    for name in TERM_NAMES:
        collected[name].append(terms[name].detach().cpu().reshape(-1))


def parameter_group_norm(loss, named_parameters, retain_graph=True):
    parameters = [parameter for _, parameter in named_parameters if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph,
        create_graph=False, allow_unused=True)
    squared = loss.new_zeros(())
    non_none = 0
    finite = True
    for gradient in gradients:
        if gradient is not None:
            non_none += 1
            finite = finite and bool(torch.isfinite(gradient).all())
            squared = squared + gradient.detach().square().sum()
    return {
        "parameters_with_gradient": non_none,
        "norm": math.sqrt(squared.item()),
        "finite": finite,
    }


def model_parameter_groups(model):
    return {
        "fluid_decoder": list(model.fluid_decoder.named_parameters()),
        "transformer": list(model.blocks.named_parameters()),
        "fluid_encoder": list(model.to_fluid_token.named_parameters()),
        "particle_decoder": list(model.particle_decoder.named_parameters()),
    }


def numpy_metrics(prediction, reference):
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    difference = prediction - reference
    pred_centered = prediction - prediction.mean()
    ref_centered = reference - reference.mean()
    denominator = np.sqrt(
        np.sum(pred_centered ** 2) * np.sum(ref_centered ** 2))
    pred_rms = float(np.sqrt(np.mean(prediction ** 2)))
    ref_rms = float(np.sqrt(np.mean(reference ** 2)))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-30)),
        "correlation": float(
            np.sum(pred_centered * ref_centered) / max(denominator, 1e-30)),
        "prediction_rms": pred_rms,
        "reference_rms": ref_rms,
        "rms_ratio": pred_rms / max(ref_rms, 1e-30),
    }
