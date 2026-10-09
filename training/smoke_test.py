"""One-batch validation of dataset, pointwise decoder, EOS, AD and backward."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, DataConfig, ModelConfig, TrainConfig
from dataset import CFDDEMDataset, denormalize_pressure, read_fluid_file
from inference import build_model, run_autoregressive
from physics_losses import gas_density_from_pressure
from train import set_seed, train_one_batch


def assert_finite_metrics(metrics):
    for name, value in metrics.items():
        if not np.isfinite(value):
            raise AssertionError(f"{name} is not finite: {value}")


def main():
    data_cfg, model_cfg = DataConfig(), ModelConfig(model_scale="small")
    set_seed(42)
    device = torch.device("cpu")

    print("TIME DISCRETIZATION CHECK")
    print(f"  raw CFD output dt={data_cfg.original_dt:.9g} s")
    print(f"  frame interval={data_cfg.frame_interval} raw output files per ML step")
    print(f"  actual ML dt used in continuity residual={data_cfg.ml_dt:.9g} s")
    if data_cfg.ml_dt != 0.1:
        raise AssertionError("unexpected ML time interval")

    print("DATA FILE CHECK")
    for source in DATA_SOURCES:
        fluid_count = len([name for name in os.listdir(source["fluid_dir"]) if name.startswith("fluid_")])
        particle_count = len([name for name in os.listdir(source["particle_dir"]) if name.startswith("particles_")])
        print(source["name"], "fluid_files=", fluid_count, "particle_files=", particle_count)
        for frame in (0, 100, 10000, 20000):
            velocity, voidfraction, pressure = read_fluid_file(
                os.path.join(source["fluid_dir"], f"fluid_{frame}.txt"), data_cfg)
            print(
                f"  frame={frame} EP_G=[{voidfraction.min():.6g},{voidfraction.max():.6g}] "
                f"P_G=[{pressure.min():.6g},{pressure.max():.6g}] "
                f"velocity=[{velocity.min():.6g},{velocity.max():.6g}]"
            )

    dataset = CFDDEMDataset(data_cfg, model_cfg)
    batch = dataset[0]
    print("DATASET samples=", len(dataset), "channels=ux,uy,uz,EP_G,P_G_normalized")
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(" ", name, tuple(value.shape), value.dtype)

    previous_cells = batch["input_fluid_cell_state"]
    next_cells = batch["gt_fluid"]
    query_cells = batch["output_fluid_coords"]
    if previous_cells.shape != next_cells.shape or previous_cells.shape != (1, data_cfg.n_cells, 5):
        raise AssertionError("n and n+1 fluid states are not aligned cell representations")
    if query_cells.shape != (1, data_cfg.n_cells, 3):
        raise AssertionError("cell-state/query-coordinate alignment failed")
    print("CELL ALIGNMENT n, n+1, xyz", tuple(previous_cells.shape),
          tuple(next_cells.shape), tuple(query_cells.shape))

    model = build_model(model_cfg, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("PARAMETERS", parameter_count)

    # The decoder consumes normalized coordinates, but normalization happens
    # after physical coordinates enter the Torch graph. Therefore gradients
    # requested w.r.t. physical_coords include the chain-rule scale 1/L.
    physical_coords = batch["output_fluid_coords"].detach().clone().requires_grad_(True)
    normalized_coords = model._normalize_fluid_coords(physical_coords)
    coordinate_probe = normalized_coords.sum()
    coordinate_gradient = torch.autograd.grad(coordinate_probe, physical_coords)[0]
    expected_gradient = 1.0 / (model.domain_max - model.domain_min)
    if not torch.allclose(coordinate_gradient[0, 0], expected_gradient, rtol=1e-6, atol=1e-6):
        raise AssertionError("coordinate normalization chain rule is incorrect")
    print("COORDINATE AD d(x*,y*,z*)/d(x,y,z)=",
          coordinate_gradient[0, 0].tolist(), "1/m")

    # Empirical diagonal-Jacobian check: moving query 0 must not alter other
    # query outputs. This remains true after training because the decoder has
    # cross-attention to latent KV but no query-query operation.
    with torch.no_grad():
        latent = torch.randn(1, 18, model_cfg.dim)
        query = torch.rand(1, 4, 3)
        condition = model.conditioner(torch.tensor([[0.60]]))
        decoded = model.fluid_decoder(latent, query, {"cond": condition})
        changed_query = query.clone()
        changed_query[:, 0, 0] += 0.1
        changed = model.fluid_decoder(latent, changed_query, {"cond": condition})
        cross_query_change = (decoded[:, 1:] - changed[:, 1:]).abs().max().item()
    print("QUERY INDEPENDENCE max off-diagonal change", cross_query_change)
    if cross_query_change != 0.0:
        raise AssertionError("fluid decoder queries are coupled")

    label_pressure = denormalize_pressure(batch["gt_fluid"][..., 4:5], data_cfg)
    if not (1.0e4 < label_pressure.min() < label_pressure.max() < 1.0e7):
        raise AssertionError("pressure denormalization failed")
    eos_density = gas_density_from_pressure(
        torch.tensor(101320.0), data_cfg.molecular_weight,
        data_cfg.gas_constant, data_cfg.gas_temperature)
    print("EOS rho(P=101320 Pa)=", eos_density.item())
    p_star_probe = torch.tensor(0.25)
    p_pa_probe = p_star_probe * data_cfg.pressure_scale + data_cfg.pressure_ref
    rho_probe = gas_density_from_pressure(
        p_pa_probe, data_cfg.molecular_weight, data_cfg.gas_constant,
        data_cfg.gas_temperature)
    print(f"EOS CHAIN P*={p_star_probe.item():.6g} -> P={p_pa_probe.item():.6g} Pa "
          f"-> rho={rho_probe.item():.9g} kg/m^3")
    if abs(eos_density.item() - 1.1853) > 1e-3:
        raise AssertionError("ideal-gas EOS magnitude is wrong")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    baseline_cfg = TrainConfig(use_mass_constraint=False, lambda_mass=0.0, device="cpu")
    baseline_metrics, baseline_prediction = train_one_batch(
        model, batch, optimizer, device, data_cfg, baseline_cfg)
    assert baseline_prediction["fluid"].shape == (1, data_cfg.n_cells, 5)
    assert_finite_metrics(baseline_metrics)
    predicted_pressure = denormalize_pressure(
        baseline_prediction["fluid"][..., 4:5].detach(), data_cfg)
    if not (1.0e4 < predicted_pressure.min() and predicted_pressure.max() < 1.0e7):
        raise AssertionError("predicted pressure is not on the expected physical scale")
    print("PREDICTED P_G range Pa", predicted_pressure.min().item(), predicted_pressure.max().item())
    print("BASELINE", baseline_metrics)

    physics_cfg = TrainConfig(use_mass_constraint=True, lambda_mass=1e-8, device="cpu")
    physics_metrics, physics_prediction = train_one_batch(
        model, batch, optimizer, device, data_cfg, physics_cfg)
    assert physics_prediction["fluid"].shape == (1, data_cfg.n_cells, 5)
    assert physics_metrics["mass_loss"] > 0.0
    assert_finite_metrics(physics_metrics)
    if not any(parameter.grad is not None for parameter in model.fluid_decoder.parameters()):
        raise AssertionError("fluid decoder did not receive gradients")
    print("PHYSICS", physics_metrics)

    rollout = run_autoregressive(
        model, model_cfg, data_cfg, DATA_SOURCES[0], start_frame=0,
        num_steps=2, device=device, compute_mass_metrics=True)
    if len(rollout) != 2 or not np.isfinite(rollout[-1]["mass_residual_rms"]):
        raise AssertionError("autoregressive rollout failed")
    print("ROLLOUT step2", {
        key: rollout[-1][key]
        for key in ("mass_residual_mae", "mass_residual_rms")
    })
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
