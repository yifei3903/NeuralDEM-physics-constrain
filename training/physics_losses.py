"""Physics losses and diagnostics for compressible gas flow.

The legacy residual keeps its finite-difference time term. Stage 6 adds a
separately named continuous spacetime residual using relative-time AD.
"""

import torch


def denormalize_pressure(pressure_normalized: torch.Tensor, pressure_ref: float,
                         pressure_scale: float) -> torch.Tensor:
    return pressure_normalized * pressure_scale + pressure_ref


def gas_density_from_pressure(
    pressure: torch.Tensor,
    molecular_weight: float = 29.0,
    gas_constant: float = 8314.56,
    gas_temperature: float = 298.15,
) -> torch.Tensor:
    """Ideal-gas EOS in SI units: rho=P*MW/(R*T)."""
    return pressure * molecular_weight / (gas_constant * gas_temperature)


def _pointwise_axis_derivative(
    values: torch.Tensor,
    physical_coords: torch.Tensor,
    axis: int,
    create_graph: bool,
) -> torch.Tensor:
    """Return d(values_i)/d(coords_i[axis]) for query-independent outputs."""
    gradient = torch.autograd.grad(
        outputs=values.sum(),
        inputs=physical_coords,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return gradient[..., axis:axis + 1]


def compute_mass_residual(
    predicted_fluid: torch.Tensor,
    previous_fluid: torch.Tensor,
    physical_coords: torch.Tensor,
    dt: float,
    pressure_ref: float,
    pressure_scale: float,
    molecular_weight: float = 29.0,
    gas_constant: float = 8314.56,
    gas_temperature: float = 298.15,
    create_graph: bool = True,
    return_terms: bool = False,
) -> torch.Tensor:
    """Compressible continuity residual at the n+1 cell-center queries.

    Channel order is [ux, uy, uz, eps_g, normalized P_g].
    """
    if predicted_fluid.shape != previous_fluid.shape:
        raise ValueError(f"predicted/previous shape mismatch: {predicted_fluid.shape} vs {previous_fluid.shape}")
    if predicted_fluid.shape[-1] != 5:
        raise ValueError("fluid state must have five channels")
    if physical_coords.shape[:-1] != predicted_fluid.shape[:-1] or physical_coords.shape[-1] != 3:
        raise ValueError("coordinate shape must match fluid points and end in xyz")
    if not physical_coords.requires_grad:
        raise ValueError("physical_coords.requires_grad must be True for spatial AD")

    velocity_np1 = predicted_fluid[..., 0:3]
    eps_np1 = predicted_fluid[..., 3:4]
    pressure_np1 = denormalize_pressure(predicted_fluid[..., 4:5], pressure_ref, pressure_scale)
    pressure_n = denormalize_pressure(previous_fluid[..., 4:5], pressure_ref, pressure_scale)

    rho_np1 = gas_density_from_pressure(
        pressure_np1, molecular_weight, gas_constant, gas_temperature)
    rho_n = gas_density_from_pressure(
        pressure_n, molecular_weight, gas_constant, gas_temperature)
    mass_np1 = eps_np1 * rho_np1
    mass_n = previous_fluid[..., 3:4] * rho_n

    temporal = (mass_np1 - mass_n) / dt
    flux = mass_np1 * velocity_np1
    directional_divergence = tuple(
        _pointwise_axis_derivative(flux[..., axis:axis + 1], physical_coords, axis, create_graph)
        for axis in range(3)
    )
    divergence = sum(directional_divergence)
    residual = temporal + divergence
    if return_terms:
        return residual, {
            "temporal": temporal,
            "divergence": divergence,
            "dJx_dx": directional_divergence[0],
            "dJy_dy": directional_divergence[1],
            "dJz_dz": directional_divergence[2],
        }
    return residual


def compute_mass_loss(*args, **kwargs):
    result = compute_mass_residual(*args, **kwargs)
    if isinstance(result, tuple):
        residual, terms = result
        return residual.square().mean(), residual, terms
    return result.square().mean(), result


def spacetime_continuity_residual(
    predicted_fluid: torch.Tensor,
    physical_coords: torch.Tensor,
    tau_query: torch.Tensor,
    pressure_ref: float,
    pressure_scale: float,
    molecular_weight: float = 29.0,
    gas_constant: float = 8314.56,
    gas_temperature: float = 298.15,
    create_graph: bool = True,
    return_terms: bool = False,
) -> torch.Tensor:
    """Continuous-time compressible continuity residual at each query.

    Computes d(epsilon*rho)/d(tau) + div(epsilon*rho*u). Both coordinates
    and tau must contain one autograd variable per query point. No discrete
    previous-state time difference is used here.
    """
    if predicted_fluid.shape[-1] != 5:
        raise ValueError("fluid state must have five channels")
    if physical_coords.shape != (*predicted_fluid.shape[:-1], 3):
        raise ValueError("physical_coords must match prediction queries and end in xyz")
    if tau_query.shape != (*predicted_fluid.shape[:-1], 1):
        raise ValueError("tau_query must match prediction queries and end in one scalar")
    if not physical_coords.requires_grad:
        raise ValueError("physical_coords.requires_grad must be True for spatial AD")
    if not tau_query.requires_grad:
        raise ValueError("tau_query.requires_grad must be True for time AD")

    velocity = predicted_fluid[..., 0:3]
    epsilon = predicted_fluid[..., 3:4]
    pressure = denormalize_pressure(
        predicted_fluid[..., 4:5], pressure_ref, pressure_scale)
    rho = gas_density_from_pressure(
        pressure, molecular_weight, gas_constant, gas_temperature)
    mass = epsilon * rho
    flux = mass * velocity

    dm_dt = torch.autograd.grad(
        outputs=mass.sum(),
        inputs=tau_query,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0]
    directional_divergence = tuple(
        _pointwise_axis_derivative(
            flux[..., axis:axis + 1], physical_coords, axis, create_graph)
        for axis in range(3)
    )
    divergence = sum(directional_divergence)
    residual = dm_dt + divergence
    if return_terms:
        return residual, {
            "dm_dt": dm_dt,
            "divergence": divergence,
            "dJx_dx": directional_divergence[0],
            "dJy_dy": directional_divergence[1],
            "dJz_dz": directional_divergence[2],
            "mass": mass,
            "rho": rho,
        }
    return residual


def spacetime_continuity_loss(*args, **kwargs):
    result = spacetime_continuity_residual(*args, **kwargs)
    if isinstance(result, tuple):
        residual, terms = result
        return residual.square().mean(), residual, terms
    return result.square().mean(), result
