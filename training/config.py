"""Reduced NeuralDEM configuration for the two compressible CFD-DEM runs."""

from dataclasses import dataclass


DATA_SOURCES = [
    {
        "name": "gidaspow_060",
        "fluid_dir": r"D:\mfixflie\revise\0.6ms\Gidaspow\fluid",
        "particle_dir": r"D:\mfixflie\revise\0.6ms\Gidaspow\particles",
        "inlet_velocity": 0.60,
        "start_frame": 0,
        "end_frame": 20000,
    },
    {
        "name": "gidaspow_074",
        "fluid_dir": r"D:\mfixflie\revise\Gidaspow\fluid",
        "particle_dir": r"D:\mfixflie\revise\Gidaspow\particles",
        "inlet_velocity": 0.74,
        "start_frame": 0,
        "end_frame": 20000,
    },
]


MODEL_SCALES = {
    "original": dict(dim=768, depth=12, num_heads=12, num_supernodes=100),
    # Removing mixing reduces parameters, so dim=156 keeps the requested model
    # close to 9.58M while retaining four attention blocks and four heads.
    "small": dict(dim=156, depth=4, num_heads=4, num_supernodes=100),
}


@dataclass
class DataConfig:
    # Physical time between two consecutive raw CFD/VTK output files.
    original_dt: float = 0.001
    # Number of raw output-file intervals skipped between ML states.
    frame_interval: int = 100
    # Backward-compatible explicit value; checked against ml_dt below.
    dt: float = 0.1

    x_len: float = 0.044
    y_len: float = 0.12
    z_len: float = 0.01
    nx: int = 12
    ny: int = 24
    nz: int = 3
    n_cells: int = 864
    max_particles: int = 10000

    # P* = (P - pressure_ref) / pressure_scale.
    pressure_ref: float = 101325.0
    pressure_scale: float = 1000.0
    molecular_weight: float = 29.0       # kg/kmol
    gas_temperature: float = 298.15      # K
    gas_constant: float = 8314.56        # J/(kmol K)

    def __post_init__(self):
        if self.n_cells != self.nx * self.ny * self.nz:
            raise ValueError("n_cells must equal nx * ny * nz")
        if abs(self.dt - self.ml_dt) > 1e-12:
            raise ValueError(f"dt={self.dt} does not match ML frame spacing {self.ml_dt}")

    @property
    def ml_dt(self) -> float:
        """Physical time between the two states in one ML transition."""
        return self.original_dt * self.frame_interval


@dataclass
class ModelConfig:
    model_scale: str = "small"
    dim: int | None = None
    num_heads: int | None = None
    depth: int | None = None
    num_supernodes: int | None = None

    patch_size: tuple = (4, 4, 3)
    resolution: tuple = (12, 24, 3)
    radius: float = 0.003
    domain_min: tuple = (0.0, 0.0, 0.0)
    domain_max: tuple = (0.044, 0.12, 0.01)
    particle_radius: float = 0.0006
    tau_max: float = 0.1

    def __post_init__(self):
        if self.model_scale not in MODEL_SCALES:
            raise ValueError(f"Unknown model_scale={self.model_scale!r}")
        preset = MODEL_SCALES[self.model_scale]
        for name, value in preset.items():
            if getattr(self, name) is None:
                setattr(self, name, value)
        if self.dim % self.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if any(r % p for r, p in zip(self.resolution, self.patch_size)):
            raise ValueError("resolution must be divisible by patch_size")


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 1
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    loss_particle_weight: float = 1.0
    loss_fluid_weight: float = 3.0
    loss_velocity_weight: float = 1.0
    loss_voidfraction_weight: float = 1.0
    loss_pressure_weight: float = 1.0
    use_mass_constraint: bool = True
    lambda_mass: float = 0.0

    ar_test_every: int = 5
    ar_test_steps: int = 20
    metric_steps: tuple = (1, 5, 10, 20, 50)
    save_dir: str = r"D:\NeuralDEM\NeuralDEM-stage6b\training\checkpoints_reduced"
    save_every: int = 10
    log_every: int = 1
    device: str = "cuda"

    # Deterministic small-data diagnostic. The ordinary training path keeps
    # this disabled; small_overfit.py enables it explicitly.
    debug_overfit: bool = False
    debug_num_pairs: int = 20
    debug_source_index: int = 0
    debug_epochs: int = 10
    debug_output_dir: str = r"D:\NeuralDEM\NeuralDEM-stage6b\training\small_overfit_results"


@dataclass
class ContinuousTimeConfig:
    """Fixed Stage-6 protocol; these values are not a hyperparameter sweep."""

    tau_max: float = 0.1
    min_offset: int = 1
    max_offset: int = 100
    queries_per_anchor: int = 4
    num_anchor_states: int = 20
    smoke_epochs: int = 5
    train_epochs: int = 100
    evaluation_taus: tuple = (0.01, 0.025, 0.05, 0.075, 0.1)
    derivative_taus: tuple = (0.02, 0.04, 0.06, 0.08)
    output_dir: str = r"D:\NeuralDEM\NeuralDEM-stage6b\training\continuous_time_results_stage6"


@dataclass
class Stage6BConfig:
    """Fixed controlled Stage-6B mass-physics experiment."""

    lambda_mass: float = 1e-3
    train_epochs: int = 100
    checkpoint_epochs: tuple = (1, 25, 50, 75, 100)
    output_dir: str = r"D:\NeuralDEM\NeuralDEM-stage6b\training\stage6b_mass_physics_results"
