"""CFD-DEM dataset for pointwise, compressible-fluid NeuralDEM training."""

import csv
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config import DATA_SOURCES, DataConfig


FLUID_CHANNELS = ("U_G_x", "U_G_y", "U_G_z", "EP_G", "P_G_normalized")


def normalize_pressure(pressure: np.ndarray, cfg: DataConfig) -> np.ndarray:
    return (pressure - cfg.pressure_ref) / cfg.pressure_scale


def denormalize_pressure(pressure_normalized, cfg: DataConfig):
    return pressure_normalized * cfg.pressure_scale + cfg.pressure_ref


def _raw_to_canonical(values: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """File order (z,x,y,C) -> NeuralDEM canonical order (x,y,z,C)."""
    channels = values.shape[-1]
    return values.reshape(cfg.nz, cfg.nx, cfg.ny, channels).transpose(1, 2, 0, 3).reshape(-1, channels)


def read_fluid_file(filepath: str, cfg: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read U_G, EP_G and P_G by header name and return canonical x,y,z order."""
    with open(filepath, "r", newline="", encoding="utf-8-sig") as handle:
        header = [name.strip().strip('"') for name in next(csv.reader(handle))]

    required = ["EP_G", "P_G", "Gas_Velocity:0", "Gas_Velocity:1", "Gas_Velocity:2"]
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"{filepath}: missing fluid fields {missing}; header={header}")

    data = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] != cfg.n_cells:
        raise ValueError(f"{filepath}: expected {cfg.n_cells} cells, got {data.shape[0]}")
    if data.shape[1] != len(header):
        raise ValueError(f"{filepath}: header/data column mismatch")

    index = {name: i for i, name in enumerate(header)}
    velocity = data[:, [index[f"Gas_Velocity:{axis}"] for axis in range(3)]]
    voidfraction = data[:, index["EP_G"]:index["EP_G"] + 1]
    pressure = data[:, index["P_G"]:index["P_G"] + 1]
    if not (1.0e4 < float(pressure.min()) and float(pressure.max()) < 1.0e7):
        raise ValueError(f"{filepath}: implausible pressure range {pressure.min()}..{pressure.max()} Pa")

    return (
        _raw_to_canonical(velocity, cfg),
        _raw_to_canonical(voidfraction, cfg),
        _raw_to_canonical(pressure, cfg),
    )


def read_particle_file(filepath: str, max_particles: int) -> Tuple[np.ndarray, np.ndarray, int]:
    data = np.loadtxt(filepath, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 6:
        raise ValueError(f"{filepath}: expected at least 6 particle columns")
    n = min(data.shape[0], max_particles)
    return data[:n, :3], data[:n, -3:], n


def build_fluid_coords(cfg: DataConfig) -> np.ndarray:
    """Cell centers in canonical flattened (x,y,z) order."""
    x = (np.arange(cfg.nx, dtype=np.float32) + 0.5) * (cfg.x_len / cfg.nx)
    y = (np.arange(cfg.ny, dtype=np.float32) + 0.5) * (cfg.y_len / cfg.ny)
    z = (np.arange(cfg.nz, dtype=np.float32) + 0.5) * (cfg.z_len / cfg.nz)
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)


def patchify_fluid(fluid_data: np.ndarray, cfg: DataConfig,
                   patch_size: Tuple[int, int, int]) -> np.ndarray:
    """Canonical (N,C) cell field -> (num_patches, patch_volume*C)."""
    px, py, pz = patch_size
    channels = fluid_data.shape[-1]
    volume = fluid_data.reshape(cfg.nx, cfg.ny, cfg.nz, channels)
    patches = volume.reshape(
        cfg.nx // px, px, cfg.ny // py, py, cfg.nz // pz, pz, channels,
    ).transpose(0, 2, 4, 1, 3, 5, 6)
    return patches.reshape(-1, px * py * pz * channels).astype(np.float32, copy=False)


def build_patch_coords(fluid_coords: np.ndarray, cfg: DataConfig,
                       patch_size: Tuple[int, int, int]) -> np.ndarray:
    patched = patchify_fluid(fluid_coords, cfg, patch_size)
    return patched.reshape(len(patched), -1, 3).mean(axis=1).astype(np.float32)


def select_supernodes(particle_coords: np.ndarray, num_supernodes: int,
                      rng: np.random.RandomState) -> np.ndarray:
    n = particle_coords.shape[0]
    if n <= num_supernodes:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=num_supernodes, replace=False).astype(np.int64)


class CFDDEMDataset(Dataset):
    def __init__(self, cfg: DataConfig, model_cfg, train: bool = True,
                 sources: List[dict] | None = None):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.sources = DATA_SOURCES if sources is None else sources
        self.samples = []

        for src in self.sources:
            for key in ("fluid_dir", "particle_dir"):
                if not os.path.isdir(src[key]):
                    raise FileNotFoundError(f"Missing {key}: {src[key]}")
            frame_ids = list(range(src["start_frame"], src["end_frame"] + 1, cfg.frame_interval))
            for t0, t1 in zip(frame_ids[:-1], frame_ids[1:]):
                paths = (
                    os.path.join(src["fluid_dir"], f"fluid_{t0}.txt"),
                    os.path.join(src["fluid_dir"], f"fluid_{t1}.txt"),
                    os.path.join(src["particle_dir"], f"particles_{t0}.txt"),
                    os.path.join(src["particle_dir"], f"particles_{t1}.txt"),
                )
                for path in paths:
                    if not os.path.isfile(path):
                        raise FileNotFoundError(path)
                self.samples.append((src, t0, t1))

        self.fluid_coords = build_fluid_coords(cfg)
        self.patch_coords = build_patch_coords(self.fluid_coords, cfg, model_cfg.patch_size)
        self._voidfraction_min = self._estimate_voidfraction_min()

    def _estimate_voidfraction_min(self) -> float:
        mins = []
        for src in self.sources:
            frames = (src["start_frame"], (src["start_frame"] + src["end_frame"]) // 2, src["end_frame"])
            for frame in frames:
                path = os.path.join(src["fluid_dir"], f"fluid_{frame}.txt")
                _, voidfraction, _ = read_fluid_file(path, self.cfg)
                mins.append(float(voidfraction.min()))
        return min(mins)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cfg = self.cfg
        src, t0, t1 = self.samples[idx]
        fluid_dir, particle_dir = src["fluid_dir"], src["particle_dir"]

        vel_f0, eps_f0, p_f0 = read_fluid_file(os.path.join(fluid_dir, f"fluid_{t0}.txt"), cfg)
        vel_p0, coords_p0, n_p0 = read_particle_file(
            os.path.join(particle_dir, f"particles_{t0}.txt"), cfg.max_particles)
        vel_f1, eps_f1, p_f1 = read_fluid_file(os.path.join(fluid_dir, f"fluid_{t1}.txt"), cfg)
        vel_p1, _, _ = read_particle_file(
            os.path.join(particle_dir, f"particles_{t1}.txt"), cfg.max_particles)

        # Fixed channel order: ux, uy, uz, eps_g, normalized P_g.
        state_f0 = np.concatenate([vel_f0, eps_f0, normalize_pressure(p_f0, cfg)], axis=-1)
        state_f1 = np.concatenate([vel_f1, eps_f1, normalize_pressure(p_f1, cfg)], axis=-1)
        state_f0_patched = patchify_fluid(state_f0, cfg, self.model_cfg.patch_size)
        supernode_idx = select_supernodes(
            coords_p0, self.model_cfg.num_supernodes, np.random.RandomState(42 + idx))

        return {
            "inlet_velocity": torch.tensor([[src["inlet_velocity"]]], dtype=torch.float32),
            "input_particle_coords": torch.from_numpy(coords_p0),
            "input_particle_displacement": torch.from_numpy(vel_p0 * cfg.ml_dt),
            "supernode_idx": torch.from_numpy(supernode_idx),
            "input_fluid_coords": torch.from_numpy(self.patch_coords).unsqueeze(0),
            "input_fluid_state": torch.from_numpy(state_f0_patched).unsqueeze(0),
            "input_fluid_cell_state": torch.from_numpy(state_f0).unsqueeze(0),
            "output_fluid_coords": torch.from_numpy(self.fluid_coords).unsqueeze(0),
            "fluid_voidfraction_min": torch.tensor([self._voidfraction_min], dtype=torch.float32),
            "num_output_particles": torch.tensor([n_p0], dtype=torch.long),
            "gt_particle_displacement": torch.from_numpy(vel_p1 * cfg.ml_dt),
            "gt_fluid": torch.from_numpy(state_f1).unsqueeze(0),
            "source_name": src["name"],
            "t0": t0,
            "t1": t1,
        }


class ContinuousTimeCFDDEMDataset(Dataset):
    """One current state with K randomly sampled relative-time targets.

    The anchor state is independent of the sampled query times. Raw frame
    offsets are converted to physical seconds using ``cfg.original_dt``.
    """

    def __init__(
        self,
        cfg: DataConfig,
        model_cfg,
        source: dict,
        num_anchor_states: int = 20,
        queries_per_anchor: int = 4,
        min_offset: int = 1,
        max_offset: int = 100,
        seed: int = 42,
    ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.source = source
        self.queries_per_anchor = int(queries_per_anchor)
        self.min_offset = int(min_offset)
        self.max_offset = int(max_offset)
        self.rng = np.random.RandomState(seed)
        if not 1 <= self.min_offset <= self.max_offset:
            raise ValueError("offset range must be positive and ordered")
        if self.queries_per_anchor > self.max_offset - self.min_offset + 1:
            raise ValueError("queries_per_anchor exceeds available unique offsets")

        self.anchor_frames = [
            source["start_frame"] + index * cfg.frame_interval
            for index in range(num_anchor_states)
        ]
        if self.anchor_frames[-1] + self.max_offset > source["end_frame"]:
            raise ValueError("continuous-time target exceeds source end_frame")
        for directory_key, prefix in (("fluid_dir", "fluid"), ("particle_dir", "particles")):
            directory = source[directory_key]
            if not os.path.isdir(directory):
                raise FileNotFoundError(directory)
            for frame in (self.anchor_frames[0], self.anchor_frames[-1] + self.max_offset):
                path = os.path.join(directory, f"{prefix}_{frame}.txt")
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)

        self.fluid_coords = build_fluid_coords(cfg)
        self.patch_coords = build_patch_coords(
            self.fluid_coords, cfg, model_cfg.patch_size)
        probe_frames = (self.anchor_frames[0], self.anchor_frames[len(self.anchor_frames) // 2],
                        self.anchor_frames[-1] + self.max_offset)
        self._voidfraction_min = min(
            float(read_fluid_file(
                os.path.join(source["fluid_dir"], f"fluid_{frame}.txt"), cfg)[1].min())
            for frame in probe_frames
        )

    def __len__(self):
        return len(self.anchor_frames)

    def _fluid_state(self, frame: int) -> np.ndarray:
        velocity, voidfraction, pressure = read_fluid_file(
            os.path.join(self.source["fluid_dir"], f"fluid_{frame}.txt"), self.cfg)
        return np.concatenate([
            velocity, voidfraction, normalize_pressure(pressure, self.cfg),
        ], axis=-1).astype(np.float32, copy=False)

    def __getitem__(self, idx):
        offsets = np.sort(self.rng.choice(
            np.arange(self.min_offset, self.max_offset + 1),
            size=self.queries_per_anchor,
            replace=False,
        )).astype(np.int64)
        return self.get_with_offsets(idx, offsets)

    def get_with_offsets(self, idx, offsets):
        """Load deterministic relative-time targets for evaluation."""
        cfg = self.cfg
        anchor = self.anchor_frames[idx]
        offsets = np.asarray(offsets, dtype=np.int64)
        if offsets.ndim != 1 or len(offsets) == 0:
            raise ValueError("offsets must be a non-empty 1D sequence")
        if offsets.min() < self.min_offset or offsets.max() > self.max_offset:
            raise ValueError("requested offset is outside the configured range")
        tau = offsets.astype(np.float32) * np.float32(cfg.original_dt)

        state_f0 = self._fluid_state(anchor)
        vel_p0, coords_p0, n_p0 = read_particle_file(
            os.path.join(self.source["particle_dir"], f"particles_{anchor}.txt"),
            cfg.max_particles,
        )
        target_fluid = []
        target_particle_displacement = []
        for offset, tau_value in zip(offsets, tau):
            target_frame = anchor + int(offset)
            target_fluid.append(self._fluid_state(target_frame))
            vel_pk, _, n_pk = read_particle_file(
                os.path.join(self.source["particle_dir"], f"particles_{target_frame}.txt"),
                cfg.max_particles,
            )
            if n_pk < n_p0:
                raise ValueError(
                    f"target frame {target_frame} has fewer particles ({n_pk}) than anchor ({n_p0})")
            target_particle_displacement.append(vel_pk[:n_p0] * tau_value)

        state_f0_patched = patchify_fluid(
            state_f0, cfg, self.model_cfg.patch_size)
        supernode_idx = select_supernodes(
            coords_p0, self.model_cfg.num_supernodes,
            np.random.RandomState(42 + idx),
        )
        return {
            "inlet_velocity": torch.tensor(
                [[self.source["inlet_velocity"]]], dtype=torch.float32),
            "input_particle_coords": torch.from_numpy(coords_p0),
            # Keep the current-state feature scale independent of query tau.
            "input_particle_displacement": torch.from_numpy(vel_p0 * cfg.ml_dt),
            "supernode_idx": torch.from_numpy(supernode_idx),
            "input_fluid_coords": torch.from_numpy(self.patch_coords).unsqueeze(0),
            "input_fluid_state": torch.from_numpy(state_f0_patched).unsqueeze(0),
            "input_fluid_cell_state": torch.from_numpy(state_f0).unsqueeze(0),
            "output_fluid_coords": torch.from_numpy(self.fluid_coords).unsqueeze(0),
            "fluid_voidfraction_min": torch.tensor(
                [self._voidfraction_min], dtype=torch.float32),
            "num_output_particles": torch.tensor([n_p0], dtype=torch.long),
            "query_tau_values": torch.from_numpy(tau).unsqueeze(0),
            "target_offsets": torch.from_numpy(offsets).unsqueeze(0),
            "gt_particle_displacement": torch.from_numpy(
                np.stack(target_particle_displacement)).unsqueeze(0),
            "gt_fluid": torch.from_numpy(np.stack(target_fluid)).unsqueeze(0),
            "source_name": self.source["name"],
            "anchor_frame": anchor,
        }


def collate_fn(batch):
    if len(batch) != 1:
        raise ValueError("Variable-size particle data currently requires batch_size=1")
    return batch[0]
