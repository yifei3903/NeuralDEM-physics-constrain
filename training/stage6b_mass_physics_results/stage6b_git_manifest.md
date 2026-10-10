# Stage 6B Git and artifact manifest

## Provenance

- Stage 6 baseline commit: `3c3b1c9c5445a6f8d5cd0bfa2c1fcc56b2a08256`
- Stage 6 baseline branch: `baseline/stage6-continuous-time`
- Stage 6 baseline tag: `stage6-baseline`
- Stage 6B branch: `experiment/stage6b-mass-physics`
- Stage 6B experiment commit: `4a46b6ec950f2df9ba9345b4354bde2a49e39f18`
- Stage 6B completion tag: `stage6b-completed`
- Stage 7 archive commit: `fdd55f17e0b81b47372c52a96d23e9c70f35014c`
- Stage 7 archive branch: `archive/stage7-dynamics`
- Stage 7 archive tag: `stage7-completed`

Stage 6B is a direct descendant of the Stage 6 baseline. It is not derived from Stage 7.

## Experiment definition

- Source case: `gidaspow_060`
- Anchor frames: 0, 100, ..., 1900 (20 anchors)
- Relative-time queries per anchor: 4
- Query range: 0.001--0.100 s
- Epochs: 100
- Optimizer updates: 2,000
- Seed: 42
- Optimizer: AdamW, learning rate 1e-4, weight decay 1e-4
- Mass-loss weight: 1e-3
- Model parameters: 9,532,700
- Elapsed training time: 1,662.057 s
- NaN/Inf: none

## Tracked Stage 6B files

- `training/config.py`
- `training/stage6b_utils.py`
- `training/stage6b_physics_smoke_test.py`
- `training/stage6b_train.py`
- `training/stage6b_evaluate.py`
- `training/stage6b_mass_physics_results/stage6b_config.yaml`
- `training/stage6b_mass_physics_results/physics_smoke_test.json`
- `training/stage6b_mass_physics_results/stage6b_training_history.csv`
- `training/stage6b_mass_physics_results/training_summary.json`
- `training/stage6b_mass_physics_results/evaluation_summary.json`
- `training/stage6b_mass_physics_results/fixed_horizon_metrics.csv`
- `training/stage6b_mass_physics_results/physics_diagnostics.csv`
- `training/stage6b_mass_physics_results/time_derivative_metrics.csv`
- `training/stage6b_mass_physics_results/flux_divergence_metrics.csv`
- `training/stage6b_mass_physics_results/rollout_metrics.csv`
- `training/stage6b_mass_physics_results/stage6_vs_stage6b.csv`
- `training/stage6b_mass_physics_results/stage6b_summary.md`
- `training/stage6b_mass_physics_results/stage6b_git_manifest.md`

Checkpoint binaries are intentionally ignored by Git and retained on disk in `training/stage6b_mass_physics_results/checkpoints/`.

## Artifact integrity

- `checkpoints/final.pt`: `99AB933A3B1469DF5DE921D2CFD319552D9B100D4B67F7D39B6BA33B5AB6E171`
- `stage6b_training_history.csv`: `9BF63913D110BE62ADEBD25DDF0F2C37DF464A0966AECFEAB8B7E73E0F30A883`
- `stage6_vs_stage6b.csv`: `970F383967BCFD3720347B97B56F88A11798B70B597687A986F005BD74A7AB66`

The final checkpoint was successfully deserialized on 2026-10-10 and records epoch 100. All primary evaluation CSV files have their expected row counts and finite numeric values. The merged comparison table contains expected blank cells where a metric does not apply to a particular row type.

## Protected-code audit

Relative to the Stage 6 baseline, no changes are present under `src/`, nor in:

- `training/dataset.py`
- `training/continuous_time_utils.py`
- `training/physics_losses.py`

This confirms that Stage 6B did not alter the model body, dataset definition, spacetime derivative implementation, or Stage 6 physics-loss implementation.

## Worktrees at completion

- `D:/NeuralDEM/NeuralDEM-main`: `archive/stage7-dynamics`
- `D:/NeuralDEM/NeuralDEM-stage6b`: `experiment/stage6b-mass-physics`

The tag `stage6b-completed` identifies the finalized Stage 6B report state. The experiment implementation and numeric outputs were first frozen in the experiment commit listed above; the completion tag additionally includes this report and manifest.
