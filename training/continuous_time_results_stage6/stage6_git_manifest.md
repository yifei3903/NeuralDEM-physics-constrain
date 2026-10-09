# Stage 6 Git baseline manifest

- Branch: `baseline/stage6-continuous-time`
- Validated source snapshot commit: `49ede8028da47291a5072f286e71d946db554cf4`
- Baseline tag: `stage6-baseline`
- Worktree used for validation: `D:\NeuralDEM\NeuralDEM-stage6b`
- Result directory: `D:\NeuralDEM\NeuralDEM-stage6b\training\continuous_time_results_stage6`
- Summary: `D:\NeuralDEM\NeuralDEM-stage6b\training\continuous_time_results_stage6\stage6_summary.md`
- Epoch-100 checkpoint: `D:\NeuralDEM\NeuralDEM-stage6b\training\continuous_time_results_stage6\checkpoints\continuous_time_epoch_100.pt`
- Checkpoint SHA256: `BBDC8F07BD768D36F0D38E01D99FDBC47D45B3DE3CA2535DBD2C85C88F0013B4`
- Shared initialization: `D:\NeuralDEM\NeuralDEM-stage6b\training\small_overfit_results\shared_initial_state.pt`
- Shared-initialization SHA256: `83247C30CEC60FD0EDDE678B6BD0397EF434EC7672C444B42F1D19F58B20A7ED`

## Baseline protocol

- Model: Reduced NeuralDEM, 9,532,700 parameters.
- Data-only: `lambda_mass=0`.
- Input/output: `(S_n,tau) -> S_(n+tau)`.
- Fluid output: `[B,K,864,5]`; particle displacement: `[B,K,Np,3]`.
- Channels: `[u,v,w,epsilon,P_star]`.
- `K=4`; one state encoding is reused by all four pointwise spacetime queries.
- Random tau support: `{0.001,...,0.100} s`, resampled every anchor/epoch.
- 20 anchor frames: `0,100,...,1900`.
- 100 epochs; seed 42; AdamW; learning rate `1e-4`; weight decay `1e-4`; batch size 1 anchor.

## Restoration validation

- Stage-7-only `predict_dynamics`, `DynamicsConfig`, `LocalDynamicsDataset`, scripts, and result references are absent.
- Epoch-100 checkpoint strict load: no missing or unexpected keys.
- Parameter count: 9,532,700.
- Smoke output: fluid `[1,4,864,5]`, particle `[1,4,9240,3]`.
- Encoder calls for K=4: 1.
- Query independence: changing one query changed that query by `8.06e-5`; all other queries changed by exactly 0.
- `d(tau/0.1)/dtau=10 s^-1`.
- Pointwise time-gradient shape: `[1,4,864,1]`; time and spatial AD are finite.
- All reproduced Stage-6 files below have SHA256 exactly equal to the archived originals:
  - `horizon_accuracy.csv`
  - `rollout_smoke.json`
  - `smoke_test.json`
  - `spacetime_continuity.csv`
  - `spacetime_continuity_summary.json`
  - `temporal_trajectory_samples.csv`
  - `time_derivative_summary.json`
  - `time_derivative_vs_cfd.csv`
- Reproduced continuity residual RMS: 3.66963, 3.96971, 4.43335, 4.88186 at tau 0.02, 0.04, 0.06, 0.08 s.

## Runtime

- Python: 3.10.19
- PyTorch: 2.5.1
- CUDA available: False
- Platform: Windows-10-10.0.26200-SP0

Checkpoint files are intentionally ignored by Git and retained on disk. Small metrics, summaries, and this manifest are tracked.
