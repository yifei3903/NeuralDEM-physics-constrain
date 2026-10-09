# Stage 7 Git archive manifest

- Branch: `archive/stage7-dynamics`
- Source snapshot commit: `6b7c28ede2f49d521a79a3e28a1ddb456633e374`
- Completed archive tag: `stage7-completed`
- Result directory: `D:\NeuralDEM\NeuralDEM-main\training\continuous_dynamics_results_stage7`
- Summary: `D:\NeuralDEM\NeuralDEM-main\training\continuous_dynamics_results_stage7\stage7_summary.md`
- Checkpoints are retained on disk and intentionally ignored by Git.

## Checkpoint SHA256

| checkpoint | SHA256 |
|---|---|
| `epoch_100.pt` | `D5FB0AE4B53A43F114ACACED67B0F62A9A9D7D67EB4E48C3514F2BE362A86492` |
| `epoch_300.pt` | `62E1C6E0326573CDC810E4585FA5933C6FB37936552BE2123317C67505D3D0FF` |
| `epoch_500.pt` | `A7A774BDBFF8EA2373FF17302C629B55108686ED61592FD2994F1E041A6DE292` |

## Training protocol

- Model: Reduced NeuralDEM, 9,532,700 parameters.
- Task: data-only learned local fluid dynamics with explicit Euler.
- Source: `gidaspow_060`.
- Fluid rate: `[du/dt,dv/dt,dw/dt,dm_star/dt,dP_star/dt]`.
- `dt_dyn=0.001 s`; 20 independently sampled 0.1-s blocks per epoch.
- 500 epochs, 10,000 optimizer updates, batch size 1.
- Seed 42; AdamW; learning rate `1e-4`; weight decay `1e-4`.
- Initialization: `training/small_overfit_results/shared_initial_state.pt`, SHA256 `83247C30CEC60FD0EDDE678B6BD0397EF434EC7672C444B42F1D19F58B20A7ED`.
- No mass/continuity loss, momentum equation, finite-volume layer, history window, or Stage-8 physics update.

## Runtime

- Python: `3.10.19` (Anaconda build)
- Python executable: `D:\miniforge\envs\spyder\python.exe`
- PyTorch: `2.5.1`
- CUDA available: `False`
- Platform: `Windows-10-10.0.26200-SP0`

## Stage 7 source changes

Modified relative to the Stage-6 behavior:

- `src/neural_dem/modules/dit_perceiver_decoder.py`
- `src/neural_dem/fbed_model.py`
- `training/config.py`
- `training/dataset.py`

Added for Stage 7:

- `training/dynamics_utils.py`
- `training/dynamics_smoke_test.py`
- `training/dynamics_train.py`
- `training/evaluate_dynamics.py`
- `training/validate_dynamics_rate.py`
- `training/dynamics_rollout.py`

All small CSV/JSON/Markdown diagnostics are tracked. The `.pt` checkpoint files remain at the paths above but are not committed.
