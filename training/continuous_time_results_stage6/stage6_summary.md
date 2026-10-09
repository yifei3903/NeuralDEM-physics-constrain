# Stage 6: continuous relative-time NeuralDEM

## Scope and protocol

- One current state only: `S_n`; no history window.
- Relative physical query time: `tau=(target_frame-anchor_frame)*0.001 s`.
- Training support: `tau in {0.001, ..., 0.100} s`, four offsets sampled again for every anchor and epoch.
- 20 anchor states from the 0.60 m/s case: frames 0, 100, ..., 1900.
- One state encoding followed by K decoder queries; encoder/Transformer is not repeated for K.
- Data-only training: `lambda_mass=0`; no continuity loss was optimized.
- Seed 42, AdamW, learning rate `1e-4`, weight decay `1e-4`, batch size one anchor.
- Five-epoch smoke checkpoint and fixed 100-epoch prototype checkpoint.
- 9,532,700 trainable parameters. The separate sinusoidal time embeddings add no trainable parameters.
- Training time: 1035.802 s (17.263 min) on CPU. No NaN/Inf.

The final randomly sampled epoch has data loss 0.06471. The best sampled epoch is epoch 78 at 0.04550, and the last-ten-epoch mean is 0.06591. The objective was still noisy and improving; training was nevertheless stopped at the pre-declared 100 epochs.

## Architecture and compatibility

The fluid and particle decoders keep their original pointwise Perceiver cross-attention. Their query features are now

`query_mlp(spatial_sincos(x,y,z) + time_sincos(tau/0.1))`.

There is no query-query self-attention. `FbedModel.encode_state()` produces the latent state once, and `decode_queries()` expands the same latent over K relative times. Explicit continuous queries return fluid `[B,K,864,5]` and particle displacement `[B,K,Np,3]`.

`query_tau=None` constructs `tau=0.1 s` internally and preserves the legacy output shapes. The deterministic time-frequency buffers are non-persistent, so both the Stage-5 shared initialization and Stage-5 epoch-500 checkpoint still load with `strict=True` and the original 141 state-dict keys.

For continuous particle prediction, every tau queries displacement for the same current particles. During rollout, the next particle coordinates are updated by `x_(n+tau)=x_n+Delta_x(tau)`. The current-state encoder feature remains at its historical 0.1-s displacement scale so it does not change meaning when `tau_test` changes.

## Pointwise time-AD smoke test

| check | result |
|---|---:|
| K-query fluid shape | `[1,4,864,5]` |
| K-query particle shape | `[1,4,9240,3]` |
| `autograd.grad(m.sum(), tau_query)` shape | `[1,4,864,1]` |
| encoder calls for K=4 | 1 |
| `d(tau/0.1)/d(tau)` | `10.0 s^-1` |
| changed query response for `delta_tau=1e-4 s` | `8.06e-5` max abs |
| maximum change at every other query | `0.0` |
| initial parameter gradient norm | `117.36` |
| finite time gradient / residual | yes / yes |

Thus each cell owns an independent time variable. The returned gradient is pointwise, not a `[B,K]` spatial sum.

## Fixed-horizon teacher-forced accuracy

All metrics use the same 20 anchor states. Pressure RMSE is in Pa; velocity RMSE is in m/s.

| tau (s) | u RMSE / corr | v RMSE / corr | w RMSE / corr | epsilon RMSE / corr | pressure RMSE / corr | particle displacement RMSE / corr |
|---:|---:|---:|---:|---:|---:|---:|
| 0.010 | 0.0672 / 0.265 | 0.1822 / 0.845 | 0.0271 / 0.121 | 0.0633 / 0.964 | 20.84 / 0.939 | 0.00116 / 0.024 |
| 0.025 | 0.0683 / 0.288 | 0.1958 / 0.828 | 0.0251 / 0.150 | 0.0604 / 0.968 | 16.34 / 0.968 | 0.00135 / 0.052 |
| 0.050 | 0.0593 / 0.249 | 0.1879 / 0.843 | 0.0234 / 0.202 | 0.0575 / 0.972 | 17.14 / 0.970 | 0.00194 / 0.036 |
| 0.075 | 0.0542 / 0.150 | 0.1564 / 0.883 | 0.0208 / 0.058 | 0.0624 / 0.967 | 16.99 / 0.971 | 0.00244 / -0.007 |
| 0.100 | 0.0545 / 0.123 | 0.1490 / 0.892 | 0.0208 / 0.036 | 0.0665 / 0.963 | 16.64 / 0.976 | 0.00348 / -0.018 |

The prototype learns v, epsilon, and pressure much better than u, w, and particle displacement. Accuracy is not monotonic in tau because the target dynamics and field variance differ by horizon. These are prototype facts, not a tuned accuracy result.

## Does tau control a continuous prediction?

`temporal_trajectory_samples.csv` contains 6000 rows: four anchors, three interior cells, five channels, and all 100 query times from 0.001 to 0.100 s.

Across those trajectories, mean prediction ranges over the 0.099-s interval are:

| channel | mean range | maximum range | maximum adjacent 0.001-s jump |
|---|---:|---:|---:|
| u | 0.00526 m/s | 0.01391 | 0.000508 |
| v | 0.02374 m/s | 0.08503 | 0.002336 |
| w | 0.00264 m/s | 0.00620 | 0.000132 |
| epsilon | 0.02872 | 0.10677 | 0.002064 |
| pressure | 2.468 Pa | 5.305 | 0.133 Pa |

The outputs are not repeated copies and vary smoothly with tau. However, their temporal variation is strongly smoothed relative to the raw CFD temporal changes.

## Time AD versus raw-CFD temporal finite difference

CFD reference derivatives use `[q(t+0.001)-q(t-0.001)]/0.002`. Pressure derivatives are in Pa/s and mass derivatives in kg m^-3 s^-1.

| tau (s) | quantity | AD RMS | CFD-FD RMS | RMS ratio | relative L2 | correlation |
|---:|---|---:|---:|---:|---:|---:|
| 0.02 | epsilon_dt | 0.111 | 4.800 | 0.0232 | 0.9996 | 0.0314 |
| 0.02 | pressure_dt | 38.03 | 1740.55 | 0.0218 | 0.9965 | 0.0655 |
| 0.02 | mass_dt | 0.132 | 5.693 | 0.0232 | 0.9996 | 0.0316 |
| 0.04 | mass_dt | 0.228 | 6.258 | 0.0364 | 0.9990 | 0.0467 |
| 0.06 | mass_dt | 0.321 | 5.032 | 0.0638 | 1.0039 | -0.0289 |
| 0.08 | mass_dt | 0.422 | 5.136 | 0.0821 | 1.0054 | -0.0259 |

Pointwise AD is operational, but it does not yet represent the instantaneous CFD derivative. The learned mass derivative is only 2.3% to 8.2% of the CFD temporal-FD RMS and has near-zero correlation. This is consistent with temporal smoothing under randomly sampled field-value supervision.

## Data-only continuous spacetime continuity baseline

The new diagnostic uses only

`R = d(epsilon*rho)/d(tau) + dJx/dx + dJy/dy + dJz/dz`.

It does not include the legacy finite difference `(m_(n+1)-m_n)/0.1`.

| tau (s) | RMS(A) | RMS(B) | RMS(R) | corr(A,B) | cancellation ratio | RMS dJx/dx | RMS dJy/dy | RMS dJz/dz |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.132 | 3.710 | 3.670 | -0.334 | 0.955 | 0.428 | 3.673 | 0.407 |
| 0.04 | 0.228 | 3.993 | 3.970 | -0.126 | 0.941 | 0.430 | 3.943 | 0.409 |
| 0.06 | 0.321 | 4.392 | 4.433 | 0.101 | 0.941 | 0.427 | 4.323 | 0.409 |
| 0.08 | 0.422 | 4.718 | 4.882 | 0.362 | 0.950 | 0.421 | 4.631 | 0.405 |

The data-only residual is not small and is dominated by y-direction divergence. A small residual was not assumed or enforced; these values are the baseline for a future physics comparison.

## Rollout interface smoke

- `tau_test=0.1 s`: five autoregressive steps completed with finite predictions.
- `tau_test=0.05 s`: the same checkpoint completed two autoregressive steps with finite predictions.
- At 0.1 s, velocity MAE ranges from 0.0383 to 0.0718 m/s, epsilon MAE from 0.0270 to 0.0520, and pressure MAE from 6.66 to 15.13 Pa over steps 1-5.
- At 0.05 s, step-1/2 velocity MAE is 0.1730/0.0650 m/s, epsilon MAE is 0.0304/0.0271, and pressure MAE is 13.62/9.67 Pa.

This is only an interface and numerical-finiteness check, not a claim that the rollout is accurate or stable for long horizons.

## Direct answers

1. **Can one S_n query multiple future times?** Yes. One forward accepts independent `[B,K,N,1]` relative-time queries and returns K future fields/displacements.
2. **Is one latent reused?** Yes. The K=4 smoke hook counted exactly one state encoder call.
3. **Does tau enter both branches?** Yes. Both particle and fluid Perceiver queries add their own 1D time embedding; particle targets are `velocity_(n+tau)*tau` under the existing displacement convention.
4. **Is tau relative physical time?** Yes: `(target_frame-anchor_frame)*0.001 s`. Absolute simulation time and frame number are not model inputs.
5. **Do predictions change continuously with tau?** Yes, and all sampled trajectories are non-constant and smooth, but too temporally smooth relative to CFD.
6. **Is pointwise dq/dtau computable?** Yes. Gradients are finite, query-independent, and have shape `[B,K,864,1]`.
7. **How consistent is dm/dtau_AD with CFD-FD?** Poorly: relative L2 is about 1.0, correlation is about -0.029 to 0.047, and AD RMS is 2.3%-8.2% of CFD-FD RMS.
8. **What is the data-only continuous continuity residual?** RMS 3.67, 3.97, 4.43, and 4.88 at tau 0.02, 0.04, 0.06, and 0.08 s, respectively.
9. **Does 0.1-s rollout work?** Yes, five finite steps.
10. **Does 0.05-s rollout work with the same model?** Yes, two finite steps.
11. **Any NaN, Inf, or broken gradient?** No. All training and diagnostic tensors are finite; no gradient path is broken.
12. **Overall result?** The continuous relative-time representation and its pointwise AD plumbing are valid. The 100-epoch data-only prototype has not learned physically accurate instantaneous temporal derivatives, transverse fields, or particle displacement. No physics optimization was started.

## Code changes

- `src/neural_dem/modules/dit_perceiver_decoder.py`: separate pointwise time embedding and in-graph `tau/0.1` normalization.
- `src/neural_dem/fbed_model.py`: one-time state encoder, multi-tau decoding, fluid-only diagnostic decoding, particle time conditioning, legacy shape/default compatibility.
- `training/config.py`: fixed Stage-6 protocol configuration and `tau_max`.
- `training/dataset.py`: random K-offset continuous-time targets plus deterministic evaluation offsets.
- `training/physics_losses.py`: separately named continuous spacetime residual; legacy residual retained.
- `training/inference.py`: optional 0.05/0.1-s continuous-time rollout while retaining the legacy call.
- `training/continuous_time_utils.py`: pointwise query construction and unchanged multi-horizon data loss.
- `training/continuous_time_smoke_test.py`: time normalization, gradient shape, encoder reuse, and query independence checks.
- `training/continuous_time_train.py`: fixed 5/100-epoch data-only prototype.
- `training/evaluate_continuous_time.py`: fixed-horizon, trajectory, and rollout diagnostics.
- `training/validate_time_derivative.py`: AD versus CFD temporal-FD and continuous continuity diagnostics.

No Stage 1-5 result or checkpoint was overwritten. No loss weighting, normalization, model width/depth, Transformer block, optimizer, physics weight, history encoder, momentum equation, control-volume loss, or conservation projection was introduced.
