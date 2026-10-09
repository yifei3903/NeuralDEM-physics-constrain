# Stage 5: pure-data baseline representation / fitting audit

## Experiment contract

- Source: 0.60 m/s Gidaspow case only.
- Samples: the fixed 20 consecutive pairs `(0 -> 100), ..., (1900 -> 2000)`.
- Teacher-forced one-step prediction only; `dt_ML = 0.1 s`.
- Shared initialization: `training/small_overfit_results/shared_initial_state.pt`.
- Seed 42, AdamW, learning rate `1e-4`, weight decay `1e-4`, batch size 1.
- Pure data baseline throughout: `lambda_mass = 0`; no physics loss or architecture changes.
- Checkpoints: 0, 10, 25, 50, 100, 200, 300, 500 epochs.
- Runtime: 3444.894 s (57.415 min), CPU. No NaN/Inf.

The epoch-300 window test did not declare a plateau, so the pre-declared extension to epoch 500 was used. The final epoch is not the minimum instantaneous training-loss epoch because the unchanged stochastic training procedure produces visible oscillations.

## Normalization and loss audit

| channel | physical min | physical max | target mean | target std | normalization |
|---|---:|---:|---:|---:|---|
| u (m/s) | -0.41562 | 0.41775 | 0.000600 | 0.054679 | identity |
| v (m/s) | 0.09002 | 2.12490 | 0.794417 | 0.325370 | identity |
| w (m/s) | -0.16274 | 0.18067 | -0.00000749 | 0.020592 | identity |
| epsilon | 0.42000 | 1.00000 | 0.845082 | 0.238469 | identity |
| pressure (Pa) | 101320 | 101560 | 101358.856 | 57.264 | `P*=(P-101325)/1000` |

Because u, v, and w are not standardized, their normalized standard deviations are the same values shown above. The v standard deviation is 5.95 times u and 15.80 times w; its variance is about 35.4 and 249.7 times larger, respectively.

The implemented loss is `L_fluid = mean(MSE_u, MSE_v, MSE_w) + MSE_epsilon + MSE_Pstar`, followed by `L_data = L_particle + 3 L_fluid`. Therefore the effective channel weights in `L_data` are `u:v:w:epsilon:P* = 1:1:1:3:3`. The five channels are not equally weighted, and equal formal weights among u/v/w do not compensate their very different physical/normalized scales.

## Teacher-forced field fitting

| epoch | field | RMSE | relative L2 | correlation | pred/CFD std |
|---:|---|---:|---:|---:|---:|
| 10 | u | 0.05546 m/s | 1.014 | 0.008 | 0.042 |
| 10 | v | 0.18667 m/s | 0.217 | 0.820 | 0.825 |
| 10 | w | 0.02860 m/s | 1.389 | -0.012 | 0.187 |
| 10 | epsilon | 0.12157 | 0.138 | 0.875 | 0.913 |
| 10 | pressure | 23.73 Pa | 0.000234 | 0.923 | 0.876 |
| 200 | u | 0.04028 m/s | 0.737 | 0.680 | 0.613 |
| 200 | v | 0.10537 m/s | 0.123 | 0.948 | 0.903 |
| 200 | w | 0.02070 m/s | 1.005 | 0.129 | 0.293 |
| 200 | epsilon | 0.04048 | 0.046 | 0.987 | 0.940 |
| 200 | pressure | 14.71 Pa | 0.000145 | 0.986 | 0.839 |
| 500 | u | 0.03127 m/s | 0.572 | 0.822 | 0.824 |
| 500 | v | 0.07774 m/s | 0.0906 | 0.971 | 0.986 |
| 500 | w | 0.01778 m/s | 0.863 | 0.505 | 0.496 |
| 500 | epsilon | 0.02023 | 0.0230 | 0.997 | 0.996 |
| 500 | pressure | 6.40 Pa | 0.0000631 | 0.996 | 0.926 |

At epoch 500, density derived from the EOS has RMSE `7.49e-5 kg/m^3`, relative L2 `6.31e-5`, and correlation 0.9962.

The baseline did **not** fully overfit every fluid channel. It fit v, epsilon, and pressure well; u became useful but remained imperfect; w improved substantially from epoch 10 but remained the weakest channel.

## Centered fluctuations and smoothing

| epoch | field | fluctuation relative L2 | centered correlation | fluctuation pred/CFD RMS |
|---:|---|---:|---:|---:|
| 10 | u | 1.001 | -0.013 | 0.041 |
| 10 | w | 1.021 | -0.012 | 0.187 |
| 10 | Jx | 1.006 | -0.041 | 0.077 |
| 10 | Jz | 1.027 | -0.008 | 0.226 |
| 500 | u | 0.569 | 0.822 | 0.822 |
| 500 | v | 0.237 | 0.972 | 0.986 |
| 500 | w | 0.854 | 0.521 | 0.483 |
| 500 | Jx | 0.610 | 0.793 | 0.796 |
| 500 | Jy | 0.747 | 0.665 | 0.685 |
| 500 | Jz | 0.876 | 0.482 | 0.493 |

The final u field is mainly a reduced-amplitude but correctly located fluctuation field. The final w field has both effects: it is strongly smoothed (only 48.3% of CFD fluctuation RMS) and its spatial pattern is only moderately correct (centered correlation 0.521). It is not merely a full-amplitude fluctuation shifted to the wrong cells.

## Mass-flux fitting

| epoch | Jx rel. L2 / corr. | Jy rel. L2 / corr. | Jz rel. L2 / corr. |
|---:|---:|---:|---:|
| 10 | 1.030 / -0.017 | 0.203 / -0.013 | 1.643 / -0.008 |
| 100 | 0.935 / 0.366 | 0.146 / 0.248 | 1.090 / 0.034 |
| 200 | 0.768 / 0.641 | 0.135 / 0.429 | 1.010 / 0.133 |
| 500 | 0.614 / 0.792 | 0.113 / 0.660 | 0.899 / 0.446 |

Jx improved strongly, Jy raw relative L2 was always helped by its large mean but its centered correlation rose to only 0.665, and Jz improved but remained poor and smoothed.

## NN-FD versus CFD-FD (interior cells)

| epoch | quantity | relative L2 | correlation | NN/CFD RMS |
|---:|---|---:|---:|---:|
| 10 | dJx/dx | 1.000 | -0.012 | 0.016 |
| 10 | dJy/dy | 1.331 | -0.011 | 0.889 |
| 10 | dJz/dz | 1.018 | -0.030 | 0.093 |
| 200 | dJx/dx | 0.882 | 0.472 | 0.439 |
| 200 | dJy/dy | 0.974 | 0.256 | 0.418 |
| 200 | dJz/dz | 1.001 | 0.099 | 0.253 |
| 500 | dJx/dx | 0.754 | 0.657 | 0.665 |
| 500 | dJy/dy | 0.857 | 0.491 | 0.572 |
| 500 | dJz/dz | 0.923 | 0.364 | 0.437 |
| 500 | divergence | 0.963 | 0.308 | 0.532 |

The derivative structure improves naturally as cell-center fields improve, but is still inaccurate. This is not Case D: the corresponding centered u/w/Jx/Jz fields are not yet fitted well enough for a clean derivative test.

For y-z cancellation, the all-cell baseline evolves from `corr(Dy,Dz)=-0.0448, C_yz=0.9238` at epoch 10 to `-0.0562, 0.6878` at epoch 500. The CFD reference is approximately `-0.3582, 0.7128`. Cancellation magnitude becomes close, but the directional anticorrelation remains much weaker than CFD; it does not collapse to the near-perfect `-0.996` anticorrelation previously observed in the physics run.

## Per-channel gradient audit

The gradients are not disconnected. At epoch 500, effective gradient norms `(decoder, encoder, transformer)` are:

| channel | fixed-batch channel loss | decoder | encoder | transformer |
|---|---:|---:|---:|---:|
| u | 9.94e-4 | 4.94e-2 | 1.22e-3 | 2.83e-5 |
| v | 1.41e-2 | 8.26e-1 | 1.22e-2 | 2.77e-4 |
| w | 4.21e-4 | 1.09e-1 | 1.29e-3 | 2.49e-5 |
| epsilon | 3.38e-4 | 5.70e-1 | 8.21e-3 | 2.26e-4 |
| pressure | 4.00e-5 | 1.55e-1 | 2.11e-3 | 3.21e-5 |

At initialization the effective decoder norms were u 15.99, v 60.85, w 23.62, epsilon 99.23, and pressure 34.47. Thus u/w can backpropagate, but their effective influence is materially smaller than the dominant channels after the easy mean/low-frequency fit. The zero encoder/transformer gradients at epoch 0 are caused by the model's initially closed residual gates; nonzero gradients appear after training.

## Direct answers and classification

1. **Did it truly overfit all 20 pairs?** No. v, epsilon, and pressure are fitted well; u is partially fitted; w remains materially underfit. Total loss alone would hide this.
2. **Why were u/w poor at 10 epochs?** Ten epochs were strongly under-trained: both fields, fluxes, and derivatives improve through epoch 500. However, the identity normalization plus MSE means that v variance dominates u/w, while epsilon and pressure receive 3x explicit weight. The diagnosis is therefore under-training compounded by loss/normalization imbalance, not a broken gradient path.
3. **Smoothing or misplaced fluctuations?** Both, especially for w. At epoch 500 u has centered correlation 0.822 and std ratio 0.822; w has correlation 0.521 and std ratio 0.483. w is predominantly over-smoothed and also spatially imperfect.
4. **Do Jx/Jy/Jz improve?** Yes, all improve, especially Jx. Jz remains the weakest and retains relative L2 0.899 with correlation 0.446.
5. **Do NN-FD derivatives improve naturally?** Yes, in the broad checkpoint trend, but final derivative errors remain large, especially dJz/dz and divergence.
6. **Can the current architecture express these spatial structures?** It demonstrably expresses substantial u/w structure that was absent at epoch 10, so the earlier failure was not proof of a hard representation limit. The evidence best supports a **mixed diagnosis: primary under-training plus loss/normalization imbalance**. A hard architecture representation limitation is not established because u/w were still improving rather than cleanly plateaued, but complete memorization of w/high-frequency derivatives was not demonstrated even after 500 epochs.

No physics term, reweighting, normalization change, architecture change, rollout, or next-stage training was performed.
