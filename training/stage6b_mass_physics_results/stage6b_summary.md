# Stage 6B scientific audit

Stage 6 commit: `3c3b1c9c5445a6f8d5cd0bfa2c1fcc56b2a08256` (`baseline/stage6-continuous-time`, tag `stage6-baseline`)

Stage 6B experiment commit: `4a46b6ec950f2df9ba9345b4354bde2a49e39f18` (`experiment/stage6b-mass-physics`)

Stage 7 archive commit: `fdd55f17e0b81b47372c52a96d23e9c70f35014c` (`archive/stage7-dynamics`, tag `stage7-completed`)

## Scope and result integrity

This audit resumed after the completed Stage 6B run. It did not retrain the model or rerun the completed unified evaluation.

- Stage 6B: 100 epochs, 2,000 optimizer updates, `lambda_mass=1e-3`, seed 42, 9,532,700 parameters.
- Training time: 1,662.057 s (27.701 min), compared with 1,035.802 s (17.263 min) for Stage 6.
- Training and evaluation report no NaN or Inf.
- The final checkpoint loads successfully and records epoch 100.
- Evaluation inventory is complete: 60 fixed-horizon rows, 8 continuity rows, 24 time-AD rows, 56 flux/gradient rows, 14 rollout rows, and 49 comparison rows.
- Final checkpoint SHA-256: `99AB933A3B1469DF5DE921D2CFD319552D9B100D4B67F7D39B6BA33B5AB6E171`.
- The Stage 6B protected model core, dataset, continuous-time utilities, and `physics_losses.py` are unchanged from Stage 6. Stage 6B adds training/evaluation drivers and configuration only.

## Final training losses

Both runs used the same initialization, data, sampled time queries, optimizer, and 100-epoch schedule. The final sampled-tau statistics are identical: min 0.002 s, max 0.099 s, mean 0.050775 s.

| quantity | Stage 6 | Stage 6B | Stage 6B change |
|---|---:|---:|---:|
| total data loss | 0.0647096 | 0.0662661 | +2.41% |
| particle loss | 5.77596e-6 | 9.57097e-6 | +65.70% |
| fluid data loss | 0.0215679 | 0.0220855 | +2.40% |
| u loss | 0.00372945 | 0.00373676 | +0.20% |
| v loss | 0.0376449 | 0.0376143 | -0.08% |
| w loss | 0.000601687 | 0.000589667 | -2.00% |
| epsilon loss | 0.00714198 | 0.00768973 | +7.67% |
| pressure loss | 0.000433931 | 0.000415557 | -4.23% |

Stage 6B final total loss is 0.0666351. Its raw mass loss is 0.368970, and the weighted contribution is 0.000368970, only 0.5537% of total loss. Thus even a small weighted term materially reshapes the learned derivatives while slightly degrading total data fit.

## Fixed-horizon field accuracy

Mean RMSE over evaluation horizons 0.01, 0.025, 0.05, 0.075, and 0.10 s changed as follows:

| field | Stage 6 | Stage 6B | change |
|---|---:|---:|---:|
| u | 0.0606876 | 0.0609731 | +0.47% |
| v | 0.174242 | 0.171527 | -1.56% |
| w | 0.0234474 | 0.0235331 | +0.37% |
| epsilon | 0.0620363 | 0.0626878 | +1.05% |
| pressure (Pa) | 17.5918 | 16.8049 | -4.47% |
| particle displacement | 0.00207353 | 0.00262051 | +26.38% |

The horizon dependence is mixed. Stage 6B improves long-horizon `v` (5.56% at 0.10 s), epsilon (4.00% at 0.10 s), and pressure (14.17% at 0.10 s), but slightly worsens `u` and `w` at every horizon. Particle displacement is worse at every fixed horizon, with the largest cost at short horizons (+76.38% at 0.01 s) and the smallest at 0.10 s (+4.31%). The pressure benefit emerges only after 0.025 s; at 0.01 s pressure RMSE is 2.91% worse.

## Continuous spacetime mass residual

Definitions:

`A = d(epsilon*rho)/dtau`, `Dx = dJx/dx`, `Dy = dJy/dy`, `Dz = dJz/dz`, `B = Dx+Dy+Dz`, and `R = A+B`.

Mean diagnostics over tau = 0.02, 0.04, 0.06, and 0.08 s:

| quantity | Stage 6 | Stage 6B | interpretation |
|---|---:|---:|---|
| RMS(A) | 0.275704 | 0.191929 | smaller on average, but not consistently at 0.02 s |
| RMS(Dx) | 0.426350 | 0.351904 | modest suppression |
| RMS(Dy) | 4.142655 | 0.713156 | dominant 82.8% suppression |
| RMS(Dz) | 0.407315 | 0.379925 | nearly unchanged |
| RMS(B) | 4.203251 | 0.502885 | 88.0% suppression |
| RMS(R) | 4.238638 | 0.672389 | 84.1% reduction |
| corr(A,B) | 0.000755 | 0.834418 | strongly positive, not the required anticorrelation |
| corr(Dy,Dz) | 0.247969 | -0.264851 | directional anticorrelation introduced |
| A/B cancellation ratio | 0.946515 | 0.967702 | no improved A-versus-B cancellation |
| Dy/Dz cancellation ratio | 0.935491 | 0.727442 | substantially more directional cancellation |

At individual horizons, Stage 6B reduces RMS(R) by 80.0%, 83.0%, 85.7%, and 86.8%. RMS(B) falls by 85.9% to 89.2%, driven principally by RMS(Dy), which falls by 80.6% to 84.3%. By contrast, corr(A,B) is +0.826 to +0.845, so `A` and `B` reinforce rather than satisfy `A approximately -B`.

**Scientific classification:** the residual reduction is not evidence that the model learned the intended continuity balance. It is primarily divergence-amplitude suppression, especially suppression of `Dy`, plus additional artificial cancellation between `Dy` and `Dz`. The strongly positive `corr(A,B)` rules out genuine temporal-spatial cancellation as the main mechanism.

The last stochastic training epoch has different sampled-query statistics from the fixed audit and reports corr(A,B) = -0.121, corr(Dy,Dz) = -0.598, and a Dy/Dz cancellation ratio of 0.571. The controlled fixed-tau audit is the appropriate model comparison and confirms that the apparent conservation improvement does not transfer to the correct A-versus-B structure.

## Time-AD versus CFD temporal finite difference

For `dm/dtau`, Stage 6B remains poorly aligned with CFD temporal finite differences:

| tau (s) | Stage 6 corr | Stage 6B corr | Stage 6 RMS ratio | Stage 6B RMS ratio |
|---:|---:|---:|---:|---:|
| 0.02 | 0.0316 | -0.0263 | 0.0232 | 0.0414 |
| 0.04 | 0.0467 | -0.0349 | 0.0364 | 0.0322 |
| 0.06 | -0.0289 | 0.0392 | 0.0638 | 0.0346 |
| 0.08 | -0.0259 | 0.0105 | 0.0821 | 0.0304 |

All relative-L2 errors remain approximately 1.0 and correlations remain near zero. Stage 6B does not learn the CFD temporal derivative pattern. The derivative amplitude is still only 3.0% to 4.1% of the CFD reference RMS; the apparent amplitude change at 0.02 s is not accompanied by correlation improvement.

## Flux, gradient, and divergence fidelity

Flux-field accuracy is essentially unchanged or slightly worse. Mean correlations change from 0.216 to 0.178 for `Jx`, 0.071 to 0.026 for `Jy`, and 0.096 to 0.053 for `Jz`. `Jy` amplitude remains close to the CFD reference, but its spatial correlation is poor.

The gradient comparison gives the decisive result:

| quantity | Stage 6 pred/reference RMS | Stage 6B pred/reference RMS | Stage 6B corr | Stage 6B relative L2 |
|---|---:|---:|---:|---:|
| dJx/dx | 0.0506 | 0.0420 | 0.0061 | 1.0008 |
| dJy/dy | 0.4328 | 0.0741 | -0.0003 | 1.0028 |
| dJz/dz | 0.0374 | 0.0350 | 0.0095 | 1.0001 |
| divergence | 0.3134 | 0.0372 | 0.0107 | 1.0002 |

The Stage 6B divergence RMSE decreases from 14.2454 to 13.5655 only because the prediction collapses toward zero while the CFD reference RMS is 13.5628. Correlation remains approximately zero. This is zero-predictor behavior, not improved flux-gradient or divergence fidelity.

## Autoregressive rollout

No rollout contains NaN, Inf, non-positive pressure, or epsilon below zero. Both models do produce epsilon above one; Stage 6B reduces the count substantially.

For `tau_test=0.10 s`, averaging steps 1--5, Stage 6B changes:

- velocity MAE: 0.051699 to 0.049943 (3.40% better);
- epsilon MAE: 0.037261 to 0.035384 (5.04% better);
- pressure MAE: 10.0078 Pa to 8.0670 Pa (19.39% better);
- particle-position RMSE: 0.004690 to 0.003614 (22.95% better);
- epsilon-above-one cells: 544.6 to 145.8 on average (73.23% fewer).

For `tau_test=0.05 s`, averaging steps 1--2, velocity MAE is 1.28% worse and epsilon MAE is 22.36% worse, while pressure MAE improves 15.41%, particle-position RMSE improves 27.90%, and epsilon-above-one count falls from 537 to 196.

Rollout therefore shows some practically useful long-step stability and pressure/particle benefits, but they are horizon- and field-dependent and do not demonstrate correct local conservation dynamics.

## Final judgment

Stage 6B imposes a clear tradeoff rather than a uniform prediction improvement:

1. The fixed-horizon field result is mixed: pressure and long-horizon `v` improve, `u`/`w` slightly worsen, epsilon is mixed, and particle displacement is substantially worse on average.
2. Long-step rollout metrics improve overall, especially pressure, particle position, and epsilon-bound violations; short-step epsilon and velocity do not improve.
3. The mass residual is numerically much smaller, but the mechanism is not physically faithful. `A` and `B` are strongly positively correlated, while the predicted divergence and especially `Dy` are suppressed toward zero and `Dy`/`Dz` develop artificial cancellation.
4. Time-AD, flux-gradient, and divergence correlations with CFD remain near zero. Therefore the current physics constraint reduces its own residual without learning the corresponding CFD derivative fields.

Consequently, Stage 6B provides no defensible evidence that the mass constraint improved local conservation fidelity. It does provide evidence that the present residual can be optimized through derivative suppression and cancellation, while producing selective long-horizon prediction benefits at the cost of data fit and short-horizon particle accuracy. No further model was selected and no additional training was started.

## Artifacts

- `stage6b_training_history.csv`: complete 100-epoch training record.
- `fixed_horizon_metrics.csv`: field and particle accuracy at five horizons.
- `physics_diagnostics.csv`: A/Dx/Dy/Dz/B/R and cancellation diagnostics.
- `time_derivative_metrics.csv`: time-AD versus CFD temporal-FD.
- `flux_divergence_metrics.csv`: flux/gradient/divergence fidelity.
- `rollout_metrics.csv`: 0.10 s five-step and 0.05 s two-step rollouts.
- `stage6_vs_stage6b.csv`: unified long-form comparison table.
- `stage6b_git_manifest.md`: provenance, hashes, and reproduction pointers.

