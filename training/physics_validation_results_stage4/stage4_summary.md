# Stage 4 diagnostic summary

No model was trained or modified. Main scientific comparisons use the common interior mask.

## Automatic diagnostic classification

| model | case | diagnosis |
|---|---:|---|
| baseline | 2 | predicted discrete spatial field is not accurate enough |
| physics_1e-03 | 2 | predicted discrete spatial field is not accurate enough |
| physics_1e-02 | 2 | predicted discrete spatial field is not accurate enough |

## Temporal schemes: interior-only

| scheme | residual MAE | residual RMS |
|---|---:|---:|
| FE | 3.98196 | 6.5495 |
| BE_current | 4.07531 | 6.41966 |
| endpoint_trapezoid | 3.09613 | 4.6797 |
| midpoint | 4.39458 | 7.09942 |
| full_time_average | 2.91406 | 4.32035 |

## Flux-value accuracy: interior-only

| model | flux | relative L2 | correlation |
|---|---|---:|---:|
| baseline | Jx | 1.02409 | -0.0425249 |
| baseline | Jy | 0.245669 | -0.448702 |
| baseline | Jz | 1.18083 | 0.0853714 |
| physics_1e-03 | Jx | 1.0967 | 0.054687 |
| physics_1e-03 | Jy | 0.226069 | -0.476259 |
| physics_1e-03 | Jz | 1.00296 | -0.0458733 |
| physics_1e-02 | Jx | 1.00268 | 0.0891943 |
| physics_1e-02 | Jy | 0.237058 | -0.503954 |
| physics_1e-02 | Jz | 4.78227 | -0.0641198 |

## Derivative-source isolation: interior-only divergence

| model | comparison | relative L2 | correlation |
|---|---|---:|---:|
| baseline | nn_fd_vs_cfd_fd | 1.31234 | -0.0204728 |
| baseline | nn_ad_vs_nn_fd | 0.017552 | 0.99999 |
| baseline | nn_ad_vs_cfd_fd | 1.30718 | -0.0204721 |
| physics_1e-03 | nn_fd_vs_cfd_fd | 1.06507 | -0.0239315 |
| physics_1e-03 | nn_ad_vs_nn_fd | 0.0407007 | 0.999226 |
| physics_1e-03 | nn_ad_vs_cfd_fd | 1.06381 | -0.0232957 |
| physics_1e-02 | nn_fd_vs_cfd_fd | 1.01925 | 0.00875173 |
| physics_1e-02 | nn_ad_vs_nn_fd | 0.0254528 | 0.999666 |
| physics_1e-02 | nn_ad_vs_cfd_fd | 1.01877 | 0.00880491 |

## y-z cancellation: interior-only

| model | representation | corr(Dy,Dz) | C_yz |
|---|---|---:|---:|
| CFD_reference | CFD-FD | -0.358211 | 0.712845 |
| baseline | NN-FD | -0.496422 | 0.962119 |
| baseline | NN-AD | -0.476758 | 0.962101 |
| physics_1e-03 | NN-FD | -0.740574 | 0.35118 |
| physics_1e-03 | NN-AD | -0.742505 | 0.348044 |
| physics_1e-02 | NN-FD | -0.996138 | 0.130023 |
| physics_1e-02 | NN-AD | -0.996256 | 0.127672 |

Full raw metrics are stored in the adjacent CSV and JSON files.
