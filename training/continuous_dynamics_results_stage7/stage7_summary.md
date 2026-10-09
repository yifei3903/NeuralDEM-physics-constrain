# Stage 7 — Local Continuous Dynamics NeuralDEM

## 结论先行

Stage 7 的软件结构已经把流体输出从 future-state query 改为局部生成器接口

```text
S_n -> G_theta(S_n) -> [du/dt,dv/dt,dw/dt,dm_star/dt,dP_star/dt]
    -> explicit Euler (dt=0.001 s) -> S_(n+1)
```

但这次固定设置的 500-epoch、data-only 训练并没有在数值上学到可信的局部瞬时动力学。预测 rate 的 RMS 仅为 CFD reference 的约 `0.02%~36.6%`，速度和 `m_star` 的相关系数接近零；1-ms learned update 也没有优于 zero-rate identity baseline。换句话说：**生成器语义和积分接口成立，但训练得到的算子主要退化为近零 rate，不能声称已学到真实 local dynamics。**

100-step 真正自回归积分全部保持 finite，没有 NaN/Inf 或非法 rollout 状态；不过误差明显累积，轨迹对 `u/w/epsilon/m_star` 仍严重过度平滑，pressure 则出现平滑但不准确的单向漂移。本阶段没有加入任何质量守恒训练或 Stage-8 更新。

## 固定实验协议

- 数据：仅 `gidaspow_060`，20 个 block，起点为 `0,100,...,1900`。
- 每个 epoch：每个 block 独立随机抽取 `r in [0,99]`，训练 `frame(block+r) -> frame(block+r+1)`；20 updates/epoch。
- 时间积分：explicit Euler，`dt_dyn=0.001 s`；没有积分步长扫描。
- 训练：500 epochs，共 10,000 updates，seed 42，AdamW，`lr=1e-4`，`weight_decay=1e-4`，batch size 1。
- 模型：9,532,700 parameters，保持 reduced NeuralDEM 的 encoder、双分支 Transformer、supernode pooling、fluid patch embedding 和 Perceiver decoder 规模。
- 初始化：`training/small_overfit_results/shared_initial_state.pt`。
- 加载审计：141 keys loaded；newly initialized、missing、unexpected keys 均为空。
- 总耗时：2875.02 s（47.92 min，CPU）。
- checkpoint：epoch 100、300、500 均已保存。
- 训练全程没有 NaN/Inf；没有按中途结果自动延长训练或调整优化器。

固定参考密度由 EOS 直接计算：

```text
rho_ref = P_ref * MW / (R*T)
        = 101325 * 29 / (8314.56 * 298.15)
        = 1.185333408 kg/m^3
```

外部 encoder state 仍为 `[u,v,w,epsilon,P_star]`。Euler 内部 state 为 `[u,v,w,m_star,P_star]`，其中

```text
rho(P) = P*MW/(R*T)
m_star = epsilon*rho(P)/rho_ref
m_star_(n+1) = m_star_n + dt*dm_star_dt
epsilon_(n+1) = rho_ref*m_star_(n+1)/rho(P_(n+1))
```

训练监督仍然只来自积分后的下一帧 data loss，**没有使用 CFD finite-difference derivative label 作为 loss**。

## Smoke test 与梯度路径

| 检查 | 结果 |
|---|---:|
| fluid rate shape | `[1,864,5]` |
| particle displacement shape | `[9240,3]` |
| rate / Euler state / m_star / epsilon / pressure / rho | 全部 finite |
| `dt=0` internal max error | `0` |
| `dt=0` external-state max error | `5.96e-8` |
| Euler internal identity max error | `1.17e-7` |
| 初始 fluid-decoder gradient norm | `2.18e-4` |
| 一次 optimizer step 后 Transformer gradient norm | `6.86e-4` |
| 一次 optimizer step 后 fluid-encoder gradient norm | `2.80e-8` |

初始化时 Transformer 和 fluid encoder 的 gradient tensor 存在但 norm 为零，这是原模型 zero-gated decoder 路径的结果；完成一次 optimizer update 后，两者都收到非零梯度。因此 Euler integration 到 decoder、Transformer、fluid encoder 的反向链路已验证。

Stage-6 backward compatibility 也单独复查：`continuous_time_epoch_100.pt` strict load 的 missing/unexpected keys 均为空，显式四个 `query_tau` 的输出仍为 fluid `[1,4,864,5]`、particle `[1,4,9240,3]`，全部 finite。

## 训练行为与 1-ms one-step 准确度

第 500 epoch 的随机 20-transition 平均 total data loss 为 `5.462e-4`，fluid loss 为 `1.821e-4`，particle loss 为 `8.134e-9`。最佳单个训练 epoch 是 epoch 169，loss 为 `3.267e-4`；训练后期仍存在由每 epoch 重新采样 transition 引起的明显波动，最后 50 epochs 平均 loss 为 `2.045e-3`。

下面的 one-step 指标使用固定的内部测试 frames。为了判断很小的 1-ms loss 是否真的来自 learned dynamics，额外给出不更新状态的 zero-rate identity baseline。

| field | learned RMSE | identity RMSE | learned relative L2 | learned corr |
|---|---:|---:|---:|---:|
| u | 0.008512 | 0.008512 | 0.1408 | 0.9902 |
| v | 0.015381 | 0.015377 | 0.01777 | 0.9990 |
| w | 0.006330 | 0.006330 | 0.2649 | 0.9647 |
| epsilon | 0.006276 | 0.006276 | 0.007149 | 0.9997 |
| pressure (Pa) | 1.7985 | 1.7414 | `1.77e-5` | 0.9995 |
| particle displacement (m) | `7.370e-5` | `3.415e-5` | 2.159 | -0.0635 |

所有 learned one-step RMSE 都与 identity 相当或略差。因此 one-step pipeline 能训练、能 backward、能输出下一帧，但这些低误差主要说明相邻 CFD frames 很接近，**不能作为 generator 已学到局部变化率的证据**。

## Rate 与 CFD temporal difference

预测表示为 `[du/dt,dv/dt,dw/dt,dm_star/dt,dP_star/dt]`。`P_star` 是现有 pressure normalization，不是 Pa；训练时没有使用下表 reference。

### CFD forward difference

| quantity | pred RMS | reference RMS | RMS ratio | RMSE | relative L2 | corr |
|---|---:|---:|---:|---:|---:|---:|
| du/dt | 0.01372 | 8.51169 | 0.001612 | 8.51176 | 1.00001 | 0.0232 |
| dv/dt | 0.20101 | 15.37665 | 0.01307 | 15.38143 | 1.00031 | 0.00095 |
| dw/dt | 0.001467 | 6.33000 | 0.000232 | 6.33000 | 1.00000 | -0.00762 |
| dm_star/dt | 0.01103 | 6.27932 | 0.001756 | 6.27935 | 1.00000 | 0.00844 |
| dP_star/dt | 0.48499 | 1.74138 | 0.27851 | 1.79863 | 1.03288 | -0.0346 |

### CFD centered difference

| quantity | pred RMS | reference RMS | RMS ratio | RMSE | relative L2 | corr |
|---|---:|---:|---:|---:|---:|---:|
| du/dt | 0.01372 | 6.73025 | 0.002039 | 6.73034 | 1.00001 | 0.0304 |
| dv/dt | 0.20101 | 12.44724 | 0.01615 | 12.44993 | 1.00022 | 0.0192 |
| dw/dt | 0.001467 | 4.85764 | 0.000302 | 4.85764 | 1.00000 | -0.00633 |
| dm_star/dt | 0.01103 | 4.79729 | 0.002298 | 4.79733 | 1.00001 | 0.00979 |
| dP_star/dt | 0.48499 | 1.32637 | 0.36565 | 1.35406 | 1.02088 | -0.3221 |

结论很明确：除 pressure rate 具有较大的 RMS 外，网络 rate 的振幅远小于 CFD；所有 rate correlation 都很低。这个 checkpoint 学到的是近零、弱状态相关的 rate，而不是真实 CFD temporal derivative。

## 与 Stage 6 的概念性比较

Stage 6 的 mass AD（量纲 mass rate）相对 CFD temporal FD 的 RMS ratio 在 `0.0232~0.0821`，correlation 在 `-0.0289~0.0467`；Stage 7 的 `dm_star/dt` ratio 为 `0.00176`（forward）或 `0.00230`（centered），correlation 为 `0.00844/0.00979`。mass 与 m_star 的固定 `rho_ref` 缩放不影响 relative L2、correlation 或 RMS ratio。

所以 Stage 7 的 learned mass rate **没有比 Stage 6 更接近真实时间导数**；它反而更接近 zero rate。由于 Stage 6/7 的目标、训练预算及评估时刻不同，这里只作诊断，不宣称严格的架构优劣。

下面列出 Stage-6 direct teacher-forced query 与 Stage-7 repeated-Euler rollout 的 RMSE，帮助理解两种时间表示；二者不是同一种评估协议，不能直接当成公平排行榜。

| horizon | model/protocol | u | v | w | epsilon | pressure (Pa) |
|---:|---|---:|---:|---:|---:|---:|
| 0.01 s | Stage 6 direct query | 0.0672 | 0.1822 | 0.0271 | 0.0633 | 20.84 |
| 0.01 s | Stage 7, 10 Euler steps | 0.0950 | 0.4335 | 0.0418 | 0.0299 | 15.25 |
| 0.05 s | Stage 6 direct query | 0.0593 | 0.1879 | 0.0234 | 0.0575 | 17.14 |
| 0.05 s | Stage 7, 50 Euler steps | 0.0942 | 0.5337 | 0.0352 | 0.0974 | 41.77 |
| 0.10 s | Stage 6 direct query | 0.0545 | 0.1490 | 0.0208 | 0.0665 | 16.64 |
| 0.10 s | Stage 7, 100 Euler steps | 0.0780 | 0.4764 | 0.0237 | 0.1091 | 56.29 |

## 真正自回归 rollout

rollout 使用 4 个 anchors `[0,500,1000,1500]`。每个 1-ms step 都重新编码当前预测状态、预测 rate、Euler 更新流体、以 particle `query_tau=0.001` 更新颗粒、重新 patchify；中间没有读取真实 CFD 状态。

| steps / horizon | u RMSE | v RMSE | w RMSE | epsilon RMSE | pressure RMSE (Pa) | particle-position RMSE (m) |
|---|---:|---:|---:|---:|---:|---:|
| 10 / 0.01 s | 0.0950 | 0.4335 | 0.04184 | 0.02988 | 15.25 | 0.000738 |
| 50 / 0.05 s | 0.0942 | 0.5337 | 0.03516 | 0.09736 | 41.77 | 0.004321 |
| 100 / 0.10 s | 0.0780 | 0.4764 | 0.02372 | 0.10909 | 56.29 | 0.007228 |

数值意义上的稳定性检查通过：所有 100-step trajectories finite，没有 NaN/Inf、negative pressure 或 rollout 中的 epsilon 越界。但预测意义上的稳定性较差：epsilon、pressure 和 particle-position error 随 horizon 显著增长，速度 correlation 也偏低。因此这里的“稳定”只表示积分没有数值爆炸，不表示长期预测准确。

## Trajectory smoothness

`dynamics_trajectory_samples.csv` 保存 4 anchors × 3 cells × 6 channels × 101 times（0 到 0.1 s）的预测和 CFD 值。Stage-7 prediction 在 0.1 s 内的平均 range 为：

| channel | Stage-7 mean range | max adjacent 1-ms jump | Stage-6 mean range |
|---|---:|---:|---:|
| u | 0.00124 | `1.38e-5` | 0.00526 |
| v | 0.01938 | `2.02e-4` | 0.02374 |
| w | 0.000253 | `5.74e-6` | 0.00264 |
| epsilon | 0.00152 | `1.72e-5` | 0.02872 |
| m_star | 0.00114 | `1.24e-5` | n/a |
| pressure (Pa) | 48.32 | 0.492 | 2.468 |

`u/w/epsilon/m_star` 比 Stage 6 还更平滑，符合近零 rate 的导数诊断；`v` 的变化幅度相近。pressure 并非近似常量，而是非常平滑地累计约 48 Pa 漂移，其 sampled-trajectory prediction/CFD correlation 仅约 0.12，因此也不是正确的时间动力学。结论是：**过度平滑问题没有解决，只是 pressure 通道表现为平滑的错误漂移。**

## 非法状态审计

没有添加任何 epsilon、pressure、m_star 或 rate clamp。

| scope | epsilon<0 | epsilon>1 | P<=0 | NaN | Inf | epsilon range | pressure range (Pa) |
|---|---:|---:|---:|---:|---:|---|---|
| 所有训练 one-step predictions | 0 | 2,851,458 | 0 | 0 | 0 | `[0.419778,1.000364]` | `[101069.99,101620.34]` |
| 4-anchor 100-step rollout | 0 | 0 | 0 | 0 | 0 | `[0.418566,0.999984]` | `[101320.48,101598.52]` |

训练 one-step 的 epsilon>1 数量大，但最大值只超出 1 约 `3.64e-4`；该事实按要求原样记录，未用 clamp 隐藏。最终 rollout 本身没有出现非法状态。

## 对 14 个必答问题的直接回答

1. **是否已从 future-state predictor 改成 local dynamics generator？** 代码/API 和输出语义上是：Stage-7 先输出 rate，再由 Euler 得到 future state；但实证上该 checkpoint 没有学到可信的 local CFD dynamics，主要退化为近零 rate。
2. **fluid dynamics decoder 是否完全不依赖 tau？** 是。`predict_dynamics()` 调用 spatial-only decoder 路径，fluid query 仅含归一化的 `(x,y,z)`。Stage-6 time-conditioned 路径单独保留。
3. **未来 horizon 是否只由 integration step × number of steps 决定？** 是。Stage 7 固定 `dt_dyn=0.001 s`；10/50/100 steps 分别对应 0.01/0.05/0.10 s。
4. **五个 rate 是否 finite？** 是，smoke、训练、rate validation 和 rollout 均 finite。
5. **1-ms prediction 能否正常学习？** pipeline 可优化且 loss 很低，但未优于 zero-rate identity；因此不能说学到了有意义的 one-step dynamics。
6. **与 CFD forward difference 一致程度？** velocity/mass rate relative L2 约 1、correlation 约 0、RMS ratio `0.00023~0.0131`；pressure ratio 0.279 但 correlation -0.0346。整体不一致。
7. **与 CFD centered difference 一致程度？** 同样不一致；velocity/mass RMS ratio `0.00030~0.0161`，correlation `-0.0063~0.0304`；pressure ratio 0.366、correlation -0.322。
8. **dm_star/dt 是否比 Stage-6 dm/dtau_AD 更接近真实 derivative？** 否。Stage-7 RMS ratio 0.00176/0.00230，低于 Stage-6 的 0.0232~0.0821，相关性也仍接近零。
9. **10/50/100-step rollout 是否稳定？** 数值上均 finite 且无非法 rollout 状态；准确度上误差明显累积，不能称为可靠的长期稳定预测。
10. **trajectory 是否仍过度平滑？** 是，尤其 `u/w/epsilon/m_star`；pressure 是平滑但错误的累计漂移。
11. **是否有 epsilon<0、epsilon>1、P<=0、NaN、Inf？** 训练 one-step 有 2,851,458 个轻微 epsilon>1 值（max 1.000364），其余为零；100-step rollout 五类计数全部为零。
12. **particle branch 是否保持 Stage-6 行为？** 是。仍使用 continuous displacement decoder，每一步 `query_tau_particle=0.001 s`，没有改为 particle ODE，也没有修改 particle loss/matching。但本 checkpoint 的 particle one-step/rollout accuracy 较差，不应把接口保持解释为性能保持。
13. **修改了哪些文件？** 见下一节。
14. **是否完全没有使用被禁止的方法？** 是：没有 mass loss、continuity training、physics-defined `dm/dt`、face flux、finite volume、migration probability、history window 或 momentum equation；`lambda_mass=0`。

## 文件变更

修改：

- `src/neural_dem/modules/dit_perceiver_decoder.py`：增加可选的 spatial-only decode 开关；原有 time-conditioned 默认路径不变。
- `src/neural_dem/fbed_model.py`：增加 `predict_dynamics()`；同一次 current-state encoding 生成 spatial fluid rate，并让 particle branch 查询固定 1 ms displacement；原 Stage-6 `forward(...,query_tau=...)` 保留。
- `training/config.py`：增加固定的 Stage-7 `DynamicsConfig`。
- `training/dataset.py`：增加按 20 个 block 独立随机采样 one-frame transition 的 `LocalDynamicsDataset`。

新增：

- `training/dynamics_utils.py`：EOS、`m_star` 转换、Euler update、原 data loss、invalid-state 统计和严格初始化加载。
- `training/dynamics_smoke_test.py`：shape、finite、Euler identity、dt=0 和梯度链路测试。
- `training/dynamics_train.py`：固定 500-epoch data-only Stage-7 训练。
- `training/evaluate_dynamics.py`：one-step learned/identity 对照与非法状态审计。
- `training/validate_dynamics_rate.py`：forward/centered CFD rate comparison 与 Stage-6 derivative 摘要。
- `training/dynamics_rollout.py`：10/50/100-step 真正自回归 rollout 和 trajectory CSV。

`training/inference.py` 无需修改。Stage 1~6 的 checkpoint 和结果未覆盖。

## 结果文件

- `checkpoints/epoch_100.pt`
- `checkpoints/epoch_300.pt`
- `checkpoints/epoch_500.pt`
- `training_curve.csv`
- `one_step_accuracy.csv`
- `rate_vs_cfd_forward.csv`
- `rate_vs_cfd_centered.csv`
- `rate_summary.json`
- `rollout_accuracy.csv`
- `dynamics_trajectory_samples.csv`
- `invalid_state_statistics.json`
- `smoke_test.json`
- `training_summary.json`

Stage 7 到此停止。没有启动 Stage 8，也没有实现或训练 `dm_star/dt = -(1/rho_ref) div(m u)`。
