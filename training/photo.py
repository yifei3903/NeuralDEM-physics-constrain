"""
多时间步预测 vs 真实 CFD-DEM 对比可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from config import DataConfig, DATA_SOURCES
from dataset import read_fluid_file

# ===== 配置 =====
nz, nx, ny = 3, 12, 24
predictions_dir = r"D:\NeuralDEM\NeuralDEM-main\training\predictions_reduced"
source = DATA_SOURCES[0]      # 切换工况: 0=0.60m/s, 1=0.74m/s
data_dir = source["fluid_dir"]
data_cfg = DataConfig()
start_frame = source["start_frame"]
start_frame = 5000
frame_interval = 100

step_indices = [9, 29, 49, 69, 89, 109, 129, 149]   # 预测步数索引 (0-based)

# ========================================================
# 图1: 空隙率演化
# ========================================================
fig, axes = plt.subplots(2, len(step_indices), figsize=(4 * len(step_indices), 8))

for col, step_idx in enumerate(step_indices):
    pred = np.load(f"{predictions_dir}/step_{step_idx:04d}.npz")
    pred_vf = pred["fluid_voidfraction"].flatten().reshape(nx, ny, nz)
    im = axes[0, col].imshow(pred_vf[:, :, 1].T, origin="lower", aspect="auto",
                              cmap="viridis", vmin=0.4, vmax=1.0)
    axes[0, col].set_title(f"Pred step {step_idx}", fontsize=10)
    plt.colorbar(im, ax=axes[0, col])

    real_frame = start_frame + (step_idx + 1) * frame_interval
    _, real_vf, _ = read_fluid_file(f"{data_dir}/fluid_{real_frame}.txt", data_cfg)
    real_vf = real_vf.reshape(nx, ny, nz)
    im = axes[1, col].imshow(real_vf[:, :, 1].T, origin="lower", aspect="auto",
                              cmap="viridis", vmin=0.4, vmax=1.0)
    axes[1, col].set_title(f"CFD-DEM f{real_frame}", fontsize=10)
    plt.colorbar(im, ax=axes[1, col])

axes[0, 0].set_ylabel("Predicted", fontsize=12)
axes[1, 0].set_ylabel("CFD-DEM", fontsize=12)
fig.suptitle(f"Voidfraction evolution (inlet={source['inlet_velocity']}m/s, mid-z)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{predictions_dir}/evolution_voidfraction.png", dpi=150)
plt.show()

# ========================================================
# 图2: 垂直速度 vy 演化（本算例重力和入口方向为 y）
# ========================================================
fig, axes = plt.subplots(2, len(step_indices), figsize=(4 * len(step_indices), 8))

for col, step_idx in enumerate(step_indices):
    pred = np.load(f"{predictions_dir}/step_{step_idx:04d}.npz")
    pred_vel = pred["fluid_vel"].reshape(-1, 3)
    pred_vy = pred_vel[:, 1].reshape(nx, ny, nz)
    im = axes[0, col].imshow(pred_vy[:, :, 1].T, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[0, col].set_title(f"Pred step {step_idx}", fontsize=10)
    plt.colorbar(im, ax=axes[0, col])

    real_frame = start_frame + (step_idx + 1) * frame_interval
    real_vel, _, _ = read_fluid_file(f"{data_dir}/fluid_{real_frame}.txt", data_cfg)
    real_vy = real_vel[:, 1].reshape(nx, ny, nz)
    im = axes[1, col].imshow(real_vy[:, :, 1].T, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[1, col].set_title(f"CFD-DEM f{real_frame}", fontsize=10)
    plt.colorbar(im, ax=axes[1, col])

axes[0, 0].set_ylabel("Predicted", fontsize=12)
axes[1, 0].set_ylabel("CFD-DEM", fontsize=12)
fig.suptitle(f"Vertical velocity vy evolution (inlet={source['inlet_velocity']}m/s, mid-z)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{predictions_dir}/evolution_velocity.png", dpi=150)
plt.show()

# ========================================================
# 数值误差
# ========================================================
print(f"{'Step':<10} {'MAE vf':<12} {'MAE vz':<12}")
print("-" * 34)
for step_idx in step_indices:
    pred = np.load(f"{predictions_dir}/step_{step_idx:04d}.npz")
    real_frame = start_frame + (step_idx + 1) * frame_interval
    real_vel, real_vf, _ = read_fluid_file(f"{data_dir}/fluid_{real_frame}.txt", data_cfg)
    mae_vf = np.abs(pred["fluid_voidfraction"].flatten() - real_vf.flatten()).mean()
    mae_vy = np.abs(pred["fluid_vel"].reshape(-1, 3)[:, 1] - real_vel[:, 1]).mean()
    print(f"step_{step_idx:<4d} {mae_vf:<12.4f} {mae_vy:<12.4f}")
