"""
生成预测演化 GIF 动图：空隙率 + 垂直速度，与真实 CFD-DEM 对比
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from config import DATA_SOURCES, DataConfig
from dataset import read_fluid_file

# ===== 配置 =====
nz, nx, ny = 3, 12, 24
predictions_dir = r"D:\NeuralDEM\NeuralDEM-main\training\predictions_reduced"
source = DATA_SOURCES[0]
data_dir = source["fluid_dir"]
data_cfg = DataConfig()
start_frame = 5000
frame_interval = 100
num_steps = 150

# ===== 创建画布 =====
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 预加载第一帧设置 colorbar 范围
pred0 = np.load(f"{predictions_dir}/step_0000.npz")
vf0 = pred0["fluid_voidfraction"].flatten().reshape(nx, ny, nz)
vel0 = pred0["fluid_vel"].reshape(-1, 3)[:, 1].reshape(nx, ny, nz)

writer = PillowWriter(fps=8)
writer.setup(fig, f"{predictions_dir}/evolution.gif", dpi=100)

for i in range(num_steps):
    real_frame = start_frame + (i + 1) * frame_interval

    # 读取预测
    pred = np.load(f"{predictions_dir}/step_{i:04d}.npz")
    pred_vf = pred["fluid_voidfraction"].flatten().reshape(nx, ny, nz)
    pred_vel = pred["fluid_vel"].reshape(-1, 3)
    pred_vy = pred_vel[:, 1].reshape(nx, ny, nz)

    # 读取真实 CFD-DEM
    real_vel, real_vf, _ = read_fluid_file(f"{data_dir}/fluid_{real_frame}.txt", data_cfg)
    real_vf = real_vf.reshape(nx, ny, nz)
    real_vy = real_vel[:, 1].reshape(nx, ny, nz)

    # 清除上一帧
    for ax in axes.flat:
        ax.clear()

    # 物理时间
    t_phys = (start_frame + (i + 1) * frame_interval) * 0.001

    # 左下: 预测空隙率
    im = axes[0, 0].imshow(pred_vf[:, :, 1].T, origin="lower", aspect="auto",
                           cmap="viridis", vmin=0.4, vmax=1.0)
    axes[0, 0].set_title(f"Pred voidfraction  t={t_phys:.1f}s", fontsize=10)
    axes[0, 0].set_ylabel("x (mid-z)")

    # 右下: 真实空隙率
    im2 = axes[0, 1].imshow(real_vf[:, :, 1].T, origin="lower", aspect="auto",
                            cmap="viridis", vmin=0.4, vmax=1.0)
    axes[0, 1].set_title(f"CFD-DEM voidfraction  f{real_frame}", fontsize=10)

    # 左下: 预测 vy
    im3 = axes[1, 0].imshow(pred_vy[:, :, 1].T, origin="lower", aspect="auto",
                            cmap="RdBu_r", vmin=-0.5, vmax=1.5)
    axes[1, 0].set_title(f"Pred vy  t={t_phys:.1f}s", fontsize=10)
    axes[1, 0].set_ylabel("x (mid-z)")

    # 右下: 真实 vy
    im4 = axes[1, 1].imshow(real_vy[:, :, 1].T, origin="lower", aspect="auto",
                            cmap="RdBu_r", vmin=-0.5, vmax=1.5)
    axes[1, 1].set_title(f"CFD-DEM vy  f{real_frame}", fontsize=10)

    fig.suptitle(f"Autoregressive rollout (inlet={source['inlet_velocity']} m/s, frame {i+1}/{num_steps})",
                 fontsize=12)
    plt.tight_layout()

    writer.grab_frame()
    if i % 20 == 0:
        print(f"  frame {i}/{num_steps}")

writer.finish()
print(f"Saved to {predictions_dir}/evolution.gif")
print(f"MAE vf final: {np.abs(pred_vf.flatten() - real_vf.flatten()).mean():.4f}")
print(f"MAE vy final: {np.abs(pred_vy.flatten() - real_vy.flatten()).mean():.4f}")
