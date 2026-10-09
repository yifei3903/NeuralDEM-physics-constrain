"""快速测试 — 数据加载 + forward + 统计"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from collections import Counter
from config import DataConfig, ModelConfig
from dataset import CFDDEMDataset

dc = DataConfig()
mc = ModelConfig()
ds = CFDDEMDataset(dc, mc)
print(f"Total samples: {len(ds)}")

# 统计气速分布
ic = Counter()
for i in range(0, len(ds), max(1, len(ds) // 20)):
    ic[float(ds[i]["inlet_velocity"])] += 1
print(f"Inlet distribution: {dict(ic)}")

last = ds[-1]
print(f"Last sample: inlet={last['inlet_velocity'].item():.1f}, t0={last['t0']}, t1={last['t1']}")
print(f"Voidfraction min: {ds._voidfraction_min:.4f}")
print("[OK] Data ready")

