"""Re-evaluate saved small-overfit checkpoints with diagnostic correlations."""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_SOURCES, DataConfig, ModelConfig, TrainConfig
from dataset import CFDDEMDataset
from inference import build_model
from small_overfit import evaluate


def main():
    data_cfg = DataConfig()
    model_cfg = ModelConfig(model_scale="small")
    train_cfg = TrainConfig(debug_overfit=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CFDDEMDataset(
        data_cfg, model_cfg,
        sources=[DATA_SOURCES[train_cfg.debug_source_index]])
    samples = [dataset[index] for index in range(train_cfg.debug_num_pairs)]
    runs = (("baseline", 0.0), ("physics_1e-04", 1e-4),
            ("physics_1e-03", 1e-3), ("physics_1e-02", 1e-2),
            ("physics_1e-01", 1e-1))
    output = []
    for name, lambda_mass in runs:
        checkpoint = torch.load(
            os.path.join(train_cfg.debug_output_dir, f"{name}_final.pt"),
            map_location=device, weights_only=True)
        model = build_model(model_cfg, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        metrics = evaluate(model, samples, device, data_cfg, train_cfg, lambda_mass)
        output.append({"model": name, "lambda_mass": lambda_mass, **metrics})
        print(name, "R_rms=", metrics["residual_rms"],
              "corr(A,B)=", metrics["temporal_divergence_correlation"],
              "cos(A,B)=", metrics["temporal_divergence_cosine"],
              "cancellation_ratio=", metrics["residual_cancellation_ratio"])
    path = os.path.join(train_cfg.debug_output_dir, "final_diagnostics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
