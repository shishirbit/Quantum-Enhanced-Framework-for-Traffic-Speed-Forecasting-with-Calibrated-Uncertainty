"""Ideal, channel-noise, and finite-shot evaluation on a fixed test subset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import yaml

from quarts.data import SplitData, load_timeseries, split_and_window
from quarts.metrics import laplace_nll, mae, rmse
from quarts.models import GraphTemporalBackbone, HybridForecaster, QuantumResidualCalibrator
from quarts.training import make_loader
from train_reference import load_adjacency, normalized_adjacency


@torch.no_grad()
def evaluate(model, loader, adjacency, mean, std, device, dtype=torch.float32):
    ys, mus, scales, masks = [], [], [], []
    for x, y, mask in loader:
        output = model(x.to(device=device, dtype=dtype), adjacency)
        ys.append(y.numpy() * std + mean); mus.append(output.location.cpu().numpy() * std + mean)
        scales.append(output.scale.cpu().numpy() * std); masks.append(mask.numpy())
    y, mu, scale, mask = map(np.concatenate, (ys, mus, scales, masks))
    return {"MAE": mae(y, mu, mask), "RMSE": rmse(y, mu, mask), "Laplace_NLL": laplace_nll(y, mu, scale, mask)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_experiment.yaml")
    parser.add_argument("--runs", default="artifacts/runs/confirmatory/metr_la/quantum")
    parser.add_argument("--subset", type=int, default=512)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    data = load_timeseries(ROOT / "data/raw/dcrnn/metr-la.h5")
    (_, _, test), scaler = split_and_window(data, 12, 12)
    indices = np.linspace(0, len(test.x) - 1, min(args.subset, len(test.x)), dtype=int)
    subset = SplitData(test.x[indices], test.y[indices], test.x_mask[indices], test.y_mask[indices])
    loader = make_loader(subset, 32, False, 0)
    shot_loader = make_loader(subset, 1, False, 0)
    adjacency_array = normalized_adjacency(load_adjacency(ROOT / "data/raw/dcrnn/adj_mx.pkl", 207))
    mean, std = float(scaler.mean[..., 0].item()), float(scaler.std[..., 0].item())
    scenarios = [("ideal", {})]
    scenarios += [(f"depolarizing_{p}", {"depolarizing": p}) for p in config["quantum"]["depolarizing"]]
    scenarios += [(f"amplitude_damping_{p}", {"amplitude_damping": p}) for p in config["quantum"]["amplitude_damping"]]
    scenarios += [(f"shots_{shots}", {"shots": shots}) for shots in config["quantum"]["shots"]]
    for run_dir in (ROOT / args.runs).glob("seed_*"):
        if not (run_dir / "best.pt").exists():
            continue
        output_path = run_dir / "quantum_noise.json"
        results = {}
        if output_path.exists():
            results = json.loads(output_path.read_text(encoding="utf-8")).get("results", {})
        for name, options in scenarios:
            if name in results:
                print(run_dir.name, name, "SKIP completed", flush=True)
                continue
            finite_shots = "shots" in options
            device = torch.device("cpu" if finite_shots else "cuda")
            dtype = torch.float64 if finite_shots else torch.float32
            calibrator = QuantumResidualCalibrator(
                config["model"]["hidden_dim"], 12, config["model"]["qubits"],
                config["model"]["circuit_depth"], **options
            )
            model = HybridForecaster(
                GraphTemporalBackbone(1, config["model"]["hidden_dim"], 12), calibrator
            ).to(device=device, dtype=dtype)
            model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
            adjacency = torch.from_numpy(adjacency_array).to(device=device, dtype=dtype)
            results[name] = evaluate(
                model, shot_loader if finite_shots else loader, adjacency, mean, std, device, dtype
            )
            print(run_dir.name, name, results[name], flush=True)
            output_path.write_text(
                json.dumps({"subset": indices.tolist(), "results": results}, indent=2), encoding="utf-8"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
