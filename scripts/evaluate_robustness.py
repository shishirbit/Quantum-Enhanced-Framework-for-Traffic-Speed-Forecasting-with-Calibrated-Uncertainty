"""Evaluate completed confirmatory runs under predeclared input corruptions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import yaml

from quarts.data import load_timeseries, split_and_window
from quarts.metrics import laplace_nll, mae, mape, rmse, wape
from quarts.models import GraphTemporalBackbone, HybridForecaster
from quarts.training import make_loader
from train_reference import build_calibrator, load_adjacency, normalized_adjacency

DATASETS = {
    "metr_la": ("data/raw/dcrnn/metr-la.h5", "data/raw/dcrnn/adj_mx.pkl"),
    "pems_bay": ("data/raw/dcrnn/pems-bay.h5", "data/raw/dcrnn/adj_mx_bay.pkl"),
}
REFERENCE_MODELS = {"none", "mlp", "fourier", "separable", "quantum"}


@torch.no_grad()
def evaluate(model, loader, adjacency, mean, std, scenario, seed, device):
    model.eval()
    targets, locations, scales, masks = [], [], [], []
    generator = torch.Generator(device=device).manual_seed(seed)
    for x, y, mask in loader:
        x = x.to(device)
        if scenario.startswith("missing_"):
            rate = float(scenario.split("_")[1])
            observed = torch.rand(x.shape, generator=generator, device=device) >= rate
            x = x * observed
        elif scenario.startswith("noise_"):
            rate = float(scenario.split("_")[1])
            noise = torch.randn(x.shape, generator=generator, device=device) * rate
            x = x + noise
        elif scenario.startswith("outage_"):
            steps = int(scenario.split("_")[1]) // 5
            sensors = max(1, int(x.shape[2] * 0.1))
            selected = torch.randperm(x.shape[2], generator=generator, device=device)[:sensors]
            x[:, -steps:, selected, :] = 0
        output = model(x, adjacency)
        targets.append(y.numpy() * std + mean)
        locations.append(output.location.cpu().numpy() * std + mean)
        scales.append(output.scale.cpu().numpy() * std)
        masks.append(mask.numpy())
    y = np.concatenate(targets)
    location = np.concatenate(locations)
    scale = np.concatenate(scales)
    mask = np.concatenate(masks)
    return {
        "MAE": mae(y, location, mask),
        "RMSE": rmse(y, location, mask),
        "MAPE": mape(y, location, mask),
        "WAPE": wape(y, location, mask),
        "Laplace_NLL": laplace_nll(y, location, scale, mask),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_experiment.yaml")
    parser.add_argument("--runs", default="artifacts/runs/confirmatory")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    scenarios = ["clean"]
    scenarios += [f"missing_{rate}" for rate in config["evaluation"]["missing_rates"]]
    scenarios += [f"noise_{rate}" for rate in config["evaluation"]["noise_std_rates"]]
    scenarios += ["outage_30", "outage_60", "outage_120"]
    for dataset in args.datasets:
        data_path, adjacency_path = DATASETS[dataset]
        data = load_timeseries(ROOT / data_path)
        (_, _, test), standardizer = split_and_window(
            data, config["dataset"]["input_steps"], config["dataset"]["horizons"]
        )
        adjacency = torch.from_numpy(
            normalized_adjacency(load_adjacency(ROOT / adjacency_path, data.shape[1]))
        ).cuda()
        mean = float(standardizer.mean[..., 0].item())
        std = float(standardizer.std[..., 0].item())
        loader = make_loader(test, config["training"]["batch_size"], False, 0)
        base = ROOT / args.runs / dataset
        for metrics_path in base.glob("*/seed_*/metrics.json"):
            run_dir = metrics_path.parent
            output_path = run_dir / "robustness.json"
            if output_path.exists():
                continue
            calibrator = run_dir.parent.name
            if calibrator not in REFERENCE_MODELS:
                continue
            seed = int(run_dir.name.split("_")[-1])
            model = HybridForecaster(
                GraphTemporalBackbone(data.shape[-1], config["model"]["hidden_dim"], config["dataset"]["horizons"]),
                build_calibrator(
                    calibrator,
                    config["model"]["hidden_dim"],
                    config["dataset"]["horizons"],
                    config["model"]["qubits"],
                    config["model"]["circuit_depth"],
                ),
            ).cuda()
            model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cuda", weights_only=True))
            results = {
                scenario: evaluate(model, loader, adjacency, mean, std, scenario, seed + 9001, "cuda")
                for scenario in scenarios
            }
            output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"DONE {dataset}/{calibrator}/{seed}", flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
