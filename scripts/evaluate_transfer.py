"""Zero-shot cross-network evaluation of completed METR-LA reference runs.

The learned weights are held fixed. PEMS-BAY is normalized with statistics from
its training split and evaluated on its canonical test split and graph. This
tests whether a learned local forecasting/calibration rule transfers to a new
sensor network without fine-tuning; it is not a claim of raw-unit transfer.
"""
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
from quarts.training import make_loader, predict
from train_reference import build_calibrator, load_adjacency, normalized_adjacency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_experiment.yaml")
    parser.add_argument("--source-runs", default="artifacts/runs/confirmatory/metr_la")
    parser.add_argument("--target-data", default="data/raw/dcrnn/pems-bay.h5")
    parser.add_argument("--target-adjacency", default="data/raw/dcrnn/adj_mx_bay.pkl")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    data = load_timeseries(ROOT / args.target_data)
    (_, _, test), scaler = split_and_window(
        data, config["dataset"]["input_steps"], config["dataset"]["horizons"]
    )
    loader = make_loader(test, config["training"]["batch_size"], False, 0)
    device = torch.device(args.device)
    adjacency = torch.from_numpy(
        normalized_adjacency(load_adjacency(ROOT / args.target_adjacency, data.shape[1]))
    ).to(device)
    mean = float(scaler.mean[..., 0].item())
    std = float(scaler.std[..., 0].item())

    source = ROOT / args.source_runs
    for checkpoint in sorted(source.glob("*/seed_*/best.pt")):
        run_dir = checkpoint.parent
        output = run_dir / "zero_shot_pems_bay.json"
        if output.exists() or not (run_dir / "metrics.json").exists():
            continue
        calibrator_name = run_dir.parent.name
        model = HybridForecaster(
            GraphTemporalBackbone(data.shape[-1], config["model"]["hidden_dim"], config["dataset"]["horizons"]),
            build_calibrator(
                calibrator_name,
                config["model"]["hidden_dim"],
                config["dataset"]["horizons"],
                config["model"]["qubits"],
                config["model"]["circuit_depth"],
            ),
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        prediction = predict(model, loader, adjacency, device)
        target = prediction["target"] * std + mean
        location = prediction["location"] * std + mean
        scale = prediction["scale"] * std
        mask = prediction["mask"]
        report = {
            "source_dataset": "metr_la",
            "target_dataset": "pems_bay",
            "adaptation": "none",
            "normalization": "target-training-split statistics",
            "MAE": mae(target, location, mask),
            "RMSE": rmse(target, location, mask),
            "MAPE": mape(target, location, mask),
            "WAPE": wape(target, location, mask),
            "Laplace_NLL": laplace_nll(target, location, scale, mask),
            "inference_seconds": prediction["elapsed_seconds"],
            "target_test_windows": int(len(test.x)),
        }
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"DONE {calibrator_name}/{run_dir.name}: {json.dumps(report)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
