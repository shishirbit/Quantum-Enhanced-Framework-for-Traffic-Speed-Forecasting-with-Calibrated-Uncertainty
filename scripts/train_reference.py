"""Train the QUARTS reference backbone and one residual calibrator.

Published SOTA baselines must use their verified official implementations under
the frozen protocol; this script covers the proposed model and matched controls.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from quarts.data import load_timeseries, split_and_window, synthetic_traffic
from quarts.metrics import interval_metrics, laplace_nll, mae, mape, rmse, wape
from quarts.models import (
    FourierResidualCalibrator,
    GraphTemporalBackbone,
    HybridForecaster,
    MLPResidualCalibrator,
    QuantumResidualCalibrator,
    parameter_count,
)
from quarts.reproducibility import save_manifest, set_seed
from quarts.training import make_loader, predict, save_predictions, train_epoch


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--data", help="HDF5/NPZ [time,node,feature]; omit for synthetic")
    parser.add_argument("--adjacency", help="NPY numeric adjacency; identity if omitted")
    parser.add_argument("--calibrator", choices=["quantum", "separable", "fourier", "mlp", "none"])
    parser.add_argument("--run-dir", default="artifacts/runs/reference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--qubits", type=int)
    parser.add_argument("--depth", type=int)
    return parser.parse_args()


def normalized_adjacency(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32) + np.eye(len(matrix), dtype=np.float32)
    inverse = np.maximum(matrix.sum(axis=1), 1e-6) ** -0.5
    return inverse[:, None] * matrix * inverse[None, :]


def load_adjacency(path: str | None, nodes: int) -> np.ndarray:
    if not path:
        return np.eye(nodes, dtype=np.float32)
    source = Path(path)
    if source.suffix.lower() == ".npy":
        return np.load(source).astype(np.float32)
    if source.suffix.lower() in {".pkl", ".pickle"}:
        with source.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        matrix = payload[-1] if isinstance(payload, (tuple, list)) else payload
        return np.asarray(matrix, dtype=np.float32)
    raise ValueError(f"Unsupported adjacency file: {source}")


def build_calibrator(name, hidden, horizons, qubits, depth):
    if name == "quantum":
        return QuantumResidualCalibrator(hidden, horizons, qubits, depth, entangled=True)
    if name == "separable":
        return QuantumResidualCalibrator(hidden, horizons, qubits, depth, entangled=False)
    if name == "fourier":
        return FourierResidualCalibrator(hidden, horizons, qubits, depth)
    if name == "mlp":
        return MLPResidualCalibrator(hidden, horizons, width=2 * qubits)
    return None


def main():
    args = arguments()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    config["seed"] = seed
    set_seed(seed)
    if args.data:
        data = load_timeseries(args.data)
        adjacency = load_adjacency(args.adjacency, data.shape[1])
    else:
        data, adjacency = synthetic_traffic(
            steps=480,
            nodes=int(config["dataset"]["nodes"]),
            features=int(config["dataset"]["features"]),
            seed=seed,
        )
    input_steps = int(config["dataset"]["input_steps"])
    horizons = int(config["dataset"]["horizons"])
    (train, valid, test), standardizer = split_and_window(data, input_steps, horizons)
    calibrator_name = args.calibrator or config["model"]["calibrator"]
    hidden = int(config["model"]["hidden_dim"])
    qubits = int(args.qubits if args.qubits is not None else config["model"]["qubits"])
    depth = int(args.depth if args.depth is not None else config["model"]["circuit_depth"])
    config["model"]["qubits"] = qubits
    config["model"]["circuit_depth"] = depth
    model = HybridForecaster(
        GraphTemporalBackbone(data.shape[-1], hidden, horizons),
        build_calibrator(calibrator_name, hidden, horizons, qubits, depth),
    )
    device = torch.device(args.device)
    model.to(device)
    adjacency_tensor = torch.from_numpy(normalized_adjacency(adjacency)).to(device)
    batch_size = int(config["training"]["batch_size"])
    train_loader = make_loader(train, batch_size, True, seed)
    valid_loader = make_loader(valid, batch_size, False, seed)
    test_loader = make_loader(test, batch_size, False, seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    history, best, stale = [], float("inf"), 0
    patience = int(config["training"].get("patience", 10**9))
    min_delta = float(config["training"].get("min_delta", 0.0))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    epochs = int(args.epochs if args.epochs is not None else config["training"]["epochs"])
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, adjacency_tensor, optimizer, device)
        validation = predict(model, valid_loader, adjacency_tensor, device)
        valid_nll = laplace_nll(
            validation["target"], validation["location"], validation["scale"], validation["mask"]
        )
        history.append({"epoch": epoch, "train_nll": train_loss, "valid_nll": valid_nll})
        print(json.dumps(history[-1]), flush=True)
        if valid_nll < best - min_delta:
            best = valid_nll
            stale = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    prediction = predict(model, test_loader, adjacency_tensor, device)
    target_mean = float(standardizer.mean[..., 0].item())
    target_std = float(standardizer.std[..., 0].item())
    target = prediction["target"] * target_std + target_mean
    location = prediction["location"] * target_std + target_mean
    scale = prediction["scale"] * target_std
    mask = prediction["mask"]
    horizon_steps = {"15min": 2, "30min": 5, "60min": 11}
    horizon_metrics = {}
    for label, index in horizon_steps.items():
        horizon_metrics[label] = {
            "MAE": mae(target[:, index], location[:, index], mask[:, index]),
            "RMSE": rmse(target[:, index], location[:, index], mask[:, index]),
            "MAPE": mape(target[:, index], location[:, index], mask[:, index]),
        }
    report = {
        "calibrator": calibrator_name,
        "parameters": parameter_count(model),
        "MAE": mae(target, location, mask),
        "RMSE": rmse(target, location, mask),
        "MAPE": mape(target, location, mask),
        "WAPE": wape(target, location, mask),
        "Laplace_NLL": laplace_nll(target, location, scale, mask),
        **{
            f"interval90_{key}": value
            for key, value in interval_metrics(
                target,
                location - np.log(10.0) * scale,
                location + np.log(10.0) * scale,
                mask,
                alpha=0.1,
            ).items()
        },
        "inference_seconds": prediction["elapsed_seconds"],
        "total_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "horizons": horizon_metrics,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_predictions(
        run_dir / "predictions.npz",
        {"target": target, "location": location, "scale": scale, "mask": mask},
    )
    source_files = [args.data] if args.data else []
    if args.adjacency:
        source_files.append(args.adjacency)
    save_manifest(
        run_dir / "manifest.json",
        config | {"selected_calibrator": calibrator_name},
        source_files,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
