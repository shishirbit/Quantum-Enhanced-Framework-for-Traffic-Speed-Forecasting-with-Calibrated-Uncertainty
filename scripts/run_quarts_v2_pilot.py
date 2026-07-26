"""Exploratory QUARTS v2: gated calibration of a frozen STAEformer.

This script never writes into the v1 confirmatory tree. It first caches
validation-safe local context features and frozen STAEformer forecasts, then
trains parameter-matched residual/uncertainty heads. The residual path is
bounded and initialized close to a no-op so an untrained head cannot destroy
the strong base forecast.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "STAEformer"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from lib.data_prepare import get_dataloaders_from_index_data
from model.STAEformer import STAEformer
from quarts.metrics import interval_metrics, laplace_nll, mae, mape, rmse, wape
from quarts.models import (
    FourierResidualCalibrator,
    GatedResidualCalibrator,
    MLPResidualCalibrator,
    QuantumResidualCalibrator,
    parameter_count,
)
from quarts.reproducibility import save_manifest, set_seed


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["METRLA", "PEMSBAY"], default="METRLA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibrators", nargs="+", default=["none", "mlp", "fourier", "separable", "quantum"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--point-weight", type=float, default=0.2)
    parser.add_argument("--max-train-windows", type=int, default=8000)
    parser.add_argument("--max-valid-windows", type=int, default=2000)
    parser.add_argument("--max-test-windows", type=int, default=0, help="0 keeps the complete test split")
    parser.add_argument("--max-train-instances", type=int, default=1_000_000)
    parser.add_argument("--cache", default="artifacts/cache/quarts_v2")
    parser.add_argument("--output", default="artifacts/runs/exploratory_v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def deterministic_subset(loader, limit: int, batch_size: int) -> DataLoader:
    dataset = loader.dataset
    if limit and limit < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, limit, dtype=int).tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def context_features(x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Build sensor-local, network-size-independent calibration features."""
    recent = x[..., 0].permute(0, 2, 1)
    stats = torch.stack(
        [recent[:, :, -1], recent.mean(-1), recent.std(-1), recent[:, :, -1] - recent[:, :, 0]],
        dim=-1,
    )
    features = [base.permute(0, 2, 1), stats]
    if x.shape[-1] > 1:
        tod = 2 * torch.pi * x[:, -1, :, 1]
        features.append(torch.stack([torch.sin(tod), torch.cos(tod)], dim=-1))
    if x.shape[-1] > 2:
        dow = 2 * torch.pi * x[:, -1, :, 2] / 7.0
        features.append(torch.stack([torch.sin(dow), torch.cos(dow)], dim=-1))
    return torch.cat(features, dim=-1)


@torch.no_grad()
def cache_split(model, loader, scaler, device, path: Path):
    if path.exists():
        return
    features, bases, targets, masks = [], [], [], []
    model.eval()
    for batch_index, (x, y) in enumerate(loader, start=1):
        base = model(x.to(device)).squeeze(-1).cpu()
        target = ((y.squeeze(-1) - scaler.mean) / scaler.std).float()
        features.append(context_features(x, base).half().numpy())
        bases.append(base.half().numpy())
        targets.append(target.half().numpy())
        masks.append((y.squeeze(-1) != 0).numpy())
        if batch_index % 10 == 0:
            print(f"CACHE {path.stem}: {batch_index}/{len(loader)} batches", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.concatenate(features),
        base=np.concatenate(bases),
        target=np.concatenate(targets),
        mask=np.concatenate(masks),
        mean=np.asarray(scaler.mean),
        std=np.asarray(scaler.std),
    )


@torch.no_grad()
def cache_split_from_predictions(loader, scaler, predictions_path: Path, path: Path):
    """Reuse frozen-backbone test forecasts from the completed v1 protocol."""
    if path.exists():
        return
    with np.load(predictions_path) as predictions:
        locations = predictions["location"]
        archived_masks = predictions["mask"]
    if len(locations) != len(loader.dataset):
        raise ValueError(
            f"Archived predictions contain {len(locations)} windows, "
            f"but the test loader contains {len(loader.dataset)}"
        )
    features, bases, targets, masks = [], [], [], []
    offset = 0
    for batch_index, (x, y) in enumerate(loader, start=1):
        stop = offset + len(x)
        base = torch.from_numpy(
            ((locations[offset:stop] - scaler.mean) / scaler.std).astype(np.float32)
        )
        target = ((y.squeeze(-1) - scaler.mean) / scaler.std).float()
        features.append(context_features(x, base).half().numpy())
        bases.append(base.half().numpy())
        targets.append(target.half().numpy())
        masks.append(archived_masks[offset:stop].astype(bool))
        offset = stop
        if batch_index % 25 == 0:
            print(f"CACHE test archive: {batch_index}/{len(loader)} batches", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.concatenate(features),
        base=np.concatenate(bases),
        target=np.concatenate(targets),
        mask=np.concatenate(masks),
        mean=np.asarray(scaler.mean),
        std=np.asarray(scaler.std),
    )


def build_head(name: str, hidden: int, horizons: int):
    if name == "quantum":
        core = QuantumResidualCalibrator(hidden, horizons, qubits=4, depth=2, entangled=True)
    elif name == "separable":
        core = QuantumResidualCalibrator(hidden, horizons, qubits=4, depth=2, entangled=False)
    elif name == "fourier":
        core = FourierResidualCalibrator(hidden, horizons, width=3, depth=2)
    elif name == "mlp":
        core = MLPResidualCalibrator(hidden, horizons, width=4)
    else:
        return None
    return GatedResidualCalibrator(core, horizons, residual_limit=2.0, initial_logit=-6.0)


def flatten_archive(path: Path, max_instances: int = 0, seed: int = 0):
    with np.load(path) as archive:
        features = archive["features"].reshape(-1, archive["features"].shape[-1])
        base = archive["base"].transpose(0, 2, 1).reshape(-1, archive["base"].shape[1])
        target = archive["target"].transpose(0, 2, 1).reshape(-1, archive["target"].shape[1])
        mask = archive["mask"].transpose(0, 2, 1).reshape(-1, archive["mask"].shape[1])
        mean, std = float(archive["mean"].item()), float(archive["std"].item())
    if max_instances and len(features) > max_instances:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(features), max_instances, replace=False))
        features, base, target, mask = features[selected], base[selected], target[selected], mask[selected]
    tensors = (
        torch.from_numpy(features),
        torch.from_numpy(base),
        torch.from_numpy(target),
        torch.from_numpy(mask),
    )
    return TensorDataset(*tensors), mean, std


def masked_objective(pred, target, scale, mask, point_weight):
    weight = mask.to(pred.dtype)
    absolute = torch.abs(target - pred)
    nll = torch.log(2 * scale.clamp_min(1e-6)) + absolute / scale.clamp_min(1e-6)
    denominator = weight.sum().clamp_min(1)
    return ((nll + point_weight * absolute) * weight).sum() / denominator


@torch.no_grad()
def evaluate_head(head, loader, device):
    predictions = {key: [] for key in ("base", "target", "mask", "location", "scale")}
    total_abs, total_nll, count = 0.0, 0.0, 0.0
    if head is not None:
        head.eval()
    for feature, base, target, mask in loader:
        feature = feature.to(device=device, dtype=torch.float32)
        base = base.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        mask = mask.to(device)
        if head is None:
            residual = torch.zeros_like(base)
            scale = torch.ones_like(base)
        else:
            residual, scale = head(feature[:, None, :])
            residual, scale = residual.squeeze(-1), scale.squeeze(-1)
        location = base + residual
        weight = mask.to(location.dtype)
        total_abs += float((torch.abs(target - location) * weight).sum())
        total_nll += float(
            ((torch.log(2 * scale.clamp_min(1e-6)) + torch.abs(target - location) / scale.clamp_min(1e-6)) * weight).sum()
        )
        count += float(weight.sum())
        for key, value in {"base": base, "target": target, "mask": mask, "location": location, "scale": scale}.items():
            predictions[key].append(value.cpu().numpy())
    predictions = {key: np.concatenate(value) for key, value in predictions.items()}
    return total_abs / count, total_nll / count, predictions


def validation_residual_mask(prediction) -> np.ndarray:
    """Keep a residual at a horizon only when validation MAE improves."""
    target = prediction["target"]
    mask = prediction["mask"].astype(bool)
    base_error = np.abs(target - prediction["base"])
    calibrated_error = np.abs(target - prediction["location"])
    enabled = []
    for horizon in range(target.shape[1]):
        valid = mask[:, horizon]
        improves = (
            valid.any()
            and calibrated_error[valid, horizon].mean()
            < base_error[valid, horizon].mean() - 1e-6
        )
        enabled.append(improves)
    return np.asarray(enabled, dtype=bool)


def train_head(name, train_set, valid_set, test_set, hidden, horizons, args, mean, std, run_dir):
    device = torch.device(args.device)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    valid_loader = DataLoader(valid_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)
    head = build_head(name, hidden, horizons)
    started = time.perf_counter()
    history = []
    if head is not None:
        head.to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        best_mae, best_state, stale = float("inf"), copy.deepcopy(head.state_dict()), 0
        for epoch in range(1, args.epochs + 1):
            head.train()
            losses = []
            for feature, base, target, mask in train_loader:
                feature = feature.to(device=device, dtype=torch.float32)
                base = base.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.float32)
                mask = mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                residual, scale = head(feature[:, None, :])
                loss = masked_objective(base + residual.squeeze(-1), target, scale.squeeze(-1), mask, args.point_weight)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            valid_mae, valid_nll, _ = evaluate_head(head, valid_loader, device)
            record = {
                "epoch": epoch,
                "train_objective": float(np.mean(losses)),
                "valid_MAE_normalized": valid_mae,
                "valid_NLL_normalized": valid_nll,
                "gate_mean": float(head.gate.mean().detach().cpu()),
            }
            history.append(record)
            print(name, json.dumps(record), flush=True)
            if valid_mae < best_mae - 1e-5:
                best_mae, best_state, stale = valid_mae, copy.deepcopy(head.state_dict()), 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        head.load_state_dict(best_state)
        _, _, valid_prediction = evaluate_head(head, valid_loader, device)
        head.set_residual_mask(validation_residual_mask(valid_prediction))

    _, normalized_nll, prediction = evaluate_head(head, test_loader, device)
    target = prediction["target"] * std + mean
    location = prediction["location"] * std + mean
    base = prediction["base"] * std + mean
    scale = prediction["scale"] * std
    mask = prediction["mask"]
    q = np.log(10.0) * scale
    intervals = interval_metrics(target, location - q, location + q, mask, alpha=0.1)
    report = {
        "model": f"staeformer+{name}" if name != "none" else "staeformer",
        "exploratory": True,
        "seed": args.seed,
        "calibrator_parameters": parameter_count(head) if head is not None else 0,
        "quantum_backend": (
            type(head.calibrator.quantum).__name__
            if name in {"quantum", "separable"}
            else None
        ),
        "MAE": mae(target, location, mask),
        "RMSE": rmse(target, location, mask),
        "MAPE": mape(target, location, mask),
        "WAPE": wape(target, location, mask),
        "Laplace_NLL": laplace_nll(target, location, scale, mask),
        "normalized_NLL": normalized_nll,
        "base_MAE": mae(target, base, mask),
        "interval90_coverage": intervals["coverage"],
        "interval90_mean_width": intervals["mean_width"],
        "interval90_interval_score": intervals["interval_score"],
        "interval90_calibration_error": abs(intervals["coverage"] - 0.9),
        "gate": head.gate.detach().cpu().tolist() if head is not None else [0.0] * horizons,
        "residual_horizons_enabled": (
            head.residual_enabled.detach().cpu().bool().tolist()
            if head is not None
            else [False] * horizons
        ),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    if head is not None:
        torch.save(head.state_dict(), run_dir / "best.pt")
    np.savez_compressed(run_dir / "predictions.npz", target=target, location=location, base=base, scale=scale, mask=mask)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    args = arguments()
    set_seed(args.seed)
    device = torch.device(args.device)
    with (VENDOR / "model" / "STAEformer.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)[args.dataset]
    train, valid, test, scaler = get_dataloaders_from_index_data(
        VENDOR / "data" / args.dataset,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg["batch_size"],
    )
    key = "metr_la" if args.dataset == "METRLA" else "pems_bay"
    checkpoint = ROOT / "artifacts/runs/confirmatory" / key / "staeformer" / f"seed_{args.seed}" / "best.pt"
    model = STAEformer(**cfg["model_args"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    cache_signature = (
        f"tw_{args.max_train_windows}_vw_{args.max_valid_windows}_"
        f"xw_{args.max_test_windows}"
    )
    cache_root = ROOT / args.cache / f"{key}_seed_{args.seed}" / cache_signature
    loaders = {
        "train": deterministic_subset(train, args.max_train_windows, cfg["batch_size"]),
        "valid": deterministic_subset(valid, args.max_valid_windows, cfg["batch_size"]),
        "test": deterministic_subset(test, args.max_test_windows, cfg["batch_size"]),
    }
    archived_test = checkpoint.parent / "predictions.npz"
    for split, loader in loaders.items():
        cache_path = cache_root / f"{split}.npz"
        if split == "test" and args.max_test_windows == 0 and archived_test.exists():
            cache_split_from_predictions(loader, scaler, archived_test, cache_path)
        else:
            cache_split(model, loader, scaler, device, cache_path)
        print(f"CACHE {split} complete", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.cache_only:
        return

    train_set, mean, std = flatten_archive(cache_root / "train.npz", args.max_train_instances, args.seed)
    valid_set, _, _ = flatten_archive(cache_root / "valid.npz")
    test_set, _, _ = flatten_archive(cache_root / "test.npz")
    hidden = train_set.tensors[0].shape[-1]
    reports = {}
    for name in args.calibrators:
        run_dir = ROOT / args.output / key / name / f"seed_{args.seed}"
        reports[name] = train_head(name, train_set, valid_set, test_set, hidden, 12, args, mean, std, run_dir)
        save_manifest(
            run_dir / "manifest.json",
            {
                "study": "QUARTS v2 exploratory pilot",
                "dataset": args.dataset,
                "seed": args.seed,
                "calibrator": name,
                "frozen_backbone": "official STAEformer",
                "cache_limits": {
                    "train_windows": args.max_train_windows,
                    "valid_windows": args.max_valid_windows,
                    "test_windows": args.max_test_windows,
                    "train_instances": args.max_train_instances,
                },
                "arguments": vars(args),
            },
            [checkpoint],
        )
    summary = {
        name: {key: report[key] for key in ("MAE", "RMSE", "MAPE", "WAPE", "Laplace_NLL", "base_MAE", "calibrator_parameters")}
        for name, report in reports.items()
    }
    (ROOT / args.output / key / f"pilot_seed_{args.seed}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
