"""QUARTS-UQ: uncertainty-only calibration of a frozen STAEformer.

The point forecast is never modified. Each learned head predicts conditional
Laplace scale from the same cached local features. Central prediction intervals
are recalibrated on validation data with finite-sample split conformal factors.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quarts.metrics import (
    interval_metrics,
    laplace_crps,
    laplace_nll,
    mae,
    mape,
    rmse,
    wape,
    weighted_interval_score,
)
from quarts.models import (
    FourierUncertaintyCalibrator,
    MLPUncertaintyCalibrator,
    QuantumUncertaintyCalibrator,
    parameter_count,
)
from quarts.reproducibility import save_manifest, set_seed


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["metr_la", "pems_bay"], default="metr_la")
    parser.add_argument("--backbone-seed", type=int, default=42)
    parser.add_argument("--head-seeds", nargs="+", type=int, default=[42, 73, 202])
    parser.add_argument(
        "--calibrators",
        nargs="+",
        default=["constant", "mlp", "fourier", "separable", "quantum"],
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-windows", type=int, default=2000)
    parser.add_argument("--max-valid-windows", type=int, default=1000)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--max-train-instances", type=int, default=200000)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.5])
    parser.add_argument("--cache", default="artifacts/cache/quarts_v2")
    parser.add_argument("--output", default="artifacts/runs/exploratory_uq")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def cache_directory(args) -> Path:
    signature = (
        f"tw_{args.max_train_windows}_vw_{args.max_valid_windows}_"
        f"xw_{args.max_test_windows}"
    )
    return (
        ROOT
        / args.cache
        / f"{args.dataset}_seed_{args.backbone_seed}"
        / signature
    )


def flatten_archive(path: Path, max_instances: int = 0, sample_seed: int = 42):
    with np.load(path) as archive:
        features = archive["features"].reshape(-1, archive["features"].shape[-1])
        base = archive["base"].transpose(0, 2, 1).reshape(-1, archive["base"].shape[1])
        target = archive["target"].transpose(0, 2, 1).reshape(-1, archive["target"].shape[1])
        mask = archive["mask"].transpose(0, 2, 1).reshape(-1, archive["mask"].shape[1])
        mean, std = float(archive["mean"].item()), float(archive["std"].item())
    if max_instances and len(features) > max_instances:
        rng = np.random.default_rng(sample_seed)
        selected = np.sort(rng.choice(len(features), max_instances, replace=False))
        features, base, target, mask = (
            features[selected],
            base[selected],
            target[selected],
            mask[selected],
        )
    return (
        TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(base),
            torch.from_numpy(target),
            torch.from_numpy(mask),
        ),
        mean,
        std,
    )


def build_head(name: str, hidden: int, horizons: int):
    if name == "mlp":
        return MLPUncertaintyCalibrator(hidden, horizons, width=5)
    if name == "fourier":
        return FourierUncertaintyCalibrator(hidden, horizons, width=3, depth=2)
    if name == "separable":
        return QuantumUncertaintyCalibrator(
            hidden,
            horizons,
            qubits=4,
            depth=2,
            entangled=False,
        )
    if name == "quantum":
        return QuantumUncertaintyCalibrator(
            hidden,
            horizons,
            qubits=4,
            depth=2,
            entangled=True,
        )
    if name == "constant":
        return None
    raise ValueError(f"Unknown uncertainty calibrator: {name}")


def constant_scale(dataset: TensorDataset) -> torch.Tensor:
    _, base, target, mask = dataset.tensors
    weight = mask.to(torch.float32)
    scale = (torch.abs(target.float() - base.float()) * weight).sum(0)
    scale /= weight.sum(0).clamp_min(1)
    return scale.clamp_min(1e-4)


def masked_nll(base, target, scale, mask):
    weight = mask.to(scale.dtype)
    loss = torch.log(2 * scale.clamp_min(1e-6))
    loss += torch.abs(target - base) / scale.clamp_min(1e-6)
    return (loss * weight).sum() / weight.sum().clamp_min(1)


@torch.no_grad()
def predict_scale(head, loader, device, fixed_scale=None):
    if head is not None:
        head.eval()
    scales = []
    total_nll, count = 0.0, 0.0
    for feature, base, target, mask in loader:
        base = base.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        mask = mask.to(device)
        if head is None:
            scale = fixed_scale.to(device)[None, :].expand_as(base)
        else:
            feature = feature.to(device=device, dtype=torch.float32)
            scale = head(feature[:, None, :]).squeeze(-1)
        weight = mask.to(scale.dtype)
        nll = (
            torch.log(2 * scale.clamp_min(1e-6))
            + torch.abs(target - base) / scale.clamp_min(1e-6)
        )
        total_nll += float((nll * weight).sum())
        count += float(weight.sum())
        scales.append(scale.cpu().numpy())
    return total_nll / count, np.concatenate(scales)


def conformal_factors(target, location, scale, mask, alpha):
    factors = []
    score = np.abs(target - location) / np.maximum(scale, 1e-6)
    for horizon in range(target.shape[1]):
        values = score[mask[:, horizon].astype(bool), horizon]
        rank = min(len(values), math.ceil((len(values) + 1) * (1 - alpha)))
        factors.append(float(np.partition(values, rank - 1)[rank - 1]))
    return np.asarray(factors)


def coverage_by_horizon(target, lower, upper, mask):
    values = []
    for horizon in range(target.shape[1]):
        valid = mask[:, horizon].astype(bool)
        covered = (
            (target[valid, horizon] >= lower[valid, horizon])
            & (target[valid, horizon] <= upper[valid, horizon])
        )
        values.append(float(covered.mean()))
    return values


def evaluate_report(
    name,
    head_seed,
    head,
    fixed_scale,
    valid_set,
    test_set,
    mean,
    std,
    args,
    run_dir,
    history,
    elapsed,
):
    device = torch.device(args.device)
    valid_loader = DataLoader(valid_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)
    _, valid_scale = predict_scale(head, valid_loader, device, fixed_scale)
    normalized_nll, test_scale = predict_scale(head, test_loader, device, fixed_scale)

    _, valid_base, valid_target, valid_mask = (
        tensor.numpy() for tensor in valid_set.tensors
    )
    _, test_base, test_target, test_mask = (
        tensor.numpy() for tensor in test_set.tensors
    )
    valid_base = valid_base.astype(np.float32)
    valid_target = valid_target.astype(np.float32)
    valid_mask = valid_mask.astype(bool)
    test_base = test_base.astype(np.float32)
    test_target = test_target.astype(np.float32)
    test_mask = test_mask.astype(bool)
    target = test_target * std + mean
    location = test_base * std + mean
    physical_scale = test_scale * std

    intervals = {}
    factor_report = {}
    interval_report = {}
    for alpha in sorted(set(args.alphas)):
        factors = conformal_factors(
            valid_target,
            valid_base,
            valid_scale,
            valid_mask,
            alpha,
        )
        radius = physical_scale * factors[None, :]
        lower, upper = location - radius, location + radius
        intervals[alpha] = (lower, upper)
        result = interval_metrics(target, lower, upper, test_mask, alpha=alpha)
        result["coverage_by_horizon"] = coverage_by_horizon(
            target,
            lower,
            upper,
            test_mask,
        )
        factor_report[str(alpha)] = factors.tolist()
        interval_report[str(alpha)] = result

    primary = interval_report["0.1"]
    report = {
        "study": "QUARTS-UQ exploratory",
        "dataset": args.dataset,
        "backbone_seed": args.backbone_seed,
        "head_seed": head_seed,
        "model": name,
        "point_forecast_modified": False,
        "calibrator_parameters": parameter_count(head) if head is not None else 0,
        "quantum_backend": (
            type(head.quantum).__name__
            if name in {"quantum", "separable"}
            else None
        ),
        "MAE": mae(target, location, test_mask),
        "RMSE": rmse(target, location, test_mask),
        "MAPE": mape(target, location, test_mask),
        "WAPE": wape(target, location, test_mask),
        "Laplace_NLL": laplace_nll(
            target,
            location,
            physical_scale,
            test_mask,
        ),
        "Laplace_CRPS": laplace_crps(
            target,
            location,
            physical_scale,
            test_mask,
        ),
        "conformal_WIS": weighted_interval_score(
            target,
            location,
            intervals,
            test_mask,
        ),
        "interval90_coverage": primary["coverage"],
        "interval90_mean_width": primary["mean_width"],
        "interval90_interval_score": primary["interval_score"],
        "interval90_calibration_error": abs(primary["coverage"] - 0.9),
        "normalized_NLL": normalized_nll,
        "conformal_factors": factor_report,
        "intervals": interval_report,
        "history": history,
        "elapsed_seconds": elapsed,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    if head is not None:
        torch.save(head.state_dict(), run_dir / "best.pt")
    np.savez_compressed(
        run_dir / "test_scale.npz",
        scale=test_scale.astype(np.float16),
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def train_one(name, head_seed, train_set, valid_set, test_set, mean, std, args):
    set_seed(head_seed)
    device = torch.device(args.device)
    hidden, horizons = train_set.tensors[0].shape[-1], train_set.tensors[1].shape[-1]
    head = build_head(name, hidden, horizons)
    fixed_scale = constant_scale(train_set) if head is None else None
    history = []
    started = time.perf_counter()
    if head is not None:
        head.to(device)
        train_loader = DataLoader(
            train_set,
            args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(head_seed),
        )
        valid_loader = DataLoader(valid_set, args.batch_size, shuffle=False)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4,
        )
        best_nll, best_state, stale = float("inf"), copy.deepcopy(head.state_dict()), 0
        for epoch in range(1, args.epochs + 1):
            head.train()
            losses = []
            for feature, base, target, mask in train_loader:
                feature = feature.to(device=device, dtype=torch.float32)
                base = base.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.float32)
                mask = mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                scale = head(feature[:, None, :]).squeeze(-1)
                loss = masked_nll(base, target, scale, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            valid_nll, _ = predict_scale(head, valid_loader, device)
            record = {
                "epoch": epoch,
                "train_NLL_normalized": float(np.mean(losses)),
                "valid_NLL_normalized": valid_nll,
            }
            history.append(record)
            print(
                f"{args.dataset} seed={head_seed} {name} "
                + json.dumps(record),
                flush=True,
            )
            if valid_nll < best_nll - 1e-4:
                best_nll = valid_nll
                best_state = copy.deepcopy(head.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        head.load_state_dict(best_state)

    run_dir = (
        ROOT
        / args.output
        / args.dataset
        / f"backbone_seed_{args.backbone_seed}"
        / name
        / f"head_seed_{head_seed}"
    )
    report = evaluate_report(
        name,
        head_seed,
        head,
        fixed_scale,
        valid_set,
        test_set,
        mean,
        std,
        args,
        run_dir,
        history,
        time.perf_counter() - started,
    )
    checkpoint = (
        ROOT
        / "artifacts"
        / "runs"
        / "confirmatory"
        / args.dataset
        / "staeformer"
        / f"seed_{args.backbone_seed}"
        / "best.pt"
    )
    save_manifest(
        run_dir / "manifest.json",
        {
            "study": "QUARTS-UQ exploratory",
            "dataset": args.dataset,
            "backbone_seed": args.backbone_seed,
            "head_seed": head_seed,
            "calibrator": name,
            "point_forecast_modified": False,
            "cache": str(cache_directory(args)),
            "arguments": vars(args),
        },
        [checkpoint],
    )
    print(json.dumps({key: report[key] for key in (
        "model",
        "head_seed",
        "Laplace_NLL",
        "Laplace_CRPS",
        "conformal_WIS",
        "interval90_coverage",
        "interval90_mean_width",
        "interval90_interval_score",
    )}, indent=2), flush=True)
    return report


def write_summary(reports, args):
    output = (
        ROOT
        / args.output
        / args.dataset
        / f"backbone_seed_{args.backbone_seed}"
    )
    rows = []
    keys = [
        "Laplace_NLL",
        "Laplace_CRPS",
        "conformal_WIS",
        "interval90_coverage",
        "interval90_mean_width",
        "interval90_interval_score",
        "interval90_calibration_error",
    ]
    for name in args.calibrators:
        selected = [report for report in reports if report["model"] == name]
        row = {"model": name, "runs": len(selected)}
        for key in keys:
            values = [report[key] for report in selected]
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "study": "QUARTS-UQ exploratory",
        "dataset": args.dataset,
        "backbone_seed": args.backbone_seed,
        "head_seeds": args.head_seeds,
        "point_forecast_modified": False,
        "models": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main():
    args = arguments()
    cache = cache_directory(args)
    required = [cache / f"{split}.npz" for split in ("train", "valid", "test")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Create the frozen-backbone cache before QUARTS-UQ: " + ", ".join(missing)
        )
    train_set, mean, std = flatten_archive(
        cache / "train.npz",
        args.max_train_instances,
        args.data_seed,
    )
    valid_set, _, _ = flatten_archive(cache / "valid.npz")
    test_set, _, _ = flatten_archive(cache / "test.npz")
    reports = []
    for head_seed in args.head_seeds:
        for name in args.calibrators:
            reports.append(
                train_one(
                    name,
                    head_seed,
                    train_set,
                    valid_set,
                    test_set,
                    mean,
                    std,
                    args,
                )
            )
    write_summary(reports, args)


if __name__ == "__main__":
    main()
