"""Run RAQ-STAEformer on the frozen QUARTS-RAC validation cache.

RAQ predicts separate lower and upper radii for central intervals. The
canonical METR-LA test split is not loaded or evaluated.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from run_quarts_rac import ALPHAS, load_archive, make_dataset
from quarts.metrics import interval_metrics, mae, weighted_interval_score
from quarts.models import (
    AsymmetricRegimeIntervalCalibrator,
    FourierRegimeGate,
    MLPRegimeGate,
    RegimeIntervalCalibrator,
    masked_asymmetric_weighted_interval_score,
    masked_weighted_interval_score,
    parameter_count,
)
from quarts.reproducibility import set_seed


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-seeds", nargs="+", type=int, default=[42, 73, 202])
    parser.add_argument(
        "--calibrators",
        nargs="+",
        default=["constant_asym", "fourier_sym", "mlp_asym", "fourier_asym"],
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-train-instances", type=int, default=200000)
    parser.add_argument("--cache", default="artifacts/cache/quarts_rac")
    parser.add_argument("--output", default="artifacts/runs/exploratory_raq")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_head(name: str, hidden: int, horizons: int):
    if name == "constant_asym":
        return AsymmetricRegimeIntervalCalibrator(
            None, horizons, levels=len(ALPHAS), experts=1
        )
    if name == "fourier_sym":
        return RegimeIntervalCalibrator(
            FourierRegimeGate(hidden, experts=3),
            horizons,
            levels=len(ALPHAS),
            experts=3,
        )
    if name == "mlp_asym":
        return AsymmetricRegimeIntervalCalibrator(
            MLPRegimeGate(hidden, experts=3),
            horizons,
            levels=len(ALPHAS),
            experts=3,
        )
    if name == "fourier_asym":
        return AsymmetricRegimeIntervalCalibrator(
            FourierRegimeGate(hidden, experts=3),
            horizons,
            levels=len(ALPHAS),
            experts=3,
        )
    raise ValueError(f"Unknown RAQ calibrator: {name}")


def empirical_radii(dataset: TensorDataset, asymmetric: bool):
    _, base, target, mask = dataset.tensors
    base, target, mask = base.numpy(), target.numpy(), mask.numpy()
    horizons = base.shape[1]
    if asymmetric:
        lower_error = np.maximum(base - target, 0.0)
        upper_error = np.maximum(target - base, 0.0)
        output = np.empty((horizons, len(ALPHAS), 2), dtype=np.float32)
        for h in range(horizons):
            valid = mask[:, h]
            for level, alpha in enumerate(ALPHAS):
                output[h, level, 0] = np.quantile(
                    lower_error[valid, h], 1 - alpha, method="higher"
                )
                output[h, level, 1] = np.quantile(
                    upper_error[valid, h], 1 - alpha, method="higher"
                )
        return np.maximum(output, 1e-3)
    absolute = np.abs(target - base)
    output = np.empty((horizons, len(ALPHAS)), dtype=np.float32)
    for h in range(horizons):
        valid = mask[:, h]
        for level, alpha in enumerate(ALPHAS):
            output[h, level] = np.quantile(
                absolute[valid, h], 1 - alpha, method="higher"
            )
    return np.maximum(output, 1e-3)


def initialize_head(head, radii: np.ndarray, asymmetric: bool) -> None:
    if asymmetric:
        increments = np.empty_like(radii)
        increments[:, :-1, :] = radii[:, :-1, :] - radii[:, 1:, :]
        increments[:, -1, :] = radii[:, -1, :]
        scales = np.linspace(0.75, 1.30, head.experts, dtype=np.float32)
        values = scales[:, None, None, None] * increments[None]
    else:
        increments = np.empty_like(radii)
        increments[:, :-1] = radii[:, :-1] - radii[:, 1:]
        increments[:, -1] = radii[:, -1]
        scales = np.linspace(0.75, 1.30, head.experts, dtype=np.float32)
        values = scales[:, None, None] * increments[None]
    values = np.maximum(values - 1e-4, 1e-4)
    with torch.no_grad():
        head.raw_increments.copy_(torch.from_numpy(np.log(np.expm1(values))))


def is_asymmetric(name: str) -> bool:
    return name.endswith("_asym")


@torch.no_grad()
def predict(head, dataset, batch_size: int, device, asymmetric: bool):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total, count = 0.0, 0
    lower_all, upper_all, weights_all = [], [], []
    head.eval()
    for feature, base, target, mask in loader:
        feature = feature.to(device)
        base, target, mask = base.to(device), target.to(device), mask.to(device)
        outputs = head(feature[:, None, :])
        if asymmetric:
            lower, upper, weights = outputs
            loss = masked_asymmetric_weighted_interval_score(
                target, base, lower.squeeze(2), upper.squeeze(2), mask, ALPHAS
            )
            lower_all.append(lower.squeeze(2).cpu().numpy())
            upper_all.append(upper.squeeze(2).cpu().numpy())
        else:
            radii, weights = outputs
            loss = masked_weighted_interval_score(
                target, base, radii.squeeze(2), mask, ALPHAS
            )
            radii = radii.squeeze(2)
            lower_all.append(radii.cpu().numpy())
            upper_all.append(radii.cpu().numpy())
        valid = int(mask.sum())
        total += float(loss) * valid
        count += valid
        weights_all.append(weights.squeeze(1).cpu().numpy())
    return (
        total / max(count, 1),
        np.concatenate(lower_all),
        np.concatenate(upper_all),
        np.concatenate(weights_all),
    )


def conformal_factors(dataset, lower, upper):
    _, base, target, mask = (tensor.numpy() for tensor in dataset.tensors)
    horizons = base.shape[1]
    factors = np.ones((horizons, len(ALPHAS)), dtype=np.float32)
    residual = target - base
    for h in range(horizons):
        valid = mask[:, h]
        for level, alpha in enumerate(ALPHAS):
            scores = np.where(
                residual[valid, h] < 0,
                -residual[valid, h] / np.maximum(lower[valid, h, level], 1e-6),
                residual[valid, h] / np.maximum(upper[valid, h, level], 1e-6),
            )
            quantile = min(
                1.0,
                math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores),
            )
            factors[h, level] = np.quantile(scores, quantile, method="higher")
    return factors


def evaluate(dataset, lower, upper, factors, mean, std):
    _, base, target, mask = (tensor.numpy() for tensor in dataset.tensors)
    location = base * std + mean
    observed = target * std + mean
    lower = lower * factors[None] * std
    upper = upper * factors[None] * std
    intervals = {}
    details = {}
    for level, alpha in enumerate(ALPHAS):
        bounds = (location - lower[..., level], location + upper[..., level])
        intervals[alpha] = bounds
        details[str(alpha)] = interval_metrics(
            observed, *bounds, mask, alpha=alpha
        )
    return {
        "MAE": mae(observed, location, mask),
        "conformal_WIS": weighted_interval_score(observed, location, intervals, mask),
        "interval90_coverage": details["0.1"]["coverage"],
        "interval90_mean_width": details["0.1"]["mean_width"],
        "interval90_interval_score": details["0.1"]["interval_score"],
        "interval90_calibration_error": abs(details["0.1"]["coverage"] - 0.9),
        "intervals": details,
    }


def train_one(name, seed, train_set, selection_set, calibration_set, evaluation_set, mean, std, args):
    set_seed(seed)
    device = torch.device(args.device)
    asymmetric = is_asymmetric(name)
    hidden, horizons = train_set.tensors[0].shape[-1], train_set.tensors[1].shape[-1]
    head = build_head(name, hidden, horizons).to(device)
    initialize_head(head, empirical_radii(train_set, asymmetric), asymmetric)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state, best_selection, stale = copy.deepcopy(head.state_dict()), math.inf, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train()
        total, count = 0.0, 0
        for feature, base, target, mask in loader:
            feature = feature.to(device)
            base, target, mask = base.to(device), target.to(device), mask.to(device)
            output = head(feature[:, None, :])
            if asymmetric:
                lower, upper, _ = output
                loss = masked_asymmetric_weighted_interval_score(
                    target, base, lower.squeeze(2), upper.squeeze(2), mask, ALPHAS
                )
            else:
                radii, _ = output
                loss = masked_weighted_interval_score(
                    target, base, radii.squeeze(2), mask, ALPHAS
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            valid = int(mask.sum())
            total += float(loss.detach()) * valid
            count += valid
        selection_wis, _, _, _ = predict(
            head, selection_set, args.batch_size, device, asymmetric
        )
        record = {
            "epoch": epoch,
            "train_WIS_normalized": total / max(count, 1),
            "selection_WIS_normalized": selection_wis,
        }
        history.append(record)
        print(f"raq seed={seed} {name} {json.dumps(record)}", flush=True)
        if selection_wis < best_selection - 1e-5:
            best_selection = selection_wis
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    head.load_state_dict(best_state)
    _, cal_lower, cal_upper, _ = predict(
        head, calibration_set, args.batch_size, device, asymmetric
    )
    factors = conformal_factors(calibration_set, cal_lower, cal_upper)
    _, eval_lower, eval_upper, weights = predict(
        head, evaluation_set, args.batch_size, device, asymmetric
    )
    report = evaluate(evaluation_set, eval_lower, eval_upper, factors, mean, std)
    average_weights = weights.mean(axis=0)
    entropy = -np.sum(average_weights * np.log(np.maximum(average_weights, 1e-12)))
    report.update(
        {
            "study": "RAQ-STAEformer validation-only exploratory pilot",
            "dataset": "metr_la",
            "head_seed": seed,
            "model": name,
            "canonical_test_split_used": False,
            "point_forecast_modified": False,
            "calibrator_parameters": parameter_count(head),
            "conformal_factors": factors.tolist(),
            "mean_expert_weights": average_weights.tolist(),
            "effective_experts": float(np.exp(entropy)),
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    run_dir = ROOT / args.output / "metr_la" / name / f"head_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), run_dir / "best.pt")
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("model", "head_seed", "conformal_WIS", "interval90_coverage")}, indent=2), flush=True)
    return report


def main() -> None:
    args = arguments()
    set_seed(42)
    cache = ROOT / args.cache / "metr_la_seed_42" / "tw_2000_vw_1000_ld_8"
    train_archive = load_archive(cache / "train.npz")
    valid_archive = load_archive(cache / "valid.npz")
    feature_mean = train_archive["features"].mean(axis=(0, 1), keepdims=True)
    feature_std = np.maximum(train_archive["features"].std(axis=(0, 1), keepdims=True), 1e-4)
    train_set = make_dataset(train_archive, feature_mean, feature_std, max_instances=args.max_train_instances, sample_seed=42)
    valid_windows = valid_archive["features"].shape[0]
    early_end, calibration_end = max(1, int(valid_windows * 0.4)), max(2, int(valid_windows * 0.7))
    selection_set = make_dataset(valid_archive, feature_mean, feature_std, slice(0, early_end))
    calibration_set = make_dataset(valid_archive, feature_mean, feature_std, slice(early_end, calibration_end))
    evaluation_set = make_dataset(valid_archive, feature_mean, feature_std, slice(calibration_end, None))
    reports = {name: [] for name in args.calibrators}
    for seed in args.head_seeds:
        for name in args.calibrators:
            reports[name].append(
                train_one(
                    name,
                    seed,
                    train_set,
                    selection_set,
                    calibration_set,
                    evaluation_set,
                    float(train_archive["mean"]),
                    float(train_archive["std"]),
                    args,
                )
            )
    models = []
    for name, items in reports.items():
        row = {"model": name, "runs": len(items)}
        for metric in ("conformal_WIS", "interval90_coverage", "interval90_mean_width", "interval90_interval_score", "interval90_calibration_error", "effective_experts"):
            values = np.asarray([item[metric] for item in items])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        models.append(row)
    summary = {
        "study": "RAQ-STAEformer validation-only exploratory pilot",
        "complete": True,
        "canonical_test_split_used": False,
        "validation_split": {"early_stopping": 0.4, "conformal_calibration": 0.3, "held_out_evaluation": 0.3},
        "models": models,
    }
    output = ROOT / args.output / "metr_la" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
