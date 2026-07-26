"""Validation-only SA-CQR-RAC development on PeMSD8.

The canonical PeMSD8 test split is never materialized or cached by this
runner. Model selection and conformal evaluation use rolling chronological
folds inside the validation timeline.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/STAEformer"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from quarts.metrics import interval_metrics, mae, weighted_interval_score
from quarts.models import (
    AsymmetricRegimeIntervalCalibrator,
    FourierRegimeGate,
    MLPRegimeGate,
    RegimeIntervalCalibrator,
    masked_asymmetric_pinball_loss,
    masked_asymmetric_weighted_interval_score,
    masked_weighted_interval_score,
    parameter_count,
)
from quarts.reproducibility import save_manifest, set_seed
from run_prospective_pemsd4 import cache_pems_split, deterministic_subset, train_backbone
from run_quarts_rac import ALPHAS, enriched_features, load_archive, make_dataset


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone-epochs", type=int, default=12)
    parser.add_argument("--backbone-patience", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=15)
    parser.add_argument("--head-patience", type=int, default=4)
    parser.add_argument("--backbone-batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-windows", type=int, default=4000)
    parser.add_argument("--max-valid-windows", type=int, default=0)
    parser.add_argument("--max-train-instances", type=int, default=200000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="artifacts/runs/sacqr_pemsd8_validation")
    parser.add_argument("--cache", default="artifacts/cache/sacqr_pemsd8_validation")
    return parser.parse_args()


def load_pemsd8_train_valid():
    values = np.load(ROOT / "data/raw/pemsd8/PEMS08.npz")["data"].astype(np.float32)[..., 0]
    time_steps, nodes = values.shape
    train_end = int(time_steps * 0.7)
    valid_end = int(time_steps * 0.8)
    mean = float(values[:train_end].mean())
    std = float(values[:train_end].std())
    scaled = (values - mean) / max(std, 1e-6)
    temporal = np.zeros((valid_end, nodes, 3), dtype=np.float32)
    temporal[..., 0] = scaled[:valid_end]
    temporal[..., 1] = ((np.arange(valid_end) % 288) / 288.0)[:, None]
    temporal[..., 2] = (((np.arange(valid_end) // 288) % 7).astype(np.float32))[:, None]

    def build(start, end):
        xs, ys = [], []
        for index in range(start, end - 23):
            xs.append(temporal[index : index + 12])
            ys.append(scaled[index + 12 : index + 24, :, None])
        return TensorDataset(torch.from_numpy(np.asarray(xs)), torch.from_numpy(np.asarray(ys)))

    return build(0, train_end), build(train_end, valid_end), mean, std, nodes


def dataset_slice(dataset: TensorDataset, start: int, end: int) -> TensorDataset:
    return TensorDataset(*(tensor[start:end] for tensor in dataset.tensors))


def load_adjacency(nodes):
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    with (ROOT / "data/raw/pemsd8/PEMS08.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source, target = int(row["from"]), int(row["to"])
            if source < nodes and target < nodes:
                adjacency[source, target] = 1.0
                adjacency[target, source] = 1.0
    adjacency += np.eye(nodes, dtype=np.float32)
    adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-6)
    return torch.from_numpy(adjacency)


def build_head(name, hidden, horizons):
    if name == "constant":
        return RegimeIntervalCalibrator(None, horizons, len(ALPHAS), experts=1), False
    if name == "mlp_wis":
        return RegimeIntervalCalibrator(MLPRegimeGate(hidden, experts=3), horizons, len(ALPHAS), experts=3), False
    if name == "cqr_mlp":
        gate = MLPRegimeGate(hidden, experts=3)
    elif name == "cqr_fourier":
        gate = FourierRegimeGate(hidden, experts=3)
    else:
        raise ValueError(name)
    return AsymmetricRegimeIntervalCalibrator(gate, horizons, len(ALPHAS), experts=3), True


@torch.no_grad()
def predict(head, dataset, batch_size, device, asymmetric):
    head.eval()
    lower_all, upper_all, weights_all = [], [], []
    total, count = 0.0, 0
    for feature, base, target, mask in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        feature, base, target, mask = feature.to(device), base.to(device), target.to(device), mask.to(device)
        if asymmetric:
            lower, upper, weights = head(feature[:, None, :])
            lower, upper = lower.squeeze(2), upper.squeeze(2)
            loss = masked_asymmetric_weighted_interval_score(target, base, lower, upper, mask, ALPHAS)
        else:
            radius, weights = head(feature[:, None, :])
            lower = upper = radius.squeeze(2)
            loss = masked_weighted_interval_score(target, base, lower, mask, ALPHAS)
        valid = int(mask.sum())
        total += float(loss) * valid
        count += valid
        lower_all.append(lower.cpu().numpy())
        upper_all.append(upper.cpu().numpy())
        weights_all.append(weights.squeeze(1).cpu().numpy())
    return total / max(count, 1), np.concatenate(lower_all), np.concatenate(upper_all), np.concatenate(weights_all)


def cqr_adjustment(dataset, lower_radii, upper_radii):
    _, base, target, mask = (tensor.numpy() for tensor in dataset.tensors)
    adjustment = np.zeros((base.shape[1], len(ALPHAS)), dtype=np.float32)
    for horizon in range(base.shape[1]):
        valid = mask[:, horizon]
        for level, alpha in enumerate(ALPHAS):
            lower = base[valid, horizon] - lower_radii[valid, horizon, level]
            upper = base[valid, horizon] + upper_radii[valid, horizon, level]
            scores = np.maximum(lower - target[valid, horizon], target[valid, horizon] - upper)
            probability = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
            adjustment[horizon, level] = np.quantile(scores, probability, method="higher")
    return adjustment


def physical_asymmetric_metrics(dataset, lower_radii, upper_radii, adjustment, mean, std):
    _, base, target, mask = (tensor.numpy() for tensor in dataset.tensors)
    location = base * std + mean
    observed = target * std + mean
    lower_radii = np.maximum(lower_radii + adjustment[None, :, :], 1e-6) * std
    upper_radii = np.maximum(upper_radii + adjustment[None, :, :], 1e-6) * std
    intervals, detail = {}, {}
    for level, alpha in enumerate(ALPHAS):
        lower = location - lower_radii[..., level]
        upper = location + upper_radii[..., level]
        intervals[alpha] = (lower, upper)
        detail[str(alpha)] = interval_metrics(observed, lower, upper, mask, alpha)
    return {
        "MAE": mae(observed, location, mask),
        "conformal_WIS": weighted_interval_score(observed, location, intervals, mask),
        "interval90_coverage": detail[str(ALPHAS[0])]["coverage"],
        "interval90_mean_width": detail[str(ALPHAS[0])]["mean_width"],
        "intervals": detail,
    }


def train_head(name, train_set, selection_set, folds, mean, std, args):
    set_seed(args.seed)
    device = torch.device(args.device)
    hidden, horizons = train_set.tensors[0].shape[-1], train_set.tensors[1].shape[-1]
    head, asymmetric = build_head(name, hidden, horizons)
    head = head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-3)
    loader = DataLoader(
        train_set,
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_state, best_loss, stale = copy.deepcopy(head.state_dict()), float("inf"), 0
    history = []
    for epoch in range(1, args.head_epochs + 1):
        head.train()
        for feature, base, target, mask in loader:
            feature, base, target, mask = feature.to(device), base.to(device), target.to(device), mask.to(device)
            if asymmetric:
                lower, upper, _ = head(feature[:, None, :])
                loss = masked_asymmetric_pinball_loss(
                    target, base, lower.squeeze(2), upper.squeeze(2), mask, ALPHAS
                )
            else:
                radius, _ = head(feature[:, None, :])
                loss = masked_weighted_interval_score(target, base, radius.squeeze(2), mask, ALPHAS)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        selection_wis, _, _, _ = predict(head, selection_set, args.head_batch_size, device, asymmetric)
        history.append({"epoch": epoch, "selection_WIS": selection_wis})
        print(f"pemsd8 head={name} epoch={epoch} selection_WIS={selection_wis}", flush=True)
        if selection_wis < best_loss - 1e-5:
            best_loss, best_state, stale = selection_wis, copy.deepcopy(head.state_dict()), 0
        else:
            stale += 1
            if stale >= args.head_patience:
                break
    head.load_state_dict(best_state)
    fold_reports = []
    for index, (calibration, audit) in enumerate(folds, start=1):
        _, cal_lower, cal_upper, _ = predict(head, calibration, args.head_batch_size, device, asymmetric)
        _, audit_lower, audit_upper, _ = predict(head, audit, args.head_batch_size, device, asymmetric)
        if asymmetric:
            adjustment = cqr_adjustment(calibration, cal_lower, cal_upper)
            report = physical_asymmetric_metrics(audit, audit_lower, audit_upper, adjustment, mean, std)
        else:
            # Symmetric intervals use the same additive CQR conformity score.
            adjustment = cqr_adjustment(calibration, cal_lower, cal_upper)
            report = physical_asymmetric_metrics(audit, audit_lower, audit_upper, adjustment, mean, std)
        report["fold"] = index
        fold_reports.append(report)
    result = {
        "model": name,
        "asymmetric": asymmetric,
        "parameters": parameter_count(head),
        "history": history,
        "folds": fold_reports,
        "mean_WIS": float(np.mean([fold["conformal_WIS"] for fold in fold_reports])),
        "mean_coverage90": float(np.mean([fold["interval90_coverage"] for fold in fold_reports])),
        "coverage_gate_passed": all(0.89 <= fold["interval90_coverage"] <= 0.91 for fold in fold_reports),
    }
    run_dir = ROOT / args.output / "heads" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), run_dir / "best.pt")
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    args = arguments()
    set_seed(args.seed)
    train, valid, mean, std, nodes = load_pemsd8_train_valid()
    valid_selection_end = max(1, int(len(valid) * 0.4))
    backbone_valid = dataset_slice(valid, 0, valid_selection_end)
    output = ROOT / args.output
    checkpoint = output / f"backbone_seed_{args.seed}" / "best.pt"
    model, backbone_history = train_backbone(train, backbone_valid, nodes, mean, std, args, checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = torch.device(args.device)
    adjacency = load_adjacency(nodes).to(device)
    projection = torch.randn(
        model.in_steps * model.model_dim,
        8,
        generator=torch.Generator().manual_seed(20260726),
    ).to(device)
    projection /= (model.in_steps * model.model_dim) ** 0.5
    cache = ROOT / args.cache / f"pemsd8_seed_{args.seed}"
    for name, dataset in (("train", train), ("valid", valid)):
        loader = deterministic_subset(
            dataset,
            args.max_train_windows if name == "train" else args.max_valid_windows,
            args.backbone_batch_size,
        )
        cache_pems_split(model, loader, mean, std, device, adjacency, projection, cache / f"{name}.npz")
        print(f"pemsd8 cache {name} complete", flush=True)
    train_archive, valid_archive = load_archive(cache / "train.npz"), load_archive(cache / "valid.npz")
    fmean = train_archive["features"].mean(axis=(0, 1), keepdims=True)
    fstd = np.maximum(train_archive["features"].std(axis=(0, 1), keepdims=True), 1e-4)
    train_set = make_dataset(
        train_archive,
        fmean,
        fstd,
        max_instances=args.max_train_instances,
        sample_seed=args.seed,
    )
    windows = valid_archive["features"].shape[0]
    selection_set = make_dataset(valid_archive, fmean, fstd, slice(0, int(windows * 0.4)))
    fold_slices = (
        (slice(int(windows * 0.4), int(windows * 0.6)), slice(int(windows * 0.6), int(windows * 0.7))),
        (slice(int(windows * 0.5), int(windows * 0.7)), slice(int(windows * 0.7), int(windows * 0.8))),
        (slice(int(windows * 0.6), int(windows * 0.8)), slice(int(windows * 0.8), windows)),
    )
    folds = [
        (
            make_dataset(valid_archive, fmean, fstd, calibration_slice),
            make_dataset(valid_archive, fmean, fstd, audit_slice),
        )
        for calibration_slice, audit_slice in fold_slices
    ]
    reports = [
        train_head(name, train_set, selection_set, folds, mean, std, args)
        for name in ("constant", "mlp_wis", "cqr_mlp", "cqr_fourier")
    ]
    best_baseline = min(report["mean_WIS"] for report in reports if report["model"] != "cqr_fourier")
    proposed = next(report for report in reports if report["model"] == "cqr_fourier")
    gain = (best_baseline - proposed["mean_WIS"]) / best_baseline
    gate = proposed["coverage_gate_passed"] and gain >= 0.03
    summary = {
        "study": "SA-CQR-RAC PeMSD8 validation-only development",
        "complete": True,
        "canonical_test_split_used": False,
        "test_cache_created": False,
        "models": reports,
        "best_baseline_WIS": best_baseline,
        "proposed_gain_vs_best_baseline": gain,
        "validation_gate_passed": gate,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_manifest(
        output / "manifest.json",
        {
            "study": summary["study"],
            "canonical_test_split_used": False,
            "arguments": vars(args),
        },
        [
            ROOT / "data/raw/pemsd8/PEMS08.npz",
            ROOT / "data/raw/pemsd8/PEMS08.csv",
            ROOT / "configs/sacqr_pemsd8_validation.yaml",
        ],
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
