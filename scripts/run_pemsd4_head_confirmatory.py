"""Confirmatory PeMSD4 uncertainty-head seeds and split-conformal baseline.

The PeMSD4 STAEformer backbone and frozen feature caches are reused exactly
from the prospective run. Only head initialization seeds vary (73, 202),
and the test split is read after validation selection and calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from quarts.reproducibility import save_manifest
from run_prospective_pemsd4 import predict_head, train_head
from run_quarts_rac import ALPHAS, conformal_factors, load_archive, make_dataset, physical_metrics


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[73, 202])
    parser.add_argument("--head-epochs", type=int, default=15)
    parser.add_argument("--head-patience", type=int, default=4)
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-instances", type=int, default=200000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache", default="artifacts/cache/prospective_pemsd4")
    parser.add_argument("--output", default="artifacts/runs/prospective_pemsd4_confirmatory")
    return parser.parse_args()


def split_validation(archive, feature_mean, feature_std):
    windows = archive["features"].shape[0]
    early_end = max(1, int(windows * 0.4))
    calibration_end = max(2, int(windows * 0.7))
    return (
        make_dataset(archive, feature_mean, feature_std, slice(0, early_end)),
        make_dataset(archive, feature_mean, feature_std, slice(early_end, calibration_end)),
    )


def split_conformal_baseline(calibration_set, test_set, mean, std):
    """Horizon- and alpha-specific split-conformal intervals."""
    _, cal_base, cal_target, cal_mask = (tensor.numpy() for tensor in calibration_set.tensors)
    _, test_base, _, _ = (tensor.numpy() for tensor in test_set.tensors)
    residual = np.abs(cal_target - cal_base)
    radii = np.zeros((len(test_base), test_base.shape[1], len(ALPHAS)), dtype=np.float32)
    factors = np.ones((test_base.shape[1], len(ALPHAS)), dtype=np.float32)
    for horizon in range(test_base.shape[1]):
        valid = cal_mask[:, horizon]
        values = residual[valid, horizon]
        for level, alpha in enumerate(ALPHAS):
            q = np.quantile(values, min(1.0, np.ceil((len(values) + 1) * (1 - alpha)) / len(values)), method="higher")
            radii[:, horizon, level] = max(float(q), 1e-3)
    return physical_metrics(test_set, radii, factors, mean, std)


def main():
    args = arguments()
    cache = ROOT / args.cache / "pemsd4_seed_42"
    train_archive, valid_archive, test_archive = (
        load_archive(cache / f"{name}.npz") for name in ("train", "valid", "test")
    )
    feature_mean = train_archive["features"].mean(axis=(0, 1), keepdims=True)
    feature_std = np.maximum(train_archive["features"].std(axis=(0, 1), keepdims=True), 1e-4)
    train_set = make_dataset(
        train_archive,
        feature_mean,
        feature_std,
        max_instances=args.max_train_instances,
        sample_seed=42,
    )
    selection_set, calibration_set = split_validation(valid_archive, feature_mean, feature_std)
    test_set = make_dataset(test_archive, feature_mean, feature_std)
    class Args:
        head_epochs = args.head_epochs
        head_patience = args.head_patience
        head_batch_size = args.head_batch_size
        output = args.output
        device = args.device

    reports = []
    for seed in args.seeds:
        reports.append(
            train_head(
                "fourier",
                seed,
                train_set,
                selection_set,
                calibration_set,
                test_set,
                test_archive["mean"],
                test_archive["std"],
                Args,
            )
        )
    conformal = split_conformal_baseline(
        calibration_set,
        test_set,
        test_archive["mean"],
        test_archive["std"],
    )
    conformal.update(
        {
            "study": "Prospective PeMSD4 confirmatory uncertainty evaluation",
            "dataset": "PeMSD4",
            "model": "split_conformal_horizon",
            "canonical_test_split_used": True,
            "test_used_for_selection": False,
            "point_forecast_modified": False,
        }
    )
    reports.append(conformal)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "study": "Prospective PeMSD4 confirmatory uncertainty evaluation",
        "complete": True,
        "dataset": "PeMSD4",
        "frozen_backbone_cache": str(cache),
        "seeds": args.seeds,
        "models": reports,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_manifest(
        output / "manifest.json",
        {
            "study": summary["study"],
            "dataset": "PeMSD4",
            "frozen_backbone_cache": str(cache),
            "arguments": vars(args),
        },
        [cache / "train.npz", cache / "valid.npz", cache / "test.npz"],
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
