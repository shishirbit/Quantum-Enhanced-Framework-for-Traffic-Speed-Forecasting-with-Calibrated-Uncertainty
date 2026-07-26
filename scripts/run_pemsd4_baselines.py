"""Compute lightweight independent uncertainty baselines from frozen caches."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_quarts_rac import ALPHAS, conformal_factors, load_archive, make_dataset, physical_metrics


def main():
    cache = ROOT / "artifacts/cache/prospective_pemsd4/pemsd4_seed_42"
    train_archive, valid_archive, test_archive = (
        load_archive(cache / f"{name}.npz") for name in ("train", "valid", "test")
    )
    fmean = train_archive["features"].mean(axis=(0, 1), keepdims=True)
    fstd = np.maximum(train_archive["features"].std(axis=(0, 1), keepdims=True), 1e-4)
    train = make_dataset(train_archive, fmean, fstd, max_instances=200000, sample_seed=42)
    windows = valid_archive["features"].shape[0]
    early_end, calibration_end = max(1, int(windows * 0.4)), max(2, int(windows * 0.7))
    calibration = make_dataset(valid_archive, fmean, fstd, slice(early_end, calibration_end))
    test = make_dataset(test_archive, fmean, fstd)
    _, train_base, train_target, train_mask = (x.numpy() for x in train.tensors)
    _, cal_base, cal_target, _ = (x.numpy() for x in calibration.tensors)
    _, test_base, _, _ = (x.numpy() for x in test.tensors)
    train_residual = np.abs(train_target - train_base)
    cal_residual = np.abs(cal_target - cal_base)
    reports = []
    sigma = np.sqrt(np.maximum((train_residual**2 * train_mask).sum(axis=0) / np.maximum(train_mask.sum(axis=0), 1), 1e-8))
    radii = np.zeros((len(test_base), test_base.shape[1], len(ALPHAS)), dtype=np.float32)
    cal_radii = np.zeros((len(cal_base), cal_base.shape[1], len(ALPHAS)), dtype=np.float32)
    for level, alpha in enumerate(ALPHAS):
        z = NormalDist().inv_cdf(1 - alpha / 2)
        radii[:, :, level] = sigma[None, :] * z
        cal_radii[:, :, level] = sigma[None, :] * z
    factors = conformal_factors(calibration, cal_radii)
    report = physical_metrics(test, radii, factors, test_archive["mean"], test_archive["std"])
    report.update({"model": "gaussian_scale_conformal", "dataset": "PeMSD4", "canonical_test_split_used": True, "test_used_for_selection": False, "point_forecast_modified": False})
    reports.append(report)
    out = ROOT / "artifacts/runs/prospective_pemsd4_confirmatory/baselines.json"
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
