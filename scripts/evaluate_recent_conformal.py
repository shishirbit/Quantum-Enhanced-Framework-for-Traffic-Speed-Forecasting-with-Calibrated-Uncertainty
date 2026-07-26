"""Evaluate recent-window conformal calibration on frozen independent runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from quarts.models import FourierRegimeGate, RegimeIntervalCalibrator
from run_prospective_pemsd4 import predict_head
from run_quarts_rac import ALPHAS, conformal_factors, load_archive, make_dataset, physical_metrics


def evaluate(seed, fraction):
    if seed == 42:
        run_dir = ROOT / "artifacts/runs/prospective_pemsd4"
        cache_dir = ROOT / "artifacts/cache/prospective_pemsd4/pemsd4_seed_42"
    else:
        run_dir = ROOT / f"artifacts/runs/prospective_pemsd4_seed{seed}"
        cache_dir = ROOT / f"artifacts/cache/prospective_pemsd4_seed{seed}/pemsd4_seed_{seed}"
    train_archive, valid_archive, test_archive = (load_archive(cache_dir / f"{name}.npz") for name in ("train", "valid", "test"))
    fmean = train_archive["features"].mean(axis=(0, 1), keepdims=True)
    fstd = np.maximum(train_archive["features"].std(axis=(0, 1), keepdims=True), 1e-4)
    windows = valid_archive["features"].shape[0]
    start = max(0, int(windows * (1.0 - fraction)))
    calibration = make_dataset(valid_archive, fmean, fstd, slice(start, windows))
    test = make_dataset(test_archive, fmean, fstd)
    hidden, horizons = test.tensors[0].shape[-1], test.tensors[1].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = RegimeIntervalCalibrator(FourierRegimeGate(hidden, experts=3), horizons, len(ALPHAS), experts=3).to(device)
    head.load_state_dict(torch.load(run_dir / "heads/fourier" / f"seed_{seed}" / "best.pt", map_location=device, weights_only=True))
    _, cal_radii, _, _ = predict_head(head, calibration, 4096, device)
    factors = conformal_factors(calibration, cal_radii)
    _, test_radii, _, _ = predict_head(head, test, 4096, device)
    report = physical_metrics(test, test_radii, factors, test_archive["mean"], test_archive["std"])
    report.update({"seed": seed, "calibration_fraction": fraction, "calibration_start_window": start, "calibration_windows": windows - start, "model": "fourier_recent_conformal", "test_used_for_selection": False})
    return report


def main():
    reports = []
    for seed in (42, 73, 202):
        for fraction in (0.10, 0.20, 0.30):
            reports.append(evaluate(seed, fraction))
    out = ROOT / "artifacts/analysis/pemsd4_recent_conformal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
