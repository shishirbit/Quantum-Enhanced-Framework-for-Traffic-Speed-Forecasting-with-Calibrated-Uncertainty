"""Paired bootstrap and Wilcoxon tests on independent PeMSD4 test instances."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from quarts.models import FourierRegimeGate, MLPRegimeGate, RegimeIntervalCalibrator
from run_prospective_pemsd4 import predict_head
from run_quarts_rac import ALPHAS, conformal_factors, load_archive, make_dataset


def wis_rows(dataset, radii, factors, mean, std):
    _, base, target, mask = (x.numpy() for x in dataset.tensors)
    location = base * std + mean
    observed = target * std + mean
    calibrated = radii * factors[None, :, :] * std
    valid = mask.astype(bool)
    aggregate = 0.5 * np.abs(observed - location)
    for level, alpha in enumerate(ALPHAS):
        radius = calibrated[..., level]
        lower, upper = location - radius, location + radius
        score = upper - lower + (2 / alpha) * (lower - observed) * (observed < lower) + (2 / alpha) * (observed - upper) * (observed > upper)
        aggregate += (alpha / 2) * score
    row_valid = valid.all(axis=1)
    return (aggregate.sum(axis=1) / (len(ALPHAS) + 0.5) / valid.sum(axis=1))[row_valid]


def load_head(seed, name, hidden, horizons, device, run_dir):
    if name == "mlp":
        gate = MLPRegimeGate(hidden, experts=3)
    elif name == "fourier":
        gate = FourierRegimeGate(hidden, experts=3)
    else:
        gate = None
    head = RegimeIntervalCalibrator(gate, horizons, len(ALPHAS), experts=1 if gate is None else 3).to(device)
    path = run_dir / "heads" / name / f"seed_{seed}" / "best.pt"
    head.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return head


def one_seed(seed):
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
    early_end, calibration_end = max(1, int(windows * 0.4)), max(2, int(windows * 0.7))
    calibration = make_dataset(valid_archive, fmean, fstd, slice(early_end, calibration_end))
    test = make_dataset(test_archive, fmean, fstd)
    hidden, horizons = test.tensors[0].shape[-1], test.tensors[1].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = {}
    for name in ("mlp", "fourier"):
        head = load_head(seed, name, hidden, horizons, device, run_dir)
        _, cal_radii, _, _ = predict_head(head, calibration, 4096, device)
        factors = conformal_factors(calibration, cal_radii)
        _, test_radii, _, _ = predict_head(head, test, 4096, device)
        scores[name] = wis_rows(test, test_radii, factors, test_archive["mean"], test_archive["std"])
    return scores


def main():
    reports = []
    for seed in (42, 73, 202):
        scores = one_seed(seed)
        for baseline in ("mlp",):
            diff = scores["fourier"] - scores[baseline]
            rng = np.random.default_rng(seed + 9000)
            boot = []
            for _ in range(2000):
                boot.append(float(rng.choice(diff, len(diff), replace=True).mean()))
            ci = np.quantile(boot, [0.025, 0.975]).tolist()
            stat = wilcoxon(diff, alternative="two-sided", method="approx")
            reports.append({"seed": seed, "comparison": "fourier_minus_mlp", "n_instances": int(len(diff)), "mean_difference": float(diff.mean()), "bootstrap_ci95": ci, "wilcoxon_statistic": float(stat.statistic), "wilcoxon_pvalue": float(stat.pvalue)})
    out = ROOT / "artifacts/analysis/pemsd4_paired_tests.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
