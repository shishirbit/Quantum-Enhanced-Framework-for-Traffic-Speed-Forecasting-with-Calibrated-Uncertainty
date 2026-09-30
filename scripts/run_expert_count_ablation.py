"""Reviewer-requested validation-only ablation of the number of RAC experts.

The experiment reuses the frozen METR-LA STAEformer feature cache and never
loads the canonical test split.  K is varied while the feature set, Fourier
gate width, optimizer, epoch budget, chronological selection/calibration/
evaluation split, and head seeds remain fixed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from quarts.models import (  # noqa: E402
    FourierRegimeGate,
    RegimeIntervalCalibrator,
    masked_weighted_interval_score,
    parameter_count,
)
from quarts.reproducibility import save_manifest, set_seed  # noqa: E402
from run_quarts_rac import (  # noqa: E402
    ALPHAS,
    conformal_factors,
    empirical_radii,
    initialize_experts,
    load_archive,
    make_dataset,
    physical_metrics,
    predict,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--head-seeds", nargs="+", type=int, default=[42, 73, 202])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-train-instances", type=int, default=200000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--cache",
        default="artifacts/cache/quarts_rac/metr_la_seed_42/tw_2000_vw_1000_ld_8",
    )
    parser.add_argument("--output", default="artifacts/reviewer_revision/expert_count_ablation")
    return parser.parse_args()


def train_one(
    experts: int,
    seed: int,
    train_set,
    selection_set,
    calibration_set,
    evaluation_set,
    mean: float,
    std: float,
    args: argparse.Namespace,
) -> dict:
    set_seed(seed)
    device = torch.device(args.device)
    hidden = train_set.tensors[0].shape[-1]
    horizons = train_set.tensors[1].shape[-1]
    gate = FourierRegimeGate(hidden, experts=experts, width=3, depth=2)
    head = RegimeIntervalCalibrator(
        gate,
        horizons=horizons,
        levels=len(ALPHAS),
        experts=experts,
    ).to(device)
    initialize_experts(head, empirical_radii(train_set))
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state = copy.deepcopy(head.state_dict())
    best_selection = math.inf
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train()
        for feature, base, target, mask in loader:
            feature = feature.to(device)
            base = base.to(device)
            target = target.to(device)
            mask = mask.to(device)
            radii, _ = head(feature[:, None, :])
            loss = masked_weighted_interval_score(
                target,
                base,
                radii.squeeze(2),
                mask,
                ALPHAS,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        selection_wis, _, _ = predict(head, selection_set, args.batch_size, device)
        history.append({"epoch": epoch, "selection_WIS_normalized": selection_wis})
        print(
            f"K={experts} seed={seed} epoch={epoch} selection_WIS={selection_wis:.6f}",
            flush=True,
        )
        if selection_wis < best_selection - 1e-5:
            best_selection = selection_wis
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    head.load_state_dict(best_state)
    _, calibration_radii, _ = predict(head, calibration_set, args.batch_size, device)
    factors = conformal_factors(calibration_set, calibration_radii)
    _, evaluation_radii, weights = predict(head, evaluation_set, args.batch_size, device)
    report = physical_metrics(evaluation_set, evaluation_radii, factors, mean, std)
    mean_weights = weights.mean(axis=0)
    entropy = -np.sum(mean_weights * np.log(np.maximum(mean_weights, 1e-12)))
    report.update(
        {
            "experts": experts,
            "head_seed": seed,
            "parameters": parameter_count(head),
            "mean_expert_weights": mean_weights.tolist(),
            "effective_experts": float(np.exp(entropy)),
            "mean_max_gate_probability": float(weights.max(axis=1).mean()),
            "elapsed_seconds": time.perf_counter() - started,
            "history": history,
            "canonical_test_split_used": False,
        }
    )
    run_dir = ROOT / args.output / f"k_{experts}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), run_dir / "best.pt")
    np.savez_compressed(
        run_dir / "evaluation.npz",
        radii=evaluation_radii.astype(np.float16),
        factors=factors.astype(np.float32),
        weights=weights.astype(np.float16),
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = arguments()
    cache = ROOT / args.cache
    train_archive = load_archive(cache / "train.npz")
    valid_archive = load_archive(cache / "valid.npz")
    feature_mean = train_archive["features"].mean(axis=(0, 1), keepdims=True)
    feature_std = np.maximum(
        train_archive["features"].std(axis=(0, 1), keepdims=True), 1e-4
    )
    train_set = make_dataset(
        train_archive,
        feature_mean,
        feature_std,
        max_instances=args.max_train_instances,
        sample_seed=42,
    )
    windows = valid_archive["features"].shape[0]
    early_end = max(1, int(windows * 0.4))
    calibration_end = max(early_end + 1, int(windows * 0.7))
    selection_set = make_dataset(
        valid_archive, feature_mean, feature_std, slice(0, early_end)
    )
    calibration_set = make_dataset(
        valid_archive, feature_mean, feature_std, slice(early_end, calibration_end)
    )
    evaluation_set = make_dataset(
        valid_archive, feature_mean, feature_std, slice(calibration_end, None)
    )
    reports = []
    for experts in args.experts:
        for seed in args.head_seeds:
            reports.append(
                train_one(
                    experts,
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
    summary = []
    for experts in args.experts:
        group = [item for item in reports if item["experts"] == experts]
        row = {"experts": experts, "runs": len(group)}
        for metric in (
            "conformal_WIS",
            "interval90_coverage",
            "interval90_mean_width",
            "effective_experts",
            "mean_max_gate_probability",
            "parameters",
            "elapsed_seconds",
        ):
            values = np.asarray([item[metric] for item in group], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": "Reviewer-requested validation-only Fourier-RAC expert-count ablation",
        "complete": True,
        "canonical_test_split_used": False,
        "validation_split": {
            "early_stopping": [0, early_end],
            "conformal_calibration": [early_end, calibration_end],
            "held_out_evaluation": [calibration_end, windows],
        },
        "arguments": vars(args),
        "summary": summary,
        "runs": reports,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_manifest(
        output / "manifest.json",
        {
            "study": payload["study"],
            "arguments": vars(args),
            "canonical_test_split_used": False,
        },
        [cache / "train.npz", cache / "valid.npz"],
    )
    print(json.dumps({"complete": True, "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
