"""Fail closed unless all frozen-protocol output artifacts are present."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("metr_la", "pems_bay")
REFERENCE = ("none", "mlp", "fourier", "separable", "quantum")
SEEDS = (17, 42, 73, 101, 202)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="artifacts/runs/confirmatory")
    parser.add_argument("--output", default="artifacts/analysis/completion_audit.json")
    args = parser.parse_args()
    runs = ROOT / args.runs
    missing: list[str] = []

    for dataset in DATASETS:
        for model in REFERENCE:
            for seed in SEEDS:
                run = runs / dataset / model / f"seed_{seed}"
                for artifact in ("metrics.json", "predictions.npz", "best.pt", "manifest.json", "robustness.json"):
                    if not (run / artifact).exists():
                        missing.append(str((run / artifact).relative_to(ROOT)))
                if dataset == "metr_la" and not (run / "zero_shot_pems_bay.json").exists():
                    missing.append(str((run / "zero_shot_pems_bay.json").relative_to(ROOT)))
                if dataset == "metr_la" and model == "quantum" and not (run / "quantum_noise.json").exists():
                    missing.append(str((run / "quantum_noise.json").relative_to(ROOT)))
        for seed in SEEDS:
            run = runs / dataset / "staeformer" / f"seed_{seed}"
            for artifact in ("metrics.json", "predictions.npz", "best.pt", "manifest.json"):
                if not (run / artifact).exists():
                    missing.append(str((run / artifact).relative_to(ROOT)))
        for model in ("persistence", "historical_average"):
            run = runs / dataset / model / "seed_0"
            for artifact in ("metrics.json", "predictions.npz", "manifest.json"):
                if not (run / artifact).exists():
                    missing.append(str((run / artifact).relative_to(ROOT)))

    for qubits in (2, 4, 6):
        for depth in (1, 2, 3):
            path = ROOT / "artifacts/runs/ablation/metr_la" / f"q{qubits}_d{depth}" / "seed_42/metrics.json"
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
    for artifact in ("run_metrics.csv", "summary.csv", "paired_tests.csv", "friedman_tests.csv"):
        path = ROOT / "artifacts/analysis" / artifact
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))

    report = {
        "complete": not missing,
        "expected_reference_runs": len(DATASETS) * len(REFERENCE) * len(SEEDS),
        "expected_official_baseline_runs": len(DATASETS) * len(SEEDS),
        "expected_ablation_runs": 9,
        "missing_count": len(missing),
        "missing": missing,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(f"Completion audit failed: {len(missing)} artifacts missing")


if __name__ == "__main__":
    main()
