"""Repair deterministic artifacts, then regenerate final analysis and audit."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ["scripts/evaluate_quantum_noise.py", "--config", "configs/full_experiment.yaml", "--runs", "artifacts/runs/confirmatory/metr_la/quantum"],
    ["scripts/run_naive_baselines.py", "--datasets", "metr_la", "pems_bay", "--output", "artifacts/runs/confirmatory"],
    ["scripts/analyze_results.py", "--runs", "artifacts/runs/confirmatory", "--output", "artifacts/analysis"],
    ["scripts/generate_figures.py", "--runs", "artifacts/runs/confirmatory", "--analysis", "artifacts/analysis", "--output", "artifacts/figures"],
    ["scripts/audit_completion.py", "--runs", "artifacts/runs/confirmatory", "--output", "artifacts/analysis/completion_audit.json"],
]


def main() -> None:
    for step in STEPS:
        print(f"RECOVERY START {' '.join(step)}", flush=True)
        result = subprocess.run([sys.executable, *step], cwd=ROOT)
        print(f"RECOVERY RETURN {result.returncode}", flush=True)


if __name__ == "__main__":
    main()
