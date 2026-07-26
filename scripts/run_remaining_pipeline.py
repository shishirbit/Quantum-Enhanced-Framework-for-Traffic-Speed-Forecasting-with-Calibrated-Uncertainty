"""Resumable sequential continuation after the live METR-LA matrix.

Every stage invokes an independently resumable script. A failed optional stage
is recorded and does not erase successful artifacts or prevent later analysis.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts" / "runs" / "pipeline_status.jsonl"


def run(name: str, arguments: list[str]) -> bool:
    command = [sys.executable, *arguments]
    print(f"PIPELINE START {name}", flush=True)
    started = time.time()
    result = subprocess.run(command, cwd=ROOT)
    record = {
        "stage": name,
        "returncode": result.returncode,
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "command": command,
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    print(f"PIPELINE {'DONE' if result.returncode == 0 else 'FAILED'} {name}", flush=True)
    return result.returncode == 0


def main() -> None:
    # D009: rerun the one seed that was produced before protocol freeze.
    run("rerun_metr_none_seed17", [
        "scripts/run_matrix.py", "--config", "configs/full_experiment.yaml",
        "--datasets", "metr_la", "--calibrators", "none", "--seeds", "17",
        "--output", "artifacts/runs/confirmatory",
    ])
    run("pems_reference_matrix", [
        "scripts/run_matrix.py", "--config", "configs/full_experiment.yaml",
        "--datasets", "pems_bay", "--output", "artifacts/runs/confirmatory",
    ])
    run("deterministic_baselines", [
        "scripts/run_naive_baselines.py", "--datasets", "metr_la", "pems_bay",
        "--output", "artifacts/runs/confirmatory",
    ])
    run("robustness", [
        "scripts/evaluate_robustness.py", "--config", "configs/full_experiment.yaml",
        "--runs", "artifacts/runs/confirmatory", "--datasets", "metr_la", "pems_bay",
    ])
    run("zero_shot_transfer", [
        "scripts/evaluate_transfer.py", "--config", "configs/full_experiment.yaml",
        "--source-runs", "artifacts/runs/confirmatory/metr_la",
    ])
    run("quantum_noise", [
        "scripts/evaluate_quantum_noise.py", "--config", "configs/full_experiment.yaml",
        "--runs", "artifacts/runs/confirmatory/metr_la/quantum",
    ])
    run("quantum_ablation", ["scripts/run_ablation_matrix.py"])

    stae_smoke = run("staeformer_preflight", [
        "scripts/run_staeformer.py", "--dataset", "METRLA", "--seed", "999",
        "--epochs", "1", "--patience", "1",
        "--output", "artifacts/runs/preflight/staeformer_metr_seed999",
    ])
    if stae_smoke:
        run("staeformer_matrix", [
            "scripts/run_staeformer_matrix.py", "--datasets", "METRLA", "PEMSBAY",
            "--output", "artifacts/runs/confirmatory",
        ])
    else:
        print("PIPELINE SKIP staeformer_matrix: preflight failed", flush=True)

    run("statistical_analysis", [
        "scripts/analyze_results.py", "--runs", "artifacts/runs/confirmatory",
        "--output", "artifacts/analysis",
    ])
    run("figures", [
        "scripts/generate_figures.py", "--runs", "artifacts/runs/confirmatory",
        "--analysis", "artifacts/analysis", "--output", "artifacts/figures",
    ])
    run("completion_audit", [
        "scripts/audit_completion.py", "--runs", "artifacts/runs/confirmatory",
        "--output", "artifacts/analysis/completion_audit.json",
    ])


if __name__ == "__main__":
    main()
