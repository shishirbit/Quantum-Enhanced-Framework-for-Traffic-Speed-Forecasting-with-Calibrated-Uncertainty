"""Resumable confirmatory training matrix for the frozen experiment protocol."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "metr_la": ("data/raw/dcrnn/metr-la.h5", "data/raw/dcrnn/adj_mx.pkl"),
    "pems_bay": ("data/raw/dcrnn/pems-bay.h5", "data/raw/dcrnn/adj_mx_bay.pkl"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_experiment.yaml")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--calibrators", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output", default="artifacts/runs/confirmatory")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    calibrators = args.calibrators or config["evaluation"]["calibrators"]
    seeds = args.seeds or config["evaluation"]["seeds"]
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    status_file = output / "matrix_status.jsonl"
    failures = []
    for dataset in args.datasets:
        data, adjacency = DATASETS[dataset]
        for calibrator in calibrators:
            for seed in seeds:
                run_dir = output / dataset / calibrator / f"seed_{seed}"
                metrics = run_dir / "metrics.json"
                if metrics.exists():
                    print(f"SKIP completed {dataset}/{calibrator}/{seed}", flush=True)
                    continue
                command = [
                    sys.executable,
                    str(ROOT / "scripts/train_reference.py"),
                    "--config", str(ROOT / args.config),
                    "--data", str(ROOT / data),
                    "--adjacency", str(ROOT / adjacency),
                    "--calibrator", calibrator,
                    "--seed", str(seed),
                    "--run-dir", str(run_dir),
                    "--device", "cuda",
                ]
                print(f"START {dataset}/{calibrator}/{seed}", flush=True)
                result = subprocess.run(command, cwd=ROOT)
                record = {"dataset": dataset, "calibrator": calibrator, "seed": seed, "returncode": result.returncode}
                with status_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                if result.returncode:
                    failures.append(record)
                    print(f"FAILED {record}", flush=True)
                else:
                    print(f"DONE {dataset}/{calibrator}/{seed}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} matrix run(s) failed; inspect {status_file}")


if __name__ == "__main__":
    main()
