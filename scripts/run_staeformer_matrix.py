from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["METRLA", "PEMSBAY"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73, 101, 202])
    parser.add_argument("--output", default="artifacts/runs/confirmatory")
    args = parser.parse_args()
    for dataset in args.datasets:
        key = "metr_la" if dataset == "METRLA" else "pems_bay"
        for seed in args.seeds:
            output = ROOT / args.output / key / "staeformer" / f"seed_{seed}"
            if (output / "metrics.json").exists():
                continue
            command = [sys.executable, str(ROOT / "scripts/run_staeformer.py"), "--dataset", dataset,
                       "--seed", str(seed), "--output", str(output)]
            print(f"START {dataset}/{seed}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            print(f"DONE {dataset}/{seed}", flush=True)


if __name__ == "__main__":
    main()

