"""Resumable qubit/depth sensitivity grid on the primary dataset."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for qubits in (2, 4, 6):
        for depth in (1, 2, 3):
            # q=4,d=2 is already estimated across all confirmatory seeds, but a
            # dedicated seed-42 artifact keeps the grid structurally complete.
            output = ROOT / "artifacts/runs/ablation/metr_la" / f"q{qubits}_d{depth}" / "seed_42"
            if (output / "metrics.json").exists():
                continue
            command = [
                sys.executable, str(ROOT / "scripts/train_reference.py"),
                "--config", str(ROOT / "configs/full_experiment.yaml"),
                "--data", str(ROOT / "data/raw/dcrnn/metr-la.h5"),
                "--adjacency", str(ROOT / "data/raw/dcrnn/adj_mx.pkl"),
                "--calibrator", "quantum", "--seed", "42",
                "--qubits", str(qubits), "--depth", str(depth),
                "--run-dir", str(output), "--device", "cuda",
            ]
            print(f"START q={qubits}, depth={depth}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            print(f"DONE q={qubits}, depth={depth}", flush=True)


if __name__ == "__main__":
    main()

