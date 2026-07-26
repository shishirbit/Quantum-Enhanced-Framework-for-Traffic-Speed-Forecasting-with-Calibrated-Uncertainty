"""Create SA-CQR-RAC PeMSD8 validation comparison artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts/runs/sacqr_pemsd8_validation/summary.json"
OUT_CSV = ROOT / "artifacts/analysis/sacqr_pemsd8_validation_comparison.csv"
OUT_PNG = ROOT / "artifacts/figures/sacqr_pemsd8_validation_comparison.png"


def main():
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = []
    for model in summary["models"]:
        for fold in model["folds"]:
            rows.append(
                {
                    "model": model["model"],
                    "fold": fold["fold"],
                    "WIS": fold["conformal_WIS"],
                    "coverage90": fold["interval90_coverage"],
                    "width90": fold["interval90_mean_width"],
                }
            )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = [model["model"] for model in summary["models"]]
    wis = [model["mean_WIS"] for model in summary["models"]]
    coverage = [model["mean_coverage90"] * 100 for model in summary["models"]]
    colors = ["#7f8c8d", "#3498db", "#2c3e50", "#d35400"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    axes[0].bar(names, wis, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Rolling-fold mean WIS")
    axes[0].set_title("Validation WIS (lower is better)")
    axes[1].bar(names, coverage, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].axhspan(89, 91, color="#2ecc71", alpha=0.14)
    axes[1].set_ylabel("Mean 90% coverage (%)")
    axes[1].set_title("Validation coverage")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)
    print(json.dumps({"csv": str(OUT_CSV), "figure": str(OUT_PNG)}, indent=2))


if __name__ == "__main__":
    main()
