"""Aggregate complete independent-backbone PeMSD4 runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 73, 202)
RUNS = {
    42: ROOT / "artifacts/runs/prospective_pemsd4/summary.json",
    73: ROOT / "artifacts/runs/prospective_pemsd4_seed73/summary.json",
    202: ROOT / "artifacts/runs/prospective_pemsd4_seed202/summary.json",
}
OUT_CSV = ROOT / "artifacts/analysis/pemsd4_independent_backbone_summary.csv"
OUT_PNG = ROOT / "artifacts/figures/pemsd4_independent_backbone_summary.png"


def main():
    rows = []
    for seed in SEEDS:
        summary = json.loads(RUNS[seed].read_text(encoding="utf-8"))
        for model in summary["models"]:
            rows.append({"seed": seed, "model": model["model"], "MAE": model["MAE"], "WIS": model["conformal_WIS"], "coverage90": model["interval90_coverage"], "width90": model["interval90_mean_width"]})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    matrix = {name: [row for row in rows if row["model"] == name] for name in ("constant", "mlp", "fourier")}
    mean = {name: float(np.mean([row["WIS"] for row in values])) for name, values in matrix.items()}
    sd = {name: float(np.std([row["WIS"] for row in values], ddof=1)) for name, values in matrix.items()}
    coverage = {name: float(np.mean([row["coverage90"] for row in values])) for name, values in matrix.items()}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    x = np.arange(len(SEEDS))
    for name, color in (("constant", "#7f8c8d"), ("mlp", "#3498db"), ("fourier", "#d35400")):
        vals = [row["WIS"] for row in matrix[name]]
        axes[0].plot(x, vals, marker="o", linewidth=2, label=name, color=color)
        axes[1].plot(x, [row["coverage90"] * 100 for row in matrix[name]], marker="o", linewidth=2, label=name, color=color)
    axes[0].set_title("Test WIS by independent backbone seed")
    axes[0].set_ylabel("WIS (lower is better)")
    axes[1].set_title("90% coverage by independent seed")
    axes[1].set_ylabel("Coverage (%)")
    for ax in axes:
        ax.set_xticks(x, [str(seed) for seed in SEEDS])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[1].axhspan(89, 91, color="#2ecc71", alpha=0.12)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)
    print(json.dumps({"mean_WIS": mean, "sd_WIS": sd, "mean_coverage90": coverage, "csv": str(OUT_CSV), "figure": str(OUT_PNG)}, indent=2))


if __name__ == "__main__":
    main()
