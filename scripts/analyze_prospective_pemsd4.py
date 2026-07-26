"""Create the prospective PeMSD4 comparison table and publication figure."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts" / "runs" / "prospective_pemsd4" / "summary.json"
OUT_CSV = ROOT / "artifacts" / "analysis" / "prospective_pemsd4_comparison.csv"
OUT_PNG = ROOT / "artifacts" / "figures" / "prospective_pemsd4_comparison.png"
OUT_SVG = ROOT / "artifacts" / "figures" / "prospective_pemsd4_comparison.svg"


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = []
    for item in summary["models"]:
        rows.append(
            {
                "dataset": summary["dataset"],
                "model": item["model"],
                "MAE": item["MAE"],
                "conformal_WIS": item["conformal_WIS"],
                "uncalibrated_WIS": item["uncalibrated_WIS"],
                "interval90_coverage": item["interval90_coverage"],
                "interval90_mean_width": item["interval90_mean_width"],
                "calibrator_parameters": item["calibrator_parameters"],
            }
        )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["model"] for row in rows]
    wis = [row["conformal_WIS"] for row in rows]
    coverage = [row["interval90_coverage"] * 100 for row in rows]
    widths = [row["interval90_mean_width"] for row in rows]
    colors = ["#7f8c8d", "#3498db", "#d35400"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, values, title, ylabel in zip(
        axes,
        (wis, coverage, widths),
        ("Conformal WIS (lower is better)", "90% coverage", "90% mean width"),
        ("WIS", "Coverage (%)", "Width (flow units)"),
    ):
        ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
    axes[1].axhspan(89, 91, color="#2ecc71", alpha=0.12, label="89–91% target")
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Prospective PeMSD4 interval evaluation (frozen backbone)", fontsize=12)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220)
    fig.savefig(OUT_SVG)
    plt.close(fig)
    print(json.dumps({"csv": str(OUT_CSV), "png": str(OUT_PNG), "svg": str(OUT_SVG)}, indent=2))


if __name__ == "__main__":
    main()
