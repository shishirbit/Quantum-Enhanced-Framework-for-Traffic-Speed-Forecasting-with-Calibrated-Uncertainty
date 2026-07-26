"""Aggregate prospective PeMSD4 seed results and baseline comparison."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROSPECTIVE = ROOT / "artifacts" / "runs" / "prospective_pemsd4" / "summary.json"
CONFIRMATORY = ROOT / "artifacts" / "runs" / "prospective_pemsd4_confirmatory" / "summary.json"
BASELINES = ROOT / "artifacts" / "runs" / "prospective_pemsd4_confirmatory" / "baselines.json"
OUT_CSV = ROOT / "artifacts" / "analysis" / "pemsd4_confirmatory_summary.csv"
OUT_PNG = ROOT / "artifacts" / "figures" / "pemsd4_confirmatory_summary.png"


def main() -> None:
    prospective = json.loads(PROSPECTIVE.read_text(encoding="utf-8"))
    confirmatory = json.loads(CONFIRMATORY.read_text(encoding="utf-8"))
    fourier = [m for m in prospective["models"] if m["model"] == "fourier"]
    fourier += [m for m in confirmatory["models"] if m["model"] == "fourier"]
    baseline = next(m for m in prospective["models"] if m["model"] == "constant")
    split = next(m for m in confirmatory["models"] if m["model"] == "split_conformal_horizon")
    gaussian = json.loads(BASELINES.read_text(encoding="utf-8"))[0]
    rows = []
    for model in fourier:
        rows.append({"model": "Fourier-RAC", "seed": model["head_seed"], "WIS": model["conformal_WIS"], "coverage90": model["interval90_coverage"], "width90": model["interval90_mean_width"]})
    rows.append({"model": "Constant", "seed": 42, "WIS": baseline["conformal_WIS"], "coverage90": baseline["interval90_coverage"], "width90": baseline["interval90_mean_width"]})
    rows.append({"model": "Split conformal", "seed": "pooled", "WIS": split["conformal_WIS"], "coverage90": split["interval90_coverage"], "width90": split["interval90_mean_width"]})
    rows.append({"model": "Gaussian + conformal", "seed": "pooled", "WIS": gaussian["conformal_WIS"], "coverage90": gaussian["interval90_coverage"], "width90": gaussian["interval90_mean_width"]})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    all_mean_wis = float(np.mean([m["conformal_WIS"] for m in fourier]))
    all_std_wis = float(np.std([m["conformal_WIS"] for m in fourier], ddof=1))
    mean_wis = float(np.mean([m["conformal_WIS"] for m in fourier[1:]]))
    std_wis = float(np.std([m["conformal_WIS"] for m in fourier[1:]], ddof=1))
    mean_cov = float(np.mean([m["interval90_coverage"] for m in fourier]))
    baseline_wis = split["conformal_WIS"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    labels = ["Fourier-RAC\n(seed 42)", "Fourier-RAC\n(seeds 73/202)", "Split conformal", "Gaussian +\nconformal"]
    values = [fourier[0]["conformal_WIS"], mean_wis, baseline_wis, gaussian["conformal_WIS"]]
    bars = ax.bar(labels, values, color=["#d35400", "#e67e22", "#7f8c8d", "#95a5a6"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Test conformal WIS (lower is better)")
    ax.set_title("PeMSD4 confirmatory uncertainty evaluation")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.text(0.02, 0.96, f"Fourier seeds 73/202: {mean_wis:.2f} ± {std_wis:.2f}\nMean 90% coverage: {mean_cov*100:.2f}%", transform=ax.transAxes, ha="left", va="top", fontsize=9)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=220)
    plt.close(fig)
    print(json.dumps({"fourier_all_seed_mean_WIS": all_mean_wis, "fourier_all_seed_sd_WIS": all_std_wis, "fourier_additional_seed_mean_WIS": mean_wis, "fourier_additional_seed_sd_WIS": std_wis, "fourier_mean_coverage90": mean_cov, "split_conformal_WIS": baseline_wis, "csv": str(OUT_CSV), "figure": str(OUT_PNG)}, indent=2))


if __name__ == "__main__":
    main()
