"""Summarize and plot the completed QUARTS v2 exploratory pilot."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts" / "runs" / "exploratory_v2" / "metr_la"
ANALYSIS = ROOT / "artifacts" / "analysis"
FIGURES = ROOT / "artifacts" / "figures"
ORDER = ["none", "mlp", "fourier", "separable", "quantum"]
LABELS = {
    "none": "STAEformer",
    "mlp": "STAE + MLP",
    "fourier": "STAE + Fourier",
    "separable": "STAE + separable VQC",
    "quantum": "QUARTS v2 (entangled)",
}


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = {
        name: json.loads((RUN_ROOT / name / "seed_42" / "metrics.json").read_text(encoding="utf-8"))
        for name in ORDER
    }
    base_mae = metrics["none"]["MAE"]
    rows = []
    for name in ORDER:
        item = metrics[name]
        rows.append(
            {
                "model": name,
                "label": LABELS[name],
                "MAE": item["MAE"],
                "MAE_gain_vs_STAEformer": base_mae - item["MAE"],
                "MAE_relative_gain_percent": 100 * (base_mae - item["MAE"]) / base_mae,
                "RMSE": item["RMSE"],
                "Laplace_NLL": item["Laplace_NLL"],
                "interval90_coverage": item["interval90_coverage"],
                "interval90_interval_score": item["interval90_interval_score"],
                "parameters": item["calibrator_parameters"],
            }
        )

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS / "quarts_v2_seed42_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    point_ranking = sorted(ORDER, key=lambda name: metrics[name]["MAE"])
    decision = {
        "complete": True,
        "study": "QUARTS v2 exploratory pilot",
        "dataset": "METR-LA",
        "seed": 42,
        "point_metric_ranking": point_ranking,
        "best_point_model": point_ranking[0],
        "best_uncertainty_nll_model": min(ORDER, key=lambda name: metrics[name]["Laplace_NLL"]),
        "quantum_MAE_gain_vs_STAEformer": base_mae - metrics["quantum"]["MAE"],
        "quantum_MAE_difference_vs_Fourier": metrics["quantum"]["MAE"] - metrics["fourier"]["MAE"],
        "quantum_MAE_difference_vs_separable": metrics["quantum"]["MAE"] - metrics["separable"]["MAE"],
        "promotion_gate_passed": (
            metrics["quantum"]["MAE"] < metrics["none"]["MAE"]
            and metrics["quantum"]["MAE"] < metrics["fourier"]["MAE"]
            and metrics["quantum"]["MAE"] < metrics["separable"]["MAE"]
        ),
        "decision": (
            "Do not launch a new confirmatory matrix: the entangled VQC preserved "
            "the strong backbone but did not beat the Fourier or separable controls."
        ),
    }
    (ANALYSIS / "quarts_v2_pilot_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    labels = [LABELS[name] for name in ORDER]
    gains = np.asarray([base_mae - metrics[name]["MAE"] for name in ORDER]) * 1000
    interval_scores = [metrics[name]["interval90_interval_score"] for name in ORDER]
    colors = ["#888888", "#4c78a8", "#f58518", "#54a24b", "#b279a2"]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    bars = axes[0].barh(labels, gains, color=colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("MAE improvement vs frozen STAEformer (thousandths of mph)")
    axes[0].set_title("Point-forecast effect is negligible")
    axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[0].set_xlim(left=min(-0.01, gains.min() - 0.02), right=gains.max() + 0.06)

    bars = axes[1].barh(labels, interval_scores, color=colors)
    axes[1].set_xlabel("90% interval score (lower is better)")
    axes[1].set_title("Uncertainty interval quality improves")
    axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[1].set_xlim(0, max(interval_scores) * 1.15)
    for axis in axes:
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("QUARTS v2 exploratory pilot — METR-LA, seed 42")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"quarts_v2_seed42_comparison.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QUARTS v2 analysis failed: {exc}", file=sys.stderr)
        raise
