"""Analyze the completed RAQ-STAEformer validation-only ablation."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts" / "runs" / "exploratory_raq" / "metr_la" / "summary.json"
ANALYSIS = ROOT / "artifacts" / "analysis"
FIGURES = ROOT / "artifacts" / "figures"
ORDER = ["constant_asym", "fourier_sym", "mlp_asym", "fourier_asym"]
LABELS = {
    "constant_asym": "Constant asymmetric",
    "fourier_sym": "Symmetric Fourier-RAC",
    "mlp_asym": "Asymmetric MLP",
    "fourier_asym": "RAQ-STAEformer",
}
COLORS = {
    "constant_asym": "#888888",
    "fourier_sym": "#f58518",
    "mlp_asym": "#4c78a8",
    "fourier_asym": "#b279a2",
}


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    by_model = {item["model"]: item for item in summary["models"]}
    symmetric = by_model["fourier_sym"]["conformal_WIS_mean"]
    asymmetric = by_model["fourier_asym"]["conformal_WIS_mean"]
    relative_change = 100 * (symmetric - asymmetric) / symmetric
    coverage = by_model["fourier_asym"]["interval90_coverage_mean"]
    decision = {
        "complete": True,
        "study": "RAQ-STAEformer validation-only exploratory ablation",
        "dataset": "METR-LA validation period only",
        "canonical_test_split_used": False,
        "primary_metric": "conformal_WIS",
        "best_mean_WIS_model": min(
            ORDER, key=lambda model: by_model[model]["conformal_WIS_mean"]
        ),
        "symmetric_fourier_WIS": symmetric,
        "asymmetric_fourier_WIS": asymmetric,
        "asymmetric_fourier_relative_WIS_change_percent": relative_change,
        "asymmetric_fourier_90_coverage": coverage,
        "gate": {
            "required_WIS_improvement_percent": 3.0,
            "coverage_range": [0.89, 0.91],
            "WIS_improvement_gate_passed": relative_change >= 3.0,
            "coverage_gate_passed": 0.89 <= coverage <= 0.91,
            "canonical_test_boundary_passed": not summary["canonical_test_split_used"],
        },
        "promotion_gate_passed": (
            relative_change >= 3.0 and 0.89 <= coverage <= 0.91
        ),
        "decision": (
            "Reject asymmetric RAQ intervals. Symmetric Fourier-RAC remains the "
            "strongest candidate: asymmetric Fourier increases WIS by "
            f"{abs(relative_change):.2f}% and over-covers at {coverage * 100:.2f}%."
        ),
        "next_step": (
            "Freeze symmetric Fourier-RAC and run a preregistered prospective "
            "evaluation with independent conformal baselines on a fresh dataset."
        ),
    }

    rows = []
    for model in ORDER:
        row = {"model": model, "label": LABELS[model]}
        row.update(by_model[model])
        rows.append(row)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS / "raq_staeformer_validation_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    decision_path = ANALYSIS / "raq_staeformer_validation_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    panels = [
        ("conformal_WIS", "Conformal WIS", "Lower is better"),
        ("interval90_coverage", "90% coverage", "Target: 89–91%"),
        ("interval90_interval_score", "90% interval score", "Lower is better"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    labels = [LABELS[model] for model in ORDER]
    colors = [COLORS[model] for model in ORDER]
    for axis, (metric, xlabel, title) in zip(axes, panels, strict=True):
        means = np.asarray([by_model[m][f"{metric}_mean"] for m in ORDER])
        stds = np.asarray([by_model[m][f"{metric}_std"] for m in ORDER])
        positions = np.arange(len(ORDER))
        axis.barh(positions, means, xerr=stds, color=colors, capsize=3)
        axis.set_yticks(positions, labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        for position, mean in zip(positions, means, strict=True):
            axis.text(
                mean * 0.985,
                position,
                f"{mean:.4f}",
                ha="right",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=8,
            )
        axis.set_xlim(right=max(means + stds) * 1.1)
    axes[1].axvspan(0.89, 0.91, color="#54a24b", alpha=0.12)
    figure.suptitle("RAQ-STAEformer validation-only ablation — METR-LA")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            FIGURES / f"raq_staeformer_validation_comparison.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"RAQ analysis failed: {exc}", file=sys.stderr)
        raise
