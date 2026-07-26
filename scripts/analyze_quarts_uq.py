"""Aggregate the QUARTS-UQ exploratory experiment and apply its promotion gate."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "artifacts"
    / "runs"
    / "exploratory_uq"
    / "metr_la"
    / "backbone_seed_42"
)
ANALYSIS = ROOT / "artifacts" / "analysis"
FIGURES = ROOT / "artifacts" / "figures"
ORDER = ["constant", "mlp", "fourier", "separable", "quantum"]
LEARNED_CONTROLS = ["fourier", "separable"]
LABELS = {
    "constant": "Constant scale",
    "mlp": "MLP",
    "fourier": "Fourier",
    "separable": "Separable VQC",
    "quantum": "QUARTS-UQ",
}
COLORS = {
    "constant": "#888888",
    "mlp": "#4c78a8",
    "fourier": "#f58518",
    "separable": "#54a24b",
    "quantum": "#b279a2",
}
METRICS = [
    "Laplace_NLL",
    "Laplace_CRPS",
    "conformal_WIS",
    "interval90_coverage",
    "interval90_mean_width",
    "interval90_interval_score",
    "interval90_calibration_error",
]


def load_runs() -> dict[str, list[dict]]:
    runs = {}
    for model in ORDER:
        paths = sorted(
            (RUN_ROOT / model).glob("head_seed_*/metrics.json"),
            key=lambda path: int(path.parent.name.removeprefix("head_seed_")),
        )
        if len(paths) != 3:
            raise RuntimeError(f"Expected three {model} runs, found {len(paths)}")
        runs[model] = [
            json.loads(path.read_text(encoding="utf-8")) for path in paths
        ]
    return runs


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    runs = load_runs()
    rows = []
    for model in ORDER:
        item = runs[model]
        row = {
            "model": model,
            "label": LABELS[model],
            "runs": len(item),
            "calibrator_parameters": item[0]["calibrator_parameters"],
            "MAE": item[0]["MAE"],
            "point_forecast_modified": item[0]["point_forecast_modified"],
        }
        for metric in METRICS:
            values = np.asarray([run[metric] for run in item], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)

    by_model = {row["model"]: row for row in rows}
    quantum_wis = by_model["quantum"]["conformal_WIS_mean"]
    gains = {
        control: 100.0
        * (by_model[control]["conformal_WIS_mean"] - quantum_wis)
        / by_model[control]["conformal_WIS_mean"]
        for control in LEARNED_CONTROLS
    }
    coverage = by_model["quantum"]["interval90_coverage_mean"]
    wis_gate = all(gain >= 5.0 for gain in gains.values())
    coverage_gate = 0.88 <= coverage <= 0.92
    point_gate = all(
        run["point_forecast_modified"] is False for run in runs["quantum"]
    )
    promotion_passed = wis_gate and coverage_gate and point_gate

    paired_wins = {
        control: sum(
            q["conformal_WIS"] < c["conformal_WIS"]
            for q, c in zip(runs["quantum"], runs[control], strict=True)
        )
        for control in LEARNED_CONTROLS
    }
    decision = {
        "complete": True,
        "study": "QUARTS-UQ exploratory",
        "dataset": "METR-LA",
        "backbone_seed": 42,
        "head_seeds": [42, 73, 202],
        "primary_metric": "conformal_WIS",
        "lower_is_better": True,
        "best_mean_WIS_model": min(
            ORDER, key=lambda model: by_model[model]["conformal_WIS_mean"]
        ),
        "best_mean_CRPS_model": min(
            ORDER, key=lambda model: by_model[model]["Laplace_CRPS_mean"]
        ),
        "best_mean_NLL_model": min(
            ORDER, key=lambda model: by_model[model]["Laplace_NLL_mean"]
        ),
        "best_coverage_calibration_model": min(
            ORDER,
            key=lambda model: by_model[model][
                "interval90_calibration_error_mean"
            ],
        ),
        "quantum_WIS_gain_percent_vs_Fourier": gains["fourier"],
        "quantum_WIS_gain_percent_vs_separable": gains["separable"],
        "quantum_paired_WIS_wins_vs_Fourier": paired_wins["fourier"],
        "quantum_paired_WIS_wins_vs_separable": paired_wins["separable"],
        "paired_runs": 3,
        "quantum_90_coverage": coverage,
        "gate": {
            "minimum_WIS_gain_percent_vs_each_control": 5.0,
            "coverage_range": [0.88, 0.92],
            "point_forecast_must_be_unchanged": True,
            "WIS_gate_passed": wis_gate,
            "coverage_gate_passed": coverage_gate,
            "point_forecast_gate_passed": point_gate,
        },
        "promotion_gate_passed": promotion_passed,
        "pems_bay_replication_justified": False,
        "decision": (
            "Do not promote QUARTS-UQ or launch a PEMS-BAY replication. "
            "The separable VQC is the best aggregate uncertainty model and the "
            "entangled VQC misses the predeclared 5% WIS advantage by a wide margin."
        ),
    }

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS / "quarts_uq_metr_three_seed_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ANALYSIS / "quarts_uq_metr_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    labels = [LABELS[model] for model in ORDER]
    colors = [COLORS[model] for model in ORDER]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    panels = [
        ("conformal_WIS", "Conformal WIS", "Primary score (lower is better)"),
        ("Laplace_CRPS", "Laplace CRPS", "Distribution score (lower is better)"),
        (
            "interval90_calibration_error",
            "Absolute coverage error",
            "90% calibration (lower is better)",
        ),
    ]
    for axis, (metric, xlabel, title) in zip(axes, panels, strict=True):
        means = np.asarray([by_model[m][f"{metric}_mean"] for m in ORDER])
        stds = np.asarray([by_model[m][f"{metric}_std"] for m in ORDER])
        positions = np.arange(len(ORDER))
        axis.barh(
            positions,
            means,
            xerr=stds,
            color=colors,
            alpha=0.9,
            capsize=3,
        )
        axis.set_yticks(positions, labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        panel_max = max(means + stds)
        for position, mean, std in zip(positions, means, stds, strict=True):
            small_bar = mean < 0.15 * panel_max
            axis.text(
                mean + std + 0.015 * panel_max if small_bar else mean * 0.985,
                position,
                f"{mean:.4f}",
                va="center",
                ha="left" if small_bar else "right",
                fontsize=8,
                color="black" if small_bar else "white",
                fontweight="bold",
            )
        axis.set_xlim(right=panel_max * 1.08)
    figure.suptitle(
        "QUARTS-UQ exploratory comparison — METR-LA, three head seeds"
    )
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            FIGURES / f"quarts_uq_metr_three_seed.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QUARTS-UQ analysis failed: {exc}", file=sys.stderr)
        raise
