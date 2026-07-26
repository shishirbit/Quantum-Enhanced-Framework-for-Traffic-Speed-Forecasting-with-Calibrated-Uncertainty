"""Analyze the completed validation-only QUARTS-RAC exploratory pilot."""
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
    / "exploratory_rac"
    / "metr_la"
    / "backbone_seed_42"
)
ANALYSIS = ROOT / "artifacts" / "analysis"
FIGURES = ROOT / "artifacts" / "figures"
ORDER = ["constant", "mlp", "fourier", "separable", "quantum"]
LABELS = {
    "constant": "Constant intervals",
    "mlp": "MLP regime gate",
    "fourier": "Fourier regime gate",
    "separable": "Separable VQC gate",
    "quantum": "QUARTS-RAC",
}
COLORS = {
    "constant": "#888888",
    "mlp": "#4c78a8",
    "fourier": "#f58518",
    "separable": "#54a24b",
    "quantum": "#b279a2",
}
METRICS = [
    "conformal_WIS",
    "uncalibrated_WIS",
    "interval90_coverage",
    "interval90_mean_width",
    "interval90_interval_score",
    "interval90_calibration_error",
    "effective_experts",
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
            "canonical_test_split_used": item[0]["canonical_test_split_used"],
            "point_forecast_modified": item[0]["point_forecast_modified"],
        }
        for metric in METRICS:
            values = np.asarray([run[metric] for run in item], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)
    by_model = {row["model"]: row for row in rows}

    quantum_wis = by_model["quantum"]["conformal_WIS_mean"]
    control_gains = {
        control: 100
        * (by_model[control]["conformal_WIS_mean"] - quantum_wis)
        / by_model[control]["conformal_WIS_mean"]
        for control in ("fourier", "separable")
    }
    paired_wins = {
        control: sum(
            quantum["conformal_WIS"] < baseline["conformal_WIS"]
            for quantum, baseline in zip(
                runs["quantum"],
                runs[control],
                strict=True,
            )
        )
        for control in ("fourier", "separable")
    }
    coverage = by_model["quantum"]["interval90_coverage_mean"]
    wis_gate = all(gain >= 5.0 for gain in control_gains.values())
    paired_gate = all(wins >= 2 for wins in paired_wins.values())
    coverage_gate = 0.89 <= coverage <= 0.91
    boundary_gate = all(
        run["canonical_test_split_used"] is False
        and run["point_forecast_modified"] is False
        for run in runs["quantum"]
    )
    promotion = wis_gate and paired_gate and coverage_gate and boundary_gate
    constant_wis = by_model["constant"]["conformal_WIS_mean"]
    architecture_gain = {
        model: 100
        * (constant_wis - by_model[model]["conformal_WIS_mean"])
        / constant_wis
        for model in ORDER[1:]
    }

    decision = {
        "complete": True,
        "study": "QUARTS-RAC validation-only exploratory pilot",
        "dataset": "METR-LA validation period only",
        "canonical_test_split_used": False,
        "backbone_seed": 42,
        "head_seeds": [42, 73, 202],
        "primary_metric": "conformal_WIS",
        "best_mean_WIS_model": min(
            ORDER,
            key=lambda model: by_model[model]["conformal_WIS_mean"],
        ),
        "best_mean_interval_score_model": min(
            ORDER,
            key=lambda model: by_model[model][
                "interval90_interval_score_mean"
            ],
        ),
        "WIS_gain_percent_vs_constant": architecture_gain,
        "quantum_WIS_gain_percent_vs_Fourier": control_gains["fourier"],
        "quantum_WIS_gain_percent_vs_separable": control_gains["separable"],
        "quantum_paired_WIS_wins_vs_Fourier": paired_wins["fourier"],
        "quantum_paired_WIS_wins_vs_separable": paired_wins["separable"],
        "quantum_90_coverage": coverage,
        "gate": {
            "minimum_WIS_gain_percent_vs_each_control": 5.0,
            "minimum_paired_wins_vs_each_control": 2,
            "coverage_range": [0.89, 0.91],
            "canonical_test_must_remain_unused": True,
            "point_forecast_must_be_unchanged": True,
            "WIS_gate_passed": wis_gate,
            "paired_win_gate_passed": paired_gate,
            "coverage_gate_passed": coverage_gate,
            "data_boundary_gate_passed": boundary_gate,
        },
        "promotion_gate_passed": promotion,
        "fresh_dataset_confirmation_justified": False,
        "decision": (
            "Do not promote the entangled QUARTS-RAC gate. The regime-aware "
            "architecture materially improves uncertainty scoring, but the "
            "Fourier gate is the strongest matched implementation and the "
            "entangled gate loses to Fourier in every seed."
        ),
    }

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    with (
        ANALYSIS / "quarts_rac_validation_three_seed_comparison.csv"
    ).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ANALYSIS / "quarts_rac_validation_decision.json").write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )

    labels = [LABELS[model] for model in ORDER]
    colors = [COLORS[model] for model in ORDER]
    panels = [
        ("conformal_WIS", "Conformal WIS", "Primary score (lower is better)"),
        (
            "interval90_interval_score",
            "90% interval score",
            "Sharpness and misses (lower is better)",
        ),
        (
            "effective_experts",
            "Effective experts",
            "Regime-bank utilization",
        ),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    for axis, (metric, xlabel, title) in zip(axes, panels, strict=True):
        means = np.asarray([by_model[m][f"{metric}_mean"] for m in ORDER])
        stds = np.asarray([by_model[m][f"{metric}_std"] for m in ORDER])
        positions = np.arange(len(ORDER))
        axis.barh(
            positions,
            means,
            xerr=stds,
            color=colors,
            capsize=3,
            alpha=0.9,
        )
        axis.set_yticks(positions, labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        panel_max = max(means + stds)
        for position, mean in zip(positions, means, strict=True):
            axis.text(
                mean * 0.985,
                position,
                f"{mean:.4f}",
                ha="right",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
        axis.set_xlim(right=panel_max * 1.08)
    figure.suptitle(
        "QUARTS-RAC validation-only comparison — METR-LA, three head seeds"
    )
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            FIGURES / f"quarts_rac_validation_three_seed.{suffix}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QUARTS-RAC analysis failed: {exc}", file=sys.stderr)
        raise
