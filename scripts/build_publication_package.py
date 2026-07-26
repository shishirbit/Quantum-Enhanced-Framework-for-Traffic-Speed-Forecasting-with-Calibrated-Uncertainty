"""Consolidate all completed experiments into publication tables and figures."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "artifacts/analysis"
RUNS = ROOT / "artifacts/runs"
SOURCE_FIGURES = ROOT / "artifacts/figures"
OUT = ROOT / "artifacts/publication"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
SUPPLEMENT = FIGURES / "supplement"

LABELS = {
    "historical_average": "Historical average",
    "persistence": "Persistence",
    "none": "QUARTS backbone",
    "mlp": "MLP",
    "fourier": "Fourier",
    "separable": "Separable VQC",
    "quantum": "Entangled QUARTS",
    "staeformer": "STAEformer",
    "constant": "Constant",
    "mlp_wis": "MLP-WIS",
    "cqr_mlp": "CQR-MLP",
    "cqr_fourier": "SA-CQR-Fourier",
}
COLORS = {
    "historical_average": "#9aa0a1",
    "persistence": "#bdc3c7",
    "none": "#6c757d",
    "mlp": "#3498db",
    "fourier": "#e67e22",
    "separable": "#8e44ad",
    "quantum": "#c0392b",
    "staeformer": "#2c3e50",
    "constant": "#7f8c8d",
    "mlp_wis": "#3498db",
    "cqr_mlp": "#2c3e50",
    "cqr_fourier": "#d35400",
}


def save_figure(fig, stem: str, supplement: bool = False) -> list[Path]:
    directory = SUPPLEMENT if supplement else FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    outputs = [directory / f"{stem}.png", directory / f"{stem}.pdf"]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return outputs


def write_table(frame: pd.DataFrame, name: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    frame.to_csv(path, index=False, float_format="%.6f")
    return path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def confirmatory_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(ANALYSIS / "run_metrics.csv")
    metric_columns = [
        "MAE",
        "RMSE",
        "MAPE",
        "WAPE",
        "Laplace_NLL",
        "interval90_coverage",
        "interval90_mean_width",
        "interval90_interval_score",
    ]
    rows = []
    for (dataset, model), part in runs.groupby(["dataset", "model"], sort=True):
        row = {
            "dataset": dataset,
            "model": model,
            "label": LABELS.get(model, model),
            "runs": len(part),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = part[metric].mean()
            row[f"{metric}_sd"] = part[metric].std(ddof=1)
        rows.append(row)
    performance = pd.DataFrame(rows)
    resources = (
        runs.groupby(["dataset", "model"], as_index=False)
        .agg(
            runs=("seed", "count"),
            parameters_mean=("parameters", "mean"),
            total_seconds_mean=("total_seconds", "mean"),
            total_seconds_sd=("total_seconds", "std"),
            inference_seconds_mean=("inference_seconds", "mean"),
            inference_seconds_sd=("inference_seconds", "std"),
        )
        .merge(performance[["dataset", "model", "MAE_mean"]], on=["dataset", "model"])
    )
    return performance, resources


def matched_effects(performance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(performance.dataset.unique()):
        part = performance[performance.dataset == dataset].set_index("model")
        proposed = part.loc["quantum", "MAE_mean"]
        for comparator in ("staeformer", "separable", "fourier", "mlp", "none"):
            baseline = part.loc[comparator, "MAE_mean"]
            rows.append(
                {
                    "dataset": dataset,
                    "proposed": "quantum",
                    "comparator": comparator,
                    "proposed_MAE": proposed,
                    "comparator_MAE": baseline,
                    "difference": proposed - baseline,
                    "relative_change_percent": 100 * (proposed - baseline) / baseline,
                    "winner": "QUARTS" if proposed < baseline else LABELS[comparator],
                }
            )
    return pd.DataFrame(rows)


def uncertainty_tables():
    uq = pd.read_csv(ANALYSIS / "quarts_uq_metr_three_seed_comparison.csv")
    rac = pd.read_csv(ANALYSIS / "quarts_rac_validation_three_seed_comparison.csv")
    raq = pd.read_csv(ANALYSIS / "raq_staeformer_validation_comparison.csv")
    pems4 = pd.read_csv(ANALYSIS / "prospective_pemsd4_comparison.csv")
    independent = pd.read_csv(ANALYSIS / "pemsd4_independent_backbone_summary.csv")
    sacqr = pd.read_csv(ANALYSIS / "sacqr_pemsd8_validation_comparison.csv")
    recent = load_json(ANALYSIS / "pemsd4_recent_conformal.json")
    recent_frame = pd.DataFrame(
        [
            {
                "seed": item["seed"],
                "calibration_fraction": item["calibration_fraction"],
                "WIS": item["conformal_WIS"],
                "coverage90": item["interval90_coverage"],
                "width90": item["interval90_mean_width"],
            }
            for item in recent
        ]
    )
    return uq, rac, raq, pems4, independent, sacqr, recent_frame


def decision_gate_table(
    performance,
    uq,
    rac,
    raq,
    pems4,
    independent,
    sacqr,
    recent,
):
    rows = []
    for dataset in ("metr_la", "pems_bay"):
        part = performance[performance.dataset == dataset].set_index("model")
        q, s = part.loc["quantum", "MAE_mean"], part.loc["staeformer", "MAE_mean"]
        rows.append(
            {
                "stage": "QUARTS v1 confirmatory",
                "dataset": dataset,
                "split_policy": "canonical test",
                "proposed": "Entangled QUARTS",
                "best_comparator": "STAEformer",
                "metric": "MAE",
                "proposed_value": q,
                "comparator_value": s,
                "relative_change_percent": 100 * (q - s) / s,
                "coverage90": np.nan,
                "gate": "FAIL",
                "reason": "Substantially worse point forecasting",
            }
        )
    uq_index = uq.set_index("model")
    rows.append(
        {
            "stage": "QUARTS-UQ",
            "dataset": "METR-LA",
            "split_policy": "exploratory test",
            "proposed": "QUARTS-UQ",
            "best_comparator": "Separable VQC",
            "metric": "WIS",
            "proposed_value": uq_index.loc["quantum", "conformal_WIS_mean"],
            "comparator_value": uq_index.loc["separable", "conformal_WIS_mean"],
            "relative_change_percent": 100
            * (
                uq_index.loc["quantum", "conformal_WIS_mean"]
                - uq_index.loc["separable", "conformal_WIS_mean"]
            )
            / uq_index.loc["separable", "conformal_WIS_mean"],
            "coverage90": uq_index.loc["quantum", "interval90_coverage_mean"],
            "gate": "FAIL",
            "reason": "No 5% gain; separable circuit wins",
        }
    )
    rac_index = rac.set_index("model")
    rows.append(
        {
            "stage": "QUARTS-RAC",
            "dataset": "METR-LA",
            "split_policy": "validation only",
            "proposed": "QUARTS-RAC",
            "best_comparator": "Fourier-RAC",
            "metric": "WIS",
            "proposed_value": rac_index.loc["quantum", "conformal_WIS_mean"],
            "comparator_value": rac_index.loc["fourier", "conformal_WIS_mean"],
            "relative_change_percent": 100
            * (
                rac_index.loc["quantum", "conformal_WIS_mean"]
                - rac_index.loc["fourier", "conformal_WIS_mean"]
            )
            / rac_index.loc["fourier", "conformal_WIS_mean"],
            "coverage90": rac_index.loc["quantum", "interval90_coverage_mean"],
            "gate": "FAIL",
            "reason": "Fourier gate wins all seeds",
        }
    )
    raq_index = raq.set_index("model")
    rows.append(
        {
            "stage": "RAQ-STAEformer",
            "dataset": "METR-LA",
            "split_policy": "validation only",
            "proposed": "Asymmetric Fourier-RAC",
            "best_comparator": "Symmetric Fourier-RAC",
            "metric": "WIS",
            "proposed_value": raq_index.loc["fourier_asym", "conformal_WIS_mean"],
            "comparator_value": raq_index.loc["fourier_sym", "conformal_WIS_mean"],
            "relative_change_percent": 100
            * (
                raq_index.loc["fourier_asym", "conformal_WIS_mean"]
                - raq_index.loc["fourier_sym", "conformal_WIS_mean"]
            )
            / raq_index.loc["fourier_sym", "conformal_WIS_mean"],
            "coverage90": raq_index.loc["fourier_asym", "interval90_coverage_mean"],
            "gate": "FAIL",
            "reason": "Worse WIS and overcoverage",
        }
    )
    pems4_index = pems4.set_index("model")
    rows.append(
        {
            "stage": "Fourier-RAC prospective",
            "dataset": "PeMSD4",
            "split_policy": "prospective test",
            "proposed": "Fourier-RAC",
            "best_comparator": "MLP-RAC",
            "metric": "WIS",
            "proposed_value": pems4_index.loc["fourier", "conformal_WIS"],
            "comparator_value": pems4_index.loc["mlp", "conformal_WIS"],
            "relative_change_percent": 100
            * (
                pems4_index.loc["fourier", "conformal_WIS"]
                - pems4_index.loc["mlp", "conformal_WIS"]
            )
            / pems4_index.loc["mlp", "conformal_WIS"],
            "coverage90": pems4_index.loc["fourier", "interval90_coverage"],
            "gate": "PROVISIONAL",
            "reason": "Small WIS gain; required independent backbones",
        }
    )
    independent_means = independent.groupby("model", as_index=True).mean(numeric_only=True)
    rows.append(
        {
            "stage": "Fourier-RAC independent backbones",
            "dataset": "PeMSD4",
            "split_policy": "3 independent test runs",
            "proposed": "Fourier-RAC",
            "best_comparator": "MLP-RAC",
            "metric": "WIS",
            "proposed_value": independent_means.loc["fourier", "WIS"],
            "comparator_value": independent_means.loc["mlp", "WIS"],
            "relative_change_percent": 100
            * (
                independent_means.loc["fourier", "WIS"]
                - independent_means.loc["mlp", "WIS"]
            )
            / independent_means.loc["mlp", "WIS"],
            "coverage90": independent_means.loc["fourier", "coverage90"],
            "gate": "FAIL",
            "reason": "Sign reversal and undercoverage on seed 202",
        }
    )
    seed202 = recent[(recent.seed == 202) & (recent.calibration_fraction == 0.30)].iloc[0]
    seed202_reference = recent[
        (recent.seed == 202) & (recent.calibration_fraction == 0.10)
    ].iloc[0]
    rows.append(
        {
            "stage": "Recent conformal repair",
            "dataset": "PeMSD4",
            "split_policy": "post-hoc diagnostic",
            "proposed": "Recent Fourier conformal",
            "best_comparator": "10% recent window",
            "metric": "WIS",
            "proposed_value": seed202.WIS,
            "comparator_value": seed202_reference.WIS,
            "relative_change_percent": 100
            * (seed202.WIS - seed202_reference.WIS)
            / seed202_reference.WIS,
            "coverage90": seed202.coverage90,
            "gate": "FAIL",
            "reason": "Coverage remained below 89%",
        }
    )
    sacqr_means = sacqr.groupby("model", as_index=True).mean(numeric_only=True)
    rows.append(
        {
            "stage": "SA-CQR-RAC",
            "dataset": "PeMSD8",
            "split_policy": "validation only; test untouched",
            "proposed": "SA-CQR-Fourier",
            "best_comparator": "CQR-MLP",
            "metric": "WIS",
            "proposed_value": sacqr_means.loc["cqr_fourier", "WIS"],
            "comparator_value": sacqr_means.loc["cqr_mlp", "WIS"],
            "relative_change_percent": 100
            * (
                sacqr_means.loc["cqr_fourier", "WIS"]
                - sacqr_means.loc["cqr_mlp", "WIS"]
            )
            / sacqr_means.loc["cqr_mlp", "WIS"],
            "coverage90": sacqr_means.loc["cqr_fourier", "coverage90"],
            "gate": "FAIL",
            "reason": "CQR-MLP wins; overcoverage",
        }
    )
    return pd.DataFrame(rows)


def figures(
    performance,
    resources,
    effects,
    gates,
    independent,
    recent,
    sacqr,
):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    produced = []
    datasets = ["metr_la", "pems_bay"]
    models = ["staeformer", "quantum", "separable", "fourier", "mlp", "none"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        part = performance[performance.dataset == dataset].set_index("model")
        means = [part.loc[model, "MAE_mean"] for model in models]
        errors = [part.loc[model, "MAE_sd"] for model in models]
        bars = axis.bar(
            range(len(models)),
            means,
            yerr=np.nan_to_num(errors),
            capsize=3,
            color=[COLORS[model] for model in models],
            edgecolor="black",
            linewidth=0.4,
        )
        axis.set_xticks(range(len(models)), [LABELS[model] for model in models], rotation=28, ha="right")
        axis.set_ylabel("MAE (mph)")
        axis.set_title(dataset.replace("_", "-").upper())
        axis.grid(axis="y", alpha=0.22)
        axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    produced += save_figure(fig, "figure01_confirmatory_point_forecasting")

    selected = effects[effects.comparator.isin(["staeformer", "separable", "fourier"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), width_ratios=(0.8, 1.2))
    x = np.arange(len(datasets))
    stae_values = [
        selected[
            (selected.dataset == dataset) & (selected.comparator == "staeformer")
        ]["relative_change_percent"].iloc[0]
        for dataset in datasets
    ]
    bars = axes[0].bar(x, stae_values, width=0.58, color=COLORS["staeformer"])
    axes[0].bar_label(bars, fmt="+%.1f%%", padding=2, fontsize=8)
    axes[0].set_xticks(
        x, [dataset.replace("_", "-").upper() for dataset in datasets]
    )
    axes[0].set_ylabel("Relative MAE change (%)")
    axes[0].set_title("Versus STAEformer")
    axes[0].grid(axis="y", alpha=0.22)
    width = 0.32
    for index, comparator in enumerate(["separable", "fourier"]):
        values = [
            selected[(selected.dataset == dataset) & (selected.comparator == comparator)][
                "relative_change_percent"
            ].iloc[0]
            for dataset in datasets
        ]
        bars = axes[1].bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=f"vs {LABELS[comparator]}",
            color=[COLORS["separable"], COLORS["fourier"]][index],
        )
        axes[1].bar_label(bars, fmt="%+.2f%%", padding=2, fontsize=8)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_xticks(
        x, [dataset.replace("_", "-").upper() for dataset in datasets]
    )
    axes[1].set_title("Versus capacity-matched controls")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].grid(axis="y", alpha=0.22)
    fig.suptitle("Entangled QUARTS relative MAE (positive values are worse)")
    produced += save_figure(fig, "figure02_quantum_matched_control_effects")

    uncertainty = gates[gates.metric == "WIS"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), squeeze=False)
    axis = axes[0, 0]
    colors = ["#c0392b" if gate == "FAIL" else "#e67e22" for gate in uncertainty.gate]
    values = uncertainty.relative_change_percent.fillna(0)
    axis.barh(range(len(uncertainty)), values, color=colors)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_yticks(range(len(uncertainty)), uncertainty.stage)
    axis.invert_yaxis()
    axis.set_xlabel("Relative WIS change vs best comparator (%)")
    axis.set_title("Stage-wise uncertainty effect\n(negative is better)")
    axis.grid(axis="x", alpha=0.22)
    axis = axes[0, 1]
    coverage = uncertainty.coverage90 * 100
    axis.scatter(coverage, range(len(uncertainty)), color=colors, s=42)
    axis.axvspan(89, 91, color="#2ecc71", alpha=0.13)
    axis.set_yticks(range(len(uncertainty)), uncertainty.stage)
    axis.invert_yaxis()
    axis.set_xlabel("90% interval coverage (%)")
    axis.set_title("Calibration across stages")
    axis.grid(axis="x", alpha=0.22)
    produced += save_figure(fig, "figure03_uncertainty_stage_synthesis")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), squeeze=False)
    for model, color, marker in (
        ("constant", COLORS["constant"], "o"),
        ("mlp", COLORS["mlp"], "s"),
        ("fourier", COLORS["fourier"], "^"),
    ):
        part = independent[independent.model == model].sort_values("seed")
        axes[0, 0].plot(part.seed.astype(str), part.WIS, marker=marker, label=LABELS[model], color=color)
        axes[0, 1].plot(
            part.seed.astype(str),
            100 * part.coverage90,
            marker=marker,
            label=LABELS[model],
            color=color,
        )
    axes[0, 0].set_ylabel("Test WIS")
    axes[0, 0].set_title("PeMSD4 independent backbones")
    axes[0, 1].set_ylabel("90% coverage (%)")
    axes[0, 1].set_title("Coverage stability")
    axes[0, 1].axhspan(89, 91, color="#2ecc71", alpha=0.13)
    for axis in axes[0]:
        axis.set_xlabel("Backbone seed")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    produced += save_figure(fig, "figure04_pemsd4_independent_backbones")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), squeeze=False)
    seed202 = recent[recent.seed == 202].sort_values("calibration_fraction")
    axes[0, 0].plot(
        100 * seed202.calibration_fraction,
        seed202.WIS,
        marker="o",
        color=COLORS["fourier"],
    )
    axes[0, 0].set_xlabel("Recent validation fraction (%)")
    axes[0, 0].set_ylabel("WIS")
    axes[0, 0].set_title("PeMSD4 seed 202 repair")
    for model, color, marker in (
        ("constant", COLORS["constant"], "o"),
        ("mlp_wis", COLORS["mlp_wis"], "s"),
        ("cqr_mlp", COLORS["cqr_mlp"], "^"),
        ("cqr_fourier", COLORS["cqr_fourier"], "D"),
    ):
        part = sacqr[sacqr.model == model].sort_values("fold")
        axes[0, 1].plot(
            part.fold,
            100 * part.coverage90,
            marker=marker,
            label=LABELS[model],
            color=color,
        )
    axes[0, 1].axhspan(89, 91, color="#2ecc71", alpha=0.13)
    axes[0, 1].set_xticks([1, 2, 3])
    axes[0, 1].set_xlabel("Rolling validation fold")
    axes[0, 1].set_ylabel("90% coverage (%)")
    axes[0, 1].set_title("PeMSD8 validation calibration")
    axes[0, 1].legend(frameon=False, fontsize=8)
    for axis in axes[0]:
        axis.grid(alpha=0.22)
    produced += save_figure(fig, "figure05_calibration_repair_and_transfer")

    selected_resources = resources[
        resources.model.isin(["quantum", "staeformer", "fourier", "separable"])
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    markers = {"metr_la": "o", "pems_bay": "s"}
    for model in ["staeformer", "quantum", "separable", "fourier"]:
        part = selected_resources[selected_resources.model == model]
        for dataset in datasets:
            row = part[part.dataset == dataset].iloc[0]
            ax.scatter(
                row.inference_seconds_mean,
                row.MAE_mean,
                marker=markers[dataset],
                color=COLORS[model],
                s=58,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
    from matplotlib.lines import Line2D

    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=COLORS[model],
            markeredgecolor="none",
            label=LABELS[model],
        )
        for model in ["staeformer", "quantum", "separable", "fourier"]
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[dataset],
            linestyle="",
            color="#555555",
            label=dataset.replace("_", "-").upper(),
        )
        for dataset in datasets
    ]
    ax.set_xscale("log")
    ax.set_xlabel("Mean test inference time (s, log scale)")
    ax.set_ylabel("Mean MAE (mph)")
    ax.set_title("Accuracy-efficiency trade-off")
    ax.grid(alpha=0.22)
    model_legend = ax.legend(
        handles=model_handles,
        title="Model",
        frameon=False,
        fontsize=7.5,
        title_fontsize=8,
        loc="upper right",
    )
    ax.add_artist(model_legend)
    ax.legend(
        handles=dataset_handles,
        title="Dataset",
        frameon=False,
        fontsize=7.5,
        title_fontsize=8,
        loc="center right",
    )
    produced += save_figure(fig, "figure06_accuracy_efficiency_tradeoff")
    return produced


def copy_supplementary() -> list[Path]:
    mapping = {
        "quarts_vs_staeformer_horizons": "figureS01_horizon_performance",
        "robustness_missing_metr_la": "figureS02_missing_input_metr_la",
        "robustness_missing_pems_bay": "figureS03_missing_input_pems_bay",
        "quantum_ablation_heatmap": "figureS04_quantum_ablation",
        "quantum_noise_sensitivity": "figureS05_quantum_noise",
        "quarts_v2_seed42_comparison": "figureS06_quarts_v2",
        "quarts_uq_metr_three_seed": "figureS07_quarts_uq",
        "quarts_rac_validation_three_seed": "figureS08_quarts_rac",
        "raq_staeformer_validation_comparison": "figureS09_raq_asymmetry",
        "prospective_pemsd4_comparison": "figureS10_pemsd4_prospective",
        "pemsd4_confirmatory_summary": "figureS11_pemsd4_head_seeds",
        "sacqr_pemsd8_validation_comparison": "figureS12_sacqr_validation",
    }
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    outputs = []
    for source_stem, target_stem in mapping.items():
        for suffix in (".png", ".pdf", ".svg"):
            source = SOURCE_FIGURES / f"{source_stem}{suffix}"
            if source.exists():
                target = SUPPLEMENT / f"{target_stem}{suffix}"
                shutil.copy2(source, target)
                outputs.append(target)
    return outputs


def manifest(files: list[Path]) -> Path:
    records = []
    for path in sorted(set(files)):
        records.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    output = OUT / "publication_manifest.json"
    output.write_text(
        json.dumps(
            {
                "complete": True,
                "scope": "All non-smoke completed experiments through D023",
                "excluded": ["smoke runs", "pre-protocol seed-17 run", "PeMSD8 canonical test"],
                "files": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def readme(table_paths, figure_paths, supplement_paths):
    lines = [
        "# Publication artifact package",
        "",
        "This package consolidates all non-smoke experiments completed through decision D023.",
        "PeMSD8 test results are intentionally absent because the SA-CQR validation gate failed.",
        "",
        "## Tables",
        "",
    ]
    lines += [f"- `{path.name}`" for path in table_paths]
    lines += ["", "## Main figures", ""]
    lines += [f"- `{path.name}`" for path in figure_paths if path.suffix == ".pdf"]
    lines += ["", "## Supplementary figures", ""]
    supplement_by_stem = {}
    suffix_priority = {".pdf": 0, ".svg": 1, ".png": 2}
    for path in supplement_paths:
        current = supplement_by_stem.get(path.stem)
        if current is None or suffix_priority[path.suffix] < suffix_priority[current.suffix]:
            supplement_by_stem[path.stem] = path
    lines += [
        f"- `{path.name}`"
        for path in sorted(supplement_by_stem.values(), key=lambda item: item.stem)
    ]
    lines += [
        "",
        "## Integrity exclusions",
        "",
        "- Smoke and preflight runs are excluded.",
        "- The pre-protocol METR-LA seed-17 batch-128 run is excluded.",
        "- Validation-only branches are labeled and not mixed with canonical test results.",
        "- PeMSD8 canonical test targets were not evaluated.",
        "",
    ]
    path = OUT / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    performance, resources = confirmatory_tables()
    effects = matched_effects(performance)
    uq, rac, raq, pems4, independent, sacqr, recent = uncertainty_tables()
    gates = decision_gate_table(performance, uq, rac, raq, pems4, independent, sacqr, recent)
    paired = pd.read_csv(ANALYSIS / "paired_tests.csv")
    friedman = pd.read_csv(ANALYSIS / "friedman_tests.csv")
    pems4_paired = pd.DataFrame(load_json(ANALYSIS / "pemsd4_paired_tests.json"))
    v2 = pd.read_csv(ANALYSIS / "quarts_v2_seed42_comparison.csv")
    ablation_rows = []
    for path in (RUNS / "ablation/metr_la").glob("q*_d*/seed_*/metrics.json"):
        payload = load_json(path)
        architecture = path.parents[1].name
        q, d = architecture.split("_")
        ablation_rows.append(
            {
                "qubits": int(q[1:]),
                "depth": int(d[1:]),
                "MAE": payload["MAE"],
                "RMSE": payload["RMSE"],
                "parameters": payload["parameters"],
                "inference_seconds": payload["inference_seconds"],
            }
        )
    ablation = pd.DataFrame(ablation_rows).sort_values(["qubits", "depth"])
    table_frames = [
        (performance, "table01_confirmatory_point_forecasting.csv"),
        (effects, "table02_quantum_matched_control_effects.csv"),
        (resources, "table03_resource_efficiency.csv"),
        (paired, "table04_confirmatory_paired_statistics.csv"),
        (friedman, "table04b_confirmatory_friedman_statistics.csv"),
        (ablation, "table05_quantum_architecture_ablation.csv"),
        (v2, "table06_quarts_v2_pilot.csv"),
        (uq, "table07_quarts_uq.csv"),
        (rac, "table08_quarts_rac_validation.csv"),
        (raq, "table09_raq_asymmetry_validation.csv"),
        (pems4, "table10_pemsd4_prospective.csv"),
        (independent, "table11_pemsd4_independent_backbones.csv"),
        (pems4_paired, "table12_pemsd4_paired_statistics.csv"),
        (recent, "table13_pemsd4_recent_conformal.csv"),
        (sacqr, "table14_sacqr_pemsd8_validation.csv"),
        (gates, "table15_experiment_gate_summary.csv"),
    ]
    table_paths = [write_table(frame, name) for frame, name in table_frames]
    figure_paths = figures(performance, resources, effects, gates, independent, recent, sacqr)
    supplement_paths = copy_supplementary()
    readme_path = readme(table_paths, figure_paths, supplement_paths)
    manifest_path = manifest(table_paths + figure_paths + supplement_paths + [readme_path])
    print(
        json.dumps(
            {
                "tables": len(table_paths),
                "main_figure_files": len(figure_paths),
                "supplementary_figure_files": len(supplement_paths),
                "readme": str(readme_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
