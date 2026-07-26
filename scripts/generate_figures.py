"""Generate publication-ready figures from completed, structured artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"persistence": "Persistence", "historical_average": "Historical average", "none": "Backbone", "mlp": "MLP", "fourier": "Fourier", "separable": "Separable VQC", "quantum": "QUARTS", "staeformer": "STAEformer"}
COLORS = {"persistence": "#bdc3c7", "historical_average": "#95a5a6", "none": "#7f8c8d", "mlp": "#3498db", "fourier": "#f39c12", "separable": "#9b59b6", "quantum": "#c0392b", "staeformer": "#2c3e50"}


def save(fig, output: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="artifacts/runs/confirmatory")
    parser.add_argument("--analysis", default="artifacts/analysis")
    parser.add_argument("--output", default="artifacts/figures")
    args = parser.parse_args()
    runs, analysis, output = ROOT / args.runs, ROOT / args.analysis, ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})

    metrics_path = analysis / "run_metrics.csv"
    if metrics_path.exists():
        frame = pd.read_csv(metrics_path)
        if not frame.empty:
            datasets = sorted(frame.dataset.unique())
            fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.5), squeeze=False)
            for ax, dataset in zip(axes[0], datasets):
                part = frame[frame.dataset == dataset]
                models = sorted(part.model.unique(), key=lambda x: list(LABELS).index(x) if x in LABELS else 99)
                means = [part[part.model == model].MAE.mean() for model in models]
                errors = [part[part.model == model].MAE.std(ddof=1) for model in models]
                ax.bar(range(len(models)), means, yerr=np.nan_to_num(errors), color=[COLORS.get(m, "#555555") for m in models], capsize=3)
                ax.set_xticks(range(len(models)), [LABELS.get(m, m) for m in models], rotation=30, ha="right")
                ax.set_ylabel("MAE (mph)"); ax.set_title(dataset.replace("_", " ").upper())
            save(fig, output, "confirmatory_mae")

            quantum = frame[frame.model == "quantum"]
            if not quantum.empty:
                fig, ax = plt.subplots(figsize=(5.5, 3.5))
                for dataset, part in quantum.groupby("dataset"):
                    ax.scatter(part.parameters, part.inference_seconds, label=dataset.upper(), s=42)
                ax.set_xlabel("Trainable parameters"); ax.set_ylabel("Test inference time (s)"); ax.legend(frameon=False)
                save(fig, output, "quantum_resource_profile")

            # Direct proposed-model versus verified SOTA comparison. Keep the
            # raw direction of every metric visible; lower is better.
            sota = frame[frame.model.isin(["quantum", "staeformer"])].copy()
            if set(sota.model.unique()) == {"quantum", "staeformer"}:
                metrics = ["MAE", "RMSE", "MAPE", "WAPE"]
                fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), squeeze=False)
                width = 0.36
                for ax, metric in zip(axes.ravel(), metrics):
                    x = np.arange(len(datasets))
                    for offset, model in zip((-width / 2, width / 2), ("quantum", "staeformer")):
                        means = [sota[(sota.dataset == dataset) & (sota.model == model)][metric].mean() for dataset in datasets]
                        errors = [sota[(sota.dataset == dataset) & (sota.model == model)][metric].std(ddof=1) for dataset in datasets]
                        bars = ax.bar(x + offset, means, width, yerr=np.nan_to_num(errors), capsize=3,
                                      label=LABELS[model], color=COLORS[model])
                        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
                    ax.set_xticks(x, [dataset.replace("_", " ").upper() for dataset in datasets])
                    ax.set_ylabel(f"{metric}{' (%)' if metric in ('MAPE', 'WAPE') else ' (mph)' if metric == 'MAE' else ''}")
                    ax.set_title(f"{metric} (lower is better)")
                axes[0, 0].legend(frameon=False)
                save(fig, output, "quarts_vs_staeformer_metrics")

                resource = sota.groupby(["dataset", "model"], as_index=False).agg(
                    MAE=("MAE", "mean"), inference_seconds=("inference_seconds", "mean"),
                    parameters=("parameters", "mean"), total_seconds=("total_seconds", "mean"),
                )
                fig, ax = plt.subplots(figsize=(6.2, 4.1))
                markers = {"quantum": "o", "staeformer": "s"}
                for _, row in resource.iterrows():
                    ax.scatter(row.inference_seconds, row.MAE, s=70, marker=markers[row.model],
                               color=COLORS[row.model])
                    ax.annotate(f"{LABELS[row.model]}\n{row.dataset.upper()}",
                                (row.inference_seconds, row.MAE), xytext=(5, 5),
                                textcoords="offset points", fontsize=7)
                ax.set_xscale("log")
                ax.set_xlabel("Mean test inference time (s, log scale)")
                ax.set_ylabel("Mean MAE (mph)")
                ax.set_title("Accuracy–inference trade-off")
                save(fig, output, "quarts_vs_staeformer_efficiency")

                summary_rows = []
                for dataset in datasets:
                    part = sota[sota.dataset == dataset]
                    for metric in metrics:
                        q_value = part[part.model == "quantum"][metric].mean()
                        s_value = part[part.model == "staeformer"][metric].mean()
                        summary_rows.append({
                            "dataset": dataset, "metric": metric,
                            "QUARTS_mean": q_value, "STAEformer_mean": s_value,
                            "QUARTS_relative_change_percent": 100 * (q_value - s_value) / s_value,
                        })
                pd.DataFrame(summary_rows).to_csv(analysis / "quarts_vs_staeformer.csv", index=False)

                horizon_rows = []
                for path in runs.glob("*/[qs]*/seed_*/metrics.json"):
                    model = path.parents[1].name
                    if model not in {"quantum", "staeformer"}:
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    for horizon, values in payload.get("horizons", {}).items():
                        horizon_rows.append({"dataset": path.parents[2].name, "model": model,
                                             "horizon": horizon, **values})
                if horizon_rows:
                    horizons = ["15min", "30min", "60min"]
                    horizon_frame = pd.DataFrame(horizon_rows)
                    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.7), squeeze=False)
                    for ax, dataset in zip(axes[0], datasets):
                        for model in ("quantum", "staeformer"):
                            part = horizon_frame[(horizon_frame.dataset == dataset) & (horizon_frame.model == model)]
                            means = [part[part.horizon == horizon].MAE.mean() for horizon in horizons]
                            errors = [part[part.horizon == horizon].MAE.std(ddof=1) for horizon in horizons]
                            ax.errorbar(horizons, means, yerr=np.nan_to_num(errors), marker=markers[model],
                                        label=LABELS[model], color=COLORS[model], capsize=3)
                        ax.set_title(dataset.replace("_", " ").upper())
                        ax.set_xlabel("Forecast horizon"); ax.set_ylabel("MAE (mph)")
                        ax.legend(frameon=False)
                    save(fig, output, "quarts_vs_staeformer_horizons")

    robustness_rows = []
    for path in runs.glob("*/*/seed_*/robustness.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for scenario, values in payload.items():
            robustness_rows.append({"dataset": path.parents[2].name, "model": path.parents[1].name, "scenario": scenario, **values})
    if robustness_rows:
        robust = pd.DataFrame(robustness_rows)
        for dataset in sorted(robust.dataset.unique()):
            part = robust[robust.dataset == dataset]
            scenarios = [x for x in sorted(part.scenario.unique()) if x == "clean" or x.startswith("missing_")]
            fig, ax = plt.subplots(figsize=(6.2, 3.7))
            for model in sorted(part.model.unique()):
                values = [part[(part.model == model) & (part.scenario == s)].MAE.mean() for s in scenarios]
                ax.plot(range(len(scenarios)), values, marker="o", label=LABELS.get(model, model), color=COLORS.get(model))
            ax.set_xticks(range(len(scenarios)), [s.replace("missing_", "") for s in scenarios])
            ax.set_xlabel("Missing-input rate (clean = 0)"); ax.set_ylabel("MAE (mph)"); ax.legend(frameon=False, ncol=2)
            save(fig, output, f"robustness_missing_{dataset}")

    ablation_rows = []
    for path in (ROOT / "artifacts/runs/ablation/metr_la").glob("q*_d*/seed_*/metrics.json"):
        model = path.parents[1].name
        q, d = model.split("_")
        values = json.loads(path.read_text(encoding="utf-8"))
        ablation_rows.append({"qubits": int(q[1:]), "depth": int(d[1:]), **values})
    if ablation_rows:
        table = pd.DataFrame(ablation_rows).pivot(index="qubits", columns="depth", values="MAE")
        fig, ax = plt.subplots(figsize=(4.8, 3.7)); image = ax.imshow(table.values, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(table.columns)), table.columns); ax.set_yticks(range(len(table.index)), table.index)
        ax.set_xlabel("Circuit depth"); ax.set_ylabel("Qubits")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]): ax.text(j, i, f"{table.iloc[i, j]:.3f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax, label="MAE (mph)"); save(fig, output, "quantum_ablation_heatmap")

    noise_rows = []
    for path in runs.glob("*/quantum/seed_*/quantum_noise.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))["results"]
        for scenario, values in payload.items(): noise_rows.append({"scenario": scenario, **values})
    if noise_rows:
        noise = pd.DataFrame(noise_rows).groupby("scenario", as_index=False).agg(MAE=("MAE", "mean"), SD=("MAE", "std"))
        fig, ax = plt.subplots(figsize=(7, 3.7)); ax.bar(range(len(noise)), noise.MAE, yerr=np.nan_to_num(noise.SD), color="#c0392b", capsize=3)
        ax.set_xticks(range(len(noise)), noise.scenario, rotation=35, ha="right"); ax.set_ylabel("MAE (mph)")
        save(fig, output, "quantum_noise_sensitivity")

    print(f"Figures written to {output}")


if __name__ == "__main__":
    main()
