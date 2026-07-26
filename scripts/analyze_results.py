"""Aggregate confirmatory artifacts and run paired sensor-day statistics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

from quarts.metrics import interval_metrics
from quarts.statistics import cliffs_delta, paired_block_bootstrap, wilcoxon_paired


def sensor_day_errors(path: Path, steps_per_day=288):
    archive = np.load(path)
    error = np.abs(archive["target"] - archive["location"])
    mask = archive["mask"].astype(bool)
    # [sample,horizon,node] -> [day,node], averaging all within-day forecasts.
    sample_count, _, nodes = error.shape
    days = np.arange(sample_count) // steps_per_day
    values, block_labels = [], []
    for day in np.unique(days):
        chosen = days == day
        for node in range(nodes):
            valid = mask[chosen, :, node]
            values.append(np.mean(error[chosen, :, node][valid]) if valid.any() else np.nan)
            block_labels.append(int(day))
    return np.asarray(values), np.asarray(block_labels)


def holm_adjust(pvalues):
    """Holm step-down adjusted p-values in original order."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="artifacts/runs/confirmatory")
    parser.add_argument("--output", default="artifacts/analysis")
    args = parser.parse_args()
    runs, rows = ROOT / args.runs, []
    for path in runs.glob("*/*/seed_*/metrics.json"):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "dataset": path.parents[2].name,
            "model": path.parents[1].name,
            "seed": int(path.parent.name.split("_")[-1]),
            **{key: metrics.get(key) for key in ("MAE", "RMSE", "MAPE", "WAPE", "Laplace_NLL", "parameters", "total_seconds", "inference_seconds")},
        }
        prediction_path = path.parent / "predictions.npz"
        if prediction_path.exists():
            archive = np.load(prediction_path)
            if "scale" in archive.files:
                q = np.log(10.0) * archive["scale"]
                intervals = interval_metrics(
                    archive["target"], archive["location"] - q, archive["location"] + q,
                    archive["mask"], alpha=0.1,
                )
                row.update({f"interval90_{key}": value for key, value in intervals.items()})
                row["interval90_calibration_error"] = abs(intervals["coverage"] - 0.9)
        rows.append(row)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "run_metrics.csv", index=False)
    if frame.empty:
        print("No completed runs")
        return
    summary = frame.groupby(["dataset", "model"]).agg(["mean", "std", "median"])
    summary.to_csv(output / "summary.csv")
    tests, omnibus = [], []
    for dataset in sorted(frame.dataset.unique()):
        model_errors = {}
        for model in sorted(frame[frame.dataset == dataset].model.unique()):
            pairs = [sensor_day_errors(path) for path in sorted((runs / dataset / model).glob("seed_*/predictions.npz"))]
            arrays = [pair[0] for pair in pairs]
            if arrays:
                model_errors[model] = np.nanmean(np.stack(arrays), axis=0)
        learned = frame[frame.dataset == dataset]
        complete_models = learned.groupby("model").seed.nunique()
        complete_models = complete_models[complete_models >= 5].index
        metric_wide = learned[learned.model.isin(complete_models)].pivot(index="seed", columns="model", values="MAE").dropna(axis=0)
        if metric_wide.shape[0] >= 3 and metric_wide.shape[1] >= 3:
            result = friedmanchisquare(*(metric_wide[column].values for column in metric_wide.columns))
            omnibus.append({"dataset": dataset, "metric": "MAE", "models": ";".join(metric_wide.columns), "blocks": len(metric_wide), "statistic": float(result.statistic), "pvalue": float(result.pvalue)})
        if "quantum" not in model_errors:
            continue
        for comparator, other in model_errors.items():
            if comparator == "quantum":
                continue
            valid = np.isfinite(model_errors["quantum"]) & np.isfinite(other)
            q, c = model_errors["quantum"][valid], other[valid]
            # Values are ordered day-major then sensor; resample 24-hour blocks.
            nodes = 207 if dataset == "metr_la" else 325
            blocks = np.arange(len(q)) // nodes
            tests.append({
                "dataset": dataset,
                "comparison": f"quantum-vs-{comparator}",
                **wilcoxon_paired(q, c),
                "cliffs_delta": cliffs_delta(q, c),
                **paired_block_bootstrap(q, c, blocks, iterations=5000, seed=42),
            })
    tests_frame = pd.DataFrame(tests)
    if not tests_frame.empty:
        for dataset in tests_frame.dataset.unique():
            selected = tests_frame.dataset == dataset
            tests_frame.loc[selected, "pvalue_holm"] = holm_adjust(tests_frame.loc[selected, "pvalue"].values)
        tests_frame["reject_holm_0.05"] = tests_frame["pvalue_holm"] < 0.05
    tests_frame.to_csv(output / "paired_tests.csv", index=False)
    pd.DataFrame(omnibus).to_csv(output / "friedman_tests.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
