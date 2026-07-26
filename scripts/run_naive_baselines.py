"""Canonical persistence and time-of-day historical-average baselines."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from quarts.data import chronological_boundaries, load_timeseries
from quarts.metrics import mae, mape, rmse, wape
from quarts.reproducibility import save_manifest

DATASETS = {"metr_la": "data/raw/dcrnn/metr-la.h5", "pems_bay": "data/raw/dcrnn/pems-bay.h5"}


def build_arrays(data: np.ndarray, input_steps=12, horizons=12):
    bounds = chronological_boundaries(len(data))
    train = data[bounds[0][0]:bounds[0][1], :, 0]
    test_start, test_end = bounds[2]
    test = data[test_start:test_end, :, 0]
    count = len(test) - input_steps - horizons + 1
    target = np.stack([test[input_steps + h:input_steps + h + count] for h in range(horizons)], axis=1)
    mask = np.isfinite(target) & (target != 0)

    latest = np.zeros((count, data.shape[1]), dtype=np.float32)
    seen = np.zeros_like(latest, dtype=bool)
    for lag in range(input_steps):
        values = test[lag:lag + count]
        valid = np.isfinite(values) & (values != 0)
        latest[valid] = values[valid]
        seen |= valid
    sensor_fallback = np.nanmean(np.where((train != 0) & np.isfinite(train), train, np.nan), axis=0)
    latest = np.where(seen, latest, sensor_fallback)
    persistence = np.repeat(latest[:, None, :], horizons, axis=1)

    slots = np.empty((288, data.shape[1]), dtype=np.float32)
    for slot in range(288):
        values = train[np.arange(slot, len(train), 288)]
        slots[slot] = np.nanmean(np.where((values != 0) & np.isfinite(values), values, np.nan), axis=0)
    slots = np.where(np.isfinite(slots), slots, sensor_fallback[None])
    sample = np.arange(count)[:, None]
    horizon = np.arange(horizons)[None, :]
    target_slots = (test_start + sample + input_steps + horizon) % 288
    historical = slots[target_slots]
    return target.astype(np.float32), mask, {"persistence": persistence, "historical_average": historical.astype(np.float32)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output", default="artifacts/runs/confirmatory")
    args = parser.parse_args()
    for dataset in args.datasets:
        source = ROOT / DATASETS[dataset]
        started = time.perf_counter()
        target, mask, predictions = build_arrays(load_timeseries(source))
        for model, prediction in predictions.items():
            run = ROOT / args.output / dataset / model / "seed_0"
            required = (run / "metrics.json", run / "predictions.npz", run / "manifest.json")
            if all(path.exists() for path in required):
                continue
            run.mkdir(parents=True, exist_ok=True)
            report = {
                "calibrator": model, "seed": 0, "parameters": 0,
                "MAE": mae(target, prediction, mask), "RMSE": rmse(target, prediction, mask),
                "MAPE": mape(target, prediction, mask), "WAPE": wape(target, prediction, mask),
                "Laplace_NLL": None, "inference_seconds": 0.0,
                "total_seconds": time.perf_counter() - started,
                "peak_gpu_memory_bytes": 0,
                "horizons": {
                    label: {"MAE": mae(target[:, index], prediction[:, index], mask[:, index]),
                            "RMSE": rmse(target[:, index], prediction[:, index], mask[:, index]),
                            "MAPE": mape(target[:, index], prediction[:, index], mask[:, index])}
                    for label, index in {"15min": 2, "30min": 5, "60min": 11}.items()
                },
            }
            np.savez_compressed(run / "predictions.npz", target=target, location=prediction, mask=mask)
            (run / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            save_manifest(run / "manifest.json", {"dataset": dataset, "model": model, "deterministic": True}, [source])
            print(f"DONE {dataset}/{model}: {json.dumps(report)}", flush=True)


if __name__ == "__main__":
    main()
