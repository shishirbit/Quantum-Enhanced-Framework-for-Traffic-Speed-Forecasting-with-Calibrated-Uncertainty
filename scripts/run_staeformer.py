"""Reproducible wrapper around the verified official STAEformer implementation."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "STAEformer"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml

from lib.data_prepare import get_dataloaders_from_index_data
from lib.utils import MaskedMAELoss
from model.STAEformer import STAEformer
from quarts.metrics import mae, mape, rmse, wape
from quarts.models import parameter_count
from quarts.reproducibility import save_manifest, set_seed


@torch.no_grad()
def predict(model, loader, scaler):
    model.eval()
    targets, predictions = [], []
    start = time.perf_counter()
    for x, y in loader:
        out = scaler.inverse_transform(model(x.cuda()))
        targets.append(y.numpy().squeeze(-1))
        predictions.append(out.cpu().numpy().squeeze(-1))
    return np.concatenate(targets), np.concatenate(predictions), time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["METRLA", "PEMSBAY"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    set_seed(args.seed)
    with (VENDOR / "model" / "STAEformer.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)[args.dataset]
    train, valid, test, scaler = get_dataloaders_from_index_data(
        VENDOR / "data" / args.dataset,
        tod=cfg.get("time_of_day"), dow=cfg.get("day_of_week"), batch_size=cfg["batch_size"]
    )
    model = STAEformer(**cfg["model_args"]).cuda()
    criterion = MaskedMAELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["milestones"], gamma=cfg["lr_decay_rate"])
    best, stale, best_state, history = float("inf"), 0, None, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, y in train:
            optimizer.zero_grad(set_to_none=True)
            out = scaler.inverse_transform(model(x.cuda()))
            loss = criterion(out, y.cuda())
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        scheduler.step()
        model.eval(); validation = []
        with torch.no_grad():
            for x, y in valid:
                validation.append(float(criterion(scaler.inverse_transform(model(x.cuda())), y.cuda())))
        record = {"epoch": epoch, "train_mae_loss": float(np.mean(losses)), "valid_mae_loss": float(np.mean(validation))}
        history.append(record); print(json.dumps(record), flush=True)
        if record["valid_mae_loss"] < best - 1e-4:
            best, stale, best_state = record["valid_mae_loss"], 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    target, prediction, inference = predict(model, test, scaler)
    mask = target != 0
    horizons = {}
    for label, index in {"15min": 2, "30min": 5, "60min": 11}.items():
        horizons[label] = {
            "MAE": mae(target[:, index], prediction[:, index], mask[:, index]),
            "RMSE": rmse(target[:, index], prediction[:, index], mask[:, index]),
            "MAPE": mape(target[:, index], prediction[:, index], mask[:, index]),
        }
    report = {
        "calibrator": "STAEformer",
        "seed": args.seed,
        "parameters": parameter_count(model),
        "MAE": mae(target, prediction, mask), "RMSE": rmse(target, prediction, mask),
        "MAPE": mape(target, prediction, mask), "WAPE": wape(target, prediction, mask),
        "Laplace_NLL": None, "inference_seconds": inference,
        "total_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "horizons": horizons, "history": history,
    }
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / "best.pt")
    np.savez_compressed(output / "predictions.npz", target=target, location=prediction, mask=mask)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_manifest(output / "manifest.json", {"dataset": args.dataset, "seed": args.seed, "official_config": cfg})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

