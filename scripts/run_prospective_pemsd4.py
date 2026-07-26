"""Prospective PeMSD4 evaluation of frozen symmetric Fourier-RAC.

The backbone is trained only on PeMSD4 train windows. The uncertainty heads
use the first validation segment for early stopping and the next segment for
conformal calibration. The canonical test split is read only for final
evaluation after all parameters are frozen.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "STAEformer"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from model.STAEformer import STAEformer
from quarts.metrics import interval_metrics, mae, rmse, weighted_interval_score
from quarts.models import (
    FourierRegimeGate,
    MLPRegimeGate,
    RegimeIntervalCalibrator,
    masked_weighted_interval_score,
    parameter_count,
)
from quarts.reproducibility import save_manifest, set_seed
from run_quarts_rac import (
    ALPHAS,
    cache_split,
    conformal_factors,
    enriched_features,
    load_archive,
    make_dataset,
    physical_metrics,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone-epochs", type=int, default=12)
    parser.add_argument("--backbone-patience", type=int, default=4)
    parser.add_argument("--head-epochs", type=int, default=15)
    parser.add_argument("--head-patience", type=int, default=4)
    parser.add_argument("--backbone-batch-size", type=int, default=16)
    parser.add_argument("--head-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-windows", type=int, default=4000)
    parser.add_argument("--max-valid-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--max-train-instances", type=int, default=200000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="artifacts/runs/prospective_pemsd4")
    parser.add_argument("--cache", default="artifacts/cache/prospective_pemsd4")
    return parser.parse_args()


def deterministic_subset(loader, limit: int, batch_size: int):
    # Accept either a Dataset (used by the PeMSD4 loader) or an existing
    # DataLoader (keeps this helper compatible with the earlier runners).
    dataset = loader.dataset if hasattr(loader, "dataset") else loader
    if limit and limit < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, limit, dtype=int).tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_pemsd4():
    path = ROOT / "data" / "raw" / "pemsd4" / "PEMS04.npz"
    values = np.load(path)["data"].astype(np.float32)[..., 0]
    time_steps, nodes = values.shape
    tod = (np.arange(time_steps) % 288) / 288.0
    dow = ((np.arange(time_steps) // 288) % 7).astype(np.float32)
    train_end = int(time_steps * 0.7)
    valid_end = int(time_steps * 0.8)
    mean = float(values[:train_end].mean())
    std = float(values[:train_end].std())
    scaled = (values - mean) / max(std, 1e-6)
    temporal = np.zeros((time_steps, nodes, 3), dtype=np.float32)
    temporal[..., 0] = scaled
    temporal[..., 1] = tod[:, None]
    temporal[..., 2] = dow[:, None]
    input_steps, horizon = 12, 12

    def build(start, end):
        xs, ys = [], []
        for window_start in range(start, end - input_steps - horizon + 1):
            xs.append(temporal[window_start : window_start + input_steps])
            ys.append(
                scaled[
                    window_start + input_steps : window_start + input_steps + horizon
                ][..., None]
            )
        return TensorDataset(torch.from_numpy(np.asarray(xs)), torch.from_numpy(np.asarray(ys)))

    return (
        build(0, train_end),
        build(train_end, valid_end),
        build(valid_end, time_steps),
        mean,
        std,
        nodes,
    )


def load_adjacency(nodes: int) -> torch.Tensor:
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    with (ROOT / "data" / "raw" / "pemsd4" / "PEMS04.csv").open(
        encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source, target = int(row["from"]), int(row["to"])
            if source < nodes and target < nodes:
                adjacency[source, target] = 1.0
                adjacency[target, source] = 1.0
    adjacency += np.eye(nodes, dtype=np.float32)
    adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-6)
    return torch.from_numpy(adjacency)


@torch.no_grad()
def cache_pems_split(model, loader, mean, std, device, adjacency, projection, path):
    if path.exists():
        return
    captured = {}

    def capture(_module, inputs):
        captured["latent"] = inputs[0].detach()

    hook = model.output_proj.register_forward_pre_hook(capture)
    features, bases, targets, masks = [], [], [], []
    try:
        for x, y in loader:
            x_device = x.to(device)
            base = model(x_device).squeeze(-1)
            feature = enriched_features(
                x_device,
                base,
                captured["latent"],
                adjacency,
                projection,
            )
            features.append(feature.cpu().half().numpy())
            bases.append(base.cpu().half().numpy())
            targets.append(y.squeeze(-1).half().numpy())
            masks.append(np.ones_like(y.squeeze(-1).numpy(), dtype=bool))
    finally:
        hook.remove()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.concatenate(features),
        base=np.concatenate(bases),
        target=np.concatenate(targets),
        mask=np.concatenate(masks),
        mean=np.asarray(mean),
        std=np.asarray(std),
    )


def train_backbone(train, valid, nodes, mean, std, args, checkpoint):
    device = torch.device(args.device)
    model = STAEformer(
        num_nodes=nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=3e-4)
    loader = deterministic_subset(train, args.max_train_windows, args.backbone_batch_size)
    valid_loader = deterministic_subset(valid, args.max_valid_windows, args.backbone_batch_size)
    best_state, best_loss, stale = None, float("inf"), 0
    history = []
    for epoch in range(1, args.backbone_epochs + 1):
        model.train()
        train_losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device).squeeze(-1)
            prediction = model(x).squeeze(-1)
            loss = (prediction - y).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            valid_losses = [
                float((model(x.to(device)).squeeze(-1) - y.to(device).squeeze(-1)).abs().mean())
                for x, y in valid_loader
            ]
        record = {
            "epoch": epoch,
            "train_MAE_normalized": float(np.mean(train_losses)),
            "valid_MAE_normalized": float(np.mean(valid_losses)),
        }
        history.append(record)
        print(f"pemsd4 backbone {json.dumps(record)}", flush=True)
        if record["valid_MAE_normalized"] < best_loss - 1e-4:
            best_loss = record["valid_MAE_normalized"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.backbone_patience:
                break
    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    return model, history


def train_head(
    name,
    seed,
    train_set,
    selection_set,
    calibration_set,
    test_set,
    mean,
    std,
    args,
):
    set_seed(seed)
    device = torch.device(args.device)
    hidden, horizons = train_set.tensors[0].shape[-1], train_set.tensors[1].shape[-1]
    if name == "constant":
        head = RegimeIntervalCalibrator(None, horizons, len(ALPHAS), experts=1)
    elif name == "mlp":
        head = RegimeIntervalCalibrator(
            MLPRegimeGate(hidden, experts=3), horizons, len(ALPHAS), experts=3
        )
    else:
        head = RegimeIntervalCalibrator(
            FourierRegimeGate(hidden, experts=3), horizons, len(ALPHAS), experts=3
        )
    head = head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-3)
    loader = DataLoader(
        train_set,
        batch_size=args.head_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state, best_loss, stale = copy.deepcopy(head.state_dict()), float("inf"), 0
    history = []
    for epoch in range(1, args.head_epochs + 1):
        head.train()
        for feature, base, target, mask in loader:
            feature, base, target, mask = (
                feature.to(device),
                base.to(device),
                target.to(device),
                mask.to(device),
            )
            radii, _ = head(feature[:, None, :])
            loss = masked_weighted_interval_score(
                target, base, radii.squeeze(2), mask, ALPHAS
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        selection_wis, _, _, _ = predict_head(
            head, selection_set, args.head_batch_size, device
        )
        history.append({"epoch": epoch, "selection_WIS_normalized": selection_wis})
        print(f"pemsd4 head={name} seed={seed} epoch={epoch} selection_WIS={selection_wis}", flush=True)
        if selection_wis < best_loss - 1e-5:
            best_loss = selection_wis
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.head_patience:
                break
    head.load_state_dict(best_state)
    _, cal_radii, _, _ = predict_head(head, calibration_set, args.head_batch_size, device)
    factors = conformal_factors(calibration_set, cal_radii)
    _, test_radii, _, weights = predict_head(head, test_set, args.head_batch_size, device)
    report = physical_metrics(test_set, test_radii, factors, mean, std)
    report.update(
        {
            "study": "Prospective PeMSD4 Fourier-RAC evaluation",
            "dataset": "PeMSD4",
            "model": name,
            "head_seed": seed,
            "canonical_test_split_used": True,
            "test_used_for_selection": False,
            "point_forecast_modified": False,
            "calibrator_parameters": parameter_count(head),
            "coverage_calibration_factors": factors.tolist(),
            "mean_expert_weights": weights.mean(axis=0).tolist(),
            "history": history,
        }
    )
    run_dir = ROOT / args.output / "heads" / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), run_dir / "best.pt")
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@torch.no_grad()
def predict_head(head, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    radii, weights = [], []
    total, count = 0.0, 0
    head.eval()
    for feature, base, target, mask in loader:
        feature, base, target, mask = (
            feature.to(device),
            base.to(device),
            target.to(device),
            mask.to(device),
        )
        predicted, gates = head(feature[:, None, :])
        predicted = predicted.squeeze(2)
        loss = masked_weighted_interval_score(target, base, predicted, mask, ALPHAS)
        valid = int(mask.sum())
        total += float(loss) * valid
        count += valid
        radii.append(predicted.cpu().numpy())
        weights.append(gates.squeeze(1).cpu().numpy())
    return total / max(count, 1), np.concatenate(radii), np.concatenate(radii), np.concatenate(weights)


def main():
    args = arguments()
    set_seed(args.seed)
    train, valid, test, mean, std, nodes = load_pemsd4()
    output = ROOT / args.output
    checkpoint = output / f"backbone_seed_{args.seed}" / "best.pt"
    device = torch.device(args.device)
    if checkpoint.exists():
        model = STAEformer(
            num_nodes=nodes,
            in_steps=12,
            out_steps=12,
            steps_per_day=288,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=80,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        history = []
    else:
        model, history = train_backbone(train, valid, nodes, mean, std, args, checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adjacency = load_adjacency(nodes).to(device)
    projection = torch.randn(
        model.in_steps * model.model_dim,
        8,
        generator=torch.Generator().manual_seed(20260723),
    ).to(device)
    projection /= (model.in_steps * model.model_dim) ** 0.5
    cache = ROOT / args.cache / f"pemsd4_seed_{args.seed}"
    loaders = {
        "train": deterministic_subset(train, args.max_train_windows, args.backbone_batch_size),
        "valid": deterministic_subset(valid, args.max_valid_windows, args.backbone_batch_size),
        "test": deterministic_subset(test, args.max_test_windows, args.backbone_batch_size),
    }
    for split, loader in loaders.items():
        cache_pems_split(
            model,
            loader,
            mean,
            std,
            device,
            adjacency,
            projection,
            cache / f"{split}.npz",
        )
        print(f"pemsd4 cache {split} complete", flush=True)
    train_archive, valid_archive, test_archive = (
        load_archive(cache / f"{split}.npz") for split in ("train", "valid", "test")
    )
    feature_mean = train_archive["features"].mean(axis=(0, 1), keepdims=True)
    feature_std = np.maximum(train_archive["features"].std(axis=(0, 1), keepdims=True), 1e-4)
    train_set = make_dataset(train_archive, feature_mean, feature_std, max_instances=args.max_train_instances, sample_seed=42)
    valid_windows = valid_archive["features"].shape[0]
    early_end, calibration_end = max(1, int(valid_windows * 0.4)), max(2, int(valid_windows * 0.7))
    selection_set = make_dataset(valid_archive, feature_mean, feature_std, slice(0, early_end))
    calibration_set = make_dataset(valid_archive, feature_mean, feature_std, slice(early_end, calibration_end))
    test_set = make_dataset(test_archive, feature_mean, feature_std)
    reports = []
    for name in ("constant", "mlp", "fourier"):
        reports.append(
            train_head(
                name,
                args.seed,
                train_set,
                selection_set,
                calibration_set,
                test_set,
                mean,
                std,
                args,
            )
        )
    summary = {
        "study": "Prospective PeMSD4 Fourier-RAC evaluation",
        "complete": True,
        "dataset": "PeMSD4",
        "canonical_test_split_used": True,
        "test_used_for_selection": False,
        "backbone_history": history,
        "models": reports,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_manifest(
        output / "manifest.json",
        {
            "study": "Prospective PeMSD4 Fourier-RAC evaluation",
            "dataset": "PeMSD4",
            "source_record": "Zenodo 7816008",
            "canonical_test_split_used": True,
            "test_used_for_selection": False,
            "arguments": vars(args),
        },
        [
            ROOT / "data" / "raw" / "pemsd4" / "PEMS04.npz",
            ROOT / "data" / "raw" / "pemsd4" / "PEMS04.csv",
            ROOT / "configs" / "prospective_pemsd4_fourier_rac.yaml",
        ],
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
