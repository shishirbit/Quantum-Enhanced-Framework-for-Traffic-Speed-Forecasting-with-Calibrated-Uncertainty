"""Run the validation-only QUARTS-RAC exploratory pilot.

The frozen STAEformer point forecast is unchanged. Learned heads only select
among a shared bank of ordered central-interval experts. The canonical test
split is deliberately never loaded by this script.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
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
from torch.utils.data import DataLoader, Subset, TensorDataset

from lib.data_prepare import get_dataloaders_from_index_data
from model.STAEformer import STAEformer
from quarts.metrics import interval_metrics, mae, weighted_interval_score
from quarts.models import (
    FourierRegimeGate,
    MLPRegimeGate,
    QuantumRegimeGate,
    RegimeIntervalCalibrator,
    masked_weighted_interval_score,
    parameter_count,
)
from quarts.reproducibility import save_manifest, set_seed

ALPHAS = (0.1, 0.2, 0.5)
FEATURE_NAMES = (
    [f"base_h{index + 1}" for index in range(12)]
    + ["local_last", "local_mean", "local_std", "local_trend"]
    + ["neighbor_last", "neighbor_mean", "neighbor_std", "neighbor_trend"]
    + ["disagreement_last", "disagreement_mean", "disagreement_std", "disagreement_trend"]
    + [
        "network_mean_last",
        "network_mean_mean",
        "network_mean_std",
        "network_mean_trend",
        "network_std_last",
        "network_std_mean",
        "network_std_std",
        "network_std_trend",
    ]
    + ["tod_sin", "tod_cos", "dow_sin", "dow_cos"]
    + [f"latent_projection_{index + 1}" for index in range(8)]
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["METRLA"], default="METRLA")
    parser.add_argument("--backbone-seed", type=int, default=42)
    parser.add_argument("--head-seeds", nargs="+", type=int, default=[42, 73, 202])
    parser.add_argument(
        "--calibrators",
        nargs="+",
        default=["constant", "mlp", "fourier", "separable", "quantum"],
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-train-windows", type=int, default=2000)
    parser.add_argument("--max-valid-windows", type=int, default=1000)
    parser.add_argument("--max-train-instances", type=int, default=200000)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--cache", default="artifacts/cache/quarts_rac")
    parser.add_argument("--output", default="artifacts/runs/exploratory_rac")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def deterministic_subset(loader: DataLoader, limit: int, batch_size: int) -> DataLoader:
    dataset = loader.dataset
    if limit and limit < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, limit, dtype=int).tolist()
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_adjacency() -> torch.Tensor:
    path = ROOT / "data" / "raw" / "dcrnn" / "adj_mx.pkl"
    with path.open("rb") as handle:
        _, _, adjacency = pickle.load(handle, encoding="latin1")
    adjacency = np.asarray(adjacency, dtype=np.float32)
    adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-6)
    return torch.from_numpy(adjacency)


def enriched_features(
    x: torch.Tensor,
    base: torch.Tensor,
    latent: torch.Tensor,
    adjacency: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Create interaction-rich, network-size-independent sensor features."""

    recent = x[..., 0].float().permute(0, 2, 1)
    local = torch.stack(
        [
            recent[..., -1],
            recent.mean(-1),
            recent.std(-1),
            recent[..., -1] - recent[..., 0],
        ],
        dim=-1,
    )
    neighbor = torch.einsum("nm,bmf->bnf", adjacency, local)
    disagreement = local - neighbor
    network_mean = local.mean(1, keepdim=True).expand_as(local)
    network_std = local.std(1, keepdim=True).expand_as(local)
    tod = 2 * torch.pi * x[:, -1, :, 1]
    dow = 2 * torch.pi * x[:, -1, :, 2] / 7.0
    cyclical = torch.stack(
        [torch.sin(tod), torch.cos(tod), torch.sin(dow), torch.cos(dow)],
        dim=-1,
    )
    projected = latent @ projection
    return torch.cat(
        [
            base.permute(0, 2, 1),
            local,
            neighbor,
            disagreement,
            network_mean,
            network_std,
            cyclical,
            projected,
        ],
        dim=-1,
    )


@torch.no_grad()
def cache_split(
    model: STAEformer,
    loader: DataLoader,
    scaler,
    device: torch.device,
    adjacency: torch.Tensor,
    projection: torch.Tensor,
    path: Path,
) -> None:
    if path.exists():
        return
    captured: dict[str, torch.Tensor] = {}

    def capture_projection_input(_module, inputs):
        captured["latent"] = inputs[0].detach()

    hook = model.output_proj.register_forward_pre_hook(capture_projection_input)
    features, bases, targets, masks = [], [], [], []
    try:
        for batch_index, (x, y) in enumerate(loader, start=1):
            x_device = x.to(device)
            base = model(x_device).squeeze(-1)
            latent = captured["latent"]
            feature = enriched_features(
                x_device,
                base,
                latent,
                adjacency,
                projection,
            )
            target = ((y.squeeze(-1) - scaler.mean) / scaler.std).float()
            features.append(feature.cpu().half().numpy())
            bases.append(base.cpu().half().numpy())
            targets.append(target.half().numpy())
            masks.append((y.squeeze(-1) != 0).numpy())
            if batch_index % 10 == 0:
                print(
                    f"CACHE {path.stem}: {batch_index}/{len(loader)} batches",
                    flush=True,
                )
    finally:
        hook.remove()
    if not features:
        raise RuntimeError(f"No batches were available for {path.stem}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.concatenate(features),
        base=np.concatenate(bases),
        target=np.concatenate(targets),
        mask=np.concatenate(masks),
        mean=np.asarray(scaler.mean),
        std=np.asarray(scaler.std),
        feature_names=np.asarray(FEATURE_NAMES),
    )


def load_archive(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path) as archive:
        return {
            "features": archive["features"].astype(np.float32),
            "base": archive["base"].astype(np.float32),
            "target": archive["target"].astype(np.float32),
            "mask": archive["mask"].astype(bool),
            "mean": float(archive["mean"].item()),
            "std": float(archive["std"].item()),
        }


def make_dataset(
    archive: dict[str, np.ndarray | float],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    window_slice: slice | None = None,
    max_instances: int = 0,
    sample_seed: int = 42,
) -> TensorDataset:
    selection = window_slice if window_slice is not None else slice(None)
    features = archive["features"][selection]
    base = archive["base"][selection]
    target = archive["target"][selection]
    mask = archive["mask"][selection]
    features = (features - feature_mean) / feature_std
    features = features.reshape(-1, features.shape[-1])
    base = base.transpose(0, 2, 1).reshape(-1, base.shape[1])
    target = target.transpose(0, 2, 1).reshape(-1, target.shape[1])
    mask = mask.transpose(0, 2, 1).reshape(-1, mask.shape[1])
    if max_instances and len(features) > max_instances:
        rng = np.random.default_rng(sample_seed)
        selected = np.sort(rng.choice(len(features), max_instances, replace=False))
        features, base, target, mask = (
            features[selected],
            base[selected],
            target[selected],
            mask[selected],
        )
    return TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(base.astype(np.float32)),
        torch.from_numpy(target.astype(np.float32)),
        torch.from_numpy(mask),
    )


def build_head(name: str, hidden: int, horizons: int) -> RegimeIntervalCalibrator:
    if name == "constant":
        return RegimeIntervalCalibrator(None, horizons, len(ALPHAS), experts=1)
    if name == "mlp":
        gate = MLPRegimeGate(hidden, experts=3, width=5)
    elif name == "fourier":
        gate = FourierRegimeGate(hidden, experts=3, width=3, depth=2)
    elif name in {"separable", "quantum"}:
        gate = QuantumRegimeGate(
            hidden,
            experts=3,
            qubits=4,
            depth=2,
            entangled=name == "quantum",
        )
    else:
        raise ValueError(f"Unknown calibrator: {name}")
    return RegimeIntervalCalibrator(gate, horizons, len(ALPHAS), experts=3)


def empirical_radii(dataset: TensorDataset) -> np.ndarray:
    _, base, target, mask = dataset.tensors
    residual = np.abs(target.numpy() - base.numpy())
    valid = mask.numpy()
    radii = np.empty((base.shape[1], len(ALPHAS)), dtype=np.float32)
    for horizon in range(base.shape[1]):
        values = residual[:, horizon][valid[:, horizon]]
        for level, alpha in enumerate(ALPHAS):
            radii[horizon, level] = np.quantile(
                values,
                1 - alpha,
                method="higher",
            )
    return np.maximum(radii, 1e-3)


def initialize_experts(
    head: RegimeIntervalCalibrator,
    radii: np.ndarray,
) -> None:
    increments = np.empty_like(radii)
    increments[:, :-1] = radii[:, :-1] - radii[:, 1:]
    increments[:, -1] = radii[:, -1]
    scales = np.linspace(0.75, 1.30, head.experts, dtype=np.float32)
    values = scales[:, None, None] * increments[None, :, :]
    values = np.maximum(values - 1e-4, 1e-4)
    raw = np.log(np.expm1(values))
    with torch.no_grad():
        head.raw_increments.copy_(torch.from_numpy(raw))


@torch.no_grad()
def predict(
    head: RegimeIntervalCalibrator,
    dataset: TensorDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total, count = 0.0, 0
    all_radii, all_weights = [], []
    head.eval()
    for feature, base, target, mask in loader:
        feature = feature.to(device)
        base = base.to(device)
        target = target.to(device)
        mask = mask.to(device)
        radii, weights = head(feature[:, None, :])
        radii = radii.squeeze(2)
        loss = masked_weighted_interval_score(target, base, radii, mask, ALPHAS)
        valid = int(mask.sum())
        total += float(loss) * valid
        count += valid
        all_radii.append(radii.cpu().numpy())
        all_weights.append(weights.squeeze(1).cpu().numpy())
    return total / max(count, 1), np.concatenate(all_radii), np.concatenate(all_weights)


def conformal_factors(
    dataset: TensorDataset,
    radii: np.ndarray,
) -> np.ndarray:
    _, base, target, mask = (tensor.numpy() for tensor in dataset.tensors)
    residual = np.abs(target - base)
    factors = np.ones((base.shape[1], len(ALPHAS)), dtype=np.float32)
    for horizon in range(base.shape[1]):
        valid = mask[:, horizon]
        for level, alpha in enumerate(ALPHAS):
            scores = residual[valid, horizon] / np.maximum(
                radii[valid, horizon, level],
                1e-6,
            )
            probability = min(
                1.0,
                math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores),
            )
            factors[horizon, level] = np.quantile(
                scores,
                probability,
                method="higher",
            )
    return factors


def physical_metrics(
    dataset: TensorDataset,
    radii: np.ndarray,
    factors: np.ndarray,
    mean: float,
    std: float,
) -> dict:
    _, base, target, mask = (tensor.numpy() for tensor in dataset.tensors)
    location = base * std + mean
    observed = target * std + mean
    calibrated = radii * factors[None, :, :] * std
    intervals = {}
    detail = {}
    for level, alpha in enumerate(ALPHAS):
        radius = calibrated[..., level]
        lower, upper = location - radius, location + radius
        intervals[alpha] = (lower, upper)
        detail[str(alpha)] = interval_metrics(
            observed,
            lower,
            upper,
            mask,
            alpha=alpha,
        )
    uncalibrated = {
        alpha: (
            location - radii[..., level] * std,
            location + radii[..., level] * std,
        )
        for level, alpha in enumerate(ALPHAS)
    }
    return {
        "MAE": mae(observed, location, mask),
        "conformal_WIS": weighted_interval_score(
            observed,
            location,
            intervals,
            mask,
        ),
        "uncalibrated_WIS": weighted_interval_score(
            observed,
            location,
            uncalibrated,
            mask,
        ),
        "interval90_coverage": detail["0.1"]["coverage"],
        "interval90_mean_width": detail["0.1"]["mean_width"],
        "interval90_interval_score": detail["0.1"]["interval_score"],
        "interval90_calibration_error": abs(detail["0.1"]["coverage"] - 0.9),
        "intervals": detail,
    }


def train_one(
    name: str,
    head_seed: int,
    train_set: TensorDataset,
    selection_set: TensorDataset,
    calibration_set: TensorDataset,
    evaluation_set: TensorDataset,
    mean: float,
    std: float,
    args: argparse.Namespace,
) -> dict:
    set_seed(head_seed)
    device = torch.device(args.device)
    hidden = train_set.tensors[0].shape[-1]
    horizons = train_set.tensors[1].shape[-1]
    head = build_head(name, hidden, horizons).to(device)
    initialize_experts(head, empirical_radii(train_set))
    optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(head_seed)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    best_state = copy.deepcopy(head.state_dict())
    best_selection = math.inf
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        head.train()
        total, count = 0.0, 0
        for feature, base, target, mask in loader:
            feature = feature.to(device)
            base = base.to(device)
            target = target.to(device)
            mask = mask.to(device)
            radii, _ = head(feature[:, None, :])
            loss = masked_weighted_interval_score(
                target,
                base,
                radii.squeeze(2),
                mask,
                ALPHAS,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            valid = int(mask.sum())
            total += float(loss.detach()) * valid
            count += valid
        selection_wis, _, _ = predict(
            head,
            selection_set,
            args.batch_size,
            device,
        )
        record = {
            "epoch": epoch,
            "train_WIS_normalized": total / max(count, 1),
            "selection_WIS_normalized": selection_wis,
        }
        history.append(record)
        print(
            f"metr_la seed={head_seed} {name} {json.dumps(record)}",
            flush=True,
        )
        if selection_wis < best_selection - 1e-5:
            best_selection = selection_wis
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    head.load_state_dict(best_state)
    _, calibration_radii, _ = predict(
        head,
        calibration_set,
        args.batch_size,
        device,
    )
    factors = conformal_factors(calibration_set, calibration_radii)
    _, evaluation_radii, weights = predict(
        head,
        evaluation_set,
        args.batch_size,
        device,
    )
    report = physical_metrics(
        evaluation_set,
        evaluation_radii,
        factors,
        mean,
        std,
    )
    average_weights = weights.mean(axis=0)
    entropy = -np.sum(
        average_weights * np.log(np.maximum(average_weights, 1e-12))
    )
    report.update(
        {
            "study": "QUARTS-RAC validation-only exploratory pilot",
            "dataset": "metr_la",
            "backbone_seed": args.backbone_seed,
            "head_seed": head_seed,
            "model": name,
            "canonical_test_split_used": False,
            "point_forecast_modified": False,
            "calibrator_parameters": parameter_count(head),
            "quantum_backend": (
                type(head.gate.quantum).__name__
                if name in {"separable", "quantum"}
                else None
            ),
            "conformal_factors": factors.tolist(),
            "mean_expert_weights": average_weights.tolist(),
            "effective_experts": float(np.exp(entropy)),
            "mean_max_gate_probability": float(weights.max(axis=1).mean()),
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    run_dir = (
        ROOT
        / args.output
        / "metr_la"
        / f"backbone_seed_{args.backbone_seed}"
        / name
        / f"head_seed_{head_seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), run_dir / "best.pt")
    np.savez_compressed(
        run_dir / "evaluation_radii.npz",
        radii=evaluation_radii.astype(np.float16),
        factors=factors,
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "model",
                    "head_seed",
                    "conformal_WIS",
                    "uncalibrated_WIS",
                    "interval90_coverage",
                    "interval90_interval_score",
                    "effective_experts",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def aggregate(reports: dict[str, list[dict]], args: argparse.Namespace) -> dict:
    metrics = [
        "conformal_WIS",
        "uncalibrated_WIS",
        "interval90_coverage",
        "interval90_mean_width",
        "interval90_interval_score",
        "interval90_calibration_error",
        "effective_experts",
    ]
    models = []
    for name in args.calibrators:
        item = {"model": name, "runs": len(reports[name])}
        for metric in metrics:
            values = np.asarray([run[metric] for run in reports[name]])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std())
        models.append(item)
    return {
        "study": "QUARTS-RAC validation-only exploratory pilot",
        "complete": True,
        "dataset": "metr_la",
        "backbone_seed": args.backbone_seed,
        "head_seeds": args.head_seeds,
        "canonical_test_split_used": False,
        "validation_split": {
            "early_stopping": 0.4,
            "conformal_calibration": 0.3,
            "held_out_evaluation": 0.3,
        },
        "models": models,
    }


def main() -> None:
    args = arguments()
    if args.latent_dim != 8:
        raise ValueError("The frozen feature specification requires --latent-dim 8")
    set_seed(args.backbone_seed)
    device = torch.device(args.device)
    with (VENDOR / "model" / "STAEformer.yaml").open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)[args.dataset]
    train, valid, _test, scaler = get_dataloaders_from_index_data(
        VENDOR / "data" / args.dataset,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg["batch_size"],
    )
    del _test
    checkpoint = (
        ROOT
        / "artifacts"
        / "runs"
        / "confirmatory"
        / "metr_la"
        / "staeformer"
        / f"seed_{args.backbone_seed}"
        / "best.pt"
    )
    model = STAEformer(**cfg["model_args"]).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adjacency = load_adjacency().to(device)
    projection_generator = torch.Generator().manual_seed(20260723)
    projection = torch.randn(
        model.in_steps * model.model_dim,
        args.latent_dim,
        generator=projection_generator,
    )
    projection /= math.sqrt(model.in_steps * model.model_dim)
    projection = projection.to(device)
    signature = f"tw_{args.max_train_windows}_vw_{args.max_valid_windows}_ld_8"
    cache_root = (
        ROOT
        / args.cache
        / f"metr_la_seed_{args.backbone_seed}"
        / signature
    )
    loaders = {
        "train": deterministic_subset(
            train,
            args.max_train_windows,
            cfg["batch_size"],
        ),
        "valid": deterministic_subset(
            valid,
            args.max_valid_windows,
            cfg["batch_size"],
        ),
    }
    for split, loader in loaders.items():
        cache_split(
            model,
            loader,
            scaler,
            device,
            adjacency,
            projection,
            cache_root / f"{split}.npz",
        )
        print(f"CACHE {split} complete", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.cache_only:
        return

    train_archive = load_archive(cache_root / "train.npz")
    valid_archive = load_archive(cache_root / "valid.npz")
    feature_mean = train_archive["features"].mean(axis=(0, 1), keepdims=True)
    feature_std = train_archive["features"].std(axis=(0, 1), keepdims=True)
    feature_std = np.maximum(feature_std, 1e-4)
    train_set = make_dataset(
        train_archive,
        feature_mean,
        feature_std,
        max_instances=args.max_train_instances,
        sample_seed=args.backbone_seed,
    )
    valid_windows = valid_archive["features"].shape[0]
    early_stop_end = max(1, int(valid_windows * 0.4))
    calibration_end = max(early_stop_end + 1, int(valid_windows * 0.7))
    if calibration_end >= valid_windows:
        raise RuntimeError("At least four validation windows are required")
    selection_set = make_dataset(
        valid_archive,
        feature_mean,
        feature_std,
        slice(0, early_stop_end),
    )
    calibration_set = make_dataset(
        valid_archive,
        feature_mean,
        feature_std,
        slice(early_stop_end, calibration_end),
    )
    evaluation_set = make_dataset(
        valid_archive,
        feature_mean,
        feature_std,
        slice(calibration_end, None),
    )
    reports: dict[str, list[dict]] = {name: [] for name in args.calibrators}
    for head_seed in args.head_seeds:
        for name in args.calibrators:
            report = train_one(
                name,
                head_seed,
                train_set,
                selection_set,
                calibration_set,
                evaluation_set,
                float(train_archive["mean"]),
                float(train_archive["std"]),
                args,
            )
            reports[name].append(report)
            run_dir = (
                ROOT
                / args.output
                / "metr_la"
                / f"backbone_seed_{args.backbone_seed}"
                / name
                / f"head_seed_{head_seed}"
            )
            save_manifest(
                run_dir / "manifest.json",
                {
                    "study": "QUARTS-RAC validation-only exploratory pilot",
                    "arguments": vars(args),
                    "canonical_test_split_used": False,
                    "feature_names": FEATURE_NAMES,
                    "validation_windows": {
                        "early_stopping": [0, early_stop_end],
                        "conformal_calibration": [
                            early_stop_end,
                            calibration_end,
                        ],
                        "held_out_evaluation": [
                            calibration_end,
                            valid_windows,
                        ],
                    },
                },
                [
                    checkpoint,
                    ROOT / "configs" / "quarts_rac_exploratory.yaml",
                ],
            )
    summary = aggregate(reports, args)
    summary_path = (
        ROOT
        / args.output
        / "metr_la"
        / f"backbone_seed_{args.backbone_seed}"
        / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
