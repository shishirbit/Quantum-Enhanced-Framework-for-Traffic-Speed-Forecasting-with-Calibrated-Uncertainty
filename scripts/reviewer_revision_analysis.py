"""Generate reviewer-requested dependence-aware and diagnostic analyses.

All analyses operate on frozen predictions, feature caches, and fitted heads.
No model is selected from the canonical test results.  Test-set subgroup
summaries are explicitly descriptive post hoc diagnostics requested during
peer review.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from quarts.models import (  # noqa: E402
    FourierRegimeGate,
    MLPRegimeGate,
    RegimeIntervalCalibrator,
    parameter_count,
)
from run_quarts_rac import ALPHAS, load_archive, make_dataset  # noqa: E402

SEEDS = (17, 42, 73, 101, 202)
BLOCK_LENGTH = 288
BOOTSTRAP_REPLICATES = 5000


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", nargs="+", default=["point", "pemsd4", "circuit"])
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="artifacts/reviewer_revision")
    return parser.parse_args()


def circular_block_bootstrap(
    values: np.ndarray,
    replicates: int,
    block_length: int = BLOCK_LENGTH,
    seed: int = 20260930,
) -> np.ndarray:
    """Bootstrap a mean by resampling circular contiguous time blocks."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite values supplied to block bootstrap")
    rng = np.random.default_rng(seed)
    block_length = min(block_length, values.size)
    blocks_needed = math.ceil(values.size / block_length)
    result = np.empty(replicates, dtype=np.float64)
    offsets = np.arange(block_length)
    for index in range(replicates):
        starts = rng.integers(0, values.size, size=blocks_needed)
        sample_index = (starts[:, None] + offsets[None, :]) % values.size
        result[index] = values[sample_index.ravel()[: values.size]].mean()
    return result


def hierarchical_block_bootstrap(
    by_seed: list[np.ndarray],
    replicates: int,
    block_length: int = BLOCK_LENGTH,
    seed: int = 20260930,
) -> np.ndarray:
    """Resample independent training seeds, then time blocks within each seed."""
    rng = np.random.default_rng(seed)
    result = np.empty(replicates, dtype=np.float64)
    offsets = np.arange(block_length)
    count = len(by_seed)
    for replicate in range(replicates):
        selected = rng.integers(0, count, size=count)
        seed_means = []
        for selected_seed in selected:
            values = np.asarray(by_seed[selected_seed], dtype=np.float64)
            length = len(values)
            local_block = min(block_length, length)
            blocks_needed = math.ceil(length / local_block)
            starts = rng.integers(0, length, size=blocks_needed)
            indices = (starts[:, None] + offsets[:local_block][None, :]) % length
            seed_means.append(values[indices.ravel()[:length]].mean())
        result[replicate] = np.mean(seed_means)
    return result


def ci(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def masked_origin_mae(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    target = archive["target"]
    location = archive["location"]
    mask = archive["mask"].astype(bool)
    error = np.where(mask, np.abs(target - location), np.nan)
    return np.nanmean(error, axis=(1, 2))


def point_analysis(output: Path, replicates: int) -> None:
    comparisons = (
        ("quantum", "separable"),
        ("quantum", "fourier"),
        ("quantum", "staeformer"),
    )
    rows = []
    seed_rows = []
    for dataset in ("metr_la", "pems_bay"):
        for left, right in comparisons:
            differences = []
            for seed in SEEDS:
                left_path = (
                    ROOT
                    / "artifacts/runs/confirmatory"
                    / dataset
                    / left
                    / f"seed_{seed}"
                    / "predictions.npz"
                )
                right_path = (
                    ROOT
                    / "artifacts/runs/confirmatory"
                    / dataset
                    / right
                    / f"seed_{seed}"
                    / "predictions.npz"
                )
                with np.load(left_path) as left_archive, np.load(right_path) as right_archive:
                    left_mae = masked_origin_mae(left_archive)
                    right_mae = masked_origin_mae(right_archive)
                    common = min(len(left_mae), len(right_mae))
                    # Official STAEformer retains 18 boundary-crossing windows
                    # that precede the strictly within-split QUARTS windows.
                    # The common chronological origins are therefore aligned
                    # at the end of both arrays.
                    difference = left_mae[-common:] - right_mae[-common:]
                    difference = difference[np.isfinite(difference)]
                differences.append(difference)
                bootstrap = circular_block_bootstrap(
                    difference,
                    replicates,
                    seed=20260930 + seed,
                )
                low, high = ci(bootstrap)
                seed_rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"{left}_minus_{right}",
                        "seed": seed,
                        "time_origins": len(difference),
                        "block_length_origins": BLOCK_LENGTH,
                        "mean_MAE_difference": difference.mean(),
                        "block_bootstrap_ci_low": low,
                        "block_bootstrap_ci_high": high,
                    }
                )
            means = np.asarray([values.mean() for values in differences])
            bootstrap = hierarchical_block_bootstrap(
                differences,
                replicates,
                seed=20260930 + sum(ord(character) for character in dataset + left + right),
            )
            low, high = ci(bootstrap)
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"{left}_minus_{right}",
                    "independent_seeds": len(differences),
                    "mean_MAE_difference": means.mean(),
                    "seed_SD": means.std(ddof=1),
                    "hierarchical_block_ci_low": low,
                    "hierarchical_block_ci_high": high,
                    "directional_wins_left": int(np.sum(means < 0)),
                    "directional_wins_right": int(np.sum(means > 0)),
                }
            )
    pd.DataFrame(rows).to_csv(output / "dependence_aware_point_summary.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output / "dependence_aware_point_by_seed.csv", index=False)


def pems_paths(seed: int) -> tuple[Path, Path]:
    suffix = "" if seed == 42 else f"_seed{seed}"
    run_root = ROOT / "artifacts/runs" / f"prospective_pemsd4{suffix}"
    cache_root = ROOT / "artifacts/cache" / f"prospective_pemsd4{suffix}" / f"pemsd4_seed_{seed}"
    return run_root, cache_root


def build_pems_head(name: str, checkpoint: Path, device: torch.device):
    hidden, horizons, experts = 44, 12, 3
    if name == "mlp":
        gate = MLPRegimeGate(hidden, experts=experts, width=5)
    elif name == "fourier":
        gate = FourierRegimeGate(hidden, experts=experts, width=3, depth=2)
    else:
        raise ValueError(name)
    head = RegimeIntervalCalibrator(gate, horizons, len(ALPHAS), experts).to(device)
    head.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    head.eval()
    return head


@torch.no_grad()
def predict_radii_weights(head, dataset, device: torch.device, batch_size: int = 4096):
    radii, weights = [], []
    for feature, _base, _target, _mask in DataLoader(dataset, batch_size=batch_size):
        predicted, gate = head(feature.to(device)[:, None, :])
        radii.append(predicted.squeeze(2).cpu().numpy())
        weights.append(gate.squeeze(1).cpu().numpy())
    return np.concatenate(radii), np.concatenate(weights)


def wis_components(
    target: np.ndarray,
    location: np.ndarray,
    radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    absolute_error = np.abs(target - location)
    aggregate = 0.5 * absolute_error
    for level, alpha in enumerate(ALPHAS):
        radius = radii[..., level]
        lower = location - radius
        upper = location + radius
        score = (
            upper
            - lower
            + (2.0 / alpha) * (lower - target) * (target < lower)
            + (2.0 / alpha) * (target - upper) * (target > upper)
        )
        aggregate += (alpha / 2.0) * score
    wis = aggregate / (len(ALPHAS) + 0.5)
    radius90 = radii[..., 0]
    coverage90 = np.abs(target - location) <= radius90
    width90 = 2.0 * radius90
    return wis, coverage90, width90


def summarize_condition(
    rows: list[dict],
    seed: int,
    model: str,
    dimension: str,
    labels: np.ndarray,
    wis: np.ndarray,
    coverage: np.ndarray,
    width: np.ndarray,
) -> None:
    labels = np.asarray(labels)
    for label in np.unique(labels):
        selected = labels == label
        rows.append(
            {
                "seed": seed,
                "model": model,
                "dimension": dimension,
                "level": str(label),
                "observations": int(selected.sum()),
                "WIS": float(wis[selected].mean()),
                "coverage90": float(coverage[selected].mean()),
                "width90": float(width[selected].mean()),
            }
        )


def quantile_labels(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    boundaries = np.quantile(finite, [0.25, 0.5, 0.75])
    return np.asarray(["Q1", "Q2", "Q3", "Q4"])[np.digitize(finite, boundaries)]


def pemsd4_analysis(output: Path, replicates: int, device_name: str) -> None:
    device = torch.device(device_name)
    seed_metrics = []
    condition_rows = []
    origin_differences = []
    node_rows = []
    regime_rows = []
    for seed in SEEDS:
        run_root, cache_root = pems_paths(seed)
        required = [
            cache_root / "train.npz",
            cache_root / "test.npz",
            run_root / "heads/mlp" / f"seed_{seed}" / "best.pt",
            run_root / "heads/fourier" / f"seed_{seed}" / "best.pt",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing five-backbone inputs: " + ", ".join(missing))
        train_archive = load_archive(cache_root / "train.npz")
        test_archive = load_archive(cache_root / "test.npz")
        feature_mean = train_archive["features"].mean(axis=(0, 1), keepdims=True)
        feature_std = np.maximum(
            train_archive["features"].std(axis=(0, 1), keepdims=True), 1e-4
        )
        test_set = make_dataset(test_archive, feature_mean, feature_std)
        windows, horizons, nodes = test_archive["target"].shape
        target = test_archive["target"] * float(test_archive["std"]) + float(test_archive["mean"])
        location = test_archive["base"] * float(test_archive["std"]) + float(test_archive["mean"])
        by_model = {}
        for model in ("mlp", "fourier"):
            checkpoint = run_root / "heads" / model / f"seed_{seed}" / "best.pt"
            metrics_path = checkpoint.parent / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            factors = np.asarray(metrics["coverage_calibration_factors"], dtype=np.float32)
            head = build_pems_head(model, checkpoint, device)
            flat_radii, flat_weights = predict_radii_weights(head, test_set, device)
            radii = flat_radii.reshape(windows, nodes, horizons, len(ALPHAS)).transpose(0, 2, 1, 3)
            radii = radii * factors[None, :, None, :] * float(test_archive["std"])
            weights = flat_weights.reshape(windows, nodes, -1)
            wis, coverage, width = wis_components(target, location, radii)
            by_model[model] = {
                "wis": wis,
                "coverage": coverage,
                "width": width,
                "weights": weights,
            }
            seed_metrics.append(
                {
                    "seed": seed,
                    "model": model,
                    "WIS": float(wis.mean()),
                    "coverage90": float(coverage.mean()),
                    "width90": float(width.mean()),
                    "parameters": parameter_count(head),
                    "mean_expert_weights": json.dumps(weights.mean(axis=(0, 1)).tolist()),
                    "effective_experts": float(
                        np.exp(
                            -np.sum(
                                weights.mean(axis=(0, 1))
                                * np.log(np.maximum(weights.mean(axis=(0, 1)), 1e-12))
                            )
                        )
                    ),
                }
            )
            horizon_labels = np.broadcast_to(
                np.arange(1, horizons + 1)[None, :, None], target.shape
            )
            summarize_condition(
                condition_rows, seed, model, "forecast_horizon", horizon_labels, wis, coverage, width
            )
            summarize_condition(
                condition_rows,
                seed,
                model,
                "traffic_intensity_quartile",
                quantile_labels(target.ravel()).reshape(target.shape),
                wis,
                coverage,
                width,
            )
            residual = np.abs(target - location)
            summarize_condition(
                condition_rows,
                seed,
                model,
                "base_residual_quartile",
                quantile_labels(residual.ravel()).reshape(residual.shape),
                wis,
                coverage,
                width,
            )
            local_std = test_archive["features"][..., 14]
            local_labels = quantile_labels(local_std.ravel()).reshape(local_std.shape)
            local_labels = np.broadcast_to(local_labels[:, None, :], target.shape)
            summarize_condition(
                condition_rows,
                seed,
                model,
                "recent_local_volatility_quartile",
                local_labels,
                wis,
                coverage,
                width,
            )
            dominant = np.argmax(weights, axis=-1) + 1
            dominant = np.broadcast_to(dominant[:, None, :], target.shape)
            summarize_condition(
                condition_rows,
                seed,
                model,
                "dominant_interval_expert",
                dominant,
                wis,
                coverage,
                width,
            )
            if model == "fourier":
                dominant_sensor = np.argmax(weights, axis=-1) + 1
                target_level = target.mean(axis=1)
                base_residual = np.abs(target - location).mean(axis=1)
                local_std_sensor = test_archive["features"][..., 14]
                for expert in range(1, weights.shape[-1] + 1):
                    selected_expert = dominant_sensor == expert
                    regime_rows.append(
                        {
                            "seed": seed,
                            "expert": expert,
                            "share": float(selected_expert.mean()),
                            "mean_traffic_intensity": float(target_level[selected_expert].mean()) if selected_expert.any() else np.nan,
                            "mean_recent_local_volatility": float(local_std_sensor[selected_expert].mean()) if selected_expert.any() else np.nan,
                            "mean_absolute_base_residual": float(base_residual[selected_expert].mean()) if selected_expert.any() else np.nan,
                        }
                    )
            for node in range(nodes):
                node_rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "node": node,
                        "coverage90": float(coverage[:, :, node].mean()),
                        "width90": float(width[:, :, node].mean()),
                        "WIS": float(wis[:, :, node].mean()),
                    }
                )
            del head
            if device.type == "cuda":
                torch.cuda.empty_cache()
        difference = by_model["fourier"]["wis"].mean(axis=(1, 2)) - by_model["mlp"]["wis"].mean(axis=(1, 2))
        origin_differences.append(difference)
        bootstrap = circular_block_bootstrap(difference, replicates, seed=20260930 + seed)
        low, high = ci(bootstrap)
        seed_metrics.append(
            {
                "seed": seed,
                "model": "fourier_minus_mlp",
                "WIS": float(difference.mean()),
                "coverage90": np.nan,
                "width90": np.nan,
                "parameters": np.nan,
                "mean_expert_weights": "",
                "effective_experts": np.nan,
                "block_bootstrap_ci_low": low,
                "block_bootstrap_ci_high": high,
            }
        )
    seed_frame = pd.DataFrame(seed_metrics)
    seed_frame.to_csv(output / "pemsd4_five_backbone_metrics.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(output / "pemsd4_conditional_calibration.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(output / "pemsd4_expert_regime_characteristics.csv", index=False)
    node_frame = pd.DataFrame(node_rows)
    node_frame.to_csv(output / "pemsd4_node_calibration.csv", index=False)
    replication_means = np.asarray([values.mean() for values in origin_differences])
    bootstrap = hierarchical_block_bootstrap(origin_differences, replicates, seed=20260930)
    low, high = ci(bootstrap)
    replication_summary = {
        "experimental_unit": "independently trained STAEformer backbone",
        "independent_seeds": list(SEEDS),
        "comparison": "Fourier-RAC minus MLP-RAC WIS",
        "mean_difference": float(replication_means.mean()),
        "seed_SD": float(replication_means.std(ddof=1)),
        "hierarchical_block_ci95": [low, high],
        "Fourier_wins": int(np.sum(replication_means < 0)),
        "MLP_wins": int(np.sum(replication_means > 0)),
        "bootstrap_unit": "five-minute forecast origin in circular 288-origin daily blocks, nested within backbone seed",
        "post_hoc_reviewer_requested": True,
    }
    (output / "pemsd4_replication_inference.json").write_text(
        json.dumps(replication_summary, indent=2), encoding="utf-8"
    )
    summary_rows = []
    for model in ("mlp", "fourier"):
        selected = seed_frame[seed_frame.model == model]
        summary_rows.append(
            {
                "model": model,
                "backbones": len(selected),
                "WIS_mean": selected.WIS.mean(),
                "WIS_SD": selected.WIS.std(ddof=1),
                "coverage90_mean": selected.coverage90.mean(),
                "coverage90_SD": selected.coverage90.std(ddof=1),
                "width90_mean": selected.width90.mean(),
                "width90_SD": selected.width90.std(ddof=1),
                "effective_experts_mean": selected.effective_experts.mean(),
                "effective_experts_SD": selected.effective_experts.std(ddof=1),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output / "pemsd4_five_backbone_summary.csv", index=False)
    node_summary = (
        node_frame.groupby("model")
        .agg(
            node_coverage_median=("coverage90", "median"),
            node_coverage_q1=("coverage90", lambda x: x.quantile(0.25)),
            node_coverage_q3=("coverage90", lambda x: x.quantile(0.75)),
            node_coverage_min=("coverage90", "min"),
            node_width_median=("width90", "median"),
            node_width_q1=("width90", lambda x: x.quantile(0.25)),
            node_width_q3=("width90", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    node_summary.to_csv(output / "pemsd4_node_calibration_summary.csv", index=False)


def circuit_analysis(output: Path) -> None:
    import pennylane as qml

    rng = np.random.default_rng(20260930)
    qubits, depth, samples = 4, 2, 256
    states = {}
    specs_rows = []
    for entangled in (False, True):
        device = qml.device("default.qubit", wires=qubits)

        @qml.qnode(device)
        def circuit(inputs, weights):
            for layer in range(depth):
                for wire in range(qubits):
                    qml.RY(inputs[wire], wires=wire)
                    qml.Rot(*weights[layer, wire], wires=wire)
                if entangled:
                    for wire in range(qubits):
                        qml.CNOT(wires=[wire, (wire + 1) % qubits])
            return qml.state()

        inputs = rng.uniform(-math.pi, math.pi, size=(samples, qubits))
        weights = rng.uniform(0.0, 2.0 * math.pi, size=(samples, depth, qubits, 3))
        sample_states = np.asarray([circuit(inputs[i], weights[i]) for i in range(samples)])
        states[entangled] = sample_states
        resources = qml.specs(circuit)(inputs[0], weights[0])["resources"]
        specs_rows.append(
            {
                "circuit": "entangled" if entangled else "separable",
                "qubits": qubits,
                "reupload_layers": depth,
                "trainable_rotation_parameters": depth * qubits * 3,
                "data_RY_gates": depth * qubits,
                "trainable_Rot_gates": depth * qubits,
                "elementary_trainable_rotations": depth * qubits * 3,
                "CNOT_gates": depth * qubits if entangled else 0,
                "PennyLane_logical_depth": resources.depth,
                "head_trainable_parameters": 276,
                "complete_QUARTS_parameters": 4140,
                "initialization": "independent Uniform(0, 2pi)",
            }
        )

    def meyer_wallach(state: np.ndarray) -> float:
        tensor = state.reshape(*([2] * qubits))
        purities = []
        for wire in range(qubits):
            matrix = np.moveaxis(tensor, wire, 0).reshape(2, -1)
            reduced = matrix @ matrix.conj().T
            purities.append(np.real(np.trace(reduced @ reduced)))
        return float(2.0 * (1.0 - np.mean(purities)))

    def expressibility_kl(sample_states: np.ndarray) -> float:
        first = sample_states[0::2]
        second = sample_states[1::2]
        fidelity = np.abs(np.sum(first.conj() * second, axis=1)) ** 2
        bins = np.linspace(0.0, 1.0, 26)
        empirical, _ = np.histogram(fidelity, bins=bins, density=False)
        empirical = empirical / empirical.sum()
        dimension = 2**qubits
        expected = (1.0 - bins[:-1]) ** (dimension - 1) - (1.0 - bins[1:]) ** (dimension - 1)
        expected = expected / expected.sum()
        return float(
            np.sum(
                empirical
                * np.log((empirical + 1e-12) / (expected + 1e-12))
            )
        )

    diagnostics = []
    for entangled, sample_states in states.items():
        entanglement = np.asarray([meyer_wallach(state) for state in sample_states])
        diagnostics.append(
            {
                "circuit": "entangled" if entangled else "separable",
                "expressibility_KL_to_Haar": expressibility_kl(sample_states),
                "Meyer_Wallach_mean": entanglement.mean(),
                "Meyer_Wallach_SD": entanglement.std(ddof=1),
                "random_circuits": samples,
            }
        )
    design = pd.DataFrame(specs_rows).merge(pd.DataFrame(diagnostics), on="circuit")
    design.to_csv(output / "circuit_structure_expressivity.csv", index=False)

    ablation = pd.read_csv(ROOT / "artifacts/publication/tables/table05_quantum_architecture_ablation.csv")
    ablation["statevector_dimension"] = 2 ** ablation.qubits
    ablation["CNOT_gates"] = ablation.qubits * ablation.depth
    ablation["trainable_rotation_parameters"] = 3 * ablation.qubits * ablation.depth
    ablation["circuit_instance_inference_seconds"] = ablation.inference_seconds
    ablation.to_csv(output / "simulator_scaling.csv", index=False)

    resource = pd.read_csv(ROOT / "artifacts/publication/tables/table03_resource_efficiency.csv")
    quantum = resource[resource.model.isin(["quantum", "separable", "fourier", "staeformer"])].copy()
    quantum.to_csv(output / "controlled_resource_comparison.csv", index=False)

    dataset_sizes = {
        "metr_la": {"train": 23967, "validation": 3404, "test": 6832, "nodes": 207},
        "pems_bay": {"train": 36458, "validation": 5188, "test": 10401, "nodes": 325},
    }
    evaluation_rows = []
    for dataset, size in dataset_sizes.items():
        metrics = json.loads(
            (
                ROOT
                / "artifacts/runs/confirmatory"
                / dataset
                / "quantum/seed_42/metrics.json"
            ).read_text(encoding="utf-8")
        )
        epochs = len(metrics["history"])
        batched_calls = epochs * (
            math.ceil(size["train"] / 256) + math.ceil(size["validation"] / 256)
        ) + math.ceil(size["test"] / 256)
        circuit_instances = (
            epochs * (size["train"] + size["validation"]) + size["test"]
        ) * size["nodes"]
        evaluation_rows.append(
            {
                "dataset": dataset,
                "epochs": epochs,
                "batch_size": 256,
                "batched_quantum_forward_calls": batched_calls,
                "statevector_circuit_instances": circuit_instances,
                "peak_GPU_memory_GiB": metrics["peak_gpu_memory_bytes"] / (1024**3),
                "training_plus_inference_seconds": metrics["total_seconds"],
                "test_inference_seconds": metrics["inference_seconds"],
                "backend": "analytic statevector simulator on classical GPU",
            }
        )
    pd.DataFrame(evaluation_rows).to_csv(output / "quantum_simulator_cost.csv", index=False)


def main() -> None:
    args = arguments()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if "point" in args.sections:
        point_analysis(output, args.bootstrap_replicates)
    if "pemsd4" in args.sections:
        pemsd4_analysis(output, args.bootstrap_replicates, args.device)
    if "circuit" in args.sections:
        circuit_analysis(output)
    audit = {
        "complete": True,
        "sections": args.sections,
        "bootstrap_replicates": args.bootstrap_replicates,
        "block_length_origins": BLOCK_LENGTH,
        "block_interpretation": "one day at five-minute resolution",
        "independent_replication_unit": "training seed",
        "test_subgroup_analysis": "post hoc descriptive diagnostic requested during peer review",
    }
    (output / "analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
