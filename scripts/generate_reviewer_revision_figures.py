"""Create publication figures added in response to peer review."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "artifacts/reviewer_revision"
FIGURES = ROOT / "manuscript/quarts_springer/figures"
SUPPLEMENT = FIGURES / "supplement"

BLUE = "#D9E8F7"
GREEN = "#DDEED8"
YELLOW = "#FFF0C7"
RED = "#F8D8D3"
PURPLE = "#E8DDF2"
INK = "#202020"


def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, width, height, text, color, fontsize=9, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=color,
        edgecolor=INK,
        linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, text=None, color=INK, style="-"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color=color,
        linestyle=style,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.025,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )


def experiment_flow() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.975, "Experimental sequence and decision logic", ha="center", va="top", fontsize=14, fontweight="bold")
    ax.text(0.225, 0.925, "Experimental stage", ha="center", fontsize=10, fontweight="bold")
    ax.text(0.755, 0.925, "Decision and consequence", ha="center", fontsize=10, fontweight="bold")
    stage_x, stage_w, decision_x, decision_w, height = 0.05, 0.35, 0.55, 0.40, 0.105
    ys = [0.78, 0.63, 0.48, 0.33, 0.18]
    stage_texts = [
        "Stage 1: Point forecasting\nRQ1 and RQ2",
        "Stage 2: Uncertainty heads\nRQ3",
        "Stage 3: Regime calibration\nValidation only",
        "Stage 4: Prospective PeMSD4\nProtected test",
        "Stage 5: Independent backbones\nReplication for RQ4",
    ]
    decision_texts = [
        "Advance only if the primary metric improves\nand all declared criteria are satisfied",
        "FAIL: entangled branches retained\nas negative evidence",
        "PASS: Fourier RAC selected on validation\nbefore test access",
        "PROVISIONAL PASS: initial PeMSD4 result\nrequires independent replication",
        "FAIL: five-backbone effect is unstable\nand the block-bootstrap interval crosses zero",
    ]
    stage_colors = [BLUE, BLUE, GREEN, GREEN, GREEN]
    decision_colors = [YELLOW, RED, YELLOW, YELLOW, RED]
    for index, y in enumerate(ys):
        box(ax, (stage_x, y), stage_w, height, stage_texts[index], stage_colors[index], fontsize=9.4, weight="bold")
        box(ax, (decision_x, y), decision_w, height, decision_texts[index], decision_colors[index], fontsize=8.2)
        arrow(ax, (stage_x + stage_w, y + height / 2), (decision_x, y + height / 2))
        if index < len(ys) - 1:
            arrow(ax, (stage_x + stage_w / 2, y), (stage_x + stage_w / 2, ys[index + 1] + height))
    box(ax, (0.38, 0.025), 0.57, 0.085, "Supplementary post hoc diagnostics\nExpert count, conditional coverage, circuit expressivity, and simulator cost", PURPLE, fontsize=7.8)
    arrow(ax, (0.75, ys[-1]), (0.75, 0.11), color="#674188", style="--")
    save(fig, FIGURES / "figure07_experimental_flow")


def system_architecture() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.6))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title("(a) End-to-end QUARTS workflow", fontsize=13, fontweight="bold", pad=12)
    input_specs = [
        (0.02, "Historical traffic"),
        (0.35, "Road graph"),
        (0.68, "Time covariates"),
    ]
    for xpos, label in input_specs:
        box(ax, (xpos, 0.88), 0.30, 0.075, label, BLUE, fontsize=8.4, weight="bold")
        arrow(ax, (xpos + 0.15, 0.88), (0.5, 0.82))
    box(ax, (0.15, 0.70), 0.70, 0.12, "Chronological preprocessing\nTraining-only scaling and target masking", BLUE, fontsize=9.0)
    arrow(ax, (0.5, 0.70), (0.5, 0.64))
    box(ax, (0.15, 0.53), 0.70, 0.11, "Forecasting backbone\nGraph temporal model or frozen STAEformer", GREEN, fontsize=9.0, weight="bold")
    arrow(ax, (0.5, 0.53), (0.5, 0.47))
    box(ax, (0.24, 0.38), 0.52, 0.09, "Base forecast and node-level latent state", GREEN, fontsize=8.7)
    arrow(ax, (0.5, 0.38), (0.5, 0.32))
    box(ax, (0.18, 0.22), 0.64, 0.10, "Matched calibration head\nPoint correction, scale, or interval radii", YELLOW, fontsize=8.8, weight="bold")
    arrow(ax, (0.5, 0.22), (0.5, 0.16))
    box(ax, (0.14, 0.07), 0.72, 0.09, "Chronological conformal adjustment", YELLOW, fontsize=8.7)
    arrow(ax, (0.5, 0.07), (0.5, 0.015))
    ax.text(0.5, 0.002, "Final forecast, predictive intervals, and validation decision", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

    ax = axes[1]
    ax.set_title("(b) Matched heads and entangled VQC", fontsize=13, fontweight="bold", pad=12)
    box(ax, (0.01, 0.72), 0.29, 0.12, "Common base forecast\nand latent features", GREEN, fontsize=8.2, weight="bold")
    heads = [
        (0.86, "MLP (tanh)"),
        (0.72, "Fourier (sin and cos)"),
        (0.58, "Separable VQC"),
        (0.44, "Entangled VQC"),
    ]
    for y, label in heads:
        box(ax, (0.38, y), 0.27, 0.085, label, YELLOW, fontsize=8.6)
        arrow(ax, (0.30, 0.78), (0.38, y + 0.042))
        arrow(ax, (0.65, y + 0.042), (0.76, 0.71))
    box(ax, (0.76, 0.64), 0.22, 0.14, "Identical outputs\nand objective", YELLOW, fontsize=9.0, weight="bold")

    frame = FancyBboxPatch((0.03, 0.04), 0.94, 0.34, boxstyle="round,pad=0.02,rounding_size=0.025", facecolor="#FAFAFA", edgecolor=INK, linewidth=1.1)
    ax.add_patch(frame)
    ax.text(0.5, 0.35, "Entangled VQC internal path", ha="center", va="center", fontsize=10.2, fontweight="bold")
    blocks = [
        (0.07, 0.15, 0.18, "Linear\ncompression", BLUE),
        (0.30, 0.15, 0.20, "$R_Y$ encoding\nand Rot gates", BLUE),
        (0.55, 0.15, 0.18, "Ring CNOT\nentanglement", BLUE),
        (0.78, 0.13, 0.16, "Pauli $Z$\nand readout", PURPLE),
    ]
    for xpos, ypos, width, label, color in blocks:
        box(ax, (xpos, ypos), width, 0.10, label, color, fontsize=8.3)
    for left, right in ((0.25, 0.30), (0.50, 0.55), (0.73, 0.78)):
        arrow(ax, (left, 0.20), (right, 0.20))
    save(fig, FIGURES / "figure00_system_architecture")


def circuit_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)
    for ax, entangled, title in zip(
        axes,
        (False, True),
        ("Separable VQC", "Entangled VQC with ring CNOT"),
    ):
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.65, 4.05)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold")
        for qubit in range(4):
            y = 3.25 - qubit
            ax.plot([0.35, 9.65], [y, y], color=INK, linewidth=1.0)
            ax.text(0.15, y, rf"$q_{{{qubit}}}$", ha="right", va="center", fontsize=9)
        for layer in range(2):
            offset = 0.75 + layer * 4.55
            for qubit in range(4):
                y = 3.25 - qubit
                box(ax, (offset, y - 0.20), 0.78, 0.40, r"$R_Y(z)$", BLUE, fontsize=8)
                box(ax, (offset + 1.05, y - 0.20), 0.78, 0.40, "Rot", GREEN, fontsize=8)
            if entangled:
                cnot_x = offset + 2.20
                for qubit in range(4):
                    control_y = 3.25 - qubit
                    target_y = 3.25 - ((qubit + 1) % 4)
                    x = cnot_x + 0.30 * qubit
                    ax.plot([x, x], [control_y, target_y], color="#9B2C2C", linewidth=1.0)
                    ax.plot(x, control_y, "o", color="#9B2C2C", markersize=4)
                    ax.plot(x, target_y, marker="o", markerfacecolor="white", markeredgecolor="#9B2C2C", markersize=8)
                    ax.plot([x - 0.06, x + 0.06], [target_y, target_y], color="#9B2C2C", linewidth=1.0)
                    ax.plot([x, x], [target_y - 0.06, target_y + 0.06], color="#9B2C2C", linewidth=1.0)
                ax.text(offset + 2.65, 3.88, "CNOT ring", ha="center", va="center", fontsize=8.0, color="#9B2C2C")
            ax.text(offset + 1.45, -0.40, f"Reupload layer {layer + 1}", ha="center", va="center", fontsize=8)
        for qubit in range(4):
            y = 3.25 - qubit
            box(ax, (9.00, y - 0.20), 0.55, 0.40, "$Z$", PURPLE, fontsize=8)
        ax.text(9.28, 3.88, "Measure", ha="center", va="center", fontsize=8.0)
        ax.text(5.0, 3.67, "Same compression, rotations, initialization, measurement, and readout", ha="center", va="center", fontsize=7.7)
    fig.text(
        0.5,
        0.01,
        "Only the two ring CNOT layers differ: the separable circuit has 0 CNOT gates and logical depth 4; the entangled circuit has 8 CNOT gates and logical depth 12.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, FIGURES / "figure08_circuit_comparison")


def quantitative_robustness_figure() -> None:
    seed_path = ANALYSIS / "pemsd4_five_backbone_metrics.csv"
    condition_path = ANALYSIS / "pemsd4_conditional_calibration.csv"
    k_path = ANALYSIS / "expert_count_ablation/summary.json"
    if not (seed_path.exists() and condition_path.exists() and k_path.exists()):
        return
    seeds = pd.read_csv(seed_path)
    conditions = pd.read_csv(condition_path)
    k_data = json.loads(k_path.read_text(encoding="utf-8"))["summary"]
    k_frame = pd.DataFrame(k_data)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2))

    ax = axes[0, 0]
    pivot = seeds[seeds.model.isin(["mlp", "fourier"])].pivot(index="seed", columns="model", values="WIS")
    difference = pivot.fourier - pivot.mlp
    colors = ["#2F855A" if value < 0 else "#C53030" for value in difference]
    ax.bar([str(seed) for seed in difference.index], difference, color=colors)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_ylabel("Fourier minus MLP WIS")
    ax.set_xlabel("Independent backbone seed")
    ax.set_title("(a) Replication level effect")

    ax = axes[0, 1]
    horizon = conditions[
        (conditions.dimension == "forecast_horizon") & (conditions.model == "fourier")
    ].copy()
    horizon["level_num"] = horizon.level.astype(int)
    summary = horizon.groupby("level_num").coverage90.agg(["mean", "std"]).reset_index()
    ax.errorbar(summary.level_num * 5, summary["mean"], yerr=summary["std"], marker="o", color="#2B6CB0", capsize=3)
    ax.axhline(0.9, color=INK, linestyle="--", linewidth=0.9)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("90% interval coverage")
    ax.set_ylim(0.80, 0.96)
    ax.set_title("(b) Coverage by horizon")

    ax = axes[1, 0]
    intensity = conditions[
        (conditions.dimension == "traffic_intensity_quartile") & (conditions.model == "fourier")
    ]
    summary = intensity.groupby("level").agg(coverage=("coverage90", "mean"), width=("width90", "mean")).reindex(["Q1", "Q2", "Q3", "Q4"])
    positions = np.arange(4)
    ax.bar(positions - 0.18, summary.coverage, width=0.36, color="#63B3ED", label="coverage")
    ax.set_ylim(0.75, 1.0)
    ax.set_xticks(positions, summary.index)
    ax.set_xlabel("Traffic intensity quartile")
    ax.set_ylabel("Coverage")
    twin = ax.twinx()
    twin.bar(positions + 0.18, summary.width, width=0.36, color="#F6AD55", label="width")
    twin.set_ylabel("Mean interval width")
    ax.set_title("(c) Conditional calibration")

    ax = axes[1, 1]
    ax.errorbar(
        k_frame.experts,
        k_frame.conformal_WIS_mean,
        yerr=k_frame.conformal_WIS_sd,
        marker="o",
        color="#805AD5",
        capsize=3,
        label="WIS",
    )
    ax.set_xlabel("Number of interval experts K")
    ax.set_ylabel("WIS")
    twin = ax.twinx()
    twin.plot(k_frame.experts, k_frame.effective_experts_mean, marker="s", color="#2F855A", label="effective experts")
    twin.set_ylabel("Effective experts")
    ax.set_title("(d) Validation only expert ablation")

    fig.tight_layout()
    save(fig, FIGURES / "figure09_robustness_diagnostics")


def main() -> None:
    system_architecture()
    experiment_flow()
    circuit_comparison()
    quantitative_robustness_figure()


if __name__ == "__main__":
    main()
