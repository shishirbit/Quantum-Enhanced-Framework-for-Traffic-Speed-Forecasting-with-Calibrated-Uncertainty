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
    fig, ax = plt.subplots(figsize=(12.2, 5.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.965, "Prespecified confirmatory path", ha="center", va="top", fontsize=12, fontweight="bold")
    widths, height = 0.165, 0.16
    x = [0.015, 0.213, 0.411, 0.609, 0.807]
    texts = [
        "Stage 1\nPoint forecasting\nRQ1 and RQ2",
        "Stage 2\nUncertainty heads\nRQ3",
        "Stage 3\nRegime calibration\nValidation only",
        "Stage 4\nProspective PeMSD4\nProtected test",
        "Stage 5\nIndependent backbone\nreplication, RQ4",
    ]
    colors = [BLUE, BLUE, GREEN, GREEN, GREEN]
    for xpos, text, color in zip(x, texts, colors):
        box(ax, (xpos, 0.65), widths, height, text, color, fontsize=9, weight="bold")
    for index in range(4):
        arrow(ax, (x[index] + widths, 0.73), (x[index + 1], 0.73), "gate")

    gate_texts = [
        "Advance if primary metric improves\nand coverage is 0.89 to 0.91",
        "Entangled branches failed\nRetained as negative evidence",
        "Fourier RAC selected\non validation before test access",
        "Initial PeMSD4 result\npassed provisionally",
        "Five backbones and block bootstrap\nFailed robustness gate",
    ]
    gate_colors = [YELLOW, RED, YELLOW, YELLOW, RED]
    y = [0.32, 0.32, 0.32, 0.32, 0.32]
    for xpos, ypos, text, color in zip(x, y, gate_texts, gate_colors):
        box(ax, (xpos, ypos), widths, 0.15, text, color, fontsize=7.2)
    arrow(ax, (x[0] + widths / 2, 0.65), (x[0] + widths / 2, y[0] + 0.15))
    arrow(ax, (x[1] + widths / 2, 0.65), (x[1] + widths / 2, y[1] + 0.15), color="#9B2C2C")
    arrow(ax, (x[2] + widths / 2, 0.65), (x[2] + widths / 2, y[2] + 0.15))
    arrow(ax, (x[3] + widths / 2, 0.65), (x[3] + widths / 2, y[3] + 0.15))
    arrow(ax, (x[4] + widths / 2, 0.65), (x[4] + widths / 2, y[4] + 0.15), color="#9B2C2C")

    box(
        ax,
        (0.58, 0.03),
        0.39,
        0.13,
        "Reviewer requested post hoc diagnostics\nK ablation, conditional coverage,\ncircuit expressivity, and simulator cost",
        PURPLE,
        fontsize=8.3,
    )
    arrow(ax, (0.89, 0.32), (0.89, 0.16), text="diagnostic only", color="#674188", style="--")
    save(fig, FIGURES / "figure07_experimental_flow")


def circuit_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), sharey=True)
    for ax, entangled, title in zip(
        axes,
        (False, True),
        ("Separable VQC", "Entangled VQC with ring CNOT"),
    ):
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.7, 3.7)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold")
        for qubit in range(4):
            y = 3 - qubit
            ax.plot([0.25, 9.75], [y, y], color=INK, linewidth=1.0)
            ax.text(0.05, y, rf"$q_{{{qubit}}}$", ha="right", va="center", fontsize=9)
        for layer in range(2):
            offset = 0.65 + layer * 4.35
            for qubit in range(4):
                y = 3 - qubit
                box(ax, (offset, y - 0.23), 0.75, 0.46, r"$R_Y(z)$", BLUE, fontsize=8)
                box(ax, (offset + 1.0, y - 0.23), 0.88, 0.46, "Rot", GREEN, fontsize=8)
            if entangled:
                cnot_x = offset + 2.35
                for qubit in range(4):
                    control_y = 3 - qubit
                    target_y = 3 - ((qubit + 1) % 4)
                    x = cnot_x + 0.27 * qubit
                    ax.plot([x, x], [control_y, target_y], color="#9B2C2C", linewidth=1.0)
                    ax.plot(x, control_y, "o", color="#9B2C2C", markersize=4)
                    ax.plot(x, target_y, marker="o", markerfacecolor="white", markeredgecolor="#9B2C2C", markersize=8)
                    ax.plot([x - 0.06, x + 0.06], [target_y, target_y], color="#9B2C2C", linewidth=1.0)
                    ax.plot([x, x], [target_y - 0.06, target_y + 0.06], color="#9B2C2C", linewidth=1.0)
            ax.text(offset + 1.65, -0.5, f"reupload layer {layer + 1}", ha="center", va="center", fontsize=8)
        annotation = (
            "24 trainable rotation parameters\n8 data rotations, 0 CNOT, logical depth 4"
            if not entangled
            else "24 trainable rotation parameters\n8 data rotations, 8 CNOT, logical depth 12"
        )
        ax.text(5.0, 3.55, annotation, ha="center", va="top", fontsize=8.5)
    fig.text(
        0.5,
        0.01,
        "Both circuits use identical compression, parameter initialization, local rotations, measurement, readout, and optimizer. Only the ring CNOT operations differ.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, FIGURES / "figure08_circuit_comparison")


def quantitative_revision_figure() -> None:
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
    save(fig, FIGURES / "figure09_revision_diagnostics")


def main() -> None:
    experiment_flow()
    circuit_comparison()
    quantitative_revision_figure()


if __name__ == "__main__":
    main()
