"""Generate the journal figure for the QUARTS system architecture."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manuscript" / "quarts_springer" / "figures"

COLORS = {
    "input": "#e8f1f8",
    "classical": "#d9ead3",
    "feature": "#fff2cc",
    "quantum": "#eadcf8",
    "output": "#fce5cd",
    "audit": "#eeeeee",
    "ink": "#222222",
}


def box(
    axis,
    x,
    y,
    width,
    height,
    text,
    color,
    fontsize=8.2,
    linewidth=1.0,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=color,
        edgecolor=COLORS["ink"],
        linewidth=linewidth,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
    )
    return patch


def arrow(axis, start, end, connectionstyle="arc3", linewidth=1.1):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=linewidth,
        color=COLORS["ink"],
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
    )
    axis.add_patch(patch)
    return patch


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(13.4, 7.2))
    grid = figure.add_gridspec(2, 1, height_ratios=(1.18, 0.82), hspace=0.28)

    top = figure.add_subplot(grid[0, 0])
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.axis("off")
    top.text(
        0.0,
        1.02,
        "(a) End to end forecasting and calibration workflow",
        fontsize=10,
        weight="bold",
        ha="left",
    )

    widths = [0.135, 0.13, 0.15, 0.14, 0.15, 0.13]
    xs = [0.015, 0.18, 0.34, 0.52, 0.69, 0.855]
    y_top = 0.63
    height = 0.23
    labels = [
        "Historical traffic tensor\nRoad graph\nTemporal covariates",
        "Chronological split\nTrain only scaling\nMask preservation",
        "Forecasting backbone\nGraph temporal model in v1\nFrozen STAEformer later",
        "Base forecast\nand local latent state",
        "Residual and regime\nfeature construction",
        "Matched calibration head\nEntangled VQC or control",
    ]
    fills = [
        COLORS["input"],
        COLORS["classical"],
        COLORS["classical"],
        COLORS["feature"],
        COLORS["feature"],
        COLORS["quantum"],
    ]
    top_boxes = [
        box(top, x, y_top, w, height, label, fill)
        for x, w, label, fill in zip(xs, widths, labels, fills)
    ]
    for left, right in zip(top_boxes[:-1], top_boxes[1:]):
        arrow(
            top,
            (left.get_x() + left.get_width(), left.get_y() + left.get_height() / 2),
            (right.get_x(), right.get_y() + right.get_height() / 2),
        )

    output_box = box(
        top,
        0.805,
        0.25,
        0.18,
        0.20,
        "Point correction and\nscale or interval radii",
        COLORS["output"],
    )
    conformal_box = box(
        top,
        0.58,
        0.25,
        0.17,
        0.20,
        "Split conformal or\nconformal quantile calibration",
        COLORS["output"],
    )
    final_box = box(
        top,
        0.355,
        0.25,
        0.17,
        0.20,
        "Final multiple horizon\nforecast and intervals",
        COLORS["output"],
    )
    gate_box = box(
        top,
        0.095,
        0.25,
        0.20,
        0.20,
        "Validation decision gate\nAccuracy, WIS, coverage\nand seed consistency",
        COLORS["audit"],
    )
    arrow(
        top,
        (
            top_boxes[-1].get_x() + top_boxes[-1].get_width() / 2,
            top_boxes[-1].get_y(),
        ),
        (output_box.get_x() + output_box.get_width() / 2, output_box.get_y() + output_box.get_height()),
    )
    arrow(
        top,
        (output_box.get_x(), output_box.get_y() + output_box.get_height() / 2),
        (
            conformal_box.get_x() + conformal_box.get_width(),
            conformal_box.get_y() + conformal_box.get_height() / 2,
        ),
    )
    arrow(
        top,
        (conformal_box.get_x(), conformal_box.get_y() + conformal_box.get_height() / 2),
        (
            final_box.get_x() + final_box.get_width(),
            final_box.get_y() + final_box.get_height() / 2,
        ),
    )
    arrow(
        top,
        (final_box.get_x(), final_box.get_y() + final_box.get_height() / 2),
        (
            gate_box.get_x() + gate_box.get_width(),
            gate_box.get_y() + gate_box.get_height() / 2,
        ),
    )
    top.text(
        0.095,
        0.12,
        "Canonical test evaluation occurs only after the prespecified gate is passed.",
        fontsize=8,
        ha="left",
        color="#555555",
    )

    bottom = figure.add_subplot(grid[1, 0])
    bottom.set_xlim(0, 1)
    bottom.set_ylim(0, 1)
    bottom.axis("off")
    bottom.text(
        0.0,
        1.04,
        "(b) Parameter matched head comparison and entangled circuit",
        fontsize=10,
        weight="bold",
        ha="left",
    )

    controls = [
        ("MLP\nTanh layers", COLORS["classical"]),
        ("Fourier\nSine and cosine", COLORS["classical"]),
        ("Separable VQC\nNo CNOT gates", COLORS["quantum"]),
        ("Entangled VQC\nRing CNOT gates", COLORS["quantum"]),
    ]
    control_x = [0.01, 0.165, 0.32, 0.475]
    control_boxes = [
        box(bottom, x, 0.49, 0.135, 0.27, text, color, fontsize=8.1)
        for x, (text, color) in zip(control_x, controls)
    ]
    bottom.text(
        0.31,
        0.23,
        "Identical local features, optimization budget, forecast outputs, and objective",
        fontsize=8.1,
        ha="center",
        va="center",
        color="#444444",
    )
    bottom.plot([0.025, 0.595], [0.36, 0.36], color=COLORS["ink"], linewidth=1.0)
    for candidate in control_boxes:
        arrow(
            bottom,
            (candidate.get_x() + candidate.get_width() / 2, 0.36),
            (candidate.get_x() + candidate.get_width() / 2, candidate.get_y()),
            linewidth=0.85,
        )

    stages = [
        ("Linear\ncompression", COLORS["feature"]),
        ("Repeated quantum block\nRY encoding, Rot gates,\nand ring CNOT gates", COLORS["quantum"]),
        ("Pauli Z\nmeasurement", COLORS["quantum"]),
        ("Classical\nreadout", COLORS["output"]),
    ]
    stage_x = [0.66, 0.745, 0.865, 0.94]
    stage_widths = [0.07, 0.105, 0.06, 0.055]
    stage_boxes = []
    for x, width, (label, fill) in zip(stage_x, stage_widths, stages):
        stage_boxes.append(
            box(
                bottom,
                x,
                0.36,
                width,
                0.33,
                label,
                fill,
                fontsize=7.0,
                linewidth=0.85,
            )
        )
    for left, right in zip(stage_boxes[:-1], stage_boxes[1:]):
        arrow(
            bottom,
            (left.get_x() + left.get_width(), left.get_y() + left.get_height() / 2),
            (right.get_x(), right.get_y() + right.get_height() / 2),
            linewidth=0.8,
        )
    bottom.text(
        0.83,
        0.78,
        "Entangled variational quantum circuit",
        fontsize=7.6,
        ha="center",
        color="#555555",
    )
    bottom.text(
        0.83,
        0.17,
        "Reference circuit: four qubits, depth two, analytic state vector simulation",
        fontsize=7.7,
        ha="center",
        color="#555555",
    )

    figure.savefig(
        OUTPUT / "figure00_system_architecture.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        OUTPUT / "figure00_system_architecture.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
