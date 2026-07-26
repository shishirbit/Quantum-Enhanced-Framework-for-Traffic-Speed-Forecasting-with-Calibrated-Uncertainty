"""Generate the publication system architecture for QUARTS."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_OUTPUT = ROOT / "manuscript" / "quarts_springer" / "figures"
PUBLICATION_OUTPUT = ROOT / "artifacts" / "publication" / "figures"

COLORS = {
    "data": "#e8f1f8",
    "backbone": "#d9ead3",
    "feature": "#fff2cc",
    "quantum": "#eadcf8",
    "output": "#fce5cd",
    "decision": "#eeeeee",
    "ink": "#20242a",
    "muted": "#50565e",
    "bus": "#48515c",
}


def add_box(
    axis,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    fontsize: float = 9.0,
    linewidth: float = 1.05,
):
    """Add a consistently styled box and return its patch."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        facecolor=color,
        edgecolor=COLORS["ink"],
        linewidth=linewidth,
        zorder=3,
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
        linespacing=1.18,
        zorder=4,
    )
    return patch


def anchor(patch, side: str) -> tuple[float, float]:
    """Return the midpoint of one box boundary."""
    x, y = patch.get_x(), patch.get_y()
    width, height = patch.get_width(), patch.get_height()
    points = {
        "left": (x, y + height / 2),
        "right": (x + width, y + height / 2),
        "top": (x + width / 2, y + height),
        "bottom": (x + width / 2, y),
    }
    return points[side]


def add_arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    linewidth: float = 1.25,
):
    """Draw a straight arrow between boundary anchors."""
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        color=COLORS["ink"],
        shrinkA=4,
        shrinkB=4,
        zorder=2,
    )
    axis.add_patch(patch)
    return patch


def add_elbow_arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    bend_y: float,
    linewidth: float = 1.25,
):
    """Draw an orthogonal connector with one horizontal segment."""
    start_x, start_y = start
    end_x, end_y = end
    vertices = [
        (start_x, start_y),
        (start_x, bend_y),
        (end_x, bend_y),
        (end_x, end_y),
    ]
    path = MplPath(
        vertices,
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
    )
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        color=COLORS["ink"],
        zorder=2,
    )
    axis.add_patch(patch)
    return patch


def build_main_workflow(axis) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.0,
        1.015,
        "(a) End to end forecasting and uncertainty calibration",
        fontsize=10.4,
        weight="bold",
        ha="left",
        color=COLORS["ink"],
    )

    top_y, top_h, top_w = 0.69, 0.22, 0.19
    top_x = [0.02, 0.275, 0.53, 0.785]
    top_labels = [
        "Historical traffic tensor\nRoad graph\nTemporal covariates",
        "Chronological\npreprocessing\nTraining only scaling\nTarget masking",
        "Forecasting backbone\nGraph temporal model\nor frozen STAEformer",
        "Base forecast and\nnode level latent state",
    ]
    top_colors = [
        COLORS["data"],
        COLORS["backbone"],
        COLORS["backbone"],
        COLORS["feature"],
    ]
    top_boxes = [
        add_box(axis, x, top_y, top_w, top_h, label, color)
        for x, label, color in zip(top_x, top_labels, top_colors)
    ]
    for left, right in zip(top_boxes[:-1], top_boxes[1:]):
        add_arrow(axis, anchor(left, "right"), anchor(right, "left"))

    lower_y, lower_h, lower_w = 0.28, 0.22, 0.14
    lower_x = [0.835, 0.67, 0.505, 0.34, 0.175, 0.01]
    lower_labels = [
        "Residual and\nregime features",
        "Matched calibration\nhead",
        "Point correction,\nscale, or radii",
        "Chronological\nconformal adjustment",
        "Final forecast and\npredictive intervals",
        "Validation gate\nAccuracy, WIS,\ncoverage, seeds",
    ]
    lower_colors = [
        COLORS["feature"],
        COLORS["quantum"],
        COLORS["output"],
        COLORS["output"],
        COLORS["output"],
        COLORS["decision"],
    ]
    lower_boxes = [
        add_box(
            axis,
            x,
            lower_y,
            lower_w,
            lower_h,
            label,
            color,
            fontsize=8.4,
        )
        for x, label, color in zip(lower_x, lower_labels, lower_colors)
    ]

    add_elbow_arrow(
        axis,
        anchor(top_boxes[-1], "bottom"),
        anchor(lower_boxes[0], "top"),
        bend_y=0.585,
    )
    for right, left in zip(lower_boxes[:-1], lower_boxes[1:]):
        add_arrow(axis, anchor(right, "left"), anchor(left, "right"))

    axis.text(
        0.01,
        0.12,
        "The canonical test split is evaluated only after the prespecified validation gate passes.",
        fontsize=8.2,
        ha="left",
        color=COLORS["muted"],
    )


def build_matched_controls(axis) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.0,
        1.02,
        "(b) Matched calibration heads and the entangled circuit",
        fontsize=10.4,
        weight="bold",
        ha="left",
        color=COLORS["ink"],
    )

    axis.text(
        0.255,
        0.91,
        "Matched head comparison",
        fontsize=8.8,
        weight="bold",
        ha="center",
        color=COLORS["ink"],
    )
    input_box = add_box(
        axis,
        0.015,
        0.48,
        0.13,
        0.22,
        "Identical latent\nfeatures",
        COLORS["feature"],
        fontsize=8.4,
    )
    head_x = [0.175, 0.285, 0.395, 0.505]
    head_labels = [
        "MLP\nTanh",
        "Fourier\nSin and cos",
        "Separable VQC\nNo CNOT",
        "Entangled VQC\nRing CNOT",
    ]
    head_colors = [
        COLORS["backbone"],
        COLORS["backbone"],
        COLORS["quantum"],
        COLORS["quantum"],
    ]
    head_boxes = [
        add_box(axis, x, 0.48, 0.095, 0.22, label, color, fontsize=7.8)
        for x, label, color in zip(head_x, head_labels, head_colors)
    ]
    output_box = add_box(
        axis,
        0.64,
        0.48,
        0.13,
        0.22,
        "Identical outputs\nand objective",
        COLORS["output"],
        fontsize=8.4,
    )

    input_bus_x = 0.155
    output_bus_x = 0.62
    bus_y_top = 0.76
    bus_y_bottom = 0.42
    axis.plot(
        [input_bus_x, input_bus_x],
        [bus_y_bottom, bus_y_top],
        color=COLORS["bus"],
        linewidth=1.15,
        zorder=1,
    )
    axis.plot(
        [input_bus_x, 0.595],
        [bus_y_top, bus_y_top],
        color=COLORS["bus"],
        linewidth=1.15,
        zorder=1,
    )
    axis.plot(
        [0.155, output_bus_x],
        [bus_y_bottom, bus_y_bottom],
        color=COLORS["bus"],
        linewidth=1.15,
        zorder=1,
    )
    axis.plot(
        [output_bus_x, output_bus_x],
        [bus_y_bottom, bus_y_top],
        color=COLORS["bus"],
        linewidth=1.15,
        zorder=1,
    )
    add_arrow(axis, anchor(input_box, "right"), (input_bus_x, 0.59), linewidth=1.05)
    for candidate in head_boxes:
        center_x = anchor(candidate, "top")[0]
        add_arrow(axis, (center_x, bus_y_top), anchor(candidate, "top"), linewidth=1.0)
        add_arrow(
            axis,
            anchor(candidate, "bottom"),
            (center_x, bus_y_bottom),
            linewidth=1.0,
        )
    add_arrow(axis, (output_bus_x, 0.59), anchor(output_box, "left"), linewidth=1.05)

    axis.text(
        0.385,
        0.31,
        "Same features, parameter scale, training budget, horizons, and loss",
        fontsize=7.8,
        ha="center",
        color=COLORS["muted"],
    )

    axis.plot(
        [0.795, 0.795],
        [0.13, 0.91],
        color="#b7bbc0",
        linewidth=0.9,
        zorder=1,
    )
    axis.text(
        0.895,
        0.91,
        "Entangled VQC",
        fontsize=8.8,
        weight="bold",
        ha="center",
        color=COLORS["ink"],
    )
    stage_x, stage_w, stage_h = 0.82, 0.16, 0.135
    stage_y = [0.70, 0.51, 0.32, 0.13]
    stage_labels = [
        "Linear\nmap",
        "RY and Rot reuploading",
        "Ring CNOT entanglement",
        "Pauli Z measurement\nand classical readout",
    ]
    stage_colors = [
        COLORS["feature"],
        COLORS["quantum"],
        COLORS["quantum"],
        COLORS["output"],
    ]
    stage_boxes = [
        add_box(
            axis,
            stage_x,
            y,
            stage_w,
            stage_h,
            label,
            color,
            fontsize=7.6,
            linewidth=0.9,
        )
        for y, label, color in zip(stage_y, stage_labels, stage_colors)
    ]
    for upper, lower in zip(stage_boxes[:-1], stage_boxes[1:]):
        add_arrow(
            axis,
            anchor(upper, "bottom"),
            anchor(lower, "top"),
            linewidth=0.95,
        )


def main() -> None:
    MANUSCRIPT_OUTPUT.mkdir(parents=True, exist_ok=True)
    PUBLICATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(8.2, 7.3))
    grid = figure.add_gridspec(2, 1, height_ratios=(1.15, 0.85), hspace=0.22)
    build_main_workflow(figure.add_subplot(grid[0, 0]))
    build_matched_controls(figure.add_subplot(grid[1, 0]))
    figure.subplots_adjust(left=0.025, right=0.985, top=0.96, bottom=0.045)

    for output in (MANUSCRIPT_OUTPUT, PUBLICATION_OUTPUT):
        figure.savefig(
            output / "figure00_system_architecture.pdf",
            bbox_inches="tight",
        )
        figure.savefig(
            output / "figure00_system_architecture.png",
            dpi=320,
            bbox_inches="tight",
        )
        figure.savefig(
            output / "figure00_system_architecture.svg",
            bbox_inches="tight",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
