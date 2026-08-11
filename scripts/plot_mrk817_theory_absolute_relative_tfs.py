#!/usr/bin/env python3
"""Plot Mrk 817 absolute disk TFs and fitted UVW2-relative Gamma kernels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "Mrk817/results/tf_comparison/theory_forward_gamma"
)
DEFAULT_OUTPUT = (
    DEFAULT_INPUT / "Mrk817_theory_absolute_and_relative_transfer_functions"
)

PARAMETER_SET = "mrk817_lewin2024"
MODES = ("finite_difference", "linearized")
BANDS = ("UVW2", "UVM2", "UVW1", "U", "B", "V")
BAND_LABELS = {
    "UVW2": "UVW2 (1928 Å)",
    "UVM2": "UVM2 (2246 Å)",
    "UVW1": "UVW1 (2600 Å)",
    "U": "U (3465 Å)",
    "B": "B (4392 Å)",
    "V": "V (5468 Å)",
}
BAND_COLORS = {
    "UVW2": "#25282D",
    "UVM2": "#4878A8",
    "UVW1": "#5A9A55",
    "U": "#D47A15",
    "B": "#C65353",
    "V": "#9A6A91",
}
MODE_STYLE = {
    "finite_difference": {
        "label": "finite difference",
        "color": "#D47A15",
        "linestyle": "-",
        "linewidth": 2.15,
    },
    "linearized": {
        "label": "linearized",
        "color": "#4878A8",
        "linestyle": "--",
        "linewidth": 1.75,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def gamma_k2(delay: np.ndarray, c: float, w: float) -> np.ndarray:
    """Unit-area shifted shape-2 Gamma density."""
    value = np.zeros_like(delay)
    offset = delay - c
    mask = offset >= 0.0
    scaled = offset[mask] / w
    value[mask] = scaled * np.exp(-scaled) / w
    return value


def plot_absolute_panel(
    axis: plt.Axes,
    absolute: pd.DataFrame,
) -> None:
    for band in BANDS:
        for mode in MODES:
            item = absolute[
                (absolute["band"] == band)
                & (absolute["response_mode"] == mode)
            ].sort_values("time_days")
            axis.plot(
                item["time_days"],
                item["density_per_day"],
                color=BAND_COLORS[band],
                linestyle=MODE_STYLE[mode]["linestyle"],
                linewidth=(2.25 if mode == "finite_difference" else 1.25),
                alpha=(0.96 if mode == "finite_difference" else 0.78),
            )

    axis.set_xlim(0.0, 5.0)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Absolute delay [day]")
    axis.set_ylabel(r"Unit-area absolute TF $\psi_\lambda(t)$ [day$^{-1}$]")
    axis.set_title(
        "Absolute theoretical disk transfer functions",
        loc="left",
        fontsize=13,
        pad=9,
    )

    band_handles = [
        Line2D(
            [0],
            [0],
            color=BAND_COLORS[band],
            lw=2.4,
            label=BAND_LABELS[band],
        )
        for band in BANDS
    ]
    mode_handles = [
        Line2D(
            [0],
            [0],
            color="#30343A",
            lw=1.8,
            ls=MODE_STYLE[mode]["linestyle"],
            label=MODE_STYLE[mode]["label"],
        )
        for mode in MODES
    ]
    first_legend = axis.legend(
        handles=band_handles,
        ncol=3,
        frameon=False,
        fontsize=9.3,
        loc="upper right",
        columnspacing=1.2,
        handlelength=2.5,
    )
    axis.add_artist(first_legend)
    axis.legend(
        handles=mode_handles,
        frameon=False,
        fontsize=9.3,
        loc="center right",
        bbox_to_anchor=(1.0, 0.50),
    )


def plot_delta_panel(axis: plt.Axes) -> None:
    axis.axvspan(-0.75, 0.0, color="#ECEDEF", zorder=0)
    axis.axvline(0.0, color="#7C8188", lw=0.9, zorder=1)
    axis.annotate(
        "",
        xy=(0.0, 0.88),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": "#25282D", "lw": 2.0},
    )
    axis.text(
        0.05,
        0.72,
        r"$K_{\rm W2\leftarrow W2}(t)=\delta(t)$",
        fontsize=10,
        color="#25282D",
    )
    axis.text(
        0.05,
        0.46,
        "unit area; arrow height is schematic",
        fontsize=8.5,
        color="#5A6068",
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([])


def plot_relative_panel(
    axis: plt.Axes,
    band: str,
    summary: pd.DataFrame,
    delay: np.ndarray,
    output_rows: list[dict[str, float | str]],
) -> None:
    axis.axvspan(delay.min(), 0.0, color="#ECEDEF", zorder=0)
    axis.axvline(0.0, color="#7C8188", lw=0.9, zorder=1)

    ymax = 0.0
    annotations: list[str] = []
    for mode in MODES:
        row = summary[
            (summary["target_band"] == band)
            & (summary["response_mode"] == mode)
        ].iloc[0]
        c = float(row["c_days"])
        w = float(row["w_days"])
        amplitude = float(row["amplitude_A"])
        tau = float(row["gamma_centroid_days"])
        kernel = gamma_k2(delay, c, w)
        ymax = max(ymax, float(kernel.max()))
        style = MODE_STYLE[mode]
        axis.plot(
            delay,
            kernel,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
        short = "FD" if mode == "finite_difference" else "LIN"
        annotations.append(
            rf"{short}: $c={c:.2f}$, $w={w:.2f}$, "
            rf"$\tau={tau:.2f}$, $A={amplitude:.2f}$"
        )
        for time_value, density in zip(delay, kernel):
            output_rows.append(
                {
                    "parameter_set": PARAMETER_SET,
                    "response_mode": mode,
                    "driver_band": "UVW2",
                    "target_band": band,
                    "delay_days": float(time_value),
                    "unit_area_relative_kernel_per_day": float(density),
                    "amplitude_A": amplitude,
                    "effective_relative_kernel_per_day": float(
                        amplitude * density
                    ),
                    "c_days": c,
                    "w_days": w,
                    "gamma_centroid_days": tau,
                }
            )

    axis.set_ylim(0.0, ymax * 1.18)
    axis.text(
        0.98,
        0.96,
        "\n".join(annotations),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.1,
        color="#33373D",
        linespacing=1.25,
    )


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9DDE1", lw=0.7, alpha=0.72)
    axis.tick_params(direction="out", length=4, width=0.8)


def main() -> None:
    args = parse_args()
    absolute_path = args.input / "theory_absolute_transfer_functions.csv"
    summary_path = args.input / "theory_forward_gamma_summary.csv"
    absolute = pd.read_csv(absolute_path)
    summary = pd.read_csv(summary_path)

    absolute = absolute[absolute["parameter_set"] == PARAMETER_SET].copy()
    summary = summary[
        (summary["parameter_set"] == PARAMETER_SET)
        & (summary["fit_domain"] == "full")
    ].copy()
    if absolute.empty or len(summary) != 10:
        raise RuntimeError("Expected two modes for six absolute and five relative TFs.")

    delay = np.linspace(-0.75, 4.0, 3801)
    output_rows: list[dict[str, float | str]] = []

    figure = plt.figure(figsize=(14.2, 10.0))
    grid = figure.add_gridspec(
        3,
        3,
        height_ratios=(1.50, 1.0, 1.0),
        hspace=0.40,
        wspace=0.26,
    )
    absolute_axis = figure.add_subplot(grid[0, :])
    plot_absolute_panel(absolute_axis, absolute)
    style_axis(absolute_axis)

    relative_axes = [
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[1, 2]),
        figure.add_subplot(grid[2, 0]),
        figure.add_subplot(grid[2, 1]),
        figure.add_subplot(grid[2, 2]),
    ]
    plot_delta_panel(relative_axes[0])
    for axis, band in zip(relative_axes[1:], BANDS[1:]):
        plot_relative_panel(axis, band, summary, delay, output_rows)

    for axis, band in zip(relative_axes, BANDS):
        axis.set_xlim(delay.min(), delay.max())
        axis.set_title(BAND_LABELS[band], fontsize=11.3, pad=6)
        axis.set_xlabel("Relative delay to UVW2 [day]")
        style_axis(axis)
    relative_axes[0].set_ylabel(r"Relative TF $K_{\lambda\leftarrow W2}(t)$")
    relative_axes[3].set_ylabel(
        r"Unit-area relative TF $K_{\lambda\leftarrow W2}(t)$ [day$^{-1}$]"
    )

    figure.suptitle(
        "Mrk 817 Swift theoretical absolute and UVW2-relative transfer functions",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.965,
        "Lewin et al. (2024) temperature/inclination parameters; "
        "relative panels show full-domain fitted positive k=2 Gamma kernels",
        ha="center",
        va="top",
        fontsize=10,
        color="#50555C",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.925, bottom=0.065)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    pdf_path = args.output.with_suffix(".pdf")
    csv_path = args.output.with_name(args.output.name + "_relative_kernels.csv")
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    print(png_path)
    print(pdf_path)
    print(csv_path)


if __name__ == "__main__":
    main()
