#!/usr/bin/env python3
"""Plot weighted parameter posteriors for all ten completed F9 recovery fits.

The two native Gamma component tuples are relabelled sample-by-sample by
increasing centroid.  Every raw nested-sampling row and its final importance
weight is retained.  For readability, each histogram displays the central
99% weighted x-range and folds the remaining <=1% mass into the two edge
bins.  Width histograms use equal bins in log(width).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from collect_mica_results import (  # noqa: E402
    component_parameter_arrays,
    param_names,
    reorder_component_columns,
    weighted_quantile,
)


OUT = HERE / "outputs" / "recovery_parameter_histograms_20260910"
CASES = [
    "disk_only",
    "response_within",
    "random_within",
    "response_beyond",
    "random_beyond",
]
CASE_LABELS = {
    "disk_only": "Disk only",
    "response_within": "Response: centroid 60 d",
    "random_within": "Independent: shorter scale",
    "response_beyond": "Response: centroid 800 d",
    "random_beyond": "Independent: ultra-slow",
}
BANDS = ["M2", "B"]
BLUE = "#2457A6"
ORANGE = "#C67C18"
GREEN = "#2F6B4F"
INK = "#252525"
LIGHT = "#D9D9D9"
Q68 = np.array([0.15865, 0.5, 0.84135])
QVIEW = np.array([0.005, 0.995])


def normalized_weights(table: np.ndarray) -> np.ndarray:
    weights = np.asarray(table[:, -1], dtype=float)
    valid = np.isfinite(weights) & (weights >= 0)
    if not valid.all() or weights.sum() <= 0:
        raise ValueError("Invalid MICA posterior weights")
    return weights / weights.sum()


def read_truth() -> dict[tuple[str, str], dict]:
    payload = json.loads((HERE / "outputs" / "truth.json").read_text())
    records: dict[tuple[str, str], dict] = {}
    for case in payload["cases"]:
        for band, values in case["bands"].items():
            records[(case["name"], band)] = values
    disk_truth = json.loads((HERE / "cases" / "disk_only" / "truth.json").read_text())
    # disk_only/truth.json has the same case-record layout used below.
    if "bands" in disk_truth:
        for band, values in disk_truth["bands"].items():
            records[("disk_only", band)] = values
    return records


def read_posterior(case: str, band: str) -> dict[str, np.ndarray]:
    run = (
        HERE
        / "cases"
        / case
        / "runs"
        / "mica"
        / "2comp"
        / f"run_W2_to_{band}_2comp_gamma"
    )
    data = run / "data"
    sample = np.loadtxt(data / "sample1d.txt_2", ndmin=2)
    weights = normalized_weights(np.loadtxt(data / "weights.txt", ndmin=2))
    names = param_names(data / "para_names_line.txt_2")
    if len(sample) != len(weights):
        raise ValueError(f"Sample/weight mismatch: {run}")
    ordered, _ = reorder_component_columns(
        sample, names, 2, "gamma", {"component_order": "moment1"}
    )
    _, width, amplitude, centroid = component_parameter_arrays(
        ordered, names, 2, "gamma"
    )
    fraction = amplitude / amplitude.sum(axis=0)
    return {
        "weights": weights,
        "centroid": centroid,
        "width": width,
        "fraction": fraction,
    }


def truth_values(case: str, band: str, truth: dict[tuple[str, str], dict]):
    values = truth[(case, band)]
    short_centroid = float(values["short_relative_centroid_days"])
    if case == "disk_only":
        return [[short_centroid, None, 1.0], [None, None, 0.0]]
    if case.startswith("response"):
        gain = float(values["slow_scale"])
        long_fraction = gain / (1.0 + gain)
        return [
            [short_centroid, None, 1.0 - long_fraction],
            [
                float(values["injected_slow_centroid_days"]),
                float(values["injected_slow_width_days"]),
                long_fraction,
            ],
        ]
    return [[short_centroid, None, None], [None, None, None]]


def draw_histogram(ax, values, weights, color, log_x=False, truth=None):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    lo, hi = weighted_quantile(values, weights, QVIEW)
    if not np.isfinite(lo + hi) or hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if log_x:
        lo = max(float(lo), np.nextafter(0.0, 1.0))
        bins = np.geomspace(lo, float(hi), 49)
    else:
        pad = 0.035 * (float(hi) - float(lo))
        lo, hi = float(lo) - pad, float(hi) + pad
        bins = np.linspace(lo, hi, 49)
    clipped = np.clip(values, bins[0], bins[-1])
    mass, _ = np.histogram(clipped, bins=bins, weights=weights)
    ax.stairs(mass, bins, fill=True, color=color, alpha=0.26, linewidth=0.9)
    ax.stairs(mass, bins, color=color, linewidth=1.15)
    q16, median, q84 = weighted_quantile(values, weights, Q68)
    ax.axvspan(q16, q84, color=LIGHT, alpha=0.42, zorder=0)
    ax.axvline(median, color=INK, linewidth=1.05)
    if truth is not None and np.isfinite(truth):
        ax.axvline(truth, color=GREEN, linestyle="--", linewidth=1.1)
    if log_x:
        ax.set_xscale("log")
        ticks = np.geomspace(bins[0], bins[-1], 3)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{tick:.2g}" for tick in ticks])
        ax.minorticks_off()
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.text(
        0.98,
        0.93,
        f"{median:.3g} [{q16:.3g}, {q84:.3g}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=INK,
    )


def make_band_figure(band: str, truth: dict[tuple[str, str], dict]):
    columns = [
        (0, "centroid", "Short centroid [d]", False),
        (0, "width", "Short Gamma scale w [d]", True),
        (0, "fraction", "Short area fraction", False),
        (1, "centroid", "Long centroid [d]", False),
        (1, "width", "Long Gamma scale w [d]", True),
        (1, "fraction", "Long area fraction", False),
    ]
    figure, axes = plt.subplots(5, 6, figsize=(19.2, 13.2), squeeze=False)
    for row, case in enumerate(CASES):
        posterior = read_posterior(case, band)
        references = truth_values(case, band, truth)
        for col, (component, key, title, log_x) in enumerate(columns):
            ax = axes[row, col]
            reference_index = {"centroid": 0, "width": 1, "fraction": 2}[key]
            draw_histogram(
                ax,
                posterior[key][component],
                posterior["weights"],
                BLUE if component == 0 else ORANGE,
                log_x=log_x,
                truth=references[component][reference_index],
            )
            if row == 0:
                ax.set_title(title, fontsize=10.2, pad=8)
            if col == 0:
                ax.set_ylabel(CASE_LABELS[case], fontsize=9.5, labelpad=14)
            ax.tick_params(axis="x", labelsize=7.4)
    figure.suptitle(
        f"F9 {band}: weighted two-Gamma parameter posteriors for recovery experiments",
        fontsize=16,
        y=0.985,
    )
    figure.text(
        0.5,
        0.957,
        "Rows are injection conditions; components are relabelled in every joint sample by increasing centroid. "
        "Black: median; grey: central 68.27%; green dashed: available injected benchmark.",
        ha="center",
        fontsize=9.2,
    )
    figure.text(
        0.5,
        0.018,
        "All 10,000 raw rows retain their final importance weights. Each panel uses its own central 99% x-window; "
        "the remaining <=1% posterior mass is retained in edge bins. Width axes are logarithmic. "
        "Independent-signal cases have no true long response parameters.",
        ha="center",
        fontsize=8.2,
    )
    figure.subplots_adjust(left=0.09, right=0.988, top=0.925, bottom=0.065, hspace=0.50, wspace=0.30)
    return figure


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth = read_truth()
    figures = []
    for band in BANDS:
        figure = make_band_figure(band, truth)
        figure.savefig(OUT / f"F9_{band}_recovery_parameter_histograms.png", dpi=190)
        figures.append(figure)
    with PdfPages(OUT / "F9_recovery_parameter_histograms.pdf") as pdf:
        for figure in figures:
            pdf.savefig(figure)
            plt.close(figure)
    print(OUT)


if __name__ == "__main__":
    main()
