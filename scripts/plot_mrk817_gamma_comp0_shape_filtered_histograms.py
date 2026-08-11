#!/usr/bin/env python3
"""Plot Mrk 817 component-0 posteriors after an L-/centroid shape cut."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DEFAULT_ARCHIVE = Path("Mrk817/runs/mica/legacy_archives/gamma0_100_2comp.zip")
DEFAULT_OUTPUT = Path(
    "Mrk817/results/tf_comparison/"
    "Mrk817_gamma2_comp0_shape_filtered_histograms"
)
BANDS = ("UVM2", "UVW1", "U", "B", "V")
FWHM_PER_W = 2.446386
RATIO_MIN = 0.8
RATIO_MAX = 1.2


def load_band(archive: zipfile.ZipFile, band: str) -> dict[str, np.ndarray]:
    member = (
        f"2comp/run_UVW2_to_{band}_2comp_gamma/data/"
        "posterior_sample1d.txt_2"
    )
    sample = np.loadtxt(io.BytesIO(archive.read(member)), comments="#")
    amplitude0 = np.exp(sample[:, 4])
    width0 = np.exp(sample[:, 6])
    centroid0 = sample[:, 5] + 2.0 * width0
    amplitude1 = np.exp(sample[:, 7])
    centroid1 = sample[:, 8] + 2.0 * np.exp(sample[:, 9])
    fraction0 = amplitude0 / (amplitude0 + amplitude1)
    fwhm0 = FWHM_PER_W * width0
    ratio = 2.0 * width0 / centroid0

    ordered = centroid0 < centroid1
    selected = ordered & (ratio >= RATIO_MIN) & (ratio <= RATIO_MAX)
    return {
        "centroid0_ordered": centroid0[ordered],
        "fwhm0_ordered": fwhm0[ordered],
        "fraction0_ordered": fraction0[ordered],
        "ratio_ordered": ratio[ordered],
        "centroid0_selected": centroid0[selected],
        "fwhm0_selected": fwhm0[selected],
        "fraction0_selected": fraction0[selected],
        "ratio_selected": ratio[selected],
        "n_all": np.array([sample.shape[0]]),
        "n_ordered": np.array([np.sum(ordered)]),
        "n_selected": np.array([np.sum(selected)]),
    }


def plot_edges(
    ordered: np.ndarray, selected: np.ndarray, bins: int = 45
) -> np.ndarray:
    lo, hi = np.quantile(ordered, [0.01, 0.99])
    if selected.size:
        lo = min(lo, float(np.min(selected)))
        hi = max(hi, float(np.max(selected)))
    if hi <= lo:
        lo -= 0.5
        hi += 0.5
    pad = 0.025 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, bins + 1)


def interval_text(values: np.ndarray) -> str:
    if values.size == 0:
        return "no retained samples"
    if values.size == 1:
        return f"single value: {values[0]:.3f}"
    q16, q50, q84 = np.quantile(values, [0.16, 0.5, 0.84])
    return f"{q50:.3f} [{q16:.3f}, {q84:.3f}]"


def write_outputs(data: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    sample_path = output.with_name(output.name + "_posterior.csv")
    summary_path = output.with_name(output.name + "_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["band", "centroid0_days", "fwhm0_days", "f0", "Lminus_over_tau0"]
        )
        for band in BANDS:
            item = data[band]
            for row in zip(
                item["centroid0_selected"],
                item["fwhm0_selected"],
                item["fraction0_selected"],
                item["ratio_selected"],
            ):
                writer.writerow([band, *[f"{value:.10g}" for value in row]])

    fields = [
        "band",
        "n_all",
        "n_tau0_lt_tau1",
        "n_shape_selected",
        "fraction_of_ordered",
        "quantity",
        "q16",
        "median",
        "q84",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for band in BANDS:
            item = data[band]
            n_selected = int(item["n_selected"][0])
            for quantity in ("centroid0", "fwhm0", "fraction0", "ratio"):
                values = item[f"{quantity}_selected"]
                if values.size:
                    q16, q50, q84 = np.quantile(values, [0.16, 0.5, 0.84])
                else:
                    q16 = q50 = q84 = np.nan
                writer.writerow(
                    {
                        "band": band,
                        "n_all": int(item["n_all"][0]),
                        "n_tau0_lt_tau1": int(item["n_ordered"][0]),
                        "n_shape_selected": n_selected,
                        "fraction_of_ordered": (
                            n_selected / int(item["n_ordered"][0])
                        ),
                        "quantity": quantity,
                        "q16": f"{q16:.10g}",
                        "median": f"{q50:.10g}",
                        "q84": f"{q84:.10g}",
                    }
                )


def plot(data: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    ink = "#25282D"
    muted = "#59616B"
    grid = "#E2E6EA"
    baseline = "#8F969E"
    selected_color = "#C46B3B"
    selected_edge = "#7D3D1F"

    quantities = (
        ("centroid0", r"Centroid $\tau_0=c_0+2w_0$ (days)"),
        ("fwhm0", r"Single-component FWHM $=2.446386w_0$ (days)"),
        ("fraction0", r"Amplitude fraction $f_0=A_0/(A_0+A_1)$"),
    )
    fig, axes = plt.subplots(3, 5, figsize=(16.0, 9.2))

    for column, band in enumerate(BANDS):
        item = data[band]
        n_ordered = int(item["n_ordered"][0])
        n_selected = int(item["n_selected"][0])
        for row, (quantity, xlabel) in enumerate(quantities):
            ax = axes[row, column]
            ordered = item[f"{quantity}_ordered"]
            selected = item[f"{quantity}_selected"]
            edges = plot_edges(ordered, selected)

            ax.hist(
                ordered,
                bins=edges,
                density=True,
                histtype="step",
                color=baseline,
                linewidth=1.15,
                label=r"Ordered posterior: $\tau_0<\tau_1$",
            )
            if selected.size >= 2:
                ax.hist(
                    selected,
                    bins=edges,
                    density=True,
                    histtype="stepfilled",
                    color=selected_color,
                    edgecolor=selected_edge,
                    linewidth=0.65,
                    alpha=0.66,
                    label="Shape-selected posterior",
                )
                ax.axvline(
                    np.median(selected), color=selected_edge, lw=1.2, zorder=4
                )
            elif selected.size == 1:
                ax.axvline(
                    selected[0], color=selected_color, lw=2.0, zorder=4
                )
            else:
                ax.text(
                    0.5,
                    0.48,
                    "No retained samples",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=selected_edge,
                    weight="bold",
                )

            ax.set_xlim(edges[0], edges[-1])
            ax.set_xlabel(xlabel, fontsize=8.6)
            ax.grid(True, axis="y", color=grid, linewidth=0.65, alpha=0.85)
            ax.set_axisbelow(True)
            ax.tick_params(direction="in", which="both", top=True, right=True,
                           labelsize=8)
            ax.minorticks_on()
            ax.text(
                0.04,
                0.94,
                interval_text(selected),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.9,
                color=ink,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": grid,
                    "alpha": 0.90,
                },
            )

        axes[0, column].set_title(
            f"{band}\nN={n_selected}/{n_ordered} "
            f"({n_selected/n_ordered:.1%})",
            fontsize=10.8,
            color=ink,
            weight="bold",
        )

    for row in range(3):
        axes[row, 0].set_ylabel("Conditional posterior density", fontsize=9)

    handles = [
        Line2D([0], [0], color=baseline, lw=1.3,
               label=r"Before shape cut: $\tau_0<\tau_1$"),
        Patch(facecolor=selected_color, edgecolor=selected_edge, alpha=0.66,
              label=r"After $0.8\leq L_-/\tau_0\leq1.2$"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.983),
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    fig.suptitle(
        "Mrk 817 two-Gamma posterior: component-0 shape-selected distributions",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=15,
        color=ink,
    )
    fig.text(
        0.055,
        0.946,
        r"Base selection: $\tau_0<\tau_1$. Additional shape cut: "
        r"$0.8\leq L_-/\tau_0=2w_0/(c_0+2w_0)\leq1.2$, equivalent to "
        r"$-w_0/3\leq c_0\leq w_0/2$. Histograms are density-normalized.",
        ha="left",
        fontsize=9.2,
        color=muted,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.07,
        top=0.885,
        wspace=0.27,
        hspace=0.46,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as archive:
        data = {band: load_band(archive, band) for band in BANDS}
    plot(data, args.output)
    write_outputs(data, args.output)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_name(args.output.name + "_posterior.csv"))
    print(args.output.with_name(args.output.name + "_summary.csv"))


if __name__ == "__main__":
    main()
