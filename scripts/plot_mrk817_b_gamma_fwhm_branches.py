#!/usr/bin/env python3
"""Compare B-band two-Gamma posterior branches split by component-0 FWHM."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ARCHIVE = Path("Mrk817/runs/mica/legacy_archives/gamma0_100_2comp.zip")
DEFAULT_OUTPUT = Path(
    "Mrk817/results/tf_comparison/Mrk817_B_gamma2_comp0_fwhm_branches"
)
MEMBER = (
    "2comp/run_UVW2_to_B_2comp_gamma/data/posterior_sample1d.txt_2"
)
FWHM_PER_W = 2.446386


def load(archive_path: Path, threshold: float) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]
]:
    with zipfile.ZipFile(archive_path) as archive:
        sample = np.loadtxt(io.BytesIO(archive.read(MEMBER)), comments="#")
    if sample.ndim != 2 or sample.shape[1] < 10:
        raise ValueError(f"expected at least 10 posterior columns in {MEMBER}")

    amplitude0 = np.exp(sample[:, 4])
    center0 = sample[:, 5]
    width0 = np.exp(sample[:, 6])
    amplitude1 = np.exp(sample[:, 7])
    center1 = sample[:, 8]
    width1 = np.exp(sample[:, 9])
    centroid0 = center0 + 2.0 * width0
    centroid1 = center1 + 2.0 * width1
    ordered = centroid0 < centroid1
    fwhm0 = FWHM_PER_W * width0

    quantities = {
        "center0": center0,
        "mode0": center0 + width0,
        "centroid0": centroid0,
        "fwhm0": fwhm0,
        "fraction0": amplitude0 / (amplitude0 + amplitude1),
        "amplitude0": amplitude0,
        "amplitude1": amplitude1,
        "amplitude_total": amplitude0 + amplitude1,
        "center1": center1,
        "mode1": center1 + width1,
        "centroid1": centroid1,
        "fwhm1": FWHM_PER_W * width1,
    }
    masks = {
        "narrow": ordered & (fwhm0 < threshold),
        "broad": ordered & (fwhm0 > threshold),
    }
    branches = {
        branch: {key: values[mask] for key, values in quantities.items()}
        for branch, mask in masks.items()
    }
    counts = {
        "all": sample.shape[0],
        "ordered": int(np.sum(ordered)),
        "narrow": int(np.sum(masks["narrow"])),
        "broad": int(np.sum(masks["broad"])),
    }
    return branches, masks, counts


def fd_edges(values: np.ndarray, minimum: int = 24, maximum: int = 100):
    plot_lo, plot_hi = np.quantile(values, [0.01, 0.99])
    visible = values[(values >= plot_lo) & (values <= plot_hi)]
    q25, q75 = np.quantile(visible, [0.25, 0.75])
    width = 2.0 * (q75 - q25) / np.cbrt(values.size)
    span = plot_hi - plot_lo
    if width <= 0.0 or span <= 0.0:
        bins = minimum
    else:
        bins = int(np.clip(np.ceil(span / width), minimum, maximum))
    return np.linspace(plot_lo, plot_hi, bins + 1)


def interval(values: np.ndarray, digits: int = 3) -> str:
    q16, q50, q84 = np.quantile(values, [0.16, 0.5, 0.84])
    return f"{q50:.{digits}f} [{q16:.{digits}f}, {q84:.{digits}f}]"


def write_summary(
    branches: dict[str, dict[str, np.ndarray]],
    counts: dict[str, int],
    output: Path,
) -> None:
    path = output.with_name(output.name + "_summary.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "branch",
        "n",
        "fraction_of_ordered",
        "parameter",
        "q16",
        "median",
        "q84",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for branch, quantities in branches.items():
            for parameter, values in quantities.items():
                q16, q50, q84 = np.quantile(values, [0.16, 0.5, 0.84])
                writer.writerow(
                    {
                        "branch": branch,
                        "n": counts[branch],
                        "fraction_of_ordered": counts[branch] / counts["ordered"],
                        "parameter": parameter,
                        "q16": f"{q16:.10g}",
                        "median": f"{q50:.10g}",
                        "q84": f"{q84:.10g}",
                    }
                )


def plot(
    branches: dict[str, dict[str, np.ndarray]],
    counts: dict[str, int],
    threshold: float,
    output: Path,
) -> None:
    ink = "#25282D"
    muted = "#59616B"
    grid = "#E2E6EA"
    colors = {"narrow": "#C46B3B", "broad": "#3B6F91"}
    labels = {
        "narrow": rf"Narrow: FWHM$_0<{threshold:g}$ d",
        "broad": rf"Broad: FWHM$_0>{threshold:g}$ d",
    }
    panels = (
        ("center0", r"Comp0 onset $c_0$ (days)"),
        ("mode0", r"Comp0 mode $c_0+w_0$ (days)"),
        ("centroid0", r"Comp0 centroid $c_0+2w_0$ (days)"),
        ("fwhm0", r"Comp0 FWHM (days)"),
        ("fraction0", r"Fraction $f_0=A_0/(A_0+A_1)$"),
        ("amplitude0", r"Amplitude $A_0$"),
        ("amplitude1", r"Amplitude $A_1$"),
        ("amplitude_total", r"Total amplitude $A_0+A_1$"),
        ("center1", r"Comp1 onset $c_1$ (days)"),
        ("mode1", r"Comp1 mode $c_1+w_1$ (days)"),
        ("centroid1", r"Comp1 centroid $c_1+2w_1$ (days)"),
        ("fwhm1", r"Comp1 FWHM (days)"),
    )

    fig, axes = plt.subplots(3, 4, figsize=(15.6, 10.0))
    legend_handles = []
    for ax, (parameter, title) in zip(axes.ravel(), panels):
        combined = np.concatenate(
            [branches["narrow"][parameter], branches["broad"][parameter]]
        )
        edges = fd_edges(combined)
        for branch, linestyle in (("narrow", "--"), ("broad", "-")):
            values = branches[branch][parameter]
            filled = ax.hist(
                values,
                bins=edges,
                density=True,
                histtype="stepfilled",
                color=colors[branch],
                alpha=0.20,
            )
            outlined = ax.hist(
                values,
                bins=edges,
                density=True,
                histtype="step",
                color=colors[branch],
                linewidth=1.25,
                linestyle=linestyle,
                label=labels[branch],
            )
            ax.axvline(
                np.median(values),
                color=colors[branch],
                linewidth=1.1,
                linestyle=linestyle,
            )
            if not legend_handles:
                legend_handles.append(outlined[2][0])
            elif len(legend_handles) == 1 and branch == "broad":
                legend_handles.append(outlined[2][0])

        ax.set_title(title, loc="left", fontsize=10.3, color=ink, weight="bold")
        ax.grid(True, axis="y", color=grid, linewidth=0.65, alpha=0.85)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", which="both", top=True, right=True,
                       labelsize=8.3)
        ax.minorticks_on()
        ax.text(
            0.98,
            0.96,
            "n: " + interval(branches["narrow"][parameter]) + "\n"
            "b: " + interval(branches["broad"][parameter]),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.7,
            color=ink,
            bbox={
                "boxstyle": "round,pad=0.23",
                "facecolor": "white",
                "edgecolor": grid,
                "alpha": 0.88,
            },
        )
        ax.set_xlim(edges[0], edges[-1])

    for row in range(3):
        axes[row, 0].set_ylabel("Conditional posterior density", fontsize=9)

    fig.legend(
        legend_handles,
        [
            f"Narrow branch: N={counts['narrow']} "
            f"({counts['narrow']/counts['ordered']:.1%})",
            f"Broad branch: N={counts['broad']} "
            f"({counts['broad']/counts['ordered']:.1%})",
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.982),
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    fig.suptitle(
        "Mrk 817 B-band two-Gamma posterior: component-0 width branches",
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=15,
        color=ink,
    )
    fig.text(
        0.055,
        0.953,
        r"Base selection: $\tau_0<\tau_1$. Branch split uses the "
        rf"single-component FWHM$_0={FWHM_PER_W:.6f}w_0$ at {threshold:g} d."
        "\n"
        "Each branch is density-normalized; axes show the combined 1st--99th "
        "percentile and annotations use all samples: median [16th, 84th].",
        ha="left",
        va="top",
        fontsize=9.3,
        color=muted,
        linespacing=1.25,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.06,
        top=0.865,
        wspace=0.28,
        hspace=0.32,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    branches, _, counts = load(args.archive, args.threshold)
    plot(branches, counts, args.threshold, args.output)
    write_summary(branches, counts, args.output)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_name(args.output.name + "_summary.csv"))


if __name__ == "__main__":
    main()
