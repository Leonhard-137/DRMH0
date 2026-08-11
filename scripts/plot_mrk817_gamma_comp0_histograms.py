#!/usr/bin/env python3
"""Plot component-0 posterior histograms for the Mrk 817 two-Gamma fits."""

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
    "Mrk817/results/tf_comparison/Mrk817_gamma2_comp0_histograms"
)
BANDS = ("UVM2", "UVW1", "U", "B", "V")
FWHM_PER_W = 2.446386


def load_band(archive: zipfile.ZipFile, band: str) -> dict[str, np.ndarray]:
    member = (
        f"2comp/run_UVW2_to_{band}_2comp_gamma/data/"
        "posterior_sample1d.txt_2"
    )
    sample = np.loadtxt(io.BytesIO(archive.read(member)), comments="#")
    if sample.ndim != 2 or sample.shape[1] < 10:
        raise ValueError(f"expected at least 10 posterior columns in {member}")

    amplitude0 = np.exp(sample[:, 4])
    width0 = np.exp(sample[:, 6])
    centroid0 = sample[:, 5] + 2.0 * width0
    amplitude1 = np.exp(sample[:, 7])
    centroid1 = sample[:, 8] + 2.0 * np.exp(sample[:, 9])
    fraction0 = amplitude0 / (amplitude0 + amplitude1)
    selected = centroid0 < centroid1

    return {
        "centroid0": centroid0[selected],
        "fwhm0": FWHM_PER_W * width0[selected],
        "fraction0": fraction0[selected],
        "n_all": np.array([sample.shape[0]]),
        "n_selected": np.array([np.sum(selected)]),
    }


def fd_bins(values: np.ndarray, minimum: int = 30, maximum: int = 260) -> int:
    """Freedman-Diaconis bins, capped for stable small-panel rendering."""
    q25, q75 = np.quantile(values, [0.25, 0.75])
    width = 2.0 * (q75 - q25) / np.cbrt(values.size)
    span = np.ptp(values)
    if width <= 0.0 or span <= 0.0:
        return minimum
    return int(np.clip(np.ceil(span / width), minimum, maximum))


def interval_label(values: np.ndarray, digits: int) -> str:
    q16, q50, q84 = np.quantile(values, [0.16, 0.5, 0.84])
    return (
        f"{q50:.{digits}f} "
        f"(-{q50-q16:.{digits}f}/+{q84-q50:.{digits}f})"
    )


def write_summary(data: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    path = output.with_name(output.name + "_summary.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "band",
        "n_all",
        "n_tau0_lt_tau1",
        "selected_fraction",
        "quantity",
        "q16",
        "median",
        "q84",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for band in BANDS:
            item = data[band]
            for quantity in ("centroid0", "fwhm0", "fraction0"):
                q16, q50, q84 = np.quantile(
                    item[quantity], [0.16, 0.5, 0.84]
                )
                writer.writerow(
                    {
                        "band": band,
                        "n_all": int(item["n_all"][0]),
                        "n_tau0_lt_tau1": int(item["n_selected"][0]),
                        "selected_fraction": (
                            item["n_selected"][0] / item["n_all"][0]
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
    blue = "#3B6F91"
    blue_edge = "#244A64"
    orange = "#C46B3B"
    orange_edge = "#7D3D1F"

    quantities = (
        ("centroid0", r"Centroid $\tau_0=c_0+2w_0$ (days)", 3),
        ("fwhm0", r"Single-component FWHM $=2.446386w_0$ (days)", 3),
        ("fraction0", r"Amplitude fraction $f_0=A_0/(A_0+A_1)$", 3),
    )

    fig, axes = plt.subplots(3, 5, figsize=(16.0, 9.2))

    for column, band in enumerate(BANDS):
        item = data[band]
        for row, (key, xlabel, digits) in enumerate(quantities):
            ax = axes[row, column]
            values = item[key]
            color = orange if band == "U" else blue
            edge = orange_edge if band == "U" else blue_edge
            bins = fd_bins(values)

            ax.hist(
                values,
                bins=bins,
                density=True,
                color=color,
                edgecolor=edge,
                linewidth=0.35,
                alpha=0.78,
            )
            q16, q50, q84 = np.quantile(values, [0.16, 0.5, 0.84])
            ax.axvline(q50, color=ink, lw=1.25, zorder=3)
            ax.axvline(q16, color=ink, lw=0.9, ls=":", alpha=0.8, zorder=3)
            ax.axvline(q84, color=ink, lw=0.9, ls=":", alpha=0.8, zorder=3)
            ax.set_yscale("log")
            ax.set_xlabel(xlabel, fontsize=8.7)
            ax.grid(True, axis="y", color=grid, linewidth=0.65, alpha=0.85)
            ax.set_axisbelow(True)
            ax.tick_params(direction="in", which="both", top=True, right=True,
                           labelsize=8)
            ax.minorticks_on()

            if key == "fraction0":
                ax.set_xlim(0.0, 1.0)
            else:
                lo, hi = np.min(values), np.max(values)
                pad = 0.025 * (hi - lo) if hi > lo else 0.1
                ax.set_xlim(lo - pad, hi + pad)

            ax.text(
                0.04,
                0.95,
                interval_label(values, digits),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.0,
                color=ink,
                bbox={
                    "boxstyle": "round,pad=0.23",
                    "facecolor": "white",
                    "edgecolor": grid,
                    "alpha": 0.90,
                },
            )

        selected = int(item["n_selected"][0])
        total = int(item["n_all"][0])
        axes[0, column].set_title(
            f"{band}\nN={selected}/{total}",
            fontsize=11,
            color=ink,
            weight="bold",
        )

    for row in range(3):
        axes[row, 0].set_ylabel("Posterior density (log scale)", fontsize=9)

    fig.suptitle(
        "Mrk 817 two-Gamma posterior: component-0 distributions",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=15,
        color=ink,
    )
    fig.text(
        0.055,
        0.949,
        r"Selection: $\tau_0<\tau_1$ only; no additional U-band cut. "
        r"Solid line: median; dotted lines: 16th and 84th percentiles. "
        "Log-density axes retain visibility of low-mass secondary structure.",
        ha="left",
        fontsize=9.4,
        color=muted,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.07,
        top=0.895,
        wspace=0.27,
        hspace=0.45,
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
    write_summary(data, args.output)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_name(args.output.name + "_summary.csv"))


if __name__ == "__main__":
    main()
