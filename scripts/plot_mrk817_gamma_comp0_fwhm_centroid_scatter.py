#!/usr/bin/env python3
"""Plot component-0 FWHM versus centroid for all Mrk 817 two-Gamma bands."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


DEFAULT_ARCHIVE = Path("Mrk817/runs/mica/legacy_archives/gamma0_100_2comp.zip")
DEFAULT_OUTPUT = Path(
    "Mrk817/results/tf_comparison/"
    "Mrk817_gamma2_comp0_fwhm_centroid_scatter"
)
BANDS = ("UVM2", "UVW1", "U", "B", "V")
FWHM_PER_W = 2.446386


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
    selected = centroid0 < centroid1
    return {
        "fwhm0": FWHM_PER_W * width0[selected],
        "centroid0": centroid0[selected],
        "fraction0": fraction0[selected],
        "n_all": np.array([sample.shape[0]]),
        "n_selected": np.array([np.sum(selected)]),
    }


def padded_limits(values: np.ndarray, fraction: float = 0.04) -> tuple[float, float]:
    lo, hi = np.min(values), np.max(values)
    pad = fraction * (hi - lo) if hi > lo else 0.1
    return lo - pad, hi + pad


def write_samples(data: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    path = output.with_name(output.name + "_posterior.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band", "fwhm0_days", "centroid0_days", "f0"])
        for band in BANDS:
            item = data[band]
            for fwhm, centroid, fraction in zip(
                item["fwhm0"], item["centroid0"], item["fraction0"]
            ):
                writer.writerow(
                    [band, f"{fwhm:.10g}", f"{centroid:.10g}", f"{fraction:.10g}"]
                )


def plot(data: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    ink = "#25282D"
    muted = "#59616B"
    grid = "#E2E6EA"
    blue = "#356F95"
    reference = "#B45D36"

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.5))
    axes_flat = axes.ravel()

    for ax, band in zip(axes_flat[:5], BANDS):
        item = data[band]
        fwhm = item["fwhm0"]
        centroid = item["centroid0"]
        ax.scatter(
            fwhm,
            centroid,
            s=11,
            color=blue,
            alpha=0.34,
            linewidths=0,
            rasterized=True,
        )
        ax.axvline(1.0, color=reference, ls="--", lw=1.1, alpha=0.9)
        ax.axhline(0.0, color=ink, ls=":", lw=0.9, alpha=0.65)
        xlo, xhi = padded_limits(fwhm)
        ylo, yhi = padded_limits(centroid)
        ax.set_xlim(max(0.0, xlo), xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_title(band, loc="left", fontsize=12, color=ink, weight="bold")
        ax.set_xlabel(r"Comp0 FWHM $=2.446386w_0$ (days)")
        ax.set_ylabel(r"Comp0 centroid $\tau_0=c_0+2w_0$ (days)")
        ax.grid(True, color=grid, linewidth=0.65, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()

        qx = np.quantile(fwhm, [0.16, 0.5, 0.84])
        qy = np.quantile(centroid, [0.16, 0.5, 0.84])
        correlation = np.corrcoef(fwhm, centroid)[0, 1]
        n_selected = int(item["n_selected"][0])
        n_all = int(item["n_all"][0])
        ax.text(
            0.97,
            0.04,
            rf"$N={n_selected}/{n_all}$" + "\n"
            + rf"median FWHM$_0={qx[1]:.3f}$ d" + "\n"
            + rf"median $\tau_0={qy[1]:.3f}$ d" + "\n"
            + rf"$r={correlation:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.3,
            color=ink,
            bbox={
                "boxstyle": "round,pad=0.27",
                "facecolor": "white",
                "edgecolor": grid,
                "alpha": 0.90,
            },
        )

        if band == "UVM2":
            inset = inset_axes(ax, width="43%", height="43%", loc="lower left",
                               borderpad=1.2)
            inset.scatter(
                fwhm,
                centroid,
                s=7,
                color=blue,
                alpha=0.30,
                linewidths=0,
                rasterized=True,
            )
            inset.axvline(1.0, color=reference, ls="--", lw=0.8, alpha=0.9)
            inset.axhline(0.0, color=ink, ls=":", lw=0.7, alpha=0.65)
            ix0, ix1 = np.quantile(fwhm, [0.05, 0.95])
            iy0, iy1 = np.quantile(centroid, [0.05, 0.95])
            xpad = 0.08 * (ix1 - ix0)
            ypad = 0.08 * (iy1 - iy0)
            inset.set_xlim(max(0.0, ix0 - xpad), ix1 + xpad)
            inset.set_ylim(iy0 - ypad, iy1 + ypad)
            inset.tick_params(direction="in", labelsize=6.5, top=True, right=True)
            inset.set_title("central 90%", fontsize=7.2, color=muted)
            mark_inset(ax, inset, loc1=1, loc2=3, fc="none", ec=muted,
                       lw=0.65, alpha=0.55)

    summary = axes_flat[5]
    summary.axis("off")
    summary.text(
        0.04,
        0.90,
        "Definitions",
        fontsize=12,
        color=ink,
        weight="bold",
        transform=summary.transAxes,
    )
    summary.text(
        0.04,
        0.80,
        r"Selection: $\tau_0<\tau_1$ only" + "\n\n"
        r"Centroid: $\tau_0=c_0+2w_0$" + "\n"
        r"FWHM: $2.446386w_0$" + "\n\n"
        "Orange dashed line: FWHM = 1 d\n"
        "Grey dotted line: centroid = 0 d\n\n"
        "UVM2 inset enlarges its central 90%;\n"
        "the main panel retains every selected sample.",
        fontsize=10,
        color=muted,
        va="top",
        linespacing=1.35,
        transform=summary.transAxes,
    )

    fig.suptitle(
        "Mrk 817 two-Gamma posterior: component-0 FWHM versus centroid",
        x=0.065,
        y=0.985,
        ha="left",
        fontsize=15,
        color=ink,
    )
    fig.text(
        0.065,
        0.947,
        r"All posterior samples satisfying $\tau_0<\tau_1$; no additional "
        r"U-band cut. Each point is one posterior draw.",
        ha="left",
        fontsize=9.5,
        color=muted,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.075,
        top=0.895,
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as archive:
        data = {band: load_band(archive, band) for band in BANDS}
    plot(data, args.output)
    write_samples(data, args.output)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_name(args.output.name + "_posterior.csv"))


if __name__ == "__main__":
    main()
