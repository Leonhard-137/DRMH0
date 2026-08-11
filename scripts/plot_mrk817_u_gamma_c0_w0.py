#!/usr/bin/env python3
"""Inspect the U-band two-Gamma posterior in the native center--width plane."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_POSTERIOR = Path(
    "Mrk817/past/runs/past/gamma0_100/2comp/"
    "run_UVW2_to_U_2comp_gamma/data/posterior_sample1d.txt_2"
)
DEFAULT_OUTPUT = Path(
    "Mrk817/results/tf_comparison/Mrk817_U_gamma2_c0_w0_selection"
)


def load_posterior(path: Path) -> dict[str, np.ndarray]:
    sample = np.loadtxt(path, comments="#")
    if sample.ndim != 2 or sample.shape[1] < 10:
        raise ValueError(f"expected at least 10 posterior columns in {path}")

    amplitude0 = np.exp(sample[:, 4])
    center0 = sample[:, 5]
    width0 = np.exp(sample[:, 6])
    centroid0 = center0 + 2.0 * width0
    amplitude1 = np.exp(sample[:, 7])
    centroid1 = sample[:, 8] + 2.0 * np.exp(sample[:, 9])
    fraction0 = amplitude0 / (amplitude0 + amplitude1)
    selected = (centroid0 < centroid1) & (centroid0 > 0.1)
    return {
        "center0": center0,
        "width0": width0,
        "centroid0": centroid0,
        "centroid1": centroid1,
        "fraction0": fraction0,
        "selected": selected,
    }


def plot(data: dict[str, np.ndarray], output: Path) -> None:
    center0 = data["center0"]
    width0 = data["width0"]
    fraction0 = data["fraction0"]
    selected = data["selected"]

    ink = "#22252A"
    grid = "#E4E7EB"
    grey = "#B9BDC4"
    cmap = "Blues"
    vmin, vmax = np.quantile(fraction0, [0.01, 0.99])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharex=True, sharey=True)
    scatter = axes[0].scatter(
        center0,
        width0,
        c=fraction0,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=17,
        alpha=0.58,
        linewidths=0,
        rasterized=True,
    )
    axes[0].scatter(
        center0[~selected],
        width0[~selected],
        s=12,
        facecolors="none",
        edgecolors=grey,
        linewidths=0.4,
        alpha=0.42,
        rasterized=True,
    )
    axes[0].set_title("Full posterior", loc="left", fontsize=12, color=ink)

    axes[1].scatter(
        center0[selected],
        width0[selected],
        c=fraction0[selected],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=19,
        alpha=0.62,
        linewidths=0,
        rasterized=True,
    )
    axes[1].set_title("Conditional posterior", loc="left", fontsize=12, color=ink)

    xlo, xhi = np.quantile(center0, [0.005, 0.995])
    ylo, yhi = np.quantile(width0, [0.005, 0.995])
    xpad = 0.06 * (xhi - xlo)
    ypad = 0.07 * (yhi - ylo)
    xgrid = np.linspace(xlo - xpad, xhi + xpad, 400)

    for ax in axes:
        # For Gamma, tau_0 = c_0 + 2 w_0. These are constant-centroid lines.
        for tau, style, alpha in [(-0.2, ":", 0.45), (0.0, ":", 0.55),
                                  (0.1, "--", 0.95), (0.3, ":", 0.45)]:
            wline = (tau - xgrid) / 2.0
            good = (wline >= ylo - ypad) & (wline <= yhi + ypad)
            ax.plot(xgrid[good], wline[good], color=ink, lw=1.0,
                    ls=style, alpha=alpha)
        ax.text(
            0.03,
            0.96,
            r"dashed: $\tau_0=0.1$ d",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#4B5058",
        )
        ax.grid(True, color=grid, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()
        ax.set_xlabel(r"Gamma center, $c_0$ (days)")
        ax.set_xlim(xlo - xpad, xhi + xpad)
        ax.set_ylim(max(0.0, ylo - ypad), yhi + ypad)
    axes[0].set_ylabel(r"Gamma width, $w_0$ (days)")

    corr_all = float(np.corrcoef(center0, width0)[0, 1])
    corr_selected = float(np.corrcoef(center0[selected], width0[selected])[0, 1])
    n_selected = int(np.sum(selected))
    axes[1].text(
        0.97,
        0.05,
        rf"$N={n_selected}$ ({n_selected/len(center0):.1%})" + "\n"
        + rf"$r(c_0,w_0)={corr_selected:.2f}$",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=ink,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": grid},
    )

    cbar = fig.colorbar(scatter, ax=axes, fraction=0.028, pad=0.025)
    cbar.set_label(r"Amplitude fraction, $f_0=A_0/(A_0+A_1)$")
    cbar.outline.set_linewidth(0.7)
    fig.suptitle("Mrk 817 U-band two-Gamma posterior: center versus width",
                 x=0.08, ha="left", fontsize=14, color=ink)
    fig.text(
        0.08,
        0.91,
        rf"Filter: $\tau_0<\tau_1$ and $\tau_0>0.1$ d; "
        rf"full-posterior $r(c_0,w_0)={corr_all:.2f}$. "
        r"Color shows $f_0$.",
        ha="left",
        fontsize=9.5,
        color="#4B5058",
    )
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.14, top=0.82, wspace=0.16)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(load_posterior(args.posterior), args.output)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
