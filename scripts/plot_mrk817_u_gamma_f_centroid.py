#!/usr/bin/env python3
"""Plot the U-band two-Gamma posterior in centroid--amplitude-fraction space."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_POSTERIOR = Path(
    "Mrk817/past/runs/past/gamma0_100/2comp/"
    "run_UVW2_to_U_2comp_gamma/data/posterior_sample1d.txt_2"
)
DEFAULT_OUTPUT = Path(
    "Mrk817/results/tf_comparison/Mrk817_U_gamma2_f_centroid_selection"
)


def posterior_columns(path: Path) -> dict[str, np.ndarray]:
    sample = np.loadtxt(path, comments="#")
    if sample.ndim != 2 or sample.shape[1] < 10:
        raise ValueError(f"expected at least 10 posterior columns in {path}")

    amplitude0 = np.exp(sample[:, 4])
    centroid0 = sample[:, 5] + 2.0 * np.exp(sample[:, 6])
    amplitude1 = np.exp(sample[:, 7])
    centroid1 = sample[:, 8] + 2.0 * np.exp(sample[:, 9])
    amplitude_sum = amplitude0 + amplitude1
    fraction0 = amplitude0 / amplitude_sum
    fraction1 = amplitude1 / amplitude_sum
    selected = (centroid0 < centroid1) & (centroid0 > 0.1)

    return {
        "sample": np.arange(len(sample), dtype=int),
        "centroid0": centroid0,
        "centroid1": centroid1,
        "fraction0": fraction0,
        "fraction1": fraction1,
        "selected": selected,
    }


def write_table(columns: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "tau0_days", "tau1_days", "f0", "f1", "selected"])
        for row in zip(
            columns["sample"],
            columns["centroid0"],
            columns["centroid1"],
            columns["fraction0"],
            columns["fraction1"],
            columns["selected"],
        ):
            writer.writerow([row[0], *[f"{value:.10g}" for value in row[1:5]], int(row[5])])


def quantile_text(x: np.ndarray) -> str:
    p16, p50, p84 = np.quantile(x, [0.16, 0.5, 0.84])
    return rf"${p50:.3f}_{{-{p50-p16:.3f}}}^{{+{p84-p50:.3f}}}$"


def plot(columns: dict[str, np.ndarray], output: Path) -> None:
    tau0 = columns["centroid0"]
    f0 = columns["fraction0"]
    selected = columns["selected"]
    n_all = len(tau0)
    n_selected = int(np.sum(selected))

    blue = "#2F6B9A"
    blue_dark = "#19486A"
    grey = "#B9BDC4"
    ink = "#22252A"
    grid = "#E4E7EB"

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True)

    ax = axes[0]
    ax.scatter(tau0[~selected], f0[~selected], s=13, color=grey, alpha=0.38,
               linewidths=0, rasterized=True, label="Removed")
    ax.scatter(tau0[selected], f0[selected], s=15, facecolors="none",
               edgecolors=blue, alpha=0.65, linewidths=0.55,
               rasterized=True, label="Retained")
    ax.set_title("Full posterior", loc="left", fontsize=12, color=ink)
    ax.legend(frameon=False, loc="lower right", fontsize=9, handletextpad=0.4)

    ax = axes[1]
    ax.scatter(tau0[selected], f0[selected], s=16, color=blue, alpha=0.48,
               linewidths=0, rasterized=True)
    ax.axvline(np.median(tau0[selected]), color=blue_dark, lw=1.1, ls="--")
    ax.axhline(np.median(f0[selected]), color=blue_dark, lw=1.1, ls=":")
    corr = float(np.corrcoef(tau0[selected], f0[selected])[0, 1])
    ax.set_title("Conditional posterior", loc="left", fontsize=12, color=ink)
    ax.text(
        0.97,
        0.05,
        rf"$N={n_selected}$ ({n_selected/n_all:.1%})" + "\n" + rf"$r={corr:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=ink,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": grid},
    )

    for ax in axes:
        ax.axvline(0.1, color=ink, lw=1.0, ls=(0, (4, 3)), alpha=0.85)
        ax.grid(True, color=grid, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()
        ax.set_xlabel(r"Component 0 centroid, $\tau_0=c_0+2w_0$ (days)")
    axes[0].set_ylabel(r"Component 0 amplitude fraction, $f_0=A_0/(A_0+A_1)$")

    # Use robust limits so a few remote posterior points do not hide the main relation.
    xlo, xhi = np.quantile(tau0, [0.005, 0.995])
    ylo, yhi = np.quantile(f0, [0.005, 0.995])
    xpad = 0.06 * (xhi - xlo)
    ypad = 0.08 * (yhi - ylo)
    axes[0].set_xlim(xlo - xpad, xhi + xpad)
    axes[0].set_ylim(max(0.0, ylo - ypad), min(1.0, yhi + ypad))

    full_tau = quantile_text(tau0)
    selected_tau = quantile_text(tau0[selected])
    full_f = quantile_text(f0)
    selected_f = quantile_text(f0[selected])
    fig.suptitle("Mrk 817 U-band two-Gamma posterior: centroid versus amplitude fraction",
                 x=0.08, ha="left", fontsize=14, color=ink)
    fig.text(
        0.08,
        0.91,
        rf"Filter: $\tau_0<\tau_1$ and $\tau_0>0.1$ d.  "
        rf"$\tau_0$: {full_tau} $\rightarrow$ {selected_tau} d;  "
        rf"$f_0$: {full_f} $\rightarrow$ {selected_f}.",
        ha="left",
        fontsize=9.5,
        color="#4B5058",
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.99, 0.86), w_pad=2.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=Path, default=DEFAULT_POSTERIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    columns = posterior_columns(args.posterior)
    plot(columns, args.output)
    write_table(columns, args.output.with_name(args.output.name + "_posterior.csv"))
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_name(args.output.name + "_posterior.csv"))


if __name__ == "__main__":
    main()
