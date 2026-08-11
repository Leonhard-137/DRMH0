#!/usr/bin/env python3
"""Plot L-/relative-centroid ratio distributions for Mrk 817 two-Gamma fits."""

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
    "Mrk817/results/tf_comparison/Mrk817_gamma2_comp0_Lminus_over_taurel"
)
BANDS = ("UVM2", "UVW1", "U", "B", "V")
RATIO_MIN = 0.8
RATIO_MAX = 1.2
Z_LIMIT = 4.1


def signed_log10(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log10(1.0 + np.abs(values))


def load_band(archive: zipfile.ZipFile, band: str) -> dict[str, np.ndarray]:
    member = (
        f"2comp/run_UVW2_to_{band}_2comp_gamma/data/"
        "posterior_sample1d.txt_2"
    )
    sample = np.loadtxt(io.BytesIO(archive.read(member)), comments="#")
    width0 = np.exp(sample[:, 6])
    centroid0 = sample[:, 5] + 2.0 * width0
    centroid1 = sample[:, 8] + 2.0 * np.exp(sample[:, 9])
    ordered = centroid0 < centroid1
    centroid = centroid0[ordered]
    lminus = 2.0 * width0[ordered]
    ratio = lminus / centroid
    return {
        "centroid0": centroid,
        "lminus0": lminus,
        "ratio": ratio,
        "n_all": np.array([sample.shape[0]]),
        "n_ordered": np.array([np.sum(ordered)]),
    }


def write_outputs(data: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    sample_path = output.with_name(output.name + "_posterior.csv")
    summary_path = output.with_name(output.name + "_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with sample_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band", "tau0_rel_days", "Lminus0_days", "Lminus_over_tau0"])
        for band in BANDS:
            item = data[band]
            for row in zip(item["centroid0"], item["lminus0"], item["ratio"]):
                writer.writerow([band, *[f"{value:.10g}" for value in row]])

    fields = [
        "band",
        "n_ordered",
        "fraction_tau0_le_0",
        "fraction_ratio_0p8_to_1p2",
        "ratio_all_q16",
        "ratio_all_median",
        "ratio_all_q84",
        "ratio_positive_q16",
        "ratio_positive_median",
        "ratio_positive_q84",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for band in BANDS:
            item = data[band]
            ratio = item["ratio"]
            positive = ratio[item["centroid0"] > 0.0]
            q_all = np.quantile(ratio, [0.16, 0.5, 0.84])
            q_positive = np.quantile(positive, [0.16, 0.5, 0.84])
            writer.writerow(
                {
                    "band": band,
                    "n_ordered": int(item["n_ordered"][0]),
                    "fraction_tau0_le_0": np.mean(item["centroid0"] <= 0.0),
                    "fraction_ratio_0p8_to_1p2": np.mean(
                        (ratio >= RATIO_MIN) & (ratio <= RATIO_MAX)
                    ),
                    "ratio_all_q16": f"{q_all[0]:.10g}",
                    "ratio_all_median": f"{q_all[1]:.10g}",
                    "ratio_all_q84": f"{q_all[2]:.10g}",
                    "ratio_positive_q16": f"{q_positive[0]:.10g}",
                    "ratio_positive_median": f"{q_positive[1]:.10g}",
                    "ratio_positive_q84": f"{q_positive[2]:.10g}",
                }
            )


def plot(data: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    ink = "#25282D"
    muted = "#59616B"
    grid = "#E2E6EA"
    blue = "#3B6F91"
    blue_edge = "#244A64"
    orange = "#C46B3B"

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.2))
    axes_flat = axes.ravel()
    edges = np.linspace(-Z_LIMIT, Z_LIMIT, 111)
    z_cut_lo, z_cut_hi = signed_log10(np.array([RATIO_MIN, RATIO_MAX]))

    tick_values = np.array([-1000, -100, -10, -1, 0, 1, 10, 100, 1000], dtype=float)
    tick_positions = signed_log10(tick_values)
    tick_labels = [r"$-10^3$", r"$-10^2$", r"$-10$", r"$-1$", "0",
                   r"$1$", r"$10$", r"$10^2$", r"$10^3$"]

    for ax, band in zip(axes_flat[:5], BANDS):
        item = data[band]
        ratio = item["ratio"]
        transformed = signed_log10(ratio)
        clipped = np.clip(transformed, -Z_LIMIT, Z_LIMIT)
        ax.axvspan(z_cut_lo, z_cut_hi, color=orange, alpha=0.18, zorder=0)
        ax.hist(
            clipped,
            bins=edges,
            density=True,
            color=blue,
            edgecolor=blue_edge,
            linewidth=0.35,
            alpha=0.78,
        )
        ax.axvline(0.0, color=ink, ls=":", lw=0.9, alpha=0.7)
        ax.axvline(z_cut_lo, color=orange, ls="--", lw=1.0)
        ax.axvline(z_cut_hi, color=orange, ls="--", lw=1.0)
        in_cut = (ratio >= RATIO_MIN) & (ratio <= RATIO_MAX)
        if np.any(in_cut):
            ax.plot(
                transformed[in_cut],
                np.full(np.sum(in_cut), 0.025),
                "|",
                transform=ax.get_xaxis_transform(),
                color=orange,
                markersize=8,
                markeredgewidth=1.0,
                alpha=0.85,
                zorder=5,
            )
        ax.set_xlim(-Z_LIMIT, Z_LIMIT)
        ax.set_xticks(tick_positions, tick_labels)
        ax.set_title(band, loc="left", fontsize=12, color=ink, weight="bold")
        ax.set_xlabel(r"$R=L_-/\tau_{0,\rm rel}$ (signed-log display)")
        ax.set_ylabel("Posterior density per signed-log interval")
        ax.grid(True, axis="y", color=grid, linewidth=0.65, alpha=0.85)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", which="both", top=True, right=True,
                       labelsize=8)
        ax.minorticks_on()

        positive = ratio[item["centroid0"] > 0.0]
        q16, q50, q84 = np.quantile(positive, [0.16, 0.5, 0.84])
        nonpositive_fraction = np.mean(item["centroid0"] <= 0.0)
        n_cut = int(np.sum(in_cut))
        cut_fraction = n_cut / len(ratio)
        clipped_fraction = np.mean(np.abs(transformed) > Z_LIMIT)
        text = (
            rf"$N={len(ratio)}$" + "\n"
            + rf"$P(\tau_0\leq0)={100.0*nonpositive_fraction:.1f}\%$" + "\n"
            + rf"positive-$\tau_0$: $R={q50:.2f}$ "
              rf"$[{q16:.2f},{q84:.2f}]$" + "\n"
            + rf"$P(0.8\leq R\leq1.2)={100.0*cut_fraction:.2f}\%$ "
              rf"$({n_cut}/{len(ratio)})$"
        )
        if clipped_fraction > 0.0:
            text += "\n" + rf"edge-clipped: {clipped_fraction:.2%}"
        ax.text(
            0.97,
            0.95,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.1,
            color=ink,
            bbox={
                "boxstyle": "round,pad=0.27",
                "facecolor": "white",
                "edgecolor": grid,
                "alpha": 0.90,
            },
        )

    notes = axes_flat[5]
    notes.axis("off")
    notes.text(
        0.04,
        0.90,
        "How to read the axis",
        fontsize=12,
        color=ink,
        weight="bold",
        transform=notes.transAxes,
    )
    notes.text(
        0.04,
        0.80,
        r"$R=2w_0/(c_0+2w_0)$" + "\n\n"
        r"Displayed coordinate:" + "\n"
        r"$\mathrm{sign}(R)\log_{10}(1+|R|)$" + "\n"
        "but tick labels report R itself.\n\n"
        "Negative R means a negative relative centroid.\n"
        "Orange band marks 0.8–1.2; orange rug\n"
        "ticks mark the individual accepted samples.\n"
        "Extreme values occur when the centroid\n"
        "approaches zero; values beyond the displayed\n"
        r"$|R|\simeq10^4$ edge are clipped to that edge.",
        fontsize=10,
        color=muted,
        va="top",
        linespacing=1.35,
        transform=notes.transAxes,
    )

    fig.suptitle(
        r"Mrk 817 two-Gamma posterior: $L_-/\tau_{0,\rm rel}$ distributions",
        x=0.065,
        y=0.982,
        ha="left",
        fontsize=15,
        color=ink,
    )
    fig.text(
        0.065,
        0.942,
        r"Selection: $\tau_0<\tau_1$ only. Here $L_-=2w_0$; "
        r"no shape cut and no additional U-band cut are applied.",
        ha="left",
        fontsize=9.5,
        color=muted,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.075,
        top=0.885,
        wspace=0.29,
        hspace=0.34,
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
