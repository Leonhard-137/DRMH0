#!/usr/bin/env python3
"""Plot Mrk 817 Gamma component-0 posteriors after requiring c0 > 0."""

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
    "Mrk817_gamma2_comp0_cpositive_histograms"
)
BANDS = ("UVM2", "UVW1", "U", "B", "V")


def load_band(archive: zipfile.ZipFile, band: str) -> dict[str, np.ndarray]:
    member = (
        f"2comp/run_UVW2_to_{band}_2comp_gamma/data/"
        "posterior_sample1d.txt_2"
    )
    sample = np.loadtxt(io.BytesIO(archive.read(member)), comments="#")
    amplitude0 = np.exp(sample[:, 4])
    center0 = sample[:, 5]
    width0 = np.exp(sample[:, 6])
    centroid0 = center0 + 2.0 * width0
    amplitude1 = np.exp(sample[:, 7])
    centroid1 = sample[:, 8] + 2.0 * np.exp(sample[:, 9])
    fraction0 = amplitude0 / (amplitude0 + amplitude1)

    ordered = centroid0 < centroid1
    selected = ordered & (center0 > 0.0)
    result: dict[str, np.ndarray] = {
        "n_all": np.array([sample.shape[0]]),
        "n_ordered": np.array([np.sum(ordered)]),
        "n_selected": np.array([np.sum(selected)]),
        "center0_selected": center0[selected],
    }
    for name, values in (
        ("centroid0", centroid0),
        ("two_width0", 2.0 * width0),
        ("fraction0", fraction0),
    ):
        result[f"{name}_ordered"] = values[ordered]
        result[f"{name}_selected"] = values[selected]
    return result


def panel_edges(ordered: np.ndarray, selected: np.ndarray, key: str) -> np.ndarray:
    if key == "fraction0":
        return np.linspace(0.0, 1.0, 41)
    lo, hi = np.quantile(ordered, [0.005, 0.995])
    if selected.size:
        lo = min(lo, float(np.min(selected)))
        hi = max(hi, float(np.max(selected)))
    if hi <= lo:
        lo -= 0.5
        hi += 0.5
    pad = 0.03 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, 46)


def interval_text(values: np.ndarray) -> str:
    if values.size == 0:
        return "no retained samples"
    if values.size == 1:
        return f"single sample: {values[0]:.3f}"
    q16, q50, q84 = np.quantile(values, [0.16, 0.50, 0.84])
    return f"{q50:.3f} [{q16:.3f}, {q84:.3f}]"


def write_outputs(data: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    posterior_path = output.with_name(output.name + "_posterior.csv")
    summary_path = output.with_name(output.name + "_summary.csv")

    with posterior_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["band", "c0_days", "centroid0_days", "two_w0_days", "f0"]
        )
        for band in BANDS:
            item = data[band]
            for row in zip(
                item["center0_selected"],
                item["centroid0_selected"],
                item["two_width0_selected"],
                item["fraction0_selected"],
            ):
                writer.writerow([band, *[f"{value:.10g}" for value in row]])

    fields = [
        "band", "n_all", "n_tau0_lt_tau1", "n_c0_gt_0",
        "fraction_of_ordered", "quantity", "q16", "median", "q84",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for band in BANDS:
            item = data[band]
            n_ordered = int(item["n_ordered"][0])
            n_selected = int(item["n_selected"][0])
            for quantity in ("centroid0", "two_width0", "fraction0"):
                values = item[f"{quantity}_selected"]
                if values.size:
                    q16, q50, q84 = np.quantile(values, [0.16, 0.50, 0.84])
                else:
                    q16 = q50 = q84 = np.nan
                writer.writerow(
                    {
                        "band": band,
                        "n_all": int(item["n_all"][0]),
                        "n_tau0_lt_tau1": n_ordered,
                        "n_c0_gt_0": n_selected,
                        "fraction_of_ordered": n_selected / n_ordered,
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
    orange = "#C46B3B"
    orange_edge = "#7D3D1F"
    quantities = (
        ("centroid0", r"Centroid $\tau_{0,\rm rel}=c_0+2w_0$ (days)"),
        ("two_width0", r"Left extent $2w_0$ (days)"),
        ("fraction0", r"Amplitude fraction $f_0=A_0/(A_0+A_1)$"),
    )

    fig, axes = plt.subplots(5, 3, figsize=(13.5, 14.0))
    for row, band in enumerate(BANDS):
        item = data[band]
        n_ordered = int(item["n_ordered"][0])
        n_selected = int(item["n_selected"][0])
        for column, (key, xlabel) in enumerate(quantities):
            ax = axes[row, column]
            ordered = item[f"{key}_ordered"]
            selected = item[f"{key}_selected"]
            edges = panel_edges(ordered, selected, key)

            ax.hist(
                ordered,
                bins=edges,
                density=True,
                histtype="step",
                color=baseline,
                linewidth=1.15,
            )
            if selected.size >= 2:
                ax.hist(
                    selected,
                    bins=edges,
                    density=True,
                    histtype="stepfilled",
                    color=orange,
                    edgecolor=orange_edge,
                    linewidth=0.65,
                    alpha=0.66,
                )
                ax.axvline(np.median(selected), color=orange_edge, lw=1.2)
            elif selected.size == 1:
                ax.axvline(selected[0], color=orange, lw=2.2)
            else:
                ax.text(
                    0.5, 0.48, "No retained samples", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9.2,
                    color=orange_edge, weight="bold",
                )

            if selected.size:
                ax.plot(
                    selected,
                    np.full(selected.size, 0.025),
                    "|",
                    transform=ax.get_xaxis_transform(),
                    color=orange_edge,
                    markersize=8,
                    markeredgewidth=0.9,
                    alpha=0.85,
                    zorder=5,
                )
            ax.set_xlim(edges[0], edges[-1])
            ax.set_xlabel(xlabel, fontsize=8.8)
            ax.grid(True, axis="y", color=grid, linewidth=0.65, alpha=0.85)
            ax.set_axisbelow(True)
            ax.tick_params(
                direction="in", which="both", top=True, right=True, labelsize=8
            )
            ax.minorticks_on()
            ax.text(
                0.97, 0.94, interval_text(selected), transform=ax.transAxes,
                ha="right", va="top", fontsize=7.9, color=ink,
                bbox={
                    "boxstyle": "round,pad=0.22", "facecolor": "white",
                    "edgecolor": grid, "alpha": 0.90,
                },
            )

        axes[row, 0].set_ylabel(
            f"{band}\nconditional density\n"
            f"N={n_selected}/{n_ordered} ({n_selected/n_ordered:.2%})",
            fontsize=9.2,
        )

    handles = [
        Line2D([0], [0], color=baseline, lw=1.3,
               label=r"Ordered posterior: $\tau_0<\tau_1$"),
        Patch(facecolor=orange, edgecolor=orange_edge, alpha=0.66,
              label=r"Retained posterior: $\tau_0<\tau_1$ and $c_0>0$"),
        Line2D([0], [0], color=orange_edge, marker="|", linestyle="None",
               markersize=8, label="Individual retained samples (rug)"),
    ]
    fig.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.965),
        frameon=False, ncol=3, fontsize=9,
    )
    fig.suptitle(
        r"Mrk 817 two-Gamma posterior after requiring $c_0>0$",
        x=0.065, y=0.992, ha="left", fontsize=15, color=ink,
    )
    fig.text(
        0.065, 0.968,
        r"Selection: $\tau_0<\tau_1$ and $c_0>0$ only. "
        r"Orange intervals are median [16th, 84th percentiles]. "
        "Sparse rows should be read as individual posterior samples, not smooth distributions.",
        ha="left", fontsize=9.2, color=muted,
    )
    fig.subplots_adjust(
        left=0.095, right=0.985, bottom=0.055, top=0.925,
        wspace=0.24, hspace=0.54,
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
