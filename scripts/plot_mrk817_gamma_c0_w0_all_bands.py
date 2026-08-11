#!/usr/bin/env python3
"""Compare the component-0 two-Gamma posterior across Mrk 817 bands."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


DEFAULT_ARCHIVE = Path("Mrk817/runs/mica/legacy_archives/gamma0_100_2comp.zip")
DEFAULT_OUTPUT = Path(
    "Mrk817/results/tf_comparison/Mrk817_gamma2_c0_w0_all_bands"
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
    return {
        "center0": center0[ordered],
        "width0": width0[ordered],
        "centroid0": centroid0[ordered],
        "fraction0": fraction0[ordered],
        "n_all": np.array([len(sample)]),
        "n_ordered": np.array([np.sum(ordered)]),
    }


def quantile_text(values: np.ndarray) -> str:
    lo, med, hi = np.quantile(values, [0.16, 0.5, 0.84])
    return rf"{med:.2f}_{{-{med-lo:.2f}}}^{{+{hi-med:.2f}}}\,\mathrm{{d}}"


def robust_limits(values: np.ndarray, lower: float = 0.005, upper: float = 0.995):
    lo, hi = np.quantile(values, [lower, upper])
    pad = 0.07 * (hi - lo)
    return lo - pad, hi + pad


def write_summary(data: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    path = output.with_name(output.name + "_summary.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "band",
                "n_all",
                "n_tau0_lt_tau1",
                "ordered_fraction",
                "tau0_q16_days",
                "tau0_median_days",
                "tau0_q84_days",
                "fraction_tau0_lt_0p1",
                "corr_c0_w0",
                "f0_median",
            ],
        )
        writer.writeheader()
        for band in BANDS:
            item = data[band]
            tau = item["centroid0"]
            q16, q50, q84 = np.quantile(tau, [0.16, 0.5, 0.84])
            writer.writerow(
                {
                    "band": band,
                    "n_all": int(item["n_all"][0]),
                    "n_tau0_lt_tau1": int(item["n_ordered"][0]),
                    "ordered_fraction": item["n_ordered"][0] / item["n_all"][0],
                    "tau0_q16_days": q16,
                    "tau0_median_days": q50,
                    "tau0_q84_days": q84,
                    "fraction_tau0_lt_0p1": np.mean(tau < 0.1),
                    "corr_c0_w0": np.corrcoef(
                        item["center0"], item["width0"]
                    )[0, 1],
                    "f0_median": np.median(item["fraction0"]),
                }
            )


def plot(data: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    ink = "#25282D"
    muted = "#555B65"
    grid = "#E2E6EA"
    accent = "#BE4B3B"
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.75, vmax=1.0, clip=True)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.2))
    axes = axes.ravel()

    for ax, band in zip(axes[:5], BANDS):
        item = data[band]
        c0 = item["center0"]
        w0 = item["width0"]
        tau0 = item["centroid0"]
        f0 = item["fraction0"]
        xlo, xhi = robust_limits(c0)
        ylo, yhi = robust_limits(w0)
        ylo = max(0.0, ylo)

        ax.scatter(
            c0,
            w0,
            c=f0,
            cmap=cmap,
            norm=norm,
            s=10,
            alpha=0.48,
            linewidths=0,
            rasterized=True,
        )

        q16, q50, q84 = np.quantile(tau0, [0.16, 0.5, 0.84])
        xgrid = np.linspace(xlo, xhi, 500)
        for tau, style, width, alpha in (
            (q16, ":", 0.9, 0.55),
            (q50, "-", 1.3, 0.85),
            (q84, ":", 0.9, 0.55),
        ):
            wline = (tau - xgrid) / 2.0
            good = (wline >= ylo) & (wline <= yhi)
            ax.plot(
                xgrid[good], wline[good], color=ink, ls=style,
                lw=width, alpha=alpha,
            )

        if band == "U":
            wcut = (0.1 - xgrid) / 2.0
            good = (wcut >= ylo) & (wcut <= yhi)
            ax.plot(
                xgrid[good], wcut[good], color=accent, ls="--", lw=1.6,
                label=r"previous cut: $\tau_0=0.1$ d",
            )
            ax.legend(loc="lower left", frameon=True, fontsize=7.8)

        correlation = np.corrcoef(c0, w0)[0, 1]
        ax.text(
            0.04,
            0.96,
            rf"$\tau_0={quantile_text(tau0)}$" + "\n"
            + rf"$r(c_0,w_0)={correlation:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            color=ink,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": grid,
                "alpha": 0.88,
            },
        )
        ax.set_title(band, loc="left", fontsize=12, color=ink, weight="bold")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel(r"Gamma center, $c_0$ (days)")
        ax.set_ylabel(r"Gamma width, $w_0$ (days)")
        ax.grid(True, color=grid, linewidth=0.65, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.minorticks_on()

    summary_ax = axes[5]
    positions = np.arange(len(BANDS))
    for position, band in zip(positions, BANDS):
        tau = data[band]["centroid0"]
        parts = summary_ax.violinplot(
            tau,
            positions=[position],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            quantiles=[[0.16, 0.5, 0.84]],
            points=200,
        )
        for body in parts["bodies"]:
            body.set_facecolor(cmap(norm(np.median(data[band]["fraction0"]))))
            body.set_edgecolor(ink)
            body.set_alpha(0.68)
        if "cquantiles" in parts:
            parts["cquantiles"].set_color(ink)
            parts["cquantiles"].set_linewidth(1.0)

    summary_ax.axhline(0.1, color=accent, ls="--", lw=1.2, alpha=0.9)
    summary_ax.axhline(0.0, color=ink, ls=":", lw=0.9, alpha=0.65)
    summary_ax.set_xticks(positions, BANDS)
    summary_ax.set_ylabel(r"Component-0 centroid, $\tau_0=c_0+2w_0$ (days)")
    summary_ax.set_title("Centroid comparison", loc="left", fontsize=12,
                         color=ink, weight="bold")
    summary_ax.set_ylim(-0.45, 1.75)
    summary_ax.grid(True, axis="y", color=grid, linewidth=0.65, alpha=0.8)
    summary_ax.set_axisbelow(True)
    summary_ax.tick_params(direction="in", which="both", top=True, right=True)
    summary_ax.minorticks_on()
    summary_ax.text(
        0.04,
        0.96,
        "violins: ordered posterior\nlines: 16th / 50th / 84th pct.",
        transform=summary_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        color=muted,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": grid,
            "alpha": 0.9,
        },
    )

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar_axis = fig.add_axes([0.91, 0.17, 0.014, 0.64])
    cbar = fig.colorbar(scalar, cax=cbar_axis)
    cbar.set_label(r"Amplitude fraction, $f_0=A_0/(A_0+A_1)$")
    cbar.outline.set_linewidth(0.7)

    fig.suptitle(
        "Mrk 817 two-Gamma posterior: component-0 center–width geometry",
        x=0.07,
        ha="left",
        fontsize=15,
        color=ink,
    )
    fig.text(
        0.07,
        0.93,
        r"Common selection only: $\tau_0<\tau_1$. Solid/dotted diagonals mark "
        r"the median/68% interval of $\tau_0$ in each band; color is $f_0$.",
        ha="left",
        fontsize=9.5,
        color=muted,
    )
    fig.subplots_adjust(
        left=0.07, right=0.88, bottom=0.08, top=0.89, wspace=0.29, hspace=0.31
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
