#!/usr/bin/env python3
"""Render the poster's U-excess comparison from the manuscript result table.

The input contains posterior 16th/50th/84th percentiles, in percent, for
100 * (S_U / S_hat_U_without_U - 1).  S = lambda * R_lambda; the prediction
comes from a free-index power-law fit excluding U.  No fits are rerun here.
The paired intervals describe each estimate, not uncertainty in their change.

Run with the existing astro environment's Python from any working directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


POSTER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = POSTER_DIR.parents[2]
INPUT = PROJECT_DIR / "DRMH0/tables/relaxed/sed_u_excess.csv"
OUTPUT = POSTER_DIR / "assets/u_excess_comparison"

SOURCE_LABELS = {
    "Mrk817_a": "Mrk 817",
    "Mrk142": "Mrk 142",
    "Mrk335": "Mrk 335",
    "F9_a": "Fairall 9",
    "Mrk279_b": "Mrk 279",
    "Mrk509": "Mrk 509",
    "NGC5548": "NGC 5548",
    "NGC4151_a": "NGC 4151",
    "NGC6814": "NGC 6814",
    "NGC2617_a": "NGC 2617",
}
BLUE = "#326793"
GRAY = "#7C858B"
INK = "#273943"
EXCEPTION = "#946023"


def load_estimates():
    """Check one before/after pair per source, retaining the paper's order."""
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pairs = {}
    order = []
    for row in rows:
        source, stage = row["source"], row["stage"]
        if source not in pairs:
            pairs[source] = {}
            order.append(source)
        if stage not in ("before", "after") or stage in pairs[source]:
            raise ValueError(f"Unexpected or duplicate stage: {source}, {stage}")
        if row["fit_model"] != "free" or row["excluded_band"] != "U":
            raise ValueError(f"Unexpected excess definition for {source}")
        values = np.array([
            float(row[f"u_excess_percent_p{q}"]) for q in (16, 50, 84)
        ])
        if not np.all(np.isfinite(values)) or not np.all(np.diff(values) >= 0):
            raise ValueError(f"Invalid posterior percentiles: {source}, {stage}")
        pairs[source][stage] = values
    if order != list(SOURCE_LABELS):
        raise ValueError("The source population/order differs from the poster sample")
    if any(set(pair) != {"before", "after"} for pair in pairs.values()):
        raise ValueError("Missing paired estimate")
    before = np.array([pairs[source]["before"] for source in order])
    after = np.array([pairs[source]["after"] for source in order])
    return order, before, after


def main():
    order, before, after = load_estimates()
    decreases = after[:, 1] < before[:, 1]
    print(f"Median U-excess decreases: {decreases.sum()}/{len(order)}")
    print("Median increases: " + ", ".join(
        SOURCE_LABELS[source] for source, lower in zip(order, decreases) if not lower
    ))
    print(f"Mrk 817: {before[0, 1]:.6f}% -> {after[0, 1]:.6f}%")
    if decreases.sum() != 8:
        raise ValueError("The expected eight-of-ten comparison has changed")

    # The astro environment's cached font inventory predates the system Lato
    # install; register the existing font explicitly without rebuilding caches.
    lato_font = Path("/usr/share/fonts/truetype/lato/Lato-Regular.ttf")
    if lato_font.is_file():
        font_manager.fontManager.addfont(str(lato_font))
    font_manager.findfont("Lato", fallback_to_default=False)
    plt.rcParams.update({
        "font.family": "Lato", "font.size": 18,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.labelsize": 20, "xtick.labelsize": 18,
        "ytick.labelsize": 18, "legend.fontsize": 18,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.9, "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(11.9, 7.7), facecolor="white")
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.125, top=0.875)
    positions = np.arange(len(order), dtype=float)
    offset = 0.115
    for y, first, second in zip(positions, before[:, 1], after[:, 1]):
        ax.plot([first, second], [y - offset, y + offset],
                color="#C1C8CC", linewidth=1.6, zorder=1)

    handles = []
    for values, y, marker, color, fill, label in (
        (before, positions - offset, "s", GRAY, "white", "Before decomposition"),
        (after, positions + offset, "o", BLUE, BLUE, "Short-lag component"),
    ):
        handles.append(ax.errorbar(
            values[:, 1], y,
            xerr=np.vstack((values[:, 1] - values[:, 0],
                            values[:, 2] - values[:, 1])),
            fmt=marker, markersize=8.7, markerfacecolor=fill,
            markeredgecolor=color, markeredgewidth=1.8,
            ecolor=color, elinewidth=1.7, capsize=4, capthick=1.4,
            color=color, linestyle="none", label=label, zorder=3,
        ))

    ax.set_yticks(positions, [SOURCE_LABELS[source] for source in order])
    for tick, lower in zip(ax.get_yticklabels(), decreases):
        if not lower:
            tick.set_color(EXCEPTION)
    ax.set_ylim(len(order) - 0.48, -0.52)
    ax.set_xlim(-1.5, 56)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_xlabel("U-band excess (%)", labelpad=12)
    ax.axvline(0, color="#8E989E", linewidth=1.0, zorder=0)
    ax.grid(axis="x", color="#E3E7EA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#89959D")
    ax.tick_params(axis="y", length=0, pad=13)
    ax.tick_params(axis="x", length=5, pad=7)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.985),
               ncol=2, frameon=False, handletextpad=0.6, columnspacing=1.6)

    # Preserve the requested physical canvas instead of tight-cropping its bounds.
    for suffix in (".pdf", ".png"):
        destination = OUTPUT.with_suffix(suffix)
        fig.savefig(destination, dpi=220)
        print(f"Wrote {destination}")
    plt.close(fig)
    print("Canvas: 11.9 x 7.7 inches; PNG: 2618 x 1694 pixels (220 dpi).")


def mrk817_diagnostics():
    """Replot saved SED points and beta=4/3 lag inference; no new fit."""
    source = PROJECT_DIR / "DRMH0/fig/paper/sed_lag/Fig3_Mrk817_a_sed_lag.json"
    with source.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    lag_path = Path(meta["short_lags"])
    with lag_path.open(newline="", encoding="utf-8") as handle:
        lags = list(csv.DictReader(handle))
    with lag_path.with_name("Mrk817_a_lambda_tau_fit_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        fit = next(row for row in csv.DictReader(handle) if row["fit"] == "b4_3")
    tau0 = float(fit["tau0"])
    low = tau0 - float(fit["tau0_err_low"])
    high = tau0 + float(fit["tau0_err_high"])
    # Cross-check the plotted saved interval against its posterior, without refitting.
    posterior = np.genfromtxt(lag_path.with_name("tau0_posterior.txt"), names=True)["tau0_b4_3"]
    saved_interval = np.percentile(posterior, [15.865, 50, 84.135])
    print("Saved tau0 quantiles:", saved_interval)
    assert np.allclose(saved_interval, [low, tau0, high], rtol=1e-6)
    print("Fit-summary tau0 interval:", low, tau0, high)
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15,
                         "xtick.labelsize": 14, "ytick.labelsize": 14})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    fig.subplots_adjust(left=.085, right=.99, bottom=.21, top=.82, wspace=.36)
    ax, lag_ax = axes
    handles = []
    for key, color, marker, fill, label in [
        ("total", GRAY, "s", "white", "Before decomposition"),
        ("short", BLUE, "o", BLUE, "Short-lag component"),
    ]:
        points = meta["sed_display_scaling"][key]["scaled_points"]
        x = np.array([p["wavelength_angstrom"] for p in points])
        q = np.array([[p[f"scaled_p{k}"] for k in (16, 50, 84)] for p in points])
        handles.append(ax.errorbar(
            x, q[:, 1], yerr=[q[:, 1]-q[:, 0], q[:, 2]-q[:, 1]],
            fmt=marker, color=color, mfc=fill, mec=color, ms=5.8,
            capsize=3, lw=1.25, label=label, zorder=3))
    xx = np.linspace(1850, 5650, 250)
    ax.plot(xx, (xx/1928.)**(-4/3), "--", color=INK, lw=1.5)
    ax.set(xscale="log", yscale="log", xlim=(1800, 5650), ylim=(.13, 1.65),
           ylabel="Scaled variable SED")
    ax.set_xticks([2000, 3000, 5000], ["2000", "3000", "5000"])
    ax.set_yticks([.2, .5, 1], ["0.2", "0.5", "1.0"])
    ax.minorticks_off()
    ax.text(.95, .90, r"$\lambda^{-4/3}$ reference", transform=ax.transAxes,
            ha="right", fontsize=13)
    wave = np.array([float(r["lambda_obs"]) for r in lags])
    tau = np.array([float(r["tau"]) for r in lags])
    err = np.array([[float(r[k]) for r in lags]
                    for k in ("tau_err_low", "tau_err_high")])
    lag_ax.errorbar(wave[1:], tau[1:], yerr=err[:, 1:], fmt="o", color=BLUE,
                    ms=5.8, capsize=3, lw=1.25, zorder=3)
    ref_handle, = lag_ax.plot(wave[0], 0, "x", color=INK, ms=5.8,
                              zorder=4, label="UVW2 reference")
    handles.append(ref_handle)
    shape = (xx/float(fit["lambda0"]))**(4/3)-1
    lag_ax.fill_between(xx, low*shape, high*shape, color="#E5E7E9")
    lag_ax.plot(xx, tau0*shape, "--", color=INK, lw=1.5)
    lag_ax.axhline(0, color="#A9B1B7", lw=.8, ls=":")
    lag_ax.set(xlim=(1800, 5650), ylim=(-.055, .74), ylabel="Lag vs. UVW2 (day)")
    lag_ax.set_xticks([2000, 3000, 4000, 5000])
    lag_ax.set_yticks([0, .2, .4, .6])
    lag_ax.text(.06, .90, r"$\beta=4/3$ fit", transform=lag_ax.transAxes, fontsize=13)
    for panel in axes:
        panel.set_xlabel("Observed wavelength (Å)", labelpad=6)
        panel.axvline(3465, color="#A9B1B7", lw=.8, ls=":")
        panel.text(3465, .69 if panel is lag_ax else 1.43, "U",
                   ha="center", va="top", color=GRAY, fontsize=12)
        panel.tick_params(direction="out", width=.8, length=3.5)
        for spine in panel.spines.values():
            spine.set_linewidth(.9)
            spine.set_color("#6B7780")
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.53, 1.01), fontsize=14)
    for suffix in (".pdf", ".png"):
        target = POSTER_DIR / ("assets/mrk817_diagnostics" + suffix)
        fig.savefig(target, dpi=220)
        print(f"Wrote {target}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    mrk817_diagnostics()
