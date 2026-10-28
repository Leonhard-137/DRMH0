#!/usr/bin/env python3
"""Plot NGC 4593 Gamma component-area fraction before/after tau_short > 0."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/leonhard/sci001/NGC4593/runs/mica/gamma2/2comp")
OUT = Path("/home/leonhard/sci001/NGC4593/raw/ngc4593_4particle_f_before_after")
BANDS = ("W2", "M2", "W1", "U", "B", "V")
TARGETS = BANDS[1:]


def load_band(band: str):
    data_dir = ROOT / f"run_W2_to_{band}_2comp_gamma" / "data"
    sample = np.loadtxt(data_dir / "sample1d.txt_2", comments="#")
    weights = np.loadtxt(data_dir / "weights.txt")[:, 2]
    weights = np.clip(weights, 0.0, None)
    weights /= weights.sum()

    width0 = np.exp(sample[:, 6])
    width1 = np.exp(sample[:, 9])
    tau0 = sample[:, 5] + 2.0 * width0
    tau1 = sample[:, 8] + 2.0 * width1
    amp0 = np.exp(sample[:, 4])
    amp1 = np.exp(sample[:, 7])
    frac0_native = amp0 / (amp0 + amp1)

    # Keep the same convention used in the centroid plots: comp0 is the
    # shorter-centroid component on every posterior draw.
    short_is_native0 = tau0 <= tau1
    tau_short = np.where(short_is_native0, tau0, tau1)
    frac_short = np.where(short_is_native0, frac0_native, 1.0 - frac0_native)
    selected = tau_short > 0.0
    retained_mass = float(weights[selected].sum())
    selected_weights = weights[selected] / retained_mass
    return frac_short, weights, selected, selected_weights, retained_mass


def qlabel(values: np.ndarray, weights: np.ndarray) -> str:
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    cdf = np.cumsum(w) / w.sum()
    q = [float(np.interp(p, cdf, x)) for p in (0.16, 0.50, 0.84)]
    return f"{q[1]:.3f}  [{q[0]:.3f}, {q[2]:.3f}]"


def main() -> None:
    ink = "#25282D"
    grid = "#E2E6EA"
    blue = "#3B6F91"
    orange = "#C46B3B"
    bins = np.linspace(0.0, 1.0, 51)

    fig, axes = plt.subplots(
        6, 2, figsize=(9.6, 16.0), sharex=True,
        gridspec_kw={"hspace": 0.42, "wspace": 0.18},
    )
    fig.suptitle(
        "NGC 4593 Gamma2 (4-particle): component-area fraction f0",
        fontsize=17, y=0.995,
    )
    fig.text(
        0.5, 0.972,
        r"f0 = A_short/(A_short + A_long); weighted posterior; comp0 = shorter centroid per draw",
        ha="center", va="top", fontsize=10.5, color="#59616B",
    )
    axes[0, 0].set_title("All posterior samples", fontsize=12)
    axes[0, 1].set_title(r"After selection: $\tau_{\rm comp0}>0$", fontsize=12)

    for row, band in enumerate(BANDS):
        ax_all, ax_sel = axes[row]
        for ax in (ax_all, ax_sel):
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks(np.linspace(0.0, 1.0, 6))
            ax.grid(axis="y", color=grid, linewidth=0.65, alpha=0.9)
            ax.set_axisbelow(True)
            ax.tick_params(direction="in", top=True, right=True, labelsize=9)
            ax.axvline(0.5, color="#9AA1A8", lw=0.7, ls=":", zorder=0)
        ax_all.text(0.03, 0.90, band, transform=ax_all.transAxes,
                    fontsize=12, fontweight="bold", color=ink)

        if band == "W2":
            for ax in (ax_all, ax_sel):
                ax.axvline(1.0, color=blue, lw=2.4)
                ax.text(0.04, 0.72, "driver anchor\nf0 = 1 fixed",
                        transform=ax.transAxes, fontsize=10, color=ink,
                        va="top")
            continue

        values, weights, selected, selected_weights, retained_mass = load_band(band)
        ax_all.hist(values, bins=bins, weights=weights, density=True,
                    color=blue, alpha=0.68, edgecolor=blue, linewidth=0.45)
        ax_sel.hist(values[selected], bins=bins, weights=selected_weights,
                    density=True, color=orange, alpha=0.68,
                    edgecolor=orange, linewidth=0.45)

        for ax, vals, ws in (
            (ax_all, values, weights),
            (ax_sel, values[selected], selected_weights),
        ):
            order = np.argsort(vals)
            xs = vals[order]
            cdf = np.cumsum(ws[order]) / ws.sum()
            q16, q50, q84 = [float(np.interp(p, cdf, xs))
                             for p in (0.16, 0.50, 0.84)]
            ax.axvline(q50, color=ink, lw=1.25)
            ax.axvline(q16, color=ink, lw=0.8, ls=":", alpha=0.8)
            ax.axvline(q84, color=ink, lw=0.8, ls=":", alpha=0.8)
            ax.text(0.97, 0.90, qlabel(vals, ws), transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.3, color=ink,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                          "edgecolor": grid, "alpha": 0.9})

        ax_all.text(0.03, 0.75, f"N={len(values):,}",
                    transform=ax_all.transAxes, fontsize=8.5, color="#59616B")
        ax_sel.text(0.03, 0.75, f"mass={retained_mass:.3f}",
                    transform=ax_sel.transAxes, fontsize=8.5, color="#59616B")

    for ax in axes[-1]:
        ax.set_xlabel(r"area fraction $f_0$", fontsize=10.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("posterior density", fontsize=9.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
