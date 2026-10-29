#!/usr/bin/env python3
"""NGC 6814 Gamma centroid posteriors after tau_comp0 > 0 selection."""

# 复用 NGC6814 的重标记读取器，绘制 tau_comp0>0 条件后的质心分布。
# 理论参考值和输出目录硬编码；筛后的权重需重新归一，不能沿用全后验概率解释。
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_ngc6814_gamma_centroids import BANDS, load_band, weighted_quantiles


OUT = Path("/home/leonhard/sci001/NGC6814/raw/ngc6814_gamma_centroids_comp0_gt0")
THEORY = {"W2": 0.0, "M2": 0.010990, "W1": 0.024642,
          "U": 0.059693, "B": 0.100468, "V": 0.151628}


def main() -> None:
    colors = {"comp0": "#3B6F91", "comp1": "#C46B3B", "total": "#3A8754"}
    quantities = ("comp0", "comp1", "total")
    data = {}
    weights = {}
    retained = {}
    for band in BANDS:
        item, w = load_band(band)
        keep = (np.ones_like(item["comp0"], dtype=bool)
                if band == "W2" else item["comp0"] > 0.0)
        retained[band] = float(w[keep].sum())
        w = w[keep] / retained[band]
        data[band] = {key: value[keep] for key, value in item.items()}
        weights[band] = w

    fig, axes = plt.subplots(
        6, 3, figsize=(14.0, 17.0),
        gridspec_kw={"hspace": 0.42, "wspace": 0.20},
    )
    fig.suptitle("NGC 6814 Gamma2: posterior after $\\tau_{\\rm comp0}>0$",
                 fontsize=18, y=0.995)
    fig.text(
        0.5, 0.972,
        "Posterior weights renormalized after selection; components reordered by centroid per draw.",
        ha="center", va="top", fontsize=10.5, color="#59616B",
    )
    for col, quantity in enumerate(quantities):
        axes[0, col].set_title(
            {"comp0": "comp0 (short)", "comp1": "comp1 (long)",
             "total": "total"}[quantity], fontsize=12)

    for row, band in enumerate(BANDS):
        for col, quantity in enumerate(quantities):
            ax = axes[row, col]
            vals = data[band][quantity]
            w = weights[band]
            ax.grid(axis="y", color="#E2E6EA", linewidth=0.65, alpha=0.9)
            ax.set_axisbelow(True)
            ax.tick_params(direction="in", top=True, right=True, labelsize=8.5)
            ax.axvline(0.0, color="#59616B", lw=0.8, alpha=0.8)

            if band == "W2":
                ax.axvline(0.0, color=colors[quantity], lw=2.2)
                ax.text(0.04, 0.78, "driver anchor\n$\\tau=0$ fixed",
                        transform=ax.transAxes, fontsize=10, color="#25282D",
                        va="top")
                ax.set_xlim(-1.0, 1.0)
                ax.set_ylim(0.0, 1.0)
            else:
                if quantity == "comp0":
                    bins = np.concatenate((
                        np.linspace(-10.0, 2.0, 121),
                        np.linspace(2.0, 90.0, 45)[1:],
                    ))
                else:
                    bins = np.linspace(-12.0, 90.0, 81)
                ax.hist(vals, bins=bins, weights=w, density=True,
                        color=colors[quantity], alpha=0.68,
                        edgecolor=colors[quantity], linewidth=0.38)
                q16, q50, q84 = weighted_quantiles(vals, w)
                ax.axvline(q50, color="#25282D", lw=1.25)
                ax.axvline(q16, color="#25282D", lw=0.8, ls=":", alpha=0.8)
                ax.axvline(q84, color="#25282D", lw=0.8, ls=":", alpha=0.8)
                if quantity in ("comp0", "total"):
                    ax.axvline(THEORY[band], color="#6B4E9B", lw=1.15,
                               ls="--", alpha=0.95)
                ax.text(
                    0.97, 0.91,
                    f"{q50:.2f} [{q16:.2f}, {q84:.2f}] d",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.4,
                    color="#25282D",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                          "edgecolor": "#E2E6EA", "alpha": 0.9},
                )
                ax.set_xlim(-10.0, 2.0) if quantity == "comp0" else ax.set_xlim(-12.0, 90.0)

            if col == 0:
                ax.set_ylabel(f"{band}\nposterior density", fontsize=9.8)
                if band != "W2":
                    ax.text(0.04, 0.77, f"retained={retained[band]:.3f}",
                            transform=ax.transAxes, fontsize=8.5, color="#59616B")
            if row == len(BANDS) - 1:
                ax.set_xlabel(r"centroid $\tau$ [d, relative to W2]", fontsize=9.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    rows = ["band,retained_mass,theory_lag_days,component,q16_days,median_days,q84_days,median_minus_theory_days"]
    for band in BANDS:
        for quantity in quantities:
            q16, q50, q84 = weighted_quantiles(data[band][quantity], weights[band])
            rows.append(
                f"{band},{retained[band]:.8g},{THEORY[band]:.8g},{quantity},"
                f"{q16:.8g},{q50:.8g},{q84:.8g},{q50-THEORY[band]:.8g}"
            )
    OUT.with_name(OUT.name + "_summary.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    plt.close(fig)
    print(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".pdf"))
    print(OUT.with_name(OUT.name + "_summary.csv"))


if __name__ == "__main__":
    main()
