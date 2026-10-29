#!/usr/bin/env python3
"""Summarize and plot NGC 6814 two-Gamma posterior centroids."""

# 固定 NGC6814：读取 sample1d+weights，按质心重排后给短、长与总核质心。
# 总核质心是面积加权均值；W2 的全零数组是参考锚，不能当作额外观测数据。
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/leonhard/sci001/NGC6814/runs/mica/gamma2/2comp")
OUT = Path("/home/leonhard/sci001/NGC6814/raw/ngc6814_gamma_centroid_distributions")
BANDS = ("W2", "M2", "W1", "U", "B", "V")


def weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    cdf = np.cumsum(w) / np.sum(w)
    return tuple(float(np.interp(p, cdf, x)) for p in (0.16, 0.50, 0.84))


# 短/长按每个抽样的质心排序，total=sum(A_i*tau_i)/sum(A_i)；不能直接平均两分量延迟。
def load_band(band: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if band == "W2":
        zero = np.zeros(10000, dtype=float)
        return {"comp0": zero, "comp1": zero, "total": zero}, np.full(10000, 0.0001)

    data_dir = ROOT / f"run_W2_to_{band}_2comp_gamma" / "data"
    sample = np.loadtxt(data_dir / "sample1d.txt_2", comments="#")
    weights = np.loadtxt(data_dir / "weights.txt")[:, 2]
    weights = np.clip(weights, 0.0, None)
    weights /= weights.sum()

    tau_native = np.column_stack(
        (sample[:, 5] + 2.0 * np.exp(sample[:, 6]),
         sample[:, 8] + 2.0 * np.exp(sample[:, 9]))
    )
    amplitudes = np.column_stack((np.exp(sample[:, 4]), np.exp(sample[:, 7])))
    order = np.argsort(tau_native, axis=1)
    taus = np.take_along_axis(tau_native, order, axis=1)
    amps = np.take_along_axis(amplitudes, order, axis=1)
    total = np.sum(amps * taus, axis=1) / np.sum(amps, axis=1)
    return {"comp0": taus[:, 0], "comp1": taus[:, 1], "total": total}, weights


def main() -> None:
    colors = {"comp0": "#3B6F91", "comp1": "#C46B3B", "total": "#3A8754"}
    labels = {"comp0": "comp0 (short)", "comp1": "comp1 (long)", "total": "total"}
    quantities = ("comp0", "comp1", "total")
    all_data = {}
    all_weights = {}
    for band in BANDS:
        all_data[band], all_weights[band] = load_band(band)

    fig, axes = plt.subplots(
        6, 3, figsize=(14.0, 17.0), sharex=False,
        gridspec_kw={"hspace": 0.42, "wspace": 0.20},
    )
    fig.suptitle("NGC 6814 Gamma2: centroid posterior distributions",
                 fontsize=18, y=0.995)
    fig.text(
        0.5, 0.972,
        "No centroid selection; posterior-weighted. Components are reordered per draw by centroid.",
        ha="center", va="top", fontsize=10.5, color="#59616B",
    )

    for col, quantity in enumerate(quantities):
        axes[0, col].set_title(labels[quantity], fontsize=12)

    for row, band in enumerate(BANDS):
        data = all_data[band]
        weights = all_weights[band]
        for col, quantity in enumerate(quantities):
            ax = axes[row, col]
            values = data[quantity]
            color = colors[quantity]
            ax.grid(axis="y", color="#E2E6EA", linewidth=0.65, alpha=0.9)
            ax.set_axisbelow(True)
            ax.tick_params(direction="in", top=True, right=True, labelsize=8.5)
            ax.axvline(0.0, color="#59616B", lw=0.8, alpha=0.8)

            if band == "W2":
                ax.axvline(0.0, color=color, lw=2.2)
                ax.text(0.04, 0.78, "driver anchor\n$\\tau=0$ fixed",
                        transform=ax.transAxes, fontsize=10, color="#25282D",
                        va="top")
                ax.set_xlim(-1.0, 1.0)
                ax.set_ylim(0.0, 1.0)
            else:
                if quantity == "comp0":
                    # The short component has rare long-tail draws.  A
                    # single automatic range would make the near-zero core
                    # one wide bin, so refine the displayed core explicitly.
                    bins = np.concatenate((
                        np.linspace(-10.0, 2.0, 121),
                        np.linspace(2.0, 90.0, 45)[1:],
                    ))
                else:
                    bins = np.linspace(-12.0, 90.0, 81)
                ax.hist(values, bins=bins, weights=weights, density=True,
                        color=color, alpha=0.68, edgecolor=color, linewidth=0.38)
                q16, q50, q84 = weighted_quantiles(values, weights)
                ax.axvline(q50, color="#25282D", lw=1.25)
                ax.axvline(q16, color="#25282D", lw=0.8, ls=":", alpha=0.8)
                ax.axvline(q84, color="#25282D", lw=0.8, ls=":", alpha=0.8)
                ax.text(
                    0.97, 0.91,
                    f"{q50:.2f} [{q16:.2f}, {q84:.2f}] d",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.4,
                    color="#25282D",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                          "edgecolor": "#E2E6EA", "alpha": 0.9},
                )
                # The short component is concentrated near zero; keep its
                # negative tail but do not let the long-component range hide
                # the structure.  Long and total retain the full window.
                if quantity == "comp0":
                    ax.set_xlim(-10.0, 2.0)
                else:
                    ax.set_xlim(-12.0, 90.0)

            if col == 0:
                ax.set_ylabel(f"{band}\nposterior density", fontsize=9.8)
            if row == len(BANDS) - 1:
                ax.set_xlabel(r"centroid $\tau$ [d, relative to W2]", fontsize=9.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    rows = ["band,component,q16_days,median_days,q84_days"]
    for band in BANDS:
        for quantity in quantities:
            q16, q50, q84 = weighted_quantiles(all_data[band][quantity], all_weights[band])
            rows.append(f"{band},{quantity},{q16:.8g},{q50:.8g},{q84:.8g}")
    OUT.with_name(OUT.name + "_summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    plt.close(fig)
    print(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".pdf"))
    print(OUT.with_name(OUT.name + "_summary.csv"))


if __name__ == "__main__":
    main()
