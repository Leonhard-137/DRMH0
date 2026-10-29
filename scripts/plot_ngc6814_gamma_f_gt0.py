#!/usr/bin/env python3
"""Plot NGC 6814 comp0 area fraction before/after tau_comp0 > 0."""

# 固定 NGC6814：比较按质心定义的短分量面积占比在 tau_short>0 前后的变化。
# 同一抽样中振幅和质心必须同步重排，避免把一个分量的时间配给另一个分量的面积。
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_ngc6814_gamma_centroids import BANDS, ROOT, weighted_quantiles


OUT = Path("/home/leonhard/sci001/NGC6814/raw/ngc6814_gamma_f_comp0_gt0")
THEORY = {"W2": 0.0, "M2": 0.010990, "W1": 0.024642,
          "U": 0.059693, "B": 0.100468, "V": 0.151628}


# 面积 f_short=A_short/(A0+A1)，不是总核峰高占比；先按 tau 重排 A 后再算 f。
def load_band_with_f(band: str):
    if band == "W2":
        zero = np.zeros(10000, dtype=float)
        return {"comp0": zero, "comp1": zero, "total": zero,
                "f_comp0": np.ones(10000, dtype=float)}, np.full(10000, 0.0001)

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
    f_comp0 = amps[:, 0] / np.sum(amps, axis=1)
    return {"comp0": taus[:, 0], "comp1": taus[:, 1], "total": total,
            "f_comp0": f_comp0}, weights


def main() -> None:
    data = {}
    weights = {}
    retained = {}
    for band in BANDS:
        values, w = load_band_with_f(band)
        keep = np.ones_like(w, dtype=bool) if band == "W2" else values["comp0"] > 0.0
        retained[band] = float(w[keep].sum())
        data[band] = {key: value[keep] for key, value in values.items()}
        weights[band] = w[keep] / retained[band]

    # Six rows (bands) by two columns: unfiltered and filtered f_comp0.
    fig, axes = plt.subplots(6, 2, figsize=(9.6, 15.5),
                             gridspec_kw={"hspace": 0.52, "wspace": 0.20})
    fig.suptitle("NGC 6814 Gamma2: comp0 area fraction $f$", fontsize=17, y=0.995)
    fig.text(0.5, 0.972,
             "Left: full posterior; right: posterior restricted to comp0 centroid > 0.",
             ha="center", va="top", fontsize=10.5, color="#59616B")
    axes[0, 0].set_title("Before selection", fontsize=12)
    axes[0, 1].set_title("After $\\tau_{\\rm comp0}>0$", fontsize=12)

    # Use the same f range and bins for every panel so the before/after shift is direct.
    bins = np.linspace(0.0, 1.0, 41)
    color = "#3B6F91"
    rows = [
        "band,retained_mass,theory_lag_days,"
        "f_before_q16,f_before_median,f_before_q84,"
        "f_after_q16,f_after_median,f_after_q84"
    ]
    for row, band in enumerate(BANDS):
        # Reconstruct the full, unfiltered arrays only for the plot/statistics.
        full_values, full_weights = load_band_with_f(band)
        post_values = data[band]["f_comp0"]
        post_weights = weights[band]
        q_before = weighted_quantiles(full_values["f_comp0"], full_weights)
        q_after = weighted_quantiles(post_values, post_weights)
        rows.append(
            f"{band},{retained[band]:.8g},{THEORY[band]:.8g},"
            f"{q_before[0]:.8g},{q_before[1]:.8g},{q_before[2]:.8g},"
            f"{q_after[0]:.8g},{q_after[1]:.8g},{q_after[2]:.8g}"
        )
        for col, (values, w, q) in enumerate(
            ((full_values["f_comp0"], full_weights, q_before),
             (post_values, post_weights, q_after))
        ):
            ax = axes[row, col]
            ax.hist(values, bins=bins, weights=w, density=True,
                    color=color, alpha=0.70, edgecolor=color, linewidth=0.4)
            ax.axvline(q[1], color="#25282D", lw=1.25)
            ax.axvline(q[0], color="#25282D", lw=0.8, ls=":", alpha=0.85)
            ax.axvline(q[2], color="#25282D", lw=0.8, ls=":", alpha=0.85)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(bottom=0.0)
            ax.grid(axis="y", color="#E2E6EA", linewidth=0.65, alpha=0.9)
            ax.set_axisbelow(True)
            ax.tick_params(direction="in", top=True, right=True, labelsize=8.5)
            ax.text(0.97, 0.90, f"{q[1]:.3f} [{q[0]:.3f}, {q[2]:.3f}]",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.4,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                          "edgecolor": "#E2E6EA", "alpha": 0.9})
            if col == 0:
                ax.set_ylabel(f"{band}\\nposterior density", fontsize=9.8)
            if row == len(BANDS) - 1:
                ax.set_xlabel(r"$f_{\rm comp0}=A_{\rm comp0}/(A_0+A_1)$", fontsize=9.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    OUT.with_name(OUT.name + "_summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    plt.close(fig)
    print(OUT.with_suffix(".png"))
    print(OUT.with_suffix(".pdf"))
    print(OUT.with_name(OUT.name + "_summary.csv"))


if __name__ == "__main__":
    main()
