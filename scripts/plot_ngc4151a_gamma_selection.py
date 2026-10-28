#!/usr/bin/env python3
"""Summarize NGC 4151a two-component posteriors and make requested plots.

The Gamma centroids follow the MICA convention used by collect_mica_results.py:
tau = center + 2 exp(log_sigma).  Components are reordered by centroid in every
posterior draw, so comp0 is the short component and comp1 is the delayed one.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ngc4151a-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


WORKSPACE = Path("/home/leonhard/sci001")
SOURCE = WORKSPACE / "NGC4151_a"
GAMMA_ROOT = SOURCE / "runs/mica/gamma2/2comp"
GAUSSIAN_ROOT = SOURCE / "runs/mica/gaussian2/2comp"
THEORY_FILE = SOURCE / "raw/theoretical_disk_lags.csv"
OUTDIR = SOURCE / "raw"

BANDS = ("W2", "M2", "W1", "U", "B", "V")
RESPONSES = BANDS[1:]
BLR_RL_CENTER = 6.2
BLR_RL_LOW = 3.1
BLR_RL_HIGH = 12.4
W2_TO_OPTICAL_CONTINUUM_LAG = 0.8
BLR_W2_CENTER = BLR_RL_CENTER + W2_TO_OPTICAL_CONTINUUM_LAG
BLR_W2_LOW = BLR_RL_LOW + W2_TO_OPTICAL_CONTINUUM_LAG
BLR_W2_HIGH = BLR_RL_HIGH + W2_TO_OPTICAL_CONTINUUM_LAG
HBETA_RESPONSE_LAG = 6.21
BLR_THEORY_SOURCE = "https://academic.oup.com/mnras/article/520/2/1807/6982927"
HBETA_SOURCE = "https://arxiv.org/abs/2208.02966"
CONTINUUM_LAG_SOURCE = "https://academic.oup.com/view-large/531148283"

COMPARISON_CSV = OUTDIR / "ngc4151a_comp0_centroid_comparison.csv"
SELECTION_CSV = OUTDIR / "ngc4151a_gamma_selection_summary.csv"
BLR_CSV = OUTDIR / "ngc4151a_gamma_comp1_blr_comparison.csv"
HIST_BASE = OUTDIR / "ngc4151a_gamma_selection_histograms"
BLR_BASE = OUTDIR / "ngc4151a_gamma_comp1_blr_comparison"


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: tuple[float, ...] = (0.16, 0.50, 0.84),
) -> tuple[float, ...]:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    x = np.asarray(values[mask], dtype=float)
    w = np.asarray(weights[mask], dtype=float)
    if x.size == 0 or w.sum() <= 0.0:
        raise ValueError("No finite, positive-weight posterior samples")
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return tuple(float(np.interp(p, cdf, x)) for p in probabilities)


def probability(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(weights[np.asarray(mask, dtype=bool)].sum() / weights.sum())


def theory_lags() -> dict[str, float]:
    output: dict[str, float] = {}
    with THEORY_FILE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output[row["band"]] = float(row["theory_relative_lag_days"])
    return output


def load_weights(data_dir: Path) -> np.ndarray:
    raw = np.loadtxt(data_dir / "weights.txt")
    weights = np.asarray(raw[:, 2], dtype=float)
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 0.0:
        raise ValueError(f"Posterior weights have zero mass: {data_dir}")
    return weights / weights.sum()


def load_gamma(band: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    data_dir = GAMMA_ROOT / f"run_W2_to_{band}_2comp_gamma/data"
    sample = np.loadtxt(data_dir / "sample1d.txt_2", comments="#")
    weights = load_weights(data_dir)

    tau_native = np.column_stack(
        (
            sample[:, 5] + 2.0 * np.exp(sample[:, 6]),
            sample[:, 8] + 2.0 * np.exp(sample[:, 9]),
        )
    )
    amp_native = np.column_stack((np.exp(sample[:, 4]), np.exp(sample[:, 7])))
    order = np.argsort(tau_native, axis=1)
    tau = np.take_along_axis(tau_native, order, axis=1)
    amp = np.take_along_axis(amp_native, order, axis=1)
    f0 = amp[:, 0] / amp.sum(axis=1)
    return {"comp0": tau[:, 0], "comp1": tau[:, 1], "f": f0}, weights


def load_gaussian(band: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    data_dir = GAUSSIAN_ROOT / f"run_W2_to_{band}_2comp_gaussian/data"
    sample = np.loadtxt(data_dir / "sample1d.txt_2", comments="#")
    weights = load_weights(data_dir)
    tau_native = np.column_stack((sample[:, 5], sample[:, 8]))
    amp_native = np.column_stack((np.exp(sample[:, 4]), np.exp(sample[:, 7])))
    order = np.argsort(tau_native, axis=1)
    tau = np.take_along_axis(tau_native, order, axis=1)
    amp = np.take_along_axis(amp_native, order, axis=1)
    f0 = amp[:, 0] / amp.sum(axis=1)
    return {"comp0": tau[:, 0], "comp1": tau[:, 1], "f": f0}, weights


def selected_gamma(
    values: dict[str, np.ndarray], weights: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    keep = (values["comp0"] > 0.0) & (values["comp0"] < values["comp1"])
    retained = float(weights[keep].sum())
    if retained <= 0.0:
        raise ValueError("Selection retained no posterior mass")
    selected = {key: item[keep] for key, item in values.items()}
    return selected, weights[keep] / retained, retained


def qrow(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    q16, q50, q84 = weighted_quantiles(values, weights)
    return {"q16": q16, "median": q50, "q84": q84}


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_tables(
    theory: dict[str, float],
    gamma: dict[str, dict[str, np.ndarray]],
    gamma_w: dict[str, np.ndarray],
    selected: dict[str, dict[str, np.ndarray]],
    selected_w: dict[str, np.ndarray],
    retained: dict[str, float],
    gaussian: dict[str, dict[str, np.ndarray]],
    gaussian_w: dict[str, np.ndarray],
) -> None:
    comparison_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    blr_rows: list[dict[str, object]] = []

    comparison_rows.append(
        {
            "band": "W2",
            "theory_days": 0.0,
            "gamma_full_q16_days": 0.0,
            "gamma_full_median_days": 0.0,
            "gamma_full_q84_days": 0.0,
            "gamma_selected_q16_days": 0.0,
            "gamma_selected_median_days": 0.0,
            "gamma_selected_q84_days": 0.0,
            "gamma_retained_probability": 1.0,
            "gaussian_q16_days": 0.0,
            "gaussian_median_days": 0.0,
            "gaussian_q84_days": 0.0,
            "gamma_selected_minus_theory_days": 0.0,
            "gaussian_minus_theory_days": 0.0,
            "note": "driver anchor; not a fitted response posterior",
        }
    )
    selection_rows.append(
        {
            "band": "W2",
            "retained_probability": 1.0,
            "comp0_full_q16_days": 0.0,
            "comp0_full_median_days": 0.0,
            "comp0_full_q84_days": 0.0,
            "comp0_selected_q16_days": 0.0,
            "comp0_selected_median_days": 0.0,
            "comp0_selected_q84_days": 0.0,
            "f_full_q16": 1.0,
            "f_full_median": 1.0,
            "f_full_q84": 1.0,
            "f_selected_q16": 1.0,
            "f_selected_median": 1.0,
            "f_selected_q84": 1.0,
            "note": "driver anchor; f=1 by convention",
        }
    )

    for band in RESPONSES:
        g0 = qrow(gamma[band]["comp0"], gamma_w[band])
        gs0 = qrow(selected[band]["comp0"], selected_w[band])
        ga0 = qrow(gaussian[band]["comp0"], gaussian_w[band])
        gf = qrow(gamma[band]["f"], gamma_w[band])
        gsf = qrow(selected[band]["f"], selected_w[band])
        comparison_rows.append(
            {
                "band": band,
                "theory_days": theory[band],
                "gamma_full_q16_days": g0["q16"],
                "gamma_full_median_days": g0["median"],
                "gamma_full_q84_days": g0["q84"],
                "gamma_selected_q16_days": gs0["q16"],
                "gamma_selected_median_days": gs0["median"],
                "gamma_selected_q84_days": gs0["q84"],
                "gamma_retained_probability": retained[band],
                "gaussian_q16_days": ga0["q16"],
                "gaussian_median_days": ga0["median"],
                "gaussian_q84_days": ga0["q84"],
                "gamma_selected_minus_theory_days": gs0["median"] - theory[band],
                "gaussian_minus_theory_days": ga0["median"] - theory[band],
                "note": "",
            }
        )
        selection_rows.append(
            {
                "band": band,
                "retained_probability": retained[band],
                "comp0_full_q16_days": g0["q16"],
                "comp0_full_median_days": g0["median"],
                "comp0_full_q84_days": g0["q84"],
                "comp0_selected_q16_days": gs0["q16"],
                "comp0_selected_median_days": gs0["median"],
                "comp0_selected_q84_days": gs0["q84"],
                "f_full_q16": gf["q16"],
                "f_full_median": gf["median"],
                "f_full_q84": gf["q84"],
                "f_selected_q16": gsf["q16"],
                "f_selected_median": gsf["median"],
                "f_selected_q84": gsf["q84"],
                "note": "selection: comp0>0 and comp0<comp1 after per-draw centroid sorting",
            }
        )

        for label, values, weights in (
            ("full", gamma[band], gamma_w[band]),
            ("selected", selected[band], selected_w[band]),
        ):
            q1 = qrow(values["comp1"], weights)
            f1 = qrow(1.0 - values["f"], weights)
            blr_rows.append(
                {
                    "band": band,
                    "posterior": label,
                    "comp1_q16_days": q1["q16"],
                    "comp1_median_days": q1["median"],
                    "comp1_q84_days": q1["q84"],
                    "comp1_fraction_q16": f1["q16"],
                    "comp1_fraction_median": f1["median"],
                    "comp1_fraction_q84": f1["q84"],
                    "median_over_blr_rl_center": q1["median"] / BLR_RL_CENTER,
                    "median_minus_blr_w2_center_days": q1["median"] - BLR_W2_CENTER,
                    "p_comp1_in_blr_w2_band": probability(
                        (values["comp1"] >= BLR_W2_LOW)
                        & (values["comp1"] <= BLR_W2_HIGH),
                        weights,
                    ),
                    "q16_q84_overlaps_blr_w2_band": (
                        q1["q84"] >= BLR_W2_LOW and q1["q16"] <= BLR_W2_HIGH
                    ),
                    "blr_rl_center_days": BLR_RL_CENTER,
                    "blr_rl_0p3dex_band_days": f"{BLR_RL_LOW}-{BLR_RL_HIGH}",
                    "w2_to_optical_continuum_lag_days": W2_TO_OPTICAL_CONTINUUM_LAG,
                    "blr_expected_relative_w2_center_days": BLR_W2_CENTER,
                    "blr_expected_relative_w2_band_days": f"{BLR_W2_LOW:.1f}-{BLR_W2_HIGH:.1f}",
                    "blr_theory_source": BLR_THEORY_SOURCE,
                    "continuum_lag_source": CONTINUUM_LAG_SOURCE,
                    "hbeta_response_lag_days": HBETA_RESPONSE_LAG,
                    "hbeta_source": HBETA_SOURCE,
                }
            )

    write_csv(COMPARISON_CSV, comparison_rows, list(comparison_rows[0]))
    write_csv(SELECTION_CSV, selection_rows, list(selection_rows[0]))
    write_csv(BLR_CSV, blr_rows, list(blr_rows[0]))


def setup_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#E2E6EA", linewidth=0.65, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", top=True, right=True, labelsize=8.0)


def quantile_annotation(ax: plt.Axes, values: np.ndarray, weights: np.ndarray) -> None:
    q16, q50, q84 = weighted_quantiles(values, weights)
    ax.axvline(q50, color="#25282D", lw=1.2, zorder=4)
    ax.axvline(q16, color="#25282D", lw=0.8, ls=":", alpha=0.9, zorder=4)
    ax.axvline(q84, color="#25282D", lw=0.8, ls=":", alpha=0.9, zorder=4)
    ax.text(
        0.97,
        0.90,
        f"{q50:.3f} [{q16:.3f}, {q84:.3f}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#E2E6EA",
            "alpha": 0.90,
        },
    )


def make_histograms(
    theory: dict[str, float],
    gamma: dict[str, dict[str, np.ndarray]],
    gamma_w: dict[str, np.ndarray],
    selected: dict[str, dict[str, np.ndarray]],
    selected_w: dict[str, np.ndarray],
    retained: dict[str, float],
) -> None:
    fig, axes = plt.subplots(
        len(BANDS),
        4,
        figsize=(16.0, 17.4),
        gridspec_kw={"hspace": 0.50, "wspace": 0.24},
    )
    fig.suptitle(
        "NGC 4151a Gamma2: posterior before and after short-lag selection",
        fontsize=17,
        y=0.995,
    )
    fig.text(
        0.5,
        0.978,
        "Selection: comp0 centroid > 0 and comp0 < comp1; components sorted by centroid per draw; posterior-weighted.",
        ha="center",
        va="top",
        fontsize=10.0,
        color="#59616B",
    )
    titles = (
        "comp0 centroid: full",
        "comp0 centroid: selected",
        r"$f_{\rm comp0}$: full",
        r"$f_{\rm comp0}$: selected",
    )
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11.5)

    colors = ("#3B6F91", "#C46B3B", "#3B6F91", "#C46B3B")
    for row, band in enumerate(BANDS):
        for col in range(4):
            setup_axis(axes[row, col])
            if col < 2:
                axes[row, col].axvline(0.0, color="#59616B", lw=0.8, alpha=0.75)
            if col == 0:
                axes[row, col].set_ylabel(f"{band}\nposterior density", fontsize=9.2)

        if band == "W2":
            for col, anchor in enumerate((0.0, 0.0, 1.0, 1.0)):
                ax = axes[row, col]
                ax.axvline(anchor, color=colors[col], lw=2.2)
                ax.text(
                    0.05,
                    0.76,
                    "driver anchor\nfixed, not fitted",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8.8,
                    color="#25282D",
                )
                ax.set_xlim((-0.5, 0.5) if col < 2 else (0.0, 1.0))
                ax.set_ylim(0.0, 1.0)
            continue

        full0 = gamma[band]["comp0"]
        post0 = selected[band]["comp0"]
        wf = gamma_w[band]
        ws = selected_w[band]
        lo_full, hi_full = weighted_quantiles(full0, wf, (0.025, 0.975))
        lo_sel, hi_sel = weighted_quantiles(post0, ws, (0.025, 0.975))
        lo = min(lo_full, lo_sel, 0.0, theory[band])
        hi = max(hi_full, hi_sel, 0.0, theory[band])
        span = max(hi - lo, 0.05)
        lo -= 0.08 * span
        hi += 0.08 * span
        bins0 = np.linspace(lo, hi, 49)

        for col, (values, weights) in enumerate(((full0, wf), (post0, ws))):
            ax = axes[row, col]
            ax.hist(
                values,
                bins=bins0,
                weights=weights,
                density=True,
                color=colors[col],
                alpha=0.65,
                edgecolor=colors[col],
                linewidth=0.38,
            )
            ax.axvline(theory[band], color="#6B4E9B", lw=1.25, ls="--", zorder=5)
            quantile_annotation(ax, values, weights)
            ax.set_xlim(lo, hi)
            ax.set_ylim(bottom=0.0)
        axes[row, 1].text(
            0.03,
            0.88,
            f"retained={retained[band]:.3f}",
            transform=axes[row, 1].transAxes,
            ha="left",
            va="top",
            fontsize=7.8,
            color="#59616B",
        )

        bins_f = np.linspace(0.0, 1.0, 41)
        for col, (values, weights) in enumerate(
            ((gamma[band]["f"], wf), (selected[band]["f"], ws)), start=2
        ):
            ax = axes[row, col]
            ax.hist(
                values,
                bins=bins_f,
                weights=weights,
                density=True,
                color=colors[col],
                alpha=0.65,
                edgecolor=colors[col],
                linewidth=0.38,
            )
            quantile_annotation(ax, values, weights)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(bottom=0.0)

    for col in range(4):
        axes[-1, col].set_xlabel(
            r"centroid relative to W2 [days]" if col < 2 else r"$f_{\rm comp0}$",
            fontsize=9.2,
        )
    fig.legend(
        handles=[
            Line2D([0], [0], color="#25282D", lw=1.2, label="median"),
            Line2D([0], [0], color="#25282D", lw=0.8, ls=":", label="16th/84th"),
            Line2D([0], [0], color="#6B4E9B", lw=1.25, ls="--", label="thin-disc theory"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.002),
        fontsize=9.0,
    )
    fig.text(
        0.995,
        0.005,
        "Centroid panels display the central 95% posterior range per band.",
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#59616B",
    )
    fig.savefig(HIST_BASE.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(HIST_BASE.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_blr_plot(
    gamma: dict[str, dict[str, np.ndarray]],
    gamma_w: dict[str, np.ndarray],
    selected: dict[str, dict[str, np.ndarray]],
    selected_w: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    y = np.arange(len(RESPONSES), dtype=float)
    full_stats = np.array(
        [weighted_quantiles(gamma[b]["comp1"], gamma_w[b]) for b in RESPONSES]
    )
    selected_stats = np.array(
        [weighted_quantiles(selected[b]["comp1"], selected_w[b]) for b in RESPONSES]
    )

    ax.axvspan(
        BLR_W2_LOW,
        BLR_W2_HIGH,
        color="#8A6BB8",
        alpha=0.18,
        label="predicted BLR band relative to W2 (3.9-13.2 d)",
    )
    ax.axvline(
        BLR_W2_CENTER,
        color="#59616B",
        lw=1.0,
        ls="--",
        label="predicted center relative to W2 (7.0 d)",
    )
    for stats, offset, color, marker, label, fill in (
        (full_stats, -0.11, "#3B6F91", "s", "full posterior", "none"),
        (selected_stats, 0.11, "#C46B3B", "o", "selected posterior", "#C46B3B"),
    ):
        q16, q50, q84 = stats[:, 0], stats[:, 1], stats[:, 2]
        ax.errorbar(
            q50,
            y + offset,
            xerr=np.vstack((q50 - q16, q84 - q50)),
            fmt=marker,
            ms=6.0,
            mfc=fill,
            mec=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=3.0,
            label=label,
            zorder=4,
        )
    ax.set_yticks(y, RESPONSES)
    ax.invert_yaxis()
    ax.set_xlabel("Gamma comp1 centroid relative to W2 [days]")
    ax.set_ylabel("response band")
    ax.set_title("NGC 4151a Gamma2: delayed centroid compared with BLR scales", pad=12)
    ax.text(
        0.5,
        1.01,
        "NGC 4151 R-L prediction: 6.2 light-days with 0.3 dex scatter; shifted by +0.8 d for comparison to W2-relative lags.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="#59616B",
    )
    ax.grid(axis="x", color="#E2E6EA", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=True,
        fontsize=8.8,
    )
    fig.savefig(BLR_BASE.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(BLR_BASE.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    theory = theory_lags()
    gamma: dict[str, dict[str, np.ndarray]] = {}
    gamma_w: dict[str, np.ndarray] = {}
    selected: dict[str, dict[str, np.ndarray]] = {}
    selected_w: dict[str, np.ndarray] = {}
    retained: dict[str, float] = {}
    gaussian: dict[str, dict[str, np.ndarray]] = {}
    gaussian_w: dict[str, np.ndarray] = {}

    for band in RESPONSES:
        gamma[band], gamma_w[band] = load_gamma(band)
        selected[band], selected_w[band], retained[band] = selected_gamma(
            gamma[band], gamma_w[band]
        )
        gaussian[band], gaussian_w[band] = load_gaussian(band)

    make_tables(
        theory,
        gamma,
        gamma_w,
        selected,
        selected_w,
        retained,
        gaussian,
        gaussian_w,
    )
    make_histograms(theory, gamma, gamma_w, selected, selected_w, retained)
    make_blr_plot(gamma, gamma_w, selected, selected_w)

    for path in (
        COMPARISON_CSV,
        SELECTION_CSV,
        BLR_CSV,
        HIST_BASE.with_suffix(".png"),
        HIST_BASE.with_suffix(".pdf"),
        BLR_BASE.with_suffix(".png"),
        BLR_BASE.with_suffix(".pdf"),
    ):
        print(path)


if __name__ == "__main__":
    main()
