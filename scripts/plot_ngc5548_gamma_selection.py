#!/usr/bin/env python3
"""Unified NGC 5548 two-Gamma/two-Gaussian centroid comparison.

The Gamma transfer-function centroid is tau = c + 2 exp(log w); the Gaussian
centroid is tau = c.  Both models are relabelled draw by draw by increasing
centroid.  The requested Gamma selection is then applied to the relabelled
posterior: 0 < tau0 < tau1.  Nested-sampling weights are retained and
renormalized after selection.
"""

# 统一 NGC5548/Mrk509 的两核比较，源由 UNIFIED_SOURCE 环境变量选择。
# NGC5548 保留原时间约定；Mrk509 质心除以 1.0341。输出 analysis/gamma2_gaussian2_unified。
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ngc5548-unified-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


WORKSPACE = Path("/home/leonhard/sci001")
SOURCE_KEY = os.environ.get("UNIFIED_SOURCE", "NGC5548")
BANDS = ("W2", "M2", "W1", "U", "B", "V")
RESPONSES = BANDS[1:]
WAVELENGTHS = {"W2": 1928.0, "M2": 2246.0, "W1": 2600.0, "U": 3465.0, "B": 4392.0, "V": 5468.0}

if SOURCE_KEY == "NGC5548":
    SOURCE = WORKSPACE / "NGC5548"
    DISPLAY_SOURCE = "NGC 5548"
    FILE_PREFIX = "ngc5548"
    TIME_DILATION_FACTOR = 1.0  # Preserve the original project convention.
    UVW2_VS_1158 = 0.82
    H_BETA_VS_1158 = 6.23
    H_BETA_VS_5100 = 4.17
    OPTICAL_VS_1158 = 2.23
    BLR_OBS_W2 = H_BETA_VS_1158 - UVW2_VS_1158
    BLR_RL_5100_CENTER = 5.0 * H_BETA_VS_5100
    OPTICAL_VS_W2 = OPTICAL_VS_1158 - UVW2_VS_1158
    BLR_RL_SCATTER_DEX = 0.26
    BLR_RL_W2_CENTER = BLR_RL_5100_CENTER + OPTICAL_VS_W2
    BLR_RL_W2_LOW = BLR_RL_5100_CENTER / (10.0**BLR_RL_SCATTER_DEX) + OPTICAL_VS_W2
    BLR_RL_W2_HIGH = BLR_RL_5100_CENTER * (10.0**BLR_RL_SCATTER_DEX) + OPTICAL_VS_W2
    RL_BAND_LABEL = "historical R-L band"
    RL_CENTER_LABEL = "historical R-L center"
    OBS_HBETA_LABEL = "2014 H-beta observed"
    BLR_CONTEXT_NOTE = "H-beta is a guide only: broadband comp1 may contain diffuse-continuum or low-frequency response."
    REFERENCES = {
        "disk_theory": "https://arxiv.org/abs/1811.07956",
        "hbeta_measurement": "https://arxiv.org/abs/1702.01177",
        "uvw2_shift": "https://arxiv.org/abs/1510.05648",
        "rl_relation": "https://arxiv.org/abs/2209.10853",
    }
elif SOURCE_KEY == "Mrk509":
    SOURCE = WORKSPACE / "Mrk509"
    DISPLAY_SOURCE = "Mrk 509"
    FILE_PREFIX = "mrk509"
    TIME_DILATION_FACTOR = 1.0341
    # Rest-frame H-beta R-L prediction from the Bentz et al. (2013) Clean fit,
    # using the campaign L_5100=1.2e44 erg/s.  The W2-to-5100 shift uses the
    # project's thin-disc normalization R/c=0.318 d at rest UVW2.
    BLR_RL_5100_CENTER = 10.0 ** (1.555 + 0.542 * np.log10(1.2))
    OPTICAL_VS_W2 = 0.318 * ((5100.0 / (1928.0 / TIME_DILATION_FACTOR)) ** (4.0 / 3.0) - 1.0)
    BLR_RL_SCATTER_DEX = 0.13
    BLR_RL_W2_CENTER = BLR_RL_5100_CENTER + OPTICAL_VS_W2
    BLR_RL_W2_LOW = BLR_RL_5100_CENTER / (10.0**BLR_RL_SCATTER_DEX) + OPTICAL_VS_W2
    BLR_RL_W2_HIGH = BLR_RL_5100_CENTER * (10.0**BLR_RL_SCATTER_DEX) + OPTICAL_VS_W2
    BLR_OBS_W2 = 79.6 + OPTICAL_VS_W2
    RL_BAND_LABEL = "Bentz R-L prediction band"
    RL_CENTER_LABEL = "Bentz R-L prediction"
    OBS_HBETA_LABEL = "historical H-beta observed"
    BLR_CONTEXT_NOTE = "H-beta is a scale guide only: broadband comp1 may contain diffuse-continuum or low-frequency response."
    REFERENCES = {
        "disk_theory_and_l5100": "https://academic.oup.com/mnras/article/542/3/2572/8233645",
        "rl_relation": "https://arxiv.org/abs/1303.1742",
        "hbeta_measurement": "https://arxiv.org/abs/astro-ph/0407299",
    }
else:
    raise ValueError(f"Unsupported UNIFIED_SOURCE={SOURCE_KEY!r}")

GAMMA_ROOT = SOURCE / "runs/mica/gamma2/2comp"
GAUSSIAN_ROOT = SOURCE / "runs/mica/gaussian2/2comp"
THEORY_FILE = SOURCE / "raw/theoretical_disk_lags.csv"
OUTDIR = SOURCE / "analysis/gamma2_gaussian2_unified"

P16 = 0.15865
P84 = 0.84135


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, probs=(P16, 0.5, P84)) -> tuple[float, ...]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    x = values[ok]
    w = weights[ok]
    if x.size == 0 or w.sum() <= 0.0:
        raise ValueError("No finite positive-weight samples")
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return tuple(float(np.interp(p, cdf, x)) for p in probs)


def probability(mask: np.ndarray, weights: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    return float(np.asarray(weights, dtype=float)[mask].sum() / np.asarray(weights, dtype=float).sum())


def kish_ess(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float(1.0 / np.sum(w * w))


def load_weights(data_dir: Path) -> np.ndarray:
    raw = np.loadtxt(data_dir / "weights.txt")
    weights = np.asarray(raw[:, -1], dtype=float)
    weights = np.where(np.isfinite(weights), np.clip(weights, 0.0, None), 0.0)
    if weights.sum() <= 0.0:
        raise ValueError(f"Zero posterior mass: {data_dir}")
    return weights / weights.sum()


def gamma_data_dir(band: str) -> Path:
    suffix = "_4particle" if band == "V" else ""
    return GAMMA_ROOT / f"run_W2_to_{band}_2comp_gamma{suffix}/data"


# Gamma 先求 c+2w 再重标记，并除源专属 TIME_DILATION_FACTOR；振幅同步重排但不随时间单位缩放。
def load_gamma(band: str) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    data_dir = gamma_data_dir(band)
    sample = np.loadtxt(data_dir / "sample1d.txt_2", comments="#")
    weights = load_weights(data_dir)
    if sample.shape[0] != weights.shape[0]:
        raise ValueError(f"Sample/weight row mismatch: {data_dir}")
    native_tau = np.column_stack(
        (sample[:, 5] + 2.0 * np.exp(sample[:, 6]), sample[:, 8] + 2.0 * np.exp(sample[:, 9]))
    )
    native_amp = np.column_stack((np.exp(sample[:, 4]), np.exp(sample[:, 7])))
    order = np.argsort(native_tau, axis=1)
    tau = np.take_along_axis(native_tau, order, axis=1) / TIME_DILATION_FACTOR
    amp = np.take_along_axis(native_amp, order, axis=1)
    f0 = amp[:, 0] / amp.sum(axis=1)
    run_name = data_dir.parent.name
    return {"comp0": tau[:, 0], "comp1": tau[:, 1], "f": f0}, weights, run_name


# Gaussian 质心就是 c；这里与 Gamma 使用相同的按质心重标记和源时间换算规则。
def load_gaussian(band: str) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    data_dir = GAUSSIAN_ROOT / f"run_W2_to_{band}_2comp_gaussian/data"
    sample = np.loadtxt(data_dir / "sample1d.txt_2", comments="#")
    weights = load_weights(data_dir)
    if sample.shape[0] != weights.shape[0]:
        raise ValueError(f"Sample/weight row mismatch: {data_dir}")
    native_tau = np.column_stack((sample[:, 5], sample[:, 8]))
    native_amp = np.column_stack((np.exp(sample[:, 4]), np.exp(sample[:, 7])))
    order = np.argsort(native_tau, axis=1)
    tau = np.take_along_axis(native_tau, order, axis=1) / TIME_DILATION_FACTOR
    amp = np.take_along_axis(native_amp, order, axis=1)
    f0 = amp[:, 0] / amp.sum(axis=1)
    return {"comp0": tau[:, 0], "comp1": tau[:, 1], "f": f0}, weights, data_dir.parent.name


# 筛选是条件化后验，权重必须除 retained；同时报告 retained 与 Kish ESS 才能识别少量样本支配。
def select_gamma(values: dict[str, np.ndarray], weights: np.ndarray):
    keep = (values["comp0"] > 0.0) & (values["comp0"] < values["comp1"])
    retained = float(weights[keep].sum())
    if retained <= 0.0:
        raise ValueError("Selection retained zero posterior mass")
    selected = {key: value[keep] for key, value in values.items()}
    selected_weights = weights[keep] / retained
    return selected, selected_weights, retained


def theory_lags() -> dict[str, float]:
    result: dict[str, float] = {}
    with THEORY_FILE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["band"]] = float(row["theory_relative_lag_days"])
    return result


def stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    q16, q50, q84 = weighted_quantiles(values, weights)
    return {"q16": q16, "median": q50, "q84": q84}


def interval_text(s: dict[str, float], digits: int = 3) -> str:
    return f"{s['median']:.{digits}f} [{s['q16']:.{digits}f}, {s['q84']:.{digits}f}]"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def density_rows(values: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mass, edges = np.histogram(values, bins=bins, weights=weights)
    widths = np.diff(edges)
    density = mass / widths
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def make_histogram_figure(
    theory: dict[str, float],
    gamma: dict[str, dict[str, np.ndarray]],
    gamma_w: dict[str, np.ndarray],
    selected: dict[str, dict[str, np.ndarray]],
    selected_w: dict[str, np.ndarray],
    retained: dict[str, float],
) -> None:
    fig, axes = plt.subplots(len(RESPONSES), 4, figsize=(16.0, 15.0), gridspec_kw={"hspace": 0.50, "wspace": 0.24})
    fig.suptitle(f"{DISPLAY_SOURCE} two-Gamma posterior before and after centroid selection", fontsize=17, y=0.995)
    fig.text(
        0.5,
        0.977,
        r"Selection: $0<\tau_0<\tau_1$ after per-draw centroid relabelling; weighted posterior; $f_0=A_0/(A_0+A_1)$.",
        ha="center",
        va="top",
        fontsize=10,
        color="#59616B",
    )
    titles = (r"$\tau_0$: full", r"$\tau_0$: selected", r"$f_0$: full", r"$f_0$: selected")
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11.5)
    colors = ("#3B6F91", "#C46B3B", "#3B6F91", "#C46B3B")

    for row, band in enumerate(RESPONSES):
        full0 = gamma[band]["comp0"]
        selected0 = selected[band]["comp0"]
        wf = gamma_w[band]
        ws = selected_w[band]
        lo_f, hi_f = weighted_quantiles(full0, wf, (0.005, 0.995))
        lo_s, hi_s = weighted_quantiles(selected0, ws, (0.005, 0.995))
        lo = min(lo_f, lo_s, theory[band], 0.0)
        hi = max(hi_f, hi_s, theory[band], 0.0)
        span = max(hi - lo, 0.05)
        bins_tau = np.linspace(lo - 0.06 * span, hi + 0.06 * span, 51)

        for col in range(4):
            ax = axes[row, col]
            ax.grid(axis="y", color="#E2E6EA", linewidth=0.65)
            ax.set_axisbelow(True)
            ax.tick_params(direction="in", top=True, right=True, labelsize=8)
            if col == 0:
                ax.set_ylabel(f"{band}\nposterior density", fontsize=9)

        for col, (values, weights) in enumerate(((full0, wf), (selected0, ws))):
            ax = axes[row, col]
            ax.hist(values, bins=bins_tau, weights=weights, density=True, color=colors[col], alpha=0.68, edgecolor=colors[col], linewidth=0.35)
            q16, q50, q84 = weighted_quantiles(values, weights)
            ax.axvline(q50, color="#25282D", lw=1.2)
            ax.axvline(q16, color="#25282D", lw=0.8, ls=":")
            ax.axvline(q84, color="#25282D", lw=0.8, ls=":")
            ax.axvline(theory[band], color="#6B4E9B", lw=1.25, ls="--")
            ax.axvline(0.0, color="#59616B", lw=0.7, alpha=0.7)
            ax.text(0.97, 0.90, f"{q50:.3f} [{q16:.3f}, {q84:.3f}]", transform=ax.transAxes, ha="right", va="top", fontsize=7.7, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#E2E6EA", "alpha": 0.90})
            ax.set_xlim(bins_tau[0], bins_tau[-1])
            ax.set_ylim(bottom=0.0)
        axes[row, 1].text(0.03, 0.88, f"retained={retained[band]:.3f}", transform=axes[row, 1].transAxes, ha="left", va="top", fontsize=7.8, color="#59616B")

        bins_f = np.linspace(0.0, 1.0, 41)
        for col, (values, weights) in enumerate(((gamma[band]["f"], wf), (selected[band]["f"], ws)), start=2):
            ax = axes[row, col]
            ax.hist(values, bins=bins_f, weights=weights, density=True, color=colors[col], alpha=0.68, edgecolor=colors[col], linewidth=0.35)
            q16, q50, q84 = weighted_quantiles(values, weights)
            ax.axvline(q50, color="#25282D", lw=1.2)
            ax.axvline(q16, color="#25282D", lw=0.8, ls=":")
            ax.axvline(q84, color="#25282D", lw=0.8, ls=":")
            ax.text(0.97, 0.90, f"{q50:.3f} [{q16:.3f}, {q84:.3f}]", transform=ax.transAxes, ha="right", va="top", fontsize=7.7, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#E2E6EA", "alpha": 0.90})
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(bottom=0.0)

    for col in range(4):
        axes[-1, col].set_xlabel("rest-frame centroid relative to W2 [days]" if col < 2 else r"$f_0$", fontsize=9.2)
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
        fontsize=9,
    )
    fig.text(0.995, 0.005, "Centroid axes show the central 99% posterior range per band.", ha="right", va="bottom", fontsize=8, color="#59616B")
    for suffix in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{FILE_PREFIX}_gamma_selection_histograms.{suffix}", dpi=240 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def make_blr_figure(
    gamma: dict[str, dict[str, np.ndarray]],
    gamma_w: dict[str, np.ndarray],
    selected: dict[str, dict[str, np.ndarray]],
    selected_w: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    y = np.arange(len(RESPONSES), dtype=float)
    full_stats = np.asarray([weighted_quantiles(gamma[b]["comp1"], gamma_w[b]) for b in RESPONSES])
    selected_stats = np.asarray([weighted_quantiles(selected[b]["comp1"], selected_w[b]) for b in RESPONSES])
    ax.axvspan(BLR_RL_W2_LOW, BLR_RL_W2_HIGH, color="#8A6BB8", alpha=0.18, label=f"{RL_BAND_LABEL} ({BLR_RL_W2_LOW:.1f}-{BLR_RL_W2_HIGH:.1f} d)")
    ax.axvline(BLR_RL_W2_CENTER, color="#6B4E9B", lw=1.25, ls="--", label=f"{RL_CENTER_LABEL} ({BLR_RL_W2_CENTER:.1f} d)")
    ax.axvline(BLR_OBS_W2, color="#59616B", lw=1.15, ls=":", label=f"{OBS_HBETA_LABEL} ({BLR_OBS_W2:.1f} d)")
    for sample_stats, offset, color, marker, label, fill in (
        (full_stats, -0.11, "#3B6F91", "s", "full posterior", "none"),
        (selected_stats, 0.11, "#C46B3B", "o", "selected posterior", "#C46B3B"),
    ):
        q16, q50, q84 = sample_stats[:, 0], sample_stats[:, 1], sample_stats[:, 2]
        ax.errorbar(q50, y + offset, xerr=np.vstack((q50 - q16, q84 - q50)), fmt=marker, ms=6.2, mfc=fill, mec=color, ecolor=color, elinewidth=1.3, capsize=3, label=label, zorder=4)
    ax.set_yticks(y, RESPONSES)
    ax.invert_yaxis()
    ax.set_xlabel("rest-frame Gamma comp1 centroid relative to W2 [days]")
    ax.set_ylabel("response band")
    ax.set_title(f"{DISPLAY_SOURCE} two-Gamma comp1 centroids versus H-beta BLR scales", pad=12)
    ax.text(0.5, 1.01, BLR_CONTEXT_NOTE, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#59616B")
    ax.grid(axis="x", color="#E2E6EA", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=True, fontsize=8.5)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{FILE_PREFIX}_gamma_comp1_blr_comparison.{suffix}", dpi=240 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    theory = theory_lags()
    gamma: dict[str, dict[str, np.ndarray]] = {}
    gamma_w: dict[str, np.ndarray] = {}
    gamma_runs: dict[str, str] = {}
    selected: dict[str, dict[str, np.ndarray]] = {}
    selected_w: dict[str, np.ndarray] = {}
    retained: dict[str, float] = {}
    gaussian: dict[str, dict[str, np.ndarray]] = {}
    gaussian_w: dict[str, np.ndarray] = {}
    gaussian_runs: dict[str, str] = {}

    for band in RESPONSES:
        gamma[band], gamma_w[band], gamma_runs[band] = load_gamma(band)
        selected[band], selected_w[band], retained[band] = select_gamma(gamma[band], gamma_w[band])
        gaussian[band], gaussian_w[band], gaussian_runs[band] = load_gaussian(band)

    comparison_rows: list[dict[str, object]] = [
        {
            "band": "W2",
            "wavelength_angstrom": WAVELENGTHS["W2"],
            "theory_days": 0.0,
            "gamma_full_q16_days": 0.0,
            "gamma_full_median_days": 0.0,
            "gamma_full_q84_days": 0.0,
            "gamma_selected_q16_days": 0.0,
            "gamma_selected_median_days": 0.0,
            "gamma_selected_q84_days": 0.0,
            "gamma_retained_probability": 1.0,
            "gamma_selected_ess": 0.0,
            "gaussian_q16_days": 0.0,
            "gaussian_median_days": 0.0,
            "gaussian_q84_days": 0.0,
            "gaussian_ess": 0.0,
            "gamma_full_display": "0 (driver anchor)",
            "gamma_selected_display": "0 (driver anchor)",
            "gaussian_display": "0 (driver anchor)",
            "theory_display": "0.000",
            "gamma_run": "driver anchor; not fitted",
            "gaussian_run": "driver anchor; not fitted",
        }
    ]
    selection_rows: list[dict[str, object]] = []
    blr_rows: list[dict[str, object]] = []
    comp0_chart_rows: list[dict[str, object]] = []
    blr_chart_rows: list[dict[str, object]] = []
    comp0_density_rows: list[dict[str, object]] = []
    f_density_rows: list[dict[str, object]] = []

    comp0_chart_rows.extend(
        {"band": "W2", "series": series, "median_days": 0.0, "wavelength_angstrom": WAVELENGTHS["W2"], "retained_probability": 1.0}
        for series in ("thin-disc theory", "2Gamma full", "2Gamma selected", "2Gaussian full")
    )

    all_comp0_values = np.concatenate([gamma[b]["comp0"] for b in RESPONSES])
    all_comp0_weights = np.concatenate([gamma_w[b] / len(RESPONSES) for b in RESPONSES])
    global_lo, global_hi = weighted_quantiles(all_comp0_values, all_comp0_weights, (0.002, 0.998))
    global_lo = min(global_lo, min(theory[b] for b in RESPONSES), 0.0)
    global_hi = max(global_hi, max(theory[b] for b in RESPONSES), 0.0)
    global_span = max(global_hi - global_lo, 0.1)
    tau_bins = np.linspace(global_lo - 0.05 * global_span, global_hi + 0.05 * global_span, 61)
    f_bins = np.linspace(0.0, 1.0, 51)

    for band in RESPONSES:
        g0 = stats(gamma[band]["comp0"], gamma_w[band])
        gs0 = stats(selected[band]["comp0"], selected_w[band])
        ga0 = stats(gaussian[band]["comp0"], gaussian_w[band])
        gf = stats(gamma[band]["f"], gamma_w[band])
        gsf = stats(selected[band]["f"], selected_w[band])
        comparison_rows.append(
            {
                "band": band,
                "wavelength_angstrom": WAVELENGTHS[band],
                "theory_days": theory[band],
                "gamma_full_q16_days": g0["q16"],
                "gamma_full_median_days": g0["median"],
                "gamma_full_q84_days": g0["q84"],
                "gamma_selected_q16_days": gs0["q16"],
                "gamma_selected_median_days": gs0["median"],
                "gamma_selected_q84_days": gs0["q84"],
                "gamma_retained_probability": retained[band],
                "gamma_selected_ess": kish_ess(selected_w[band]),
                "gaussian_q16_days": ga0["q16"],
                "gaussian_median_days": ga0["median"],
                "gaussian_q84_days": ga0["q84"],
                "gaussian_ess": kish_ess(gaussian_w[band]),
                "gamma_full_display": interval_text(g0),
                "gamma_selected_display": interval_text(gs0),
                "gaussian_display": interval_text(ga0),
                "theory_display": f"{theory[band]:.3f}",
                "gamma_run": gamma_runs[band],
                "gaussian_run": gaussian_runs[band],
            }
        )
        selection_rows.append(
            {
                "band": band,
                "retained_probability": retained[band],
                "full_ess": kish_ess(gamma_w[band]),
                "selected_ess": kish_ess(selected_w[band]),
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
                "gamma_run": gamma_runs[band],
            }
        )
        for series, median in (
            ("thin-disc theory", theory[band]),
            ("2Gamma full", g0["median"]),
            ("2Gamma selected", gs0["median"]),
            ("2Gaussian full", ga0["median"]),
        ):
            comp0_chart_rows.append({"band": band, "series": series, "median_days": median, "wavelength_angstrom": WAVELENGTHS[band], "retained_probability": retained[band], "gamma_selected_q16_days": gs0["q16"], "gamma_selected_q84_days": gs0["q84"]})

        for posterior, values, weights in (("full", gamma[band], gamma_w[band]), ("selected", selected[band], selected_w[band])):
            q1 = stats(values["comp1"], weights)
            f1 = stats(1.0 - values["f"], weights)
            in_rl = probability((values["comp1"] >= BLR_RL_W2_LOW) & (values["comp1"] <= BLR_RL_W2_HIGH), weights)
            blr_rows.append(
                {
                    "band": band,
                    "posterior": posterior,
                    "comp1_q16_days": q1["q16"],
                    "comp1_median_days": q1["median"],
                    "comp1_q84_days": q1["q84"],
                    "comp1_display": interval_text(q1),
                    "f1_q16": f1["q16"],
                    "f1_median": f1["median"],
                    "f1_q84": f1["q84"],
                    "f1_display": interval_text(f1),
                    "p_comp1_in_rl_band": in_rl,
                    "q16_q84_overlaps_rl_band": bool(q1["q84"] >= BLR_RL_W2_LOW and q1["q16"] <= BLR_RL_W2_HIGH),
                    "rl_center_relative_w2_days": BLR_RL_W2_CENTER,
                    "rl_low_days": BLR_RL_W2_LOW,
                    "rl_high_days": BLR_RL_W2_HIGH,
                    "observed_hbeta_relative_w2_days": BLR_OBS_W2,
                    "gamma_run": gamma_runs[band],
                }
            )
            blr_chart_rows.append({"band": band, "series": f"2Gamma {posterior}", "centroid_days": q1["median"], "q16_days": q1["q16"], "q84_days": q1["q84"], "f1_median": f1["median"], "p_in_rl_band": in_rl})

            centers, densities = density_rows(values["comp0"], weights, tau_bins)
            for center, density in zip(centers, densities):
                comp0_density_rows.append({"band": band, "posterior": posterior, "bin_center_days": float(center), "posterior_density": float(density), "retained_probability": retained[band], "theory_days": theory[band]})
            centers, densities = density_rows(values["f"], weights, f_bins)
            for center, density in zip(centers, densities):
                f_density_rows.append({"band": band, "posterior": posterior, "bin_center_f0": float(center), "posterior_density": float(density), "retained_probability": retained[band]})

    for band in RESPONSES:
        blr_chart_rows.extend(
            [
                {"band": band, "series": RL_CENTER_LABEL, "centroid_days": BLR_RL_W2_CENTER, "q16_days": BLR_RL_W2_LOW, "q84_days": BLR_RL_W2_HIGH, "f1_median": None, "p_in_rl_band": None},
                {"band": band, "series": OBS_HBETA_LABEL, "centroid_days": BLR_OBS_W2, "q16_days": None, "q84_days": None, "f1_median": None, "p_in_rl_band": None},
            ]
        )

    write_csv(OUTDIR / f"{FILE_PREFIX}_comp0_centroid_comparison.csv", comparison_rows)
    write_csv(OUTDIR / f"{FILE_PREFIX}_gamma_selection_summary.csv", selection_rows)
    write_csv(OUTDIR / f"{FILE_PREFIX}_gamma_comp1_blr_comparison.csv", blr_rows)
    make_histogram_figure(theory, gamma, gamma_w, selected, selected_w, retained)
    make_blr_figure(gamma, gamma_w, selected, selected_w)

    report_data = {
        "comparison_rows": comparison_rows,
        "selection_rows": selection_rows,
        "blr_rows": blr_rows,
        "comp0_chart_rows": comp0_chart_rows,
        "blr_chart_rows": blr_chart_rows,
        "comp0_density_rows": comp0_density_rows,
        "f_density_rows": f_density_rows,
        "references": REFERENCES,
        "definitions": {
            "gamma_centroid": "tau=c+2*exp(logw)",
            "gaussian_centroid": "tau=c",
            "relabel": "sort complete component parameter tuples by centroid within each draw",
            "selection": "0<tau0<tau1 after relabelling",
            "f0": "A_short/(A_short+A_long)",
            "quantiles": "weighted 15.865/50/84.135 percentiles",
            "time_frame": "rest frame" if TIME_DILATION_FACTOR != 1.0 else "project convention",
            "time_dilation_factor": TIME_DILATION_FACTOR,
            "v_gamma_run": gamma_runs["V"],
        },
    }
    (OUTDIR / f"{FILE_PREFIX}_unified_report_data.json").write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "output_dir": str(OUTDIR),
        "comparison": comparison_rows,
        "blr_selected": [row for row in blr_rows if row["posterior"] == "selected"],
        "blr_scales": {"rl_center": BLR_RL_W2_CENTER, "rl_band": [BLR_RL_W2_LOW, BLR_RL_W2_HIGH], "observed_hbeta": BLR_OBS_W2},
    }, indent=2))


if __name__ == "__main__":
    main()
