#!/usr/bin/env python3
"""NGC 2617_a two-Gamma centroid/fraction analysis.

This script intentionally uses the already completed MICA posterior runs.  It
does not fit a new model.  Gamma component centroids are
tau = center + 2 exp(log(width)); components are relabelled per draw by
increasing centroid and f is the short-component amplitude fraction.
"""

# NGC2617_a 的两 Gamma 后验诊断：读取已完成的 sample1d+weights，写 raw/ 下的表与图。
# 绝对工作区和 BLR 比较数值均在本文件固定；W2 是人为零点锚，不是拟合出的无误差测量。
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ngc2617a-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


WORKSPACE = Path("/home/leonhard/sci001")
SOURCE = WORKSPACE / "NGC2617_a"
GAMMA_ROOT = SOURCE / "runs/mica/gamma2/2comp"
THEORY_FILE = SOURCE / "raw/theoretical_disk_lags.csv"
OUTDIR = SOURCE / "raw"

BANDS = ("W2", "M2", "W1", "U", "B", "V")
RESPONSES = BANDS[1:]
WAVELENGTHS = {"W2": 1928.0, "M2": 2246.0, "W1": 2600.0, "U": 3465.0, "B": 4392.0, "V": 5468.0}

# The 2013 NGC 2617 paper gives the H-beta R-L estimate used here.
BLR_RL_OPTICAL_CENTER = 9.0
BLR_RL_OPTICAL_LOW = 8.0
BLR_RL_OPTICAL_HIGH = 10.0
BLR_RL_SOURCE = "https://arxiv.org/abs/1310.2241"
OBS_HBETA_OPTICAL = 6.4
OBS_HBETA_OPTICAL_ERR = 0.5
OBS_HBETA_SOURCE = "https://arxiv.org/abs/1610.00008"

P16 = 0.16
P50 = 0.50
P84 = 0.84


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, probs=(P16, P50, P84)) -> tuple[float, ...]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    x = values[ok]
    w = weights[ok]
    if x.size == 0 or w.sum() <= 0.0:
        raise ValueError("No finite positive-weight samples")
    order = np.argsort(x)
    x = x[order]
    cdf = np.cumsum(w[order])
    cdf /= cdf[-1]
    return tuple(float(np.interp(p, cdf, x)) for p in probs)


# Kish ESS=(sum w)^2/sum(w^2) 反映权重集中程度，不是链自相关的有效样本数。
def kish_ess(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return float(1.0 / np.sum(weights * weights))


def probability(mask: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return float(weights[np.asarray(mask, dtype=bool)].sum() / weights.sum())


def stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    q16, q50, q84 = weighted_quantiles(values, weights)
    return {"q16": q16, "median": q50, "q84": q84}


def load_weights(data_dir: Path) -> np.ndarray:
    raw = np.loadtxt(data_dir / "weights.txt")
    weights = np.asarray(raw[:, -1], dtype=float)
    weights = np.where(np.isfinite(weights), np.clip(weights, 0.0, None), 0.0)
    if weights.sum() <= 0.0:
        raise ValueError(f"Posterior weights have zero mass: {data_dir}")
    return weights / weights.sum()


# Gamma 的 tau=c+2exp(logw)，面积 A=exp(logA)；按 tau 同步重排，再计算 f_short=A_short/(A0+A1)。
def load_gamma(band: str) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    if band == "W2":
        # W2 is the driver anchor, not a fitted response band.
        n = 10000
        return {
            "comp0": np.zeros(n),
            "comp1": np.zeros(n),
            "f": np.full(n, np.nan),
        }, np.full(n, 1.0 / n), "driver anchor; not fitted"

    run_name = f"run_W2_to_{band}_2comp_gamma"
    data_dir = GAMMA_ROOT / run_name / "data"
    sample = np.loadtxt(data_dir / "sample1d.txt_2", comments="#")
    weights = load_weights(data_dir)
    if sample.ndim != 2 or sample.shape[1] < 10 or sample.shape[0] != weights.shape[0]:
        raise ValueError(f"Sample/weight shape mismatch: {data_dir}")

    native_tau = np.column_stack(
        (
            sample[:, 5] + 2.0 * np.exp(sample[:, 6]),
            sample[:, 8] + 2.0 * np.exp(sample[:, 9]),
        )
    )
    native_amp = np.column_stack((np.exp(sample[:, 4]), np.exp(sample[:, 7])))
    order = np.argsort(native_tau, axis=1)
    tau = np.take_along_axis(native_tau, order, axis=1)
    amp = np.take_along_axis(native_amp, order, axis=1)
    f0 = amp[:, 0] / amp.sum(axis=1)
    return {"comp0": tau[:, 0], "comp1": tau[:, 1], "f": f0}, weights, run_name


# 保留 0<tau_short<tau_long，并用保留的后验质量重新归一权重；记录 retained 是必要的。
def select_gamma(values: dict[str, np.ndarray], weights: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    keep = (values["comp0"] > 0.0) & (values["comp0"] < values["comp1"])
    retained = float(weights[keep].sum())
    if retained <= 0.0:
        raise ValueError("Centroid selection retained zero posterior mass")
    selected = {key: value[keep] for key, value in values.items()}
    selected_weights = weights[keep] / retained
    return selected, selected_weights, retained


def read_theory() -> dict[str, float]:
    result: dict[str, float] = {}
    with THEORY_FILE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["band"]] = float(row["theory_relative_lag_days"])
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_histograms(
    theory: dict[str, float],
    gamma: dict[str, dict[str, np.ndarray]],
    gamma_w: dict[str, np.ndarray],
    selected: dict[str, dict[str, np.ndarray]],
    selected_w: dict[str, np.ndarray],
    retained: dict[str, float],
) -> Path:
    fig, axes = plt.subplots(6, 4, figsize=(16.0, 15.0), gridspec_kw={"hspace": 0.52, "wspace": 0.24})
    fig.suptitle("NGC 2617_a: 2Gamma comp0 centroid and f distributions", fontsize=17, y=0.995)
    fig.text(
        0.5,
        0.976,
        r"Selection: $0<\tau_{0}<\tau_{1}$ after per-draw centroid sorting; posterior-weighted; $f=A_{0}/(A_{0}+A_{1})$.",
        ha="center",
        va="top",
        fontsize=10,
        color="#59616B",
    )
    titles = (r"$\tau_0$: raw", r"$\tau_0$: filtered", r"$f$: raw", r"$f$: filtered")
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=11.5)

    colors = ("#3B6F91", "#C46B3B", "#3B6F91", "#C46B3B")
    for row, band in enumerate(BANDS):
        for col in range(4):
            ax = axes[row, col]
            ax.grid(axis="y", color="#E2E6EA", linewidth=0.65)
            ax.set_axisbelow(True)
            ax.tick_params(direction="in", top=True, right=True, labelsize=8)
            if col == 0:
                retained_text = "driver" if band == "W2" else f"retained={retained[band]:.3f}"
                ax.set_ylabel(f"{band}\n{retained_text}\nposterior density", fontsize=8.7)

        if band == "W2":
            for col in (0, 1):
                ax = axes[row, col]
                ax.axvline(0.0, color=colors[col], lw=2.0)
                ax.set_xlim(-1.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.text(0.05, 0.82, "driver anchor\n$\\tau=0$ fixed", transform=ax.transAxes, fontsize=9, va="top")
            for col in (2, 3):
                ax = axes[row, col]
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.0)
                ax.text(0.5, 0.55, "driver anchor\n(no $f$)", transform=ax.transAxes, ha="center", va="center", fontsize=9)
            continue

        full0, wf = gamma[band]["comp0"], gamma_w[band]
        sel0, ws = selected[band]["comp0"], selected_w[band]
        lo_full, hi_full = weighted_quantiles(full0, wf, (0.005, 0.995))
        lo_sel, hi_sel = weighted_quantiles(sel0, ws, (0.005, 0.995))
        lo = min(lo_full, lo_sel, theory[band], 0.0)
        hi = max(hi_full, hi_sel, theory[band], 0.0)
        span = max(hi - lo, 0.1)
        bins_tau = np.linspace(lo - 0.06 * span, hi + 0.06 * span, 51)

        for col, (values, weights) in enumerate(((full0, wf), (sel0, ws))):
            ax = axes[row, col]
            ax.hist(values, bins=bins_tau, weights=weights, density=True, color=colors[col], alpha=0.68, edgecolor=colors[col], linewidth=0.35)
            q16, q50, q84 = weighted_quantiles(values, weights)
            ax.axvline(q50, color="#25282D", lw=1.2)
            ax.axvline(q16, color="#25282D", lw=0.8, ls=":")
            ax.axvline(q84, color="#25282D", lw=0.8, ls=":")
            ax.axvline(theory[band], color="#6B4E9B", lw=1.25, ls="--")
            ax.axvline(0.0, color="#59616B", lw=0.7, alpha=0.75)
            ax.text(0.97, 0.90, f"{q50:.3f} [{q16:.3f}, {q84:.3f}]", transform=ax.transAxes, ha="right", va="top", fontsize=7.6, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#E2E6EA", "alpha": 0.9})
            ax.set_xlim(bins_tau[0], bins_tau[-1])
            ax.set_ylim(bottom=0.0)

        bins_f = np.linspace(0.0, 1.0, 41)
        for col, (values, weights) in enumerate(((gamma[band]["f"], wf), (selected[band]["f"], ws)), start=2):
            ax = axes[row, col]
            ax.hist(values, bins=bins_f, weights=weights, density=True, color=colors[col], alpha=0.68, edgecolor=colors[col], linewidth=0.35)
            q16, q50, q84 = weighted_quantiles(values, weights)
            ax.axvline(q50, color="#25282D", lw=1.2)
            ax.axvline(q16, color="#25282D", lw=0.8, ls=":")
            ax.axvline(q84, color="#25282D", lw=0.8, ls=":")
            ax.text(0.97, 0.90, f"{q50:.3f} [{q16:.3f}, {q84:.3f}]", transform=ax.transAxes, ha="right", va="top", fontsize=7.6, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#E2E6EA", "alpha": 0.9})
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(bottom=0.0)

    for col in range(4):
        axes[-1, col].set_xlabel("centroid relative to W2 [days]" if col < 2 else r"$f=A_0/(A_0+A_1)$", fontsize=9.2)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#25282D", lw=1.2, label="median"),
            Line2D([0], [0], color="#25282D", lw=0.8, ls=":", label="16th/84th"),
            Line2D([0], [0], color="#6B4E9B", lw=1.25, ls="--", label="thin-disc theory (centroid panels)"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.002),
        fontsize=9,
    )
    fig.text(0.995, 0.005, "Centroid panels show the central 99% posterior range per response band.", ha="right", va="bottom", fontsize=8, color="#59616B")
    output = OUTDIR / "ngc2617_a_2gamma_comp0_f_histograms.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def make_blr_plot(
    gamma: dict[str, dict[str, np.ndarray]],
    gamma_w: dict[str, np.ndarray],
    selected: dict[str, dict[str, np.ndarray]],
    selected_w: dict[str, np.ndarray],
    blr_center: float,
    blr_low: float,
    blr_high: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    y = np.arange(len(RESPONSES), dtype=float)
    ax.axvspan(blr_low, blr_high, color="#8A6BB8", alpha=0.18, label=f"BLR R-L band ({blr_low:.1f}-{blr_high:.1f} d relative W2)")
    ax.axvline(blr_center, color="#6B4E9B", lw=1.25, ls="--", label=f"BLR R-L center ({blr_center:.1f} d relative W2)")
    observed = OBS_HBETA_OPTICAL + (blr_center - BLR_RL_OPTICAL_CENTER)
    ax.axvline(observed, color="#59616B", lw=1.1, ls=":", label=f"observed H-beta guide ({observed:.1f} d relative W2)")

    full_stats = np.asarray([weighted_quantiles(gamma[b]["comp1"], gamma_w[b]) for b in RESPONSES])
    selected_stats = np.asarray([weighted_quantiles(selected[b]["comp1"], selected_w[b]) for b in RESPONSES])
    for sample_stats, offset, color, marker, label, fill in (
        (full_stats, -0.11, "#3B6F91", "s", "raw posterior", "none"),
        (selected_stats, 0.11, "#C46B3B", "o", "filtered posterior", "#C46B3B"),
    ):
        q16, q50, q84 = sample_stats[:, 0], sample_stats[:, 1], sample_stats[:, 2]
        ax.errorbar(q50, y + offset, xerr=np.vstack((q50 - q16, q84 - q50)), fmt=marker, ms=6.2, mfc=fill, mec=color, ecolor=color, elinewidth=1.3, capsize=3, label=label, zorder=4)
    ax.set_yticks(y, RESPONSES)
    ax.invert_yaxis()
    ax.set_xlabel("2Gamma comp1 centroid relative to W2 [days]")
    ax.set_ylabel("response band")
    ax.set_title("NGC 2617_a: delayed 2Gamma comp1 centroid versus BLR size", pad=12)
    ax.text(0.5, 1.01, "BLR reference: R_BLR=9±1 light-days from the 2013 R-L estimate; shifted by the local W2-to-5100 thin-disc lag.", transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#59616B")
    ax.grid(axis="x", color="#E2E6EA", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=True, fontsize=8.5)
    output = OUTDIR / "ngc2617_a_2gamma_comp1_blr_comparison.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    theory = read_theory()
    gamma: dict[str, dict[str, np.ndarray]] = {}
    gamma_w: dict[str, np.ndarray] = {}
    gamma_runs: dict[str, str] = {}
    selected: dict[str, dict[str, np.ndarray]] = {}
    selected_w: dict[str, np.ndarray] = {}
    retained: dict[str, float] = {}

    for band in BANDS:
        gamma[band], gamma_w[band], gamma_runs[band] = load_gamma(band)
        if band == "W2":
            selected[band], selected_w[band], retained[band] = gamma[band], gamma_w[band], 1.0
        else:
            selected[band], selected_w[band], retained[band] = select_gamma(gamma[band], gamma_w[band])

    # Interpolate the local disk-theory lag from W2 to 5100 A.  The BLR R-L
    # estimate is measured against optical continuum, while MICA centroids are
    # relative to W2; this small shift keeps the comparison in one coordinate.
    tau_w2_to_5100 = float(np.interp(5100.0, [WAVELENGTHS["B"], WAVELENGTHS["V"]], [theory["B"], theory["V"]]))
    blr_center_w2 = BLR_RL_OPTICAL_CENTER + tau_w2_to_5100
    blr_low_w2 = BLR_RL_OPTICAL_LOW + tau_w2_to_5100
    blr_high_w2 = BLR_RL_OPTICAL_HIGH + tau_w2_to_5100
    observed_hbeta_w2 = OBS_HBETA_OPTICAL + tau_w2_to_5100

    comp0_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    blr_rows: list[dict[str, object]] = []
    for band in BANDS:
        if band == "W2":
            comp0_rows.append({
                "band": band,
                "wavelength_angstrom": WAVELENGTHS[band],
                "theory_relative_lag_days": 0.0,
                "gamma_raw_q16_days": 0.0,
                "gamma_raw_median_days": 0.0,
                "gamma_raw_q84_days": 0.0,
                "gamma_filtered_q16_days": 0.0,
                "gamma_filtered_median_days": 0.0,
                "gamma_filtered_q84_days": 0.0,
                "gamma_retained_probability": 1.0,
                "gamma_raw_ess": kish_ess(gamma_w[band]),
                "gamma_filtered_ess": kish_ess(selected_w[band]),
                "gaussian_raw_q16_days": "",
                "gaussian_raw_median_days": "",
                "gaussian_raw_q84_days": "",
                "gaussian_status": "NGC2617_a 2Gaussian posterior not present",
                "gamma_run": gamma_runs[band],
            })
            continue

        raw0 = stats(gamma[band]["comp0"], gamma_w[band])
        sel0 = stats(selected[band]["comp0"], selected_w[band])
        rawf = stats(gamma[band]["f"], gamma_w[band])
        self = stats(selected[band]["f"], selected_w[band])
        comp0_rows.append({
            "band": band,
            "wavelength_angstrom": WAVELENGTHS[band],
            "theory_relative_lag_days": theory[band],
            "gamma_raw_q16_days": raw0["q16"],
            "gamma_raw_median_days": raw0["median"],
            "gamma_raw_q84_days": raw0["q84"],
            "gamma_filtered_q16_days": sel0["q16"],
            "gamma_filtered_median_days": sel0["median"],
            "gamma_filtered_q84_days": sel0["q84"],
            "gamma_retained_probability": retained[band],
            "gamma_raw_ess": kish_ess(gamma_w[band]),
            "gamma_filtered_ess": kish_ess(selected_w[band]),
            "gaussian_raw_q16_days": "",
            "gaussian_raw_median_days": "",
            "gaussian_raw_q84_days": "",
            "gaussian_status": "NGC2617_a 2Gaussian posterior not present",
            "gamma_run": gamma_runs[band],
        })
        selection_rows.append({
            "band": band,
            "retained_probability": retained[band],
            "raw_ess": kish_ess(gamma_w[band]),
            "filtered_ess": kish_ess(selected_w[band]),
            "comp0_raw_q16_days": raw0["q16"],
            "comp0_raw_median_days": raw0["median"],
            "comp0_raw_q84_days": raw0["q84"],
            "comp0_filtered_q16_days": sel0["q16"],
            "comp0_filtered_median_days": sel0["median"],
            "comp0_filtered_q84_days": sel0["q84"],
            "f_raw_q16": rawf["q16"],
            "f_raw_median": rawf["median"],
            "f_raw_q84": rawf["q84"],
            "f_filtered_q16": self["q16"],
            "f_filtered_median": self["median"],
            "f_filtered_q84": self["q84"],
            "selection_definition": "comp0 > 0 and comp0 < comp1 after per-draw centroid sorting",
        })
        for label, values, weights in (("raw", gamma[band], gamma_w[band]), ("filtered", selected[band], selected_w[band])):
            q1 = stats(values["comp1"], weights)
            p_in_blr = probability((values["comp1"] >= blr_low_w2) & (values["comp1"] <= blr_high_w2), weights)
            blr_rows.append({
                "band": band,
                "posterior": label,
                "comp1_q16_days": q1["q16"],
                "comp1_median_days": q1["median"],
                "comp1_q84_days": q1["q84"],
                "blr_rl_optical_center_days": BLR_RL_OPTICAL_CENTER,
                "blr_rl_optical_low_days": BLR_RL_OPTICAL_LOW,
                "blr_rl_optical_high_days": BLR_RL_OPTICAL_HIGH,
                "w2_to_5100_theory_lag_days": tau_w2_to_5100,
                "blr_relative_w2_center_days": blr_center_w2,
                "blr_relative_w2_low_days": blr_low_w2,
                "blr_relative_w2_high_days": blr_high_w2,
                "comp1_median_minus_blr_center_days": q1["median"] - blr_center_w2,
                "comp1_median_over_blr_center": q1["median"] / blr_center_w2,
                "posterior_mass_in_blr_band": p_in_blr,
                "observed_hbeta_optical_days": OBS_HBETA_OPTICAL,
                "observed_hbeta_optical_err_days": OBS_HBETA_OPTICAL_ERR,
                "observed_hbeta_relative_w2_days": observed_hbeta_w2,
                "blr_rl_source": BLR_RL_SOURCE,
                "observed_hbeta_source": OBS_HBETA_SOURCE,
            })

    comp0_path = OUTDIR / "ngc2617_a_2gamma_comp0_vs_theory.csv"
    selection_path = OUTDIR / "ngc2617_a_2gamma_selection_summary.csv"
    blr_path = OUTDIR / "ngc2617_a_2gamma_comp1_blr_comparison.csv"
    write_csv(comp0_path, comp0_rows)
    write_csv(selection_path, selection_rows)
    write_csv(blr_path, blr_rows)

    hist_path = make_histograms(theory, gamma, gamma_w, selected, selected_w, retained)
    blr_plot_path = make_blr_plot(gamma, gamma_w, selected, selected_w, blr_center_w2, blr_low_w2, blr_high_w2)

    metadata = {
        "source": "NGC2617_a",
        "gamma_runs_root": str(GAMMA_ROOT),
        "theory_file": str(THEORY_FILE),
        "gamma_centroid_definition": "tau = center + 2*exp(log(width))",
        "component_relabelling": "sort the two component centroids within each posterior draw; comp0 is shorter",
        "f_definition": "f = A0/(A0+A1) after relabelling",
        "selection": "comp0 > 0 and comp0 < comp1",
        "gaussian_status": "No NGC2617_a 2Gaussian posterior/run is present in the current workspace; Gaussian columns are blank rather than substituted.",
        "blr_reference": "2013 NGC2617 R-L estimate R_BLR=9 +/- 1 light-days, shifted to W2-relative coordinates using local thin-disc theory.",
        "blr_rl_source": BLR_RL_SOURCE,
        "observed_hbeta_source": OBS_HBETA_SOURCE,
        "w2_to_5100_theory_lag_days": tau_w2_to_5100,
        "blr_relative_w2_center_days": blr_center_w2,
        "blr_relative_w2_band_days": [blr_low_w2, blr_high_w2],
        "outputs": {
            "comp0_table": str(comp0_path),
            "selection_table": str(selection_path),
            "blr_table": str(blr_path),
            "histogram_png": str(hist_path),
            "histogram_pdf": str(hist_path.with_suffix(".pdf")),
            "blr_png": str(blr_plot_path),
            "blr_pdf": str(blr_plot_path.with_suffix(".pdf")),
        },
    }
    metadata_path = OUTDIR / "ngc2617_a_2gamma_analysis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(comp0_path)
    print(selection_path)
    print(blr_path)
    print(hist_path)
    print(hist_path.with_suffix(".pdf"))
    print(blr_plot_path)
    print(blr_plot_path.with_suffix(".pdf"))
    print(metadata_path)
    print(f"BLR relative-to-W2 reference: {blr_center_w2:.4f} d ({blr_low_w2:.4f}-{blr_high_w2:.4f} d)")


if __name__ == "__main__":
    main()
