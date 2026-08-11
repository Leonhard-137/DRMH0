#!/usr/bin/env python3
"""Low-cost checks for the Mrk817 two-component Gaussian/Gamma fits.

The script uses only the saved, equal-weight posterior samples.  It does not
run MICA.  It performs two checks after imposing tau0 < tau1:

1. Compare invariants of the *total* Gaussian and Gamma transfer functions.
2. Importance-reweight the Gamma samples from flat(log w0, log w1) to
   flat(w0, w1), for which the posterior weight is proportional to w0*w1.

For the second check, w0-only reweighting is also reported as a diagnostic.
It is not a coherent replacement of both width priors; it isolates whether an
unconstrained second component is dominating the full w0*w1 weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammainc, ndtr


BANDS = ("UVM2", "UVW1", "U", "B", "V")
MODELS = ("Gaussian", "Gamma")
MODEL_COLORS = {"Gaussian": "#3976A8", "Gamma": "#D97732"}
SQRT_2PI = math.sqrt(2.0 * math.pi)
LAG_LOW = -10.0
LAG_HIGH = 100.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--gamma-archive",
        type=Path,
        default=root / "Mrk817/runs/mica/legacy_archives/gamma0_100_2comp.zip",
    )
    parser.add_argument(
        "--gaussian-root",
        type=Path,
        default=root / "Mrk817/runs/mica/gaussian2_uvw2_lag_m10_100/2comp",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Mrk817/results/tf_comparison",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_gamma(archive: Path, band: str) -> tuple[np.ndarray, str]:
    member = (
        f"2comp/run_UVW2_to_{band}_2comp_gamma/data/"
        "posterior_sample1d.txt_2"
    )
    prior_member = (
        f"2comp/run_UVW2_to_{band}_2comp_gamma/data/para_names_line.txt_2"
    )
    with zipfile.ZipFile(archive) as zf:
        with zf.open(member) as handle:
            samples = np.loadtxt(handle)
        prior_text = zf.read(prior_member).decode("utf-8", errors="replace")
    return np.atleast_2d(samples), prior_text


def read_gaussian(root: Path, band: str) -> tuple[np.ndarray, str]:
    data_dir = root / f"run_UVW2_to_{band}_2comp_gaussian/data"
    samples = np.loadtxt(data_dir / "posterior_sample1d.txt_2")
    prior_text = (data_dir / "para_names_line.txt_2").read_text(errors="replace")
    return np.atleast_2d(samples), prior_text


def unpack_samples(raw: np.ndarray, model: str) -> dict[str, np.ndarray]:
    a0 = np.exp(raw[:, 4])
    c0 = raw[:, 5]
    w0 = np.exp(raw[:, 6])
    a1 = np.exp(raw[:, 7])
    c1 = raw[:, 8]
    w1 = np.exp(raw[:, 9])
    if model == "Gamma":
        tau0 = c0 + 2.0 * w0
        tau1 = c1 + 2.0 * w1
    else:
        tau0 = c0.copy()
        tau1 = c1.copy()
    keep = tau0 < tau1
    total_a = a0 + a1
    return {
        "a0": a0[keep],
        "c0": c0[keep],
        "w0": w0[keep],
        "a1": a1[keep],
        "c1": c1[keep],
        "w1": w1[keep],
        "tau0": tau0[keep],
        "tau1": tau1[keep],
        "f0": (a0 / total_a)[keep],
        "raw_n": np.array([raw.shape[0]], dtype=int),
    }


def component_cdf(t: np.ndarray, c: np.ndarray, w: np.ndarray, model: str) -> np.ndarray:
    if model == "Gaussian":
        return ndtr((t - c) / w)
    x = (t - c) / w
    xp = np.maximum(x, 0.0)
    # Regularized lower incomplete gamma is stable even for very small x.
    ans = gammainc(2.0, xp)
    return np.where(x > 0.0, ans, 0.0)


def mixture_cdf(t: np.ndarray, p: dict[str, np.ndarray], model: str) -> np.ndarray:
    return p["f0"] * component_cdf(t, p["c0"], p["w0"], model) + (
        1.0 - p["f0"]
    ) * component_cdf(t, p["c1"], p["w1"], model)


def mixture_density(
    t: np.ndarray, p: dict[str, np.ndarray], model: str
) -> np.ndarray:
    # Candidate grids are shaped (sample, candidate); align per-sample
    # parameters with an explicit trailing axis in that case.
    def aligned(value: np.ndarray) -> np.ndarray:
        return value[:, None] if t.ndim == 2 else value

    c0 = aligned(p["c0"])
    w0 = aligned(p["w0"])
    c1 = aligned(p["c1"])
    w1 = aligned(p["w1"])
    f0 = aligned(p["f0"])
    if model == "Gaussian":
        z0 = (t - c0) / w0
        z1 = (t - c1) / w1
        d0 = np.exp(-0.5 * z0 * z0) / (SQRT_2PI * w0)
        d1 = np.exp(-0.5 * z1 * z1) / (SQRT_2PI * w1)
    else:
        x0 = (t - c0) / w0
        x1 = (t - c1) / w1
        # np.where evaluates both branches; clip before exp to avoid an
        # irrelevant exp(-negative-large) overflow below the shifted support.
        x0_positive = np.maximum(x0, 0.0)
        x1_positive = np.maximum(x1, 0.0)
        d0 = np.where(x0 > 0.0, x0_positive * np.exp(-x0_positive) / w0, 0.0)
        d1 = np.where(x1 > 0.0, x1_positive * np.exp(-x1_positive) / w1, 0.0)
    return f0 * d0 + (1.0 - f0) * d1


def mixture_quantiles(
    p: dict[str, np.ndarray], model: str, probs: tuple[float, ...]
) -> np.ndarray:
    """Vectorized bisection over the mathematical, untruncated kernels."""
    if model == "Gaussian":
        low = np.minimum(p["c0"] - 12.0 * p["w0"], p["c1"] - 12.0 * p["w1"])
        high = np.maximum(p["c0"] + 12.0 * p["w0"], p["c1"] + 12.0 * p["w1"])
    else:
        low = np.minimum(p["c0"], p["c1"])
        high = np.maximum(p["c0"] + 50.0 * p["w0"], p["c1"] + 50.0 * p["w1"])
    answer = []
    for prob in probs:
        lo = low.copy()
        hi = high.copy()
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            below = mixture_cdf(mid, p, model) < prob
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        answer.append(0.5 * (lo + hi))
    return np.column_stack(answer)


def mixture_modes(p: dict[str, np.ndarray], model: str) -> np.ndarray:
    """Find each total-mixture mode using a coarse vector grid plus refinement."""
    n = p["f0"].size
    modes = np.empty(n)
    chunk_size = 400
    if model == "Gaussian":
        local_grid = np.linspace(-4.0, 4.0, 161)
        component_modes = (p["c0"], p["c1"])
    else:
        local_grid = np.linspace(0.0, 4.0, 201)
        component_modes = (p["c0"] + p["w0"], p["c1"] + p["w1"])
    between_grid = np.linspace(0.0, 1.0, 161)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        q = {key: value[start:stop] for key, value in p.items() if value.size == n}
        if model == "Gaussian":
            cand0 = q["c0"][:, None] + q["w0"][:, None] * local_grid
            cand1 = q["c1"][:, None] + q["w1"][:, None] * local_grid
            m0 = q["c0"]
            m1 = q["c1"]
        else:
            cand0 = q["c0"][:, None] + q["w0"][:, None] * local_grid
            cand1 = q["c1"][:, None] + q["w1"][:, None] * local_grid
            m0 = q["c0"] + q["w0"]
            m1 = q["c1"] + q["w1"]
        between = m0[:, None] + (m1 - m0)[:, None] * between_grid
        candidates = np.concatenate((cand0, cand1, between), axis=1)
        density = mixture_density(candidates, q, model)
        best = candidates[np.arange(stop - start), np.argmax(density, axis=1)]

        # The coarse spacing is <= 0.05 w (Gaussian) or 0.02 w (Gamma).
        # Refine around the nearest component mode with a bounded scalar search.
        for j, guess in enumerate(best):
            i = start + j
            nearest_zero = abs(guess - component_modes[0][i]) <= abs(
                guess - component_modes[1][i]
            )
            scale = p["w0"][i] if nearest_zero else p["w1"][i]
            half = 0.09 * scale

            def negative_density(x: float) -> float:
                one = {
                    "c0": np.array([p["c0"][i]]),
                    "w0": np.array([p["w0"][i]]),
                    "c1": np.array([p["c1"][i]]),
                    "w1": np.array([p["w1"][i]]),
                    "f0": np.array([p["f0"][i]]),
                }
                return -float(mixture_density(np.array([x]), one, model)[0])

            refined = minimize_scalar(
                negative_density,
                bounds=(guess - half, guess + half),
                method="bounded",
                options={"xatol": 1.0e-8, "maxiter": 60},
            )
            modes[i] = refined.x if refined.success else guess
    return modes


def truncated_first_moment(
    bound: float, c: np.ndarray, w: np.ndarray, model: str
) -> np.ndarray:
    """Integral from -infinity to bound of t times a normalized component."""
    if model == "Gaussian":
        z = (bound - c) / w
        phi = np.exp(-0.5 * z * z) / SQRT_2PI
        return c * ndtr(z) - w * phi
    x = (bound - c) / w
    xp = np.maximum(x, 0.0)
    f2 = gammainc(2.0, xp)
    gamma3_raw = 2.0 * gammainc(3.0, xp)
    ans = c * f2 + w * gamma3_raw
    return np.where(x > 0.0, ans, 0.0)


def compute_invariants(p: dict[str, np.ndarray], model: str) -> dict[str, np.ndarray]:
    quantiles = mixture_quantiles(p, model, (0.1, 0.5, 0.9))
    centroid = p["f0"] * p["tau0"] + (1.0 - p["f0"]) * p["tau1"]
    p_before_zero = mixture_cdf(np.zeros_like(p["f0"]), p, model)
    cdf_low = mixture_cdf(np.full_like(p["f0"], LAG_LOW), p, model)
    cdf_high = mixture_cdf(np.full_like(p["f0"], LAG_HIGH), p, model)
    window_mass = cdf_high - cdf_low
    first0 = truncated_first_moment(LAG_HIGH, p["c0"], p["w0"], model) - truncated_first_moment(
        LAG_LOW, p["c0"], p["w0"], model
    )
    first1 = truncated_first_moment(LAG_HIGH, p["c1"], p["w1"], model) - truncated_first_moment(
        LAG_LOW, p["c1"], p["w1"], model
    )
    window_centroid = (p["f0"] * first0 + (1.0 - p["f0"]) * first1) / window_mass
    return {
        "peak": mixture_modes(p, model),
        "centroid": centroid,
        "t10": quantiles[:, 0],
        "t50": quantiles[:, 1],
        "t90": quantiles[:, 2],
        "t90_minus_t10": quantiles[:, 2] - quantiles[:, 0],
        "p_before_zero": p_before_zero,
        "window_mass_m10_100": window_mass,
        "window_centroid_m10_100": window_centroid,
    }


def empirical_quantiles(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    probs = np.array([0.16, 0.50, 0.84])
    if weights is None:
        return np.quantile(values, probs)
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return np.interp(probs, cdf, x, left=x[0], right=x[-1])


def normalized_importance_weights(log_weight: np.ndarray) -> np.ndarray:
    shifted = log_weight - np.max(log_weight)
    weight = np.exp(shifted)
    return weight / np.sum(weight)


def effective_sample_size(weight: np.ndarray) -> float:
    return float(1.0 / np.sum(np.square(weight)))


def robust_edges(groups: list[np.ndarray], bins: int = 48) -> np.ndarray:
    all_values = np.concatenate(groups)
    lo, hi = np.quantile(all_values, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite plot limit")
    if hi <= lo:
        pad = max(1.0e-6, abs(lo) * 0.01)
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, bins + 1)


def clipped(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    tiny = (edges[-1] - edges[0]) * 1.0e-9
    return np.clip(values, edges[0] + tiny, edges[-1] - tiny)


def add_quantile_summary(
    rows: list[dict[str, object]],
    *,
    band: str,
    model_or_scheme: str,
    metric: str,
    values: np.ndarray,
    weights: np.ndarray | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    q16, q50, q84 = empirical_quantiles(values, weights)
    row: dict[str, object] = {
        "band": band,
        "model_or_scheme": model_or_scheme,
        "metric": metric,
        "q16": q16,
        "median": q50,
        "q84": q84,
    }
    if extra:
        row.update(extra)
    rows.append(row)


def plot_invariants(df: pd.DataFrame, output_base: Path) -> None:
    metrics = ("peak", "centroid", "t10", "t50", "t90")
    labels = {
        "peak": "total peak (d)",
        "centroid": "total centroid (d)",
        "t10": r"$t_{10}$ (d)",
        "t50": r"$t_{50}$ (d)",
        "t90": r"$t_{90}$ (d)",
    }
    fig, axes = plt.subplots(len(BANDS), len(metrics), figsize=(17.0, 13.0), constrained_layout=True)
    for row, band in enumerate(BANDS):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            groups = [
                df.loc[(df.band == band) & (df.model == model), metric].to_numpy()
                for model in MODELS
            ]
            edges = robust_edges(groups)
            for model, values in zip(MODELS, groups):
                q16, q50, q84 = np.quantile(values, [0.16, 0.50, 0.84])
                ax.hist(
                    clipped(values, edges),
                    bins=edges,
                    density=True,
                    histtype="step",
                    linewidth=1.45,
                    color=MODEL_COLORS[model],
                    label=model if row == 0 and col == 0 else None,
                )
                ax.axvline(q50, color=MODEL_COLORS[model], linewidth=1.0, alpha=0.85)
                short = "Gau" if model == "Gaussian" else "Gam"
                y = 0.96 if model == "Gaussian" else 0.83
                ax.text(
                    0.02,
                    y,
                    f"{short}: {q50:.3g} [{q16:.3g}, {q84:.3g}]",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7.3,
                    color=MODEL_COLORS[model],
                )
            if row == 0:
                ax.set_title(labels[metric], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{band}\ndensity", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.16, linewidth=0.5)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(
        "Mrk817: invariants of the total two-component transfer function\n"
        r"ordered samples ($\tau_0<\tau_1$); mathematical full support; "
        "display limits are 0.5--99.5% with outside samples edge-clipped",
        fontsize=13,
    )
    fig.savefig(output_base.with_suffix(".png"), dpi=210)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_reweight(
    by_band: dict[str, dict[str, np.ndarray]], output_base: Path
) -> None:
    variables = ("tau0", "two_w0", "f0")
    labels = {"tau0": r"$\tau_0=c_0+2w_0$ (d)", "two_w0": r"$2w_0$ (d)", "f0": r"$f_0$"}
    colors = {"original": "#555555", "flat_w0_w1": "#D97732", "flat_w0_only": "#3976A8"}
    fig, axes = plt.subplots(len(BANDS), len(variables), figsize=(12.4, 13.0), constrained_layout=True)
    for row, band in enumerate(BANDS):
        item = by_band[band]
        n = item["f0"].size
        ess = effective_sample_size(item["weight_flat_w0_w1"])
        for col, variable in enumerate(variables):
            ax = axes[row, col]
            values = item[variable]
            if variable == "f0":
                edges = np.linspace(max(0.0, np.quantile(values, 0.002)), min(1.0, np.quantile(values, 0.998)), 49)
            else:
                edges = robust_edges([values])
            xplot = clipped(values, edges)
            ax.hist(
                xplot,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.35,
                color=colors["original"],
                label="original" if row == 0 and col == 0 else None,
            )
            ax.hist(
                xplot,
                bins=edges,
                weights=item["weight_flat_w0_w1"],
                density=True,
                histtype="stepfilled",
                alpha=0.24,
                linewidth=1.35,
                color=colors["flat_w0_w1"],
                label=r"flat $w_0,w_1$ (main)" if row == 0 and col == 0 else None,
            )
            ax.hist(
                xplot,
                bins=edges,
                weights=item["weight_flat_w0_only"],
                density=True,
                histtype="step",
                linestyle="--",
                linewidth=1.25,
                color=colors["flat_w0_only"],
                label=r"flat $w_0$ only (diagnostic)" if row == 0 and col == 0 else None,
            )
            med_original = empirical_quantiles(values)[1]
            med_full = empirical_quantiles(values, item["weight_flat_w0_w1"])[1]
            med_w0 = empirical_quantiles(values, item["weight_flat_w0_only"])[1]
            ax.text(
                0.02,
                0.96,
                f"orig {med_original:.4g} | full {med_full:.4g} | w0 {med_w0:.4g}",
                transform=ax.transAxes,
                va="top",
                fontsize=7.5,
            )
            if row == 0:
                ax.set_title(labels[variable], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{band}\nESS {ess:.0f}/{n} ({ess/n:.1%})\ndensity", fontsize=9.4)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.16, linewidth=0.5)
    axes[0, 0].legend(frameon=False, fontsize=8.3, loc="center right")
    fig.suptitle(
        r"Mrk817 Gamma posterior: importance reweighting from flat $\log w$ to flat $w$"
        "\n"
        r"ordered samples ($\tau_0<\tau_1$); main weight $W\propto w_0w_1$; "
        "0.5--99.5% display limits",
        fontsize=13,
    )
    fig.savefig(output_base.with_suffix(".png"), dpi=210)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    invariant_records: list[pd.DataFrame] = []
    invariant_summary: list[dict[str, object]] = []
    reweight_records: list[pd.DataFrame] = []
    reweight_summary: list[dict[str, object]] = []
    validation: dict[str, object] = {
        "scope": "existing posterior only; no MICA rerun",
        "selection": "tau0 < tau1",
        "full_support": True,
        "lag_window_for_diagnostic": [LAG_LOW, LAG_HIGH],
        "bands": {},
    }
    gamma_for_plot: dict[str, dict[str, np.ndarray]] = {}

    with zipfile.ZipFile(args.gamma_archive) as gamma_zip:
        gamma_names = set(gamma_zip.namelist())

    for band in BANDS:
        raw_gamma, gamma_prior = read_gamma(args.gamma_archive, band)
        raw_gaussian, gaussian_prior = read_gaussian(args.gaussian_root, band)
        raw_by_model = {"Gaussian": raw_gaussian, "Gamma": raw_gamma}
        validation["bands"][band] = {
            "gamma_raw_n": int(raw_gamma.shape[0]),
            "gaussian_raw_n": int(raw_gaussian.shape[0]),
            "gamma_width_prior_is_log": all(
                "component_sigma      LOG" in line
                for line in gamma_prior.splitlines()
                if "component_sigma" in line
            ),
            "gaussian_width_prior_is_log": all(
                "component_sigma      LOG" in line
                for line in gaussian_prior.splitlines()
                if "component_sigma" in line
            ),
        }

        # If archived input light curves are present in both runs, compare them.
        input_match: dict[str, bool] = {}
        gamma_prefix = f"2comp/run_UVW2_to_{band}_2comp_gamma/data/"
        gaussian_data_dir = args.gaussian_root / f"run_UVW2_to_{band}_2comp_gaussian/data"
        for name in ("data_input.txt",):
            gamma_member = gamma_prefix + name
            gaussian_path = gaussian_data_dir / name
            if gamma_member in gamma_names and gaussian_path.exists():
                with zipfile.ZipFile(args.gamma_archive) as zf:
                    gamma_hash = sha256_bytes(zf.read(gamma_member))
                input_match[name] = gamma_hash == sha256_bytes(gaussian_path.read_bytes())
        validation["bands"][band]["input_file_hash_match"] = input_match

        for model in MODELS:
            p = unpack_samples(raw_by_model[model], model)
            n = p["f0"].size
            validation["bands"][band][f"{model.lower()}_ordered_n"] = int(n)
            inv = compute_invariants(p, model)
            quantile_errors = []
            for metric, prob in (("t10", 0.1), ("t50", 0.5), ("t90", 0.9)):
                quantile_errors.append(np.max(np.abs(mixture_cdf(inv[metric], p, model) - prob)))
            validation["bands"][band][f"{model.lower()}_max_cdf_quantile_error"] = float(
                max(quantile_errors)
            )
            if model == "Gaussian":
                component_mode0 = p["c0"]
                component_mode1 = p["c1"]
                lower_mode_bound = np.minimum(p["c0"], p["c1"])
                upper_mode_bound = np.maximum(p["c0"], p["c1"])
            else:
                component_mode0 = p["c0"] + p["w0"]
                component_mode1 = p["c1"] + p["w1"]
                lower_mode_bound = np.minimum(p["c0"], p["c1"])
                upper_mode_bound = np.maximum(component_mode0, component_mode1)
            peak_density = mixture_density(inv["peak"], p, model)
            component_peak_density = np.maximum(
                mixture_density(component_mode0, p, model),
                mixture_density(component_mode1, p, model),
            )
            validation["bands"][band][f"{model.lower()}_mode_check"] = {
                "max_component_mode_density_minus_reported_peak_density": float(
                    np.max(component_peak_density - peak_density)
                ),
                "below_theoretical_bound_count": int(
                    np.count_nonzero(inv["peak"] < lower_mode_bound - 1.0e-6)
                ),
                "above_theoretical_bound_count": int(
                    np.count_nonzero(inv["peak"] > upper_mode_bound + 1.0e-6)
                ),
            }
            frame = pd.DataFrame(inv)
            frame.insert(0, "sample", np.arange(n))
            frame.insert(0, "model", model)
            frame.insert(0, "band", band)
            invariant_records.append(frame)
            for metric, values in inv.items():
                add_quantile_summary(
                    invariant_summary,
                    band=band,
                    model_or_scheme=model,
                    metric=metric,
                    values=values,
                    extra={"n": n},
                )

        gamma = unpack_samples(raw_gamma, "Gamma")
        n = gamma["f0"].size
        gamma["two_w0"] = 2.0 * gamma["w0"]
        gamma["weight_flat_w0_w1"] = normalized_importance_weights(
            np.log(gamma["w0"]) + np.log(gamma["w1"])
        )
        gamma["weight_flat_w0_only"] = normalized_importance_weights(np.log(gamma["w0"]))
        gamma_for_plot[band] = gamma
        ess_full = effective_sample_size(gamma["weight_flat_w0_w1"])
        ess_w0 = effective_sample_size(gamma["weight_flat_w0_only"])
        validation["bands"][band]["reweight"] = {
            "weight_sum_flat_w0_w1": float(np.sum(gamma["weight_flat_w0_w1"])),
            "weight_sum_flat_w0_only": float(np.sum(gamma["weight_flat_w0_only"])),
            "ess_flat_w0_w1": ess_full,
            "ess_fraction_flat_w0_w1": ess_full / n,
            "ess_flat_w0_only": ess_w0,
            "ess_fraction_flat_w0_only": ess_w0 / n,
            "max_weight_flat_w0_w1": float(np.max(gamma["weight_flat_w0_w1"])),
        }
        reweight_frame = pd.DataFrame(
            {
                "band": band,
                "sample": np.arange(n),
                "c0": gamma["c0"],
                "w0": gamma["w0"],
                "tau0": gamma["tau0"],
                "two_w0": gamma["two_w0"],
                "f0": gamma["f0"],
                "c1": gamma["c1"],
                "w1": gamma["w1"],
                "tau1": gamma["tau1"],
                "weight_flat_w0_w1": gamma["weight_flat_w0_w1"],
                "weight_flat_w0_only": gamma["weight_flat_w0_only"],
            }
        )
        reweight_records.append(reweight_frame)
        schemes = {
            "original_flat_logw": None,
            "reweighted_flat_w0_w1": gamma["weight_flat_w0_w1"],
            "diagnostic_flat_w0_only": gamma["weight_flat_w0_only"],
        }
        for scheme, weights in schemes.items():
            ess = float(n) if weights is None else effective_sample_size(weights)
            for variable in ("tau0", "two_w0", "f0", "tau1", "w1"):
                add_quantile_summary(
                    reweight_summary,
                    band=band,
                    model_or_scheme=scheme,
                    metric=variable,
                    values=gamma[variable],
                    weights=weights,
                    extra={"n": n, "ess": ess, "ess_fraction": ess / n},
                )

    invariant_df = pd.concat(invariant_records, ignore_index=True)
    reweight_df = pd.concat(reweight_records, ignore_index=True)
    invariant_base = args.output_dir / "Mrk817_gaussian_gamma_total_tf_invariants"
    reweight_base = args.output_dir / "Mrk817_gamma_logw_reweighted_comp0"
    invariant_df.to_csv(invariant_base.with_name(invariant_base.name + "_posterior.csv"), index=False)
    pd.DataFrame(invariant_summary).to_csv(
        invariant_base.with_name(invariant_base.name + "_summary.csv"), index=False
    )
    reweight_df.to_csv(reweight_base.with_name(reweight_base.name + "_posterior.csv"), index=False)
    pd.DataFrame(reweight_summary).to_csv(
        reweight_base.with_name(reweight_base.name + "_summary.csv"), index=False
    )
    plot_invariants(invariant_df, invariant_base)
    plot_reweight(gamma_for_plot, reweight_base)
    validation_path = args.output_dir / "Mrk817_total_tf_and_logw_reweight_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {invariant_base}.png/.pdf and CSV files")
    print(f"Wrote {reweight_base}.png/.pdf and CSV files")
    print(f"Wrote {validation_path}")


if __name__ == "__main__":
    main()
