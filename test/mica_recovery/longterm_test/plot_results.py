#!/usr/bin/env python3
"""Plot completed longterm_test fits: transfer functions, light curves, corner.

Examples:
  python plot_results.py --config experiment_real_disk_mismatch.yaml
  python plot_results.py --config experiment_broad.yaml --suite known_response --compare-input
  python plot_results.py --case disk_only --model gaussian --plots transfer

Outputs: CASE/results/{gaussian,gamma}/{transfer_function,light_curves,corner}.{jpg,pdf}
and summary.json. All draws are retained; TF/corner component labels are ordered
by centroid within each joint draw (C0 <= C1). Total TF quantiles are evaluated
after summing components in each draw, including tails outside the native saved
grid. Native pline light curves retain explicitly labelled native indices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, MaxNLocator
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincinv, ndtr, ndtri

from fit import build_plan
from prepare import DEFAULT_CONFIG, load_config
from plot_inputs import build_transfer_functions
from plot_style import (COLORS, COMPONENT_COLORS, SHORT_DASH, REFERENCE_DASH,
                        configure_style, observations, save_figure, style_axis)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from collect_mica_results import configured_run_name, mica_run_root, param_names, reorder_component_columns
from plot_mica_transfer import kernel, read_component_parameters, read_total, QUANTILES
from plot_mica_decomposition_paper import blocks

SUITES = ("all", "gaussian", "independent", "shared_slow", "known_response",
          "spectral_shared", "real_disk_mismatch", "paired_ablation")
PLOTS = ("transfer", "lightcurves", "corner")
MASS = 0.999
# Explicit project display choice, not a Gaussian/sigma interpretation of 2D.
CORNER_MASSES = (0.683, 0.954)


def quantiles(values):
    return np.quantile(values, QUANTILES).tolist()


def component_cdf(lag, centers, widths, kind):
    u = (lag - centers) / widths
    if kind == "gaussian":
        return ndtr(u)
    if kind == "gamma":
        return gammainc(2.0, np.maximum(u, 0.0))
    raise ValueError(f"Unsupported kernel family: {kind}")


def negative_fractions(amp, centers, widths, kind):
    parts = component_cdf(0.0, centers, widths, kind)
    total = np.sum(amp * parts, axis=0) / np.sum(amp, axis=0)
    return parts, total


def response_window(amp, centers, widths, kind, mass=MASS):
    """Central mass interval of the equally averaged unit-area draw kernels."""
    if not 0 < mass < 1:
        raise ValueError("Response mass must be between zero and one")
    tail = (1.0 - mass) / 2.0
    inv = ndtri if kind == "gaussian" else lambda p: gammaincinv(2.0, p)
    lo = float(np.min(centers + widths * inv(tail / 2.0)))
    hi = float(np.max(centers + widths * inv(1.0 - tail / 2.0)))
    weights = amp / amp.sum(axis=0)

    def cdf(t):
        return float(np.mean(np.sum(weights * component_cdf(t, centers, widths, kind), axis=0)))

    return (brentq(lambda t: cdf(t) - tail, lo, hi),
            brentq(lambda t: cdf(t) - (1.0 - tail), lo, hi))


def transfer_grid(centers, widths, kind, limits):
    """Resolve sub-day peaks even when the full window spans thousands of days."""
    lo, hi = limits
    pieces = [np.linspace(lo, hi, 1800), np.array([0.0])]
    for c, w in zip(centers, widths):
        for probability in (0.1585, 0.5, 0.8415):
            origin = float(np.quantile(c, probability))
            scale = float(np.quantile(w, probability))
            offsets = np.linspace(-6, 6, 600) if kind == "gaussian" else np.linspace(0, 12, 600)
            pieces.append(origin + scale * offsets)
        if kind == "gamma":
            # Pointwise median/interval onsets can be much sharper than a
            # uniform grid: retain exact empirical ranks and nearby points.
            ranks = (len(c) - 1) * np.asarray(QUANTILES)
            indices = (len(c) - 1 - np.r_[np.floor(ranks), np.ceil(ranks)]).astype(int)
            for onset in np.unique(np.sort(c)[indices]):
                pieces.append(np.r_[onset, onset + np.median(w) * np.geomspace(1e-9, 1, 40)])
    grid = np.unique(np.concatenate(pieces))
    return grid[(grid >= lo) & (grid <= hi)]


def transfer_quantiles(lag, amp, centers, widths, kind):
    """Total first, quantiles second. Never sum pointwise component medians."""
    total = np.empty((len(lag), 3))
    parts = np.empty((2, len(lag)))
    for first in range(0, len(lag), 160):
        section = slice(first, first + 160)
        draws = [kernel(lag[section], amp[j], centers[j], widths[j], kind) for j in range(2)]
        total[section] = np.quantile(draws[0] + draws[1], QUANTILES, axis=0).T
        for j in range(2):
            parts[j, section] = np.median(draws[j], axis=0)
    return total, parts


def load_run(job, *, component_sum_tolerance_sigma=0.0):
    case = job["case"]
    saved = load_config(case / "config/source_config.yaml")
    fit = dict(saved[job["stage"]])
    # Relabel whole joint draws before any component quantiles: C0 has the
    # smaller centroid, C1 the larger. Native pline averages cannot be
    # relabelled after averaging, so their legends use distinct native names.
    fit["component_order"] = "moment1"
    if fit.get("ncomp") != 2 or fit["type_tf"] not in ("gaussian", "gamma"):
        raise ValueError("This entrypoint expects the configured two-Gaussian/two-Gamma fits")
    run = mica_run_root(case, fit) / configured_run_name(fit, job["response"])
    state = json.loads((run / "run_mica_state.json").read_text())
    if state.get("status") != "complete":
        raise RuntimeError(f"Fit is not complete: {run}")
    data = run / "data"
    amp, centers, widths, centroids, kinds, nswapped = read_component_parameters(run, fit)
    names = param_names(data / "para_names_line.txt_2")
    samples = np.loadtxt(data / "posterior_sample1d.txt_2", ndmin=2)
    samples, swapped = reorder_component_columns(samples, names, 2, kinds[0], fit)
    if int(np.count_nonzero(swapped)) != nswapped or np.any(centroids[0] > centroids[1]):
        raise ValueError(f"Inconsistent centroid relabelling: {run}")
    native = read_total(data / "tranfunc_0_1.txt_2")
    # plotfig.py samples a uniform grid then prints %e. Recreate that grid
    # before comparing narrow peaks at large lags (up to 5e-5 d rounding).
    native_lag = np.linspace(native[0, 0], native[-1, 0], len(native))
    if not np.allclose(native_lag, native[:, 0], rtol=1e-6, atol=1e-6):
        native_lag = native[:, 0]
    predicted, _ = transfer_quantiles(native_lag, amp, centers, widths, kinds[0])
    error = float(np.max(np.abs(predicted - native[:, 1:])))
    tolerance = 2e-4 * max(float(np.max(native[:, 2])), 1e-10)
    if error > tolerance:
        raise ValueError(f"Posterior/native TF disagreement ({error:g} > {tolerance:g}): {run}")
    observed = [np.loadtxt(run / "input" / name, ndmin=2) for name in ("cont.txt", "line.txt")]
    reconstruction = blocks(data / "pall.txt_2")
    components = [blocks(data / f"pline.txt_2_comp{j}")[1] for j in range(2)]
    for curve in components:
        if not np.array_equal(curve[:, 0], reconstruction[1][:, 0]):
            raise ValueError(f"Component reconstruction grid mismatch: {run}")
    total_curve = reconstruction[1]
    if not all(np.isfinite(curve).all() for curve in [total_curve, *components]):
        raise ValueError(f"Non-finite native reconstruction: {run}")
    residual = components[0][:, 1] + components[1][:, 1] - total_curve[:, 1]
    strict_tolerance = 3e-5 + 3e-6 * np.abs(total_curve[:, 1])
    strict_pass = bool(np.all(np.abs(residual) <= strict_tolerance))
    # This optional display tolerance never changes native arrays or inference.
    # Keep strict checking by default; record small discrepancies explicitly.
    sigma_ratio = (float(np.max(np.abs(residual) / total_curve[:, 2]))
                   if np.all(total_curve[:, 2] > 0) else None)
    component_sum_check = dict(
        strict_pass=strict_pass, max_abs_residual=float(np.max(np.abs(residual))),
        rms_residual=float(np.sqrt(np.mean(residual**2))),
        strict_failed_points=int(np.count_nonzero(np.abs(residual) > strict_tolerance)),
        max_total_sigma_ratio=sigma_ratio,
        allowed_total_sigma_ratio=float(component_sum_tolerance_sigma),
        native_arrays_modified=False)
    if not strict_pass:
        if (sigma_ratio is None or component_sum_tolerance_sigma <= 0
                or sigma_ratio > component_sum_tolerance_sigma):
            raise ValueError(f"Native component means do not sum to total: {run}; "
                             f"diagnostics={component_sum_check}")
        warnings.warn(f"Native component sum discrepancy retained in plots: {run}; "
                      f"diagnostics={component_sum_check}", RuntimeWarning)
    evidence = np.loadtxt(data / "evidence.txt", ndmin=2)
    lnz = float(evidence[evidence[:, 0] == 2, 1].item())
    norm_table = np.loadtxt(data / "normalization.txt", ndmin=2)
    norms = [float(norm_table[(norm_table[:, 0] == 0) & (norm_table[:, 1] == j), 2].item())
             for j in range(2)]
    # Native TFs map internally normalized fluxes; native light-curve files
    # have already been converted back to the saved input flux units.
    bounds = {}
    for line in (data / "para_names_line.txt_2").read_text().splitlines():
        fields = line.split()
        if fields and not fields[0].startswith("#"):
            bounds[fields[1]] = [float(fields[3]), float(fields[4])]
    return dict(case=case, run=run, fit=fit, kind=kinds[0], amp=amp, centers=centers,
                widths=widths, centroids=centroids, samples=samples, names=names,
                observed=observed, reconstruction=reconstruction, components=components,
                bands=(fit["driver"], job["response"]), lnz=lnz, norms=norms,
                bounds=bounds, native_tf_max_error=error, component_order="moment1",
                component_swapped_draws=nswapped, lightcurve_component_order="native",
                component_sum_check=component_sum_check)


def input_reference(config, run):
    """Only compare kernels with the same observed-driver definition."""
    name = run["case"].name
    if name in config.get("known_response", {}).get("cases", {}):
        label = "Injected W2-to-U kernel"
    elif name in config.get("independent_response", {}).get("cases", {}):
        label = "Injected correlated-response kernel"
    elif name in config.get("spectral_shared", {}).get("cases", {}):
        label = "Designed effective predictor"
    else:
        return None
    lag, definitions = build_transfer_functions(config)
    definition = definitions[name]
    lag = definition.get("lag", lag)
    values = definition["total"] * run["norms"][0] / run["norms"][1]
    return lag, values, label


def plot_transfer(run, output, reference=None):
    limits = response_window(run["amp"], run["centers"], run["widths"], run["kind"])
    lag = transfer_grid(run["centers"], run["widths"], run["kind"], limits)
    total, parts = transfer_quantiles(lag, run["amp"], run["centers"], run["widths"], run["kind"])
    fraction, total_fraction = negative_fractions(run["amp"], run["centers"], run["widths"], run["kind"])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    for index, ax in enumerate(axes):
        log = index == 1
        display = lambda y: np.ma.masked_less_equal(y, 0) if log else y
        ax.fill_between(lag, display(total[:, 0]), display(total[:, 2]),
                        color=COLORS["UNCERTAINTY"], alpha=.32, linewidth=0, zorder=1)
        ax.plot(lag, display(total[:, 1]), color="black", lw=1.7, zorder=2,
                label=rf"Total ($f_-$ = {100*np.median(total_fraction):.1f}%)")
        for j, color in enumerate(COMPONENT_COLORS):
            ax.plot(lag, display(parts[j]), color=color, lw=1.35, linestyle=SHORT_DASH, zorder=3,
                    label=rf"C{j} ($f_-$ = {100*np.median(fraction[j]):.1f}%)")
        if reference is not None:
            ax.plot(reference[0], display(reference[1]), color=COLORS["REFERENCE"], ls=REFERENCE_DASH, lw=1.25,
                    label=reference[2], zorder=4)
        ax.axvline(0, color=COLORS["GUIDE"], lw=.9, zorder=0)
        ax.set_xlim(*limits)
        ax.set_xlabel(r"$\tau$ (days)", fontsize=13)
        ax.set_ylabel(r"Response (day$^{-1}$)", fontsize=13)
        maximum = float(np.max(total[:, 2]))
        if reference is not None:
            inside = (reference[0] >= limits[0]) & (reference[0] <= limits[1])
            if np.any(inside):
                maximum = max(maximum, float(np.max(reference[1][inside])))
        if log:
            ax.set_yscale("log")
            ax.set_ylim(maximum * 1e-6, maximum * 1.15)
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        else:
            ax.set_ylim(0, maximum * 1.12)
        style_axis(ax, y_on_right=log)
    axes[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.suptitle(f"{run['bands'][0]} to {run['bands'][1]} — {run['kind'].capitalize()}", fontsize=14)
    fig.subplots_adjust(left=.085, right=.915, bottom=.18, top=.87, wspace=.18)
    save_figure(fig, output)
    return dict(lag_limits_days=list(limits), enclosed_response_mass=MASS,
                window_definition="central interval of posterior-averaged unit-area total kernels",
                total_negative_fraction=quantiles(total_fraction),
                component_negative_fractions=[quantiles(value) for value in fraction],
                reference=None if reference is None else reference[2],
                total_estimator="sum within each posterior draw, then pointwise 15.85/50/84.15 percentiles",
                native_grid_max_absolute_difference=run["native_tf_max_error"])


def truth_components(run, band_index):
    band = run["bands"][band_index]
    path = run["case"] / "components" / f"{band}_components.txt"
    if not path.is_file():
        # Inference-only cases reuse old input directories through symlinks.
        path = (run["case"] / "input").resolve().parent / "components" / f"{band}_components.txt"
    if not path.is_file():
        return None
    with path.open() as handle:
        names = handle.readline().lstrip("# ").split()
    ablation_schema = ["time", "illumination_variation", "shared_long_variation",
                       "shared_short_variation", "measurement_noise",
                       "disk_illumination_variation", "blr_illumination_variation"]
    is_ablation = names == ablation_schema
    if not names or names[0] != "time" or (not is_ablation and names[-1] != "measurement_noise"):
        raise ValueError(f"Unknown component table schema: {path}")
    values = np.loadtxt(path, ndmin=2)
    observed = run["observed"][band_index]
    if values.shape != (len(observed), len(names)):
        raise ValueError(f"Truth/observation dimensions differ: {path}")
    np.testing.assert_allclose(values[:, 0], observed[:, 0], rtol=1e-8, atol=1e-6)
    # The final two ablation columns resolve illumination into subparts;
    # they are not additional contributions to the observed flux.
    noise_index = 4 if is_ablation else len(names) - 1
    if is_ablation:
        np.testing.assert_allclose(values[:, 5] + values[:, 6], values[:, 1], rtol=1e-7, atol=1e-7)
    manifest = path.parent.parent / "config" / "generation.json"
    baseline = float(json.loads(manifest.read_text()).get("baseline_flux", 1.0)) if manifest.is_file() else 1.0
    np.testing.assert_allclose(baseline + values[:, 1:noise_index + 1].sum(axis=1),
                               observed[:, 1], rtol=1e-7, atol=1e-7)
    return dict(path=str(path), names=names[1:noise_index], values=values[:, 1:noise_index],
                time=values[:, 0], total=baseline + values[:, 1:noise_index].sum(axis=1))


def interpolate(curve, times):
    if np.min(times) < curve[0, 0] or np.max(times) > curve[-1, 0]:
        raise ValueError("Native reconstruction does not span observations")
    return np.interp(times, curve[:, 0], curve[:, 1])


def plot_lightcurves(run, output):
    fig, axes = plt.subplots(5, 1, figsize=(10.0, 10.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, .85, 2.2, .85, 2.1]})
    metrics = {}
    truths = [truth_components(run, j) for j in range(2)]
    for j, (band, observed, model, truth) in enumerate(zip(run["bands"], run["observed"],
                                                         run["reconstruction"], truths)):
        ax, residual_ax = axes[2*j:2*j+2]
        ax.fill_between(model[:, 0], model[:, 1] - model[:, 2], model[:, 1] + model[:, 2],
                        color=COLORS["UNCERTAINTY"], alpha=.32, lw=0)
        ax.plot(model[:, 0], model[:, 1], color=COLORS["PRIMARY"], lw=1.7, label="Total reconstruction")
        observations(ax, *observed.T, label="Observed")
        mean_at_times = interpolate(model, observed[:, 0])
        residuals = (observed[:, 1] - mean_at_times) / observed[:, 2]
        rms = float(np.sqrt(np.mean(residuals**2)))
        row = dict(standardized_residual_rms=rms, observed_count=len(observed),
                   native_grid_points=len(model), truth_nrmse=None)
        if truth is not None:
            ax.plot(truth["time"], truth["total"], color=COLORS["REFERENCE"], lw=1.25,
                    ls=REFERENCE_DASH, label="Noiseless input")
            denominator = float(np.std(truth["total"]))
            if denominator > 0:
                row["truth_nrmse"] = float(np.sqrt(np.mean((mean_at_times-truth["total"])**2)) / denominator)
            row["truth_file"] = truth["path"]
        metrics[band] = row
        annotation = rf"$R_{{\rm res}}={rms:.2f}$"
        if row["truth_nrmse"] is not None:
            annotation += f"; NRMSE = {row['truth_nrmse']:.2f}"
        ax.text(.02, .93, band, ha="left", va="top", transform=ax.transAxes, fontsize=13)
        ax.set_ylabel("Flux", fontsize=12)
        ax.margins(y=.20)
        if j == 0:
            ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=3)
        residual_ax.axhspan(-1, 1, color=COLORS["GUIDE"], alpha=.32, zorder=0)
        residual_ax.axhline(0, color=COLORS["GUIDE"], lw=.9)
        for level in (-1, 1):
            residual_ax.axhline(level, color=COLORS["GUIDE"], lw=.6)
        residual_ax.plot(observed[:, 0], residuals, "o", color=COLORS["INK"], ms=2.3, mew=0)
        residual_ax.text(.98, .90, annotation, ha="right", va="top", transform=residual_ax.transAxes, fontsize=10)
        residual_ax.set_ylabel(r"$(f-\hat f)/\sigma$", fontsize=11)
        extent = max(3., float(np.max(np.abs(residuals))) * 1.4)
        residual_ax.set_ylim(-extent, extent)
    ax = axes[-1]
    observed = run["observed"][1]
    for j, (component, color) in enumerate(zip(run["components"], COMPONENT_COLORS)):
        offset = float(np.mean(interpolate(component, observed[:, 0])))
        flux = component[:, 1] - offset
        ax.fill_between(component[:, 0], flux-component[:, 2], flux+component[:, 2], color=color, alpha=.20, lw=0)
        ax.plot(component[:, 0], flux, color=color, lw=1.5, ls="-" if j == 0 else SHORT_DASH, label=f"Native {j}")
    # Physical truth has its own labels; it is not automatically identified
    # with C0 or C1. Three-path mixed-driver tables do not map onto two kernels.
    truth = truths[1]
    if truth is not None and truth["names"] == ["disk_variation", "gaussian_variation"]:
        for j in range(2):
            ax.plot(truth["time"], truth["values"][:, j], color=COLORS["REFERENCE"],
                    ls=REFERENCE_DASH, lw=1.25, marker="*" if j else None,
                    markevery=max(1, len(truth["time"]) // 18), ms=4.5,
                    label=("Injected disk", "Injected Gaussian")[j])
    ax.axhline(0, color=COLORS["GUIDE"], lw=.9, zorder=0)
    ax.text(.02, .94, f"{run['bands'][1]} components", transform=ax.transAxes, va="top", fontsize=12)
    ax.set_ylabel(r"$\Delta$ flux", fontsize=12)
    ax.set_xlabel("Time (days)", fontsize=13)
    ax.margins(y=.25)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    tmin = min(curve[0, 0] for curve in run["observed"])
    tmax = max(curve[-1, 0] for curve in run["observed"])
    for ax in axes:
        style_axis(ax, y_on_right=True)
        ax.set_xlim(tmin - .02*(tmax-tmin), tmax + .02*(tmax-tmin))
    fig.suptitle(rf"{run['kind'].capitalize()} — $\ln Z={run['lnz']:.2f}$", y=.98, fontsize=14)
    fig.subplots_adjust(left=.09, right=.90, top=.94, bottom=.07, hspace=.06)
    save_figure(fig, output)
    return metrics


def corner_columns(run):
    columns, labels, priors, summaries = [], [], [], {}
    physical = run["widths"] * (np.sqrt(2) if run["kind"] == "gamma" else 1.)
    for j in range(2):
        for symbol, values, suffix, is_log in (
            ("A", run["amp"][j], "amplitude", True),
            (r"\mu", run["centroids"][j], "centroid", False),
            (r"\sigma", physical[j], "sigma", True),
        ):
            key = f"{j}-th_component_{suffix}"
            low, high = run["bounds"][key]
            if is_log:
                offset = .5*np.log(2) if suffix == "sigma" and run["kind"] == "gamma" else 0.
                limits = [(low+offset)/np.log(10), (high+offset)/np.log(10)]
                label = rf"$\log_{{10}} {symbol}_{j}$"
            else:
                limits = [low, high]
                label = rf"${symbol}_{j}$ (d)"
            columns.append(np.log10(values) if is_log else values)
            labels.append(label)
            priors.append(limits)
            summaries[f"C{j}_{suffix}"] = dict(q15_85_q50_q84_15=quantiles(values),
                                                physical_bounds=(np.power(10., limits).tolist() if is_log else limits))
    for name, label in (("taud", r"$\log_{10}\tau_{\rm DRW}$"),
                         ("sigmad", r"$\log_{10}\hat\sigma_{\rm DRW}$")):
        values = run["samples"][:, run["names"].index(name)] / np.log(10)
        columns.append(values)
        labels.append(label)
        # The continuum empirical prior is joint and not a pair of fixed
        # physical bounds. Do not turn placeholder para_names bounds into it.
        priors.append(None)
        summaries[name] = dict(q15_85_q50_q84_15=quantiles(np.power(10., values)))
    return np.column_stack(columns), labels, priors, summaries


def binned_density_levels(counts, masses=CORNER_MASSES):
    """Highest-density thresholds of the existing unsmoothed count histogram.

    Bins have equal area in the displayed (including log10) coordinates.
    Ties may enclose more than the target mass; do not perturb data to split
    ties or fabricate distinct contours when both probabilities coincide.
    """
    ordered = np.sort(counts[counts > 0])[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    return np.unique([ordered[min(np.searchsorted(cumulative, p), len(ordered)-1)]
                      for p in masses])


def plot_corner(run, output):
    values, labels, priors, summaries = corner_columns(run)
    ndim = values.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=(14.6, 14.6))
    ranges = []
    for j in range(ndim):
        low, high = np.min(values[:, j]), np.max(values[:, j])
        pad = max(float(high-low) * .06, 1e-5)
        ranges.append((low-pad, high+pad))
    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            ax.set_xlim(*ranges[col])
            # Shared rectangular grid bounds below make every panel square.
            style_axis(ax, y_on_right=False, major_bins=None)
            ax.xaxis.set_major_locator(MaxNLocator(3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
            if row == col:
                q = np.quantile(values[:, col], QUANTILES)
                ax.axvspan(q[0], q[2], color=COLORS["UNCERTAINTY"], alpha=.70, lw=0, zorder=0)
                ax.hist(values[:, col], bins=30, range=ranges[col], histtype="step",
                        color=COLORS["INK"], linestyle="-", linewidth=.9)
                ax.axvline(q[1], color=COLORS["HIGHLIGHT"], ls="-", lw=.9)
                symbol = labels[col].split("$")[1]
                ax.set_title(rf"${symbol}={q[1]:.3g}_{{-{q[1]-q[0]:.2g}}}^{{+{q[2]-q[1]:.2g}}}$",
                             fontsize=10, pad=7)
                ax.set_ylim(bottom=0)
                # Keep count-axis ticks, but no probability/count numbers.
                ax.tick_params(axis="y", labelleft=False, labelright=False)
            else:
                counts, xedges, yedges = np.histogram2d(values[:, col], values[:, row],
                                                       bins=28, range=[ranges[col], ranges[row]])
                ax.pcolormesh(xedges, yedges, np.ma.masked_less(counts.T, 1), cmap="Greys",
                              norm=LogNorm(vmin=1, vmax=max(1.01, counts.max())),
                              rasterized=True, zorder=0)
                x = .5*(xedges[:-1]+xedges[1:])
                y = .5*(yedges[:-1]+yedges[1:])
                levels = binned_density_levels(counts)
                # nextafter permits a contour around a flat maximum without
                # shifting the probability thresholds by an arbitrary amount.
                ax.contour(x, y, counts.T, levels=np.nextafter(levels, 0),
                           colors=COLORS["INK"], linestyles="-",
                           linewidths=np.linspace(.8, 1.2, len(levels)))
                ax.set_ylim(*ranges[row])
            # Show prior edges on marginals only, separate from median marks.
            if row == col and priors[col] is not None:
                for boundary in priors[col]:
                    if ranges[col][0] <= boundary <= ranges[col][1]:
                        ax.axvline(boundary, color=COLORS["UNCERTAINTY"], ls=SHORT_DASH, lw=.9)
            ax.tick_params(which="major", labelsize=9.5)
            if row == ndim-1:
                ax.set_xlabel(labels[col], fontsize=11)
            else:
                ax.tick_params(labelbottom=False)
            if col == 0 and row > 0:
                ax.set_ylabel(labels[row], fontsize=11)
            elif row != col:
                ax.tick_params(labelleft=False)
    # Keep the upper triangle blank. Method/interval definitions and the
    # sample count belong in README and summary.json, not figure annotations.
    fig.subplots_adjust(left=.075, right=.98, top=.98, bottom=.075, hspace=.10, wspace=.10)
    save_figure(fig, output)
    return summaries


def plot_jobs(plan, config_path, *, plots=PLOTS, compare_input=False, output_subdir="results",
              component_sum_tolerance_sigma=0.0):
    config = load_config(Path(config_path).resolve())
    configure_style()
    for job in plan:
        run = load_run(job, component_sum_tolerance_sigma=component_sum_tolerance_sigma)
        output = run["case"] / output_subdir / run["kind"]
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summary.update(run=str(run["run"]), component_order=run["component_order"],
                       component_swapped_draws=run["component_swapped_draws"],
                       component_definition="C0 has smaller centroid; C1 has larger centroid in each joint draw",
                       lightcurve_component_order="native", posterior_draws=len(run["samples"]),
                       ln_evidence=run["lnz"], quantiles=list(QUANTILES),
                       tf_normalization="native MICA normalized-flux response; no per-component renormalization",
                       lightcurve_normalization="saved input flux units",
                       component_sum_check=run["component_sum_check"])
        reference = input_reference(config, run) if compare_input and "transfer" in plots else None
        if compare_input and "transfer" in plots and reference is None:
            print(f"{run['case'].name}: no comparable total observed-driver kernel; input overlay omitted", flush=True)
        if "transfer" in plots:
            summary["transfer"] = plot_transfer(run, output / "transfer_function", reference)
        if "lightcurves" in plots:
            summary["lightcurves"] = plot_lightcurves(run, output / "light_curves")
        if "corner" in plots:
            summary["parameters"] = plot_corner(run, output / "corner")
            summary["corner"] = dict(
                marginal_interval=[QUANTILES[0], QUANTILES[2]],
                contour_target_masses=list(CORNER_MASSES), bins_1d=30, bins_2d=28,
                density_coordinates="displayed coordinates; A, width and DRW scales in log10",
                contour_estimator="equal-area highest-count bins; no smoothing; ties retained; interpolated isolines approximate mass",
                style="image-creator with project Times New Roman and JPG/PDF overrides")
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"wrote {run['case'].name}/{run['kind']}: {', '.join(plots)} (JPG/PDF), summary.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite", choices=SUITES, default="all")
    parser.add_argument("--case", action="append", help="Select a configured case; repeat for several cases")
    parser.add_argument("--model", choices=("both", "gaussian", "gamma"), default="both")
    parser.add_argument("--plots", nargs="+", choices=PLOTS, default=list(PLOTS))
    parser.add_argument("--compare-input", action="store_true", help="Overlay an applicable observed-driver kernel/effective predictor")
    parser.add_argument("--output-subdir", default="results", help="Case-relative directory for generated figures.")
    parser.add_argument("--component-sum-tolerance-sigma", type=float, default=0.0,
                        help="Opt-in plot-only warning tolerance in total-reconstruction sigma; default strict.")
    args = parser.parse_args()
    if not 0 <= args.component_sum_tolerance_sigma <= 0.1:
        parser.error("component-sum-tolerance-sigma must be between 0 and 0.1")
    if Path(args.output_subdir).name != args.output_subdir or args.output_subdir in {".", ".."}:
        parser.error("--output-subdir must be a single directory name")
    plan = build_plan(args.config, suite=args.suite, cases=args.case, model=args.model)
    plot_jobs(plan, args.config, plots=args.plots, compare_input=args.compare_input,
              output_subdir=args.output_subdir,
              component_sum_tolerance_sigma=args.component_sum_tolerance_sigma)


if __name__ == "__main__":
    main()
