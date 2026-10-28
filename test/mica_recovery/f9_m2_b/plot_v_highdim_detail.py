#!/usr/bin/env python3
"""Describe the six-dimensional F9 V broad-prior posterior without refitting.

The three tau_s intervals used below are display slices, selected before this
script was written.  They are neither algorithmically discovered clusters nor
physical branches.  Every statistic uses the saved joint ``sample1d`` rows and
the final column of ``weights.txt``.  The two exchangeable Gamma tuples are
sorted together by their native centroid in every posterior row.

For the current MICA Gamma output the native location column is already the
first-moment centroid.  The Gamma onset is derived only when evaluating a
transfer function: onset = centroid - 2 * width.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

# The desktop sandbox does not provide a writable default Matplotlib cache.
# Keep its small cache under the explicitly requested output directory.
_MPL_CONFIG_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "highdim_broad_20260905"
    / ".mplconfig"
)
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.special import gammainc
import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RUN_DIR = (
    PROJECT_ROOT
    / "F9_a/runs/mica/gamma/broad_np4/2comp/run_W2_to_V_2comp_gamma"
)
DEFAULT_OUTPUT_DIR = HERE / "outputs" / "highdim_broad_20260905"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from collect_mica_results import (  # noqa: E402
    arr,
    component_parameter_arrays,
    effective_sample_size,
    normalized_weights,
    param_names,
    reorder_component_columns,
    weighted_quantile,
)


QUANTILES = np.array([0.15865, 0.5, 0.84135])
DISPLAY_QUANTILES = np.array([0.005, 0.995])
COORDINATE_LABELS = (
    r"$\tau_{\rm s}$ [d]",
    r"$\ln w_{\rm s}$",
    r"$\ln A_{\rm s}$",
    r"$\tau_{\rm l}$ [d]",
    r"$\ln w_{\rm l}$",
    r"$\ln A_{\rm l}$",
)
COORDINATE_KEYS = (
    "tau_s_days",
    "log_width_s",
    "log_area_s",
    "tau_l_days",
    "log_width_l",
    "log_area_l",
)
DESCRIPTIVE_INTERVALS = (
    ("tau_s_lt_1", r"$\tau_{\rm s}<1$ d", "#0072B2"),
    ("tau_s_1_to_2", r"$1\leq\tau_{\rm s}<2$ d", "#D55E00"),
    ("tau_s_ge_2", r"$\tau_{\rm s}\geq2$ d", "#009E73"),
)
CDF_WINDOWS = (
    ("near", -15.0, 15.0, 801),
    ("wide", -150.0, 200.0, 1401),
)


def q68_dict(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Weighted 16th, 50th, and 84th percentiles using the project helper."""
    quantile = weighted_quantile(values, weights, QUANTILES)
    return {
        "p16": float(quantile[0]),
        "p50": float(quantile[1]),
        "p84": float(quantile[2]),
    }


def weighted_quantile_axis0(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: np.ndarray = QUANTILES,
    *,
    chunk_columns: int = 96,
) -> np.ndarray:
    """Compute exact weighted quantiles for every column without resampling.

    ``values`` is processed in small lag-grid chunks so that the wide CDF
    panel never creates a large all-draw sorting array.  The interpolation is
    the same definition used by ``collect_mica_results.weighted_quantile``.
    """
    values = np.asarray(values, float)
    weights = normalized_weights(weights)
    if values.ndim != 2 or values.shape[0] != len(weights):
        raise ValueError("CDF values must be n_draw by n_lag and match weights")
    if not np.isfinite(values).all():
        raise ValueError("CDF values contain a non-finite entry")

    output = np.empty((len(quantiles), values.shape[1]), dtype=float)
    for first in range(0, values.shape[1], chunk_columns):
        last = min(first + chunk_columns, values.shape[1])
        block = values[:, first:last]
        order = np.argsort(block, axis=0, kind="mergesort")
        sorted_values = np.take_along_axis(block, order, axis=0)
        sorted_weights = weights[order]
        cdf = np.cumsum(sorted_weights, axis=0)
        cdf /= cdf[-1:, :]
        columns = np.arange(last - first)

        for iq, target in enumerate(quantiles):
            index = np.sum(cdf < target, axis=0)
            result = np.empty(last - first, dtype=float)
            at_first = index == 0
            result[at_first] = sorted_values[0, at_first]
            use_linear = ~at_first
            if np.any(use_linear):
                ii = index[use_linear]
                jj = columns[use_linear]
                c0 = cdf[ii - 1, jj]
                c1 = cdf[ii, jj]
                v0 = sorted_values[ii - 1, jj]
                v1 = sorted_values[ii, jj]
                fraction = np.divide(
                    target - c0,
                    c1 - c0,
                    out=np.zeros_like(c0),
                    where=(c1 > c0),
                )
                result[use_linear] = v0 + fraction * (v1 - v0)
            output[iq, first:last] = result
    return output


def load_joint_posterior() -> dict[str, Any]:
    """Load and moment-sort the two full Gamma tuples in the saved V run."""
    data_dir = RUN_DIR / "data"
    config = yaml.safe_load(
        (PROJECT_ROOT / "F9_a/config/source_config.yaml").read_text(encoding="utf-8")
    )
    fit = dict(config["mica_gamma2_broad"])
    if fit.get("component_order") != "moment1":
        raise RuntimeError("this diagnostic requires the configured moment1 ordering")

    sample = arr(data_dir / "sample1d.txt_2")
    raw_weights = arr(data_dir / "weights.txt")[:, -1]
    if len(sample) != len(raw_weights):
        raise RuntimeError("sample1d and weights have different numbers of rows")
    weights = normalized_weights(raw_weights)
    names = param_names(data_dir / "para_names_line.txt_2")
    required = {
        f"{component}-th_component_{field}"
        for component in (0, 1)
        for field in ("amplitude", "centroid", "sigma")
    }
    if not required.issubset(names):
        raise RuntimeError(
            "run-specific parameter file does not use the expected native Gamma centroid fields"
        )

    ordered, swapped = reorder_component_columns(sample, names, 2, "gamma", fit)
    onsets, widths, amplitudes, centroids = component_parameter_arrays(
        ordered, names, 2, "gamma"
    )
    if not np.all(centroids[0] <= centroids[1]):
        raise RuntimeError("moment1 tuple sorting failed")

    index = {name: names.index(name) for name in names}
    coordinates = np.column_stack(
        (
            centroids[0],
            ordered[:, index["0-th_component_sigma"]],
            ordered[:, index["0-th_component_amplitude"]],
            centroids[1],
            ordered[:, index["1-th_component_sigma"]],
            ordered[:, index["1-th_component_amplitude"]],
        )
    )
    if not np.isfinite(coordinates).all():
        raise RuntimeError("non-finite coordinate after tuple sorting")
    return {
        "fit": fit,
        "sample": ordered,
        "weights": weights,
        "raw_weight_sum": float(np.sum(raw_weights)),
        "names": names,
        "swapped": swapped,
        "centroids": centroids,
        "widths": widths,
        "amplitudes": amplitudes,
        "onsets": onsets,
        "coordinates": coordinates,
    }


def display_ranges(
    coordinates: np.ndarray, weights: np.ndarray
) -> list[list[float]]:
    """Weighted local display ranges; no posterior row is removed."""
    ranges: list[list[float]] = []
    for column in coordinates.T:
        lower, upper = weighted_quantile(column, weights, DISPLAY_QUANTILES)
        if not lower < upper:
            pad = max(1.0, abs(float(lower)) * 0.05)
            lower, upper = lower - pad, upper + pad
        pad = 0.03 * (upper - lower)
        ranges.append([float(lower - pad), float(upper + pad)])
    return ranges


def plot_corner(
    coordinates: np.ndarray,
    weights: np.ndarray,
    axis_ranges: list[list[float]],
    output_dir: Path,
) -> None:
    """Draw an all-weighted six-coordinate corner plot at local visual scale."""
    ncoord = coordinates.shape[1]
    figure, axes = plt.subplots(ncoord, ncoord, figsize=(14.2, 14.0))
    cmap = plt.get_cmap("Blues")

    for row in range(ncoord):
        for column in range(ncoord):
            axis = axes[row, column]
            if row < column:
                axis.axis("off")
                continue
            if row == column:
                axis.hist(
                    coordinates[:, column],
                    bins=46,
                    range=axis_ranges[column],
                    weights=weights,
                    color="#3B6EA5",
                    histtype="stepfilled",
                    alpha=0.72,
                )
                q16, q50, q84 = weighted_quantile(
                    coordinates[:, column], weights, QUANTILES
                )
                axis.axvline(q50, color="#172B4D", linewidth=1.05)
                axis.axvline(q16, color="#172B4D", linewidth=0.65, linestyle="--")
                axis.axvline(q84, color="#172B4D", linewidth=0.65, linestyle="--")
                axis.set_xlim(axis_ranges[column])
                axis.set_yticks([])
            else:
                density, xedge, yedge = np.histogram2d(
                    coordinates[:, column],
                    coordinates[:, row],
                    bins=46,
                    range=[axis_ranges[column], axis_ranges[row]],
                    weights=weights,
                )
                positive = density[density > 0.0]
                if len(positive):
                    axis.pcolormesh(
                        xedge,
                        yedge,
                        np.ma.masked_less_equal(density.T, 0.0),
                        cmap=cmap,
                        norm=LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive))),
                        shading="auto",
                    )
                axis.set_xlim(axis_ranges[column])
                axis.set_ylim(axis_ranges[row])

            axis.xaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
            if row != ncoord - 1:
                axis.set_xticklabels([])
            else:
                axis.set_xlabel(COORDINATE_LABELS[column], fontsize=10)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
            if column != 0 or row == column:
                axis.set_yticklabels([])
            else:
                axis.set_ylabel(COORDINATE_LABELS[row], fontsize=10)

    figure.suptitle(
        "F9 V broad-prior posterior: six original joint coordinates\n"
        "moment1 tuple sorting; native $\\tau$ is Gamma centroid, onset = $\\tau-2w$",
        fontsize=15,
        y=0.982,
    )
    figure.text(
        0.5,
        0.012,
        "All 10,000 weighted posterior rows enter every statistic. Axis limits are only weighted 0.5–99.5% display scaling; no posterior truncation.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.075, top=0.91, wspace=0.08, hspace=0.08)
    figure.savefig(output_dir / "V_corner.png", dpi=220)
    figure.savefig(output_dir / "V_corner.pdf")
    plt.close(figure)


def gamma_cdf(
    lag_days: np.ndarray,
    centroid_days: np.ndarray,
    width_days: np.ndarray,
) -> np.ndarray:
    """Unit-area k=2 Gamma CDF for each posterior draw and lag-grid point."""
    onset_days = centroid_days - 2.0 * width_days
    x = (lag_days[None, :] - onset_days[:, None]) / width_days[:, None]
    return gammainc(2.0, np.maximum(x, 0.0))


def total_cdf_quantiles(
    lag_days: np.ndarray,
    centroids: np.ndarray,
    widths: np.ndarray,
    amplitudes: np.ndarray,
    weights: np.ndarray,
    *,
    draw_chunk: int = 384,
) -> np.ndarray:
    """Quantiles of the per-draw unit-area total Gamma transfer CDF.

    Each draw is normalized by A_s+A_l before any posterior quantile is
    taken.  The finite display windows are not re-normalized, so a broad
    tail legitimately leaves the CDF below one at the right-hand edge.
    """
    output = np.empty((len(QUANTILES), len(lag_days)), dtype=float)
    # Build the complete matrix one draw block at a time.  It is bounded here
    # (at most 10k x 1401), and the following weighted sort is itself chunked.
    cdf = np.empty((amplitudes.shape[1], len(lag_days)), dtype=float)
    for first in range(0, len(cdf), draw_chunk):
        last = min(first + draw_chunk, len(cdf))
        area = amplitudes[:, first:last].sum(axis=0)
        if np.any(area <= 0.0):
            raise RuntimeError("non-positive total Gamma area")
        cdf_s = gamma_cdf(lag_days, centroids[0, first:last], widths[0, first:last])
        cdf_l = gamma_cdf(lag_days, centroids[1, first:last], widths[1, first:last])
        cdf[first:last] = (
            amplitudes[0, first:last, None] * cdf_s
            + amplitudes[1, first:last, None] * cdf_l
        ) / area[:, None]
    if np.min(cdf) < -1.0e-12 or np.max(cdf) > 1.0 + 1.0e-12:
        raise RuntimeError("Gamma CDF falls outside [0, 1]")
    output[:] = weighted_quantile_axis0(cdf, weights, QUANTILES)
    return output


def interval_masks(tau_short: np.ndarray) -> dict[str, np.ndarray]:
    """Return only the requested descriptive, exhaustive tau_s intervals."""
    masks = {
        "tau_s_lt_1": tau_short < 1.0,
        "tau_s_1_to_2": (tau_short >= 1.0) & (tau_short < 2.0),
        "tau_s_ge_2": tau_short >= 2.0,
    }
    if not np.all(np.sum(np.vstack(tuple(masks.values())), axis=0) == 1):
        raise RuntimeError("descriptive tau_s intervals are not an exhaustive partition")
    return masks


def group_summary(
    mask: np.ndarray,
    weights: np.ndarray,
    centroids: np.ndarray,
    widths: np.ndarray,
    amplitudes: np.ndarray,
    onsets: np.ndarray,
) -> dict[str, Any]:
    """Conditional summary with normalized weights inside one display slice."""
    conditional_weights = normalized_weights(weights[mask])
    fraction_short = amplitudes[0, mask] / amplitudes[:, mask].sum(axis=0)
    total_centroid = (
        amplitudes[:, mask] * centroids[:, mask]
    ).sum(axis=0) / amplitudes[:, mask].sum(axis=0)
    total_variance = (
        amplitudes[:, mask]
        * (2.0 * widths[:, mask] ** 2 + (centroids[:, mask] - total_centroid) ** 2)
    ).sum(axis=0) / amplitudes[:, mask].sum(axis=0)
    return {
        "posterior_mass": float(np.sum(weights[mask])),
        "raw_draw_count": int(np.sum(mask)),
        "conditional_effective_sample_size": float(effective_sample_size(conditional_weights)),
        "tau_s_days": q68_dict(centroids[0, mask], conditional_weights),
        "tau_l_days": q68_dict(centroids[1, mask], conditional_weights),
        "onset_s_days": q68_dict(onsets[0, mask], conditional_weights),
        "onset_l_days": q68_dict(onsets[1, mask], conditional_weights),
        "width_s_days": q68_dict(widths[0, mask], conditional_weights),
        "width_l_days": q68_dict(widths[1, mask], conditional_weights),
        "area_fraction_s": q68_dict(fraction_short, conditional_weights),
        "total_centroid_days": q68_dict(total_centroid, conditional_weights),
        "total_sd_days": q68_dict(np.sqrt(total_variance), conditional_weights),
    }


def descriptive_tf(
    posterior: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Plot conditional total-CDF summaries for the requested display slices."""
    weights = posterior["weights"]
    centroids = posterior["centroids"]
    widths = posterior["widths"]
    amplitudes = posterior["amplitudes"]
    onsets = posterior["onsets"]
    masks = interval_masks(centroids[0])

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    all_window_quantiles: dict[str, dict[str, Any]] = {}
    group_statistics: dict[str, Any] = {}

    for key, label, color in DESCRIPTIVE_INTERVALS:
        group_statistics[key] = {
            "description": label,
            "interpretation": "descriptive tau_s interval; not an algorithmic cluster or physical branch",
            **group_summary(masks[key], weights, centroids, widths, amplitudes, onsets),
        }

    for axis, (name, low, high, npoint) in zip(axes, CDF_WINDOWS):
        lag = np.linspace(low, high, npoint)
        per_window: dict[str, Any] = {
            "lag_window_days": [float(low), float(high)],
            "n_grid_points": int(npoint),
            "cdf_definition": "[A_s F_s(t)+A_l F_l(t)]/(A_s+A_l) for each draw before conditional weighted quantiles",
            "groups": {},
        }
        for key, label, color in DESCRIPTIVE_INTERVALS:
            mask = masks[key]
            conditional_weights = normalized_weights(weights[mask])
            quantile = total_cdf_quantiles(
                lag,
                centroids[:, mask],
                widths[:, mask],
                amplitudes[:, mask],
                conditional_weights,
            )
            axis.fill_between(lag, quantile[0], quantile[2], color=color, alpha=0.18)
            axis.plot(
                lag,
                quantile[1],
                color=color,
                linewidth=1.8,
                label=(
                    f"{label}; mass={group_statistics[key]['posterior_mass']:.1%}, "
                    f"ESS={group_statistics[key]['conditional_effective_sample_size']:.0f}"
                ),
            )
            report_lags = np.array([-15.0, -5.0, 0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 50.0, 100.0, 200.0])
            report_lags = report_lags[(report_lags >= low) & (report_lags <= high)]
            per_window["groups"][key] = {
                "cdf_q16_q50_q84_at_days": {
                    f"{day:g}": [
                        float(np.interp(day, lag, quantile[i])) for i in range(3)
                    ]
                    for day in report_lags
                }
            }
        axis.set_xlim(low, high)
        axis.set_ylim(-0.02, 1.02)
        axis.axhline(1.0, color="0.5", linewidth=0.65, linestyle=":")
        axis.set_xlabel("Lag relative to W2 [observer-frame day]")
        axis.grid(alpha=0.22)
        axis.set_title(f"{low:g} to {high:g} d window")
        all_window_quantiles[name] = per_window

    axes[0].set_ylabel("Cumulative total TF area")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    figure.suptitle(
        "F9 V broad-prior total Gamma transfer functions: descriptive $\\tau_{\\rm s}$ slices\n"
        "Each draw is unit-area normalized before weighted quantiles; slices are not clusters or physical branches",
        fontsize=13.5,
        y=0.98,
    )
    figure.text(
        0.5,
        0.012,
        "The finite x-windows are only views: a broad-tail CDF may remain below one at +200 d. Ribbons show weighted 16–84% intervals.",
        ha="center",
        va="bottom",
        fontsize=8.8,
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.12, top=0.84, wspace=0.08)
    figure.savefig(output_dir / "V_descriptive_tf.png", dpi=220)
    figure.savefig(output_dir / "V_descriptive_tf.pdf")
    plt.close(figure)
    return {"groups": group_statistics, "cdf_windows": all_window_quantiles}


def build_metadata(
    posterior: dict[str, Any], axis_ranges: list[list[float]], tf_summary: dict[str, Any]
) -> dict[str, Any]:
    """Create a compact, machine-readable record of the figure calculations."""
    coordinates = posterior["coordinates"]
    weights = posterior["weights"]
    return {
        "status": "descriptive_visualization_from_existing_posterior_no_mica_run",
        "run_dir": str(RUN_DIR),
        "input_files": {
            "sample": str(RUN_DIR / "data/sample1d.txt_2"),
            "weights": str(RUN_DIR / "data/weights.txt"),
            "parameter_names": str(RUN_DIR / "data/para_names_line.txt_2"),
        },
        "posterior_contract": {
            "n_raw_joint_draws": int(len(weights)),
            "raw_weight_last_column_sum": posterior["raw_weight_sum"],
            "normalized_weight_sum": float(np.sum(weights)),
            "effective_sample_size": float(effective_sample_size(weights)),
            "component_order": "moment1; complete (logA, centroid, logw) tuples swapped per draw",
            "swapped_posterior_mass": float(np.sum(weights[posterior["swapped"]])),
            "native_gamma_location": "centroid (first moment), not onset",
            "derived_onset": "onset = centroid - 2 * width",
            "cdf_normalization": "each draw uses [A_s F_s+A_l F_l]/(A_s+A_l) before weighted quantiles",
        },
        "corner_coordinates": {
            "order": list(COORDINATE_KEYS),
            "axis_labels": list(COORDINATE_LABELS),
            "display_ranges_days_or_log_units": {
                key: value for key, value in zip(COORDINATE_KEYS, axis_ranges)
            },
            "display_range_definition": "weighted 0.5% to 99.5% quantiles plus 3% padding; display only, no calculation-time posterior cut",
            "full_posterior_q16_q50_q84": {
                key: q68_dict(coordinates[:, index], weights)
                for index, key in enumerate(COORDINATE_KEYS)
            },
        },
        "descriptive_intervals": {
            "warning": "These are predeclared descriptive tau_s intervals, not algorithmic clusters and not physical branches.",
            "definitions": {
                "tau_s_lt_1": "tau_s < 1 day",
                "tau_s_1_to_2": "1 <= tau_s < 2 day",
                "tau_s_ge_2": "tau_s >= 2 day",
            },
            **tf_summary,
        },
        "files": {
            "corner_png": "V_corner.png",
            "corner_pdf": "V_corner.pdf",
            "descriptive_tf_png": "V_descriptive_tf.png",
            "descriptive_tf_pdf": "V_descriptive_tf.pdf",
            "statistics_json": "V_descriptive_tf.json",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for V_corner, V_descriptive_tf, and the JSON record.",
    )
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    posterior = load_joint_posterior()
    axis_ranges = display_ranges(posterior["coordinates"], posterior["weights"])
    plot_corner(posterior["coordinates"], posterior["weights"], axis_ranges, output_dir)
    tf_summary = descriptive_tf(posterior, output_dir)
    metadata = build_metadata(posterior, axis_ranges, tf_summary)
    (output_dir / "V_descriptive_tf.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "files": metadata["files"],
                "n_draws": metadata["posterior_contract"]["n_raw_joint_draws"],
                "ess": metadata["posterior_contract"]["effective_sample_size"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
