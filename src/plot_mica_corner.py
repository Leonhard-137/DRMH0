#!/usr/bin/env python3
"""Plot saved MICA joint posteriors in each case's result/corner directory.

Examples::

    python main.py NGC6814 corner --stage mica_gaussian2_free
    python src/plot_mica_corner.py Mrk142 --stage mica_gamma2_free100_host

Every response band gets a vector PDF and 300 dpi PNG. No JSON sidecar is
written. The plotted parameters
are ln(sigmad), ln(taud), followed by (ln(A_i), lag_i, ln(w_i)) for
every component, directly in native sampling coordinates. Historical center
columns remain centers; current centroid columns remain centroids.
All equal-weight posterior_sample1d draws are used, with the existing stage's
component-label rule; two-Gamma components are always sorted by centroid.
There are no posterior cuts, raw-sample weights, or new fits. Ranges include
every draw. The shared reader supports
both historical center columns and current centroid columns.

Marginal strips/titles use the native 15.85/50/84.15 percentiles (68.3% equal-tail
interval); the 25-bin marginal histograms are not smoothed. Every lower panel
shows all raw draws. Two-dimensional contours target 68.3% and 95.4% probability
on a 40-bin histogram with Gaussian smoothing (default sigma: one bin), not
1D sigma contours. Physical bandwidth is capped at one quarter of each raw
central interval. Saved marginal bounds and centroid ordering constrain the
density. Runs with fewer than 100 saved or distinct joint draws use scatter
only in the lower triangle; this is a display safeguard, not a convergence
acceptance threshold. Panels whose central interval spans fewer than three
bins also use scatter only: a full-range grid cannot resolve their core.
Repeated resampled draws are retained, not deduplicated.
No explanatory text is put inside the figure. Style follows the bundled
image-creator.mplstyle and corner recipe.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, NullFormatter, ScalarFormatter
from matplotlib.patches import Polygon
import numpy as np
from scipy.ndimage import gaussian_filter

from collect_mica_results import (
    configured_run_name, configured_stages, mica_run_root, model_label,
    ncomp_from_config, read_yaml, component_location_name,
)
from plot_mica_transfer import read_joint_parameters, transfer_stage_config


ROOT = Path(__file__).resolve().parents[1]
STYLE = Path(__file__).with_name("image-creator.mplstyle")
QUANTILES = (0.1585, 0.5, 0.8415)
CONTOUR_MASSES = (0.683, 0.954)
MIN_DENSITY_DRAWS = 100
PNG_DPI = 300
INK = "#222222"
GRAY = "#B8B8B8"
MEDIAN = "#E69F00"


def save_figure_atomic(fig, path: Path, **kwargs):
    """Render beside the destination and replace it only after a clean save."""
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fig.savefig(temporary, **kwargs)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sampling_bounds(run: Path, columns: list[str], relabel: bool):
    """Saved native-coordinate bounds; zero-width DRW placeholders are unknown."""
    by_name = {}
    ncomp = (len(columns) - 2) // 3
    for line in (run / "data" / f"para_names_line.txt_{ncomp}").read_text().splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0].startswith("#"):
            continue
        low, high = float(fields[3]), float(fields[4])
        if np.isfinite([low, high]).all() and low < high:
            by_name[fields[1]] = (low, high)
    bounds = [by_name.get(name) for name in columns]
    if relabel and len(columns) == 8:
        # Either native component can occupy either displayed label.
        for left, right in zip(range(2, 5), range(5, 8)):
            a, b = bounds[left], bounds[right]
            union = None if a is None or b is None else (min(a[0], b[0]), max(a[1], b[1]))
            bounds[left] = bounds[right] = union
    return bounds


def read_corner_samples(run: Path, fit: dict):
    """Keep DRW and component coordinates from the same complete joint rows."""
    fit = dict(fit)
    if model_label(fit.get("type_tf", "gaussian")) == "gamma" and ncomp_from_config(fit) == 2:
        fit["component_order"] = "moment1"
    native, native_names, kinds, nswapped = read_joint_parameters(run, fit)
    values = [native[:, native_names.index(name)] for name in ("sigmad", "taud")]
    symbols = [r"\ln\sigma_{\rm DRW}", r"\ln\tau_{\rm DRW}"]
    labels = [rf"${symbol}$" for symbol in symbols]
    names = ["ln_sigmad", "ln_taud"]
    native_columns = ["sigmad", "taud"]
    location_names = []
    for i in range(len(kinds)):
        location = component_location_name(native_names, i)
        location_names.append(location)
        is_centroid = location.endswith("_centroid")
        lag_symbol = rf"\tau_{{{i}}}" if is_centroid else rf"c_{{{i}}}"
        columns = (f"{i}-th_component_amplitude", location, f"{i}-th_component_sigma")
        native_columns.extend(columns)
        values.extend(native[:, native_names.index(name)] for name in columns)
        symbols.extend((rf"\ln A_{{{i}}}", lag_symbol, rf"\ln w_{{{i}}}"))
        labels.extend((rf"$\ln A_{{{i}}}$", rf"${lag_symbol}$ (day)", rf"$\ln w_{{{i}}}$"))
        names.extend((f"ln_area{i}", f"tau{i}" if is_centroid else f"center{i}", f"ln_width{i}"))
    samples = np.column_stack(values)
    ncomp = len(kinds)
    settings = {}
    for line in (run / "param/param_input").read_text().splitlines():
        fields = line.split("#", 1)[0].split()
        if len(fields) >= 2:
            settings[fields[0]] = fields[1]
    relabel = str(fit.get("component_order", "native")).strip().lower() == "moment1"
    ordered = relabel or (all(kind == "gaussian" for kind in kinds)
                          and settings.get("TypeLagPrior") == "0")
    ordered_pairs = [(3 + 3*i, 6 + 3*i) for i in range(ncomp - 1)
                     if ordered and location_names[i].endswith("_centroid")
                     and location_names[i + 1].endswith("_centroid")]
    info = {
        "posterior_file": str(run / "data" / f"posterior_sample1d.txt_{ncomp}"),
        "parameter_names_file": str(run / "data" / f"para_names_line.txt_{ncomp}"),
        "native_parameter_file": str(run / "param" / "param_input"),
        "nposterior": len(samples), "nunique_joint_draws": len(np.unique(samples, axis=0)),
        "component_order": fit.get("component_order", "native"),
        "nswapped": nswapped, "kernels": kinds, "parameter_names": names,
        "native_location_columns": location_names, "coordinates": "native_sampling",
        "prior_bounds": sampling_bounds(run, native_columns, relabel),
        "ordered_pairs": ordered_pairs,
    }
    return samples, symbols, labels, info


def sample_ranges(samples: np.ndarray, prior_bounds=None) -> list[tuple[float, float]]:
    """Include all draws, padding constant dimensions without adding noise."""
    limits = []
    if prior_bounds is None:
        prior_bounds = [None] * samples.shape[1]
    if len(prior_bounds) != samples.shape[1]:
        raise ValueError("prior bounds must match the plotted dimensions")
    for values, bound in zip(samples.T, prior_bounds):
        low, high = float(values.min()), float(values.max())
        padding = 0.025 * (high - low) if high > low else max(abs(low) * 0.025, 1e-6)
        lower, upper = low - padding, high + padding
        if bound is not None:
            if low < bound[0] or high > bound[1]:
                raise ValueError("posterior sample falls outside its saved prior bounds")
            lower, upper = max(lower, bound[0]), min(upper, bound[1])
        limits.append((lower, upper))
    return limits


def smooth_histogram(counts, sigma, reflect_edges, ordered_mask=None):
    """Smooth bin mass; reflect only at saved hard bounds, never at sample extrema."""
    density = np.asarray(counts, dtype=float).copy()
    pads = [int(np.ceil(4 * width)) for width in sigma]
    for axis, (pad, reflect) in enumerate(zip(pads, reflect_edges)):
        if pad == 0:
            continue
        widths = [(0, 0)] * density.ndim
        widths[axis] = (pad, pad)
        density = np.pad(density, widths, mode="symmetric")
        for enabled, section in zip(reflect, (slice(0, pad), slice(-pad, None))):
            if not enabled:
                index = [slice(None)] * density.ndim
                index[axis] = section
                density[tuple(index)] = 0.0
    density = gaussian_filter(density, sigma=sigma, mode="constant")
    density = density[tuple(slice(pad, -pad) if pad else slice(None) for pad in pads)]
    if ordered_mask is not None:
        density = np.where(ordered_mask, density, 0.0)
    if density.sum() > 0:
        density *= np.sum(counts) / density.sum()
    return density


def contour_thresholds(counts: np.ndarray) -> np.ndarray:
    """Highest-count bin thresholds for the named enclosed probability masses."""
    ordered = np.sort(counts[counts > 0.0])[::-1]
    if len(ordered) < 2:
        return np.array([])
    cumulative = np.cumsum(ordered) / ordered.sum()
    indices = np.searchsorted(cumulative, CONTOUR_MASSES)
    thresholds = np.unique(ordered[np.minimum(indices, len(ordered) - 1)])
    return thresholds[thresholds < ordered[0]]


def parameter_title(symbol: str, quantiles: np.ndarray) -> str:
    low, median, high = map(float, quantiles)
    extent = max(abs(low), abs(median), abs(high))
    exponent = int(np.floor(np.log10(extent))) if extent > 0.0 else 0
    exponent = exponent if exponent >= 3 or exponent <= -3 else 0
    scale = 10.0**exponent
    middle, minus, plus = median / scale, (median - low) / scale, (high - median) / scale
    error = max(minus, plus)
    decimals = max(0, min(5, 1 - int(np.floor(np.log10(error))))) if error > 0 else 2
    value = f"{middle:.{decimals}f}_{{-{minus:.{decimals}f}}}^{{+{plus:.{decimals}f}}}"
    if exponent:
        value = rf"({value})\,10^{{{exponent}}}"
    return rf"${symbol}={value}$"


def make_corner(samples: np.ndarray, symbols: list[str], labels: list[str], bins: int = 40,
                *, hist_bins: int = 25, smooth: float = 1.0, prior_bounds=None,
                ordered_pairs=()):
    """Smooth only the displayed 2D density; keep all draws and raw quantiles."""
    if samples.ndim != 2 or len(samples) < 1 or not np.isfinite(samples).all():
        raise ValueError("corner requires a nonempty finite joint posterior")
    ndim = samples.shape[1]
    if ndim != len(labels) or ndim != len(symbols) or bins < 2 or hist_bins < 2:
        raise ValueError("inconsistent corner labels or invalid histogram bins")
    if not np.isfinite(smooth) or smooth < 0:
        raise ValueError("smoothing must be finite and nonnegative")
    prior_bounds = [None] * ndim if prior_bounds is None else prior_bounds
    limits = sample_ranges(samples, prior_bounds)
    constant = np.ptp(samples, axis=0) == 0.0
    quantiles = np.quantile(samples, QUANTILES, axis=0)
    sparse = len(samples) < MIN_DENSITY_DRAWS or len(np.unique(samples, axis=0)) < MIN_DENSITY_DRAWS
    edges = [np.linspace(low, high, bins + 1) for low, high in limits]
    bin_widths = np.array([(high - low) / bins for low, high in limits])
    central_widths = quantiles[2] - quantiles[0]
    sigma_bins = np.minimum(smooth, 0.25 * central_widths / bin_widths)
    unresolved = (central_widths < 3 * bin_widths) & ~constant
    reflect_edges = [(bound is not None and low == bound[0],
                      bound is not None and high == bound[1])
                     for (low, high), bound in zip(limits, prior_bounds)]
    with plt.style.context(STYLE):
        side = 1.85 * ndim + 0.9
        fig = plt.figure(figsize=(side, side))
        grid = fig.add_gridspec(ndim, ndim, left=0.075, right=0.99,
                               bottom=0.075, top=0.96, wspace=0.06, hspace=0.075)
        for row in range(ndim):
            for column in range(row + 1):
                ax = fig.add_subplot(grid[row, column])
                ax.set_box_aspect(1)
                ax.set_xlim(*limits[column])
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
                for axis in (ax.xaxis, ax.yaxis):
                    formatter = ScalarFormatter(useOffset=False, useMathText=True)
                    formatter.set_powerlimits((-3, 3))
                    axis.set_major_formatter(formatter)
                    axis.set_minor_locator(AutoMinorLocator(4))
                    axis.set_minor_formatter(NullFormatter())
                outer_x, outer_y = row == ndim - 1, column == 0 and row > 0
                ax.tick_params(which="both", direction="in", top=True, right=True,
                               labelbottom=outer_x, labelleft=outer_y)
                ax.xaxis.offsetText.set_visible(outer_x)
                ax.yaxis.offsetText.set_visible(outer_y)
                if outer_x:
                    ax.set_xlabel(labels[column])
                if outer_y:
                    ax.set_ylabel(labels[row])

                if row == column:
                    low, median, high = quantiles[:, column]
                    ax.axvspan(low, high, color=GRAY, alpha=0.70, linewidth=0)
                    marginal_edges = np.linspace(*limits[column], hist_bins + 1)
                    ax.hist(samples[:, column], bins=marginal_edges, density=True,
                            histtype="step", color=INK, linestyle="-", linewidth=1.15)
                    ax.axvline(median, color=MEDIAN, linestyle="-", linewidth=0.9)
                    ax.set_ylim(bottom=0)
                    ax.yaxis.set_major_formatter(NullFormatter())
                    ax.yaxis.offsetText.set_visible(False)
                    ax.set_title(parameter_title(symbols[column], quantiles[:, column]),
                                 fontsize=10.5, pad=5)
                else:
                    ax.set_ylim(*limits[row])
                    scatter_only = (sparse or constant[column] or constant[row]
                                    or unresolved[column] or unresolved[row])
                    ax.scatter(samples[:, column], samples[:, row], s=6,
                               color=INK, alpha=0.25 if scatter_only else 0.08,
                               linewidths=0, zorder=2)
                    if not scatter_only:
                        counts, xedges, yedges = np.histogram2d(
                            samples[:, column], samples[:, row], bins=(edges[column], edges[row]))
                        x = (xedges[:-1] + xedges[1:]) * 0.5
                        y = (yedges[:-1] + yedges[1:]) * 0.5
                        ordered = (column, row) in ordered_pairs
                        allowed = x[:, None] <= y[None, :] if ordered else None
                        density = smooth_histogram(
                            counts, sigma_bins[[column, row]],
                            [reflect_edges[column], reflect_edges[row]], allowed)
                        mesh = ax.pcolormesh(
                            xedges, yedges, np.ma.masked_equal(density.T, 0), cmap="Greys",
                            vmin=0, shading="flat", alpha=0.35, zorder=1)
                        clip = None
                        if ordered:
                            low = min(xedges[0], yedges[0])
                            high = max(xedges[-1], yedges[-1])
                            clip = Polygon([(low, low), (low, high), (high, high)],
                                           closed=True, transform=ax.transData)
                            mesh.set_clip_path(clip)
                        levels = contour_thresholds(density)
                        if len(levels):
                            # Extend to the true edges without forcing density to zero
                            # at a prior wall (which would invent an interior peak).
                            xx, yy = np.r_[xedges[0], x, xedges[-1]], np.r_[yedges[0], y, yedges[-1]]
                            contours = ax.contour(
                                xx, yy, np.pad(density.T, 1, mode="edge"), levels=levels,
                                colors=INK, linestyles="-", zorder=3,
                                linewidths=np.linspace(0.8, 1.1, len(levels)))
                            if clip is not None:
                                contours.set_clip_path(clip)
        return fig, {"quantiles": quantiles.T.tolist(), "ranges": limits,
                     "constant_dimensions": np.flatnonzero(constant).tolist(),
                     "density_omitted_dimensions": np.flatnonzero(unresolved).tolist(),
                     "smoothing_sigma_bins": sigma_bins.tolist(),
                     "smoothing_bandwidths": (sigma_bins * bin_widths).tolist(),
                     "lower_triangle": "scatter_only" if sparse else "scatter_density_contours"}


def plot_source_stage(source_dir: Path, stage: str, *, bands: list[str] | None = None,
                      bins: int = 40, hist_bins: int = 25, smooth: float = 1.0):
    cfg = read_yaml(source_dir / "config/source_config.yaml")
    fit = transfer_stage_config(cfg, stage)
    source = str(cfg.get("source", source_dir.name))
    model, ncomp = model_label(fit.get("type_tf", "gaussian")), ncomp_from_config(fit)
    responses = list(fit.get("responses", []))
    if not responses or not fit.get("driver"):
        raise ValueError(f"{stage}: missing driver/responses")
    if bands:
        unknown = set(bands) - set(responses)
        if unknown:
            raise ValueError(f"unknown response bands: {sorted(unknown)}")
        responses = [band for band in responses if band in bands]
    if Path(source).name != source:
        raise ValueError("source must be a plain filename label")
    root = mica_run_root(source_dir, fit)
    # Validate every selected band before starting the stage's figures.
    inputs = [(band, read_corner_samples(root / configured_run_name(fit, band), fit))
              for band in responses]
    output = root / "result" / "corner"
    output.mkdir(parents=True, exist_ok=True)
    products = []
    for band, (samples, symbols, labels, info) in inputs:
        stem = output / f"{source}_{band}_{ncomp}comp_{model}_corner"
        fig, statistics = make_corner(samples, symbols, labels, bins, hist_bins=hist_bins,
                                      smooth=smooth, prior_bounds=info["prior_bounds"],
                                      ordered_pairs=info["ordered_pairs"])
        try:
            with plt.style.context(STYLE):
                # The fixed GridSpec margins already contain all labels.  Avoiding
                # bbox_inches="tight" prevents a second full-canvas render.  Keep
                # the PDF fully vector: Matplotlib's mixed vector/raster renderer
                # otherwise allocates a full-canvas buffer for many lower panels.
                # PNG remains a true 300 dpi export.
                save_figure_atomic(
                    fig, stem.with_suffix(".pdf"),
                    metadata={"Title": f"{source} {band}: {stage} posterior",
                              "Creator": "plot_mica_corner.py; image-creator"})
                save_figure_atomic(fig, stem.with_suffix(".png"), dpi=PNG_DPI)
        finally:
            plt.close(fig)
        record = {"source": source, "stage": stage, "band": band, "driver": fit["driver"],
                  "model": model, **info, **statistics,
                  "marginal_quantiles": QUANTILES, "contour_probability_masses": CONTOUR_MASSES,
                  "bins": bins, "hist_bins": hist_bins, "smoothing": smooth,
                  "weights": "equal", "selection": "all",
                  "parameter_definitions": {"tau": "physical centroid, observer-frame days",
                                            "center": "native kernel center/onset, observer-frame days",
                                            "ln_sigmad": "natural log of native MICA DRW amplitude parameter",
                                            "ln_taud": "natural log of DRW timescale in observer-frame days",
                                            "ln_width": "natural log of kernel scale in observer-frame days",
                                            "ln_area": "natural log of integrated response, native MICA normalization"},
                  "pdf": str(stem.with_suffix(".pdf")), "png": str(stem.with_suffix(".png"))}
        products.append(record)
        detail = " (sparse posterior: scatter only)" if statistics["lower_triangle"] == "scatter_only" else ""
        print(f"[ok] {source}/{stage}/{band}: {len(samples)} posterior draws{detail} -> {stem.name}", flush=True)
    return products


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="source directory name(s) or paths")
    parser.add_argument("--stage", action="append", help="repeat to select cases; default: configured mica_stages")
    parser.add_argument("--band", action="append", help="optional response band(s); default: every configured band")
    parser.add_argument("--bins", type=int, default=40, help="histogram bins per axis (default: 40)")
    parser.add_argument("--hist-bins", type=int, default=25, help="unsmoothed marginal bins (default: 25)")
    parser.add_argument("--smooth", type=float, default=1.0,
                        help="2D Gaussian sigma in bins, capped by core width; 0 disables (default: 1)")
    args = parser.parse_args(argv)
    failed = False
    for name in args.sources:
        source_dir = (ROOT / Path(name).expanduser()).resolve()
        try:
            cfg = read_yaml(source_dir / "config/source_config.yaml")
            stages = list(dict.fromkeys(args.stage)) if args.stage else configured_stages(cfg)
        except (OSError, ValueError) as error:
            print(f"[error] {name}: {error}", file=sys.stderr)
            failed = True
            continue
        for stage in stages:
            try:
                plot_source_stage(source_dir, stage, bands=args.band, bins=args.bins,
                                  hist_bins=args.hist_bins, smooth=args.smooth)
            except (OSError, ValueError, KeyError, FloatingPointError) as error:
                print(f"[error] {name}/{stage}: {error}", file=sys.stderr)
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
