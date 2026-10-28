"""Render the ten-source temperature-map atlas from completed fit products.

This script does not prepare observations, import a fitter, or modify inputs.
All ten objects, in the manuscript sample-table order, must be present. Optional
band-subset tests are omitted explicitly, never replaced with a baseline map.
Run with the project's astro Python and an output-scoped MPLCONFIGDIR.
"""
from __future__ import annotations

import argparse
import csv
import gc
from dataclasses import dataclass
import io
import json
from pathlib import Path
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, NullLocator
from PIL import Image


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
SAMPLE = PROJECT / "DRMH0/tables/sample_observations.csv"
XI = (1, 10, 100, 1000)
WIDTHS = (7, 28, 56)
INK = "#222222"
ERROR = "#B8B8B8"
GUIDE = "#D9D9D9"
# Explicit paper-style overrides of the skill's generic model role colors.
FIT_RED = "#E53935"
FIT_CYAN = "#00A9C9"
CMAP = plt.get_cmap("coolwarm", 20)
TIME_LABEL = r"$t-t_{\min}$ [observer days]"
RADIUS_LABEL = r"$\log_{10}(R/R_{\rm in})$"
NORMAL_LABEL = r"$(\delta T/T)/q_{99,\,\xi=10}$"
DISPLAY = {
    "Mrk817_a": "Mrk 817 (a)", "Mrk142": "Mrk 142",
    "Mrk335": "Mrk 335", "F9_a": "Fairall 9 (a)",
    "Mrk279_b": "Mrk 279 (b)", "Mrk509": "Mrk 509",
    "NGC5548": "NGC 5548", "NGC4151_a": "NGC 4151 (a)",
    "NGC6814": "NGC 6814", "NGC2617_a": "NGC 2617 (a)",
}


@dataclass
class Source:
    name: str
    data: dict
    metrics: object
    summary: dict

    @property
    def title(self):
        return DISPLAY.get(self.name, self.name)

    @property
    def scale(self):
        return max(float(np.quantile(np.abs(self.data["q_xi10"]), .99)), 1e-15)


def sample_order(path=SAMPLE):
    # Some project exports contain literal escaped CRLF characters. Normalize
    # those separators in memory without touching the manuscript's source file.
    content = path.read_text(encoding="utf-8-sig")
    if "\\r\\n" in content and content.count("\n") < 3:
        content = content.replace("\\r\\n", "\n")
    rows = list(csv.DictReader(io.StringIO(content)))
    names = [row["source"].strip() for row in rows]
    if len(names) != 10 or len(set(names)) != 10:
        raise ValueError(f"Expected exactly ten unique sample sources: {names}")
    return names


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def finite_array(data, key, shape=None):
    a = np.asarray(data[key])
    if shape is not None and a.shape != shape:
        raise ValueError(f"{key}: shape {a.shape}, expected {shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{key}: non-finite values")
    return a


def validate_source(source):
    z = source.data
    nd, nb, nr = len(z["t"]), len(z["bands"]), len(z["u"])
    if not nd or nb < 2 or nr < 2:
        raise ValueError(f"{source.name}: empty or degenerate input")
    for key in ("t", "flux", "error", "bi"):
        finite_array(z, key, (nd,))
    if np.any(z["error"] <= 0):
        raise ValueError(f"{source.name}: nonpositive observational errors")
    if not np.issubdtype(z["bi"].dtype, np.integer):
        raise ValueError(f"{source.name}: noninteger band indices")
    if set(z["bi"].tolist()) != set(range(nb)):
        raise ValueError(f"{source.name}: band indices do not cover band labels")
    for key, shape in (("wavelengths", (nb,)), ("u", (nr,)),
                       ("edges", (nr + 1,)), ("temp", (nr,)),
                       ("kernel", (nb, nr)), ("tp", (250,))):
        finite_array(z, key, shape)
    for key in ("wavelengths", "u", "edges", "temp"):
        if np.any(z[key] <= 0):
            raise ValueError(f"{source.name}: nonpositive {key}")
    for key in ("tp", "u", "edges"):
        if np.any(np.diff(z[key]) <= 0):
            raise ValueError(f"{source.name}: {key} must increase strictly")
    if not (np.isclose(z["t"].min(), 0, atol=1e-7) and
            np.allclose(z["tp"][[0, -1]], [z["t"].min(), z["t"].max()])):
        raise ValueError(f"{source.name}: observer-time origin or bounds mismatch")
    for xi in XI:
        finite_array(z, f"q_xi{xi}", (nr, 250))
        finite_array(z, f"pred_xi{xi}", (nd,))
    for mode in ("low", "high"):
        for width in WIDTHS:
            tag = f"{mode}{width}"
            finite_array(z, f"q_{tag}", (nr, 250))
            finite_array(z, f"flux_{tag}", (nd,))
            finite_array(z, f"pred_{tag}", (nd,))
    for nt in (100, 500):
        finite_array(z, f"tp_nt{nt}", (nt,))
        finite_array(z, f"q_nt{nt}", (nr, nt))
        if np.any(np.diff(z[f"tp_nt{nt}"]) <= 0):
            raise ValueError(f"{source.name}: invalid Nt={nt} time grid")
    for key in ("q_without_U", "q_swift6"):
        if key in z:
            finite_array(z, key, (nr, 250))
    if source.summary.get("source", source.name) != source.name:
        raise ValueError(f"{source.name}: summary source mismatch")
    for key, expected in (("n_data", nd), ("n_bands", nb)):
        if key in source.summary and source.summary[key] != expected:
            raise ValueError(f"{source.name}: {key} mismatch in summary")
    return source


def load_sources():
    names = sample_order()
    required = [ROOT / "derived/input_manifest.json", ROOT / "physical_parameters.json"]
    required += [OUT / source / name for source in names
                 for name in ("fits.npz", "metrics.json", "summary.json")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Atlas requires all ten sources; missing:\n" + "\n".join(missing))
    # Validate that provenance products are readable, but never substitute their
    # possibly pre-trim counts for the actual fit arrays used in the figures.
    read_json(required[0])
    read_json(required[1])
    result = []
    for name in names:
        folder = OUT / name
        with np.load(folder / "fits.npz", allow_pickle=False) as loaded:
            z = {key: loaded[key] for key in loaded.files}
        result.append(validate_source(Source(name, z, read_json(folder / "metrics.json"),
                                             read_json(folder / "summary.json"))))
    return result


def configure_axes(ax, categorical_y=False):
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(NullLocator() if categorical_y else AutoMinorLocator(4))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(False)


def time_edges(tp):
    tp = np.asarray(tp)
    return np.r_[tp[0], (tp[:-1] + tp[1:]) / 2, tp[-1]]


def map_axis(ax, source, key="q_xi10", *, absolute=False, tp=None):
    z = source.data
    tp = z["tp"] if tp is None else tp
    divisor = 1.0 if absolute else source.scale
    limit = .2 if absolute else 1.0
    mesh = ax.pcolormesh(time_edges(tp), np.log10(z["edges"]), z[key] / divisor,
                         shading="flat", cmap=CMAP, vmin=-limit, vmax=limit,
                         rasterized=True)
    ax.set_xlim(tp[0], tp[-1])
    ax.set_ylim(np.log10(z["edges"][[0, -1]]))
    ax.xaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True, min_n_ticks=3))
    configure_axes(ax)
    return mesh


class Atlas:
    def __init__(self, pdf=None):
        self.entries = []

    def save(self, fig, name, caption):
        target = FIG / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # Check all visible text against the actual tight-export bounding box,
        # including axis/colorbar labels and titles. Visual inspection remains
        # necessary for overlaps, data legibility and ambiguous visual meaning.
        tight = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
        pad = .06 * fig.dpi
        # MaxNLocator creates candidate labels beyond the view interval. Their
        # Text.visible flag can remain true although Axis.draw never draws them.
        # Use the same tick selection as Axis.draw; do not weaken real clipping
        # checks or expand the scientific axis limits to include these labels.
        undrawn_tick_labels = set()
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                drawn = {id(tick) for tick in axis._update_ticks()} if ax.axison and axis.get_visible() else set()
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    if id(tick) not in drawn:
                        undrawn_tick_labels.update((id(tick.label1), id(tick.label2)))
        outside = []
        for item in fig.findobj(matplotlib.text.Text):
            if not item.get_visible() or not item.get_text() or id(item) in undrawn_tick_labels:
                continue
            b = item.get_window_extent(renderer)
            if b.width and b.height and (b.x0 < tight.x0-pad or b.x1 > tight.x1+pad or
                                        b.y0 < tight.y0-pad or b.y1 > tight.y1+pad):
                outside.append(item.get_text())
        if outside:
            raise RuntimeError(f"{name}: text outside export bounds: {outside}")
        for suffix, dpi in ((".pdf", 300), (".png", 300), ("_preview.jpg", 110)):
            fig.savefig(str(target) + suffix, dpi=dpi, bbox_inches="tight", pad_inches=.06)
        with Image.open(str(target) + ".png") as png:
            png_size, png_dpi = list(png.size), list(png.info.get("dpi", []))
        with Image.open(str(target) + "_preview.jpg") as jpg:
            preview_size = list(jpg.size)
        self.entries.append(dict(page=len(self.entries)+1, name=name, caption=caption,
                                 png_dimensions=png_size, png_dpi=png_dpi,
                                 preview_dimensions=preview_size,
                                 ignored_undrawn_tick_labels=len(undrawn_tick_labels),
                                 text_outside_export_bounds=outside))
        plt.close(fig)
        # Release cyclic artist/renderer references before the next page.
        fig.clear()
        gc.collect()
        (OUT / "render_progress.json").write_text(json.dumps(dict(
            status="rendering", pages_completed=len(self.entries),
            last_page=name, figures=self.entries), indent=2), encoding="utf-8")
        print(f"[{len(self.entries):02d}] {name}: PNG {png_size[0]}x{png_size[1]}", flush=True)

    def merge(self):
        # Merge finalized PDFs, avoiding PdfPages retaining all raster buffers.
        staging = OUT / "temperature_mapping_atlas.building.pdf"
        writer = PdfWriter()
        for entry in self.entries:
            page = PdfReader(str(FIG / (entry["name"] + ".pdf")))
            if len(page.pages) != 1:
                raise ValueError(f"Expected one page: {entry['name']}")
            writer.append(page)
        writer.add_metadata({"/Title": "Ten-source AGN temperature mapping - diagnostic atlas",
            "/Subject": "Regularization, time filtering, light curves and coverage",
            "/Creator": "sample10/plot_results.py"})
        with staging.open("wb") as handle:
            writer.write(handle)
        writer.close()
        if len(PdfReader(str(staging)).pages) != len(self.entries):
            raise ValueError("Merged atlas page count mismatch")
        staging.replace(OUT / "temperature_mapping_atlas.pdf")
        (OUT / "render_progress.json").write_text(json.dumps(dict(
            status="complete", pages_completed=len(self.entries),
            figures=self.entries), indent=2), encoding="utf-8")


def common_colorbar(fig, mesh, label=NORMAL_LABEL, *, overview=False):
    cax = fig.add_axes([.922, .14, .016, .72] if overview else [.933, .14, .014, .72])
    bar = fig.colorbar(mesh, cax=cax, extend="both")
    bar.set_label(label, fontsize=11)
    bar.ax.tick_params(labelsize=9)
    return bar


def baseline_overviews(sources, atlas):
    for absolute, name in ((True, "01_baseline_absolute"), (False, "02_baseline_normalized")):
        fig, axes = plt.subplots(5, 2, figsize=(12, 13))
        for i, (ax, source) in enumerate(zip(axes.flat, sources)):
            mesh = map_axis(ax, source, absolute=absolute)
            ax.set_title(source.title, loc="left")
            if i % 2 == 0:
                ax.set_ylabel(RADIUS_LABEL)
        fig.subplots_adjust(left=.077, right=.89, bottom=.065, top=.955, wspace=.18, hspace=.40)
        fig.supxlabel(TIME_LABEL, y=.023)
        fig.suptitle(r"Ten-source temperature mapping: $N_t=250,\ \xi=10$", y=.99)
        common_colorbar(fig, mesh, r"$\delta T/T$" if absolute else NORMAL_LABEL, overview=True)
        caption = ("Baseline temperature maps in the exact manuscript sample order. "
                   "Each panel uses its own full observed time span; horizontal widths do not represent equal durations. ")
        caption += ("A single absolute color scale, deltaT/T = [-0.2, 0.2], applies to all sources. "
                    "Values outside this range saturate, not disappear; saturated cell fractions are listed below. "
                    if absolute else
                    "Each source is divided by its own baseline 99th percentile of |deltaT/T|. "
                    "The common displayed range [-1,1] is dimensionless and does not imply equal physical amplitudes. ")
        atlas.save(fig, name, caption)


def map_comparison_pages(sources, atlas, kind):
    if kind == "regularization":
        keys = [f"q_xi{x}" for x in XI]
        titles = [rf"$\xi={x}$" for x in XI]
        prefix = "03_regularization"
        explanation = "Four regularization strengths; all cells in a source row use that source's xi=10 baseline q99, not per-panel renormalization."
    elif kind == "filter28":
        keys = ["q_xi10", "q_low28", "q_high28"]
        titles = ["Original", "Low-pass: 28 d", "High-pass: 28 d"]
        prefix = "04_filter28"
        explanation = "Original, 28-day-FWHM low-pass and corresponding high-pass reconstructions. A source row shares its original xi=10 q99 scale."
    elif kind == "timegrid":
        keys = ["q_nt100", "q_xi10", "q_nt500"]
        titles = [r"$N_t=100$", r"$N_t=250$", r"$N_t=500$"]
        prefix = "05_time_grid"
        explanation = "Nt=100/250/500 maps, displayed on their actual native time grids and the common source-baseline q99 scale. A fixed xi across grid sizes is not a fixed continuum regularization strength; this is sensitivity, not a convergence demonstration."
    elif kind == "subsets":
        keys = ["q_xi10", "q_without_U", "q_swift6"]
        titles = ["All input bands", "Without U/u", "Swift six bands"]
        prefix = "06_band_subsets"
        explanation = "Available band-subset sensitivity maps at the baseline color scale. Blank optional panels denote an unavailable test and do not denote zero signal. Parameter and subset definitions are inherited from the fit summary, not inferred from morphology."
    else:
        raise ValueError(kind)
    for block in range(2):
        subset = sources[block*5:(block+1)*5]
        fig, axes = plt.subplots(5, len(keys), figsize=(15 if len(keys)==4 else 13.5, 12), squeeze=False)
        missing = []
        for row, source in enumerate(subset):
            for col, key in enumerate(keys):
                ax = axes[row, col]
                if key not in source.data:
                    ax.axis("off")
                    if row == 0:
                        ax.set_title(titles[col])
                    missing.append(f"{source.name}/{key}")
                    continue
                tp = source.data.get("tp_" + key[2:]) if kind == "timegrid" and key != "q_xi10" else None
                mesh = map_axis(ax, source, key, tp=tp)
                if row == 0:
                    ax.set_title(titles[col])
                if col == 0:
                    ax.set_ylabel(source.title + "\n" + RADIUS_LABEL, fontsize=10)
                else:
                    ax.tick_params(labelleft=False)
        fig.subplots_adjust(left=.088, right=.91, bottom=.067, top=.945, wspace=.06, hspace=.27)
        fig.supxlabel(TIME_LABEL, y=.024)
        common_colorbar(fig, mesh)
        caption = explanation + (" Unavailable: " + ", ".join(missing) + "." if missing else " All requested panels are available.")
        atlas.save(fig, f"{prefix}_{block+1}", caption)


def filter_page(source, atlas):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharex=True, sharey=True)
    for row, mode in enumerate(("low", "high")):
        for col, width in enumerate(WIDTHS):
            mesh = map_axis(axes[row, col], source, f"q_{mode}{width}")
            axes[row, col].set_title(f"{'Low' if row == 0 else 'High'}-pass: {width} d")
        axes[row, 0].set_ylabel(RADIUS_LABEL)
    fig.subplots_adjust(left=.07, right=.91, bottom=.115, top=.87, wspace=.05, hspace=.20)
    fig.supxlabel(TIME_LABEL, y=.028)
    fig.suptitle(source.title, y=.965)
    common_colorbar(fig, mesh)
    atlas.save(fig, f"{source.name}/01_low_high_pass",
               f"{source.title}: Gaussian low-pass (top) and original-minus-low-pass (bottom), FWHM 7/28/56 observer days. "
               f"All six panels use the original xi=10 q99={source.scale:.6g}. "
               "Errors and smoothing-induced covariance are not propagated into a new independent-noise likelihood. "
               "Filtering across gaps and constant endpoint extension can influence structures near boundaries or long gaps. "
               "A filter width comparable to the campaign duration does not provide multiple independently resolved cycles.")


def band_grid(source):
    nb = len(source.data["bands"])
    cols = 3 if nb <= 9 else 4
    rows = int(np.ceil(nb / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, max(4.7, 2.0*rows+1.5)),
                             sharex=True, squeeze=False)
    for ax in axes.flat[nb:]:
        ax.axis("off")
    fig.subplots_adjust(left=.074, right=.984, bottom=.125 if rows<3 else .09,
                        top=.84 if rows<3 else .9, wspace=.25, hspace=.38)
    return fig, axes


def band_indices(z, b):
    idx = np.flatnonzero(z["bi"] == b)
    return idx[np.argsort(z["t"][idx], kind="stable")]


def observed(ax, t, y, err):
    ax.errorbar(t, y, yerr=err, fmt="none", ecolor=ERROR,
                elinewidth=.55, alpha=.55, capsize=0, zorder=1)
    ax.plot(t, y, linestyle="none", marker="o", color=INK,
            markersize=2.1, markeredgewidth=0, zorder=3)


def lightcurve_page(source, atlas):
    z = source.data
    fig, axes = band_grid(source)
    for b, band in enumerate(z["bands"]):
        ax = axes.flat[b]
        idx = band_indices(z, b)
        t, f, e = z["t"][idx], z["flux"][idx], z["error"][idx]
        shift = (np.max(f) + np.min(f)) / 2
        scale = max(float(np.ptp(f)/2), float(np.median(e)), np.finfo(float).eps)
        observed(ax, t, (f-shift)/scale, e/scale)
        ax.plot(t, (z["pred_xi1"][idx]-shift)/scale, color=FIT_RED, lw=.9, zorder=2,
                label=r"$\xi=1$")
        ax.plot(t, (z["pred_xi1000"][idx]-shift)/scale, color=FIT_CYAN, lw=.95, zorder=2,
                label=r"$\xi=1000$")
        ax.set_title(str(band), loc="left", fontsize=10)
        ax.set_xlim(z["t"].min(), z["t"].max())
        ax.xaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
        configure_axes(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.985,.995), ncol=2)
    fig.suptitle(source.title + " — observed and reconstructed light curves", y=.99)
    fig.supxlabel(TIME_LABEL, y=.025)
    fig.supylabel("Normalized observed flux", x=.014)
    atlas.save(fig, f"{source.name}/02_lightcurve_fits",
               f"{source.title}: every fitted band, black observed points with gray capless input-error bars; "
               "red xi=1 and cyan xi=1000 model predictions at the actual observation times. "
               "Lines connect these predictions only; no continuous between-observation prediction was recomputed. "
               "One affine transform per band is applied identically to data, errors and both models: "
               "subtract the data midrange and divide by max(data half-range, median input error). "
               "No multiplicative refit, outlier clipping or fixed vertical clipping is applied.")


def residual_page(source, atlas):
    z = source.data
    fig, axes = band_grid(source)
    for b, band in enumerate(z["bands"]):
        ax = axes.flat[b]
        idx = band_indices(z, b)
        r = (z["flux"][idx] - z["pred_xi10"][idx]) / z["error"][idx]
        observed(ax, z["t"][idx], r, np.ones(len(idx)))
        ax.axhline(0, color=GUIDE, lw=.8, zorder=0)
        ax.set_title(str(band), loc="left", fontsize=10)
        ax.set_xlim(z["t"].min(), z["t"].max())
        ax.xaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
        configure_axes(ax)
    fig.suptitle(source.title + r" — residuals ($\xi=10$)", y=.99)
    fig.supxlabel(TIME_LABEL, y=.025)
    fig.supylabel(r"$(F_{\rm obs}-F_{\rm model})/\sigma_{\rm input}$", x=.014)
    atlas.save(fig, f"{source.name}/03_residuals",
               f"{source.title}: baseline residuals divided by the input per-point errors used in the fit. "
               "The gray error bars have unit length in these units; they are not the full posterior residual covariance. "
               "Every residual and error bar is retained by autoscaling; different bands may have different vertical ranges. "
               "Neither residual coherence nor a U/u residual identifies a BLR or disk component by itself.")


def coverage_page(source, atlas):
    z = source.data
    nb = len(z["bands"])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, max(5.1, .205*nb+2.2)),
                                gridspec_kw={"width_ratios": [1.45, 1]})
    wave = z["wavelengths"]
    norm = LogNorm(vmin=float(np.min(wave)), vmax=float(np.max(wave)))
    colors = plt.get_cmap("viridis")
    for b, band in enumerate(z["bands"]):
        idx = band_indices(z, b)
        color = colors(norm(wave[b]))
        ax.plot(z["t"][idx], np.full(len(idx), b), linestyle="none", marker="|",
                markersize=4, markeredgewidth=.8, color=color)
        kernel = z["kernel"][b]
        peak = np.max(np.abs(kernel))
        if peak > 0:
            bx.plot(np.log10(z["u"]), kernel/peak, color=color, lw=1)
    ax.set(yticks=np.arange(nb), yticklabels=[str(b) for b in z["bands"]],
           xlabel=TIME_LABEL, title="Observed sampling", xlim=(0, z["t"].max()),
           ylim=(nb-.5, -.5))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    configure_axes(ax, categorical_y=True)
    bx.set(xlabel=RADIUS_LABEL, ylabel="Kernel / peak", title="Radial response kernels",
           xlim=np.log10(z["edges"][[0,-1]]))
    bx.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    configure_axes(bx)
    fig.suptitle(source.title, y=.98)
    fig.subplots_adjust(left=.105, right=.89, bottom=.14, top=.86, wspace=.3)
    cax = fig.add_axes([.917,.20,.015,.57])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colors), cax=cax)
    cb.set_label(r"Observed wavelength [$\AA$]")
    cb.ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.geomspace(wave.min(),wave.max(),4)))
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))
    cb.ax.yaxis.set_minor_locator(NullLocator())
    atlas.save(fig, f"{source.name}/04_coverage_kernels",
               f"{source.title}: actual per-band observation times (no inferred continuous coverage) and "
               "the fit's radial response kernels, each divided by its own peak. Colors encode observed wavelength, "
               "not physical component identity. Kernel overlap means that the radial grid cells are not independently "
               "resolved. The full-span maps include extrapolation/interpolation and regularization-dominated regions "
               "where simultaneous multiband coverage is absent.")


def markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_captions(sources, atlas, elapsed):
    # Main fit summaries can acquire provenance clarifications after rendering.
    # Refresh captions independently; the frozen numerical arrays need no redraw.
    for entry in atlas.entries:
        if entry["name"].startswith("06_band_subsets_"):
            block = int(entry["name"].rsplit("_", 1)[1]) - 1
            repeated = [s.name for s in sources[block*5:(block+1)*5]
                        if s.summary.get("subset_tests", {}).get(
                            "swift6", {}).get("status", "").lower().startswith("identical to primary")]
            marker = " Primary Swift-only repeats:"
            entry["caption"] = entry["caption"].split(marker)[0]
            if repeated:
                entry["caption"] += (marker + " " + ", ".join(repeated) + ". "
                    "For these sources the primary data already are the same six Swift bands: "
                    "the Swift-six map is a copy of the baseline, not a separate refit or experiment. "
                    "A map correlation of one is tautological and is not evidence of robustness.")
    lines = ["# Ten-source temperature mapping: figure captions", "",
             "## Scope and interpretation", "",
             "These are regularized inverse-model temperature maps, not direct images of the disk. "
             "No propagation trajectories, wave speeds, significance contours or pixel uncertainties are inferred by this plotting script. "
             "Color saturation is display-only; fit arrays are not clipped. No pixel-level posterior uncertainties are supplied.", "",
             "The baseline is Nr=50, Nt=250 and xi=10 unless a source summary explicitly records otherwise. "
             "A common xi does not mean equivalent physical smoothing across sources or time-grid sizes: "
             "cadence, baseline length, errors, kernels and priors differ. The shared time axis notation is observer days "
             "since each source's own first observation, not a synchronized event or rest-frame timeline.", "",
             "Low/high-pass maps use the supplied filtered data fits. Smoothing induces correlated noise, "
             "and that covariance has not been propagated into a new likelihood. Their nominal chi-squared values "
             "must not be interpreted as independent-noise goodness-of-fit tests or compared as formal evidences. "
             "Finite coverage, gaps and endpoint extension are especially relevant when 28/56 days approaches the campaign duration.", "",
             "Independent q99 normalization is only used to compare morphology. It cannot support a claim of equal "
             "amplitude, equal significance, absence of disk variability or identification of disk/BLR/magnetosonic-wave components. "
             "Parameters, input provenance and quality warnings remain in the input manifest, physical parameter file and per-source summaries.", "",
             "## Baseline scales and coverage", "",
             "Absolute-map saturation is the fraction of all radial-time cells with |deltaT/T| > 0.2; "
             "it is not a fraction of statistically significant or independently resolved pixels.", "",
             "| Source | Bands | Data | Span (d) | Nt=250 spacing (d) | q99 | chi2/N | Absolute saturation |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in sources:
        z = s.data
        chi = float(np.mean(((z["flux"]-z["pred_xi10"])/z["error"])**2))
        saturation = 100*float(np.mean(np.abs(z["q_xi10"])>.2))
        lines.append(f"| {s.name} | {len(z['bands'])} | {len(z['t'])} | {np.ptp(z['t']):.3f} | "
                     f"{np.median(np.diff(z['tp'])):.4f} | {s.scale:.5g} | {chi:.4f} | {saturation:.3f}% |")
    lines += ["", "## Source-specific warnings", ""]
    for s in sources:
        lines += [f"### {s.name}", ""]
        warnings = s.summary.get("quality_warnings", [])
        if isinstance(warnings, str):
            warnings = [warnings]
        if not warnings:
            lines.append("No additional warnings supplied by the fit summary; the common limitations above still apply.")
        for warning in warnings:
            lines.append("- " + markdown_cell(warning))
        absent = [key for key in ("q_without_U", "q_swift6") if key not in s.data]
        if absent:
            lines += ["", "Optional panels not available: " + ", ".join(absent) + "."]
        lines += [""]
    lines += ["## Atlas page index and captions", ""]
    for entry in atlas.entries:
        lines += [f"### Page {entry['page']}: {entry['name']}", "", entry["caption"], "",
                  f"[PDF](figures/{entry['name']}.pdf) · [300-dpi PNG](figures/{entry['name']}.png) · "
                  f"[110-dpi preview](figures/{entry['name']}_preview.jpg)", ""]
    lines += ["## Export validation", "",
              f"{len(atlas.entries)} atlas pages; every page exported as vector PDF, 300-dpi PNG and 110-dpi JPEG. "
              f"Render time: {elapsed:.1f} seconds. See figures/render_validation.json for raster dimensions, "
              "stored DPI and text-bound checks. These automated bounds checks do not replace visual inspection.", ""]
    (OUT / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")
    (FIG / "render_validation.json").write_text(json.dumps(dict(
        pages=len(atlas.entries), render_seconds=elapsed, sources=[s.name for s in sources],
        figures=atlas.entries), indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true", help="Validate all inputs without exporting figures")
    modes.add_argument("--captions-only", action="store_true",
                       help="Refresh captions and their validation metadata from current summaries; do not rerender")
    args = parser.parse_args()
    started = time.perf_counter()
    sources = load_sources()
    if args.preflight:
        print(json.dumps({"valid_sources": [s.name for s in sources],
                          "total_points": sum(len(s.data["t"]) for s in sources)}, indent=2))
        return
    if args.captions_only:
        previous = read_json(FIG / "render_validation.json")
        if previous["sources"] != [s.name for s in sources]:
            raise ValueError("Existing figure metadata has a different source order")
        atlas = Atlas(None)
        atlas.entries = previous["figures"]
        write_captions(sources, atlas, previous["render_seconds"])
        print(f"Refreshed captions for {len(atlas.entries)} existing pages; no figures rerendered")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    with plt.style.context(ROOT / "paper.mplstyle"):
        atlas = Atlas()
        baseline_overviews(sources, atlas)
        for kind in ("regularization", "filter28", "timegrid", "subsets"):
            map_comparison_pages(sources, atlas, kind)
        for source in sources:
            filter_page(source, atlas)
            lightcurve_page(source, atlas)
            residual_page(source, atlas)
            coverage_page(source, atlas)
        atlas.merge()
        elapsed = time.perf_counter()-started
        write_captions(sources, atlas, elapsed)
    print(f"Completed {len(atlas.entries)} pages in {elapsed:.1f} s: {OUT/'temperature_mapping_atlas.pdf'}", flush=True)


if __name__ == "__main__":
    main()
