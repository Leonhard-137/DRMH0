#!/usr/bin/env python3
"""Build compact Results diagnostics from completed FFA and MICA products.

This module is deliberately a postprocessor: it reads existing sampler and
reconstruction products, and writes only figures and descriptive summaries.
It never changes an FFA or MICA run directory.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from collect_mica_results import (
    configured_run_name,
    configured_stages,
    mica_analysis_dir,
    mica_run_root,
    model_kinds,
    model_label,
    ncomp_from_config,
    stage_config,
)


ROOT = Path(__file__).resolve().parents[1]
ZERO_HOST_MODES = {"max_x0", "max_zero", "max_zero_point"}


def resolve_source_dir(path: Path) -> Path:
    """Resolve a source directory relative to the project root."""
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_config_path(source_dir: Path, path: Path | None) -> Path:
    """Return the source configuration path."""
    if path is None:
        return source_dir / "config" / "source_config.yaml"
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (source_dir / path).resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def source_path(source_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else source_dir / path


def band_key(band: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(band)).strip("_")


def total_ffa_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the total-light FFA configuration in either supported layout."""
    ffa = dict(config.get("fflux", {}) or {})
    if not ffa:
        ffa = dict(config.get("fflux_analysis", {}) or {})
    if not ffa:
        raise KeyError("missing total-light FFA configuration ('fflux')")

    component = dict(ffa.get("component", {}) or {})
    component.setdefault("name", "total")
    if str(component["name"]).lower() != "total":
        raise ValueError("the Results FFA/host figure requires component.name=total")
    ffa["component"] = component
    ffa.setdefault("source", config.get("source"))
    ffa.setdefault("bands", list((config.get("bands", {}) or {}).keys()))
    wavelengths = dict(config.get("bands", {}) or {})
    wavelengths.update(ffa.get("wavelengths", {}) or {})
    ffa["wavelengths"] = wavelengths
    if not ffa["bands"]:
        raise ValueError("total-light FFA configuration has no bands")
    missing = [str(band) for band in ffa["bands"] if str(band) not in wavelengths]
    if missing:
        raise ValueError(f"missing wavelengths for FFA bands: {missing}")
    return ffa


def _configured_dir(
    source_dir: Path,
    ffa: dict[str, Any],
    group: str,
    key: str,
    default: Path,
) -> Path:
    value = dict(ffa.get(group, {}) or {}).get(key)
    return default if value is None else source_path(source_dir, value)


def _single_product(directory: Path, pattern: str, description: str) -> Path:
    products = sorted(directory.glob(pattern))
    if len(products) != 1:
        raise FileNotFoundError(
            f"expected one {description} matching {directory / pattern}, "
            f"found {len(products)}"
        )
    return products[0]


def _load_light_curve(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"light curve not found: {path}")
    data = np.loadtxt(path, usecols=(0, 1, 2))
    data = data.reshape(1, -1) if data.ndim == 1 else data
    if data.ndim != 2 or data.shape[1] != 3 or len(data) < 2:
        raise ValueError(f"expected an N x 3 light curve in {path}, got {data.shape}")
    if not np.isfinite(data).all() or np.any(data[:, 2] < 0.0):
        raise ValueError(f"invalid values in light curve: {path}")
    if np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError(f"light-curve times are not strictly increasing: {path}")
    return data


def _ffa_products(
    source_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    ffa = total_ffa_config(config)
    source = str(ffa.get("source") or source_dir.name)
    component = str(ffa["component"]["name"])
    root = source_path(source_dir, ffa.get("out", "fflux"))
    fit_dir = _configured_dir(source_dir, ffa, "fit", "sample_dir", root / "fit")
    analysis_dir = _configured_dir(
        source_dir, ffa, "analysis", "output_dir", root / "analysis"
    )

    nband = len(ffa["bands"])
    components = analysis_dir / f"{source}_{component}_{nband}band_components.csv"
    if not components.is_file():
        components = _single_product(
            analysis_dir, "*_components.csv", "FFA component table"
        )
    posterior = analysis_dir / f"{source}_{component}_{nband}band_flux_posterior.txt"
    if not posterior.is_file():
        posterior = _single_product(
            analysis_dir, "*_flux_posterior.txt", "FFA flux posterior"
        )
    reconstructed_x = fit_dir / "reconstructed_X.txt"
    if not reconstructed_x.is_file():
        raise FileNotFoundError(f"reconstructed FFA driver not found: {reconstructed_x}")
    return ffa, analysis_dir, components, posterior, reconstructed_x


def _validate_x_gal(
    posterior: pd.DataFrame,
    bands: list[str],
    gal_ref: str,
) -> float:
    """Validate and return the posterior-median configured host anchor."""
    if "X_gal" not in posterior:
        raise KeyError("FFA flux posterior is missing X_gal")
    x_gal_samples = pd.to_numeric(posterior["X_gal"], errors="coerce").to_numpy(float)
    if not np.isfinite(x_gal_samples).all():
        raise ValueError("FFA X_gal posterior contains non-finite values")

    x0_samples = []
    for band in bands:
        column = f"X0_{band_key(band)}"
        if column not in posterior:
            raise KeyError(f"FFA flux posterior is missing {column}")
        values = pd.to_numeric(posterior[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"FFA posterior column {column} contains non-finite values")
        x0_samples.append(values)

    if gal_ref in ZERO_HOST_MODES:
        expected = np.max(np.vstack(x0_samples), axis=0)
    else:
        if gal_ref not in bands:
            raise ValueError(
                f"configured FFA sed.gal_ref={gal_ref!r} is neither a band nor a "
                f"supported zero-host mode"
            )
        expected = x0_samples[bands.index(gal_ref)]
    if not np.allclose(x_gal_samples, expected, rtol=1.0e-10, atol=1.0e-10):
        maximum = float(np.max(np.abs(x_gal_samples - expected)))
        raise ValueError(
            f"stored X_gal does not follow configured gal_ref={gal_ref!r}; "
            f"maximum sample difference is {maximum:.6g}"
        )
    return float(np.median(x_gal_samples))


def make_ffa_host_figure(
    source_dir: Path,
    config: dict[str, Any],
    *,
    output_dir: Path | None = None,
    source_label: str | None = None,
    sed_legend_loc: str = "best",
    sed_upper_margin: float = 1.0,
    compact_appendix: bool = False,
    sed_legend_items: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Plot a compact flux--flux relation and host/RMS SED.

    Optional paper destinations leave the saved analysis products untouched.
    ``source_label`` adds the campaign identification needed in a sample atlas.
    ``compact_appendix`` changes only typography and placement for a pair that
    occupies half of a paper's full text width; it uses the same saved values.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ffa, analysis_dir, components_path, posterior_path, x_path = _ffa_products(
        source_dir, config
    )
    source = str(ffa.get("source") or source_dir.name)
    component = str(ffa["component"]["name"])
    bands = [str(band) for band in ffa["bands"]]
    wavelengths = {band: float(ffa["wavelengths"][band]) for band in bands}
    sed_config = dict(ffa.get("sed", {}) or {})
    gal_ref = str(sed_config.get("gal_ref", config.get("reference_band", bands[0])))

    posterior = pd.read_csv(posterior_path, sep=r"\s+")
    x_gal = _validate_x_gal(posterior, bands, gal_ref)
    components = pd.read_csv(components_path)
    if "Filter" not in components:
        raise KeyError(f"FFA component table is missing Filter: {components_path}")
    components["Filter"] = components["Filter"].astype(str)
    components = components.set_index("Filter").loc[bands]
    if "gal_ref" in components:
        stored = {str(value) for value in components["gal_ref"].dropna().unique()}
        if stored and stored != {gal_ref}:
            raise ValueError(
                f"FFA component table records gal_ref={sorted(stored)}, "
                f"but the configuration requests {gal_ref!r}"
            )
    for column in ("X_min", "X_max"):
        if column not in components:
            raise KeyError(f"FFA component table is missing {column}: {components_path}")
        values = pd.to_numeric(components[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or not np.allclose(
            values, values[0], rtol=1.0e-10, atol=1.0e-10
        ):
            raise ValueError(f"FFA component table has inconsistent {column} values")
    x_state_min = float(components["X_min"].iloc[0])
    x_state_max = float(components["X_max"].iloc[0])
    if x_state_max <= x_state_min:
        raise ValueError("FFA X_max must be greater than X_min")

    reconstructed = np.loadtxt(x_path, usecols=(0, 1))
    reconstructed = reconstructed.reshape(1, -1) if reconstructed.ndim == 1 else reconstructed
    if reconstructed.shape[1] != 2 or len(reconstructed) < 2:
        raise ValueError(f"expected an N x 2 reconstructed X(t) file: {x_path}")
    if not np.isfinite(reconstructed).all() or np.any(np.diff(reconstructed[:, 0]) <= 0.0):
        raise ValueError(f"invalid reconstructed X(t): {x_path}")

    root = source_path(source_dir, ffa.get("out", "fflux"))
    dered_dir = _configured_dir(source_dir, ffa, "fit", "input_dir", root / "dered")
    template = str(
        sed_config.get("input_template", "{source}_{band}_{component}.txt")
    )
    curves = {}
    for band in bands:
        path = dered_dir / template.format(
            source=source, band=band, component=component
        )
        curves[band] = _load_light_curve(path)

    line_posteriors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for band in bands:
        key = band_key(band)
        a_column, r_column = f"A_{key}", f"R_{key}"
        if a_column not in posterior or r_column not in posterior:
            raise KeyError(f"missing FFA posterior columns for {band}: {a_column}, {r_column}")
        a = pd.to_numeric(posterior[a_column], errors="coerce").to_numpy(float)
        r = pd.to_numeric(posterior[r_column], errors="coerce").to_numpy(float)
        if not np.isfinite(a).all() or not np.isfinite(r).all():
            raise ValueError(f"non-finite FFA A/R posterior samples for {band}")
        line_posteriors[band] = (a, r)

    # Use one actual joint-posterior draw for every line and for X_gal.  This
    # preserves both the exact linear model and the joint host-anchor
    # constraint; combining separate marginal medians would do neither.
    summary_columns = [posterior["X_gal"].to_numpy(float)]
    for band in bands:
        summary_columns.extend(line_posteriors[band])
    summary_matrix = np.column_stack(summary_columns)
    center = np.median(summary_matrix, axis=0)
    scale = np.median(np.abs(summary_matrix - center), axis=0)
    fallback = np.std(summary_matrix, axis=0)
    scale = np.where(scale > 0.0, scale, np.where(fallback > 0.0, fallback, 1.0))
    summary_draw = int(
        np.argmin(np.sum(np.square((summary_matrix - center) / scale), axis=1))
    )
    x_gal = float(summary_matrix[summary_draw, 0])

    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(bands)))
    style = {
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.15,
    }
    if compact_appendix:
        style.update({
            "font.size": 7.0, "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
        })
    with plt.rc_context(style):
        fig, (ax_ff, ax_sed) = plt.subplots(
            1, 2, figsize=(3.55, 2.03) if compact_appendix else (7.2, 3.15),
            constrained_layout=not compact_appendix,
        )
        if compact_appendix:
            fig.subplots_adjust(left=0.085, right=0.99, top=0.86,
                                bottom=0.17, wspace=0.28)
        if source_label is not None:
            fig.suptitle(source_label, fontsize=8.5 if compact_appendix else 10)

        reconstructed_min = float(np.min(reconstructed[:, 1]))
        reconstructed_max = float(np.max(reconstructed[:, 1]))
        if not np.isclose(reconstructed_min, x_state_min, rtol=0.0, atol=5.0e-6):
            raise ValueError("reconstructed X(t) minimum disagrees with FFA X_min")
        if not np.isclose(reconstructed_max, x_state_max, rtol=0.0, atol=5.0e-6):
            raise ValueError("reconstructed X(t) maximum disagrees with FFA X_max")

        # Show the sampled X range, the zero point, the adopted host anchor,
        # and the zero-flux baseline.  Keep the data-supported fit solid and
        # distinguish its extrapolation to X_gal with a dashed segment.
        ax_ff.axhline(0.0, color="k", alpha=0.5, lw=0.9, ls="--", zorder=0)
        ax_ff.axvline(
            0.0,
            color="k",
            alpha=0.5,
            lw=0.9,
            ls="--",
            label=r"$X=0$",
            zorder=0,
        )
        ax_ff.axvline(
            x_state_min,
            color="k",
            alpha=0.5,
            lw=0.9,
            ls=":",
            label=r"$X_F$",
            zorder=0,
        )
        ax_ff.axvline(
            x_state_max,
            color="k",
            alpha=0.5,
            lw=0.9,
            ls=":",
            label=r"$X_B$",
            zorder=0,
        )
        ax_ff.axvline(
            x_gal,
            color="k",
            alpha=0.8,
            lw=0.9,
            ls="-.",
            label=r"$X_0$",
            zorder=0,
        )
        x_line = np.linspace(x_state_min, x_state_max, 300)
        if x_gal < x_state_min:
            x_extension = np.linspace(x_gal, x_state_min, 200)
        elif x_gal > x_state_max:
            x_extension = np.linspace(x_state_max, x_gal, 200)
        else:
            x_extension = np.asarray([], dtype=float)
        for color, band in zip(colors, bands):
            curve = curves[band]
            within = (
                (curve[:, 0] >= reconstructed[0, 0])
                & (curve[:, 0] <= reconstructed[-1, 0])
            )
            if not np.any(within):
                raise ValueError(f"no {band} observations overlap reconstructed X(t)")
            x_observed = np.interp(
                curve[within, 0], reconstructed[:, 0], reconstructed[:, 1]
            )
            ax_ff.scatter(
                x_observed,
                curve[within, 1],
                s=5,
                color=color,
                alpha=0.28,
                edgecolors="none",
                rasterized=True,
                zorder=1,
            )
            intercept, slope = line_posteriors[band]
            ax_ff.plot(
                x_line,
                intercept[summary_draw] + slope[summary_draw] * x_line,
                color=color,
                label=band,
                zorder=2,
            )
            if x_extension.size:
                ax_ff.plot(
                    x_extension,
                    intercept[summary_draw] + slope[summary_draw] * x_extension,
                    color=color,
                    ls="--",
                    lw=1.0,
                    alpha=0.8,
                    label="_nolegend_",
                    zorder=2,
                )
        x_display_min = min(x_state_min, x_gal)
        span = x_state_max - x_display_min
        if span <= 0.0:
            raise ValueError("reconstructed X(t) and X_gal do not span a finite range")
        ax_ff.set_xlim(
            x_display_min - 0.025 * span,
            x_state_max + 0.025 * span,
        )
        ax_ff.set_xlabel(r"$X(t)$ [dimensionless]")
        ax_ff.set_ylabel(
            r"$f_\lambda$ "
            r"[$10^{-15}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]"
        )
        if not compact_appendix:
            ax_ff.legend(
                loc="upper left",
                ncol=2,
                frameon=False,
                columnspacing=0.7,
                handlelength=1.5,
                fontsize=6.7,
            )
        ax_ff.text(0.02, 0.04, "(a)", transform=ax_ff.transAxes)

        wave = np.asarray([wavelengths[band] for band in bands], dtype=float)
        delta = components["Delta_F_p50"].to_numpy(float)
        delta_lo = components["Delta_F_p16"].to_numpy(float)
        delta_hi = components["Delta_F_p84"].to_numpy(float)
        host = components["F_Galaxy_p50"].to_numpy(float)
        host_lo = components["F_Galaxy_p16"].to_numpy(float)
        host_hi = components["F_Galaxy_p84"].to_numpy(float)
        faint_agn = []
        bright_agn = []
        rms_agn = []
        rms_lo = []
        rms_hi = []
        for band in bands:
            key = band_key(band)
            required = (
                f"F_Galaxy_{key}",
                f"F_Faint_{key}",
                f"F_Bright_{key}",
                f"R_{key}",
            )
            missing = [column for column in required if column not in posterior]
            if missing:
                raise KeyError(f"FFA flux posterior is missing columns: {missing}")
            assigned_constant = pd.to_numeric(
                posterior[required[0]], errors="coerce"
            ).to_numpy(float)
            faint_total = pd.to_numeric(
                posterior[required[1]], errors="coerce"
            ).to_numpy(float)
            bright_total = pd.to_numeric(
                posterior[required[2]], errors="coerce"
            ).to_numpy(float)
            rms_samples = pd.to_numeric(
                posterior[required[3]], errors="coerce"
            ).to_numpy(float)
            samples = (assigned_constant, faint_total, bright_total, rms_samples)
            if not all(np.isfinite(values).all() for values in samples):
                raise ValueError(f"non-finite FFA SED posterior samples for {band}")
            faint_agn.append(float(np.median(faint_total - assigned_constant)))
            bright_agn.append(float(np.median(bright_total - assigned_constant)))
            r16, r50, r84 = np.percentile(rms_samples, [16.0, 50.0, 84.0])
            rms_lo.append(float(r16))
            rms_agn.append(float(r50))
            rms_hi.append(float(r84))
        faint_agn = np.asarray(faint_agn)
        bright_agn = np.asarray(bright_agn)
        rms_agn = np.asarray(rms_agn)
        rms_lo = np.asarray(rms_lo)
        rms_hi = np.asarray(rms_hi)

        # The configured band order is not required to be monotonic in
        # wavelength.  Sort every plotted quantity together so that connected
        # curves and the filled range never double back on themselves.
        wave_order = np.argsort(wave)
        plot_bands = [bands[index] for index in wave_order]
        wave = wave[wave_order]
        delta = delta[wave_order]
        delta_lo = delta_lo[wave_order]
        delta_hi = delta_hi[wave_order]
        host = host[wave_order]
        host_lo = host_lo[wave_order]
        host_hi = host_hi[wave_order]
        faint_agn = faint_agn[wave_order]
        bright_agn = bright_agn[wave_order]
        rms_agn = rms_agn[wave_order]
        rms_lo = rms_lo[wave_order]
        rms_hi = rms_hi[wave_order]

        arrays = (
            wave,
            delta,
            delta_lo,
            delta_hi,
            host,
            host_lo,
            host_hi,
            faint_agn,
            bright_agn,
            rms_agn,
            rms_lo,
            rms_hi,
        )
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError(f"non-finite SED values in {components_path}")
        if np.any(faint_agn <= 0.0) or np.any(bright_agn <= faint_agn):
            raise ValueError("invalid host-subtracted AGN faint-to-bright range")
        if np.any(
            (rms_lo <= 0.0) | (rms_agn < rms_lo) | (rms_hi < rms_agn)
        ):
            raise ValueError("invalid AGN RMS posterior intervals")

        ax_sed.fill_between(
            wave,
            wave * faint_agn,
            wave * bright_agn,
            color="#56B4E9",
            alpha=0.22,
            linewidth=0.0,
            label="Faint-to-bright range",
            zorder=0,
        )
        rms_y = wave * rms_agn
        rms_yerr = np.vstack(
            (wave * (rms_agn - rms_lo), wave * (rms_hi - rms_agn))
        )
        ax_sed.errorbar(
            wave,
            rms_y,
            yerr=rms_yerr,
            fmt="D:",
            ms=3.4,
            color="#0072B2",
            lw=1.3,
            capsize=3,
            elinewidth=1.0,
            label=r"RMS spectrum: $\lambda R_\lambda$",
            zorder=3,
        )

        delta_y = wave * delta
        delta_valid = (
            (delta_y > 0.0)
            & (delta_lo > 0.0)
            & (delta >= delta_lo)
            & (delta_hi >= delta)
        )
        if not np.all(delta_valid):
            bad = [band for band, valid in zip(plot_bands, delta_valid) if not valid]
            raise ValueError(f"non-positive or invalid difference-spectrum points: {bad}")
        ax_sed.plot(
            wave,
            delta_y,
            marker="o",
            markerfacecolor="white",
            linestyle="--",
            ms=4.2,
            color="0.28",
            lw=1.2,
            markeredgewidth=0.8,
            label="Difference SED",
            zorder=2,
        )

        # The configured zero-host anchor is exactly zero and is intentionally
        # omitted from the logarithmic host-galaxy series.
        host_valid = (host > 0.0) & (host >= host_lo) & (host_hi >= host)
        host_constrained = host_valid & (host_lo > 0.0)
        if np.any(host_constrained):
            host_y = wave[host_constrained] * host[host_constrained]
            host_yerr = np.vstack(
                (
                    wave[host_constrained]
                    * (host[host_constrained] - host_lo[host_constrained]),
                    wave[host_constrained]
                    * (host_hi[host_constrained] - host[host_constrained]),
                )
            )
            ax_sed.errorbar(
                wave[host_constrained],
                host_y,
                yerr=host_yerr,
                fmt="s--",
                ms=4.0,
                color="#D55E00",
                lw=1.4,
                markeredgewidth=0.8,
                capsize=3,
                elinewidth=1.0,
                label="Constant background",
                zorder=2,
            )
        host_crosses_zero = host_valid & (host_lo <= 0.0)
        if np.any(host_crosses_zero):
            crossing_y = wave[host_crosses_zero] * host[host_crosses_zero]
            crossing_upper = (
                wave[host_crosses_zero]
                * (host_hi[host_crosses_zero] - host[host_crosses_zero])
            )
            ax_sed.errorbar(
                wave[host_crosses_zero],
                crossing_y,
                yerr=np.vstack((np.zeros_like(crossing_y), crossing_upper)),
                fmt="s",
                ms=4.0,
                markerfacecolor="white",
                markeredgecolor="#D55E00",
                color="#D55E00",
                capsize=3,
                elinewidth=1.0,
                label=r"Background: 68% lower bound $\leq 0$",
                zorder=2,
            )
        ax_sed.set_xscale("log")
        ax_sed.set_yscale("log")
        ax_sed.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
        ax_sed.set_ylabel(
            r"$\lambda F_\lambda$ "
            r"[$10^{-15}\,\mathrm{erg\,s^{-1}\,cm^{-2}}$]"
        )
        ax_sed.minorticks_on()
        ax_sed.grid(which="major", lw=0.6, alpha=0.12)
        if sed_upper_margin != 1.0:
            lower, upper = ax_sed.get_ylim()
            ax_sed.set_ylim(lower, upper * sed_upper_margin)
        if compact_appendix:
            handles, labels = ax_sed.get_legend_handles_labels()
            if sed_legend_items is not None:
                for handle, label in zip(handles, labels):
                    sed_legend_items.setdefault(label, handle)
            # The assembled page supplies one label for each axis type and
            # one SED legend; numeric scales remain specific to each source.
            ax_ff.set_ylabel("")
            ax_sed.set_ylabel("")
            from matplotlib.ticker import LogFormatterSciNotation, LogLocator, MaxNLocator, NullFormatter
            ax_ff.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax_ff.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax_sed.xaxis.set_major_locator(LogLocator(base=10, subs=(2.0, 4.0)))
            ax_sed.xaxis.set_major_formatter(LogFormatterSciNotation(
                labelOnlyBase=False, minor_thresholds=(float("inf"), float("inf"))))
            ax_sed.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax_sed.legend(loc=sed_legend_loc, frameon=False, handlelength=1.7, fontsize=6.0)
        ax_sed.text(0.02, 0.04, "(b)", transform=ax_sed.transAxes)

        for axis in (ax_ff, ax_sed):
            axis.tick_params(which="both", direction="in", top=True, right=True)

        destination = analysis_dir if output_dir is None else Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        stem = destination / f"{source}_total_ffa_host_diagnostics"
        pdf = stem.with_suffix(".pdf")
        png = stem.with_suffix(".png")
        fig.savefig(pdf, bbox_inches=None if compact_appendix else "tight")
        fig.savefig(png, dpi=400, bbox_inches=None if compact_appendix else "tight")
        plt.close(fig)
    return {"pdf": pdf, "png": png}


def _ncomp(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        value = value[0]
    return int(value)


def _branch_relative_dir(fit: dict[str, Any], branch_id: str) -> Path:
    branches = dict(fit.get("posterior_branches", {}) or {})
    template = str(branches.get("result_dir_template", "branches/{branch_id}"))
    rendered = template.format(
        branch_id=branch_id,
        ncomp=_ncomp(fit.get("ncomp", fit.get("number_component", 2))),
        model=str(fit.get("type_tf", "gaussian")).lower(),
    )
    path = Path(rendered)
    if not rendered.strip() or path.is_absolute() or ".." in path.parts:
        raise ValueError("posterior branch result path must be a safe relative path")
    return path


def _run_name(fit: dict[str, Any], band: str) -> str:
    driver = str(fit["driver"])
    ncomp = _ncomp(fit.get("ncomp", fit.get("number_component", 2)))
    model = str(fit.get("type_tf", "gaussian")).lower()
    template = str(
        fit.get(
            "run_template",
            "run_{driver}_to_{band}_{ncomp}comp_{type_tf}",
        )
    )
    return template.format(
        driver=driver,
        band=band,
        response=band,
        ncomp=ncomp,
        type_tf=model,
    )


def _load_decomposition_response(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the driver and response blocks from one MICA pline product."""
    if not path.is_file():
        raise FileNotFoundError(f"MICA decomposition file not found: {path}")
    counts: tuple[int, int] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = re.fullmatch(r"#\s*(\d+)\s*:\s*(\d+)\s*", line)
            if match is not None:
                counts = (int(match.group(1)), int(match.group(2)))
                break
            if not line.startswith("#"):
                break
    if counts is None:
        raise ValueError(f"missing '# n_driver:n_response' header in {path}")
    data = np.loadtxt(path, usecols=(0, 1, 2))
    data = data.reshape(1, -1) if data.ndim == 1 else data
    expected = counts[0] + counts[1]
    if data.shape != (expected, 3):
        raise ValueError(
            f"MICA decomposition shape mismatch in {path}: "
            f"expected {(expected, 3)}, got {data.shape}"
        )
    if not np.isfinite(data).all() or np.any(data[:, 2] < 0.0):
        raise ValueError(f"invalid MICA decomposition values in {path}")
    driver = data[: counts[0]].copy()
    response = data[counts[0] :].copy()
    if np.any(np.diff(driver[:, 0]) <= 0.0) or np.any(np.diff(response[:, 0]) <= 0.0):
        raise ValueError(f"non-increasing MICA reconstruction times in {path}")
    return driver, response


def _observed_response_interval(path: Path) -> tuple[float, float]:
    observed = _load_light_curve(path)
    return float(observed[0, 0]), float(observed[-1, 0])


def make_mica_rms_table(
    source_dir: Path,
    config: dict[str, Any],
    stage: str,
    branch_id: str = "comp0",
) -> Path:
    """Compute descriptive RMS values for every native MICA component."""
    fit = stage_config(config, stage)
    ncomp = ncomp_from_config(fit)
    model = model_label(fit.get("type_tf", "gaussian"))
    kinds = model_kinds(model, ncomp)
    driver = str(fit.get("driver", ""))
    responses = list(dict.fromkeys(str(band) for band in (fit.get("responses") or [])))
    if not driver or not responses:
        raise ValueError(f"{stage} must define driver and response bands")
    if driver in responses:
        raise ValueError("the MICA driver must not be included in response RMS rows")

    run_root = mica_run_root(source_dir, fit)
    source = str(config.get("source") or source_dir.name)
    wavelengths = dict(config.get("bands", {}) or {})
    rows = []

    for band in responses:
        run_dir = run_root / configured_run_name(fit, band)
        decompositions = [
            _load_decomposition_response(
                run_dir / "data" / f"pline.txt_{ncomp}_comp{index}"
            )
            for index in range(ncomp)
        ]
        driver0, response0 = decompositions[0]
        for index, (driver_component, response_component) in enumerate(
            decompositions[1:], start=1
        ):
            if driver0.shape != driver_component.shape or not np.allclose(
                driver0, driver_component, rtol=0.0, atol=5.0e-7
            ):
                raise ValueError(f"component driver blocks differ for {band}/tf{index}")
            if response0.shape != response_component.shape or not np.allclose(
                response0[:, 0], response_component[:, 0], rtol=0.0, atol=5.0e-7
            ):
                raise ValueError(f"component response grids differ for {band}/tf{index}")

        observed_path = run_dir / "input" / "line.txt"
        observed_min, observed_max = _observed_response_interval(observed_path)
        times = response0[:, 0]
        use = (times >= observed_min) & (times <= observed_max)
        if np.count_nonzero(use) < 2:
            raise ValueError(
                f"fewer than two reconstructed response times fall inside the "
                f"observed {band} interval"
            )
        component_stats = []
        for _, response_component in decompositions:
            values = response_component[use, 1]
            mean = float(np.mean(values))
            component_stats.append(
                (mean, float(np.sqrt(np.mean(np.square(values - mean)))), float(np.ptp(values)))
            )
        rms0 = component_stats[0][1]
        for index, (mean, rms, peak_to_peak) in enumerate(component_stats):
            rows.append({
                "source": source,
                "mica_stage": stage,
                "selection": "full_native_posterior",
                "band": band,
                "wavelength": float(wavelengths.get(band, np.nan)),
                "component": f"tf{index}",
                "tf_kind": kinds[index],
                "n_time": int(np.count_nonzero(use)),
                "time_min": float(times[use][0]),
                "time_max": float(times[use][-1]),
                "mean": mean,
                "rms": rms,
                "peak_to_peak": peak_to_peak,
                "rms_to_tf0": rms / rms0 if rms0 > 0.0 else np.nan,
            })

    output_dir = mica_analysis_dir(source_dir, fit, "comp0") / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "mica_response_rms.csv"
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.10g")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create compact FFA/host and MICA RMS Results diagnostics."
    )
    parser.add_argument("source_dir", type=Path, help="source directory")
    parser.add_argument("--stage", help="one MICA config section; default: all mica_stages")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="configuration path relative to source_dir",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="official posterior branch (default: hubble_distance.posterior_branch)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = resolve_source_dir(args.source_dir)
    config_file = resolve_config_path(source_dir, args.config)
    config = read_yaml(config_file)
    figure = make_ffa_host_figure(source_dir, config)
    print(f"FFA/host PDF: {figure['pdf']}")
    print(f"FFA/host PNG: {figure['png']}")
    stages = [args.stage] if args.stage else configured_stages(config)
    for stage in stages:
        rms = make_mica_rms_table(source_dir, config, stage, branch_id="comp0")
        print(f"MICA RMS table [{stage}]: {rms}")


if __name__ == "__main__":
    main()
