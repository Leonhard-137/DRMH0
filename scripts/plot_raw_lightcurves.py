#!/usr/bin/env python3
"""Plot every source's raw multi-band light curves into its raw directory."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {"Filter", "MJD", "Flux", "Error"}

# Approximate observed-frame effective wavelengths, used only to order and label
# panels. X-ray bands are placed before the UV/optical bands.
WAVELENGTHS = {
    "HX": -3.0,
    "SX": -2.0,
    "X": -1.0,
    "HST1157.5": 1157.5,
    "1180A": 1180.0,
    "HST1367": 1367.0,
    "1398A": 1398.0,
    "HST1478.5": 1478.5,
    "1502A": 1502.0,
    "HST1746": 1746.0,
    "1739A": 1739.0,
    "W2": 1928.0,
    "UVW2": 1928.0,
    "M2": 2246.0,
    "UVM2": 2246.0,
    "W1": 2600.0,
    "UVW1": 2600.0,
    "U": 3465.0,
    "u": 3543.0,
    "F4225": 4225.0,
    "BG": 4369.0,
    "B": 4392.0,
    "F4500": 4500.0,
    "F4765": 4765.0,
    "g": 4776.0,
    "F5100": 5100.0,
    "VG": 5404.0,
    "V": 5468.0,
    "lV": 5468.0,
    "F5500": 5500.0,
    "F6000": 6000.0,
    "r": 6176.0,
    "F6420": 6420.0,
    "R": 6440.0,
    "F6925": 6925.0,
    "F7210": 7210.0,
    "i": 7648.0,
    "I": 8561.0,
    "z": 9157.0,
}


def inferred_wavelength(band: str) -> float:
    """Return a plotting-order wavelength when it can be inferred."""
    if band in WAVELENGTHS:
        return WAVELENGTHS[band]
    match = re.fullmatch(r"F?(\d+(?:\.\d+)?)A?", band)
    return float(match.group(1)) if match else float("inf")


def ordered_bands(values: pd.Series) -> list[str]:
    bands = [str(value) for value in pd.unique(values.dropna())]
    return sorted(bands, key=lambda band: (inferred_wavelength(band), band.casefold()))


def source_csv(source_dir: Path) -> Path | None:
    raw_dir = source_dir / "raw"
    preferred = raw_dir / f"{source_dir.name}.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(raw_dir.glob("*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def line_with_gaps(times: np.ndarray, fluxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaNs across major observing gaps so lines do not bridge seasons."""
    if len(times) < 3:
        return times, fluxes
    gaps = np.diff(times)
    positive = gaps[gaps > 0.0]
    if len(positive) == 0:
        return times, fluxes
    span = float(times[-1] - times[0])
    threshold = max(5.0 * float(np.median(positive)), 0.03 * span)
    out_t: list[float] = [float(times[0])]
    out_y: list[float] = [float(fluxes[0])]
    for index in range(1, len(times)):
        if times[index] - times[index - 1] > threshold:
            out_t.append(float("nan"))
            out_y.append(float("nan"))
        out_t.append(float(times[index]))
        out_y.append(float(fluxes[index]))
    return np.asarray(out_t), np.asarray(out_y)


def band_title(band: str) -> str:
    wave = inferred_wavelength(band)
    if not np.isfinite(wave) or wave <= 0.0:
        return band
    return f"{band}  ({wave:g} Å)"


def plot_source(source_dir: Path, csv_path: Path, *, dpi: int = 250) -> tuple[Path, Path]:
    data = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    for column in ("MJD", "Flux", "Error"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["Filter"] = data["Filter"].astype(str).str.strip()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["Filter", "MJD", "Flux", "Error"])
    data = data[data["Error"] >= 0.0]
    if data.empty:
        raise ValueError(f"{csv_path} has no finite light-curve rows")

    bands = ordered_bands(data["Filter"])
    ncols = 1 if len(bands) <= 3 else 2
    nrows = math.ceil(len(bands) / ncols)
    figsize = (7.2, max(4.2, 2.25 * nrows + 1.25)) if ncols == 1 else (
        12.2,
        max(5.6, 2.35 * nrows + 1.15),
    )

    mjd_min = float(data["MJD"].min())
    mjd_max = float(data["MJD"].max())
    mjd_zero = 100.0 * math.floor(mjd_min / 100.0)
    data = data.assign(MJD_plot=data["MJD"] - mjd_zero)

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.85,
        "axes.edgecolor": "#262626",
        "axes.labelcolor": "#262626",
        "xtick.color": "#262626",
        "ytick.color": "#262626",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            sharex=True,
            squeeze=False,
        )
        flat_axes = axes.ravel()
        color = "#1769AA"
        dark_color = "#0D4775"

        for index, band in enumerate(bands):
            axis = flat_axes[index]
            subset = data[data["Filter"] == band].sort_values("MJD_plot")
            times = subset["MJD_plot"].to_numpy(float)
            fluxes = subset["Flux"].to_numpy(float)
            errors = subset["Error"].to_numpy(float)
            line_t, line_y = line_with_gaps(times, fluxes)

            axis.plot(
                line_t,
                line_y,
                color=color,
                linewidth=0.72,
                alpha=0.62,
                zorder=1,
            )
            axis.errorbar(
                times,
                fluxes,
                yerr=errors,
                fmt="o",
                markersize=2.45,
                markerfacecolor=color,
                markeredgecolor=dark_color,
                markeredgewidth=0.22,
                linestyle="none",
                elinewidth=0.55,
                ecolor=color,
                capsize=0.9,
                alpha=0.78,
                rasterized=True,
                zorder=2,
            )
            axis.set_title(band_title(band), loc="left", fontsize=10.5, pad=4.0)
            axis.text(
                0.985,
                0.94,
                f"N = {len(subset)}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.0,
                color="#555555",
            )
            axis.grid(axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.72)
            axis.set_axisbelow(True)
            axis.tick_params(which="both", top=True, right=True, labelsize=8.5)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))
            axis.margins(x=0.018, y=0.14)

            if index % ncols == 0:
                axis.set_ylabel("Flux", fontsize=9.5)

        for index in range(len(bands), len(flat_axes)):
            flat_axes[index].set_visible(False)

        visible_last_row = range((nrows - 1) * ncols, min(nrows * ncols, len(bands)))
        for index in visible_last_row:
            flat_axes[index].set_xlabel(f"MJD − {mjd_zero:g}", fontsize=9.5)

        fig.suptitle(f"{source_dir.name} raw light curves", fontsize=15.0, y=0.995)
        fig.text(
            0.5,
            0.973,
            (
                f"Independent flux scales · error bars: 1σ · "
                f"MJD {mjd_min:.2f}–{mjd_max:.2f} · N = {len(data)}"
            ),
            ha="center",
            va="top",
            fontsize=8.8,
            color="#555555",
        )
        fig.subplots_adjust(
            left=0.075,
            right=0.985,
            bottom=0.065,
            top=0.935,
            hspace=0.34,
            wspace=0.22,
        )

        output_stem = source_dir / "raw" / f"{source_dir.name}_raw_lightcurves"
        png_path = output_stem.with_suffix(".png")
        pdf_path = output_stem.with_suffix(".pdf")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

    return png_path, pdf_path


def discover_sources(root: Path) -> list[tuple[Path, Path]]:
    found = []
    for source_dir in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        if not source_dir.is_dir() or not (source_dir / "raw").is_dir():
            continue
        csv_path = source_csv(source_dir)
        if csv_path is not None:
            found.append((source_dir, csv_path))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="*",
        help="optional source directory names; default: every source with raw CSV data",
    )
    parser.add_argument("--dpi", type=int, default=250)
    args = parser.parse_args()

    if args.sources:
        pairs = []
        for name in args.sources:
            source_dir = (ROOT / name).resolve()
            csv_path = source_csv(source_dir)
            if csv_path is None:
                raise FileNotFoundError(f"expected one raw CSV under {source_dir / 'raw'}")
            pairs.append((source_dir, csv_path))
    else:
        pairs = discover_sources(ROOT)

    if not pairs:
        raise FileNotFoundError(f"no source raw CSV files found under {ROOT}")

    for source_dir, csv_path in pairs:
        png_path, pdf_path = plot_source(source_dir, csv_path, dpi=args.dpi)
        print(f"{source_dir.name}: {png_path}")
        print(f"{source_dir.name}: {pdf_path}")


if __name__ == "__main__":
    main()
