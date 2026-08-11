#!/usr/bin/env python3
"""Overlay smooth theoretical disk TFs and UVW2-relative Gamma kernels.

The absolute transfer functions are recomputed at high radial resolution to
remove the small annulus-discretization ripple visible in the original plot.
The relative kernels use the existing full-domain finite-difference forward
fit.  All stored densities retain their unit-area normalization; only the
display curves are independently peak-normalized so that the narrow UVM2
relative kernel does not compress the absolute transfer functions.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve


ROOT = Path(__file__).resolve().parents[1]
THEORY_DIR = ROOT / "test" / "disk_transfer"
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from disktf_diff import DiskConfig, disk_transfer_function  # noqa: E402


RESULT_DIR = ROOT / "Mrk817/results/tf_comparison/theory_forward_gamma"
DEFAULT_SUMMARY = RESULT_DIR / "theory_forward_gamma_summary.csv"
DEFAULT_OUTPUT = RESULT_DIR / "Mrk817_theory_relative_vs_absolute_overlay"

BANDS = {
    "UVW2": 1928.0,
    "UVM2": 2246.0,
    "UVW1": 2600.0,
    "U": 3465.0,
    "B": 4392.0,
    "V": 5468.0,
}
BAND_COLORS = {
    "UVW2": "#24282E",
    "UVM2": "#4C78A8",
    "UVW1": "#59A14F",
    "U": "#E58A24",
    "B": "#D15357",
    "V": "#9A6FA4",
}

# Lewin et al. (2024) temperatures/inclination, with the disk code's radial
# limits and temperature exponent retained from the earlier forward fit.
DISK_PARAMETERS = {
    "Rin": 0.1,
    "Rout": 30.0,
    "R0": 1.0,
    "TB": 11400.0,
    "TF": 5900.0,
    "b": 0.75,
    "inc_deg": 19.0,
}
NR = 6400
NTAU = 4000
RESPONSE_MODE = "finite_difference"
PLOT_MIN_DAYS = -0.55
PLOT_MAX_DAYS = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def gamma_k2(delay: np.ndarray, c: float, w: float) -> np.ndarray:
    """Unit-area shifted shape-2 Gamma density."""
    density = np.zeros_like(delay, dtype=float)
    offset = delay - c
    mask = offset >= 0.0
    scaled = offset[mask] / w
    density[mask] = scaled * np.exp(-scaled) / w
    return density


def peak_normalize(values: np.ndarray) -> np.ndarray:
    peak = float(np.nanmax(values))
    if not math.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("Cannot peak-normalize a non-positive curve.")
    return np.asarray(values, dtype=float) / peak


def build_absolute_tfs() -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    time_days: np.ndarray | None = None
    dt_days: float | None = None
    curves: dict[str, np.ndarray] = {}
    for band, wavelength in BANDS.items():
        config = DiskConfig(
            **DISK_PARAMETERS,
            lam_A=wavelength,
            NR=NR,
            Ntau=NTAU,
            response_mode=RESPONSE_MODE,
        )
        band_time, density = disk_transfer_function(config)
        band_dt = float(np.median(np.diff(band_time)))
        if time_days is None:
            time_days = np.asarray(band_time, dtype=float)
            dt_days = band_dt
        elif not np.allclose(band_time, time_days, rtol=0.0, atol=1e-12):
            raise RuntimeError("Theoretical transfer functions use different grids.")
        curves[band] = np.asarray(density, dtype=float)

    assert time_days is not None and dt_days is not None
    return time_days, dt_days, curves


def build_forward_reconstruction(
    time_days: np.ndarray,
    dt_days: float,
    driver: np.ndarray,
    c: float,
    w: float,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical time and A * (absolute W2 TF convolved with K)."""
    kernel_min = math.floor(-5.0 / dt_days) * dt_days
    kernel_max = 30.0
    kernel_count = int(math.ceil((kernel_max - kernel_min) / dt_days)) + 1
    kernel_time = kernel_min + np.arange(kernel_count, dtype=float) * dt_days
    kernel = gamma_k2(kernel_time, c, w)
    kernel_area = float(np.sum(kernel) * dt_days)
    if not (0.999 <= kernel_area <= 1.001):
        raise RuntimeError(f"Relative kernel area is {kernel_area:.8f}, not unity.")
    kernel /= kernel_area

    model = amplitude * fftconvolve(driver, kernel, mode="full") * dt_days
    model_time = time_days[0] + kernel_time[0] + np.arange(model.size) * dt_days
    return model_time, model


def interpolate_for_plot(
    plot_time: np.ndarray,
    source_time: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(plot_time, source_time, values, left=0.0, right=0.0)


def style_axis(axis: plt.Axes, panel_index: int) -> None:
    axis.axvspan(PLOT_MIN_DAYS, 0.0, color="#EEF0F2", zorder=0)
    axis.axvline(0.0, color="#858A91", lw=0.8, zorder=1)
    axis.set_xlim(PLOT_MIN_DAYS, PLOT_MAX_DAYS)
    axis.set_ylim(0.0, 1.08)
    axis.grid(axis="y", color="#D9DDE1", lw=0.65, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    if panel_index % 3 != 0:
        axis.tick_params(labelleft=False)
    if panel_index < 3:
        axis.tick_params(labelbottom=False)


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    summary = summary[
        (summary["parameter_set"] == "mrk817_lewin2024")
        & (summary["response_mode"] == RESPONSE_MODE)
        & (summary["fit_domain"] == "full")
    ].copy()
    if set(summary["target_band"]) != set(BANDS) - {"UVW2"}:
        raise RuntimeError("Expected one full-domain finite-difference fit per target band.")
    summary = summary.set_index("target_band")

    absolute_time, dt_days, absolute = build_absolute_tfs()
    plot_time = np.arange(
        PLOT_MIN_DAYS,
        PLOT_MAX_DAYS + 0.25 * dt_days,
        dt_days / 2.0,
    )
    w2_plot_raw = interpolate_for_plot(
        plot_time, absolute_time, absolute["UVW2"]
    )
    w2_plot = peak_normalize(w2_plot_raw)

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.1, 8.35),
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()
    output_rows: list[dict[str, float | str]] = []

    for panel_index, (axis, band) in enumerate(zip(axes_flat, BANDS)):
        style_axis(axis, panel_index)
        color = BAND_COLORS[band]
        target_raw = interpolate_for_plot(plot_time, absolute_time, absolute[band])
        target_plot = peak_normalize(target_raw)

        if band == "UVW2":
            axis.plot(
                plot_time,
                target_plot,
                color=color,
                lw=2.5,
                label="UVW2 absolute theory TF",
            )
            axis.annotate(
                "",
                xy=(0.0, 1.0),
                xytext=(0.0, 0.0),
                arrowprops={"arrowstyle": "-|>", "color": "#2F6F9F", "lw": 2.0},
            )
            axis.text(
                0.075,
                0.90,
                r"relative self-kernel $\delta(t)$",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9.1,
                color="#2F6F9F",
            )
            c = w = tau = amplitude = float("nan")
            relative_raw = np.zeros_like(plot_time)
            relative_plot = np.zeros_like(plot_time)
            reconstruction_raw = target_raw.copy()
            reconstruction_plot = target_plot.copy()
        else:
            row = summary.loc[band]
            c = float(row["c_days"])
            w = float(row["w_days"])
            tau = float(row["gamma_centroid_days"])
            amplitude = float(row["amplitude_A"])
            relative_raw = gamma_k2(plot_time, c, w)
            relative_plot = peak_normalize(relative_raw)
            model_time, model_raw_full = build_forward_reconstruction(
                absolute_time,
                dt_days,
                absolute["UVW2"],
                c,
                w,
                amplitude,
            )
            reconstruction_raw = interpolate_for_plot(
                plot_time, model_time, model_raw_full
            )
            reconstruction_plot = peak_normalize(reconstruction_raw)

            axis.plot(
                plot_time,
                w2_plot,
                color="#757B82",
                lw=1.45,
                ls=":",
                label="UVW2 absolute theory TF",
            )
            axis.plot(
                plot_time,
                target_plot,
                color=color,
                lw=2.35,
                label=f"{band} absolute theory TF",
            )
            axis.plot(
                plot_time,
                relative_plot,
                color="#2F6F9F",
                lw=1.9,
                ls="--",
                label=r"relative Gamma $K_{\lambda\leftarrow W2}$",
            )
            axis.plot(
                plot_time,
                reconstruction_plot,
                color="#C56B25",
                lw=1.35,
                ls="-.",
                alpha=0.92,
                label=r"forward $A(\psi_{W2}*K)$",
            )
            axis.text(
                0.97,
                0.95,
                rf"$c={c:.2f}$, $w={w:.2f}$, $\tau=c+2w={tau:.2f}$ d",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.6,
                color="#373C42",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )

        axis.set_title(f"{band} ({BANDS[band]:.0f} Å)", fontsize=11.7, pad=7)

        for index, time_value in enumerate(plot_time):
            output_rows.append(
                {
                    "band": band,
                    "delay_days": float(time_value),
                    "absolute_reference": "central impulse",
                    "relative_reference": "UVW2",
                    "uvw2_absolute_unit_area_per_day": float(w2_plot_raw[index]),
                    "target_absolute_unit_area_per_day": float(target_raw[index]),
                    "relative_gamma_unit_area_per_day": float(relative_raw[index]),
                    "forward_reconstruction_per_day": float(reconstruction_raw[index]),
                    "uvw2_absolute_peak_normalized": float(w2_plot[index]),
                    "target_absolute_peak_normalized": float(target_plot[index]),
                    "relative_gamma_peak_normalized": float(relative_plot[index]),
                    "forward_reconstruction_peak_normalized": float(
                        reconstruction_plot[index]
                    ),
                    "c_days": c,
                    "w_days": w,
                    "gamma_centroid_days": tau,
                    "amplitude_A": amplitude,
                    "NR": NR,
                    "Ntau": NTAU,
                }
            )

    for axis in axes[-1, :]:
        axis.set_xlabel("Delay coordinate [day]")
    for axis in axes[:, 0]:
        axis.set_ylabel("Peak-normalized response")

    legend_handles = [
        Line2D([0], [0], color="#757B82", lw=1.45, ls=":", label="UVW2 absolute theory TF"),
        Line2D([0], [0], color="#24282E", lw=2.35, label="target-band absolute theory TF"),
        Line2D([0], [0], color="#2F6F9F", lw=1.9, ls="--", label="UVW2-relative Gamma kernel"),
        Line2D([0], [0], color="#C56B25", lw=1.35, ls="-.", label="forward convolution"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=4,
        frameon=False,
        fontsize=9.6,
        handlelength=3.0,
        columnspacing=1.6,
    )
    figure.suptitle(
        "Mrk 817 theoretical disk TFs and UVW2-relative Gamma kernels",
        fontsize=16.2,
        y=0.987,
    )
    figure.text(
        0.5,
        0.944,
        (
            "finite-difference disk response; high radial resolution "
            f"($N_R={NR}$, $N_\\tau={NTAU}$); no post-smoothing"
        ),
        ha="center",
        va="top",
        fontsize=9.8,
        color="#535961",
    )
    figure.text(
        0.5,
        0.018,
        (
            "Display curves are independently peak-normalized.  Absolute TFs use the central impulse as t=0; "
            "relative kernels use UVW2 as t=0.  Shading marks negative relative delay."
        ),
        ha="center",
        va="bottom",
        fontsize=9.1,
        color="#535961",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.805, bottom=0.090, hspace=0.24, wspace=0.10)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    pdf_path = args.output.with_suffix(".pdf")
    csv_path = args.output.with_name(args.output.name + "_curves.csv")
    figure.savefig(png_path, dpi=250, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    # Numerical QA printed for the run log.
    areas = {
        band: float(np.sum(density) * dt_days)
        for band, density in absolute.items()
    }
    print(f"dt_days={dt_days:.10f}")
    print("absolute_area_max_error=", max(abs(value - 1.0) for value in areas.values()))
    print(png_path)
    print(pdf_path)
    print(csv_path)


if __name__ == "__main__":
    main()
