#!/usr/bin/env python3
"""Plot unnormalized disk TFs and moment-matched MICA Gamma reconstructions.

For each Swift target band, this script keeps the raw finite-difference disk
response area and constructs the MICA k=2 shifted-Gamma relative kernel whose
zeroth, first, and second moments make the forward convolution match those of
the target theoretical absolute transfer function exactly in the continuous
limit.  No curve is area- or peak-normalized for display.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
from scipy.special import gammainc


ROOT = Path(__file__).resolve().parents[1]
THEORY_DIR = ROOT / "test" / "disk_transfer"
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from disktf_diff import (  # noqa: E402
    DiskConfig,
    annulus_weight,
    c_light,
    disk_transfer_function,
)


RESULT_DIR = ROOT / "Mrk817/results/tf_comparison/theory_forward_gamma"
DEFAULT_OUTPUT = RESULT_DIR / "Mrk817_theory_raw_moment_gamma_overlay"

BANDS = {
    "UVW2": 1928.0,
    "UVM2": 2246.0,
    "UVW1": 2600.0,
    "U": 3465.0,
    "B": 4392.0,
    "V": 5468.0,
}
TARGET_BANDS = tuple(band for band in BANDS if band != "UVW2")
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
PLOT_MIN_DAYS = -0.8
PLOT_MAX_DAYS = 6.0
ABSOLUTE_DISPLAY_SCALE = 1.0e6


@dataclass(frozen=True)
class Moments:
    m0: float
    m1: float
    m2: float
    mean: float
    variance: float


@dataclass(frozen=True)
class GammaSolution:
    amplitude: float
    c: float
    w: float
    centroid: float
    variance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--rin",
        type=float,
        default=DISK_PARAMETERS["Rin"],
        help="Inner disk radius in light-days.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.rin) or args.rin <= 0.0:
        parser.error("--rin must be positive and finite")
    if args.rin >= DISK_PARAMETERS["Rout"]:
        parser.error("--rin must be smaller than Rout")
    if args.output is None:
        if math.isclose(args.rin, DISK_PARAMETERS["Rin"]):
            args.output = DEFAULT_OUTPUT
        else:
            rin_tag = f"{args.rin:g}".replace(".", "p")
            args.output = RESULT_DIR / (
                f"Mrk817_theory_Rin{rin_tag}_raw_moment_gamma_overlay"
            )
    return args


def raw_annulus_area(config: DiskConfig) -> float:
    """Integral of the pre-normalization disk response in common code units."""
    edges = np.logspace(
        np.log10(config.Rin), np.log10(config.Rout), config.NR + 1
    )
    radii = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)
    frequency = c_light / (config.lam_A * 1.0e-10)
    weights = np.fromiter(
        (annulus_weight(frequency, float(radius), config) for radius in radii),
        dtype=float,
        count=radii.size,
    )
    return float(
        np.sum(
            2.0
            * np.pi
            * weights
            * radii
            * widths
            * np.cos(np.radians(config.inc_deg))
        )
    )


def density_moments(time: np.ndarray, density: np.ndarray, dt: float) -> Moments:
    m0 = float(np.sum(density) * dt)
    m1 = float(np.sum(time * density) * dt)
    m2 = float(np.sum(time * time * density) * dt)
    if not math.isfinite(m0) or m0 <= 0.0:
        raise RuntimeError(f"Non-positive density area: {m0}")
    mean = m1 / m0
    variance = m2 / m0 - mean * mean
    return Moments(m0=m0, m1=m1, m2=m2, mean=mean, variance=variance)


def build_raw_theory(rin: float) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    common_time: np.ndarray | None = None
    common_dt: float | None = None
    raw_curves: dict[str, np.ndarray] = {}
    for band, wavelength in BANDS.items():
        config = DiskConfig(
            **{**DISK_PARAMETERS, "Rin": rin},
            lam_A=wavelength,
            NR=NR,
            Ntau=NTAU,
            response_mode=RESPONSE_MODE,
        )
        time_days, unit_area_density = disk_transfer_function(config)
        dt_days = float(np.median(np.diff(time_days)))
        raw_area = raw_annulus_area(config)
        raw_density = np.asarray(unit_area_density, dtype=float) * raw_area

        if common_time is None:
            common_time = np.asarray(time_days, dtype=float)
            common_dt = dt_days
        elif not np.allclose(time_days, common_time, rtol=0.0, atol=1e-12):
            raise RuntimeError("Theoretical bands do not share a time grid.")
        raw_curves[band] = raw_density

    assert common_time is not None and common_dt is not None
    return common_time, common_dt, raw_curves


def mica_gamma(time: np.ndarray, solution: GammaSolution) -> np.ndarray:
    """MICA k=2 Gamma; component amplitude is its zeroth moment/area."""
    density = np.zeros_like(time, dtype=float)
    offset = time - solution.c
    mask = offset >= 0.0
    scaled = offset[mask] / solution.w
    density[mask] = (
        solution.amplitude * scaled * np.exp(-scaled) / solution.w
    )
    return density


def solve_gamma(driver: Moments, target: Moments) -> GammaSolution:
    delta_variance = target.variance - driver.variance
    if delta_variance <= 0.0:
        raise RuntimeError(
            "A positive convolution kernel cannot reproduce a target narrower "
            "than its driver."
        )
    amplitude = target.m0 / driver.m0
    centroid = target.mean - driver.mean
    w = math.sqrt(delta_variance / 2.0)
    c = centroid - 2.0 * w
    return GammaSolution(
        amplitude=amplitude,
        c=c,
        w=w,
        centroid=centroid,
        variance=2.0 * w * w,
    )


def forward_convolution(
    driver_time: np.ndarray,
    driver_density: np.ndarray,
    dt: float,
    solution: GammaSolution,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kernel_min = math.floor(-5.0 / dt) * dt
    kernel_max = 60.0
    count = int(math.ceil((kernel_max - kernel_min) / dt))
    kernel_edges = kernel_min + np.arange(count + 1, dtype=float) * dt
    kernel_time = 0.5 * (kernel_edges[:-1] + kernel_edges[1:])

    # Use exact Gamma CDF differences to store the continuous MICA kernel as
    # bin-average densities.  This preserves its intended raw area A without
    # renormalizing it to unity or applying any empirical scale correction.
    left = np.maximum((kernel_edges[:-1] - solution.c) / solution.w, 0.0)
    right = np.maximum((kernel_edges[1:] - solution.c) / solution.w, 0.0)
    kernel_mass = solution.amplitude * (gammainc(2.0, right) - gammainc(2.0, left))
    kernel_density = kernel_mass / dt
    model_density = fftconvolve(driver_density, kernel_density, mode="full") * dt
    model_time = (
        driver_time[0]
        + kernel_time[0]
        + np.arange(model_density.size, dtype=float) * dt
    )
    return kernel_time, kernel_density, model_time, model_density


def interpolate(
    destination_time: np.ndarray,
    source_time: np.ndarray,
    source_density: np.ndarray,
) -> np.ndarray:
    return np.interp(
        destination_time,
        source_time,
        source_density,
        left=0.0,
        right=0.0,
    )


def relative_l2_full_domain(
    target_time: np.ndarray,
    target_density: np.ndarray,
    model_time: np.ndarray,
    model_density: np.ndarray,
    dt: float,
) -> float:
    target_on_model = interpolate(model_time, target_time, target_density)
    numerator = float(np.sum((model_density - target_on_model) ** 2) * dt)
    denominator = float(np.sum(target_on_model**2) * dt)
    return numerator / denominator


def main() -> None:
    args = parse_args()
    absolute_time, dt_days, raw = build_raw_theory(args.rin)
    theory_moments = {
        band: density_moments(absolute_time, density, dt_days)
        for band, density in raw.items()
    }
    driver_moments = theory_moments["UVW2"]

    solutions: dict[str, GammaSolution] = {}
    kernels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    models: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    summaries: list[dict[str, float | str | int]] = []

    for band in TARGET_BANDS:
        target_moments = theory_moments[band]
        solution = solve_gamma(driver_moments, target_moments)
        kernel_time, kernel_density, model_time, model_density = forward_convolution(
            absolute_time,
            raw["UVW2"],
            dt_days,
            solution,
        )
        kernel_moments = density_moments(kernel_time, kernel_density, dt_days)
        model_moments = density_moments(model_time, model_density, dt_days)
        negative_model_area = float(
            np.sum(model_density[model_time < 0.0]) * dt_days
        )
        solutions[band] = solution
        kernels[band] = (kernel_time, kernel_density)
        models[band] = (model_time, model_density)

        summaries.append(
            {
                "band": band,
                "wavelength_angstrom": BANDS[band],
                "driver_band": "UVW2",
                "response_mode": RESPONSE_MODE,
                "Rin_light_days": args.rin,
                "Rout_light_days": DISK_PARAMETERS["Rout"],
                "R0_light_days": DISK_PARAMETERS["R0"],
                "TB_K": DISK_PARAMETERS["TB"],
                "TF_K": DISK_PARAMETERS["TF"],
                "b": DISK_PARAMETERS["b"],
                "inc_deg": DISK_PARAMETERS["inc_deg"],
                "NR": NR,
                "Ntau": NTAU,
                "dt_days": dt_days,
                "target_M0_raw": target_moments.m0,
                "target_M1_raw_day": target_moments.m1,
                "target_M2_raw_day2": target_moments.m2,
                "target_centroid_days": target_moments.mean,
                "target_variance_days2": target_moments.variance,
                "driver_M0_raw": driver_moments.m0,
                "driver_centroid_days": driver_moments.mean,
                "driver_variance_days2": driver_moments.variance,
                "gamma_A_area": solution.amplitude,
                "gamma_c_days": solution.c,
                "gamma_w_days": solution.w,
                "gamma_centroid_days": solution.centroid,
                "gamma_variance_days2": solution.variance,
                "kernel_M0_numeric": kernel_moments.m0,
                "kernel_M1_numeric_day": kernel_moments.m1,
                "kernel_M2_numeric_day2": kernel_moments.m2,
                "model_M0_raw": model_moments.m0,
                "model_M1_raw_day": model_moments.m1,
                "model_M2_raw_day2": model_moments.m2,
                "model_centroid_days": model_moments.mean,
                "model_variance_days2": model_moments.variance,
                "relative_error_M0": model_moments.m0 / target_moments.m0 - 1.0,
                "relative_error_M1": model_moments.m1 / target_moments.m1 - 1.0,
                "relative_error_M2": model_moments.m2 / target_moments.m2 - 1.0,
                "negative_model_area_fraction": negative_model_area
                / model_moments.m0,
                "relative_l2_full_domain": relative_l2_full_domain(
                    absolute_time,
                    raw[band],
                    model_time,
                    model_density,
                    dt_days,
                ),
            }
        )

    # One common scale per quantity: no per-curve or per-panel normalization.
    max_absolute = max(float(np.max(curve)) for curve in raw.values())
    max_model = max(float(np.max(density)) for _, density in models.values())
    absolute_ylim = 1.08 * max(max_absolute, max_model) * ABSOLUTE_DISPLAY_SCALE
    max_kernel = max(float(np.max(density)) for _, density in kernels.values())
    kernel_ylim = 1.08 * max_kernel

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.2, 8.35),
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()
    twin_axes: list[plt.Axes] = []

    plot_time = np.arange(
        PLOT_MIN_DAYS,
        PLOT_MAX_DAYS + 0.25 * dt_days,
        dt_days / 2.0,
    )
    driver_plot = interpolate(plot_time, absolute_time, raw["UVW2"])

    for panel_index, (axis, band) in enumerate(zip(axes_flat, BANDS)):
        twin = axis.twinx()
        twin_axes.append(twin)
        axis.axvspan(PLOT_MIN_DAYS, 0.0, color="#F0F1F3", zorder=0)
        axis.axvline(0.0, color="#858A91", lw=0.8, zorder=1)
        axis.plot(
            plot_time,
            driver_plot * ABSOLUTE_DISPLAY_SCALE,
            color="#858A91",
            lw=1.35,
            ls=":",
            label="UVW2 absolute theory",
        )

        if band == "UVW2":
            axis.plot(
                plot_time,
                driver_plot * ABSOLUTE_DISPLAY_SCALE,
                color="#282D33",
                lw=2.45,
                label="UVW2 absolute theory",
            )
            twin.annotate(
                "",
                xy=(0.0, 0.92 * kernel_ylim),
                xytext=(0.0, 0.0),
                arrowprops={"arrowstyle": "-|>", "color": "#2F6F9F", "lw": 2.0},
            )
            twin.text(
                0.075,
                0.91,
                r"$K_{W2\leftarrow W2}=\delta(t)$, area = 1",
                transform=twin.transAxes,
                ha="left",
                va="top",
                fontsize=8.9,
                color="#2F6F9F",
            )
            driver = theory_moments["UVW2"]
            axis.text(
                0.97,
                0.93,
                rf"$10^6M_0={driver.m0*ABSOLUTE_DISPLAY_SCALE:.3f}$, "
                rf"$\mu={driver.mean:.3f}$ d",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.8,
                color="#3D4248",
            )
        else:
            target_plot = interpolate(plot_time, absolute_time, raw[band])
            model_time, model_density = models[band]
            model_plot = interpolate(plot_time, model_time, model_density)
            kernel_time, kernel_density = kernels[band]
            kernel_plot = interpolate(plot_time, kernel_time, kernel_density)
            solution = solutions[band]
            target = theory_moments[band]

            axis.plot(
                plot_time,
                target_plot * ABSOLUTE_DISPLAY_SCALE,
                color="#282D33",
                lw=2.35,
                label="target absolute theory",
            )
            axis.plot(
                plot_time,
                model_plot * ABSOLUTE_DISPLAY_SCALE,
                color="#C56B25",
                lw=1.85,
                ls="-.",
                label=r"moment-matched $\psi_{W2}*K$",
            )
            twin.plot(
                plot_time,
                kernel_plot,
                color="#2F6F9F",
                lw=1.9,
                ls="--",
                label="raw MICA Gamma relative kernel",
            )
            axis.text(
                0.97,
                0.94,
                (
                    rf"$A=M_0(K)={solution.amplitude:.3f}$, "
                    rf"$\tau_K={solution.centroid:.3f}$ d"
                    "\n"
                    rf"$c={solution.c:.3f}$, $w={solution.w:.3f}$ d; "
                    rf"$10^6M_0^{{target}}={target.m0*ABSOLUTE_DISPLAY_SCALE:.3f}$"
                ),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.35,
                color="#3D4248",
                linespacing=1.28,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
            )

        axis.set_title(f"{band} ({BANDS[band]:.0f} Å)", fontsize=11.6, pad=7)
        axis.set_xlim(PLOT_MIN_DAYS, PLOT_MAX_DAYS)
        axis.set_ylim(0.0, absolute_ylim)
        twin.set_ylim(0.0, kernel_ylim)
        axis.grid(axis="y", color="#D9DDE1", lw=0.65, alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        twin.spines["top"].set_visible(False)
        twin.spines["left"].set_visible(False)
        twin.spines["right"].set_color("#2F6F9F")
        twin.tick_params(axis="y", colors="#2F6F9F", length=3.2, width=0.8)
        axis.tick_params(direction="out", length=3.5, width=0.8)
        if panel_index % 3 != 2:
            twin.tick_params(right=False, labelright=False)
            twin.spines["right"].set_visible(False)
        if panel_index < 3:
            axis.tick_params(labelbottom=False)

    for axis in axes[-1, :]:
        axis.set_xlabel("Delay coordinate [day]")
    figure.supylabel(
        r"Raw absolute response [$10^{-6}$ common units day$^{-1}$]",
        x=0.017,
        fontsize=10.5,
    )
    figure.text(
        0.985,
        0.50,
        r"Raw relative kernel $K_{\lambda\leftarrow W2}$ [day$^{-1}$]",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#2F6F9F",
    )

    legend_handles = [
        Line2D([0], [0], color="#858A91", lw=1.35, ls=":", label="UVW2 absolute theory"),
        Line2D([0], [0], color="#282D33", lw=2.35, label="target absolute theory"),
        Line2D([0], [0], color="#C56B25", lw=1.85, ls="-.", label="moment-matched absolute reconstruction"),
        Line2D([0], [0], color="#2F6F9F", lw=1.9, ls="--", label="raw MICA Gamma relative kernel"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.885),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        handlelength=3.0,
        columnspacing=1.5,
    )
    figure.suptitle(
        "Mrk 817 raw theoretical TFs and moment-matched MICA Gamma kernels",
        fontsize=16.0,
        y=0.987,
    )
    figure.text(
        0.5,
        0.945,
        (
            "finite-difference response; common raw scale; "
            f"$R_{{\\rm in}}={args.rin:g}$ light-day; "
            f"$N_R={NR}$, $N_\\tau={NTAU}$; no area or peak normalization"
        ),
        ha="center",
        va="top",
        fontsize=9.7,
        color="#535961",
    )
    figure.text(
        0.5,
        0.017,
        (
            "Left axes share one raw absolute-response scale. Right axes share one raw relative-kernel scale; "
            "for the MICA Gamma, its area is A. Full-support moments; displayed window is -0.8 to 6 d."
        ),
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="#535961",
    )
    figure.subplots_adjust(
        left=0.080,
        right=0.925,
        top=0.805,
        bottom=0.09,
        hspace=0.24,
        wspace=0.10,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output.with_suffix(".png")
    pdf_path = args.output.with_suffix(".pdf")
    summary_path = args.output.with_name(args.output.name + "_moments.csv")
    curves_path = args.output.with_name(args.output.name + "_curves.csv")
    figure.savefig(png_path, dpi=250, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    curve_rows: list[dict[str, float | str]] = []
    for band in TARGET_BANDS:
        target_plot = interpolate(plot_time, absolute_time, raw[band])
        model_time, model_density = models[band]
        model_plot = interpolate(plot_time, model_time, model_density)
        kernel_time, kernel_density = kernels[band]
        kernel_plot = interpolate(plot_time, kernel_time, kernel_density)
        for index, delay in enumerate(plot_time):
            curve_rows.append(
                {
                    "band": band,
                    "delay_days": float(delay),
                    "uvw2_absolute_raw_per_day": float(driver_plot[index]),
                    "target_absolute_raw_per_day": float(target_plot[index]),
                    "reconstructed_absolute_raw_per_day": float(model_plot[index]),
                    "relative_gamma_raw_per_day": float(kernel_plot[index]),
                }
            )
    with curves_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    frame = pd.DataFrame(summaries)
    print(f"dt_days={dt_days:.10f}")
    print(
        "max_abs_relative_moment_error=",
        float(
            frame[["relative_error_M0", "relative_error_M1", "relative_error_M2"]]
            .abs()
            .to_numpy()
            .max()
        ),
    )
    print(png_path)
    print(pdf_path)
    print(summary_path)
    print(curves_path)


if __name__ == "__main__":
    main()
