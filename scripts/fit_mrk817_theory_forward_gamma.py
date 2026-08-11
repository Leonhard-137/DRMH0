#!/usr/bin/env python3
"""Fit positive shifted-k=2 Gamma kernels between theoretical Swift responses.

This is a deterministic structural diagnostic, not an MCMC fit.  For each
target Swift band, it asks whether

    psi_target(t) ~= A * [psi_UVW2 * K_gamma(c, w)](t)

where K_gamma is a unit-area shifted Gamma density with shape k=2.  Two disk
response definitions from ``test/disk_transfer/disktf_diff.py`` are run with
the same physical parameters:

* ``finite_difference``: B_nu(T_bright) - B_nu(T_faint)
* ``linearized``: dB_nu/dT(T_mid) * (T_bright - T_faint)

The main fit includes the complete linear-convolution output.  The target is
zero outside its physical time grid, so model response leaking to t < 0 or
beyond the target support is penalized.  A positive-window-only fit is saved
as a diagnostic because omitting leaked mass can bias c strongly negative.

No existing library file is modified and no sampler is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import fftconvolve


REPO_ROOT = Path(__file__).resolve().parents[1]
THEORY_DIR = REPO_ROOT / "test" / "disk_transfer"
if str(THEORY_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_DIR))

from disktf_diff import DiskConfig, disk_transfer_function  # noqa: E402


BANDS = {
    "UVW2": 1928.0,
    "UVM2": 2246.0,
    "UVW1": 2600.0,
    "U": 3465.0,
    "B": 4392.0,
    "V": 5468.0,
}
TARGET_BANDS = tuple(name for name in BANDS if name != "UVW2")
RESPONSE_MODES = ("finite_difference", "linearized")

# Both sets retain the DiskConfig defaults for Rin, Rout, R0, b, and NR.
# The Mrk 817 temperatures and inclination are from Section 5 of
# Lewin et al. (2024, ApJ 974:271).
PARAMETER_SETS = {
    "library_default": {
        "Rin_light_days": 0.1,
        "Rout_light_days": 30.0,
        "R0_light_days": 1.0,
        "T_bright_at_R0_K": 29000.0,
        "T_faint_at_R0_K": 26500.0,
        "temperature_power_law_index": 0.75,
        "inclination_deg": 30.0,
        "NR": 800,
        "source": "DiskConfig defaults in disktf_diff.py",
    },
    "mrk817_lewin2024": {
        "Rin_light_days": 0.1,
        "Rout_light_days": 30.0,
        "R0_light_days": 1.0,
        "T_bright_at_R0_K": 11400.0,
        "T_faint_at_R0_K": 5900.0,
        "temperature_power_law_index": 0.75,
        "inclination_deg": 19.0,
        "NR": 800,
        "source": "Lewin et al. (2024, ApJ 974:271), Section 5",
    },
}

# Wide enough to contain both the moment estimates and the expected L2 fits.
C_BOUNDS_DAYS = (-10.0, 3.0)
W_BOUNDS_DAYS = (0.02, 8.0)
KERNEL_MAX_DAYS = 120.0
KERNEL_MIN_CAPTURED_AREA = 0.9995


@dataclass(frozen=True)
class DensityStats:
    area: float
    mean: float
    variance: float
    std: float


@dataclass(frozen=True)
class MomentSolution:
    possible: bool
    delta_mean: float
    delta_variance: float
    c: float
    w: float
    tau: float
    mode: float


@dataclass
class FitEvaluation:
    objective: float
    amplitude: float
    kernel: np.ndarray
    model: np.ndarray
    kernel_area_raw: float
    kernel_stats: DensityStats
    model_stats: DensityStats
    negative_mass: float
    post_target_mass: float
    target_window_mass: float
    centroid_addition_error: float
    variance_addition_error: float


@dataclass
class FitRecord:
    parameter_set: str
    response_mode: str
    fit_domain: str
    driver_band: str
    driver_wavelength_angstrom: float
    target_band: str
    target_wavelength_angstrom: float
    dt_days: float
    ntau: int
    c_days: float
    w_days: float
    two_w_days: float
    gamma_centroid_days: float
    gamma_mode_days: float
    amplitude_A: float
    relative_l2: float
    negative_mass_fraction: float
    scaled_negative_area: float
    post_target_mass_fraction: float
    target_window_mass_fraction: float
    kernel_area_raw_on_grid: float
    kernel_grid_mean_days: float
    kernel_grid_variance_days2: float
    convolved_model_area: float
    convolved_model_mean_days: float
    convolved_model_variance_days2: float
    centroid_addition_error_days: float
    variance_addition_error_days2: float
    optimizer_success: bool
    optimizer_message: str
    at_search_boundary: bool
    driver_mean_days: float
    driver_variance_days2: float
    target_mean_days: float
    target_variance_days2: float
    moment_possible: bool
    moment_delta_mean_days: float
    moment_delta_variance_days2: float
    moment_c_days: float
    moment_w_days: float
    moment_two_w_days: float
    moment_gamma_centroid_days: float
    moment_gamma_mode_days: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "Mrk817"
            / "results"
            / "tf_comparison"
            / "theory_forward_gamma"
        ),
        help="Output directory.",
    )
    parser.add_argument(
        "--ntau",
        type=int,
        default=4000,
        help=(
            "Number of theoretical delay bins. The DiskConfig default is "
            "2000; 4000 is used here to reduce shifted-Gamma discretization "
            "error without changing physical parameters."
        ),
    )
    parser.add_argument(
        "--de-maxiter",
        type=int,
        default=45,
        help="Differential-evolution generations per two-parameter fit.",
    )
    return parser.parse_args()


def normalize_density(y: np.ndarray, dt: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if np.any(~np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("Density must be finite and non-negative.")
    area = float(np.sum(y) * dt)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("Density has non-positive area.")
    return y / area


def density_stats(t: np.ndarray, y: np.ndarray, dt: float) -> DensityStats:
    area = float(np.sum(y) * dt)
    if not np.isfinite(area) or area <= 0.0:
        return DensityStats(np.nan, np.nan, np.nan, np.nan)
    mean = float(np.sum(t * y) * dt / area)
    variance = float(np.sum((t - mean) ** 2 * y) * dt / area)
    variance = max(0.0, variance)
    return DensityStats(area, mean, variance, math.sqrt(variance))


def moment_solution(driver: DensityStats, target: DensityStats) -> MomentSolution:
    delta_mean = target.mean - driver.mean
    delta_variance = target.variance - driver.variance
    if delta_variance < 0.0:
        return MomentSolution(
            False,
            delta_mean,
            delta_variance,
            np.nan,
            np.nan,
            delta_mean,
            np.nan,
        )
    w = math.sqrt(delta_variance / 2.0)
    c = delta_mean - 2.0 * w
    return MomentSolution(
        True,
        delta_mean,
        delta_variance,
        c,
        w,
        c + 2.0 * w,
        c + w,
    )


def gamma_k2_on_grid(s: np.ndarray, c: float, w: float) -> np.ndarray:
    """Continuous unit-area k=2 Gamma sampled on a signed delay grid."""
    response = np.zeros_like(s)
    u = s - c
    mask = u >= 0.0
    x = u[mask] / w
    response[mask] = x * np.exp(-x) / w
    return response


def build_theory_curves(
    parameter_set: str,
    response_mode: str,
    ntau: int,
) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
    physical_parameters = PARAMETER_SETS[parameter_set]
    curves: dict[str, np.ndarray] = {}
    common_time: np.ndarray | None = None
    common_dt: float | None = None

    for band, wavelength in BANDS.items():
        config = DiskConfig(
            Rin=physical_parameters["Rin_light_days"],
            Rout=physical_parameters["Rout_light_days"],
            R0=physical_parameters["R0_light_days"],
            TB=physical_parameters["T_bright_at_R0_K"],
            TF=physical_parameters["T_faint_at_R0_K"],
            b=physical_parameters["temperature_power_law_index"],
            inc_deg=physical_parameters["inclination_deg"],
            lam_A=wavelength,
            NR=int(physical_parameters["NR"]),
            Ntau=ntau,
            response_mode=response_mode,
        )
        time_days, response = disk_transfer_function(config)
        dt = float(np.median(np.diff(time_days)))
        response = normalize_density(response, dt)

        if common_time is None:
            common_time = np.asarray(time_days, dtype=float)
            common_dt = dt
        else:
            if not np.allclose(time_days, common_time, rtol=0.0, atol=1e-12):
                raise RuntimeError("Theoretical bands do not share one time grid.")
            if not math.isclose(dt, common_dt, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError("Theoretical bands do not share one dt.")
        curves[band] = response

    assert common_time is not None and common_dt is not None
    return common_time, common_dt, curves


class ForwardGammaProblem:
    def __init__(
        self,
        time_days: np.ndarray,
        driver: np.ndarray,
        target: np.ndarray,
        dt: float,
    ) -> None:
        self.time_days = np.asarray(time_days, dtype=float)
        self.driver = np.asarray(driver, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.dt = float(dt)
        self.driver_stats = density_stats(self.time_days, self.driver, self.dt)
        self.target_stats = density_stats(self.time_days, self.target, self.dt)

        n_negative = int(math.ceil(abs(C_BOUNDS_DAYS[0]) / self.dt))
        self.kernel_start = -n_negative * self.dt
        n_positive = int(math.ceil(KERNEL_MAX_DAYS / self.dt))
        self.kernel_time = self.kernel_start + np.arange(
            n_negative + n_positive + 1,
            dtype=float,
        ) * self.dt

        self.output_time = (
            self.time_days[0]
            + self.kernel_start
            + np.arange(
                self.driver.size + self.kernel_time.size - 1,
                dtype=float,
            )
            * self.dt
        )
        self.target_extended = np.interp(
            self.output_time,
            self.time_days,
            self.target,
            left=0.0,
            right=0.0,
        )
        self.target_energy = float(np.sum(self.target**2) * self.dt)
        self.domain_masks = {
            "full": np.ones(self.output_time.size, dtype=bool),
            "positive_window": (
                (self.output_time >= 0.0)
                & (self.output_time <= self.time_days[-1])
            ),
        }

    def evaluate(self, c: float, log_w: float, domain: str) -> FitEvaluation | None:
        w = math.exp(log_w)
        kernel_raw = gamma_k2_on_grid(self.kernel_time, c, w)
        kernel_area_raw = float(np.sum(kernel_raw) * self.dt)
        if (
            not np.isfinite(kernel_area_raw)
            or kernel_area_raw < KERNEL_MIN_CAPTURED_AREA
        ):
            return None
        kernel = kernel_raw / kernel_area_raw
        model = fftconvolve(self.driver, kernel, mode="full") * self.dt

        mask = self.domain_masks[domain]
        model_fit = model[mask]
        target_fit = self.target_extended[mask]
        denominator = float(np.dot(model_fit, model_fit))
        if not np.isfinite(denominator) or denominator <= 0.0:
            return None
        amplitude = max(0.0, float(np.dot(target_fit, model_fit) / denominator))
        residual = target_fit - amplitude * model_fit
        objective = float(
            np.sum(residual**2) * self.dt / self.target_energy
        )

        kernel_stats = density_stats(self.kernel_time, kernel, self.dt)
        model_stats = density_stats(self.output_time, model, self.dt)
        negative = self.output_time < 0.0
        post_target = self.output_time > self.time_days[-1]
        in_target = (
            (self.output_time >= self.time_days[0])
            & (self.output_time <= self.time_days[-1])
        )
        negative_mass = float(np.sum(model[negative]) * self.dt)
        post_target_mass = float(np.sum(model[post_target]) * self.dt)
        target_window_mass = float(np.sum(model[in_target]) * self.dt)

        return FitEvaluation(
            objective=objective,
            amplitude=amplitude,
            kernel=kernel,
            model=model,
            kernel_area_raw=kernel_area_raw,
            kernel_stats=kernel_stats,
            model_stats=model_stats,
            negative_mass=negative_mass,
            post_target_mass=post_target_mass,
            target_window_mass=target_window_mass,
            centroid_addition_error=(
                model_stats.mean
                - self.driver_stats.mean
                - kernel_stats.mean
            ),
            variance_addition_error=(
                model_stats.variance
                - self.driver_stats.variance
                - kernel_stats.variance
            ),
        )


def fit_problem(
    problem: ForwardGammaProblem,
    domain: str,
    seed: int,
    de_maxiter: int,
) -> tuple[np.ndarray, FitEvaluation, bool, str]:
    log_w_bounds = (math.log(W_BOUNDS_DAYS[0]), math.log(W_BOUNDS_DAYS[1]))
    bounds = (C_BOUNDS_DAYS, log_w_bounds)

    def objective(parameters: np.ndarray) -> float:
        evaluation = problem.evaluate(float(parameters[0]), float(parameters[1]), domain)
        return 1.0e12 if evaluation is None else evaluation.objective

    global_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        popsize=10,
        maxiter=de_maxiter,
        tol=1.0e-8,
        atol=1.0e-11,
        polish=False,
        workers=1,
        updating="immediate",
    )
    local_result = minimize(
        objective,
        global_result.x,
        method="Powell",
        bounds=bounds,
        options={
            "xtol": 1.0e-8,
            "ftol": 1.0e-11,
            "maxiter": 1000,
        },
    )
    candidates = [global_result, local_result]
    best = min(candidates, key=lambda result: float(result.fun))
    parameters = np.asarray(best.x, dtype=float)
    evaluation = problem.evaluate(float(parameters[0]), float(parameters[1]), domain)
    if evaluation is None:
        raise RuntimeError("Optimizer returned an invalid Gamma kernel.")
    success = bool(global_result.success or local_result.success)
    message = f"DE: {global_result.message}; local: {local_result.message}"
    return parameters, evaluation, success, message


def is_at_boundary(c: float, w: float, dt: float) -> bool:
    c_margin = max(2.0 * dt, 1.0e-3)
    log_margin = 0.02
    return bool(
        c <= C_BOUNDS_DAYS[0] + c_margin
        or c >= C_BOUNDS_DAYS[1] - c_margin
        or math.log(w) <= math.log(W_BOUNDS_DAYS[0]) + log_margin
        or math.log(w) >= math.log(W_BOUNDS_DAYS[1]) - log_margin
    )


def record_from_fit(
    parameter_set: str,
    response_mode: str,
    domain: str,
    target_band: str,
    dt: float,
    ntau: int,
    parameters: np.ndarray,
    evaluation: FitEvaluation,
    optimizer_success: bool,
    optimizer_message: str,
    problem: ForwardGammaProblem,
    moments: MomentSolution,
) -> FitRecord:
    c = float(parameters[0])
    w = math.exp(float(parameters[1]))
    return FitRecord(
        parameter_set=parameter_set,
        response_mode=response_mode,
        fit_domain=domain,
        driver_band="UVW2",
        driver_wavelength_angstrom=BANDS["UVW2"],
        target_band=target_band,
        target_wavelength_angstrom=BANDS[target_band],
        dt_days=dt,
        ntau=ntau,
        c_days=c,
        w_days=w,
        two_w_days=2.0 * w,
        gamma_centroid_days=c + 2.0 * w,
        gamma_mode_days=c + w,
        amplitude_A=evaluation.amplitude,
        relative_l2=evaluation.objective,
        negative_mass_fraction=evaluation.negative_mass,
        scaled_negative_area=evaluation.amplitude * evaluation.negative_mass,
        post_target_mass_fraction=evaluation.post_target_mass,
        target_window_mass_fraction=evaluation.target_window_mass,
        kernel_area_raw_on_grid=evaluation.kernel_area_raw,
        kernel_grid_mean_days=evaluation.kernel_stats.mean,
        kernel_grid_variance_days2=evaluation.kernel_stats.variance,
        convolved_model_area=evaluation.model_stats.area,
        convolved_model_mean_days=evaluation.model_stats.mean,
        convolved_model_variance_days2=evaluation.model_stats.variance,
        centroid_addition_error_days=evaluation.centroid_addition_error,
        variance_addition_error_days2=evaluation.variance_addition_error,
        optimizer_success=optimizer_success,
        optimizer_message=optimizer_message,
        at_search_boundary=is_at_boundary(c, w, dt),
        driver_mean_days=problem.driver_stats.mean,
        driver_variance_days2=problem.driver_stats.variance,
        target_mean_days=problem.target_stats.mean,
        target_variance_days2=problem.target_stats.variance,
        moment_possible=moments.possible,
        moment_delta_mean_days=moments.delta_mean,
        moment_delta_variance_days2=moments.delta_variance,
        moment_c_days=moments.c,
        moment_w_days=moments.w,
        moment_two_w_days=(2.0 * moments.w if moments.possible else np.nan),
        moment_gamma_centroid_days=moments.tau,
        moment_gamma_mode_days=moments.mode,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_absolute_transfer_functions(
    parameter_set: str,
    theory_by_key: dict[
        tuple[str, str], tuple[np.ndarray, float, dict[str, np.ndarray]]
    ],
    output_path: Path,
) -> None:
    colors = {
        "UVW2": "#222222",
        "UVM2": "#4C78A8",
        "UVW1": "#59A14F",
        "U": "#F28E2B",
        "B": "#E15759",
        "V": "#B279A2",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), sharex=True, sharey=True)
    for axis, response_mode in zip(axes, RESPONSE_MODES):
        time_days, _, curves = theory_by_key[(parameter_set, response_mode)]
        for band in BANDS:
            axis.plot(
                time_days,
                curves[band],
                color=colors[band],
                lw=1.8,
                label=f"{band} ({BANDS[band]:.0f} A)",
            )
        axis.set_title(response_mode.replace("_", " "))
        axis.set_xlim(0.0, 25.0)
        axis.set_xlabel("Absolute delay [day]")
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Unit-area theoretical response [day$^{-1}$]")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle(
        "Swift theoretical disk transfer functions used in forward fits\n"
        + parameter_set.replace("_", " ")
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def plot_fit_panels(
    parameter_set: str,
    problems: dict[tuple[str, str, str], ForwardGammaProblem],
    evaluations: dict[tuple[str, str, str, str], FitEvaluation],
    records: dict[tuple[str, str, str, str], FitRecord],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(TARGET_BANDS),
        len(RESPONSE_MODES),
        figsize=(13.0, 15.2),
        sharex=True,
    )
    target_color = "#222222"
    full_color = "#D17A00"
    diagnostic_color = "#3977A8"

    for column, response_mode in enumerate(RESPONSE_MODES):
        for row, target_band in enumerate(TARGET_BANDS):
            axis = axes[row, column]
            problem = problems[(parameter_set, response_mode, target_band)]
            full_eval = evaluations[
                (parameter_set, response_mode, target_band, "full")
            ]
            positive_eval = evaluations[
                (parameter_set, response_mode, target_band, "positive_window")
            ]
            full_record = records[
                (parameter_set, response_mode, target_band, "full")
            ]
            positive_record = records[
                (parameter_set, response_mode, target_band, "positive_window")
            ]

            axis.axvspan(-5.0, 0.0, color="#ECECEC", zorder=0)
            axis.axvline(0.0, color="#777777", lw=0.8, ls=":")
            axis.plot(
                problem.time_days,
                problem.target,
                color=target_color,
                lw=2.0,
                label="Target theory TF",
            )
            axis.plot(
                problem.output_time,
                full_eval.amplitude * full_eval.model,
                color=full_color,
                lw=1.8,
                label="Full-domain fit",
            )
            axis.plot(
                problem.output_time,
                positive_eval.amplitude * positive_eval.model,
                color=diagnostic_color,
                lw=1.4,
                ls="--",
                label="Positive-window diagnostic",
            )
            axis.set_xlim(-4.5, 30.0)
            axis.set_ylim(bottom=0.0)
            axis.grid(axis="y", alpha=0.22)
            axis.spines[["top", "right"]].set_visible(False)
            axis.text(
                0.985,
                0.95,
                (
                    f"full: c={full_record.c_days:.2f}, "
                    f"w={full_record.w_days:.2f}, "
                    f"tau={full_record.gamma_centroid_days:.2f}\n"
                    f"pos:  c={positive_record.c_days:.2f}, "
                    f"w={positive_record.w_days:.2f}, "
                    f"tau={positive_record.gamma_centroid_days:.2f}\n"
                    f"full negative mass={100*full_record.negative_mass_fraction:.1f}%"
                ),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )
            if column == 0:
                axis.set_ylabel(f"{target_band}\nResponse [day$^{{-1}}$]")
            if row == 0:
                axis.set_title(response_mode.replace("_", " "), fontsize=12)
            if row == len(TARGET_BANDS) - 1:
                axis.set_xlabel("Absolute delay [day]")

    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    figure.suptitle(
        "Forward shifted-Gamma fits: UVW2 theoretical response as driver\n"
        + parameter_set.replace("_", " "),
        fontsize=15,
        y=0.998,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def plot_parameter_comparison(
    parameter_set: str,
    records: dict[tuple[str, str, str, str], FitRecord],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 8.2), sharex=True)
    x = np.arange(len(TARGET_BANDS), dtype=float)
    xlabels = list(TARGET_BANDS)
    mode_colors = {
        "finite_difference": "#D17A00",
        "linearized": "#3977A8",
    }

    for response_mode in RESPONSE_MODES:
        color = mode_colors[response_mode]
        full = [
            records[(parameter_set, response_mode, band, "full")]
            for band in TARGET_BANDS
        ]
        positive = [
            records[(parameter_set, response_mode, band, "positive_window")]
            for band in TARGET_BANDS
        ]
        label = response_mode.replace("_", " ")

        axes[0, 0].plot(
            x, [record.c_days for record in full], "o-", color=color, label=label
        )
        axes[0, 0].plot(
            x,
            [record.c_days for record in positive],
            "s--",
            color=color,
            alpha=0.72,
        )
        axes[0, 0].plot(
            x,
            [record.moment_c_days for record in full],
            "^:",
            color=color,
            alpha=0.8,
        )
        axes[0, 1].plot(
            x,
            [record.gamma_centroid_days for record in full],
            "o-",
            color=color,
        )
        axes[0, 1].plot(
            x,
            [record.gamma_centroid_days for record in positive],
            "s--",
            color=color,
            alpha=0.72,
        )
        axes[0, 1].plot(
            x,
            [record.moment_gamma_centroid_days for record in full],
            "^:",
            color=color,
            alpha=0.8,
        )
        axes[0, 2].plot(
            x, [record.two_w_days for record in full], "o-", color=color
        )
        axes[0, 2].plot(
            x,
            [record.two_w_days for record in positive],
            "s--",
            color=color,
            alpha=0.72,
        )
        axes[0, 2].plot(
            x,
            [record.moment_two_w_days for record in full],
            "^:",
            color=color,
            alpha=0.8,
        )
        axes[1, 0].plot(
            x, [record.amplitude_A for record in full], "o-", color=color
        )
        axes[1, 0].plot(
            x,
            [record.amplitude_A for record in positive],
            "s--",
            color=color,
            alpha=0.72,
        )
        axes[1, 1].plot(
            x,
            [100.0 * record.negative_mass_fraction for record in full],
            "o-",
            color=color,
        )
        axes[1, 1].plot(
            x,
            [100.0 * record.negative_mass_fraction for record in positive],
            "s--",
            color=color,
            alpha=0.72,
        )
        axes[1, 2].plot(
            x, [record.relative_l2 for record in full], "o-", color=color
        )
        axes[1, 2].plot(
            x,
            [record.relative_l2 for record in positive],
            "s--",
            color=color,
            alpha=0.72,
        )

    axes[0, 0].axhline(0.0, color="#777777", lw=0.8)
    axes[0, 0].set_ylabel("Gamma onset c [day]")
    axes[0, 1].set_ylabel("Gamma centroid c+2w [day]")
    axes[0, 2].set_ylabel("2w [day]")
    axes[1, 0].axhline(1.0, color="#777777", lw=0.8, ls=":")
    axes[1, 0].set_ylabel("Analytic amplitude A")
    axes[1, 1].set_ylabel("Model mass at t<0 [%]")
    axes[1, 2].set_ylabel("Relative integrated L2")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")

    for axis in axes.flat:
        axis.set_xticks(x, xlabels)
        axis.grid(alpha=0.24)
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.01,
        "Markers: circle/solid = full domain; square/dashed = positive-window diagnostic; triangle/dotted = moment match",
        ha="center",
        fontsize=9,
    )
    figure.suptitle(
        "Forward-Gamma parameters and window sensitivity\n"
        + parameter_set.replace("_", " "),
        fontsize=15,
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.97))
    figure.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.ntau < 500:
        raise ValueError("--ntau must be at least 500.")
    args.output.mkdir(parents=True, exist_ok=True)

    theory_by_key: dict[
        tuple[str, str], tuple[np.ndarray, float, dict[str, np.ndarray]]
    ] = {}
    problems: dict[tuple[str, str, str], ForwardGammaProblem] = {}
    evaluations: dict[tuple[str, str, str, str], FitEvaluation] = {}
    records: dict[tuple[str, str, str, str], FitRecord] = {}

    for set_index, parameter_set in enumerate(PARAMETER_SETS):
        for mode_index, response_mode in enumerate(RESPONSE_MODES):
            print(f"Generating theory: {parameter_set} / {response_mode}")
            time_days, dt, curves = build_theory_curves(
                parameter_set,
                response_mode,
                args.ntau,
            )
            theory_by_key[(parameter_set, response_mode)] = (
                time_days,
                dt,
                curves,
            )

            for band_index, target_band in enumerate(TARGET_BANDS):
                problem = ForwardGammaProblem(
                    time_days,
                    curves["UVW2"],
                    curves[target_band],
                    dt,
                )
                problems[(parameter_set, response_mode, target_band)] = problem
                moments = moment_solution(problem.driver_stats, problem.target_stats)

                for domain_index, domain in enumerate(("full", "positive_window")):
                    seed = (
                        817000
                        + set_index * 10000
                        + mode_index * 1000
                        + band_index * 10
                        + domain_index
                    )
                    print(
                        f"Fitting {parameter_set:17s} {response_mode:17s} "
                        f"UVW2 -> {target_band:4s} [{domain}]"
                    )
                    parameters, evaluation, success, message = fit_problem(
                        problem,
                        domain,
                        seed,
                        args.de_maxiter,
                    )
                    evaluation_key = (
                        parameter_set,
                        response_mode,
                        target_band,
                        domain,
                    )
                    evaluations[evaluation_key] = evaluation
                    records[evaluation_key] = record_from_fit(
                        parameter_set,
                        response_mode,
                        domain,
                        target_band,
                        dt,
                        args.ntau,
                        parameters,
                        evaluation,
                        success,
                        message,
                        problem,
                        moments,
                    )

    ordered_records = [
        records[(parameter_set, response_mode, target_band, domain)]
        for parameter_set in PARAMETER_SETS
        for response_mode in RESPONSE_MODES
        for target_band in TARGET_BANDS
        for domain in ("full", "positive_window")
    ]
    summary_rows = [asdict(record) for record in ordered_records]
    summary_path = args.output / "theory_forward_gamma_summary.csv"
    write_csv(summary_path, summary_rows)

    absolute_rows: list[dict] = []
    for parameter_set in PARAMETER_SETS:
        for response_mode in RESPONSE_MODES:
            time_days, _, curves = theory_by_key[(parameter_set, response_mode)]
            for band in BANDS:
                for time_value, density in zip(time_days, curves[band]):
                    absolute_rows.append(
                        {
                            "parameter_set": parameter_set,
                            "response_mode": response_mode,
                            "band": band,
                            "wavelength_angstrom": BANDS[band],
                            "time_days": time_value,
                            "density_per_day": density,
                        }
                    )
    absolute_path = args.output / "theory_absolute_transfer_functions.csv"
    write_csv(absolute_path, absolute_rows)

    curve_rows: list[dict] = []
    for parameter_set in PARAMETER_SETS:
        for response_mode in RESPONSE_MODES:
            for target_band in TARGET_BANDS:
                problem = problems[(parameter_set, response_mode, target_band)]
                full_eval = evaluations[
                    (parameter_set, response_mode, target_band, "full")
                ]
                positive_eval = evaluations[
                    (
                        parameter_set,
                        response_mode,
                        target_band,
                        "positive_window",
                    )
                ]
                save_mask = (
                    (problem.output_time >= -5.0)
                    & (problem.output_time <= problem.time_days[-1])
                )
                for index in np.flatnonzero(save_mask):
                    curve_rows.append(
                        {
                            "parameter_set": parameter_set,
                            "response_mode": response_mode,
                            "driver_band": "UVW2",
                            "target_band": target_band,
                            "time_days": problem.output_time[index],
                            "target_density_per_day": problem.target_extended[index],
                            "full_model_density_per_day": (
                                full_eval.amplitude * full_eval.model[index]
                            ),
                            "positive_window_model_density_per_day": (
                                positive_eval.amplitude * positive_eval.model[index]
                            ),
                        }
                    )
    curves_path = args.output / "theory_forward_gamma_curves.csv"
    write_csv(curves_path, curve_rows)

    config = {
        "purpose": "Deterministic forward-convolution structural diagnostic",
        "source_file": str(THEORY_DIR / "disktf_diff.py"),
        "driver_band": "UVW2",
        "bands_angstrom": BANDS,
        "response_modes": RESPONSE_MODES,
        "parameter_sets": PARAMETER_SETS,
        "numerical_parameters": {
            "Ntau": args.ntau,
            "c_bounds_days": C_BOUNDS_DAYS,
            "w_bounds_days": W_BOUNDS_DAYS,
            "kernel_max_days": KERNEL_MAX_DAYS,
            "kernel_min_captured_area": KERNEL_MIN_CAPTURED_AREA,
            "de_maxiter": args.de_maxiter,
            "gamma_shape_k": 2.0,
            "amplitude_treatment": "analytic non-negative least squares",
            "main_fit_domain": "complete linear-convolution output",
            "diagnostic_fit_domain": "0 <= t <= theoretical target maximum",
        },
        "definitions": {
            "finite_difference": "B_nu(T_bright) - B_nu(T_faint)",
            "linearized": "dB_nu/dT(T_mid) * (T_bright - T_faint)",
            "gamma": "H(t-c)*(t-c)*exp[-(t-c)/w]/w^2",
        },
    }
    config_path = args.output / "theory_forward_gamma_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    max_model_area_error = max(
        abs(record.convolved_model_area - 1.0) for record in ordered_records
    )
    max_centroid_addition_error = max(
        abs(record.centroid_addition_error_days) for record in ordered_records
    )
    max_variance_addition_error = max(
        abs(record.variance_addition_error_days2) for record in ordered_records
    )
    min_kernel_area_raw = min(
        record.kernel_area_raw_on_grid for record in ordered_records
    )
    qa = {
        "all_optimizers_report_success": all(
            record.optimizer_success for record in ordered_records
        ),
        "any_solution_at_search_boundary": any(
            record.at_search_boundary for record in ordered_records
        ),
        "all_moment_solutions_possible": all(
            record.moment_possible for record in ordered_records
        ),
        "max_convolved_model_area_error": max_model_area_error,
        "max_centroid_addition_error_days": max_centroid_addition_error,
        "max_variance_addition_error_days2": max_variance_addition_error,
        "minimum_raw_kernel_area_on_grid": min_kernel_area_raw,
        "checks": {
            "model_area_tolerance": max_model_area_error < 1.0e-10,
            "centroid_addition_tolerance": max_centroid_addition_error < 1.0e-9,
            "variance_addition_tolerance": max_variance_addition_error < 1.0e-8,
            "kernel_capture_tolerance": (
                min_kernel_area_raw >= KERNEL_MIN_CAPTURED_AREA
            ),
        },
    }
    qa_path = args.output / "theory_forward_gamma_qa.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")

    figure_paths: list[Path] = []
    for parameter_set in PARAMETER_SETS:
        absolute_figure_path = (
            args.output / f"theory_absolute_transfer_functions_{parameter_set}.png"
        )
        fit_figure_path = (
            args.output / f"theory_forward_gamma_fits_{parameter_set}.png"
        )
        parameter_figure_path = (
            args.output / f"theory_forward_gamma_parameters_{parameter_set}.png"
        )
        plot_absolute_transfer_functions(
            parameter_set,
            theory_by_key,
            absolute_figure_path,
        )
        plot_fit_panels(
            parameter_set,
            problems,
            evaluations,
            records,
            fit_figure_path,
        )
        plot_parameter_comparison(
            parameter_set,
            records,
            parameter_figure_path,
        )
        figure_paths.extend(
            [absolute_figure_path, fit_figure_path, parameter_figure_path]
        )

    print("\nMain full-domain results")
    print("set mode band c[d] w[d] tau[d] mode[d] A neg_mass relL2")
    for parameter_set in PARAMETER_SETS:
        for response_mode in RESPONSE_MODES:
            for target_band in TARGET_BANDS:
                record = records[
                    (parameter_set, response_mode, target_band, "full")
                ]
                print(
                    f"{parameter_set:17s} {response_mode:17s} {target_band:4s} "
                    f"{record.c_days:8.4f} {record.w_days:8.4f} "
                    f"{record.gamma_centroid_days:8.4f} "
                    f"{record.gamma_mode_days:8.4f} "
                    f"{record.amplitude_A:7.4f} "
                    f"{record.negative_mass_fraction:8.4f} "
                    f"{record.relative_l2:9.5f}"
                )
    print(f"\nSummary: {summary_path}")
    print(f"Curves:  {curves_path}")
    print(f"Figures: {len(figure_paths)} PNG files in {args.output}")
    print(f"QA:      {qa_path}")


if __name__ == "__main__":
    main()
