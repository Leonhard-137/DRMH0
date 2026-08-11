"""MCMC fit of a unit-area Gaussian to a disk transfer function."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class GaussianFitSummary:
    """Posterior median and central 68% interval for one parameter."""

    median: float
    lower: float
    upper: float


def gaussian_on_grid(
    time_days: np.ndarray,
    center_days: float,
    log_width: float,
) -> np.ndarray | None:
    """Return a Gaussian density normalized on the supplied finite grid."""
    time_days = np.asarray(time_days, dtype=float)
    width_days = float(np.exp(log_width))
    standardized = (time_days - center_days) / width_days
    log_density = -0.5 * standardized**2

    max_log_density = float(np.max(log_density))
    density = np.exp(log_density - max_log_density)
    integral = float(np.trapezoid(density, time_days))
    if not np.isfinite(integral) or integral <= 0.0:
        return None
    return density / integral


def gaussian_free_area_on_grid(
    time_days: np.ndarray,
    center_days: float,
    log_width: float,
    log_area_scale: float,
) -> np.ndarray | None:
    """Return an analytic Gaussian density with a free total-area scale."""
    time_days = np.asarray(time_days, dtype=float)
    width_days = float(np.exp(log_width))
    standardized = (time_days - center_days) / width_days
    log_density = (
        log_area_scale
        - 0.5 * np.log(2.0 * np.pi)
        - log_width
        - 0.5 * standardized**2
    )
    density = np.exp(log_density)
    if np.any(~np.isfinite(density)):
        return None
    return density


def fit_gaussian_mcmc(
    time_days: np.ndarray,
    response: np.ndarray,
    *,
    nwalkers: int = 32,
    nsteps: int = 4000,
    burn: int = 1500,
    thin: int = 5,
    random_seed: int = 1234,
    progress: bool = True,
) -> dict:
    """Sample Gaussian center, log(width), and log(model scatter)."""
    rng = np.random.default_rng(random_seed)

    t = np.asarray(time_days, dtype=float)
    y = np.asarray(response, dtype=float)
    if t.ndim != 1 or y.ndim != 1:
        raise ValueError("time_days and response must be one-dimensional.")
    if t.size != y.size:
        raise ValueError("time_days and response must have the same length.")
    if t.size < 5:
        raise ValueError("At least five time bins are required.")
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
        raise ValueError("Inputs contain NaN or infinite values.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time_days must be strictly increasing.")
    if nwalkers < 10:
        raise ValueError("Use at least 10 walkers for this fit.")
    if not (0 <= burn < nsteps):
        raise ValueError("burn must satisfy 0 <= burn < nsteps.")
    if thin < 1:
        raise ValueError("thin must be at least 1.")

    y = np.clip(y, 0.0, None)
    normalization = float(np.trapezoid(y, t))
    if normalization <= 0.0:
        raise ValueError("response must have a positive integral.")
    y = y / normalization

    dt = float(np.median(np.diff(t)))
    time_span = float(t[-1] - t[0])
    y_max = float(np.max(y))
    peak_time = float(t[np.argmax(y)])
    mean_time = float(np.trapezoid(t * y, t))
    variance = float(np.trapezoid((t - mean_time) ** 2 * y, t))

    center_initial = peak_time
    width_initial = float(
        np.clip(np.sqrt(max(variance, dt**2)), dt / 20.0, 5.0 * time_span)
    )
    scatter_initial = max(0.02 * y_max, 1.0e-10)

    center_bounds = (float(t[0] - 0.25 * time_span), float(t[-1]))
    log_width_bounds = (np.log(dt / 50.0), np.log(10.0 * time_span))
    log_scatter_bounds = (
        np.log(max(1.0e-10 * y_max, 1.0e-15)),
        np.log(max(2.0 * y_max, 1.0e-12)),
    )

    # Locate the main least-squares mode before initializing the ensemble.
    # This prevents individual walkers from remaining in a broad, low-likelihood
    # Gaussian branch when the target response is strongly skewed.
    def shape_objective(parameters: np.ndarray) -> float:
        center_days, log_width = parameters
        model_density = gaussian_on_grid(t, center_days, log_width)
        if model_density is None or np.any(~np.isfinite(model_density)):
            return np.inf
        return float(np.sum((y - model_density) ** 2))

    log_width_initial = float(np.log(width_initial))
    shape_starts = (
        np.array([peak_time, log_width_initial]),
        np.array([mean_time, log_width_initial]),
        np.array([0.5 * (peak_time + mean_time), log_width_initial]),
        np.array([peak_time, log_width_initial - np.log(2.0)]),
        np.array([mean_time, log_width_initial + np.log(2.0)]),
    )
    shape_bounds = (center_bounds, log_width_bounds)
    optimized_shapes = [
        minimize(
            shape_objective,
            start,
            method="L-BFGS-B",
            bounds=shape_bounds,
        )
        for start in shape_starts
    ]
    valid_shapes = [
        item
        for item in optimized_shapes
        if item.success and np.isfinite(item.fun)
    ]
    if valid_shapes:
        best_shape = min(valid_shapes, key=lambda item: float(item.fun))
        center_initial = float(best_shape.x[0])
        width_initial = float(np.exp(best_shape.x[1]))
        optimized_curve = gaussian_on_grid(
            t, center_initial, float(best_shape.x[1])
        )
        if optimized_curve is not None:
            scatter_initial = max(
                float(np.sqrt(np.mean((y - optimized_curve) ** 2))),
                1.0e-10,
            )

    def log_prior(parameters: np.ndarray) -> float:
        center_days, log_width, log_scatter = parameters
        inside = (
            center_bounds[0] < center_days < center_bounds[1]
            and log_width_bounds[0] < log_width < log_width_bounds[1]
            and log_scatter_bounds[0] < log_scatter < log_scatter_bounds[1]
        )
        return 0.0 if inside else -np.inf

    def log_likelihood(parameters: np.ndarray) -> float:
        center_days, log_width, log_scatter = parameters
        model_density = gaussian_on_grid(t, center_days, log_width)
        if model_density is None or np.any(~np.isfinite(model_density)):
            return -np.inf

        model_scatter = float(np.exp(log_scatter))
        residual = y - model_density
        return float(
            -0.5
            * np.sum(
                (residual / model_scatter) ** 2
                + np.log(2.0 * np.pi * model_scatter**2)
            )
        )

    def log_probability(parameters: np.ndarray) -> float:
        prior = log_prior(parameters)
        if not np.isfinite(prior):
            return -np.inf
        return prior + log_likelihood(parameters)

    initial_center = np.array(
        [center_initial, np.log(width_initial), np.log(scatter_initial)],
        dtype=float,
    )
    proposal_scale = np.array(
        [max(0.25 * dt, 1.0e-8), 0.05, 0.10],
        dtype=float,
    )

    walkers: list[np.ndarray] = []
    attempts = 0
    while len(walkers) < nwalkers:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("Could not initialize walkers inside the prior bounds.")
        candidate = initial_center + rng.normal(size=3) * proposal_scale
        if np.isfinite(log_probability(candidate)):
            walkers.append(candidate)

    sampler = emcee.EnsembleSampler(nwalkers, 3, log_probability)
    sampler.run_mcmc(np.asarray(walkers), nsteps, progress=progress)

    raw_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    samples = np.column_stack(
        [raw_samples[:, 0], np.exp(raw_samples[:, 1]), np.exp(raw_samples[:, 2])]
    )
    parameter_names = ("center_days", "width_days", "model_scatter")
    percentiles = np.percentile(samples, [16.0, 50.0, 84.0], axis=0)
    summaries = {
        name: GaussianFitSummary(
            median=float(percentiles[1, index]),
            lower=float(percentiles[1, index] - percentiles[0, index]),
            upper=float(percentiles[2, index] - percentiles[1, index]),
        )
        for index, name in enumerate(parameter_names)
    }

    median_curve = gaussian_on_grid(
        t,
        summaries["center_days"].median,
        np.log(summaries["width_days"].median),
    )
    if median_curve is None:
        raise RuntimeError("The posterior median parameters produced an invalid curve.")

    return {
        "time_days": t,
        "response": y,
        "samples": samples,
        "raw_samples": raw_samples,
        "raw_parameter_names": ("center", "log(width)", "log(model scatter)"),
        "parameter_names": parameter_names,
        "summaries": summaries,
        "median_curve": median_curve,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "sampler": sampler,
    }


def fit_gaussian_free_area_mcmc(
    time_days: np.ndarray,
    response: np.ndarray,
    *,
    nwalkers: int = 32,
    nsteps: int = 4000,
    burn: int = 1500,
    thin: int = 5,
    random_seed: int = 1234,
    progress: bool = True,
) -> dict:
    """Sample Gaussian area, center, width, and model scatter."""
    rng = np.random.default_rng(random_seed)
    t = np.asarray(time_days, dtype=float)
    y = np.asarray(response, dtype=float)
    if t.ndim != 1 or y.ndim != 1:
        raise ValueError("time_days and response must be one-dimensional.")
    if t.size != y.size:
        raise ValueError("time_days and response must have the same length.")
    if t.size < 5:
        raise ValueError("At least five time bins are required.")
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(y)):
        raise ValueError("Inputs contain NaN or infinite values.")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time_days must be strictly increasing.")
    if nwalkers < 10:
        raise ValueError("Use at least 10 walkers for this fit.")
    if not (0 <= burn < nsteps):
        raise ValueError("burn must satisfy 0 <= burn < nsteps.")
    if thin < 1:
        raise ValueError("thin must be at least 1.")

    y = np.clip(y, 0.0, None)
    target_area = float(np.trapezoid(y, t))
    if not np.isfinite(target_area) or target_area <= 0.0:
        raise ValueError("response must have a positive finite integral.")

    dt = float(np.median(np.diff(t)))
    time_span = float(t[-1] - t[0])
    y_max = float(np.max(y))
    peak_time = float(t[np.argmax(y)])
    mean_time = float(np.trapezoid(t * y, t) / target_area)
    variance = float(
        np.trapezoid((t - mean_time) ** 2 * y, t) / target_area
    )

    center_initial = peak_time
    width_initial = float(
        np.clip(np.sqrt(max(variance, dt**2)), dt / 20.0, 5.0 * time_span)
    )
    area_initial = target_area
    center_bounds = (float(t[0] - 0.25 * time_span), float(t[-1]))
    log_width_bounds = (np.log(dt / 50.0), np.log(10.0 * time_span))
    log_area_bounds = (
        np.log(max(target_area * 1.0e-4, 1.0e-15)),
        np.log(max(target_area * 1.0e4, 1.0e-12)),
    )
    log_scatter_bounds = (
        np.log(max(1.0e-10 * y_max, 1.0e-15)),
        np.log(max(2.0 * y_max, 1.0e-12)),
    )

    def shape_objective(parameters: np.ndarray) -> float:
        log_area, center_days, log_width = parameters
        model_density = gaussian_free_area_on_grid(
            t, center_days, log_width, log_area
        )
        if model_density is None:
            return np.inf
        return float(np.sum((y - model_density) ** 2))

    log_width_initial = float(np.log(width_initial))
    log_area_initial = float(np.log(area_initial))
    shape_starts = (
        np.array([log_area_initial, peak_time, log_width_initial]),
        np.array([log_area_initial, mean_time, log_width_initial]),
        np.array([
            log_area_initial,
            0.5 * (peak_time + mean_time),
            log_width_initial,
        ]),
        np.array([
            log_area_initial,
            peak_time,
            log_width_initial - np.log(2.0),
        ]),
        np.array([
            log_area_initial,
            mean_time,
            log_width_initial + np.log(2.0),
        ]),
    )
    shape_bounds = (log_area_bounds, center_bounds, log_width_bounds)
    optimized_shapes = [
        minimize(
            shape_objective,
            start,
            method="L-BFGS-B",
            bounds=shape_bounds,
        )
        for start in shape_starts
    ]
    valid_shapes = [
        item for item in optimized_shapes if item.success and np.isfinite(item.fun)
    ]
    if valid_shapes:
        best_shape = min(valid_shapes, key=lambda item: float(item.fun))
        log_area_initial, center_initial, log_width_initial = map(
            float, best_shape.x
        )

    optimized_curve = gaussian_free_area_on_grid(
        t, center_initial, log_width_initial, log_area_initial
    )
    if optimized_curve is None:
        raise RuntimeError("Could not initialize the free-area Gaussian model.")
    scatter_initial = max(
        float(np.sqrt(np.mean((y - optimized_curve) ** 2))), 1.0e-10
    )

    def log_prior(parameters: np.ndarray) -> float:
        log_area, center_days, log_width, log_scatter = parameters
        inside = (
            log_area_bounds[0] < log_area < log_area_bounds[1]
            and center_bounds[0] < center_days < center_bounds[1]
            and log_width_bounds[0] < log_width < log_width_bounds[1]
            and log_scatter_bounds[0] < log_scatter < log_scatter_bounds[1]
        )
        return 0.0 if inside else -np.inf

    def log_likelihood(parameters: np.ndarray) -> float:
        log_area, center_days, log_width, log_scatter = parameters
        model_density = gaussian_free_area_on_grid(
            t, center_days, log_width, log_area
        )
        if model_density is None:
            return -np.inf
        model_scatter = float(np.exp(log_scatter))
        residual = y - model_density
        return float(
            -0.5
            * np.sum(
                (residual / model_scatter) ** 2
                + np.log(2.0 * np.pi * model_scatter**2)
            )
        )

    def log_probability(parameters: np.ndarray) -> float:
        prior = log_prior(parameters)
        if not np.isfinite(prior):
            return -np.inf
        return prior + log_likelihood(parameters)

    initial_center = np.array(
        [
            log_area_initial,
            center_initial,
            log_width_initial,
            np.log(scatter_initial),
        ],
        dtype=float,
    )
    proposal_scale = np.array(
        [0.05, max(0.25 * dt, 1.0e-8), 0.05, 0.10], dtype=float
    )
    walkers: list[np.ndarray] = []
    attempts = 0
    while len(walkers) < nwalkers:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("Could not initialize walkers inside the prior bounds.")
        candidate = initial_center + rng.normal(size=4) * proposal_scale
        if np.isfinite(log_probability(candidate)):
            walkers.append(candidate)

    sampler = emcee.EnsembleSampler(nwalkers, 4, log_probability)
    sampler.run_mcmc(np.asarray(walkers), nsteps, progress=progress)
    raw_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    samples = np.column_stack(
        [
            np.exp(raw_samples[:, 0]),
            raw_samples[:, 1],
            np.exp(raw_samples[:, 2]),
            np.exp(raw_samples[:, 3]),
        ]
    )
    parameter_names = (
        "area_scale",
        "center_days",
        "width_days",
        "model_scatter",
    )
    percentiles = np.percentile(samples, [16.0, 50.0, 84.0], axis=0)
    summaries = {
        name: GaussianFitSummary(
            median=float(percentiles[1, index]),
            lower=float(percentiles[1, index] - percentiles[0, index]),
            upper=float(percentiles[2, index] - percentiles[1, index]),
        )
        for index, name in enumerate(parameter_names)
    }
    median_curve = gaussian_free_area_on_grid(
        t,
        summaries["center_days"].median,
        np.log(summaries["width_days"].median),
        np.log(summaries["area_scale"].median),
    )
    if median_curve is None:
        raise RuntimeError("The posterior median parameters produced an invalid curve.")

    return {
        "time_days": t,
        "response": y,
        "samples": samples,
        "raw_samples": raw_samples,
        "raw_parameter_names": (
            "log(area)",
            "center",
            "log(width)",
            "log(model scatter)",
        ),
        "parameter_names": parameter_names,
        "summaries": summaries,
        "median_curve": median_curve,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "free_area": True,
        "sampler": sampler,
    }


def print_summary(result: dict) -> None:
    """Print posterior medians and central 68% intervals."""
    print("\nPosterior summary (median -lower +upper):")
    for name in result["parameter_names"]:
        item = result["summaries"][name]
        print(f"  {name:14s} = {item.median:.6g} -{item.lower:.3g} +{item.upper:.3g}")
    print(f"  acceptance     = {result['acceptance_fraction']:.3f}")


def save_results(result: dict, output_directory: str | Path) -> None:
    """Save posterior samples, summary, fitted curve, and diagnostics."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    np.savetxt(
        output_directory / "posterior_samples.csv",
        result["samples"],
        delimiter=",",
        header=",".join(result["parameter_names"]),
        comments="",
    )
    np.savetxt(
        output_directory / "fitted_curve.csv",
        np.column_stack(
            [result["time_days"], result["response"], result["median_curve"]]
        ),
        delimiter=",",
        header="time_days,response,gaussian_median",
        comments="",
    )

    with (output_directory / "summary.txt").open("w", encoding="utf-8") as handle:
        for name in result["parameter_names"]:
            item = result["summaries"][name]
            handle.write(
                f"{name} = {item.median:.10g} -{item.lower:.10g} +{item.upper:.10g}\n"
            )
        handle.write(f"acceptance_fraction = {result['acceptance_fraction']:.10g}\n")

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(
        result["time_days"],
        result["response"],
        color="#2563EB",
        linewidth=1.7,
        label="Disk response",
    )
    axis.plot(
        result["time_days"],
        result["median_curve"],
        color="#D97706",
        linestyle="--",
        linewidth=1.7,
        label=(
            "Gaussian median, area free"
            if result.get("free_area")
            else "Gaussian median"
        ),
    )
    axis.set_xlabel("Time delay [days]")
    axis.set_ylabel(
        "Response density"
        if result.get("free_area")
        else "Normalized response density"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "gaussian_fit.png", dpi=180)
    plt.close(figure)

    chain = result["sampler"].get_chain()
    raw_names = result["raw_parameter_names"]
    figure, axes = plt.subplots(
        len(raw_names),
        1,
        figsize=(8.0, 1.75 * len(raw_names)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    for index, axis in enumerate(axes):
        axis.plot(chain[:, :, index], alpha=0.25)
        axis.set_ylabel(raw_names[index])
    axes[-1].set_xlabel("MCMC step")
    figure.tight_layout()
    figure.savefig(output_directory / "trace.png", dpi=180)
    plt.close(figure)
