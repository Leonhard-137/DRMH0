"""MCMC fit of a shifted Gamma density to a disk transfer function."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


@dataclass(frozen=True)
class GammaFitSummary:
    """Posterior median and central 68% interval for one parameter."""

    median: float
    lower: float
    upper: float


def shifted_gamma_on_grid(
    time_days: np.ndarray,
    log_k: float,
    log_theta: float,
    t0_days: float,
) -> np.ndarray | None:
    """Return a shifted Gamma density normalized on the supplied finite grid."""
    time_days = np.asarray(time_days, dtype=float)
    k = float(np.exp(log_k))
    theta_days = float(np.exp(log_theta))
    shifted_time = time_days - t0_days

    density = np.zeros_like(time_days)
    positive = shifted_time > 0.0
    if not np.any(positive):
        return None

    x = shifted_time[positive]
    log_density = (
        (k - 1.0) * np.log(x)
        - x / theta_days
        - gammaln(k)
        - k * np.log(theta_days)
    )

    # Subtracting the maximum avoids unnecessary overflow/underflow.
    max_log_density = float(np.max(log_density))
    density[positive] = np.exp(log_density - max_log_density)

    integral = float(np.trapezoid(density, time_days))
    if not np.isfinite(integral) or integral <= 0.0:
        return None

    return density / integral


def shifted_gamma_free_area_on_grid(
    time_days: np.ndarray,
    log_k: float,
    log_theta: float,
    t0_days: float,
    log_area_scale: float,
) -> np.ndarray | None:
    """Return an unrenormalized shifted-Gamma density with free area scale."""
    time_days = np.asarray(time_days, dtype=float)
    k = float(np.exp(log_k))
    theta_days = float(np.exp(log_theta))
    shifted_time = time_days - t0_days

    density = np.zeros_like(time_days)
    positive = shifted_time > 0.0
    if not np.any(positive):
        return None

    x = shifted_time[positive]
    log_density = (
        log_area_scale
        + (k - 1.0) * np.log(x)
        - x / theta_days
        - gammaln(k)
        - k * np.log(theta_days)
    )
    density[positive] = np.exp(log_density)
    if np.any(~np.isfinite(density)):
        return None
    return density


def fit_shifted_gamma_mcmc(
    time_days: np.ndarray,
    response: np.ndarray,
    *,
    fixed_k: float | None = None,
    nwalkers: int = 32,
    nsteps: int = 4000,
    burn: int = 1500,
    thin: int = 5,
    random_seed: int = 1234,
    progress: bool = True,
) -> dict:
    """Fit a shifted Gamma density with emcee.

    When ``fixed_k`` is ``None``, sample log(k), log(theta), t0, and
    log(model scatter).  Otherwise, hold k fixed and sample only log(theta),
    t0, and log(model scatter).
    """
    rng = np.random.default_rng(random_seed)

    if fixed_k is not None:
        fixed_k = float(fixed_k)
        if not np.isfinite(fixed_k) or fixed_k <= 0.0:
            raise ValueError("fixed_k must be a finite positive number.")

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

    nonzero = np.flatnonzero(y > 1.0e-4 * y_max)
    first_response_time = float(t[nonzero[0]]) if nonzero.size else float(t[0])

    mean_time = float(np.trapezoid(t * y, t))
    variance = float(np.trapezoid((t - mean_time) ** 2 * y, t))
    variance = max(variance, dt**2)

    t0_initial = first_response_time - 0.5 * dt
    shifted_mean = max(mean_time - t0_initial, dt)
    k_initial = float(np.clip(shifted_mean**2 / variance, 0.3, 100.0))
    if fixed_k is None:
        theta_initial = float(
            np.clip(variance / shifted_mean, dt / 20.0, 5.0 * time_span)
        )
    else:
        theta_initial = float(
            np.clip(shifted_mean / fixed_k, dt / 20.0, 5.0 * time_span)
        )
    sigma_initial = max(0.02 * y_max, 1.0e-10)

    log_k_bounds = (np.log(0.1), np.log(200.0))
    log_theta_bounds = (np.log(dt / 50.0), np.log(10.0 * time_span))
    t0_bounds = (float(t[0] - 0.25 * time_span), peak_time)
    log_sigma_bounds = (
        np.log(max(1.0e-10 * y_max, 1.0e-15)),
        np.log(max(2.0 * y_max, 1.0e-12)),
    )

    if fixed_k is None:
        ndim = 4
        raw_parameter_names = ("log(k)", "log(theta)", "t0", "log(sigma)")

        def unpack_parameters(
            parameters: np.ndarray,
        ) -> tuple[float, float, float, float]:
            log_k, log_theta, t0_days, log_sigma = parameters
            return log_k, log_theta, t0_days, log_sigma

        initial_center = np.array(
            [
                np.log(k_initial),
                np.log(theta_initial),
                t0_initial,
                np.log(sigma_initial),
            ],
            dtype=float,
        )
        proposal_scale = np.array(
            [0.05, 0.05, max(0.25 * dt, 1.0e-8), 0.10],
            dtype=float,
        )
    else:
        ndim = 3
        fixed_log_k = float(np.log(fixed_k))
        raw_parameter_names = ("log(theta)", "t0", "log(sigma)")

        def unpack_parameters(
            parameters: np.ndarray,
        ) -> tuple[float, float, float, float]:
            log_theta, t0_days, log_sigma = parameters
            return fixed_log_k, log_theta, t0_days, log_sigma

        initial_center = np.array(
            [
                np.log(theta_initial),
                t0_initial,
                np.log(sigma_initial),
            ],
            dtype=float,
        )
        proposal_scale = np.array(
            [0.05, max(0.25 * dt, 1.0e-8), 0.10],
            dtype=float,
        )

    def log_prior(parameters: np.ndarray) -> float:
        log_k, log_theta, t0_days, log_sigma = unpack_parameters(parameters)
        inside = (
            log_theta_bounds[0] < log_theta < log_theta_bounds[1]
            and t0_bounds[0] < t0_days < t0_bounds[1]
            and log_sigma_bounds[0] < log_sigma < log_sigma_bounds[1]
        )
        if fixed_k is None:
            inside = inside and log_k_bounds[0] < log_k < log_k_bounds[1]
        return 0.0 if inside else -np.inf

    def log_likelihood(parameters: np.ndarray) -> float:
        log_k, log_theta, t0_days, log_sigma = unpack_parameters(parameters)
        model_density = shifted_gamma_on_grid(t, log_k, log_theta, t0_days)
        if model_density is None or np.any(~np.isfinite(model_density)):
            return -np.inf

        sigma = float(np.exp(log_sigma))
        residual = y - model_density
        return float(
            -0.5
            * np.sum(
                (residual / sigma) ** 2
                + np.log(2.0 * np.pi * sigma**2)
            )
        )

    def log_probability(parameters: np.ndarray) -> float:
        prior = log_prior(parameters)
        if not np.isfinite(prior):
            return -np.inf
        return prior + log_likelihood(parameters)

    walkers: list[np.ndarray] = []
    attempts = 0
    while len(walkers) < nwalkers:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("Could not initialize walkers inside the prior bounds.")
        candidate = initial_center + rng.normal(size=ndim) * proposal_scale
        if np.isfinite(log_probability(candidate)):
            walkers.append(candidate)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(np.asarray(walkers), nsteps, progress=progress)

    raw_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    if fixed_k is None:
        samples = np.column_stack(
            [
                np.exp(raw_samples[:, 0]),
                np.exp(raw_samples[:, 1]),
                raw_samples[:, 2],
                np.exp(raw_samples[:, 3]),
            ]
        )
    else:
        samples = np.column_stack(
            [
                np.full(raw_samples.shape[0], fixed_k),
                np.exp(raw_samples[:, 0]),
                raw_samples[:, 1],
                np.exp(raw_samples[:, 2]),
            ]
        )
    parameter_names = ("k", "theta_days", "t0_days", "sigma")
    percentiles = np.percentile(samples, [16.0, 50.0, 84.0], axis=0)

    summaries = {
        name: GammaFitSummary(
            median=float(percentiles[1, index]),
            lower=float(percentiles[1, index] - percentiles[0, index]),
            upper=float(percentiles[2, index] - percentiles[1, index]),
        )
        for index, name in enumerate(parameter_names)
    }

    median_curve = shifted_gamma_on_grid(
        t,
        np.log(summaries["k"].median),
        np.log(summaries["theta_days"].median),
        summaries["t0_days"].median,
    )
    if median_curve is None:
        raise RuntimeError("The posterior median parameters produced an invalid curve.")

    return {
        "time_days": t,
        "response": y,
        "samples": samples,
        "raw_samples": raw_samples,
        "raw_parameter_names": raw_parameter_names,
        "parameter_names": parameter_names,
        "summaries": summaries,
        "median_curve": median_curve,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "fixed_k": fixed_k,
        "sampler": sampler,
    }


def fit_shifted_gamma_free_area_mcmc(
    time_days: np.ndarray,
    response: np.ndarray,
    *,
    fixed_k: float,
    nwalkers: int = 32,
    nsteps: int = 4000,
    burn: int = 1500,
    thin: int = 5,
    random_seed: int = 1234,
    progress: bool = True,
) -> dict:
    """Fit area, theta, t0, and model scatter while holding k fixed."""
    rng = np.random.default_rng(random_seed)
    fixed_k = float(fixed_k)
    if not np.isfinite(fixed_k) or fixed_k <= 0.0:
        raise ValueError("fixed_k must be a finite positive number.")

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
    nonzero = np.flatnonzero(y > 1.0e-4 * y_max)
    first_response_time = float(t[nonzero[0]]) if nonzero.size else float(t[0])
    mean_time = float(np.trapezoid(t * y, t) / target_area)

    t0_initial = first_response_time - 0.5 * dt
    shifted_mean = max(mean_time - t0_initial, dt)
    theta_initial = float(
        np.clip(shifted_mean / fixed_k, dt / 20.0, 5.0 * time_span)
    )
    area_initial = target_area

    log_area_bounds = (
        np.log(max(target_area * 1.0e-4, 1.0e-15)),
        np.log(max(target_area * 1.0e4, 1.0e-12)),
    )
    log_theta_bounds = (np.log(dt / 50.0), np.log(10.0 * time_span))
    t0_bounds = (float(t[0] - 0.25 * time_span), peak_time)
    log_scatter_bounds = (
        np.log(max(1.0e-10 * y_max, 1.0e-15)),
        np.log(max(2.0 * y_max, 1.0e-12)),
    )
    fixed_log_k = float(np.log(fixed_k))

    def shape_objective(parameters: np.ndarray) -> float:
        log_area, log_theta, t0_days = parameters
        model_density = shifted_gamma_free_area_on_grid(
            t, fixed_log_k, log_theta, t0_days, log_area
        )
        if model_density is None:
            return np.inf
        return float(np.sum((y - model_density) ** 2))

    base_start = np.array(
        [np.log(area_initial), np.log(theta_initial), t0_initial],
        dtype=float,
    )
    mode_t0 = float(
        np.clip(
            peak_time - max(fixed_k - 1.0, 0.0) * theta_initial,
            t0_bounds[0] + 1.0e-10,
            t0_bounds[1] - 1.0e-10,
        )
    )
    shape_starts = (
        base_start,
        np.array([base_start[0], base_start[1], mode_t0]),
        np.array([base_start[0], base_start[1] - np.log(2.0), mode_t0]),
        np.array([base_start[0], base_start[1] + np.log(2.0), mode_t0]),
    )
    shape_bounds = (log_area_bounds, log_theta_bounds, t0_bounds)
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
        log_area_initial, log_theta_initial, t0_initial = map(float, best_shape.x)
    else:
        log_area_initial = float(base_start[0])
        log_theta_initial = float(base_start[1])

    optimized_curve = shifted_gamma_free_area_on_grid(
        t, fixed_log_k, log_theta_initial, t0_initial, log_area_initial
    )
    if optimized_curve is None:
        raise RuntimeError("Could not initialize the free-area Gamma model.")
    scatter_initial = max(
        float(np.sqrt(np.mean((y - optimized_curve) ** 2))), 1.0e-10
    )

    def log_prior(parameters: np.ndarray) -> float:
        log_area, log_theta, t0_days, log_scatter = parameters
        inside = (
            log_area_bounds[0] < log_area < log_area_bounds[1]
            and log_theta_bounds[0] < log_theta < log_theta_bounds[1]
            and t0_bounds[0] < t0_days < t0_bounds[1]
            and log_scatter_bounds[0] < log_scatter < log_scatter_bounds[1]
        )
        return 0.0 if inside else -np.inf

    def log_likelihood(parameters: np.ndarray) -> float:
        log_area, log_theta, t0_days, log_scatter = parameters
        model_density = shifted_gamma_free_area_on_grid(
            t, fixed_log_k, log_theta, t0_days, log_area
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
            log_theta_initial,
            t0_initial,
            np.log(scatter_initial),
        ],
        dtype=float,
    )
    proposal_scale = np.array(
        [0.05, 0.05, max(0.25 * dt, 1.0e-8), 0.10], dtype=float
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
            np.full(raw_samples.shape[0], fixed_k),
            np.exp(raw_samples[:, 1]),
            raw_samples[:, 2],
            np.exp(raw_samples[:, 3]),
        ]
    )
    parameter_names = ("area_scale", "k", "theta_days", "t0_days", "sigma")
    percentiles = np.percentile(samples, [16.0, 50.0, 84.0], axis=0)
    summaries = {
        name: GammaFitSummary(
            median=float(percentiles[1, index]),
            lower=float(percentiles[1, index] - percentiles[0, index]),
            upper=float(percentiles[2, index] - percentiles[1, index]),
        )
        for index, name in enumerate(parameter_names)
    }
    median_curve = shifted_gamma_free_area_on_grid(
        t,
        fixed_log_k,
        np.log(summaries["theta_days"].median),
        summaries["t0_days"].median,
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
            "log(theta)",
            "t0",
            "log(sigma)",
        ),
        "parameter_names": parameter_names,
        "summaries": summaries,
        "median_curve": median_curve,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "fixed_k": fixed_k,
        "free_area": True,
        "sampler": sampler,
    }


def print_summary(result: dict) -> None:
    """Print posterior medians and central 68% intervals."""
    print("\nPosterior summary (median -lower +upper):")
    for name in result["parameter_names"]:
        item = result["summaries"][name]
        print(f"  {name:12s} = {item.median:.6g} -{item.lower:.3g} +{item.upper:.3g}")
    print(f"  acceptance  = {result['acceptance_fraction']:.3f}")


def save_results(result: dict, output_directory: str | Path = "gamma_fit_output") -> None:
    """Save posterior samples, summary, fitted curve, and diagnostic plots."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    samples = result["samples"]
    names = result["parameter_names"]
    np.savetxt(
        output_directory / "posterior_samples.csv",
        samples,
        delimiter=",",
        header=",".join(names),
        comments="",
    )

    fitted_table = np.column_stack(
        [result["time_days"], result["response"], result["median_curve"]]
    )
    np.savetxt(
        output_directory / "fitted_curve.csv",
        fitted_table,
        delimiter=",",
        header="time_days,response,gamma_median",
        comments="",
    )

    with (output_directory / "summary.txt").open("w", encoding="utf-8") as handle:
        for name in names:
            item = result["summaries"][name]
            handle.write(
                f"{name} = {item.median:.10g} -{item.lower:.10g} +{item.upper:.10g}\n"
            )
        handle.write(f"acceptance_fraction = {result['acceptance_fraction']:.10g}\n")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(result["time_days"], result["response"], label="Disk response")
    gamma_label = "Gamma median"
    if result.get("fixed_k") is not None:
        gamma_label += f" (k={result['fixed_k']:g} fixed)"
    if result.get("free_area"):
        gamma_label += ", area free"
    ax.plot(result["time_days"], result["median_curve"], "--", label=gamma_label)
    ax.set_xlabel("Time delay [days]")
    ax.set_ylabel(
        "Response density"
        if result.get("free_area")
        else "Normalized response density"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "gamma_fit.png", dpi=180)
    plt.close(fig)

    chain = result["sampler"].get_chain()
    raw_names = result["raw_parameter_names"]
    fig, axes = plt.subplots(
        len(raw_names),
        1,
        figsize=(8.0, 1.75 * len(raw_names)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    for index, ax in enumerate(axes):
        ax.plot(chain[:, :, index], alpha=0.25)
        ax.set_ylabel(raw_names[index])
    axes[-1].set_xlabel("MCMC step")
    fig.tight_layout()
    fig.savefig(output_directory / "trace.png", dpi=180)
    plt.close(fig)
