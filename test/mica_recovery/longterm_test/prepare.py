#!/usr/bin/env python3
"""Generate configured UVW2-driver/U-response light-curve pairs only.

Outputs are MICA-style text files with columns: time, flux, flux_error.
The independent, shared_slow, spectral_shared, real_disk_mismatch and
known_response suites save sampled components as text.
Use --suite independent for disk-only / disk+independent tests, or --suite
shared_slow for the paired delayed slow signal in both bands. The default
--suite gaussian retains the original disk+Gaussian preparation behavior.
"""
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.signal import lfilter, oaconvolve
from scipy.special import log_ndtr, ndtr
import yaml

from starkey_disk_transfer import disk_moments, disk_transfer_function


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "experiment.yaml"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration is not a mapping: {path}")
    return config


def positive_float(name: str, value: object) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def load_observed(path: Path) -> np.ndarray:
    values = np.loadtxt(path)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError(f"Expected a three-column light curve in {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite value in {path}")
    if not np.all(np.diff(values[:, 0]) > 0):
        raise ValueError(f"Times are not strictly increasing in {path}")
    if not np.all(values[:, 2] > 0):
        raise ValueError(f"Flux errors must be positive in {path}")
    return values


def intrinsic_fractional_rms(values: np.ndarray) -> float:
    mean_flux = float(np.mean(values[:, 1]))
    variance = float(np.var(values[:, 1], ddof=1) - np.mean(values[:, 2] ** 2))
    if mean_flux <= 0 or variance <= 0:
        raise ValueError("The observed driver has no positive intrinsic variance")
    return sqrt(variance) / mean_flux


def kernel_grid(grid: dict) -> tuple[np.ndarray, float, int]:
    dt = positive_float("grid.dt_days", grid["dt_days"])
    lag_min = float(grid["lag_min_days"])
    lag_max = float(grid["lag_max_days"])
    if not np.isfinite(lag_min) or not np.isfinite(lag_max) or lag_min >= lag_max:
        raise ValueError("Require finite lag_min_days < lag_max_days")
    count_float = (lag_max - lag_min) / dt
    count = int(round(count_float))
    if count < 2 or not np.isclose(count_float, count, atol=1e-10):
        raise ValueError("The lag interval must contain an integer number of dt bins")
    oversample = int(grid["disk_bin_oversample"])
    if oversample < 1:
        raise ValueError("disk_bin_oversample must be at least 1")
    centers = lag_min + (np.arange(count, dtype=float) + 0.5) * dt
    return centers, dt, oversample


def disk_bin_masses(
    lag_centers: np.ndarray,
    dt: float,
    oversample: int,
    **disk_parameters: float,
) -> tuple[np.ndarray, float]:
    lag_min = lag_centers[0] - 0.5 * dt
    sub_dt = dt / oversample
    sub_lags = lag_min + (
        np.arange(len(lag_centers) * oversample, dtype=float) + 0.5
    ) * sub_dt
    density = disk_transfer_function(sub_lags, **disk_parameters)
    masses = density.reshape(len(lag_centers), oversample).sum(axis=1) * sub_dt
    sampled_area = float(masses.sum())
    if not 0.999 <= sampled_area <= 1.001:
        raise ValueError(
            f"Disk lag grid/integration is inadequate: area={sampled_area:.9f}"
        )
    masses /= sampled_area
    return masses, sampled_area


def gaussian_bin_masses(
    lag_centers: np.ndarray, dt: float, mean_days: float, sigma_days: float
) -> tuple[np.ndarray, float, float]:
    sigma = positive_float("Gaussian sigma_days", sigma_days)
    mean = float(mean_days)
    if not np.isfinite(mean):
        raise ValueError("Gaussian mean_days must be finite")
    edges = np.concatenate(
        ([lag_centers[0] - 0.5 * dt], lag_centers + 0.5 * dt)
    )
    cdf = ndtr((edges - mean) / sigma)
    masses = np.diff(cdf)
    sampled_area = float(masses.sum())
    if sampled_area < 1.0 - 1e-10:
        raise ValueError(
            f"Gaussian lag grid truncates appreciable mass: area={sampled_area:.12g}"
        )
    masses /= sampled_area
    negative_fraction = float(ndtr(-mean / sigma))
    return masses, sampled_area, negative_fraction


def known_response_kernels(config: dict) -> dict:
    """Unit-area broad Gaussian kernels, with both tails retained to 8 sigma.

    Keep these wide grids separate from the disk grid: the disk quadrature
    does not need to evaluate hundreds of days of numerically zero response.
    """
    dt = positive_float("grid.dt_days", config["grid"]["dt_days"])
    kernels = {}
    for name, spec in config.get("known_response", {}).get("cases", {}).items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid case directory name: {name!r}")
        disk_fraction = float(spec["disk_fraction"])
        gaussian_fraction = float(spec["gaussian_fraction"])
        if not np.isfinite([disk_fraction, gaussian_fraction]).all() or min(
            disk_fraction, gaussian_fraction
        ) < 0 or not np.isclose(disk_fraction + gaussian_fraction, 1.0):
            raise ValueError("Known-response fractions must be nonnegative and sum to one")
        mean = float(spec["mean_days"])
        sigma = positive_float("known_response sigma_days", spec["sigma_days"])
        if not np.isfinite(mean):
            raise ValueError("Known-response mean_days must be finite")
        lower = np.floor((mean - 8 * sigma) / dt) * dt
        upper = np.ceil((mean + 8 * sigma) / dt) * dt
        count = int(round((upper - lower) / dt))
        lag = lower + (np.arange(count) + 0.5) * dt
        mass, area, negative_fraction = gaussian_bin_masses(lag, dt, mean, sigma)
        kernels[name] = dict(lag=lag, mass=mass, sampled_area=area,
                             negative_fraction=negative_fraction,
                             disk_fraction=disk_fraction, gaussian_fraction=gaussian_fraction,
                             mean_days=mean, sigma_days=sigma)
    return kernels


def stationary_ou_from_white(
    standard_normals: np.ndarray, dt: float, tau_days: float
) -> np.ndarray:
    """Exact sampled stationary OU/DRW using a supplied paired white sequence."""
    tau = positive_float("DRW tau_days", tau_days)
    rho = np.exp(-dt / tau)
    innovations = standard_normals * sqrt(1.0 - rho * rho)
    innovations[0] = standard_normals[0]
    return lfilter([1.0], [1.0, -rho], innovations)


def convolve_and_sample(
    hidden: np.ndarray,
    kernel_mass: np.ndarray,
    hidden_start: float,
    first_lag: float,
    dt: float,
    sample_times: np.ndarray,
) -> np.ndarray:
    convolved = oaconvolve(hidden, kernel_mass, mode="full")
    location = (sample_times - hidden_start - first_lag) / dt
    lower = np.floor(location).astype(np.int64)
    fraction = location - lower
    if np.any(lower < 0) or np.any(lower + 1 >= len(convolved)):
        raise RuntimeError("Hidden light curve does not cover the requested convolution")
    return convolved[lower] * (1.0 - fraction) + convolved[lower + 1] * fraction


def case_name(tau_days: float, gaussian_name: str) -> str:
    if float(tau_days).is_integer():
        tau_label = f"{int(tau_days):04d}"
    else:
        tau_label = str(tau_days).replace(".", "p")
    return f"tau_{tau_label}d_{gaussian_name}"


def write_light_curve(
    path: Path, time: np.ndarray, flux: np.ndarray, error: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack((time, flux, error)),
        fmt="%.12g",
        header="time flux flux_error",
    )


def observation_setup(config_path: Path, config: dict) -> tuple:
    """Shared epochs, mean-flux-unit errors and W2 amplitude calibration."""
    observations = config["observations"]

    if observations["time_zero"] != "common_first_epoch":
        raise ValueError("Only time_zero=common_first_epoch is implemented")
    calibration = observations["amplitude_calibration"]
    if calibration not in (
        "observed_driver_intrinsic_fractional_rms",
        "fixed_stationary_driver_rms",
        "fixed_window_driver_rms",
    ):
        raise ValueError("Unsupported amplitude_calibration")
    if observations["noise"] != "observed_per_epoch_fractional_errors":
        raise ValueError("Unsupported noise model")

    input_dir = (config_path.parent / observations["input_dir"]).resolve()
    input_source = str(observations["input_source"])
    driver_band = str(observations["driver_band"])
    response_band = str(observations["response_band"])
    observed = {
        band: load_observed(input_dir / f"{input_source}_{band}.txt")
        for band in (driver_band, response_band)
    }

    time_zero = min(float(values[0, 0]) for values in observed.values())
    observed_time = {band: values[:, 0] - time_zero for band, values in observed.items()}
    fractional_error = {
        band: values[:, 2] / np.mean(values[:, 1])
        for band, values in observed.items()
    }
    if calibration == "fixed_stationary_driver_rms":
        target_driver_rms = positive_float(
            "driver_stationary_rms", config["spectral_shared"]["driver_stationary_rms"]
        )
    elif calibration == "fixed_window_driver_rms":
        target_driver_rms = positive_float(
            "observations.driver_window_rms", observations["driver_window_rms"]
        )
    else:
        target_driver_rms = intrinsic_fractional_rms(observed[driver_band])
    return driver_band, response_band, time_zero, observed_time, fractional_error, target_driver_rms


def spectral_shared_cases(config: dict) -> dict[str, dict]:
    """Validated parameters for the eta=1 shared spectral process."""
    suite = config.get("spectral_shared", {})
    result = {}
    for name, case in suite.get("cases", {}).items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid case directory name: {name!r}")
        spec = {**suite, **case}
        parameters = {key: positive_float(f"spectral_shared.{key}", spec[key]) for key in (
            "driver_tau_days", "driver_stationary_rms", "narrow_sigma_days",
            "narrow_gain", "slow_sigma_days", "slow_gain",
        )}
        for key in ("narrow_mean_days", "lag_days"):
            value = float(spec[key])
            if not np.isfinite(value):
                raise ValueError(f"spectral_shared.{key} must be finite")
            parameters[key] = value
        result[name] = parameters
    return result


def gaussian_ou_covariance(delta: np.ndarray, sigma: float, tau: float) -> np.ndarray:
    """Unit-variance OU covariance convolved with ONE unit-area Gaussian.

    Fourier transform: P_OU(omega)*exp(-sigma**2*omega**2/2).
    Two Gaussian filters combine by adding their variances, not widths.
    """
    delta = np.asarray(delta, dtype=float)
    if sigma == 0:
        return np.exp(-np.abs(delta) / tau)
    ratio = sigma / tau
    return (np.exp(ratio**2 / 2 - delta / tau + log_ndtr(delta / sigma - ratio))
            + np.exp(ratio**2 / 2 + delta / tau + log_ndtr(-delta / sigma - ratio)))


def spectral_component_covariance(tw: np.ndarray, tu: np.ndarray, parameters: dict) -> np.ndarray:
    """Covariance of [x(tw), s(tw), (psi_n*W)(tu), s(tu-lag)].

    x and s are independent, Px=(1-G)P, Ps=GP, and W=x+s is OU.
    No measurement noise, mean projection, gain or RMS rescaling is applied.
    This same analytic covariance can be reused by a likelihood predictor.
    """
    p = parameters
    times = [tw, tw, tu - p["narrow_mean_days"], tu - p["lag_days"]]
    widths = [0.0, 0.0, p["narrow_sigma_days"], 0.0]
    kinds = ["fast", "slow", "whole", "slow"]
    edges = np.cumsum([0] + [len(t) for t in times])
    covariance = np.empty((edges[-1], edges[-1]))
    variance = p["driver_stationary_rms"]**2
    for i in range(4):
        row = slice(edges[i], edges[i + 1])
        for j in range(i, 4):
            col = slice(edges[j], edges[j + 1])
            delta = times[i][:, None] - times[j][None, :]
            width = np.hypot(widths[i], widths[j])
            pair = {kinds[i], kinds[j]}
            if pair == {"fast", "slow"}:
                block = np.zeros_like(delta)
            elif pair == {"whole"}:
                block = gaussian_ou_covariance(delta, width, p["driver_tau_days"])
            else:
                slow = gaussian_ou_covariance(
                    delta, np.hypot(width, p["slow_sigma_days"]), p["driver_tau_days"])
                block = (gaussian_ou_covariance(delta, width, p["driver_tau_days"]) - slow
                         if "fast" in pair else slow)
            covariance[row, col] = variance * block
            covariance[col, row] = (variance * block).T
    return covariance


def sample_spectral_components(tw: np.ndarray, tu: np.ndarray, parameters: dict,
                              rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    covariance = spectral_component_covariance(tw, tu, parameters)
    # Smooth latent signals are numerically low-rank. Preserve the PSD matrix
    # with an eigensquare-root rather than adding unmodelled white jitter.
    values, vectors = eigh(covariance, check_finite=False)
    tolerance = 64 * np.finfo(float).eps * len(values) * values[-1]
    if values[0] < -tolerance:
        raise ValueError(f"Spectral covariance is not positive semidefinite: {values[0]:.6g}")
    sample = vectors @ (np.sqrt(np.maximum(values, 0)) * rng.standard_normal(len(values)))
    return tuple(np.split(sample, [len(tw), 2 * len(tw), 2 * len(tw) + len(tu)]))


def prepare_spectral_shared(config_path: Path, overwrite: bool = False) -> None:
    """Generate the eta=1 covariance-designed shared process at observed epochs.

    The short kernel is Gaussian, not Starkey; no extra W2 filter is applied.
    The broad Gaussian describes the linear predictor, NOT a deterministic
    total W-to-U response. Only component means, never realized RMS, are removed.
    """
    config_path = config_path.resolve()
    config = load_config(config_path)
    cases = spectral_shared_cases(config)
    if not cases:
        raise ValueError("No spectral-shared cases configured")
    if config["observations"]["amplitude_calibration"] != "fixed_stationary_driver_rms":
        raise ValueError("spectral_shared requires fixed_stationary_driver_rms calibration")
    experiment = config["experiment"]
    output_dir = (config_path.parent / experiment["output_dir"]).resolve()
    source = str(experiment["simulation_source"])
    driver, response, time_zero, times, errors, _ = observation_setup(config_path, config)
    bands = (driver, response)
    for name in cases:
        case = output_dir / name
        targets = [case / "input" / f"{source}_{band}.txt" for band in bands]
        targets += [case / "components" / f"{band}_components.txt" for band in bands]
        if not overwrite and any(path.exists() for path in targets):
            raise FileExistsError(f"Case already exists: {case}; use --overwrite to replace its data")
    seeds = np.random.SeedSequence(int(experiment["seed"])).spawn(len(cases))
    for (name, parameters), seed in zip(cases.items(), seeds):
        latent_seed, wnoise_seed, unoise_seed = seed.spawn(3)
        x, s, narrow, delayed = sample_spectral_components(
            times[driver], times[response], parameters, np.random.default_rng(latent_seed))
        primary = {driver: x, response: parameters["narrow_gain"] * narrow}
        slow = {driver: s, response: parameters["slow_gain"] * delayed}
        noise = {band: np.random.default_rng(child).normal(0, errors[band])
                 for band, child in zip(bands, (wnoise_seed, unoise_seed))}
        for band in bands:
            primary[band] -= primary[band].mean()
            slow[band] -= slow[band].mean()
        flux = {band: 1 + primary[band] + slow[band] + noise[band] for band in bands}
        if not all(np.isfinite(f).all() and np.all(f > 0) for f in flux.values()):
            raise ValueError(f"Invalid simulated flux in {name}; do not clip or reseed to select a preferred fit")
        case = output_dir / name
        for band in bands:
            write_light_curve(case / "input" / f"{source}_{band}.txt", times[band], flux[band], errors[band])
            path = case / "components" / f"{band}_components.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, np.column_stack((times[band], primary[band], slow[band], noise[band])),
                       fmt="%.12g", header=(
                           "time primary_variation shared_slow_variation measurement_noise\n"
                           f"band={band}; time_zero_mjd={time_zero:.9f}; flux=1+primary+shared_slow+noise\n"
                           "model: W=x+s; U=narrow_gain*(psi_n*W)+slow_gain*s(t-lag); eta=1\n"
                           "primary: x for driver, weighted narrow response for U; component means removed; no RMS rescaling\n"
                           + "; ".join(f"{key}={value:.17g}" for key, value in parameters.items()) + "\n"
                           + f"seed={experiment['seed']}; rng_case_child={seed.spawn_key[-1]}; observer_frame=true"))
        print(f"wrote {name}: {driver} n={len(times[driver])}, {response} n={len(times[response])}; "
              f"stationary driver RMS={parameters['driver_stationary_rms']:.8f}; eta=1; no RMS rescaling")


def real_disk_mismatch_cases(config: dict) -> dict[str, dict]:
    """Validate the causal-disk plus two shared-process injection parameters."""
    suite = config.get("real_disk_mismatch", {})
    defaults = {key: value for key, value in suite.items() if key != "cases"}
    result = {}
    positive = (
        "covariance_dt_days", "disk_support_days", "t1_kelvin",
        "driver_wavelength_observed_angstrom",
        "response_wavelength_observed_angstrom", "driver_tau_days",
        "blr_theta_days", "long_tau_days", "short_tau_days",
    )
    nonnegative = (
        "redshift", "response_disk_gain", "response_blr_gain",
        "blr_onset_days", "long_q", "long_response_gain",
        "long_lag_days", "short_q", "short_response_gain",
        "short_lag_days",
    )
    integer = ("covariance_n", "disk_fine_factor", "disk_origin_cells", "disk_origin_order")
    for name, case in suite.get("cases", {}).items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid case directory name: {name!r}")
        spec = {**defaults, **case}
        parameters = {key: positive_float(f"real_disk_mismatch.{key}", spec[key])
                      for key in positive}
        for key in nonnegative:
            value = float(spec[key])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"real_disk_mismatch.{key} must be finite and nonnegative")
            parameters[key] = value
        inclination = float(spec["inclination_deg"])
        if not np.isfinite(inclination) or not 0 <= inclination < 90:
            raise ValueError("real_disk_mismatch.inclination_deg must satisfy 0 <= i < 90")
        parameters["inclination_deg"] = inclination
        for key in integer:
            value = spec[key]
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"real_disk_mismatch.{key} must be an integer")
            parameters[key] = int(value)
        n = parameters["covariance_n"]
        if n < 1024 or n & (n - 1):
            raise ValueError("real_disk_mismatch.covariance_n must be a power of two >= 1024")
        if parameters["disk_fine_factor"] < 1:
            raise ValueError("real_disk_mismatch.disk_fine_factor must be at least one")
        if parameters["disk_origin_cells"] < 0:
            raise ValueError("real_disk_mismatch.disk_origin_cells must be nonnegative")
        if parameters["disk_origin_order"] < 4:
            raise ValueError("real_disk_mismatch.disk_origin_order must be at least four")
        result[name] = parameters
    return result


def _fft_covariance(spectrum: np.ndarray, lag: np.ndarray, dt: float, n: int) -> np.ndarray:
    """Interpolate a stationary covariance from its one-sided FFT spectrum."""
    spectrum = np.asarray(spectrum)
    if spectrum.shape != (n // 2 + 1,):
        raise ValueError("Spectrum does not match covariance_n")
    sampled = np.fft.irfft(spectrum, n=n) / dt
    position = np.mod(np.asarray(lag, dtype=float) / dt, n)
    lower = np.floor(position).astype(np.int64)
    fraction = position - lower
    return sampled[lower] * (1 - fraction) + sampled[(lower + 1) % n] * fraction


def _sample_real_disk_density(
    parameters: dict, wavelength_observed: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Midpoint disk density with high-order quadrature in the origin cells."""
    p = parameters
    factor = p["disk_fine_factor"]
    fine_dt = p["covariance_dt_days"] / factor
    count_float = p["disk_support_days"] / fine_dt
    count = int(round(count_float))
    if count < 2 or not np.isclose(count, count_float, atol=1e-10):
        raise ValueError("disk_support_days must contain an integer number of fine bins")
    lag = (np.arange(count, dtype=float) + 0.5) * fine_dt
    kwargs = {
        "wavelength_rest_angstrom": wavelength_observed / (1 + p["redshift"]),
        "t1_kelvin": p["t1_kelvin"],
        "inclination_deg": p["inclination_deg"],
        "redshift": p["redshift"],
    }
    density = disk_transfer_function(lag, **kwargs)
    cells = min(p["disk_origin_cells"], count)
    if cells:
        nodes, weights = np.polynomial.legendre.leggauss(p["disk_origin_order"])
        points = lag[:cells, None] + 0.5 * fine_dt * nodes
        density[:cells] = disk_transfer_function(points, **kwargs) @ weights / 2
    area = float(density.sum() * fine_dt)
    if not 0.995 <= area <= 1.005:
        raise ValueError(f"Disk integration is inadequate: area={area:.9f}")
    return lag, density, fine_dt, area


def _real_disk_transform(
    parameters: dict, wavelength_observed: float
) -> tuple[np.ndarray, float]:
    """Return the sampled causal Starkey kernel's Fourier transform."""
    p = parameters
    _, density, fine_dt, area = _sample_real_disk_density(p, wavelength_observed)
    fine_n = p["covariance_n"] * p["disk_fine_factor"]
    frequency = 2 * np.pi * np.fft.rfftfreq(
        p["covariance_n"], p["covariance_dt_days"]
    )
    transform = np.fft.rfft(density, n=fine_n)[:len(frequency)] * fine_dt
    # rfft treats density[0] as t=0; the density samples are bin midpoints.
    transform *= np.exp(-0.5j * frequency * fine_dt)
    return transform, area


def real_disk_mismatch_input_kernels(
    parameters: dict,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Physical I0-to-U kernels for display; shared processes are excluded."""
    p = parameters
    lag, disk, dt, disk_area = _sample_real_disk_density(
        p, p["response_wavelength_observed_angstrom"]
    )
    relative = lag - p["blr_onset_days"]
    blr = np.where(
        relative > 0,
        relative * np.exp(-relative / p["blr_theta_days"]) / p["blr_theta_days"]**2,
        0.0,
    )
    return lag, {
        "disk": p["response_disk_gain"] * disk,
        "blr": p["response_blr_gain"] * blr,
        "total": p["response_disk_gain"] * disk + p["response_blr_gain"] * blr,
        "disk_area": np.asarray(disk_area),
        "blr_area": np.asarray(blr.sum() * dt),
    }


def real_disk_mismatch_component_covariances(
    tw: np.ndarray, tu: np.ndarray, parameters: dict, target_driver_rms: float
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Expected observed-epoch covariances for three independent processes.

    I0 is an OU illuminating process. The two smooth shared processes are
    independent of I0 and of each other. Every actual response filter is
    causal and nonnegative, but their sum is not a deterministic W2-to-U map.
    """
    p = parameters
    tw, tu = np.asarray(tw, dtype=float), np.asarray(tu, dtype=float)
    if min(len(tw), len(tu)) < 2 or not np.isfinite(np.r_[tw, tu]).all():
        raise ValueError("Observed epochs must be finite and contain at least two points per band")
    dt, n = p["covariance_dt_days"], p["covariance_n"]
    period = dt * n
    span = max(tw.max(), tu.max()) - min(tw.min(), tu.min())
    guard = max(
        p["disk_support_days"],
        p["blr_onset_days"] + 12 * p["blr_theta_days"],
        p["long_lag_days"] + p["disk_support_days"],
        p["short_lag_days"] + p["disk_support_days"],
    )
    if period <= 2 * (span + guard):
        raise ValueError("covariance FFT period is too short for the observing window and filter support")

    frequency = 2 * np.pi * np.fft.rfftfreq(n, dt)
    driver_disk, driver_area = _real_disk_transform(
        p, p["driver_wavelength_observed_angstrom"]
    )
    response_disk, response_area = _real_disk_transform(
        p, p["response_wavelength_observed_angstrom"]
    )
    blr = (np.exp(-1j * frequency * p["blr_onset_days"])
           / (1 + 1j * frequency * p["blr_theta_days"])**2)
    illumination_psd = 2 * p["driver_tau_days"] / (
        1 + (frequency * p["driver_tau_days"])**2
    )
    long_psd = 4 * p["long_tau_days"] / (
        1 + (frequency * p["long_tau_days"])**2
    )**2
    short_psd = 4 * p["short_tau_days"] / (
        1 + (frequency * p["short_tau_days"])**2
    )**2
    filters = {
        "illumination": (
            driver_disk,
            p["response_disk_gain"] * response_disk + p["response_blr_gain"] * blr,
            illumination_psd,
        ),
        "shared_long": (
            sqrt(p["long_q"]) * driver_disk,
            sqrt(p["long_q"]) * p["long_response_gain"] * response_disk
            * np.exp(-1j * frequency * p["long_lag_days"]),
            long_psd,
        ),
        "shared_short": (
            sqrt(p["short_q"]) * driver_disk,
            sqrt(p["short_q"]) * p["short_response_gain"] * response_disk
            * np.exp(-1j * frequency * p["short_lag_days"]),
            short_psd,
        ),
    }
    dw = tw[:, None] - tw[None, :]
    du = tu[:, None] - tu[None, :]
    h = tu[None, :] - tw[:, None]
    nw = len(tw)
    covariances = {}
    for name, (w_filter, u_filter, psd) in filters.items():
        ww = _fft_covariance(psd * np.abs(w_filter)**2, dw, dt, n)
        wu = _fft_covariance(psd * u_filter * np.conj(w_filter), h, dt, n)
        uu = _fft_covariance(psd * np.abs(u_filter)**2, du, dt, n)
        covariance = np.block([[ww, wu], [wu.T, uu]])
        covariances[name] = (covariance + covariance.T) / 2

    driver_covariance = sum(cov[:nw, :nw] for cov in covariances.values())
    expected_driver_variance = np.trace(driver_covariance) / nw - np.mean(driver_covariance)
    if expected_driver_variance <= 0:
        raise ValueError("Expected sample-demeaned driver variance is not positive")
    alpha = positive_float("target_driver_rms", target_driver_rms) / sqrt(expected_driver_variance)
    covariances = {name: alpha**2 * covariance for name, covariance in covariances.items()}
    response_covariance = sum(cov[nw:, nw:] for cov in covariances.values())
    expected_response_variance = (
        np.trace(response_covariance) / len(tu) - np.mean(response_covariance)
    )
    diagnostics = {
        "alpha": alpha,
        "expected_driver_window_rms": target_driver_rms,
        "expected_response_window_rms": sqrt(expected_response_variance),
        "driver_disk_area": driver_area,
        "response_disk_area": response_area,
        "fft_period_days": period,
    }
    return covariances, diagnostics


def _sample_psd_covariance(
    covariance: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Draw from a numerically PSD covariance without adding hidden jitter."""
    values, vectors = eigh(covariance, check_finite=False)
    scale = max(float(values[-1]), np.finfo(float).tiny)
    tolerance = 256 * np.finfo(float).eps * len(values) * scale
    if values[0] < -tolerance:
        raise ValueError(
            f"Real-disk covariance is not positive semidefinite: {values[0]:.6g} "
            f"(tolerance {tolerance:.6g})"
        )
    sample = vectors @ (np.sqrt(np.maximum(values, 0)) * rng.standard_normal(len(values)))
    return sample, float(values[0])


def prepare_real_disk_mismatch(config_path: Path, overwrite: bool = False) -> None:
    """Generate the causal physical-kernel/mixed-driver possibility test."""
    config_path = config_path.resolve()
    config = load_config(config_path)
    cases = real_disk_mismatch_cases(config)
    if not cases:
        raise ValueError("No real-disk mismatch cases configured")
    if config["observations"]["amplitude_calibration"] != "fixed_window_driver_rms":
        raise ValueError("real_disk_mismatch requires fixed_window_driver_rms calibration")
    experiment = config["experiment"]
    design_id = str(config["real_disk_mismatch"].get("candidate_id", "unspecified"))
    if any(character in design_id for character in "\n;="):
        raise ValueError("real_disk_mismatch.candidate_id contains a reserved character")
    output_dir = (config_path.parent / experiment["output_dir"]).resolve()
    source = str(experiment["simulation_source"])
    driver, response, time_zero, times, errors, target_rms = observation_setup(
        config_path, config
    )
    bands = (driver, response)
    for name in cases:
        case = output_dir / name
        targets = [case / "input" / f"{source}_{band}.txt" for band in bands]
        targets += [case / "components" / f"{band}_components.txt" for band in bands]
        if not overwrite and any(path.exists() for path in targets):
            raise FileExistsError(
                f"Case already exists: {case}; use --overwrite to replace its data"
            )

    case_seeds = np.random.SeedSequence(int(experiment["seed"])).spawn(len(cases))
    for (name, parameters), case_seed in zip(cases.items(), case_seeds):
        covariances, diagnostics = real_disk_mismatch_component_covariances(
            times[driver], times[response], parameters, target_rms
        )
        children = case_seed.spawn(5)
        samples, minimum_eigenvalues = {}, {}
        for label, child in zip(
            ("illumination", "shared_long", "shared_short"), children[:3]
        ):
            samples[label], minimum_eigenvalues[label] = _sample_psd_covariance(
                covariances[label], np.random.default_rng(child)
            )
        nw = len(times[driver])
        components = {band: {} for band in bands}
        for label, sample in samples.items():
            components[driver][label] = sample[:nw]
            components[response][label] = sample[nw:]
        # Constant per-band means are nuisance parameters in the likelihood.
        # Remove them component by component, but never rescale a realization.
        for band in bands:
            for values in components[band].values():
                values -= values.mean()
        noise = {
            driver: np.random.default_rng(children[3]).normal(0, errors[driver]),
            response: np.random.default_rng(children[4]).normal(0, errors[response]),
        }
        flux = {
            band: 1 + sum(components[band].values()) + noise[band]
            for band in bands
        }
        if not all(
            np.isfinite(values).all() and np.all(values > 0)
            for values in flux.values()
        ):
            raise ValueError(
                f"Invalid simulated flux in {name}; do not clip, rescale or reseed "
                "to select a preferred fit"
            )
        case = output_dir / name
        parameters_text = "; ".join(
            f"{key}={value:.17g}" for key, value in parameters.items()
        )
        diagnostics_text = "; ".join(
            f"{key}={value:.17g}" for key, value in diagnostics.items()
        )
        eigenvalue_text = "; ".join(
            f"min_eigenvalue_{key}={value:.17g}"
            for key, value in minimum_eigenvalues.items()
        )
        for band in bands:
            write_light_curve(
                case / "input" / f"{source}_{band}.txt",
                times[band], flux[band], errors[band],
            )
            component_file = case / "components" / f"{band}_components.txt"
            component_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                component_file,
                np.column_stack((
                    times[band], components[band]["illumination"],
                    components[band]["shared_long"],
                    components[band]["shared_short"], noise[band],
                )),
                fmt="%.12g",
                header=(
                    "time illumination_variation shared_long_variation "
                    "shared_short_variation measurement_noise\n"
                    f"band={band}; time_zero_mjd={time_zero:.9f}; "
                    "flux=1+illumination+shared_long+shared_short+noise\n"
                    "model: W=D_W*(I0+sqrt(qL)*sL+sqrt(qS)*sS); "
                    "U=(a*D_U+b*B)*I0+D_U*(hL*sqrt(qL)*sL(t-dL)"
                    "+hS*sqrt(qS)*sS(t-dS))\n"
                    "I0, sL and sS are mutually independent; every physical kernel "
                    "is causal and nonnegative; component means removed; "
                    "no realization RMS rescaling\n"
                    f"{parameters_text}\n{diagnostics_text}\n{eigenvalue_text}\n"
                    f"design_id={design_id}; seed={experiment['seed']}; "
                    f"rng_case_child={case_seed.spawn_key[-1]}; observer_frame=true"
                ),
            )
        driver_signal = sum(components[driver].values())
        response_signal = sum(components[response].values())
        print(
            f"wrote {name}: {driver} n={len(times[driver])}, realized signal RMS="
            f"{np.std(driver_signal, ddof=1):.8f}; {response} n={len(times[response])}, "
            f"realized signal RMS={np.std(response_signal, ddof=1):.8f}; "
            f"alpha={diagnostics['alpha']:.12g}; no RMS rescaling"
        )


def prepare(config_path: Path, overwrite: bool = False) -> None:
    config_path = config_path.resolve()
    config = load_config(config_path)
    experiment = config["experiment"]
    grid_config = config["grid"]
    disk_config = config["disk"]
    response_config = config["response"]
    output_dir = (config_path.parent / experiment["output_dir"]).resolve()
    simulation_source = str(experiment["simulation_source"])
    driver_band, response_band, time_zero, observed_time, fractional_error, target_driver_rms = (
        observation_setup(config_path, config)
    )

    lag_centers, dt, oversample = kernel_grid(grid_config)
    redshift = float(disk_config["redshift"])
    t1_kelvin = positive_float("disk.t1_kelvin", disk_config["t1_kelvin"])
    inclination_deg = float(disk_config["inclination_deg"])
    wavelengths = disk_config["wavelength_observed_angstrom"]
    disk_mass: dict[str, np.ndarray] = {}
    disk_area: dict[str, float] = {}
    disk_truth: dict[str, dict] = {}
    for band in (driver_band, response_band):
        parameters = {
            "wavelength_rest_angstrom": float(wavelengths[band]) / (1.0 + redshift),
            "t1_kelvin": t1_kelvin,
            "inclination_deg": inclination_deg,
            "redshift": redshift,
        }
        disk_mass[band], disk_area[band] = disk_bin_masses(
            lag_centers, dt, oversample, **parameters
        )
        disk_truth[band] = disk_moments(**parameters)

    disk_fraction = float(response_config["disk_fraction"])
    gaussian_fraction = float(response_config["gaussian_fraction"])
    if min(disk_fraction, gaussian_fraction) < 0 or not np.isclose(
        disk_fraction + gaussian_fraction, 1.0
    ):
        raise ValueError("disk_fraction and gaussian_fraction must sum to 1")
    gaussian_masses: dict[str, np.ndarray] = {}
    gaussian_info: dict[str, tuple[float, float]] = {}
    for name, parameters in response_config["gaussian_cases"].items():
        masses, area, negative_fraction = gaussian_bin_masses(
            lag_centers,
            dt,
            float(parameters["mean_days"]),
            float(parameters["sigma_days"]),
        )
        gaussian_masses[str(name)] = masses
        gaussian_info[str(name)] = (area, negative_fraction)

    tau_values = [
        positive_float("DRW tau_days", value) for value in config["drw"]["tau_days"]
    ]
    if len(set(tau_values)) != len(tau_values):
        raise ValueError("DRW tau_days values must be unique")
    destinations: list[Path] = []
    for tau in tau_values:
        for gaussian_name in gaussian_masses:
            case_input = output_dir / case_name(tau, gaussian_name) / "input"
            destinations.extend(
                case_input / f"{simulation_source}_{band}.txt"
                for band in (driver_band, response_band)
            )
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        listing = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output files already exist; use --overwrite to replace them:\n{listing}"
        )

    all_times = np.concatenate(tuple(observed_time.values()))
    padding_tau = positive_float(
        "grid.hidden_padding_tau", grid_config["hidden_padding_tau"]
    )
    padding_days = padding_tau * max(tau_values)
    lag_min_edge = lag_centers[0] - 0.5 * dt
    lag_max_edge = lag_centers[-1] + 0.5 * dt
    hidden_start = np.floor(
        (all_times.min() - lag_max_edge - padding_days) / dt
    ) * dt
    hidden_end = np.ceil(
        (all_times.max() - lag_min_edge + padding_days) / dt
    ) * dt
    hidden_n = int(round((hidden_end - hidden_start) / dt)) + 1

    seeds = np.random.SeedSequence(int(experiment["seed"])).spawn(3)
    standard_normals = np.random.default_rng(seeds[0]).standard_normal(hidden_n)
    noise = {
        driver_band: np.random.default_rng(seeds[1]).normal(
            0.0, fractional_error[driver_band]
        ),
        response_band: np.random.default_rng(seeds[2]).normal(
            0.0, fractional_error[response_band]
        ),
    }

    print(
        f"Observed {driver_band} intrinsic fractional RMS: {target_driver_rms:.8f}; "
        f"common time zero MJD: {time_zero:.6f}"
    )
    print(
        f"Hidden paired grid: n={hidden_n}, dt={dt:g} d, "
        f"range=[{hidden_start:.2f}, {hidden_end:.2f}] d"
    )
    for band in (driver_band, response_band):
        print(
            f"{band} disk: sampled area={disk_area[band]:.9f}, "
            f"centroid={disk_truth[band]['mean_delay_days']:.9f} d"
        )
    for name, (area, negative_fraction) in gaussian_info.items():
        print(
            f"{name} Gaussian: sampled area={area:.12f}, "
            f"negative-lag fraction={negative_fraction:.6f}"
        )

    first_lag = float(lag_centers[0])
    for tau in tau_values:
        hidden = stationary_ou_from_white(standard_normals, dt, tau)
        driver_sample = convolve_and_sample(
            hidden,
            disk_mass[driver_band],
            hidden_start,
            first_lag,
            dt,
            observed_time[driver_band],
        )
        driver_sample -= np.mean(driver_sample)
        realized_rms = float(np.std(driver_sample, ddof=1))
        if realized_rms <= 0:
            raise RuntimeError(f"Degenerate driver realization for tau={tau:g} d")
        amplitude = target_driver_rms / realized_rms
        driver_variation = amplitude * driver_sample
        driver_flux = 1.0 + driver_variation + noise[driver_band]

        for gaussian_name, gaussian_mass in gaussian_masses.items():
            response_mass = (
                disk_fraction * disk_mass[response_band]
                + gaussian_fraction * gaussian_mass
            )
            response_sample = convolve_and_sample(
                hidden,
                response_mass,
                hidden_start,
                first_lag,
                dt,
                observed_time[response_band],
            )
            response_sample -= np.mean(response_sample)
            response_variation = amplitude * response_sample
            response_flux = 1.0 + response_variation + noise[response_band]
            if np.any(driver_flux <= 0) or np.any(response_flux <= 0):
                raise RuntimeError(
                    f"Non-positive flux for tau={tau:g}, case={gaussian_name}"
                )

            case_input = output_dir / case_name(tau, gaussian_name) / "input"
            write_light_curve(
                case_input / f"{simulation_source}_{driver_band}.txt",
                observed_time[driver_band],
                driver_flux,
                fractional_error[driver_band],
            )
            write_light_curve(
                case_input / f"{simulation_source}_{response_band}.txt",
                observed_time[response_band],
                response_flux,
                fractional_error[response_band],
            )
            print(
                f"wrote {case_input.parent.name}: "
                f"{driver_band} n={len(driver_flux)}, "
                f"signal_rms={np.std(driver_variation, ddof=1):.6f}; "
                f"{response_band} n={len(response_flux)}, "
                f"signal_rms={np.std(response_variation, ddof=1):.6f}"
            )


def build_disk_background(config_path: Path, config: dict) -> dict:
    """Build the paired clean disk signals and noise, without writing files.

    The fast hidden grid is anchored to independent_response and any configured
    known_response kernels. Adding shared_slow alone never shifts the seeded
    disk-only realization. A newly configured broad known kernel extends this
    grid, so its paired cases share each other, not the old disk_only realization.
    """
    suite = config["independent_response"]
    experiment, disk = config["experiment"], config["disk"]
    driver, response, time_zero, times, errors, target_rms = observation_setup(config_path, config)
    driver_tau = positive_float("independent_response.driver_tau_days", suite["driver_tau_days"])
    reference_taus = [positive_float("independent tau", spec["tau_days"])
                      for spec in suite["cases"].values() if spec["tau_days"] is not None]
    lag, dt, oversample = kernel_grid(config["grid"])
    kernels = {}
    for band in (driver, response):
        kernels[band], _ = disk_bin_masses(
            lag, dt, oversample,
            wavelength_rest_angstrom=float(disk["wavelength_observed_angstrom"][band]) / (1 + disk["redshift"]),
            t1_kelvin=float(disk["t1_kelvin"]),
            inclination_deg=float(disk["inclination_deg"]),
            redshift=float(disk["redshift"]),
        )
    # Associativity: disk_U convolves continuous noiseless W2, not observed W2.
    combined_mass = oaconvolve(kernels[driver], kernels[response], mode="full")
    known_kernels = known_response_kernels(config)
    longest_tau = max([driver_tau] + reference_taus)
    padding = positive_float("grid.hidden_padding_tau", config["grid"]["hidden_padding_tau"]) * longest_tau
    all_times = np.concatenate(tuple(times.values()))
    support_min = 2 * (lag[0] - dt / 2)
    support_max = 2 * (lag[-1] + dt / 2)
    for spec in known_kernels.values():
        support_min = min(support_min, lag[0] + spec["lag"][0] - dt)
        support_max = max(support_max, lag[-1] + spec["lag"][-1] + dt)
    start = np.floor((all_times.min() - support_max - padding) / dt) * dt
    end = np.ceil((all_times.max() - support_min + padding) / dt) * dt
    count = int(round((end - start) / dt)) + 1
    seeds = np.random.SeedSequence(int(experiment["seed"])).spawn(4)
    driver_white = np.random.default_rng(seeds[0]).standard_normal(count)
    independent_white = np.random.default_rng(seeds[3]).standard_normal(count)
    noise = {band: np.random.default_rng(seeds[i + 1]).normal(0, errors[band])
             for i, band in enumerate((driver, response))}
    hidden = stationary_ou_from_white(driver_white, dt, driver_tau)
    w2 = convolve_and_sample(hidden, kernels[driver], start, lag[0], dt, times[driver])
    w2 -= w2.mean()
    amplitude = target_rms / positive_float("realized driver RMS", np.std(w2, ddof=1))
    disk_signal = convolve_and_sample(hidden, combined_mass, start, 2 * lag[0], dt, times[response])
    disk_signal = amplitude * (disk_signal - disk_signal.mean())
    return dict(driver=driver, response=response, time_zero=time_zero, times=times,
                errors=errors, driver_tau=driver_tau, dt=dt, start=start, end=end,
                hidden_time=start + np.arange(count) * dt,
                independent_white=independent_white, noise=noise,
                hidden=hidden, amplitude=amplitude, lag=lag, kernels=kernels,
                known_response_kernels=known_kernels,
                clean={driver: amplitude * w2, response: disk_signal})


def prepare_known_response(config_path: Path, overwrite: bool = False) -> None:
    """W0=D*disk_W2; U=(f_disk*disk_U+f_gauss*Gaussian)*W0.

    The Gaussian is a deliberately two-sided, known W2-to-U test kernel.
    Negative-delay mass is retained, using future hidden W0 samples; this is
    an algorithmic recovery control, not a causal central-source disk kernel.
    Both contributions use the same W0 realization, not independent signals.
    """
    config_path = config_path.resolve()
    config = load_config(config_path)
    cases = known_response_kernels(config)
    if not cases:
        raise ValueError("No known-response cases configured")
    experiment = config["experiment"]
    output_dir = (config_path.parent / experiment["output_dir"]).resolve()
    source = str(experiment["simulation_source"])
    bands = tuple(str(config["observations"][key]) for key in ("driver_band", "response_band"))
    for name in cases:
        case = output_dir / name
        targets = [case / "input" / f"{source}_{band}.txt" for band in bands]
        targets += [case / "components" / f"{band}_components.txt" for band in bands]
        if not overwrite and any(path.exists() for path in targets):
            raise FileExistsError(f"Case already exists: {case}; use --overwrite to replace its data")

    background = build_disk_background(config_path, config)
    driver, response = background["driver"], background["response"]
    times, clean, noise = background["times"], background["clean"], background["noise"]
    dt = background["dt"]
    print(f"Known-response suite: driver tau={background['driver_tau']:g} d, "
          f"hidden=[{background['start']:.2f}, {background['end']:.2f}] d, dt={dt:g} d")
    for name, spec in background["known_response_kernels"].items():
        # Associativity ensures the broad component convolves W0, not raw D.
        combined = oaconvolve(background["kernels"][driver], spec["mass"], mode="full")
        first_lag = background["lag"][0] + spec["lag"][0]
        last_lag = first_lag + (len(combined) - 1) * dt
        if (times[response].min() - last_lag < background["start"] or
                times[response].max() - first_lag > background["end"]):
            raise ValueError("Hidden grid does not fully cover the two-sided response")
        raw = convolve_and_sample(background["hidden"], combined, background["start"],
                                  first_lag, dt, times[response])
        gaussian = background["amplitude"] * (raw - raw.mean())
        disk_part = {driver: clean[driver], response: spec["disk_fraction"] * clean[response]}
        gaussian_part = {driver: np.zeros_like(clean[driver]),
                         response: spec["gaussian_fraction"] * gaussian}
        flux = {band: 1 + disk_part[band] + gaussian_part[band] + noise[band] for band in bands}
        if not all(np.isfinite(values).all() and np.all(values > 0) for values in flux.values()):
            raise ValueError(f"Invalid simulated flux in {name}")
        case = output_dir / name
        for band in bands:
            write_light_curve(case / "input" / f"{source}_{band}.txt", times[band],
                              flux[band], background["errors"][band])
            component_file = case / "components" / f"{band}_components.txt"
            component_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                component_file,
                np.column_stack((times[band], disk_part[band], gaussian_part[band], noise[band])),
                fmt="%.12g",
                header=("time disk_variation gaussian_variation measurement_noise\n"
                        f"band={band}; time_zero_mjd={background['time_zero']:.9f}; flux=1+disk+gaussian+noise\n"
                        "model: W=W0=D*disk_W2; U=(f_disk*disk_U+f_gauss*Gaussian)*W0; per-band component means removed\n"
                        f"driver_tau_days={background['driver_tau']:g}; observer_frame=true; "
                        f"disk_fraction={spec['disk_fraction']:.12g}; gaussian_fraction={spec['gaussian_fraction']:.12g}; "
                        f"mean_days={spec['mean_days']:g}; sigma_days={spec['sigma_days']:g}\n"
                        f"gaussian_negative_fraction={spec['negative_fraction']:.17g}; "
                        f"gaussian_sampled_area={spec['sampled_area']:.17g}; "
                        f"gaussian_lag_edges_days=[{spec['lag'][0] - dt / 2:g},{spec['lag'][-1] + dt / 2:g}]\n"
                        f"seed={experiment['seed']}; dt_days={dt:g}; "
                        f"clean_hidden_start_days={background['start']:.12g}; clean_hidden_end_days={background['end']:.12g}"),
            )
        print(f"wrote {name}: {driver} n={len(times[driver])}, {response} n={len(times[response])}; "
              f"broad mean={spec['mean_days']:g} d, sigma={spec['sigma_days']:g} d, "
              f"negative fraction={spec['negative_fraction']:.6f}")


def prepare_independent(config_path: Path, overwrite: bool = False) -> None:
    """Y = (D * disk_W2) * disk_U + independent OU + measurement noise.

    Here disk_U defines an artificial, known W2-to-U kernel.  The independent
    term is an additive flux variation, not another transfer-function component.
    Its RNG stream is independent of D; short/long OU use paired innovations.
    """
    config_path = config_path.resolve()
    config = load_config(config_path)
    suite = config["independent_response"]
    experiment = config["experiment"]
    driver, response, time_zero, times, errors, target_rms = observation_setup(config_path, config)
    output_dir = (config_path.parent / experiment["output_dir"]).resolve()
    source = str(experiment["simulation_source"])
    driver_tau = positive_float("independent_response.driver_tau_days", suite["driver_tau_days"])
    ratio = float(suite["rms_ratio"])
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError("independent_response.rms_ratio must be nonnegative")
    cases = {
        name: None if spec["tau_days"] is None else positive_float("independent tau", spec["tau_days"])
        for name, spec in suite["cases"].items()
    }
    if not cases:
        raise ValueError("No independent-response cases configured")
    # Refuse regeneration of any existing outputs unless explicitly requested.
    for name in cases:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid case directory name: {name!r}")
        case = output_dir / name
        targets = [case / "input" / f"{source}_{band}.txt" for band in (driver, response)]
        targets += [case / "components" / f"{response}_components.txt"]
        if not overwrite and any(path.exists() for path in targets):
            raise FileExistsError(f"Case already exists: {case}; use --overwrite to replace its data")

    background = build_disk_background(config_path, config)
    dt, start, end = background["dt"], background["start"], background["end"]
    noise = background["noise"]
    driver_flux = 1 + background["clean"][driver] + noise[driver]
    disk_signal = background["clean"][response]
    disk_rms = positive_float("realized disk RMS", np.std(disk_signal, ddof=1))
    print(f"Independent suite: driver tau={driver_tau:g} d, t0={time_zero:.6f}, "
          f"hidden=[{start:.2f}, {end:.2f}] d, dt={dt:g} d, disk RMS={disk_rms:.8f}")
    for name, tau in cases.items():
        independent = np.zeros_like(disk_signal)
        if tau is not None:
            mother = stationary_ou_from_white(background["independent_white"], dt, tau)
            independent = np.interp(times[response], background["hidden_time"], mother)
            independent -= independent.mean()
            independent *= ratio * disk_rms / positive_float("independent RMS", np.std(independent, ddof=1))
        response_flux = 1 + disk_signal + independent + noise[response]
        if not all(np.isfinite(flux).all() and np.all(flux > 0) for flux in (driver_flux, response_flux)):
            raise ValueError(f"Invalid simulated flux in {name}")
        case = output_dir / name
        write_light_curve(case / "input" / f"{source}_{driver}.txt", times[driver], driver_flux, errors[driver])
        write_light_curve(case / "input" / f"{source}_{response}.txt", times[response], response_flux, errors[response])
        component_file = case / "components" / f"{response}_components.txt"
        component_file.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            component_file,
            np.column_stack((times[response], disk_signal, independent, noise[response])),
            fmt="%.12g",
            header=(f"time disk_variation independent_variation measurement_noise\n"
                    f"time_zero_mjd={time_zero:.9f}; response_flux=1+disk+independent+noise; "
                    f"driver_tau_days={driver_tau:g}; independent_tau_days={tau}; "
                    f"rms_ratio={0.0 if tau is None else ratio:g}; seed={experiment['seed']}"),
        )
        print(f"wrote {name}: {driver} n={len(driver_flux)}, {response} n={len(disk_signal)}, "
              f"independent/disk RMS={np.std(independent, ddof=1) / disk_rms:.6f}")


def prepare_shared_slow(config_path: Path, overwrite: bool = False) -> None:
    """W=W0+aW*s(t), U=disk_U*W0+aU*s(t-lag), plus measurement noise.

    One independent latent OU is shared across bands. Positive lag means U
    varies later. The amplitude is calibrated only at the W2 sampling epochs;
    each band is then centered, without rescaling the delayed U realization.
    This mixed-driver model has no injected single total W-to-U transfer kernel.
    """
    config_path = config_path.resolve()
    config = load_config(config_path)
    suite, experiment = config["shared_slow"], config["experiment"]
    tau = positive_float("shared_slow.tau_days", suite["tau_days"])
    q_w = positive_float("shared_slow.driver_rms_ratio", suite["driver_rms_ratio"])
    coefficient_ratio = positive_float("shared_slow.response_amplitude_ratio", suite["response_amplitude_ratio"])
    if coefficient_ratio <= 1:
        raise ValueError("shared_slow requires aU/aW > 1")
    cases = {name: positive_float("shared_slow lag_days", spec["lag_days"])
             for name, spec in suite["cases"].items()}
    if not cases:
        raise ValueError("No shared-slow cases configured")
    output_dir = (config_path.parent / experiment["output_dir"]).resolve()
    source = str(experiment["simulation_source"])
    bands = tuple(str(config["observations"][key]) for key in ("driver_band", "response_band"))
    for name in cases:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid case directory name: {name!r}")
        case = output_dir / name
        targets = [case / "input" / f"{source}_{band}.txt" for band in bands]
        targets += [case / "components" / f"{band}_components.txt" for band in bands]
        if not overwrite and any(path.exists() for path in targets):
            raise FileExistsError(f"Case already exists: {case}; use --overwrite to replace its data")

    background = build_disk_background(config_path, config)
    driver, response = background["driver"], background["response"]
    times, clean, noise = background["times"], background["clean"], background["noise"]
    dt = background["dt"]
    # Separate grid: do not alter the clean disk/noise realization when padding
    # for the slow timescale and delayed samples outside the observed window.
    queries = np.concatenate([times[driver]] + [times[response] - lag for lag in cases.values()])
    padding = positive_float("grid.hidden_padding_tau", config["grid"]["hidden_padding_tau"]) * tau
    start = np.floor((queries.min() - padding) / dt) * dt
    end = np.ceil((queries.max() + padding) / dt) * dt
    count = int(round((end - start) / dt)) + 1
    hidden_time = start + np.arange(count) * dt
    seed = np.random.SeedSequence(int(experiment["seed"])).spawn(5)[4]
    mother = stationary_ou_from_white(np.random.default_rng(seed).standard_normal(count), dt, tau)

    def sample_slow(query: np.ndarray) -> np.ndarray:
        if np.any(query < hidden_time[0]) or np.any(query > hidden_time[-1]):
            raise ValueError("Shared slow samples fall outside the hidden grid")
        return np.interp(query, hidden_time, mother)

    raw_w = sample_slow(times[driver])
    a_w = q_w * positive_float("clean driver RMS", np.std(clean[driver], ddof=1)) / positive_float(
        "shared slow driver RMS", np.std(raw_w, ddof=1))
    a_u = coefficient_ratio * a_w
    print(f"Shared-slow suite: tau={tau:g} d, slow hidden=[{start:.2f}, {end:.2f}] d, "
          f"dt={dt:g} d, aW={a_w:.8f}, aU={a_u:.8f}; clean driver tau={background['driver_tau']:g} d")
    for name, lag in cases.items():
        raw_u = sample_slow(times[response] - lag)
        raw_means = {driver: float(raw_w.mean()), response: float(raw_u.mean())}
        slow = {driver: a_w * (raw_w - raw_means[driver]),
                response: a_u * (raw_u - raw_means[response])}
        flux = {band: 1 + clean[band] + slow[band] + noise[band] for band in bands}
        if not all(np.isfinite(values).all() and np.all(values > 0) for values in flux.values()):
            raise ValueError(f"Invalid simulated flux in {name}")
        case = output_dir / name
        for band in bands:
            write_light_curve(case / "input" / f"{source}_{band}.txt", times[band], flux[band], background["errors"][band])
            component_file = case / "components" / f"{band}_components.txt"
            component_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                component_file,
                np.column_stack((times[band], clean[band], slow[band], noise[band])),
                fmt="%.12g",
                header=("time disk_variation shared_slow_variation measurement_noise\n"
                        f"band={band}; time_zero_mjd={background['time_zero']:.9f}; flux=1+disk+shared_slow+noise\n"
                        "model: W=W0+aW*s(t); U=disk_U*W0+aU*s(t-lag); per-band constant means removed\n"
                        f"driver_tau_days={background['driver_tau']:g}; slow_tau_days={tau:g}; lag_days={lag:g}; observer_frame=true\n"
                        f"driver_slow_to_clean_rms={q_w:.12g}; aU_over_aW={coefficient_ratio:.12g}; "
                        f"aW={a_w:.17g}; aU={a_u:.17g}; raw_slow_mean_W={raw_means[driver]:.17g}; raw_slow_mean_U={raw_means[response]:.17g}\n"
                        f"seed={experiment['seed']}; slow_rng_child=4; dt_days={dt:g}; "
                        f"slow_hidden_start_days={start:.12g}; slow_hidden_end_days={end:.12g}; "
                        f"clean_hidden_start_days={background['start']:.12g}; clean_hidden_end_days={background['end']:.12g}"),
            )
        print(f"wrote {name}: lag=+{lag:g} d; "
              f"{driver} n={len(times[driver])}, slow RMS={np.std(slow[driver], ddof=1):.8f}; "
              f"{response} n={len(times[response])}, slow RMS={np.std(slow[response], ddof=1):.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--suite",
        choices=("gaussian", "independent", "shared_slow", "known_response",
                 "spectral_shared", "real_disk_mismatch", "paired_ablation"),
        default="gaussian",
                        help="Select Gaussian echo, independent/shared signals, known kernel, or covariance-designed shared process.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace data files only for the selected suite.",
    )
    arguments = parser.parse_args()
    if arguments.suite == "paired_ablation":
        from prepare_ablation import prepare_cases
        prepare_cases(arguments.config, overwrite=arguments.overwrite)
        return
    generate = {"gaussian": prepare, "independent": prepare_independent,
                "shared_slow": prepare_shared_slow,
                "known_response": prepare_known_response,
                "spectral_shared": prepare_spectral_shared,
                "real_disk_mismatch": prepare_real_disk_mismatch}[arguments.suite]
    generate(arguments.config, overwrite=arguments.overwrite)


if __name__ == "__main__":
    main()
