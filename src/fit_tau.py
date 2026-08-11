#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from posterior_branches import (
    branch_definitions,
    branch_result_relative_dir,
    stable_seed_offset,
)


MODEL_ALIASES = {"b4_3": "4/3", "4_3": "4/3", "b2": "2", "free_beta": "free"}

MODELS = {
    "4/3": {"beta": 4.0 / 3.0, "fit": "b4_3", "label": r"$\beta=4/3$", "ls": "-"},
    "2": {"beta": 2.0, "fit": "b2", "label": r"$\beta=2$", "ls": "--"},
    "free": {"beta": None, "fit": "free", "label": r"$\beta$ free", "ls": "-."},
}

MOMENT2_MODEL_ALIASES = {
    "g8_3": "8/3",
    "8_3": "8/3",
    "g4": "4",
    "free_gamma": "free",
}

MOMENT2_MODELS = {
    "8/3": {
        "gamma": 8.0 / 3.0,
        "fit": "g8_3",
        "label": r"$\gamma=8/3$",
        "ls": "-",
    },
    "4": {
        "gamma": 4.0,
        "fit": "g4",
        "label": r"$\gamma=4$",
        "ls": "--",
    },
    "free": {
        "gamma": None,
        "fit": "free",
        "label": r"$\gamma$ free",
        "ls": "-.",
    },
}

MOMENT2_COMPONENTS = ("tf0", "tf1", "total")

MOMENT0_MODELS = {
    "standard": {
        "eta": -7.0 / 3.0,
        "label": r"standard: $\eta=-7/3$",
        "ls": "--",
    },
    "slim": {
        "eta": -1.0,
        "label": r"slim: $\eta=-1$",
        "ls": ":",
    },
    "free": {
        "eta": None,
        "label": r"$\eta$ free",
        "ls": "-",
    },
}

TF_LABELS = {
    "gaussian": "gaussian",
    "tophat": "tophat",
    "gamma": "gamma",
    "exponential": "exp",
    "exp": "exp",
}


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def rel(base: Path, path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else base / path


def q68(x) -> tuple[float, float, float]:
    p16, p50, p84 = np.percentile(np.asarray(x, float), [15.865, 50.0, 84.135])
    return float(p50), float(p50 - p16), float(p84 - p50)


def weighted_quantile(values, weights, quantiles=(0.15865, 0.5, 0.84135)):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if len(values) == 0:
        raise ValueError("no finite, positive-weight lag samples")
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    cdf = np.cumsum(weights[order])
    cdf /= cdf[-1]
    return np.interp(np.asarray(quantiles, float), cdf, values)


def model_label(model) -> str:
    model = str(model).lower()
    return TF_LABELS.get(model, model)


def ncomp_from_config(fit: dict) -> int:
    if "ncomp" in fit:
        return int(fit["ncomp"])
    nc = fit.get("number_component", [2, 2])
    return int(nc[0] if isinstance(nc, list) else nc)


def mica_run_root(source_dir: Path, cfg: dict) -> Path:
    mica = cfg.get("mica_round1", {}) or {}
    root = rel(source_dir, mica.get("output_root", "runs/mica"))
    return root / f"{ncomp_from_config(mica)}comp"


def mica_result_dir(source_dir: Path, cfg: dict) -> Path:
    out = mica_run_root(source_dir, cfg) / "result"
    out.mkdir(parents=True, exist_ok=True)
    return out


def lambda_tau_result_dir(source_dir: Path, cfg: dict, lt: dict) -> Path:
    """Use an explicit analysis result directory without moving legacy outputs."""
    if lt.get("result_dir"):
        out = rel(source_dir, lt["result_dir"])
        out.mkdir(parents=True, exist_ok=True)
        return out
    return mica_result_dir(source_dir, cfg)


def tau_model(wave, tau0, beta, lambda0):
    return tau0 * ((np.asarray(wave, float) / float(lambda0)) ** beta - 1.0)


def moment2_model(wave, moment20, gamma, lambda0):
    return moment20 * ((np.asarray(wave, float) / float(lambda0)) ** gamma - 1.0)


def gaussian_mixture_moments(centers, widths, amplitudes):
    """Return component variances, mixture centroid, and mixture variance."""
    component_variance, _, centroid, variance, fractions = (
        transfer_mixture_moments(centers, widths, amplitudes, "gaussian")
    )
    return component_variance, centroid, variance, fractions


def transfer_mixture_moments(centers, widths, amplitudes, model):
    """Return component M2, component M1, mixture M1, and mixture M2."""
    centers = np.asarray(centers, float)
    widths = np.asarray(widths, float)
    amplitudes = np.asarray(amplitudes, float)
    if centers.shape != widths.shape or centers.shape != amplitudes.shape:
        raise ValueError("centers, widths, and amplitudes must have matching shapes")
    if centers.ndim != 2:
        raise ValueError("component arrays must have shape (ncomp, nsample)")
    if not (
        np.isfinite(centers).all()
        and np.isfinite(widths).all()
        and np.isfinite(amplitudes).all()
    ):
        raise ValueError("transfer-function component parameters must be finite")
    if np.any(widths <= 0.0) or np.any(amplitudes <= 0.0):
        raise ValueError("transfer-function widths and amplitudes must be positive")

    model = model_label(model)
    if model == "gaussian":
        component_centroid = centers
        component_variance = widths**2
    elif model == "gamma":
        component_centroid = centers + 2.0 * widths
        component_variance = 2.0 * widths**2
    else:
        raise ValueError(
            "second-moment calculation supports only Gaussian and k=2 gamma "
            f"transfer functions, not {model!r}"
        )

    amplitude_sum = np.sum(amplitudes, axis=0)
    fractions = amplitudes / amplitude_sum
    mixture_centroid = np.sum(fractions * component_centroid, axis=0)
    mixture_variance = np.sum(
        fractions
        * (component_variance + (component_centroid - mixture_centroid) ** 2),
        axis=0,
    )
    return component_variance, component_centroid, mixture_centroid, mixture_variance, fractions


def sample_band(values, weights, nsamp, rng):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    weights = weights / np.sum(weights)
    return rng.choice(values, nsamp, replace=True, p=weights)


def fixed_beta_posterior(draws, waves, sigma, lambda0, beta):
    x = (waves / lambda0) ** beta - 1.0
    w = 1.0 / sigma**2
    return (draws @ (w * x)) / np.sum(w * x * x)


def free_beta_log_probability(
    theta,
    waves,
    median_tau,
    err_low,
    err_high,
    lambda0,
    tau0_bounds,
    beta_bounds,
    tau0_prior,
    likelihood,
):
    tau0, beta = map(float, theta)
    if not (tau0_bounds[0] < tau0 < tau0_bounds[1]):
        return -np.inf
    if not (beta_bounds[0] < beta < beta_bounds[1]):
        return -np.inf

    if tau0_prior == "log_uniform":
        if tau0 <= 0.0:
            return -np.inf
        log_prior = -np.log(tau0)
    elif tau0_prior == "uniform":
        log_prior = 0.0
    else:
        raise ValueError(f"unknown free_tau0_prior: {tau0_prior}")

    predicted = tau_model(waves, tau0, beta, lambda0)
    if likelihood == "split_normal":
        # The split-normal normalization is constant in theta. The lower
        # uncertainty applies when the prediction is below the measured mode.
        sigma = np.where(predicted < median_tau, err_low, err_high)
    elif likelihood == "normal":
        sigma = 0.5 * (err_low + err_high)
    else:
        raise ValueError(f"unknown free_likelihood: {likelihood}")
    if not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
        return -np.inf
    return float(log_prior - 0.5 * np.sum(((median_tau - predicted) / sigma) ** 2))


def joint_free_beta_posterior(
    waves,
    median_tau,
    err_low,
    err_high,
    lambda0,
    config,
    seed,
):
    import emcee
    from scipy.optimize import minimize

    tau0_bounds = tuple(map(float, config.get("tau0_bounds", [1.0e-3, 100.0])))
    beta_bounds = tuple(map(float, config.get("free_beta_bounds", [0.01, 5.0])))
    if len(tau0_bounds) != 2 or tau0_bounds[0] >= tau0_bounds[1]:
        raise ValueError(f"invalid tau0_bounds: {tau0_bounds}")
    if len(beta_bounds) != 2 or beta_bounds[0] >= beta_bounds[1]:
        raise ValueError(f"invalid free_beta_bounds: {beta_bounds}")

    tau0_prior = str(config.get("free_tau0_prior", "log_uniform")).lower()
    likelihood = str(config.get("free_likelihood", "split_normal")).lower()
    if tau0_prior == "log_uniform" and tau0_bounds[0] <= 0.0:
        raise ValueError("log-uniform tau0 prior requires tau0_bounds[0] > 0")
    if tau0_prior not in {"uniform", "log_uniform"}:
        raise ValueError(f"unknown free_tau0_prior: {tau0_prior}")
    if likelihood not in {"normal", "split_normal"}:
        raise ValueError(f"unknown free_likelihood: {likelihood}")

    err_low = np.asarray(err_low, float).copy()
    err_high = np.asarray(err_high, float).copy()
    regularized_error_sides = 0
    if likelihood == "split_normal":
        bad_low = ~np.isfinite(err_low) | (err_low <= 0.0)
        bad_high = ~np.isfinite(err_high) | (err_high <= 0.0)
        if np.any(bad_low & bad_high):
            raise ValueError(
                "split-normal likelihood has a band with no positive uncertainty on either side"
            )
        regularized_error_sides = int(np.sum(bad_low) + np.sum(bad_high))
        if regularized_error_sides:
            # A very low-ESS weighted posterior can put p50 and p84 (or p16)
            # on the same discrete sample. Mirror the finite opposite-side
            # interval so the conditional branch remains numerically usable,
            # while recording the intervention in the diagnostics.
            err_low[bad_low] = err_high[bad_low]
            err_high[bad_high] = err_low[bad_high]
            warnings.warn(
                "split-normal interval contained a zero/non-finite side; "
                f"mirrored {regularized_error_sides} side(s) from the finite opposite interval",
                RuntimeWarning,
            )

    nwalkers = int(config.get("free_mcmc_nwalkers", 48))
    steps = int(config.get("free_mcmc_steps", 12000))
    burn = int(float(config.get("free_mcmc_burn_frac", 0.3)) * steps)
    thin = int(config.get("free_mcmc_thin", 10))
    if nwalkers < 8:
        raise ValueError("free_mcmc_nwalkers must be at least 8")
    if not 0 < burn < steps:
        raise ValueError("free MCMC burn-in must satisfy 0 < burn < steps")
    if thin < 1:
        raise ValueError("free_mcmc_thin must be positive")

    args = (
        waves,
        median_tau,
        err_low,
        err_high,
        lambda0,
        tau0_bounds,
        beta_bounds,
        tau0_prior,
        likelihood,
    )

    def physical_log_prob(theta):
        return free_beta_log_probability(theta, *args)

    if tau0_prior == "log_uniform":
        sampling_tau0_parameter = "log_tau0"
        sampling_tau0_bounds = (np.log(tau0_bounds[0]), np.log(tau0_bounds[1]))

        def to_physical(theta):
            return np.array([np.exp(theta[0]), theta[1]], float)

        def log_prob(theta):
            physical = to_physical(theta)
            # Add the Jacobian d(tau0)/d(log tau0). It exactly cancels the
            # 1/tau0 density of the requested log-uniform physical prior.
            return physical_log_prob(physical) + float(theta[0])
    else:
        sampling_tau0_parameter = "tau0"
        sampling_tau0_bounds = tau0_bounds

        def to_physical(theta):
            return np.asarray(theta, float)

        def log_prob(theta):
            return physical_log_prob(theta)

    symmetric_sigma = 0.5 * (err_low + err_high)
    initial_beta = float(np.clip(2.0, beta_bounds[0] + 1.0e-6, beta_bounds[1] - 1.0e-6))
    x = (waves / lambda0) ** initial_beta - 1.0
    ivar = 1.0 / symmetric_sigma**2
    initial_tau0 = float(np.sum(ivar * x * median_tau) / np.sum(ivar * x * x))
    initial_tau0 = float(np.clip(initial_tau0, tau0_bounds[0] * 1.01, tau0_bounds[1] * 0.99))
    initial_sampling_tau0 = (
        np.log(initial_tau0)
        if sampling_tau0_parameter == "log_tau0"
        else initial_tau0
    )
    result = minimize(
        lambda theta: -log_prob(theta),
        x0=np.array([initial_sampling_tau0, initial_beta]),
        method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1.0e-10, "fatol": 1.0e-10},
    )
    if not result.success or not np.isfinite(log_prob(result.x)):
        raise RuntimeError(f"failed to initialize free-beta joint posterior: {result.message}")
    map_start = np.asarray(result.x, float)

    rng = np.random.default_rng(seed + 7919)
    pos = np.empty((nwalkers, 2), float)
    if sampling_tau0_parameter == "log_tau0":
        pos[:, 0] = map_start[0] + rng.normal(0.0, 0.5, nwalkers)
    else:
        pos[:, 0] = map_start[0] * np.exp(rng.normal(0.0, 0.35, nwalkers))
    pos[:, 1] = map_start[1] + rng.normal(0.0, 0.35, nwalkers)
    eps_tau = 1.0e-9 * (sampling_tau0_bounds[1] - sampling_tau0_bounds[0])
    eps_beta = 1.0e-9 * (beta_bounds[1] - beta_bounds[0])
    pos[:, 0] = np.clip(
        pos[:, 0],
        sampling_tau0_bounds[0] + eps_tau,
        sampling_tau0_bounds[1] - eps_tau,
    )
    pos[:, 1] = np.clip(pos[:, 1], beta_bounds[0] + eps_beta, beta_bounds[1] - eps_beta)
    if not np.isfinite([log_prob(row) for row in pos]).all():
        raise RuntimeError("non-finite initial state in free-beta MCMC")

    np.random.seed(seed + 104729)
    sampler = emcee.EnsembleSampler(nwalkers, 2, log_prob)
    started = time.time()
    sampler.run_mcmc(
        pos,
        steps,
        progress=bool(config.get("free_mcmc_progress", False)),
    )
    sampling_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    samples = np.array([to_physical(row) for row in sampling_samples])
    log_probability = np.array([physical_log_prob(row) for row in samples])
    if len(samples) == 0 or not np.isfinite(samples).all() or not np.isfinite(log_probability).all():
        raise RuntimeError("free-beta MCMC produced no finite posterior samples")

    try:
        autocorr = sampler.get_autocorr_time(discard=burn, tol=0)
    except Exception:
        autocorr = np.full(2, np.nan)
    best = int(np.argmax(log_probability))
    tau_coordinate = (
        (sampling_samples[:, 0] - sampling_tau0_bounds[0])
        / (sampling_tau0_bounds[1] - sampling_tau0_bounds[0])
    )
    beta_coordinate = (
        (samples[:, 1] - beta_bounds[0])
        / (beta_bounds[1] - beta_bounds[0])
    )
    edge_fraction = float(np.mean(
        (tau_coordinate < 0.01)
        | (tau_coordinate > 0.99)
        | (beta_coordinate < 0.01)
        | (beta_coordinate > 0.99)
    ))
    diagnostics = {
        "inference": "joint_mcmc",
        "sampling_tau0_parameter": sampling_tau0_parameter,
        "likelihood": likelihood,
        "regularized_error_sides": regularized_error_sides,
        "tau0_prior": tau0_prior,
        "tau0_prior_low": tau0_bounds[0],
        "tau0_prior_high": tau0_bounds[1],
        "beta_prior": "uniform",
        "beta_prior_low": beta_bounds[0],
        "beta_prior_high": beta_bounds[1],
        "nwalkers": nwalkers,
        "steps": steps,
        "burn": burn,
        "thin": thin,
        "posterior_samples": len(samples),
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_tau0": float(autocorr[0]),
        "autocorr_beta": float(autocorr[1]),
        "chain_tau0_per_autocorr": float((steps - burn) / autocorr[0]),
        "chain_beta_per_autocorr": float((steps - burn) / autocorr[1]),
        "tau0_beta_correlation": float(np.corrcoef(samples.T)[0, 1]),
        "log_tau0_beta_correlation": float(
            np.corrcoef(np.log(samples[:, 0]), samples[:, 1])[0, 1]
        ),
        "edge_fraction": edge_fraction,
        "map_tau0": float(samples[best, 0]),
        "map_beta": float(samples[best, 1]),
        "map_log_probability": float(log_probability[best]),
        "elapsed_sec": time.time() - started,
    }
    return samples, log_probability, diagnostics


def posterior_input(source_dir: Path, cfg: dict, lt: dict) -> Path:
    if lt.get("posterior_input"):
        return rel(source_dir, lt["posterior_input"])
    if lt.get("input"):
        p = rel(source_dir, lt["input"])
        if p.suffix == ".txt":
            return p
    return mica_result_dir(source_dir, cfg) / "mica_tf0_posterior.txt"


def reference(cfg: dict, lt: dict, posterior: pd.DataFrame) -> tuple[str, float]:
    mica = cfg.get("mica_round1", {}) or {}
    band = str(lt.get("lambda0_band", mica.get("driver", cfg.get("reference_band"))))
    if lt.get("lambda0") is not None:
        return band, float(lt["lambda0"])
    waves = cfg.get("bands", {}) or {}
    if band in waves:
        return band, float(waves[band])
    return band, float(posterior.loc[posterior["band"].astype(str) == band, "lambda_obs"].iloc[0])


def configured_response_bands(cfg: dict) -> list[str]:
    """Return the response-band selection used by MICA and downstream fits."""
    mica = cfg.get("mica_round1", {}) or {}
    values = cfg.get("response_bands")
    if values is None:
        values = mica.get("responses", [])
    return list(dict.fromkeys(str(band) for band in (values or [])))


def normalized_model(value) -> str:
    return MODEL_ALIASES.get(str(value), str(value))


def distance_lag_model(cfg: dict, lt: dict, models: list[str]) -> str:
    """
    Select the absolute-lag model used downstream by disk_distance.py.

    If lambda_tau.absolute_lag_model is omitted, infer beta=1/b from
    hubble_distance.disk_b. The legacy fallback is beta=4/3.
    """
    hd = cfg.get("hubble_distance", {}) or {}
    requested = lt.get("absolute_lag_model")
    disk_b = hd.get("disk_b")

    inferred = None
    if disk_b is not None:
        disk_b = float(disk_b)
        if disk_b <= 0.0:
            raise ValueError("hubble_distance.disk_b must be positive")
        beta = 1.0 / disk_b
        if np.isclose(beta, 4.0 / 3.0, rtol=0.0, atol=1.0e-10):
            inferred = "4/3"
        elif np.isclose(beta, 2.0, rtol=0.0, atol=1.0e-10):
            inferred = "2"
        else:
            raise ValueError(
                f"no fixed fit model is configured for beta=1/disk_b={beta:.8g}"
            )

    selected = normalized_model(requested) if requested is not None else (inferred or "4/3")
    if selected not in models:
        raise ValueError(
            f"absolute-lag model {selected!r} is not in lambda_tau.fit_models"
        )
    if inferred is not None and selected != inferred:
        raise ValueError(
            "lambda_tau.absolute_lag_model is inconsistent with "
            f"hubble_distance.disk_b={disk_b:.8g}; expected {inferred!r}"
        )
    return selected


def posterior_matrix(posterior, cfg, ref_band, nsamp, seed):
    rng = np.random.default_rng(seed)
    waves_cfg = cfg.get("bands", {}) or {}

    posterior = posterior.copy()
    posterior["band"] = posterior["band"].astype(str)
    grouped = {band: data for band, data in posterior.groupby("band", sort=False)}

    selected = configured_response_bands(cfg)
    if not selected:
        selected = [band for band in grouped if band != ref_band]
    selected = [band for band in selected if band != ref_band]

    missing = [band for band in selected if band not in grouped]
    if missing:
        raise ValueError(
            "response_bands missing from MICA posterior: " + ", ".join(missing)
        )

    groups = []
    mica = cfg.get("mica_round1", {}) or {}
    expected_model = model_label(mica.get("type_tf", "gaussian"))
    expected_driver = str(mica.get("driver", ref_band))
    expected_ncomp = ncomp_from_config(mica)
    for band in selected:
        data = grouped[band].copy()
        if "sample" in data and data["sample"].duplicated().any():
            raise ValueError(
                f"response band {band} contains duplicate sample identifiers; "
                "this usually means multiple MICA runs were mixed"
            )

        expected = {
            "driver_band": expected_driver,
            "model": expected_model,
            "ncomp": expected_ncomp,
        }
        for column, value in expected.items():
            if column not in data:
                warnings.warn(
                    f"MICA posterior has no {column!r} metadata; "
                    "model uniqueness cannot be fully verified",
                    RuntimeWarning,
                )
                continue
            values = data[column].dropna().astype(str).unique().tolist()
            if len(values) != 1:
                raise ValueError(
                    f"response band {band} contains multiple {column} values: {values}"
                )
            observed = model_label(values[0]) if column == "model" else values[0]
            if str(observed) != str(value):
                raise ValueError(
                    f"response band {band} has {column}={observed!r}, "
                    f"expected {value!r}"
                )

        tau = pd.to_numeric(data["tau"], errors="coerce").to_numpy(float)
        if "weight" in data:
            weight = pd.to_numeric(data["weight"], errors="coerce").to_numpy(float)
        else:
            warnings.warn(
                f"response band {band} has no posterior weights; using equal weights",
                RuntimeWarning,
            )
            weight = np.ones(len(data), float)
        valid = np.isfinite(tau) & np.isfinite(weight) & (weight > 0.0)
        tau = tau[valid]
        weight = weight[valid]
        if len(tau) == 0:
            raise ValueError(f"no finite MICA lag samples for response band: {band}")
        weight /= np.sum(weight)

        if band in waves_cfg:
            wave = float(waves_cfg[band])
        else:
            wave = float(pd.to_numeric(data["lambda_obs"], errors="coerce").dropna().iloc[0])
        groups.append((band, wave, tau, weight))

    if not groups:
        raise ValueError("no response bands selected for wavelength-lag fitting")

    groups.sort(key=lambda item: item[1])
    bands = [item[0] for item in groups]
    waves = np.array([item[1] for item in groups], float)
    lag_quantiles = np.array([
        weighted_quantile(item[2], item[3])
        for item in groups
    ])
    median_tau = lag_quantiles[:, 1]
    err_low = median_tau - lag_quantiles[:, 0]
    err_high = lag_quantiles[:, 2] - median_tau
    draws = np.column_stack([
        sample_band(item[2], item[3], nsamp, rng)
        for item in groups
    ])
    sigma = 0.5 * (err_low + err_high)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0.0):
        bad = [bands[i] for i in np.where(~np.isfinite(sigma) | (sigma <= 0.0))[0]]
        raise ValueError("invalid lag uncertainty for response band(s): " + ", ".join(bad))

    return bands, waves, draws, sigma, median_tau, err_low, err_high


def second_moment_config(lt: dict) -> dict:
    sm = dict(lt.get("second_moment", {}) or {})
    inherited = (
        "nsamp",
        "seed",
        "free_likelihood",
        "free_mcmc_nwalkers",
        "free_mcmc_steps",
        "free_mcmc_burn_frac",
        "free_mcmc_thin",
        "free_mcmc_progress",
    )
    for key in inherited:
        if key not in sm and key in lt:
            sm[key] = lt[key]
    return sm


def component_posterior_input(source_dir: Path, cfg: dict, sm: dict) -> Path:
    if sm.get("posterior_input"):
        return rel(source_dir, sm["posterior_input"])
    return mica_result_dir(source_dir, cfg) / "mica_component_posterior.txt"


def moment0_config(lt: dict) -> dict:
    m0 = dict(lt.get("moment0", {}) or {})
    for key in ("nsamp", "seed"):
        if key not in m0 and key in lt:
            m0[key] = lt[key]
    return m0


def run_moment0(
    source_dir: Path,
    cfg: dict,
    lt: dict,
    ref_band: str,
    lambda0: float,
    result_dir: Path | None = None,
):
    """Fit the physical comp0 transfer-function area as a relative SED."""
    m0 = moment0_config(lt)
    if not bool(m0.get("enabled", False)):
        return

    inp = component_posterior_input(source_dir, cfg, m0)
    if not inp.is_file():
        raise FileNotFoundError(
            f"missing MICA component posterior: {inp}; run collect_mica_results.py first"
        )
    posterior = pd.read_csv(inp, sep=r"\s+")
    if "gain" not in posterior:
        raise ValueError(
            "MICA component posterior has no physical 'gain' column; "
            "rerun collect_mica_results.py with flux-scale propagation"
        )
    posterior = posterior[posterior["component"].astype(str).eq("tf0")].copy()
    posterior["tau"] = pd.to_numeric(posterior["gain"], errors="coerce")

    nsamp = int(m0.get("nsamp", lt.get("nsamp", 10000)))
    seed = int(m0.get("seed", lt.get("seed", 12345)))
    bands, waves, draws, sigma, median, err_low, err_high = posterior_matrix(
        posterior, cfg, ref_band, nsamp, seed
    )
    if np.any(draws <= 0.0) or np.any(median <= 0.0):
        raise ValueError("comp0 physical transfer-function gains must be positive")

    models = [str(value).lower() for value in m0.get(
        "fit_models", ["standard", "slim", "free"]
    )]
    unknown = [name for name in models if name not in MOMENT0_MODELS]
    if unknown:
        raise ValueError("unknown moment0 fit model(s): " + ", ".join(unknown))
    if len(set(models)) != len(models):
        raise ValueError("moment0 fit_models contains duplicates")

    x = np.log(waves / float(lambda0))
    sigma_log = np.clip(sigma / median, 1.0e-8, None)
    fit_weight = 1.0 / np.square(sigma_log)
    denominator = float(np.sum(fit_weight * np.square(x)))
    if denominator <= 0.0:
        raise ValueError("moment0 free fit requires wavelengths distinct from lambda0")
    eta_free = np.sum(
        fit_weight[None, :] * x[None, :] * np.log(draws),
        axis=1,
    ) / denominator

    source = str(cfg.get("source", source_dir.name))
    if m0.get("result_dir"):
        out_dir = rel(source_dir, m0["result_dir"])
    else:
        out_dir = (result_dir or mica_result_dir(source_dir, cfg)) / "moment0"
    out_dir.mkdir(parents=True, exist_ok=True)

    machine = posterior.copy()
    machine["moment0"] = machine["gain"]
    machine.to_csv(
        out_dir / "mica_moment0_posterior.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )

    point_rows = [{
        "band": ref_band,
        "wavelength": lambda0,
        "moment0": 1.0,
        "moment0_err_low": 0.0,
        "moment0_err_high": 0.0,
        "is_ref": True,
    }]
    for index, band in enumerate(bands):
        point_rows.append({
            "band": band,
            "wavelength": waves[index],
            "moment0": median[index],
            "moment0_err_low": err_low[index],
            "moment0_err_high": err_high[index],
            "is_ref": False,
        })
    points = pd.DataFrame(point_rows).sort_values("wavelength")

    summary_rows = []
    eta_posteriors = {}
    for model_name in models:
        if model_name == "free":
            eta_samples = eta_free
            eta, eta_low, eta_high = q68(eta_samples)
            n_parameter = 1
        else:
            eta = float(MOMENT0_MODELS[model_name]["eta"])
            eta_low = eta_high = 0.0
            eta_samples = np.full(nsamp, eta)
            n_parameter = 0
        eta_posteriors[model_name] = eta_samples
        prediction = (waves / float(lambda0)) ** eta
        chi2 = float(np.sum(np.square((median - prediction) / sigma)))
        dof = len(waves) - n_parameter
        summary_rows.append({
            "source": source,
            "component": "tf0",
            "fit_model": model_name,
            "eta": eta,
            "eta_err_low": eta_low,
            "eta_err_high": eta_high,
            "normalization_at_lambda0": 1.0,
            "lambda0": lambda0,
            "ref_band": ref_band,
            "chi2": chi2,
            "dof": dof,
            "chi2_dof": chi2 / dof if dof > 0 else np.nan,
            "n_used": len(waves),
        })
    summary = pd.DataFrame(summary_rows)
    points.to_csv(out_dir / f"{source}_lambda_moment0_points.csv", index=False)
    summary.to_csv(out_dir / f"{source}_lambda_moment0_fit_summary.csv", index=False)
    pd.DataFrame({
        "sample": np.arange(len(eta_free)),
        "eta": eta_free,
    }).to_csv(
        out_dir / "free_eta_posterior.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )

    grid = np.logspace(
        np.log10(points["wavelength"].min() * 0.94),
        np.log10(points["wavelength"].max() * 1.06),
        500,
    )
    colors = {"standard": "#D55E00", "slim": "0.35", "free": "#0072B2"}
    with plt.rc_context({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 13,
    }):
        fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
        response = points.loc[~points["is_ref"]]
        ax.errorbar(
            response["wavelength"],
            response["moment0"],
            yerr=np.vstack([
                response["moment0_err_low"],
                response["moment0_err_high"],
            ]),
            fmt="o",
            color="black",
            capsize=3,
            label="MICA comp0",
            zorder=5,
        )
        ax.scatter([lambda0], [1.0], marker="s", color="black", label=f"{ref_band} anchor")
        for row in summary.itertuples(index=False):
            ax.plot(
                grid,
                (grid / lambda0) ** row.eta,
                color=colors[row.fit_model],
                ls=MOMENT0_MODELS[row.fit_model]["ls"],
                lw=1.8,
                label=MOMENT0_MODELS[row.fit_model]["label"],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Observed wavelength $\lambda$ ($\AA$)")
        ax.set_ylabel(r"Physical comp0 area $G_0$")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.savefig(out_dir / "moment0_comp0.png", dpi=300)
        fig.savefig(out_dir / "moment0_comp0.pdf")
        plt.close(fig)

    print(out_dir / "mica_moment0_posterior.txt")
    print(out_dir / f"{source}_lambda_moment0_points.csv")
    print(out_dir / f"{source}_lambda_moment0_fit_summary.csv")
    print(out_dir / "moment0_comp0.png")


def transfer_moment2_posterior(component_posterior: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Calculate component and total central second moments sample by sample."""
    mica = cfg.get("mica_round1", {}) or {}
    expected_model = model_label(mica.get("type_tf", "gaussian"))
    expected_ncomp = ncomp_from_config(mica)
    if expected_model not in {"gaussian", "gamma"} or expected_ncomp != 2:
        raise ValueError(
            "second-moment fitting currently requires an active two-component "
            "Gaussian or k=2 gamma MICA model"
        )

    required = {
        "run_name",
        "driver_band",
        "band",
        "model",
        "ncomp",
        "selection",
        "retained_weight",
        "lambda_obs",
        "sample",
        "component",
        "center",
        "width",
        "amp",
        "weight",
    }
    missing_columns = sorted(required.difference(component_posterior.columns))
    if missing_columns:
        raise ValueError(
            "MICA component posterior is missing columns: "
            + ", ".join(missing_columns)
        )

    posterior = component_posterior.copy()
    posterior["band"] = posterior["band"].astype(str)
    posterior["component"] = posterior["component"].astype(str)
    selected = configured_response_bands(cfg)
    if not selected:
        selected = posterior["band"].drop_duplicates().tolist()
    missing_bands = [band for band in selected if band not in set(posterior["band"])]
    if missing_bands:
        raise ValueError(
            "response_bands missing from MICA component posterior: "
            + ", ".join(missing_bands)
        )

    metadata_columns = [
        "run_name",
        "driver_band",
        "band",
        "model",
        "ncomp",
        "selection",
        "retained_weight",
        "lambda_obs",
        "sample",
    ]
    output = []
    for band in selected:
        data = posterior[posterior["band"].eq(band)].copy()
        observed_components = set(data["component"])
        if observed_components != {"tf0", "tf1"}:
            raise ValueError(
                f"response band {band} must contain exactly tf0 and tf1; "
                f"found {sorted(observed_components)}"
            )
        for component in ("tf0", "tf1"):
            subset = data[data["component"].eq(component)]
            if subset["sample"].duplicated().any():
                raise ValueError(
                    f"response band {band}, {component} contains duplicate sample identifiers"
                )

        tf0 = data[data["component"].eq("tf0")].sort_values("sample").reset_index(drop=True)
        tf1 = data[data["component"].eq("tf1")].sort_values("sample").reset_index(drop=True)
        if not np.array_equal(tf0["sample"].to_numpy(), tf1["sample"].to_numpy()):
            raise ValueError(f"tf0/tf1 posterior samples are not aligned for {band}")
        for column in (
            "run_name",
            "driver_band",
            "band",
            "model",
            "ncomp",
            "selection",
            "retained_weight",
            "lambda_obs",
            "weight",
        ):
            left = tf0[column].to_numpy()
            right = tf1[column].to_numpy()
            if np.issubdtype(left.dtype, np.number):
                equal = np.allclose(left, right, rtol=1.0e-12, atol=0.0, equal_nan=True)
            else:
                equal = np.array_equal(left.astype(str), right.astype(str))
            if not equal:
                raise ValueError(f"tf0/tf1 {column} values do not match for {band}")

        model_values = tf0["model"].astype(str).map(model_label).unique().tolist()
        ncomp_values = pd.to_numeric(tf0["ncomp"], errors="raise").unique().tolist()
        if model_values != [expected_model] or ncomp_values != [2]:
            raise ValueError(
                f"response band {band} is not a unique two-component "
                f"{expected_model} posterior"
            )

        centers = np.vstack([
            pd.to_numeric(tf0["center"], errors="coerce").to_numpy(float),
            pd.to_numeric(tf1["center"], errors="coerce").to_numpy(float),
        ])
        widths = np.vstack([
            pd.to_numeric(tf0["width"], errors="coerce").to_numpy(float),
            pd.to_numeric(tf1["width"], errors="coerce").to_numpy(float),
        ])
        amplitudes = np.vstack([
            pd.to_numeric(tf0["amp"], errors="coerce").to_numpy(float),
            pd.to_numeric(tf1["amp"], errors="coerce").to_numpy(float),
        ])
        weights = pd.to_numeric(tf0["weight"], errors="coerce").to_numpy(float)
        if not np.isfinite(weights).all() or np.any(weights < 0.0) or np.sum(weights) <= 0.0:
            raise ValueError(f"invalid posterior weights for response band {band}")

        (
            component_variance,
            component_centroid,
            total_centroid,
            total_variance,
            fractions,
        ) = (
            transfer_mixture_moments(
                centers, widths, amplitudes, expected_model
            )
        )
        base = tf0[metadata_columns].copy()
        base["weight"] = weights
        for k, component in enumerate(("tf0", "tf1")):
            frame = base.copy()
            frame["component"] = component
            frame["moment1"] = component_centroid[k]
            frame["moment2"] = component_variance[k]
            frame["amp_frac"] = fractions[k]
            output.append(frame)
        frame = base.copy()
        frame["component"] = "total"
        frame["moment1"] = total_centroid
        frame["moment2"] = total_variance
        frame["amp_frac"] = 1.0
        output.append(frame)

    result = pd.concat(output, ignore_index=True)
    if not np.isfinite(result["moment2"]).all() or np.any(result["moment2"] < 0.0):
        raise ValueError("calculated second moments must be finite and non-negative")
    component_order = {"tf0": 0, "tf1": 1, "total": 2}
    result["_component_order"] = result["component"].map(component_order)
    return result.sort_values(
        ["lambda_obs", "sample", "_component_order"]
    ).drop(columns="_component_order").reset_index(drop=True)


def gaussian_moment2_posterior(component_posterior: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Backward-compatible entry point retained for existing callers/tests."""
    return transfer_moment2_posterior(component_posterior, cfg)


def moment2_posterior_matrix(moment_posterior, cfg, ref_band, component, nsamp, seed):
    selected = moment_posterior[moment_posterior["component"].eq(component)].copy()
    if selected.empty:
        raise ValueError(f"no second-moment posterior samples for {component}")
    selected["tau"] = selected["moment2"]
    return posterior_matrix(selected, cfg, ref_band, nsamp, seed)


def normalized_moment2_model(value) -> str:
    return MOMENT2_MODEL_ALIASES.get(str(value), str(value))


def joint_free_gamma_posterior(
    waves,
    median_moment2,
    err_low,
    err_high,
    lambda0,
    config,
    seed,
):
    """Use the validated two-dimensional power-law sampler for (M20, gamma)."""
    mapped = dict(config)
    mapped["tau0_bounds"] = config.get("moment20_bounds", [1.0e-6, 1.0e4])
    mapped["free_beta_bounds"] = config.get("free_gamma_bounds", [0.01, 8.0])
    mapped["free_tau0_prior"] = config.get("moment20_prior", "log_uniform")
    samples, log_probability, diagnostics = joint_free_beta_posterior(
        waves,
        median_moment2,
        err_low,
        err_high,
        lambda0,
        mapped,
        seed,
    )

    renamed = {}
    for key, value in diagnostics.items():
        new_key = key.replace("tau0", "moment20").replace("beta", "gamma")
        if isinstance(value, str):
            value = value.replace("tau0", "moment20").replace("beta", "gamma")
        renamed[new_key] = value
    return samples, log_probability, renamed


def moment2_summary_rows(
    component,
    models,
    moment20_post,
    gamma_post,
    waves,
    median_moment2,
    sigma,
    lambda0,
):
    rows = []
    for key in models:
        moment20 = moment20_post[key]
        gamma = gamma_post[key]
        m50, mlo, mhi = q68(moment20)
        g50, glo, ghi = q68(gamma)
        predicted = moment2_model(waves, m50, g50, lambda0)
        chi2 = float(np.sum(((median_moment2 - predicted) / sigma) ** 2))
        ndim = 2 if key == "free" else 1
        dof = len(waves) - ndim
        rows.append({
            "component": component,
            "fit": MOMENT2_MODELS[key]["fit"],
            "fit_model": key,
            "moment20": m50,
            "moment20_err_low": mlo,
            "moment20_err_high": mhi,
            "gamma": g50,
            "gamma_err_low": glo if key == "free" else 0.0,
            "gamma_err_high": ghi if key == "free" else 0.0,
            "chi2": chi2,
            "dof": dof,
            "chi2_dof": chi2 / dof if dof > 0 else np.nan,
            "n_used": len(waves),
            "lambda0": lambda0,
            "inference": "joint_mcmc" if key == "free" else "posterior_propagation",
        })
    return rows


def plot_moment2_fit(points, fit_summary, lambda0, out_dir):
    fig, axes = plt.subplots(
        1,
        len(MOMENT2_COMPONENTS),
        figsize=(15.0, 4.6),
        constrained_layout=True,
    )
    for ax, component in zip(axes, MOMENT2_COMPONENTS):
        component_points = points[points["component"].eq(component)]
        response = component_points[~component_points["is_ref"]]
        ax.errorbar(
            response["wavelength"],
            response["moment2"],
            yerr=np.vstack([
                response["moment2_err_low"],
                response["moment2_err_high"],
            ]),
            fmt="o",
            capsize=3,
            label=component,
        )
        ref = component_points[component_points["is_ref"]]
        ax.scatter(ref["wavelength"], ref["moment2"], marker="s", color="black")
        grid = np.linspace(
            component_points["wavelength"].min() * 0.94,
            component_points["wavelength"].max() * 1.06,
            600,
        )
        rows = fit_summary[fit_summary["component"].eq(component)]
        for row in rows.itertuples(index=False):
            key = str(row.fit_model)
            ax.plot(
                grid,
                moment2_model(grid, row.moment20, row.gamma, lambda0),
                ls=MOMENT2_MODELS[key]["ls"],
                lw=1.7,
                label=MOMENT2_MODELS[key]["label"],
            )
        ax.axhline(0.0, lw=0.8, ls=":")
        ax.axvline(lambda0, lw=0.8, ls=":")
        positive = response.loc[response["moment2"] > 0.0, "moment2"].to_numpy(float)
        if len(positive):
            ax.set_yscale(
                "symlog",
                linthresh=max(1.0e-3, 0.5 * float(np.min(positive))),
            )
        ax.set_title(component)
        ax.set_xlabel(r"Wavelength $\lambda$ ($\AA$)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel(r"Relative second central moment (day$^2$)")
    fig.savefig(out_dir / "moment2_components.png", dpi=300)
    fig.savefig(out_dir / "moment2_components.pdf")
    plt.close(fig)


def run_second_moment(
    source_dir: Path,
    cfg: dict,
    lt: dict,
    ref_band: str,
    lambda0: float,
    result_dir: Path | None = None,
):
    sm = second_moment_config(lt)
    if not bool(sm.get("enabled", False)):
        return

    nsamp = int(sm.get("nsamp", lt.get("nsamp", 10000)))
    seed = int(sm.get("seed", lt.get("seed", 12345)))
    models = [
        normalized_moment2_model(value)
        for value in sm.get("fit_models", ["8/3", "4", "free"])
    ]
    unknown = [key for key in models if key not in MOMENT2_MODELS]
    if unknown:
        raise ValueError("unknown second-moment fit model(s): " + ", ".join(unknown))
    if len(set(models)) != len(models):
        raise ValueError("second-moment fit_models contains duplicates")

    inp = component_posterior_input(source_dir, cfg, sm)
    if not inp.is_file():
        raise FileNotFoundError(
            f"missing MICA component posterior: {inp}; run collect_mica_results.py first"
        )
    component_posterior = pd.read_csv(inp, sep=r"\s+")
    moment_posterior = transfer_moment2_posterior(component_posterior, cfg)

    if sm.get("result_dir"):
        out_dir = rel(source_dir, sm["result_dir"])
    else:
        out_dir = (result_dir or mica_result_dir(source_dir, cfg)) / "moment2"
    out_dir.mkdir(parents=True, exist_ok=True)
    moment_posterior.to_csv(
        out_dir / "mica_moment2_posterior.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )

    source = cfg.get("source", source_dir.name)
    fit_rows = []
    point_rows = []
    free_posterior_rows = []
    free_diagnostic_rows = []
    component_results = {}
    component_seed_offsets = {"tf0": 100003, "tf1": 200003, "total": 300007}

    for component in MOMENT2_COMPONENTS:
        component_seed = seed + component_seed_offsets[component]
        (
            bands,
            waves,
            draws,
            sigma,
            median_moment2,
            err_low,
            err_high,
        ) = moment2_posterior_matrix(
            moment_posterior,
            cfg,
            ref_band,
            component,
            nsamp,
            component_seed,
        )
        if "free" in models and len(bands) < 2:
            raise ValueError(
                f"free-gamma fitting requires at least two response bands for {component}"
            )

        moment20_post = {}
        gamma_post = {}
        free_log_probability = None
        free_diagnostics = None
        for key in models:
            if key in {"8/3", "4"}:
                gamma = MOMENT2_MODELS[key]["gamma"]
                moment20_post[key] = fixed_beta_posterior(
                    draws,
                    waves,
                    sigma,
                    lambda0,
                    gamma,
                )
                gamma_post[key] = np.full(nsamp, gamma)
            elif key == "free":
                joint_samples, free_log_probability, free_diagnostics = (
                    joint_free_gamma_posterior(
                        waves,
                        median_moment2,
                        err_low,
                        err_high,
                        lambda0,
                        sm,
                        component_seed,
                    )
                )
                moment20_post[key] = joint_samples[:, 0]
                gamma_post[key] = joint_samples[:, 1]

        rows = moment2_summary_rows(
            component,
            models,
            moment20_post,
            gamma_post,
            waves,
            median_moment2,
            sigma,
            lambda0,
        )
        for row in rows:
            row.update({
                "ref_band": ref_band,
                "map_moment20": np.nan,
                "map_gamma": np.nan,
                "moment20_gamma_correlation": np.nan,
            })
            if row["fit_model"] == "free" and free_diagnostics is not None:
                row["map_moment20"] = free_diagnostics["map_moment20"]
                row["map_gamma"] = free_diagnostics["map_gamma"]
                row["moment20_gamma_correlation"] = free_diagnostics[
                    "moment20_gamma_correlation"
                ]
        fit_rows.extend(rows)

        for i, band in enumerate(bands):
            point_rows.append({
                "source": source,
                "drive_band": ref_band,
                "resp_band": band,
                "band": band,
                "lambda_obs": waves[i],
                "wavelength": waves[i],
                "component": component,
                "moment2": median_moment2[i],
                "moment2_err_low": err_low[i],
                "moment2_err_high": err_high[i],
                "is_ref": False,
            })
        point_rows.append({
            "source": source,
            "drive_band": ref_band,
            "resp_band": ref_band,
            "band": ref_band,
            "lambda_obs": lambda0,
            "wavelength": lambda0,
            "component": component,
            "moment2": 0.0,
            "moment2_err_low": 0.0,
            "moment2_err_high": 0.0,
            "is_ref": True,
        })

        if free_diagnostics is not None:
            free_frame = pd.DataFrame({
                "component": component,
                "sample": np.arange(len(moment20_post["free"])),
                "moment20": moment20_post["free"],
                "gamma": gamma_post["free"],
                "log_probability": free_log_probability,
            })
            free_posterior_rows.append(free_frame)
            free_diagnostic_rows.append({"component": component, **free_diagnostics})

        component_results[component] = (moment20_post, gamma_post)

    fit_summary = pd.DataFrame(fit_rows)
    points = pd.DataFrame(point_rows).sort_values(["component", "wavelength"])
    fit_summary.to_csv(
        out_dir / f"{source}_lambda_moment2_fit_summary.csv",
        index=False,
    )
    points.to_csv(out_dir / f"{source}_lambda_moment2_points.csv", index=False)

    moment20_table = pd.DataFrame({"sample": np.arange(nsamp)})
    for component in MOMENT2_COMPONENTS:
        moment20_post, gamma_post = component_results[component]
        for key in models:
            values = moment20_post[key]
            gamma_values = gamma_post[key]
            if len(values) != nsamp:
                rng = np.random.default_rng(
                    seed + component_seed_offsets[component] + 15485863
                )
                index = rng.choice(len(values), nsamp, replace=len(values) < nsamp)
                values = values[index]
                gamma_values = gamma_values[index]
            suffix = MOMENT2_MODELS[key]["fit"]
            moment20_table[f"moment20_{component}_{suffix}"] = values
            if key == "free":
                moment20_table[f"gamma_{component}_free"] = gamma_values
    moment20_table.to_csv(
        out_dir / "moment20_posterior.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )

    if free_posterior_rows:
        pd.concat(free_posterior_rows, ignore_index=True).to_csv(
            out_dir / "free_gamma_joint_posterior.txt",
            sep=" ",
            index=False,
            float_format="%.10e",
        )
        pd.DataFrame(free_diagnostic_rows).to_csv(
            out_dir / "free_gamma_joint_summary.csv",
            index=False,
        )
    plot_moment2_fit(points, fit_summary, lambda0, out_dir)

    print(out_dir / "mica_moment2_posterior.txt")
    print(out_dir / f"{source}_lambda_moment2_points.csv")
    print(out_dir / f"{source}_lambda_moment2_fit_summary.csv")
    print(out_dir / "moment20_posterior.txt")
    if free_posterior_rows:
        print(out_dir / "free_gamma_joint_posterior.txt")
        print(out_dir / "free_gamma_joint_summary.csv")
    print(out_dir / "moment2_components.png")


def summary_rows(models, tau0_post, beta_post, waves, median_tau, sigma, lambda0):
    rows = []
    for key in models:
        tau0 = tau0_post[key]
        beta = beta_post[key]
        t50, tlo, thi = q68(tau0)
        b50, blo, bhi = q68(beta)
        pred = tau_model(waves, t50, b50, lambda0)
        chi2 = float(np.sum(((median_tau - pred) / sigma) ** 2))
        ndim = 2 if key == "free" else 1
        rows.append({
            "component": "tf0",
            "fit": MODELS[key]["fit"],
            "fit_model": key,
            "tau0": t50,
            "tau0_err_low": tlo,
            "tau0_err_high": thi,
            "beta": b50,
            "beta_err_low": blo if key == "free" else 0.0,
            "beta_err_high": bhi if key == "free" else 0.0,
            "chi2": chi2,
            "dof": len(waves) - ndim,
            "chi2_dof": chi2 / (len(waves) - ndim) if len(waves) > ndim else np.nan,
            "n_used": len(waves),
            "lambda0": lambda0,
        })
    return rows


def plot_fit(points, fit_summary, lambda0, out_dir):
    fig, ax = plt.subplots(figsize=(6.3, 4.6), constrained_layout=True)
    response = points[~points["is_ref"]]
    ax.errorbar(
        response["wavelength"],
        response["tau"],
        yerr=np.vstack([response["tau_err_low"], response["tau_err_high"]]),
        fmt="o",
        capsize=3,
        label="MICA tf0",
    )
    ref = points[points["is_ref"]]
    ax.scatter(ref["wavelength"], ref["tau"], marker="s")

    grid = np.linspace(points["wavelength"].min() * 0.94, points["wavelength"].max() * 1.06, 600)
    for row in fit_summary.itertuples(index=False):
        key = str(row.fit_model)
        selected = bool(row.used_for_distance)
        label = MODELS[key]["label"] + (" (distance/H0)" if selected else "")
        ax.plot(
            grid,
            tau_model(grid, row.tau0, row.beta, lambda0),
            ls=MODELS[key]["ls"],
            lw=2.4 if selected else 1.2,
            alpha=1.0 if selected else 0.65,
            label=label,
        )

    ax.axhline(0.0, lw=0.8, ls=":")
    ax.axvline(lambda0, lw=0.8, ls=":")
    ax.set_xlabel(r"Wavelength $\lambda$ ($\AA$)")
    ax.set_ylabel("Relative lag (days)")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "tau_tf0.png", dpi=300)
    fig.savefig(out_dir / "tau_tf0.pdf")
    plt.close(fig)


def run(source_dir: Path, cfg: dict, nsamp: int, seed: int, config_section="lambda_tau"):
    lt = cfg.get(config_section, {}) or {}
    inp = posterior_input(source_dir, cfg, lt)
    out_dir = lambda_tau_result_dir(source_dir, cfg, lt)

    posterior = pd.read_csv(inp, sep=r"\s+")
    ref_band, lambda0 = reference(cfg, lt, posterior)
    (
        bands,
        waves,
        draws,
        sigma,
        median_tau,
        err_low,
        err_high,
    ) = posterior_matrix(posterior, cfg, ref_band, nsamp, seed)
    models = [normalized_model(x) for x in lt.get("fit_models", ["4/3", "2", "free"])]
    if "free" in models and len(bands) < 2:
        raise ValueError("free-beta fitting requires at least two response bands")

    tau0_post = {}
    beta_post = {}
    free_log_probability = None
    free_diagnostics = None
    for key in models:
        if key in {"4/3", "2"}:
            beta = MODELS[key]["beta"]
            tau0_post[key] = fixed_beta_posterior(draws, waves, sigma, lambda0, beta)
            beta_post[key] = np.full(nsamp, beta)
        elif key == "free":
            joint_samples, free_log_probability, free_diagnostics = joint_free_beta_posterior(
                waves,
                median_tau,
                err_low,
                err_high,
                lambda0,
                lt,
                seed,
            )
            tau0_post[key] = joint_samples[:, 0]
            beta_post[key] = joint_samples[:, 1]
        else:
            raise ValueError(f"unknown fit model: {key}")

    fit_summary = pd.DataFrame(summary_rows(models, tau0_post, beta_post, waves, median_tau, sigma, lambda0))
    fit_summary["inference"] = fit_summary["fit_model"].map(
        lambda key: "joint_mcmc" if key == "free" else "posterior_propagation"
    )
    fit_summary["map_tau0"] = np.nan
    fit_summary["map_beta"] = np.nan
    fit_summary["tau0_beta_correlation"] = np.nan
    if free_diagnostics is not None:
        free_row = fit_summary["fit_model"].eq("free")
        fit_summary.loc[free_row, "map_tau0"] = free_diagnostics["map_tau0"]
        fit_summary.loc[free_row, "map_beta"] = free_diagnostics["map_beta"]
        fit_summary.loc[free_row, "tau0_beta_correlation"] = free_diagnostics[
            "tau0_beta_correlation"
        ]
    fit_summary["ref_band"] = ref_band
    selected_model = distance_lag_model(cfg, lt, models)
    fit_summary["used_for_distance"] = fit_summary["fit_model"].eq(selected_model)

    points = []
    for i, band in enumerate(bands):
        points.append({"source": cfg.get("source", source_dir.name), "drive_band": ref_band, "resp_band": band, "band": band, "lambda_obs": waves[i], "wavelength": waves[i], "component": "tf0", "tf_index": 0, "tau": median_tau[i], "tau_err_low": err_low[i], "tau_err_high": err_high[i], "is_ref": False})
    points.append({"source": cfg.get("source", source_dir.name), "drive_band": ref_band, "resp_band": ref_band, "band": ref_band, "lambda_obs": lambda0, "wavelength": lambda0, "component": "tf0", "tf_index": 0, "tau": 0.0, "tau_err_low": 0.0, "tau_err_high": 0.0, "is_ref": True})
    points = pd.DataFrame(points).sort_values("wavelength")

    all_bands = points["band"].astype(str).tolist()
    all_waves = points["wavelength"].to_numpy(float)

    tau0_table = pd.DataFrame({"sample": np.arange(nsamp)})
    for key in models:
        if len(tau0_post[key]) == nsamp:
            table_tau0 = tau0_post[key]
            table_beta = beta_post[key]
        else:
            rng = np.random.default_rng(seed + 15485863)
            index = rng.choice(len(tau0_post[key]), nsamp, replace=len(tau0_post[key]) < nsamp)
            table_tau0 = tau0_post[key][index]
            table_beta = beta_post[key][index]
        tau0_table[f"tau0_{MODELS[key]['fit']}"] = table_tau0
        if key == "free":
            tau0_table["beta_free"] = table_beta

    absolute_products = {}
    all_absolute_summaries = []
    wave_ratio = all_waves[None, :] / lambda0
    for key in models:
        tau0 = tau0_post[key]
        beta = beta_post[key]
        tau_abs = tau0[:, None] * wave_ratio ** beta[:, None]
        abs_table = pd.DataFrame({
            "sample": np.arange(len(tau0)),
            "tau0": tau0,
            "beta": beta,
            "fit_model": key,
        })
        for i, band in enumerate(all_bands):
            abs_table[f"tau_abs_{band}"] = tau_abs[:, i]

        abs_summary = []
        for i, band in enumerate(all_bands):
            mid, lo, hi = q68(tau_abs[:, i])
            bmid, blo, bhi = q68(beta)
            abs_summary.append({
                "band": band,
                "wavelength": all_waves[i],
                "fit": MODELS[key]["fit"],
                "fit_model": key,
                "beta": bmid,
                "beta_err_low": blo if key == "free" else 0.0,
                "beta_err_high": bhi if key == "free" else 0.0,
                "used_for_distance": key == selected_model,
                "tau_abs": mid,
                "tau_abs_err_low": lo,
                "tau_abs_err_high": hi,
            })
        absolute_products[key] = (abs_table, pd.DataFrame(abs_summary))
        all_absolute_summaries.extend(abs_summary)

    selected_abs_table, selected_abs_summary = absolute_products[selected_model]

    source = cfg.get("source", source_dir.name)
    fit_summary.to_csv(out_dir / "fit_summary.csv", index=False)
    fit_summary.to_csv(out_dir / f"{source}_lambda_tau_fit_summary.csv", index=False)
    points.to_csv(out_dir / f"{source}_lambda_tau_points.csv", index=False)
    tau0_table.to_csv(out_dir / "tau0_posterior.txt", sep=" ", index=False, float_format="%.10e")
    if free_diagnostics is not None:
        free_posterior = pd.DataFrame({
            "sample": np.arange(len(tau0_post["free"])),
            "tau0": tau0_post["free"],
            "beta": beta_post["free"],
            "log_probability": free_log_probability,
        })
        free_posterior.to_csv(
            out_dir / "free_beta_joint_posterior.txt",
            sep=" ",
            index=False,
            float_format="%.10e",
        )
        pd.DataFrame([free_diagnostics]).to_csv(
            out_dir / "free_beta_joint_summary.csv",
            index=False,
        )
    selected_abs_table.to_csv(
        out_dir / "absolute_lag_posterior.txt",
        sep=" ",
        index=False,
        float_format="%.10e",
    )
    selected_abs_summary.to_csv(out_dir / "absolute_lag_summary.csv", index=False)
    pd.DataFrame(all_absolute_summaries).to_csv(
        out_dir / "absolute_lag_summary_all_models.csv",
        index=False,
    )
    for key, (abs_table, abs_summary) in absolute_products.items():
        suffix = MODELS[key]["fit"]
        abs_table.to_csv(
            out_dir / f"absolute_lag_posterior_{suffix}.txt",
            sep=" ",
            index=False,
            float_format="%.10e",
        )
        abs_summary.to_csv(
            out_dir / f"absolute_lag_summary_{suffix}.csv",
            index=False,
        )
    plot_fit(points, fit_summary, lambda0, out_dir)
    run_moment0(source_dir, cfg, lt, ref_band, lambda0, out_dir)
    run_second_moment(source_dir, cfg, lt, ref_band, lambda0, out_dir)

    print(
        "distance/H0 absolute-lag model: "
        f"{selected_model} (beta median={np.median(beta_post[selected_model]):.8g})"
    )
    print(out_dir / "tau0_posterior.txt")
    print(out_dir / "absolute_lag_posterior.txt")
    print(out_dir / "absolute_lag_summary.csv")
    print(out_dir / "absolute_lag_summary_all_models.csv")
    if free_diagnostics is not None:
        print(out_dir / "free_beta_joint_posterior.txt")
        print(out_dir / "free_beta_joint_summary.csv")


def branch_analysis_config(
    source_dir: Path,
    cfg: dict,
    config_section: str,
    branch_id: str,
) -> dict:
    """Point lag and second-moment analysis at one collected branch."""
    definitions = branch_definitions(cfg.get("mica_round1", {}) or {})
    if branch_id not in definitions:
        available = ", ".join(definitions) if definitions else "none"
        raise ValueError(
            f"unknown or disabled posterior branch {branch_id!r}; available: {available}"
        )

    branch_cfg = copy.deepcopy(cfg)
    lt = dict(branch_cfg.get(config_section, {}) or {})
    mica_fit = branch_cfg.get("mica_round1", {}) or {}
    branch_root = (
        mica_result_dir(source_dir, branch_cfg)
        / branch_result_relative_dir(mica_fit, branch_id)
    )
    tf0_path = branch_root / "mica_tf0_posterior.txt"
    component_path = branch_root / "mica_component_posterior.txt"
    if not tf0_path.is_file() or not component_path.is_file():
        raise FileNotFoundError(
            f"missing collected posterior for branch {branch_id!r}; "
            "run collect_mica_results.py first"
        )

    lt["posterior_input"] = str(tf0_path)
    result_subdir = str(lt.get("branch_result_subdir", "")).strip()
    lt["result_dir"] = str(branch_root / result_subdir) if result_subdir else str(branch_root)
    second_moment = dict(lt.get("second_moment", {}) or {})
    second_moment["posterior_input"] = str(component_path)
    moment2_subdir = str(second_moment.get("branch_result_subdir", "")).strip()
    if moment2_subdir:
        second_moment["result_dir"] = str(branch_root / moment2_subdir)
    lt["second_moment"] = second_moment
    moment0 = dict(lt.get("moment0", {}) or {})
    moment0["posterior_input"] = str(component_path)
    moment0_subdir = str(moment0.get("branch_result_subdir", "")).strip()
    if moment0_subdir:
        moment0["result_dir"] = str(branch_root / moment0_subdir)
    lt["moment0"] = moment0
    branch_cfg[config_section] = lt
    return branch_cfg


def main():
    ap = argparse.ArgumentParser(description="Fit wavelength-lag relations from the full MICA tf0 posterior.")
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--nsamp", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--config-section", default="lambda_tau")
    branch_group = ap.add_mutually_exclusive_group()
    branch_group.add_argument("--branch", help="fit one configured posterior branch")
    branch_group.add_argument(
        "--all-branches",
        action="store_true",
        help="fit every enabled posterior branch",
    )
    args = ap.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    cfg_path = args.config or source_dir / "config/source_config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = source_dir / cfg_path
    cfg = read_yaml(cfg_path)
    lt = cfg.get(args.config_section, {}) or {}
    base_nsamp = args.nsamp or int(lt.get("nsamp", 10000))
    base_seed = args.seed if args.seed is not None else int(lt.get("seed", 12345))

    if args.all_branches:
        definitions = branch_definitions(cfg.get("mica_round1", {}) or {})
        if not definitions:
            raise ValueError("no enabled posterior branches are configured")
        branch_ids = list(definitions)
    elif args.branch:
        branch_ids = [args.branch]
    else:
        branch_ids = []

    if not branch_ids:
        run(source_dir, cfg, base_nsamp, base_seed, args.config_section)
        return

    for branch_id in branch_ids:
        print(f"\n== posterior branch: {branch_id} ==", flush=True)
        branch_cfg = branch_analysis_config(
            source_dir,
            cfg,
            args.config_section,
            branch_id,
        )
        branch_seed = base_seed + stable_seed_offset(branch_id)
        branch_cfg[args.config_section]["seed"] = branch_seed
        branch_cfg[args.config_section]["nsamp"] = base_nsamp
        run(
            source_dir,
            branch_cfg,
            base_nsamp,
            branch_seed,
            args.config_section,
        )


if __name__ == "__main__":
    main()
