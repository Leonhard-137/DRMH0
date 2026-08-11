#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from posterior_branches import branch_result_relative_dir

C_KM_S = 299792.458
C_AA_S = 2.99792458e18
FLUX_SCALE = 1.0e-15
DEFAULT_DISK_B = 3.0 / 4.0
DEFAULT_INCLINATION_DEG = 40.0
K_B_NORMALIZATION = 13.26
DEFAULT_OM0 = 0.3


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def rel(root: Path, path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def key(band: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(band)).strip("_")


def redshift(cfg: dict) -> float:
    values = [
        cfg.get("redshift"),
        (cfg.get("fflux") or {}).get("sed", {}).get("redshift"),
        (cfg.get("hubble_distance") or {}).get("redshift"),
    ]
    for value in values:
        if value is not None:
            return float(value)
    raise ValueError("missing redshift")


def bands_and_waves(cfg: dict) -> tuple[list[str], dict[str, float]]:
    """Use a distance-specific band list, or the pipeline response bands."""
    mica = cfg.get("mica_round1", {}) or {}
    hd = cfg.get("hubble_distance", {}) or {}
    reference = cfg.get("reference_band", cfg.get("driver_band", mica.get("driver")))
    if reference is None:
        raise ValueError("missing reference_band/driver_band")
    reference = str(reference)

    if hd.get("bands") is not None:
        selected = [str(band) for band in (hd.get("bands") or [])]
        bands = list(dict.fromkeys([reference, *selected]))
    else:
        responses = cfg.get("response_bands")
        if responses is None:
            responses = mica.get("responses", [])
        responses = [str(band) for band in (responses or [])]
        bands = list(dict.fromkeys([reference, *responses]))

    ff = cfg.get("fflux", {}) or {}
    waves = dict(cfg.get("bands", {}) or {})
    waves.update(ff.get("wavelengths", {}) or {})
    missing = [band for band in bands if band not in waves]
    if missing:
        raise ValueError("missing wavelengths for selected band(s): " + ", ".join(missing))

    return bands, {band: float(waves[band]) for band in bands}


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


def distance_result_dir(source_dir: Path, cfg: dict) -> Path:
    """Use hubble_distance.output_dir when configured."""
    hd = cfg.get("hubble_distance", {}) or {}
    out = (
        rel(source_dir, hd["output_dir"])
        if hd.get("output_dir")
        else mica_result_dir(source_dir, cfg)
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def flux_posterior_path(source_dir: Path, cfg: dict) -> Path:
    hd = cfg.get("hubble_distance", {}) or {}
    if hd.get("flux_posterior"):
        return rel(source_dir, hd["flux_posterior"])

    ff = cfg.get("fflux", {}) or {}
    root = rel(source_dir, ff.get("out", "fflux")) / "analysis"
    hits = []
    for path in sorted(root.glob("*_flux_posterior.txt")):
        header = path.open("r", encoding="utf-8").readline()
        if "F_Bright_int_" in header:
            hits.append(path)
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one nuclear flux posterior under {root}, found {len(hits)}")
    return hits[0]


def lag_posterior_path(source_dir: Path, cfg: dict) -> Path:
    hd = cfg.get("hubble_distance", {}) or {}
    if hd.get("lag_posterior"):
        return rel(source_dir, hd["lag_posterior"])
    return mica_result_dir(source_dir, cfg) / "absolute_lag_posterior.txt"


def tau0_posterior_path(source_dir: Path, cfg: dict) -> Path:
    hd = cfg.get("hubble_distance", {}) or {}
    if hd.get("tau0_posterior"):
        return rel(source_dir, hd["tau0_posterior"])
    return mica_result_dir(source_dir, cfg) / "tau0_posterior.txt"


def i_n(n: float, disk_b: float) -> float:
    r"""Return I_n(b) = Gamma(n/b) zeta(n/b)."""
    from scipy.special import gamma, zeta

    disk_b = float(disk_b)
    if not 0.0 < disk_b < 2.0:
        raise ValueError("disk_b must satisfy 0 < disk_b < 2")
    x = float(n) / disk_b
    return float(gamma(x) * zeta(x, 1.0))


def disk_k_b(disk_b: float) -> float:
    r"""Return K(b) = 13.26 [I_2(b)^3 / (b I_3(b)^2)]^(1/2) Mpc."""
    i2 = i_n(2.0, disk_b)
    i3 = i_n(3.0, disk_b)
    return float(K_B_NORMALIZATION * np.sqrt(i2**3 / (float(disk_b) * i3**2)))


def disk_k_b_array(disk_b) -> np.ndarray:
    """Vectorized K(b), used when the free lag slope implies a b posterior."""
    from scipy.special import gamma, zeta

    disk_b = np.asarray(disk_b, float)
    valid = np.isfinite(disk_b) & (disk_b > 0.0) & (disk_b < 2.0)
    out = np.full(disk_b.shape, np.nan, float)
    b = disk_b[valid]
    i2 = gamma(2.0 / b) * zeta(2.0 / b, 1.0)
    i3 = gamma(3.0 / b) * zeta(3.0 / b, 1.0)
    out[valid] = K_B_NORMALIZATION * np.sqrt(i2**3 / (b * i3**2))
    return out


def disk_model_settings(cfg: dict) -> dict:
    """Read the disk temperature exponent and viewing geometry."""
    hd = cfg.get("hubble_distance", {}) or {}
    disk_b = float(hd.get("disk_b", DEFAULT_DISK_B))
    if not 0.0 < disk_b < 2.0:
        raise ValueError("hubble_distance.disk_b must satisfy 0 < disk_b < 2")

    if hd.get("cos_i") is not None and hd.get("inclination_deg") is not None:
        raise ValueError("set only one of hubble_distance.cos_i and inclination_deg")
    if hd.get("cos_i") is not None:
        cos_i = float(hd["cos_i"])
        if not 0.0 < cos_i <= 1.0:
            raise ValueError("hubble_distance.cos_i must satisfy 0 < cos_i <= 1")
        inclination_deg = float(np.degrees(np.arccos(cos_i)))
    else:
        inclination_deg = float(hd.get("inclination_deg", DEFAULT_INCLINATION_DEG))
        if not 0.0 <= inclination_deg < 90.0:
            raise ValueError("hubble_distance.inclination_deg must satisfy 0 <= i < 90")
        cos_i = float(np.cos(np.deg2rad(inclination_deg)))

    k_b = (
        float(hd["k_b"])
        if hd.get("k_b") is not None
        else disk_k_b(disk_b)
    )
    if not np.isfinite(k_b) or k_b <= 0.0:
        raise ValueError("hubble_distance.k_b must be finite and positive")

    return {
        "disk_b": disk_b,
        "lag_beta": 1.0 / disk_b,
        "k_b": k_b,
        "inclination_deg": inclination_deg,
        "cos_i": cos_i,
    }


def tau0_column_for_beta(beta: float, cfg: dict) -> str:
    """Map the fixed lag-spectrum exponent to fit_tau.py's posterior column."""
    hd = cfg.get("hubble_distance", {}) or {}
    if hd.get("tau0_column"):
        return str(hd["tau0_column"])
    if np.isclose(beta, 4.0 / 3.0, rtol=0.0, atol=1.0e-10):
        return "tau0_b4_3"
    if np.isclose(beta, 2.0, rtol=0.0, atol=1.0e-10):
        return "tau0_b2"
    raise ValueError(
        f"no automatic tau0 column for lag_beta={beta:.8g}; "
        "set hubble_distance.tau0_column to a fixed-beta tau0 posterior column"
    )


def load_lag_posterior(
    source_dir: Path,
    cfg: dict,
    bands: list[str],
    waves: dict[str, float],
    lag_beta: float,
) -> tuple[pd.DataFrame, Path, str]:
    """
    Load explicit absolute lags, or reconstruct beta-consistent absolute lags.

    fit_tau.py defines relative lags as
        Delta tau(lambda) = tau0 [(lambda/lambda0)^beta - 1],
    so tau0 is the absolute lag at the reference wavelength.
    """
    hd = cfg.get("hubble_distance", {}) or {}
    if hd.get("lag_posterior"):
        path = lag_posterior_path(source_dir, cfg)
        configured_beta = hd.get("lag_beta")
        if configured_beta is not None and not np.isclose(
            float(configured_beta), lag_beta, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError(
                "hubble_distance.lag_beta is inconsistent with "
                f"1/disk_b={lag_beta:.8g}"
            )
        return pd.read_csv(path, sep=r"\s+"), path, "explicit_absolute_lag"

    tau0_path = tau0_posterior_path(source_dir, cfg)
    tau0_column = tau0_column_for_beta(lag_beta, cfg)
    if tau0_path.exists():
        tau0_table = pd.read_csv(tau0_path, sep=r"\s+")
        if tau0_column not in tau0_table.columns:
            raise ValueError(
                f"{tau0_path} has no {tau0_column!r} column. "
                "Run fit_tau.py with fit_models including the matching fixed beta."
            )
        tau0 = pd.to_numeric(tau0_table[tau0_column], errors="coerce").to_numpy(float)
        lambda0 = float(waves[bands[0]])
        lag = pd.DataFrame({"sample": np.arange(len(tau0)), "tau0": tau0})
        for band in bands:
            lag[f"tau_abs_{band}"] = tau0 * (float(waves[band]) / lambda0) ** lag_beta
        return lag, tau0_path, f"{tau0_column}: beta={lag_beta:.8g}"

    # Backward-compatible fallback for the original thin-disk calculation.
    if np.isclose(lag_beta, 4.0 / 3.0, rtol=0.0, atol=1.0e-10):
        path = lag_posterior_path(source_dir, cfg)
        return pd.read_csv(path, sep=r"\s+"), path, "legacy_absolute_lag: beta=4/3"

    raise FileNotFoundError(
        f"{tau0_path} is required for disk_b={1.0 / lag_beta:.8g}. "
        f"For disk_b=1/2, run fit_tau.py first so {tau0_column} is available."
    )


def fnu_jy(flam, wave):
    return flam * FLUX_SCALE * wave**2 / C_AA_S * 1.0e23


def g_eps(eps):
    return (1.0 - eps) / (1.0 - eps**1.5)


def disk_dist(
    tau,
    wave,
    flam,
    eps,
    disk_b=DEFAULT_DISK_B,
    cos_i=None,
    k_b=None,
):
    """Disk distance in Mpc for T proportional to R^(-b)."""
    if cos_i is None:
        cos_i = float(np.cos(np.deg2rad(DEFAULT_INCLINATION_DEG)))
    if not 0.0 < float(cos_i) <= 1.0:
        raise ValueError("cos_i must satisfy 0 < cos_i <= 1")
    if k_b is None:
        k_b = disk_k_b(disk_b)
    fnu = fnu_jy(flam, wave)
    return (
        np.asarray(k_b, float)
        * tau
        * (wave / 1.0e4) ** (-1.5)
        * (fnu / float(cos_i)) ** (-0.5)
        * g_eps(eps)
    )


def distance_flux_columns(cfg: dict, band: str) -> tuple[str, str]:
    """Return configured bright-flux and epsilon columns for one band."""
    hd = cfg.get("hubble_distance", {}) or {}
    templates = dict(hd.get("flux_columns", {}) or {})
    bkey = key(band)
    bright = str(
        templates.get("bright", "F_Bright_int_{band}")
    ).format(band=bkey, raw_band=band)
    epsilon = str(
        templates.get("epsilon", "Epsilon_int_{band}")
    ).format(band=bkey, raw_band=band)
    return bright, epsilon


def q68(values):
    p16, p50, p84 = np.percentile(np.asarray(values, float), [15.865, 50.0, 84.135])
    return float(p50), float(p50 - p16), float(p84 - p50)


def model_d70(z: float, om0: float) -> float:
    from scipy.integrate import quad
    integral = quad(lambda x: 1.0 / np.sqrt(om0 * (1.0 + x) ** 3 + 1.0 - om0), 0.0, z)[0]
    return (1.0 + z) * C_KM_S / 70.0 * integral


def build_distance_posterior(source_dir: Path, cfg: dict):
    hd = cfg.get("hubble_distance", {}) or {}
    flux_path = flux_posterior_path(source_dir, cfg)
    flux = pd.read_csv(flux_path, sep=r"\s+")

    bands, waves = bands_and_waves(cfg)
    free_disk_b = str(hd.get("disk_b", "")).strip().lower() == "free"
    if free_disk_b:
        geometry_cfg = copy.deepcopy(cfg)
        geometry_cfg["hubble_distance"] = dict(hd)
        geometry_cfg["hubble_distance"]["disk_b"] = DEFAULT_DISK_B
        model = disk_model_settings(geometry_cfg)
        lag_path = lag_posterior_path(source_dir, cfg)
        lag = pd.read_csv(lag_path, sep=r"\s+")
        if "beta" not in lag:
            raise ValueError(
                f"{lag_path} has no beta column required by the free disk model"
            )
        lag_model = "explicit_absolute_lag: beta posterior"
        model["free_disk_b"] = True
    else:
        model = disk_model_settings(cfg)
        lag, lag_path, lag_model = load_lag_posterior(
            source_dir, cfg, bands, waves, model["lag_beta"]
        )
        model["free_disk_b"] = False
    missing = []
    for band in bands:
        required_lag = f"tau_abs_{band}"
        required_flux = list(distance_flux_columns(cfg, band))
        absent = []
        if required_lag not in lag.columns:
            absent.append(required_lag)
        absent.extend(name for name in required_flux if name not in flux.columns)
        if absent:
            missing.append(f"{band}: {', '.join(absent)}")
    if missing:
        raise ValueError(
            "selected band columns missing from lag/flux posterior: " + "; ".join(missing)
        )

    nsamp = min(len(flux), len(lag), int(hd.get("nsamp", min(len(flux), len(lag)))))
    rng = np.random.default_rng(int(hd.get("seed", 12345)))
    flux = flux.iloc[rng.choice(len(flux), nsamp, replace=False)].reset_index(drop=True)
    lag = lag.iloc[rng.choice(len(lag), nsamp, replace=False)].reset_index(drop=True)

    if free_disk_b:
        lag_beta = pd.to_numeric(lag["beta"], errors="coerce").to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            disk_b = 1.0 / lag_beta
        k_b = disk_k_b_array(disk_b)
    else:
        lag_beta = np.full(nsamp, model["lag_beta"], float)
        disk_b = np.full(nsamp, model["disk_b"], float)
        k_b = np.full(nsamp, model["k_b"], float)

    distance = np.empty((nsamp, len(bands)), float)
    valid = (
        np.isfinite(lag_beta)
        & np.isfinite(disk_b)
        & np.isfinite(k_b)
        & (lag_beta > 0.0)
        & (disk_b > 0.0)
        & (disk_b < 2.0)
        & (k_b > 0.0)
    )
    for i, band in enumerate(bands):
        bright_column, epsilon_column = distance_flux_columns(cfg, band)
        tau = lag[f"tau_abs_{band}"].to_numpy(float)
        bright = flux[bright_column].to_numpy(float)
        epsilon = flux[epsilon_column].to_numpy(float)
        distance[:, i] = disk_dist(
            tau,
            waves[band],
            bright,
            epsilon,
            disk_b=disk_b,
            cos_i=model["cos_i"],
            k_b=k_b,
        )
        valid &= np.isfinite(tau) & np.isfinite(bright) & np.isfinite(epsilon)
        valid &= (tau > 0.0) & (bright > 0.0) & (epsilon > 0.0) & (epsilon < 1.0)

    valid &= np.isfinite(distance).all(axis=1) & (distance > 0.0).all(axis=1)
    distance = distance[valid]
    lag_beta = lag_beta[valid]
    disk_b = disk_b[valid]
    k_b = k_b[valid]
    if len(distance) < 3:
        raise ValueError("too few valid joint distance samples")

    posterior = pd.DataFrame({
        "sample": np.arange(len(distance)),
        "lag_beta": lag_beta,
        "disk_b": disk_b,
        "K_b_Mpc": k_b,
    })
    for i, band in enumerate(bands):
        posterior[f"D_{band}"] = distance[:, i]

    z = redshift(cfg)
    om0 = float(hd.get("om0", DEFAULT_OM0))
    d70 = model_d70(z, om0)
    b50, blo, bhi = q68(disk_b)
    beta50, betalo, betahi = q68(lag_beta)
    k50, klo, khi = q68(k_b)
    summary = []
    for i, band in enumerate(bands):
        d50, dlo, dhi = q68(distance[:, i])
        h0 = 70.0 * d70 / distance[:, i]
        h50, hlo, hhi = q68(h0)
        summary.append({
            "source": cfg.get("source", source_dir.name),
            "band": band,
            "wavelength": waves[band],
            "z": z,
            "disk_b": b50,
            "disk_b_err_low": blo,
            "disk_b_err_high": bhi,
            "lag_beta": beta50,
            "lag_beta_err_low": betalo,
            "lag_beta_err_high": betahi,
            "K_b_Mpc": k50,
            "K_b_Mpc_err_low": klo,
            "K_b_Mpc_err_high": khi,
            "inclination_deg": model["inclination_deg"],
            "cos_i": model["cos_i"],
            "lag_model": lag_model,
            "D_mpc": d50,
            "D_err_low": dlo,
            "D_err_high": dhi,
            "H0_band": h50,
            "H0_band_err_low": hlo,
            "H0_band_err_high": hhi,
        })

    log_distance = np.log(distance)
    mean_log = np.mean(log_distance, axis=0)
    cov_log = np.atleast_2d(np.cov(log_distance, rowvar=False, ddof=1))
    model["lag_model"] = lag_model
    model.update({
        "disk_b": b50,
        "disk_b_err_low": blo,
        "disk_b_err_high": bhi,
        "lag_beta": beta50,
        "lag_beta_err_low": betalo,
        "lag_beta_err_high": betahi,
        "k_b": k50,
        "k_b_err_low": klo,
        "k_b_err_high": khi,
    })
    return (
        posterior,
        pd.DataFrame(summary),
        bands,
        mean_log,
        cov_log,
        d70,
        float(np.mean(valid)),
        flux_path,
        lag_path,
        model,
    )


def log_prior(theta, h0_prior, f0_prior):
    h0, f0 = theta
    return 0.0 if h0_prior[0] < h0 < h0_prior[1] and f0_prior[0] <= f0 < f0_prior[1] else -np.inf


def log_prob(theta, mean_log, cov_log, d70, h0_prior, f0_prior):
    lp = log_prior(theta, h0_prior, f0_prior)
    if not np.isfinite(lp):
        return -np.inf
    h0, f0 = theta
    delta = mean_log - np.log(d70 * 70.0 / h0)
    cov = cov_log + (f0**2 + 1.0e-12) * np.eye(len(mean_log))
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf
    return lp - 0.5 * (delta @ np.linalg.solve(cov, delta) + logdet + len(delta) * np.log(2.0 * np.pi))


def run_h0_mcmc(
    mean_log,
    cov_log,
    d70,
    cfg,
    out: Path,
    steps=None,
    ncores=None,
    model=None,
):
    import emcee

    hd = cfg.get("hubble_distance", {}) or {}
    seed = int(hd.get("seed", 12345))
    nwalkers = int(hd.get("nwalkers", 32))
    steps = int(steps or hd.get("steps", 10000))
    ncores = int(ncores or hd.get("ncores", 1))
    burn = int(float(hd.get("burn_frac", 0.2)) * steps)
    thin = int(hd.get("thin", 10))
    h0_prior = tuple(map(float, hd.get("h0_prior", [20.0, 120.0])))
    f0_prior = tuple(map(float, hd.get("f0_prior", [0.0, 1.0])))
    model = dict(model or disk_model_settings(cfg))
    model_name = str(hd.get("model_name", "single"))

    h0_init = float(np.clip(70.0 * d70 / np.exp(np.mean(mean_log)), h0_prior[0] + 1.0, h0_prior[1] - 1.0))
    rng = np.random.default_rng(seed)
    pos = np.column_stack([
        np.clip(h0_init + rng.normal(0.0, 1.0, nwalkers), h0_prior[0] + 1.0e-5, h0_prior[1] - 1.0e-5),
        np.clip(0.05 + rng.normal(0.0, 0.01, nwalkers), f0_prior[0] + 1.0e-5, f0_prior[1] - 1.0e-5),
    ])

    args = (mean_log, cov_log, d70, h0_prior, f0_prior)
    t0 = time.time()
    if ncores > 1:
        with Pool(ncores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, 2, log_prob, args=args, pool=pool)
            sampler.run_mcmc(pos, steps, progress=True)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, 2, log_prob, args=args)
        sampler.run_mcmc(pos, steps, progress=True)

    samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    logp = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    h50, hlo, hhi = q68(samples[:, 0])
    f50, flo, fhi = q68(samples[:, 1])

    post = pd.DataFrame({"H0": samples[:, 0], "f0": samples[:, 1], "log_probability": logp})
    summary = pd.DataFrame([{
        "source": cfg.get("source", ""),
        "distance_model": model_name,
        "redshift": redshift(cfg),
        "H0": h50,
        "H0_err_low": hlo,
        "H0_err_high": hhi,
        "f0": f50,
        "f0_err_low": flo,
        "f0_err_high": fhi,
        "Om0": float(hd.get("om0", DEFAULT_OM0)),
        "disk_b": model["disk_b"],
        "lag_beta": model["lag_beta"],
        "K_b_Mpc": model["k_b"],
        "inclination_deg": model["inclination_deg"],
        "cos_i": model["cos_i"],
        "n_band": len(mean_log),
        "nwalkers": nwalkers,
        "steps": steps,
        "burn": burn,
        "thin": thin,
        "elapsed_sec": time.time() - t0,
    }])
    post.to_csv(out / "hubble_posterior.txt", sep=" ", index=False, float_format="%.10e")
    summary.to_csv(out / "hubble_summary.csv", index=False)
    return summary


def distance_branch_root(source_dir: Path, cfg: dict) -> Path:
    hd = cfg.get("hubble_distance", {}) or {}
    if hd.get("result_root"):
        return rel(source_dir, hd["result_root"])
    mica = cfg.get("mica_round1", {}) or {}
    branch_id = str(hd.get("posterior_branch", "comp0"))
    return (
        mica_run_root(source_dir, cfg)
        / "result"
        / branch_result_relative_dir(mica, branch_id)
    )


def run_distance_models(
    source_dir: Path,
    cfg: dict,
    *,
    steps=None,
    ncores=None,
    no_mcmc=False,
):
    """Run standard, slim, and free disk-distance branches independently."""
    base_hd = dict(cfg.get("hubble_distance", {}) or {})
    definitions = dict(base_hd.get("models", {}) or {})
    if not definitions:
        raise ValueError("hubble_distance.models is empty")
    branch_root = distance_branch_root(source_dir, cfg)
    lag_suffix = {
        "4/3": "b4_3",
        "b4_3": "b4_3",
        "standard": "b4_3",
        "2": "b2",
        "b2": "b2",
        "slim": "b2",
        "free": "free",
    }
    combined_distance = []
    combined_h0 = []

    for index, (model_name, raw_definition) in enumerate(definitions.items()):
        definition = dict(raw_definition or {})
        model_cfg = copy.deepcopy(cfg)
        hd = dict(base_hd)
        hd.pop("models", None)
        hd.update(definition)
        hd["model_name"] = str(model_name)
        hd["seed"] = (
            int(base_hd.get("seed", 12345))
            + int(definition.get("seed_offset", 100003 * (index + 1)))
        )
        hd.setdefault(
            "flux_posterior",
            str(branch_root / "fflux" / "analysis" / "distance_flux_posterior.txt"),
        )
        lag_model = str(definition.get("lag_fit_model", model_name)).lower()
        if lag_model not in lag_suffix:
            raise ValueError(
                f"unknown lag_fit_model {lag_model!r} for distance model {model_name!r}"
            )
        suffix = lag_suffix[lag_model]
        hd.setdefault(
            "lag_posterior",
            str(branch_root / "lag" / f"absolute_lag_posterior_{suffix}.txt"),
        )
        hd["output_dir"] = str(branch_root / "distance" / str(model_name))
        hd["h0_output_dir"] = str(branch_root / "h0" / str(model_name))
        model_cfg["hubble_distance"] = hd

        _, summary = run(
            source_dir,
            steps=steps,
            ncores=ncores,
            no_mcmc=no_mcmc,
            _cfg=model_cfg,
            _single_model=True,
        )
        combined_distance.append(summary)
        if not no_mcmc:
            h0_path = Path(hd["h0_output_dir"]) / "hubble_summary.csv"
            combined_h0.append(pd.read_csv(h0_path))

    distance_summary = pd.concat(combined_distance, ignore_index=True)
    distance_root = branch_root / "distance"
    distance_root.mkdir(parents=True, exist_ok=True)
    distance_summary.to_csv(
        distance_root / "distance_model_summary.csv",
        index=False,
    )
    if combined_h0:
        h0_root = branch_root / "h0"
        h0_root.mkdir(parents=True, exist_ok=True)
        pd.concat(combined_h0, ignore_index=True).to_csv(
            h0_root / "hubble_model_summary.csv",
            index=False,
        )
    return distance_summary


def run(
    source_dir: Path,
    steps=None,
    ncores=None,
    no_mcmc=False,
    disk_b=None,
    inclination_deg=None,
    _cfg=None,
    _single_model=False,
):
    source_dir = source_dir.expanduser().resolve()
    cfg = copy.deepcopy(_cfg) if _cfg is not None else read_yaml(
        source_dir / "config" / "source_config.yaml"
    )
    if not _single_model and (cfg.get("hubble_distance", {}) or {}).get("models"):
        return run_distance_models(
            source_dir,
            cfg,
            steps=steps,
            ncores=ncores,
            no_mcmc=no_mcmc,
        )
    if disk_b is not None or inclination_deg is not None:
        cfg["hubble_distance"] = dict(cfg.get("hubble_distance", {}) or {})
        if disk_b is not None:
            cfg["hubble_distance"]["disk_b"] = float(disk_b)
        if inclination_deg is not None:
            cfg["hubble_distance"].pop("cos_i", None)
            cfg["hubble_distance"]["inclination_deg"] = float(inclination_deg)
    out = distance_result_dir(source_dir, cfg)
    hd = cfg.get("hubble_distance", {}) or {}
    model_name = str(hd.get("model_name", "single"))

    (
        posterior,
        summary,
        bands,
        mean_log,
        cov_log,
        d70,
        valid_fraction,
        flux_path,
        lag_path,
        model,
    ) = build_distance_posterior(source_dir, cfg)
    source = cfg.get("source", source_dir.name)
    posterior.insert(1, "distance_model", model_name)
    summary.insert(1, "distance_model", model_name)
    posterior.to_csv(out / "distance_posterior.txt", sep=" ", index=False, float_format="%.10e")
    summary.to_csv(out / "distance_by_band.csv", index=False)
    pd.DataFrame({"band": bands, "mean_log_D": mean_log}).to_csv(out / "log_distance_mean.csv", index=False)
    pd.DataFrame(cov_log, index=bands, columns=bands).to_csv(out / "log_distance_covariance.csv")
    pd.DataFrame([{
        "source": source,
        "distance_model": model_name,
        "valid_fraction": valid_fraction,
        "n_sample": len(posterior),
        "disk_b": model["disk_b"],
        "disk_b_err_low": model.get("disk_b_err_low", 0.0),
        "disk_b_err_high": model.get("disk_b_err_high", 0.0),
        "lag_beta": model["lag_beta"],
        "lag_beta_err_low": model.get("lag_beta_err_low", 0.0),
        "lag_beta_err_high": model.get("lag_beta_err_high", 0.0),
        "K_b_Mpc": model["k_b"],
        "K_b_Mpc_err_low": model.get("k_b_err_low", 0.0),
        "K_b_Mpc_err_high": model.get("k_b_err_high", 0.0),
        "inclination_deg": model["inclination_deg"],
        "cos_i": model["cos_i"],
        "lag_model": model["lag_model"],
    }]).to_csv(out / "distance_run_summary.csv", index=False)

    print(f"flux posterior: {flux_path}")
    print(f"lag input: {lag_path} ({model['lag_model']})")
    print(
        f"disk model: b={model['disk_b']:.6g}, beta={model['lag_beta']:.6g}, "
        f"K(b)={model['k_b']:.6g} Mpc, i={model['inclination_deg']:.3f} deg"
    )
    print(f"valid joint samples: {len(posterior)} ({valid_fraction:.3f})")
    print(f"saved: {out / 'distance_posterior.txt'}")

    if not no_mcmc:
        h0_out = (
            rel(source_dir, hd["h0_output_dir"])
            if hd.get("h0_output_dir")
            else out
        )
        h0_out.mkdir(parents=True, exist_ok=True)
        h0 = run_h0_mcmc(
            mean_log,
            cov_log,
            d70,
            cfg,
            h0_out,
            steps=steps,
            ncores=ncores,
            model=model,
        )
        row = h0.iloc[0]
        print(f"H0 = {row.H0:.2f} +{row.H0_err_high:.2f} -{row.H0_err_low:.2f}")
    return posterior, summary


def main():
    ap = argparse.ArgumentParser(description="Distance posterior and single-source H0 fit")
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--ncores", type=int)
    ap.add_argument("--no-mcmc", action="store_true")
    ap.add_argument(
        "--disk-b",
        type=float,
        help="temperature-profile exponent b in T proportional to R^(-b)",
    )
    ap.add_argument("--inclination-deg", type=float, help="disk inclination in degrees")
    args = ap.parse_args()
    run(
        args.source_dir,
        args.steps,
        args.ncores,
        args.no_mcmc,
        args.disk_b,
        args.inclination_deg,
    )


if __name__ == "__main__":
    main()
