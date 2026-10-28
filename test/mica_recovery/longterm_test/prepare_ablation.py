"""Paired controls extending the saved one-shared-process realization.

The anchor saved (disk + BLR)*I rather than the separate BLR curve. Draw
the latter from its conditional Gaussian distribution, including residual
uncertainty. This is an extension under the original model, not recovery of
a unique historical hidden I(t). All controls share components and noise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from prepare import (
    _fft_covariance, _real_disk_transform, _sample_psd_covariance,
    load_config, load_observed, observation_setup, positive_float,
    real_disk_mismatch_cases, real_disk_mismatch_component_covariances,
    write_light_curve,
)


def anchor_setup(config_path: Path, config: dict) -> tuple[Path, dict, dict]:
    suite = config["paired_ablation"]
    anchor_config = load_config((config_path.parent / suite["anchor_config"]).resolve())
    anchor = (config_path.parent / suite["anchor_case"]).resolve()
    parameters = real_disk_mismatch_cases(anchor_config)[anchor.name]
    if parameters["short_q"] != 0:
        raise ValueError("Paired controls require a single shared process")
    if config["observations"] != anchor_config["observations"]:
        raise ValueError("Paired controls must retain anchor observation settings")
    return anchor, anchor_config, parameters


def ablation_cases(config_path: Path, config: dict) -> dict[str, dict]:
    """Physical parameters and control switches; excludes reused cases."""
    _, _, parameters = anchor_setup(config_path, config)
    result = {}
    for name, switches in config["paired_ablation"]["cases"].items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid case name: {name!r}")
        required = {"blr", "shared_driver", "shared_response"}
        if not required <= set(switches) or set(switches) - required - {"unfiltered_response_gp"}:
            raise ValueError(f"Invalid switches for {name}")
        if not all(type(value) is bool for value in switches.values()):
            raise ValueError("Ablation switches must be booleans")
        result[name] = {**parameters, **switches}
        if not switches["blr"]:
            result[name]["response_blr_gain"] = 0.0
        if switches.get("unfiltered_response_gp", False):
            if switches["shared_driver"] or not switches["shared_response"]:
                raise ValueError("Unfiltered GP requires U-only additional signal")
            gp = config["paired_ablation"]["unfiltered_response_gp"]
            result[name]["independent_gp_tau_days"] = positive_float(
                "unfiltered_response_gp.tau_days", gp["tau_days"])
            result[name]["independent_gp_amplitude"] = positive_float(
                "unfiltered_response_gp.amplitude", gp["amplitude"])
            result[name]["independent_gp_seed"] = int(config["experiment"]["seed"])
    return result


def center_blocks(matrix: np.ndarray, row_sizes: list[int],
                  col_sizes: list[int]) -> np.ndarray:
    """H_row C H_col, removing one mean per band."""
    result = np.array(matrix, dtype=float, copy=True)
    start = 0
    for size in row_sizes:
        result[start:start + size] -= result[start:start + size].mean(axis=0)
        start += size
    start = 0
    for size in col_sizes:
        result[:, start:start + size] -= result[:, start:start + size].mean(axis=1)[:, None]
        start += size
    return result


def conditional_moments(cyy: np.ndarray, czy: np.ndarray, czz: np.ndarray,
                        y: np.ndarray, band_sizes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Condition on demeaned bands, dropping their redundant last entries."""
    keep = np.ones(len(y), dtype=bool)
    keep[np.cumsum(band_sizes) - 1] = False
    factor = cho_factor(cyy[np.ix_(keep, keep)], lower=True, check_finite=True)
    cross = czy[:, keep]
    mean = cross @ cho_solve(factor, y[keep])
    covariance = czz - cross @ cho_solve(factor, cross.T)
    return mean, (covariance + covariance.T) / 2


def split_anchor_illumination(tw: np.ndarray, tu: np.ndarray, p: dict,
                              target_rms: float, alpha: float, y: np.ndarray,
                              seed: int) -> tuple[np.ndarray, dict]:
    """Draw alpha*b*B*I at U epochs conditional on saved illumination."""
    covariances, diagnostics = real_disk_mismatch_component_covariances(tw, tu, p, target_rms)
    if not np.isclose(diagnostics["alpha"], alpha, rtol=1e-12, atol=0):
        raise ValueError("Fixed alpha is inconsistent with the anchor covariance")
    dt, n = p["covariance_dt_days"], p["covariance_n"]
    omega = 2 * np.pi * np.fft.rfftfreq(n, dt)
    dw, _ = _real_disk_transform(p, p["driver_wavelength_observed_angstrom"])
    du, _ = _real_disk_transform(p, p["response_wavelength_observed_angstrom"])
    blr = p["response_blr_gain"] * np.exp(-1j * omega * p["blr_onset_days"]) / (
        1 + 1j * omega * p["blr_theta_days"]
    )**2
    u = p["response_disk_gain"] * du + blr
    psd = 2 * p["driver_tau_days"] / (1 + (omega * p["driver_tau_days"])**2)

    def covariance(row_filter, col_filter, row_times, col_times):
        return alpha**2 * _fft_covariance(
            psd * col_filter * np.conj(row_filter),
            col_times[None, :] - row_times[:, None], dt, n,
        )

    nw, nu = len(tw), len(tu)
    cyy = center_blocks(covariances["illumination"], [nw, nu], [nw, nu])
    czy = center_blocks(np.hstack((covariance(blr, dw, tu, tw),
                                  covariance(blr, u, tu, tu))), [nu], [nw, nu])
    czz = center_blocks(covariance(blr, blr, tu, tu), [nu], [nu])
    mean, conditional = conditional_moments(cyy, czy, czz, y, [nw, nu])
    residual, minimum_eigenvalue = _sample_psd_covariance(
        conditional, np.random.default_rng(seed)
    )
    sample = mean + residual
    sample -= sample.mean()
    return sample, {
        "alpha": alpha,
        "conditional_seed": seed,
        "conditional_min_eigenvalue": minimum_eigenvalue,
        "conditional_expected_residual_rms": float(np.sqrt(np.trace(conditional) / nu)),
        "conditional_realized_residual_rms": float(np.std(residual)),
        "sampled_blr_rms": float(np.std(sample)),
        "method": "Gaussian conditional augmentation of saved band-demeaned illumination; residual retained",
    }


def sample_unfiltered_gp(times: np.ndarray, tau_days: float,
                         amplitude: float, seed: int) -> np.ndarray:
    """Draw the U-only GP directly at observed epochs, without a disk filter."""
    lag = np.abs(times[:, None] - times[None, :]) / tau_days
    covariance = amplitude**2 * (1 + lag) * np.exp(-lag)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0, 1)))
    sample, _ = _sample_psd_covariance(covariance, rng)
    return sample - sample.mean()


def prepare_cases(config_path: Path, overwrite: bool = False) -> None:
    config_path = config_path.resolve()
    config = load_config(config_path)
    suite = config["paired_ablation"]
    cases = ablation_cases(config_path, config)
    anchor, anchor_config, parameters = anchor_setup(config_path, config)
    if overwrite:
        raise ValueError("Paired controls do not support overwriting existing inputs")
    output_dir = (config_path.parent / config["experiment"]["output_dir"]).resolve()
    for name in cases:
        if (output_dir / name).exists():
            raise FileExistsError(f"Case already exists: {output_dir / name}")
    reused = {}
    for name, relative in suite.get("reused_cases", {}).items():
        if Path(name).name != name or name in cases or name in (".", ".."):
            raise ValueError(f"Invalid reused case name: {name}")
        target = (config_path.parent / relative).resolve()
        link = output_dir / name
        if target != anchor or not target.is_dir():
            raise ValueError("Reused case must point to the existing anchor")
        if link.exists() or link.is_symlink():
            if not link.is_symlink() or link.resolve() != target:
                raise FileExistsError(f"Reused-case path occupied: {link}")
        reused[name] = target

    driver, response, t0, times, errors, target_rms = observation_setup(config_path, config)
    bands = (driver, response)
    source = config["experiment"]["simulation_source"]
    anchor_source = anchor_config["experiment"]["simulation_source"]
    components = {}
    for band in bands:
        path = anchor / "components" / f"{band}_components.txt"
        components[band] = np.loadtxt(path)
        data = load_observed(anchor / "input" / f"{anchor_source}_{band}.txt")
        c = components[band]
        if c.shape != (len(times[band]), 5):
            raise ValueError("Unexpected anchor component format")
        np.testing.assert_allclose(c[:, 0], times[band], atol=6e-10, rtol=0)
        np.testing.assert_allclose(data[:, 0], times[band], atol=6e-10, rtol=0)
        np.testing.assert_allclose(data[:, 2], errors[band], atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(data[:, 1], 1 + c[:, 1:].sum(axis=1), atol=8e-12, rtol=0)
        if np.any(c[:, 3] != 0):
            raise ValueError("Anchor contains a second shared process")
        header = path.read_text(encoding="utf-8").splitlines()[:8]
        saved_alpha = [float(item.split("=", 1)[1]) for line in header
                       for item in line.lstrip("# ").split("; ") if item.startswith("alpha=")]
        np.testing.assert_allclose(saved_alpha, [suite["fixed_alpha"]], rtol=1e-14, atol=0)

    blr_u, diagnostics = split_anchor_illumination(
        times[driver], times[response], parameters, target_rms, float(suite["fixed_alpha"]),
        np.r_[components[driver][:, 1], components[response][:, 1]],
        int(suite["conditional_seed"]),
    )
    disk_u = components[response][:, 1] - blr_u
    arrays = {}
    for name, p in cases.items():
        arrays[name] = {}
        raw_gp = None
        if p.get("unfiltered_response_gp", False):
            raw_gp = sample_unfiltered_gp(
                times[response], p["independent_gp_tau_days"],
                p["independent_gp_amplitude"], p["independent_gp_seed"])
        for band in bands:
            old = components[band]
            is_driver = band == driver
            disk = old[:, 1] if is_driver else disk_u
            blr = np.zeros(len(old)) if is_driver or not p["blr"] else blr_u
            illumination = old[:, 1] if is_driver or p["blr"] else disk
            shared_on = p["shared_driver"] if is_driver else p["shared_response"]
            shared = old[:, 2] if shared_on else np.zeros(len(old))
            if not is_driver and raw_gp is not None:
                shared = raw_gp
            values = np.column_stack((old[:, 0], illumination, shared, old[:, 3], old[:, 4], disk, blr))
            flux = 1 + values[:, 1:5].sum(axis=1)
            if not np.isfinite(values).all() or np.any(flux <= 0):
                raise ValueError(f"Invalid flux in {name}; do not clip or change seed")
            np.testing.assert_allclose(illumination, disk + blr, atol=1e-15, rtol=1e-14)
            arrays[name][band] = values

    # All scientific validation above precedes the first output write.
    for name, p in cases.items():
        case = output_dir / name
        parameter_text = "; ".join(f"{key}={value:.17g}" for key, value in p.items())
        process_note = (
            "illumination and noise retained; U GP redrawn directly at U epochs; "
            "GP term=A_S*S(t), no disk filtering or interband shift; "
            "GP RNG spawn_key=(0,1)"
            if p.get("unfiltered_response_gp", False)
            else "same sampled signals and noise across all controls"
        )
        for band in bands:
            values = arrays[name][band]
            path = case / "input" / f"{source}_{band}.txt"
            write_light_curve(path, values[:, 0], 1 + values[:, 1:5].sum(axis=1), errors[band])
            component_file = case / "components" / f"{band}_components.txt"
            component_file.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(component_file, values, fmt="%.12g", header=(
                "time illumination_variation shared_long_variation shared_short_variation "
                "measurement_noise disk_illumination_variation blr_illumination_variation\n"
                f"band={band}; time_zero_mjd={t0:.9f}; flux=1+columns[1:5].sum; "
                "last two columns partition illumination, not additional signals\n"
                f"{parameter_text}\nalpha={suite['fixed_alpha']:.17g}; "
                f"conditional_seed={suite['conditional_seed']}\n"
                f"anchor={anchor}; {process_note}; "
                "component means removed; no per-case RMS rescaling"
            ))
            saved = np.loadtxt(component_file)
            np.testing.assert_allclose(np.loadtxt(path)[:, 1], 1 + saved[:, 1:5].sum(axis=1),
                                       atol=8e-12, rtol=0)
        manifest = {
            "suite": "paired_ablation", "case": name, "anchor": str(anchor),
            "switches": {key: p[key] for key in ("blr", "shared_driver", "shared_response")},
            "parameters": p, "additional_process": process_note, **diagnostics,
        }
        (case / "config").mkdir(exist_ok=True)
        (case / "config" / "paired_generation.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {name}: {driver} n={len(times[driver])}, {response} n={len(times[response])}; "
              f"fixed alpha={suite['fixed_alpha']:.12g}")
    for name, target in reused.items():
        link = output_dir / name
        if not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)
        print(f"reused {name} -> {target}")
    print(json.dumps(diagnostics, indent=2))

