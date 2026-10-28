#!/usr/bin/env python3
"""Build F9_a finite-difference absolute disk kernels for W2, M2, and B.

The three unit-area kernels are responses to one latent central driver.  They
are deliberately *not* deconvolved into W2-relative kernels: an exact relative
kernel would require a regularized, potentially signed deconvolution of the W2
absolute kernel.  Downstream simulations can instead convolve the same latent
driver separately with the W2, M2, and B mass kernels and use the sampled W2
curve as the MICA driver.

The disk response comes from ``test/disk_transfer/disktf_diff.py`` in its
``finite_difference`` mode.  It is a bright-minus-faint ``B_nu`` response on a
thin, non-relativistic disk, with a central wavelength approximation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
F9_ROOT = PROJECT_ROOT / "F9_a"
CONFIG_PATH = F9_ROOT / "config" / "source_config.yaml"
THEORY_PATH = F9_ROOT / "raw" / "theoretical_disk_lags.csv"

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(PROJECT_ROOT / "test" / "disk_transfer"))
from disktf_diff import DiskConfig, disk_transfer_function  # noqa: E402


# SI constants.  The finite-difference helper also uses SI Planck functions.
G_SI = 6.67430e-11
C_SI = 2.99792458e8
SIGMA_SB_SI = 5.670374419e-8
M_SUN_KG = 1.98847e30
DAY_S = 86400.0
LIGHT_DAY_M = C_SI * DAY_S

BANDS = ("W2", "M2", "B")
NR = 1600
NTAU = 16000
ROUT_LIGHT_DAYS = 30.0
R0_LIGHT_DAYS = 1.0
DISK_BETA = 0.75


def _finite_float(value: Any, label: str) -> float:
    """Return a finite float, with a source-specific error message."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric; got {value!r}") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite; got {value!r}")
    return number


def _read_f9_inputs() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Read F9_a redshift/filter values and the project theory-row parameters."""
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    redshift = _finite_float(config.get("redshift"), "F9_a redshift")
    if redshift < 0.0:
        raise ValueError(f"F9_a redshift must be non-negative; got {redshift}")
    wavelengths = config.get("bands")
    if not isinstance(wavelengths, dict):
        raise ValueError("F9_a config has no 'bands' wavelength mapping")

    with THEORY_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        row.get("band"): row
        for row in rows
        if row.get("source") == "F9_a" and row.get("band") in BANDS
    }
    missing = [band for band in BANDS if band not in selected]
    if missing:
        raise ValueError(
            "F9_a theoretical table is missing required band(s): "
            + ", ".join(missing)
        )

    for band, row in selected.items():
        config_wavelength = _finite_float(
            wavelengths.get(band), f"F9_a config wavelength for {band}"
        )
        table_wavelength = _finite_float(
            row.get("wavelength_angstrom"),
            f"F9_a theoretical-table wavelength for {band}",
        )
        if not np.isclose(config_wavelength, table_wavelength, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"F9_a wavelength mismatch for {band}: config={config_wavelength}, "
                f"table={table_wavelength}"
            )

    reference = selected["W2"]
    physical = {
        "mbh_msun": _finite_float(reference.get("mbh_msun"), "F9_a MBH"),
        "eddington_ratio": _finite_float(
            reference.get("eddington_ratio"), "F9_a Lbol/Ledd"
        ),
        "eta": _finite_float(reference.get("eta"), "F9_a eta"),
        "x_flux_weighted": _finite_float(
            reference.get("X_flux_weighted"), "F9_a X"
        ),
        "kappa_external_to_internal": _finite_float(
            reference.get("kappa_external_to_internal"), "F9_a kappa"
        ),
        "redshift": redshift,
    }
    if physical["mbh_msun"] <= 0.0 or physical["eddington_ratio"] <= 0.0:
        raise ValueError("F9_a MBH and Lbol/Ledd must both be positive")
    if physical["eta"] <= 0.0:
        raise ValueError("F9_a eta must be positive")

    for band, row in selected.items():
        for key in ("mbh_msun", "eddington_ratio", "eta"):
            value = _finite_float(row.get(key), f"F9_a {key} for {band}")
            if not np.isclose(value, physical[key], rtol=1e-12, atol=0.0):
                raise ValueError(
                    f"F9_a table has inconsistent {key} for {band}: "
                    f"{value} versus W2 {physical[key]}"
                )

    return {
        "config": config,
        "wavelengths_observer_angstrom": {
            band: _finite_float(wavelengths[band], f"F9_a wavelength for {band}")
            for band in BANDS
        },
        "physical": physical,
    }, selected


def _temperature_scale(physical: dict[str, float]) -> dict[str, float]:
    """Set the R=1 light-day midpoint temperature from F9_a Mdot.

    The finite-difference code needs bright and faint temperatures rather than
    an accretion rate.  We preserve its default bright/faint ratio while
    rescaling both values so their arithmetic midpoint is the standard
    viscous-disk T(R0) implied by F9_a MBH, Lbol/Ledd, and eta.
    """
    mass_kg = physical["mbh_msun"] * M_SUN_KG
    l_edd_w = 1.26e31 * physical["mbh_msun"]
    mdot_kg_s = (
        physical["eddington_ratio"]
        * l_edd_w
        / (physical["eta"] * C_SI**2)
    )
    radius_m = R0_LIGHT_DAYS * LIGHT_DAY_M
    midpoint_temperature_k = (
        3.0 * G_SI * mass_kg * mdot_kg_s
        / (8.0 * np.pi * SIGMA_SB_SI * radius_m**3)
    ) ** 0.25

    defaults = DiskConfig()
    default_midpoint_k = 0.5 * (defaults.TB + defaults.TF)
    scale = midpoint_temperature_k / default_midpoint_k
    bright_temperature_k = defaults.TB * scale
    faint_temperature_k = defaults.TF * scale

    rg_light_days = G_SI * mass_kg / C_SI**2 / LIGHT_DAY_M
    return {
        "l_edd_w": float(l_edd_w),
        "mdot_kg_s": float(mdot_kg_s),
        "midpoint_temperature_r0_k": float(midpoint_temperature_k),
        "default_midpoint_temperature_k": float(default_midpoint_k),
        "temperature_scale_from_disktf_default": float(scale),
        "tb_r0_k": float(bright_temperature_k),
        "tf_r0_k": float(faint_temperature_k),
        "tb_over_tf": float(bright_temperature_k / faint_temperature_k),
        "rg_light_days": float(rg_light_days),
        "rin_light_days": float(6.0 * rg_light_days),
        "inclination_deg": float(defaults.inc_deg),
    }


def _source_edges_and_mass(
    lag_days: np.ndarray, density_per_day: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Represent a uniform centre-sampled density as exact bin masses."""
    lag_days = np.asarray(lag_days, dtype=float)
    density_per_day = np.asarray(density_per_day, dtype=float)
    if lag_days.ndim != 1 or density_per_day.shape != lag_days.shape:
        raise ValueError("source lag and density must be one-dimensional equal-length arrays")
    if lag_days.size < 2:
        raise ValueError("source transfer function needs at least two lag samples")
    step = np.diff(lag_days)
    if not np.allclose(step, step[0], rtol=0.0, atol=1e-12):
        raise ValueError("source lag grid must be uniform")
    if not np.all(np.isfinite(density_per_day)) or np.any(density_per_day < 0.0):
        raise ValueError("source transfer density must be finite and non-negative")

    half_step = 0.5 * float(step[0])
    edges = np.empty(lag_days.size + 1, dtype=float)
    edges[0] = max(0.0, float(lag_days[0] - half_step))
    edges[1:-1] = 0.5 * (lag_days[:-1] + lag_days[1:])
    edges[-1] = lag_days[-1] + half_step
    mass = density_per_day * np.diff(edges)
    return edges, mass


def _rebin_cdf_to_impulse_grid(
    source_lag_days: np.ndarray,
    source_density_per_day: np.ndarray,
    dt_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Conservatively rebin a density onto causal impulses at j*dt.

    Each output mass is the integral of the source CDF over the Voronoi cell
    centered at its lag.  The zero-lag cell is [0, dt/2]; later cells are
    [j*dt-dt/2, j*dt+dt/2].  Thus the returned masses can directly be used in
    a discrete causal convolution without point-interpolating the density.
    """
    if not np.isfinite(dt_days) or dt_days <= 0.0:
        raise ValueError(f"dt must be positive and finite; got {dt_days}")

    source_edges, source_mass = _source_edges_and_mass(
        source_lag_days, source_density_per_day
    )
    source_area = float(source_mass.sum())
    if not np.isfinite(source_area) or source_area <= 0.0:
        raise RuntimeError("source transfer function has invalid total mass")
    source_cdf = np.concatenate(([0.0], np.cumsum(source_mass)))

    n_lag = max(2, int(np.ceil(source_edges[-1] / dt_days + 0.5)))
    lag = np.arange(n_lag, dtype=float) * dt_days
    target_edges = np.concatenate(
        ([0.0], lag[1:] - 0.5 * dt_days, [lag[-1] + 0.5 * dt_days])
    )
    target_cdf = np.interp(
        target_edges,
        source_edges,
        source_cdf,
        left=0.0,
        right=source_area,
    )
    mass = np.diff(target_cdf)
    if np.min(mass) < -1.0e-13:
        raise RuntimeError("CDF rebinning produced a materially negative mass")
    mass = np.clip(mass, 0.0, None)

    captured = float(mass.sum())
    upper_lost = float(source_area - target_cdf[-1])
    lower_lost = float(target_cdf[0])
    diagnostics = {
        "source_area": source_area,
        "rebinned_area": captured,
        "rebinned_minus_source_area": captured - source_area,
        "target_lower_lost_mass": lower_lost,
        "target_upper_lost_mass": upper_lost,
        "source_grid_lower_edge_days": float(source_edges[0]),
        "source_grid_upper_edge_days": float(source_edges[-1]),
        "target_grid_upper_edge_days": float(target_edges[-1]),
        "source_first_cell_mass": float(source_mass[0]),
        "source_last_cell_mass": float(source_mass[-1]),
        "source_first_ten_cells_mass": float(source_mass[:10].sum()),
        "source_last_ten_cells_mass": float(source_mass[-10:].sum()),
    }
    return lag, mass, target_edges, diagnostics


def _mass_moments(lag_days: np.ndarray, mass: np.ndarray) -> dict[str, float]:
    """Return total mass and first two central moments of a discrete kernel."""
    area = float(np.sum(mass))
    if not np.isfinite(area) or area <= 0.0:
        raise RuntimeError("cannot calculate moments for a non-positive mass kernel")
    mu = float(np.dot(lag_days, mass) / area)
    variance = float(np.dot((lag_days - mu) ** 2, mass) / area)
    return {
        "area": area,
        "mean_lag_days": mu,
        "variance_lag_days2": variance,
        "std_lag_days": float(np.sqrt(max(variance, 0.0))),
    }


def _write_kernel_csv(
    path: Path,
    lag_days: np.ndarray,
    bin_edges_days: np.ndarray,
    mass: np.ndarray,
) -> None:
    """Save the discrete convolution contract, including its bin geometry."""
    widths = np.diff(bin_edges_days)
    table = np.column_stack(
        (
            lag_days,
            bin_edges_days[:-1],
            bin_edges_days[1:],
            mass,
            mass / widths,
        )
    )
    np.savetxt(
        path,
        table,
        delimiter=",",
        header=(
            "lag_observer_day,bin_left_observer_day,bin_right_observer_day,"
            "mass,equivalent_bin_density_per_day"
        ),
        comments="",
        fmt="%.12g",
    )


def _plot_kernels(
    output_dir: Path,
    lag_days: np.ndarray,
    masses: dict[str, np.ndarray],
    dt_days: float,
    metadata: dict[str, Any],
) -> None:
    """Write one compact PNG/PDF comparison of the three absolute kernels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"W2": "#555B66", "M2": "#2878B5", "B": "#C76D2D"}
    # The numerical grid runs to the geometric edge (about 48 observer days),
    # while essentially all of the F9 B-band response is concentrated much
    # earlier.  Crop only the rendered view; the CSV retains every mass bin.
    b_cdf = np.cumsum(masses["B"], dtype=float)
    b_cdf /= b_cdf[-1]
    b_q999 = float(np.interp(0.999, b_cdf, lag_days))
    display_upper = max(1.0, np.ceil(2.0 * b_q999) / 2.0)
    figure, axis = plt.subplots(figsize=(8.0, 4.9))
    for band in BANDS:
        moments = metadata["bands"][band]["rebinned_moments"]
        axis.step(
            lag_days,
            masses[band] / dt_days,
            where="mid",
            color=colors[band],
            linewidth=1.6,
            label=(
                f"{band} ({metadata['bands'][band]['wavelength_observer_angstrom']:.0f} Å; "
                f"μ={moments['mean_lag_days']:.3f} d)"
            ),
        )
    axis.set_xlabel("Observer-frame delay [day]")
    axis.set_ylabel("Unit-area transfer density [day$^{-1}$]")
    axis.set_title("F9_a finite-difference absolute disk transfer functions")
    axis.set_xlim(0.0, display_upper)
    axis.legend(frameon=False)
    axis.grid(alpha=0.22)
    figure.text(
        0.5,
        0.018,
        (
            f"View: 0–{display_upper:g} d; B cumulative mass reaches 99.9% at {b_q999:.2f} d. "
            "Full 0–48 d kernels are in the CSV.\n"
            "W2, M2, and B are absolute responses to one hidden driver; no W2-relative deconvolution is shown."
        ),
        ha="center",
        fontsize=7.7,
    )
    # Reserve explicit room for the two-line provenance note and the title;
    # tight_layout can otherwise crop text when this helper is called on an
    # existing figure backend.
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.19, top=0.90)
    figure.savefig(output_dir / "f9_absolute_disk_kernels.png", dpi=220)
    figure.savefig(output_dir / "f9_absolute_disk_kernels.pdf")
    plt.close(figure)


def build_kernels(output_dir: Path, dt: float = 0.02) -> dict[str, Any]:
    """Build and save common-grid F9_a absolute W2/M2/B mass kernels.

    Parameters
    ----------
    output_dir
        Directory for three CSV kernels, the JSON provenance/check record, and
        a PNG/PDF comparison plot.
    dt
        Observer-frame spacing in days.  Returned ``mass[band][j]`` is the
        probability assigned to the causal impulse at ``lag[j] == j * dt``.

    Returns
    -------
    dict
        ``{'lag': ndarray, 'mass': {band: ndarray}, 'metadata': dict}``.
    """
    dt = _finite_float(dt, "dt")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs, table_rows = _read_f9_inputs()
    physical = inputs["physical"]
    temperature = _temperature_scale(physical)
    z_factor = 1.0 + physical["redshift"]

    all_mass: dict[str, np.ndarray] = {}
    common_lag: np.ndarray | None = None
    common_edges: np.ndarray | None = None
    per_band: dict[str, dict[str, Any]] = {}

    for band in BANDS:
        wavelength_observer = inputs["wavelengths_observer_angstrom"][band]
        wavelength_rest = wavelength_observer / z_factor
        disk = DiskConfig(
            Rin=temperature["rin_light_days"],
            Rout=ROUT_LIGHT_DAYS,
            R0=R0_LIGHT_DAYS,
            TB=temperature["tb_r0_k"],
            TF=temperature["tf_r0_k"],
            b=DISK_BETA,
            inc_deg=temperature["inclination_deg"],
            lam_A=wavelength_rest,
            NR=NR,
            Ntau=NTAU,
            response_mode="finite_difference",
        )
        lag_rest, density_rest = disk_transfer_function(disk)

        # A rest-frame density transforms as psi_obs(t_obs)=psi_rest(t_rest)/(1+z).
        lag_observer_source = lag_rest * z_factor
        density_observer_source = density_rest / z_factor
        source_edges, source_mass = _source_edges_and_mass(
            lag_observer_source, density_observer_source
        )
        source_moments = _mass_moments(lag_observer_source, source_mass)

        lag, mass, bin_edges, rebin = _rebin_cdf_to_impulse_grid(
            lag_observer_source, density_observer_source, dt
        )
        rebinned_moments = _mass_moments(lag, mass)

        if common_lag is None:
            common_lag, common_edges = lag, bin_edges
        else:
            if not np.array_equal(lag, common_lag) or not np.array_equal(
                bin_edges, common_edges
            ):
                raise RuntimeError("bands did not produce a common observer-frame grid")

        if abs(source_moments["area"] - 1.0) > 5.0e-10:
            raise RuntimeError(f"{band} source kernel is not unit area")
        if abs(rebinned_moments["area"] - 1.0) > 5.0e-10:
            raise RuntimeError(f"{band} rebinned kernel is not unit area")
        if abs(rebin["target_upper_lost_mass"]) > 5.0e-12:
            raise RuntimeError(f"{band} target grid truncates source mass")

        geometric_max_observer = (
            ROUT_LIGHT_DAYS
            * (1.0 + np.sin(np.deg2rad(temperature["inclination_deg"])))
            * z_factor
        )
        source_cdf = np.concatenate(([0.0], np.cumsum(source_mass)))
        mass_above_geometric_max = max(
            0.0,
            float(
                source_moments["area"]
                - np.interp(
                    geometric_max_observer,
                    source_edges,
                    source_cdf,
                    left=0.0,
                    right=source_moments["area"],
                )
            ),
        )
        rebin["geometric_max_observer_day"] = float(geometric_max_observer)
        rebin["source_upper_grid_guard_day"] = float(
            source_edges[-1] - geometric_max_observer
        )
        rebin["mass_above_geometric_max"] = mass_above_geometric_max
        rebin["mean_rebin_error_day"] = float(
            rebinned_moments["mean_lag_days"] - source_moments["mean_lag_days"]
        )
        rebin["variance_rebin_error_day2"] = float(
            rebinned_moments["variance_lag_days2"]
            - source_moments["variance_lag_days2"]
        )

        all_mass[band] = mass
        per_band[band] = {
            "wavelength_observer_angstrom": float(wavelength_observer),
            "wavelength_rest_angstrom": float(wavelength_rest),
            "theory_scalar_absolute_lag_days": _finite_float(
                table_rows[band].get("theory_absolute_lag_days"),
                f"F9_a scalar theory lag for {band}",
            ),
            "source_moments": source_moments,
            "rebinned_moments": rebinned_moments,
            "rebin_checks": rebin,
            "relative_centroid_to_w2_days": None,
        }

    assert common_lag is not None and common_edges is not None
    w2_mean = per_band["W2"]["rebinned_moments"]["mean_lag_days"]
    for band in BANDS:
        per_band[band]["relative_centroid_to_w2_days"] = float(
            per_band[band]["rebinned_moments"]["mean_lag_days"] - w2_mean
        )

    metadata: dict[str, Any] = {
        "status": "absolute_finite_difference_kernels_generated_no_mica_run",
        "source": "F9_a",
        "input_files": {
            "source_config": str(CONFIG_PATH),
            "theoretical_disk_lags": str(THEORY_PATH),
            "transfer_function_implementation": str(
                PROJECT_ROOT / "test" / "disk_transfer" / "disktf_diff.py"
            ),
        },
        "frame_convention": {
            "input_filter_wavelength": "observer-frame central wavelength",
            "disk_wavelength": "lambda_rest=lambda_observer/(1+z)",
            "output_lag": "observer frame: lag_observer=lag_rest*(1+z)",
            "output_density": "psi_observer=psi_rest/(1+z)",
        },
        "discrete_convolution_contract": {
            "lag_definition": "lag[j] = j * dt observer-frame days",
            "mass_definition": (
                "mass[j] is the CDF integral in the causal Voronoi cell centered "
                "on lag[j]; j=0 uses [0, dt/2]"
            ),
            "use": "response = numpy.convolve(hidden_driver, mass, mode='full')",
            "relative_kernel": (
                "not constructed; W2/M2/B are separate absolute responses to one hidden driver"
            ),
        },
        "finite_difference_model": {
            "response": "B_nu(T_bright)-B_nu(T_faint) before delay projection",
            "response_mode": "finite_difference",
            "temperature_law": "T(R)=T(R0)*(R/R0)^(-0.75)",
            "central_wavelength_approximation": True,
            "unit_area_normalization": True,
            "amplitude_note": (
                "unit normalization intentionally removes absolute band gains; "
                "do not infer cross-band flux amplitudes from these masses"
            ),
        },
        "physical_inputs_from_f9_table": {
            "mbh_msun": physical["mbh_msun"],
            "eddington_ratio": physical["eddington_ratio"],
            "eta": physical["eta"],
            "x_flux_weighted": physical["x_flux_weighted"],
            "kappa_external_to_internal": physical["kappa_external_to_internal"],
            "redshift": physical["redshift"],
            "scalar_lag_note": (
                "the theoretical lag table supplies a lambda^(4/3) scalar benchmark, "
                "not a transfer-function shape or relative convolution kernel"
            ),
        },
        "simulation_assumptions": {
            "r0_light_days": R0_LIGHT_DAYS,
            "rin_light_days": temperature["rin_light_days"],
            "rin_rg": 6.0,
            "rout_light_days": ROUT_LIGHT_DAYS,
            "inclination_deg": temperature["inclination_deg"],
            "inclination_source": "DiskConfig default; simulation assumption, not F9 constraint",
            "bright_faint_ratio_source": "DiskConfig defaults rescaled at fixed ratio",
            "inner_boundary_zero_torque_factor": "omitted",
            "relativistic_transfer": "omitted",
            "disk_rotation_doppler_gravity": "omitted",
            "illumination_geometry": (
                "no lamppost/Lx geometry; a global bright-minus-faint temperature perturbation"
            ),
        },
        "temperature_scaling": temperature,
        "numerical_grid": {
            "source_nr": NR,
            "source_ntau": NTAU,
            "observer_dt_days": dt,
            "common_grid_points": int(common_lag.size),
            "common_grid_last_lag_days": float(common_lag[-1]),
        },
        "bands": per_band,
        "relative_variance_note": (
            "Only relative centroids are reported.  A variance difference is not claimed "
            "to define a physical positive W2-relative kernel."
        ),
        "files": {
            "kernels": {band: f"disk_kernel_{band}.csv" for band in BANDS},
            "metadata": "disk_parameters.json",
            "plot_png": "f9_absolute_disk_kernels.png",
            "plot_pdf": "f9_absolute_disk_kernels.pdf",
        },
    }

    for band in BANDS:
        _write_kernel_csv(
            output_dir / f"disk_kernel_{band}.csv",
            common_lag,
            common_edges,
            all_mass[band],
        )
    _plot_kernels(output_dir, common_lag, all_mass, dt, metadata)
    (output_dir / "disk_parameters.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {"lag": common_lag, "mass": all_mass, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build finite-difference F9_a absolute W2/M2/B disk kernels."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "outputs" / "kernels",
        help=(
            "Directory below test/ in which kernel CSV/JSON/figures are written "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Observer-frame discrete-kernel spacing in days (default: 0.02).",
    )
    arguments = parser.parse_args()
    result = build_kernels(arguments.output_dir, dt=arguments.dt)
    checks = {
        band: {
            "area": result["metadata"]["bands"][band]["rebinned_moments"]["area"],
            "mean_lag_days": result["metadata"]["bands"][band]["rebinned_moments"][
                "mean_lag_days"
            ],
            "relative_centroid_to_w2_days": result["metadata"]["bands"][band][
                "relative_centroid_to_w2_days"
            ],
        }
        for band in BANDS
    }
    print(json.dumps({"output_dir": str(Path(arguments.output_dir).resolve()), "bands": checks}, indent=2))


if __name__ == "__main__":
    main()
