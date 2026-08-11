"""Fit shifted-Gamma profiles to the exact disk defined in test_disk_transfer.py.

Place this file in the same directory as:

    disk_transfer.py
    gamma_mcmc.py
    test_disk_transfer.py

Run all wavelengths:

    python fit_test_disk_gamma.py

Quick workflow test:

    python fit_test_disk_gamma.py --quick

Fit one wavelength only:

    python fit_test_disk_gamma.py --wavelength 1230

Fit all wavelengths with k fixed at 2:

    python fit_test_disk_gamma.py --fixed-k 2 --output gamma_test_disk_outputs/k_fixed_2

The disk parameters and numerical grids are imported directly from
``test_disk_transfer.py``. Therefore this script cannot silently drift away
from the disk used to create ``scientific_overview.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import test_disk_transfer as disk_config
from disk_transfer import DiskTransferModel, mdot_from_eddington_ratio
from gamma_mcmc import (
    fit_shifted_gamma_free_area_mcmc,
    fit_shifted_gamma_mcmc,
    print_summary,
    save_results,
    shifted_gamma_free_area_on_grid,
    shifted_gamma_on_grid,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the exact disk in test_disk_transfer.py and fit an "
            "independent shifted-Gamma density to each selected wavelength."
        )
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=None,
        help=(
            "Fit only the nearest configured wavelength in Angstrom. "
            "The default fits every configured wavelength."
        ),
    )
    parser.add_argument(
        "--response",
        choices=("pulsed", "instant"),
        default="pulsed",
        help="Which disk response to fit. Default: pulsed.",
    )
    parser.add_argument(
        "--fit-dt",
        type=float,
        default=0.01,
        help=(
            "Approximate time spacing used inside the MCMC likelihood, in days. "
            "The final comparison curve is still evaluated on the native grid. "
            "Default: 0.01 day."
        ),
    )
    parser.add_argument(
        "--tail-factor",
        type=float,
        default=1.20,
        help=(
            "Fit through tail_factor times the disk q99.9%% lag. "
            "Default: 1.20."
        ),
    )
    parser.add_argument(
        "--fixed-k",
        type=float,
        default=None,
        help=(
            "Hold the Gamma shape k fixed at this positive value and sample "
            "only theta, t0, and model scatter. Default: sample k freely."
        ),
    )
    parser.add_argument(
        "--free-area",
        action="store_true",
        help=(
            "Fit a positive area scale without renormalizing either the "
            "cropped target or Gamma model. Requires --fixed-k."
        ),
    )
    parser.add_argument(
        "--nwalkers",
        type=int,
        default=32,
        help="Number of emcee walkers. Default: 32.",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=3000,
        help="MCMC steps per walker. Default: 3000.",
    )
    parser.add_argument(
        "--burn",
        type=int,
        default=1000,
        help="Burn-in steps discarded from every walker. Default: 1000.",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=5,
        help="Posterior thinning factor. Default: 5.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a short chain for a workflow check, not for final inference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gamma_test_disk_outputs"),
        help="Output directory. Default: gamma_test_disk_outputs.",
    )
    return parser.parse_args()


def build_exact_disk() -> tuple[DiskTransferModel, dict]:
    """Generate the disk using the constants imported from the test script."""
    mdot_g_s = mdot_from_eddington_ratio(
        disk_config.M_BH_MSUN,
        disk_config.EDDINGTON_RATIO,
        disk_config.RADIATIVE_EFFICIENCY,
    )

    model = DiskTransferModel(
        wavelengths_angstrom=disk_config.WAVELENGTHS_ANGSTROM,
        time_days=disk_config.TIME_DAYS,
        global_rmin_rg=6.0,
        rout_rg=disk_config.ROUT_RG,
        nr=disk_config.NR,
        nphi=disk_config.NPHI,
        pulse_width_days=disk_config.PULSE_WIDTH_DAYS,
    )

    prediction = model.predict(
        m_bh_msun=disk_config.M_BH_MSUN,
        mdot_g_s=mdot_g_s,
        lx_erg_s=disk_config.L_X_ERG_S,
        h_rg=disk_config.H_RG,
        inclination_deg=disk_config.INCLINATION_DEG,
        rin_rg=disk_config.RIN_RG,
        reference_index=disk_config.REFERENCE_INDEX,
    )
    return model, prediction


def select_wavelength_indices(requested: float | None) -> np.ndarray:
    wavelengths = np.asarray(
        disk_config.WAVELENGTHS_ANGSTROM,
        dtype=float,
    )
    if requested is None:
        return np.arange(wavelengths.size, dtype=int)

    index = int(np.argmin(np.abs(wavelengths - requested)))
    print(
        f"Requested {requested:.3f} A; fitting configured "
        f"wavelength {wavelengths[index]:.1f} A."
    )
    return np.array([index], dtype=int)


def block_average_density(
    time_days: np.ndarray,
    density: np.ndarray,
    *,
    time_max_days: float,
    target_dt_days: float,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop and block-average a uniformly sampled density for faster MCMC."""
    time_days = np.asarray(time_days, dtype=float)
    density = np.asarray(density, dtype=float)

    if target_dt_days <= 0.0:
        raise ValueError("fit-dt must be positive.")

    mask = time_days <= time_max_days
    time_crop = time_days[mask]
    density_crop = density[mask]

    native_dt = float(np.median(np.diff(time_crop)))
    factor = max(1, int(round(target_dt_days / native_dt)))

    if factor == 1:
        fit_time = time_crop.copy()
        fit_density = density_crop.copy()
    else:
        n_complete = (time_crop.size // factor) * factor
        if n_complete < 5 * factor:
            raise ValueError(
                "The cropped response has too few bins for the requested fit-dt."
            )

        fit_time = time_crop[:n_complete].reshape(-1, factor).mean(axis=1)
        fit_density = density_crop[:n_complete].reshape(-1, factor).mean(axis=1)

    fit_density = np.clip(fit_density, 0.0, None)
    integral = float(np.trapezoid(fit_density, fit_time))
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError("The cropped response has no positive integral.")

    if normalize:
        fit_density = fit_density / integral
    return fit_time, fit_density


def posterior_median_curve_on_native_grid(
    result: dict,
    native_time_days: np.ndarray,
) -> np.ndarray:
    summaries = result["summaries"]
    if result.get("free_area"):
        curve = shifted_gamma_free_area_on_grid(
            native_time_days,
            np.log(summaries["k"].median),
            np.log(summaries["theta_days"].median),
            summaries["t0_days"].median,
            np.log(summaries["area_scale"].median),
        )
    else:
        curve = shifted_gamma_on_grid(
            native_time_days,
            np.log(summaries["k"].median),
            np.log(summaries["theta_days"].median),
            summaries["t0_days"].median,
        )
    if curve is None:
        raise RuntimeError("Posterior median produced an invalid Gamma curve.")
    return curve


def save_native_grid_comparison(
    *,
    time_days: np.ndarray,
    disk_response: np.ndarray,
    gamma_response: np.ndarray,
    wavelength_angstrom: float,
    fit_limit_days: float,
    gamma_label: str,
    output_directory: Path,
) -> None:
    table = np.column_stack(
        [time_days, disk_response, gamma_response]
    )
    np.savetxt(
        output_directory / "full_resolution_fit.csv",
        table,
        delimiter=",",
        header="time_days,disk_response,gamma_median",
        comments="",
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(
        time_days,
        disk_response,
        linewidth=1.7,
        label="Disk response",
    )
    axis.plot(
        time_days,
        gamma_response,
        linestyle="--",
        linewidth=1.7,
        label=gamma_label,
    )
    axis.set_xlim(0.0, fit_limit_days)
    axis.set_xlabel("Time delay [days]")
    axis.set_ylabel("Response density [day$^{-1}$]")
    axis.set_title(f"{wavelength_angstrom:.0f} Angstrom")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_directory / "gamma_fit_full_resolution.png",
        dpi=200,
    )
    plt.close(figure)


def write_disk_parameter_record(
    *,
    output_directory: Path,
    model: DiskTransferModel,
    prediction: dict,
    response_key: str,
    fit_dt_days: float,
    fixed_k: float | None,
    nwalkers: int,
    nsteps: int,
    burn: int,
    thin: int,
    free_area: bool,
) -> None:
    parameters = prediction["parameters"]
    record = {
        "source_configuration": "test_disk_transfer.py",
        "response_key": response_key,
        "physical_parameters": {
            "m_bh_msun": parameters.m_bh_msun,
            "eddington_ratio": disk_config.EDDINGTON_RATIO,
            "radiative_efficiency": disk_config.RADIATIVE_EFFICIENCY,
            "mdot_g_s": parameters.mdot_g_s,
            "lx_erg_s": parameters.lx_erg_s,
            "h_rg": parameters.h_rg,
            "inclination_deg": parameters.inclination_deg,
            "rin_rg": parameters.rin_rg,
            "rout_rg": disk_config.ROUT_RG,
        },
        "native_grid": {
            "wavelengths_angstrom": (
                np.asarray(disk_config.WAVELENGTHS_ANGSTROM).tolist()
            ),
            "time_start_days": float(disk_config.TIME_DAYS[0]),
            "time_stop_days": float(disk_config.TIME_DAYS[-1]),
            "time_step_days": float(model.dt_days),
            "nr": disk_config.NR,
            "nphi": disk_config.NPHI,
            "pulse_width_days": disk_config.PULSE_WIDTH_DAYS,
        },
        "mcmc_fit_grid": {
            "target_time_step_days": fit_dt_days,
        },
        "mcmc": {
            "k_mode": "free" if fixed_k is None else "fixed",
            "fixed_k": fixed_k,
            "area_mode": "free" if free_area else "unit_area",
            "nwalkers": nwalkers,
            "nsteps": nsteps,
            "burn": burn,
            "thin": thin,
        },
    }
    (output_directory / "disk_parameters.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    arguments = parse_arguments()

    if arguments.quick:
        arguments.nsteps = 800
        arguments.burn = 250
        arguments.thin = 2

    if arguments.burn >= arguments.nsteps:
        raise ValueError("burn must be smaller than nsteps.")
    if arguments.fixed_k is not None and (
        not np.isfinite(arguments.fixed_k) or arguments.fixed_k <= 0.0
    ):
        raise ValueError("fixed-k must be a finite positive number.")
    if arguments.free_area and arguments.fixed_k is None:
        raise ValueError("free-area currently requires --fixed-k.")
    if arguments.tail_factor <= 1.0:
        raise ValueError("tail-factor should be greater than 1.")

    response_key = (
        "pulsed_response"
        if arguments.response == "pulsed"
        else "instant_response"
    )

    arguments.output.mkdir(parents=True, exist_ok=True)

    print("Generating the exact disk from test_disk_transfer.py ...")
    model, prediction = build_exact_disk()
    print("Disk calculation complete.")

    write_disk_parameter_record(
        output_directory=arguments.output,
        model=model,
        prediction=prediction,
        response_key=response_key,
        fit_dt_days=arguments.fit_dt,
        fixed_k=arguments.fixed_k,
        nwalkers=arguments.nwalkers,
        nsteps=arguments.nsteps,
        burn=arguments.burn,
        thin=arguments.thin,
        free_area=arguments.free_area,
    )

    selected_indices = select_wavelength_indices(arguments.wavelength)
    wavelengths = np.asarray(
        prediction["wavelengths_angstrom"],
        dtype=float,
    )
    native_time = np.asarray(prediction["time_days"], dtype=float)

    summary_rows: list[dict[str, float | str]] = []
    combined_results: list[
        tuple[float, np.ndarray, np.ndarray, float]
    ] = []

    for sequence, index in enumerate(selected_indices):
        wavelength = float(wavelengths[index])
        disk_response = np.asarray(
            prediction[response_key][index],
            dtype=float,
        )

        fit_limit = min(
            float(native_time[-1]),
            max(
                1.0,
                arguments.tail_factor
                * float(prediction["q999_lag_days"][index]),
            ),
        )

        fit_time, fit_response = block_average_density(
            native_time,
            disk_response,
            time_max_days=fit_limit,
            target_dt_days=arguments.fit_dt,
            normalize=not arguments.free_area,
        )

        print("\n" + "=" * 72)
        print(
            f"Fitting {wavelength:.1f} A  "
            f"({fit_time.size} likelihood bins, "
            f"0-{fit_time[-1]:.3f} d)"
        )
        if arguments.fixed_k is not None:
            print(f"Gamma shape fixed at k={arguments.fixed_k:g}")
        if arguments.free_area:
            print("Gamma area scale is free; no finite-grid renormalization")
        print("=" * 72)

        if arguments.free_area:
            result = fit_shifted_gamma_free_area_mcmc(
                fit_time,
                fit_response,
                fixed_k=arguments.fixed_k,
                nwalkers=arguments.nwalkers,
                nsteps=arguments.nsteps,
                burn=arguments.burn,
                thin=arguments.thin,
                random_seed=1234 + sequence,
                progress=True,
            )
        else:
            result = fit_shifted_gamma_mcmc(
                fit_time,
                fit_response,
                fixed_k=arguments.fixed_k,
                nwalkers=arguments.nwalkers,
                nsteps=arguments.nsteps,
                burn=arguments.burn,
                thin=arguments.thin,
                random_seed=1234 + sequence,
                progress=True,
            )
        print_summary(result)

        band_directory = (
            arguments.output / f"{wavelength:.0f}_angstrom"
        )
        save_results(result, band_directory)

        gamma_native = posterior_median_curve_on_native_grid(
            result,
            native_time,
        )
        gamma_label = "Gamma median"
        if arguments.fixed_k is not None:
            gamma_label += f" (k={arguments.fixed_k:g} fixed)"
        if arguments.free_area:
            gamma_label += ", area free"
        save_native_grid_comparison(
            time_days=native_time,
            disk_response=disk_response,
            gamma_response=gamma_native,
            wavelength_angstrom=wavelength,
            fit_limit_days=fit_limit,
            gamma_label=gamma_label,
            output_directory=band_directory,
        )

        fit_mask = native_time <= fit_limit
        residual = disk_response[fit_mask] - gamma_native[fit_mask]
        rmse = float(np.sqrt(np.mean(residual**2)))
        integrated_absolute_error = float(
            np.trapezoid(
                np.abs(disk_response - gamma_native),
                native_time,
            )
        )

        summaries = result["summaries"]
        k = summaries["k"].median
        theta = summaries["theta_days"].median
        t0 = summaries["t0_days"].median
        fitted_area = float(np.trapezoid(gamma_native, native_time))
        gamma_mean = float(
            np.trapezoid(native_time * gamma_native, native_time) / fitted_area
        )
        gamma_mode = (
            t0 + (k - 1.0) * theta
            if k > 1.0
            else t0
        )

        summary_row = {
            "wavelength_angstrom": wavelength,
            "response_type": arguments.response,
            "disk_peak_lag_days": float(
                prediction["peak_lag_days"][index]
            ),
            "disk_mean_lag_days": float(
                prediction["mean_lag_days"][index]
            ),
            "disk_median_lag_days": float(
                prediction["median_lag_days"][index]
            ),
            "gamma_k": k,
            "gamma_k_minus": summaries["k"].lower,
            "gamma_k_plus": summaries["k"].upper,
            "gamma_theta_days": theta,
            "gamma_theta_minus": summaries["theta_days"].lower,
            "gamma_theta_plus": summaries["theta_days"].upper,
            "gamma_t0_days": t0,
            "gamma_t0_minus": summaries["t0_days"].lower,
            "gamma_t0_plus": summaries["t0_days"].upper,
            "gamma_sigma": summaries["sigma"].median,
            "gamma_mean_days": gamma_mean,
            "gamma_mode_days": gamma_mode,
            "fit_limit_days": fit_limit,
            "fit_grid_bins": int(fit_time.size),
            "rmse_fit_window": rmse,
            "integrated_absolute_error": integrated_absolute_error,
            "acceptance_fraction": result["acceptance_fraction"],
        }
        if arguments.free_area:
            summary_row.update({
                "gamma_area_scale": summaries["area_scale"].median,
                "gamma_area_scale_minus": summaries["area_scale"].lower,
                "gamma_area_scale_plus": summaries["area_scale"].upper,
                "gamma_fitted_area": fitted_area,
            })
        summary_rows.append(summary_row)
        combined_results.append(
            (wavelength, disk_response, gamma_native, fit_limit)
        )

    summary_path = arguments.output / "multiwavelength_gamma_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # One compact comparison figure for all selected bands.
    n_bands = len(combined_results)
    n_columns = min(3, n_bands)
    n_rows = int(np.ceil(n_bands / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.0 * n_columns, 3.4 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for axis, (wavelength, disk_response, gamma_native, fit_limit) in zip(
        flat_axes,
        combined_results,
    ):
        mask = native_time <= fit_limit
        axis.plot(
            native_time[mask],
            disk_response[mask],
            linewidth=1.5,
            label="Disk",
        )
        combined_gamma_label = (
            "Gamma"
            if arguments.fixed_k is None
            else f"Gamma (k={arguments.fixed_k:g} fixed)"
        )
        if arguments.free_area:
            combined_gamma_label += ", area free"
        axis.plot(
            native_time[mask],
            gamma_native[mask],
            linestyle="--",
            linewidth=1.5,
            label=combined_gamma_label,
        )
        axis.set_title(f"{wavelength:.0f} Angstrom")
        axis.set_xlabel("Delay [day]")
        axis.set_ylabel("Response [day$^{-1}$]")

    for axis in flat_axes[n_bands:]:
        axis.set_visible(False)

    flat_axes[0].legend()
    figure.tight_layout()
    figure.savefig(
        arguments.output / "all_wavelength_gamma_fits.png",
        dpi=200,
    )
    plt.close(figure)

    print("\n" + "=" * 72)
    print(f"All outputs written to: {arguments.output.resolve()}")
    print(f"Summary table: {summary_path.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
