"""Fit Gaussian profiles to the exact test disk responses."""

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
from fit_test_disk_gamma import (
    block_average_density,
    build_exact_disk,
    select_wavelength_indices,
)
from gaussian_mcmc import (
    fit_gaussian_free_area_mcmc,
    fit_gaussian_mcmc,
    gaussian_free_area_on_grid,
    gaussian_on_grid,
    print_summary,
    save_results,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit one Gaussian profile to each exact disk response."
    )
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument(
        "--response", choices=("pulsed", "instant"), default="pulsed"
    )
    parser.add_argument("--fit-dt", type=float, default=0.01)
    parser.add_argument("--tail-factor", type=float, default=1.20)
    parser.add_argument("--nwalkers", type=int, default=32)
    parser.add_argument("--nsteps", type=int, default=3000)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--free-area",
        action="store_true",
        help=(
            "Fit a positive area scale without renormalizing either the "
            "cropped target or Gaussian model."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gamma_test_disk_outputs/gaussian"),
    )
    return parser.parse_args()


def posterior_median_curve_on_native_grid(
    result: dict,
    native_time_days: np.ndarray,
) -> np.ndarray:
    summaries = result["summaries"]
    if result.get("free_area"):
        curve = gaussian_free_area_on_grid(
            native_time_days,
            summaries["center_days"].median,
            np.log(summaries["width_days"].median),
            np.log(summaries["area_scale"].median),
        )
    else:
        curve = gaussian_on_grid(
            native_time_days,
            summaries["center_days"].median,
            np.log(summaries["width_days"].median),
        )
    if curve is None:
        raise RuntimeError("Posterior median produced an invalid Gaussian curve.")
    return curve


def save_native_grid_comparison(
    *,
    time_days: np.ndarray,
    disk_response: np.ndarray,
    gaussian_response: np.ndarray,
    wavelength_angstrom: float,
    fit_limit_days: float,
    free_area: bool,
    output_directory: Path,
) -> None:
    np.savetxt(
        output_directory / "full_resolution_fit.csv",
        np.column_stack([time_days, disk_response, gaussian_response]),
        delimiter=",",
        header="time_days,disk_response,gaussian_median",
        comments="",
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(
        time_days,
        disk_response,
        color="#2563EB",
        linewidth=1.7,
        label="Disk response",
    )
    axis.plot(
        time_days,
        gaussian_response,
        color="#D97706",
        linestyle="--",
        linewidth=1.7,
        label=("Gaussian median, area free" if free_area else "Gaussian median"),
    )
    axis.set_xlim(0.0, fit_limit_days)
    axis.set_xlabel("Time delay [days]")
    axis.set_ylabel("Response density [day$^{-1}$]")
    axis.set_title(f"{wavelength_angstrom:.0f} Angstrom")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "gaussian_fit_full_resolution.png", dpi=200)
    plt.close(figure)


def write_parameter_record(
    *,
    output_directory: Path,
    model,
    prediction: dict,
    response_key: str,
    arguments: argparse.Namespace,
) -> None:
    parameters = prediction["parameters"]
    record = {
        "source_configuration": "test_disk_transfer.py",
        "response_key": response_key,
        "profile": (
            "analytic Gaussian density with free area scale"
            if arguments.free_area
            else "unit-area Gaussian normalized on the finite time grid"
        ),
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
            "wavelengths_angstrom": np.asarray(
                disk_config.WAVELENGTHS_ANGSTROM
            ).tolist(),
            "time_start_days": float(disk_config.TIME_DAYS[0]),
            "time_stop_days": float(disk_config.TIME_DAYS[-1]),
            "time_step_days": float(model.dt_days),
            "nr": disk_config.NR,
            "nphi": disk_config.NPHI,
            "pulse_width_days": disk_config.PULSE_WIDTH_DAYS,
        },
        "mcmc": {
            "target_time_step_days": arguments.fit_dt,
            "tail_factor": arguments.tail_factor,
            "area_mode": "free" if arguments.free_area else "unit_area",
            "nwalkers": arguments.nwalkers,
            "nsteps": arguments.nsteps,
            "burn": arguments.burn,
            "thin": arguments.thin,
        },
    }
    (output_directory / "disk_parameters.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.quick:
        arguments.nsteps = 800
        arguments.burn = 250
        arguments.thin = 2
    if arguments.burn >= arguments.nsteps:
        raise ValueError("burn must be smaller than nsteps.")
    if arguments.tail_factor <= 1.0:
        raise ValueError("tail-factor should be greater than 1.")

    response_key = (
        "pulsed_response" if arguments.response == "pulsed" else "instant_response"
    )
    arguments.output.mkdir(parents=True, exist_ok=True)

    print("Generating the exact disk from test_disk_transfer.py ...")
    model, prediction = build_exact_disk()
    print("Disk calculation complete.")
    write_parameter_record(
        output_directory=arguments.output,
        model=model,
        prediction=prediction,
        response_key=response_key,
        arguments=arguments,
    )

    selected_indices = select_wavelength_indices(arguments.wavelength)
    wavelengths = np.asarray(prediction["wavelengths_angstrom"], dtype=float)
    native_time = np.asarray(prediction["time_days"], dtype=float)
    summary_rows: list[dict[str, float | str]] = []
    combined_results: list[tuple[float, np.ndarray, np.ndarray, float]] = []

    for sequence, index in enumerate(selected_indices):
        wavelength = float(wavelengths[index])
        disk_response = np.asarray(prediction[response_key][index], dtype=float)
        fit_limit = min(
            float(native_time[-1]),
            max(
                1.0,
                arguments.tail_factor * float(prediction["q999_lag_days"][index]),
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
            f"({fit_time.size} likelihood bins, 0-{fit_time[-1]:.3f} d)"
        )
        print("=" * 72)
        if arguments.free_area:
            print("Gaussian area scale is free; no finite-grid renormalization")
            result = fit_gaussian_free_area_mcmc(
                fit_time,
                fit_response,
                nwalkers=arguments.nwalkers,
                nsteps=arguments.nsteps,
                burn=arguments.burn,
                thin=arguments.thin,
                random_seed=2234 + sequence,
                progress=True,
            )
        else:
            result = fit_gaussian_mcmc(
                fit_time,
                fit_response,
                nwalkers=arguments.nwalkers,
                nsteps=arguments.nsteps,
                burn=arguments.burn,
                thin=arguments.thin,
                random_seed=2234 + sequence,
                progress=True,
            )
        print_summary(result)

        band_directory = arguments.output / f"{wavelength:.0f}_angstrom"
        save_results(result, band_directory)
        gaussian_native = posterior_median_curve_on_native_grid(result, native_time)
        save_native_grid_comparison(
            time_days=native_time,
            disk_response=disk_response,
            gaussian_response=gaussian_native,
            wavelength_angstrom=wavelength,
            fit_limit_days=fit_limit,
            free_area=arguments.free_area,
            output_directory=band_directory,
        )

        fit_mask = native_time <= fit_limit
        residual = disk_response[fit_mask] - gaussian_native[fit_mask]
        rmse = float(np.sqrt(np.mean(residual**2)))
        integrated_absolute_error = float(
            np.trapezoid(np.abs(disk_response - gaussian_native), native_time)
        )
        fitted_area = float(np.trapezoid(gaussian_native, native_time))
        gaussian_mean = float(
            np.trapezoid(native_time * gaussian_native, native_time)
            / fitted_area
        )
        gaussian_variance = float(
            np.trapezoid(
                (native_time - gaussian_mean) ** 2 * gaussian_native,
                native_time,
            )
            / fitted_area
        )
        summaries = result["summaries"]
        summary_row = {
            "wavelength_angstrom": wavelength,
            "response_type": arguments.response,
            "disk_peak_lag_days": float(prediction["peak_lag_days"][index]),
            "disk_mean_lag_days": float(prediction["mean_lag_days"][index]),
            "disk_median_lag_days": float(prediction["median_lag_days"][index]),
            "gaussian_center_days": summaries["center_days"].median,
            "gaussian_center_minus": summaries["center_days"].lower,
            "gaussian_center_plus": summaries["center_days"].upper,
            "gaussian_width_days": summaries["width_days"].median,
            "gaussian_width_minus": summaries["width_days"].lower,
            "gaussian_width_plus": summaries["width_days"].upper,
            "gaussian_model_scatter": summaries["model_scatter"].median,
            "gaussian_mean_days": gaussian_mean,
            "gaussian_variance_days2": gaussian_variance,
            "fit_limit_days": fit_limit,
            "fit_grid_bins": int(fit_time.size),
            "rmse_fit_window": rmse,
            "integrated_absolute_error": integrated_absolute_error,
            "acceptance_fraction": result["acceptance_fraction"],
        }
        if arguments.free_area:
            summary_row.update({
                "gaussian_area_scale": summaries["area_scale"].median,
                "gaussian_area_scale_minus": summaries["area_scale"].lower,
                "gaussian_area_scale_plus": summaries["area_scale"].upper,
                "gaussian_fitted_area": fitted_area,
            })
        summary_rows.append(summary_row)
        combined_results.append(
            (wavelength, disk_response, gaussian_native, fit_limit)
        )

    summary_path = arguments.output / "multiwavelength_gaussian_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

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
    for axis, (wavelength, disk_response, gaussian_native, fit_limit) in zip(
        flat_axes, combined_results
    ):
        mask = native_time <= fit_limit
        axis.plot(
            native_time[mask],
            disk_response[mask],
            color="#2563EB",
            linewidth=1.5,
            label="Disk",
        )
        axis.plot(
            native_time[mask],
            gaussian_native[mask],
            color="#D97706",
            linestyle="--",
            linewidth=1.5,
            label=("Gaussian, area free" if arguments.free_area else "Gaussian"),
        )
        axis.set_title(f"{wavelength:.0f} Angstrom")
        axis.set_xlabel("Delay [day]")
        axis.set_ylabel("Response [day$^{-1}$]")
    for axis in flat_axes[n_bands:]:
        axis.set_visible(False)
    flat_axes[0].legend()
    figure.tight_layout()
    figure.savefig(arguments.output / "all_wavelength_gaussian_fits.png", dpi=200)
    plt.close(figure)

    print("\n" + "=" * 72)
    print(f"All outputs written to: {arguments.output.resolve()}")
    print(f"Summary table: {summary_path.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
