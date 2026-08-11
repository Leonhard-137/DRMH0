#!/usr/bin/env python3
"""Reproduce diagnostics for the anomalous Mrk 817 U-band two-Gamma fit."""

from __future__ import annotations

import csv
import hashlib
import io
import sqlite3
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path("Mrk817/runs/mica")
ARCHIVE = ROOT / "legacy_archives/gamma0_100_2comp.zip"
OUTPUT = Path("Mrk817/results/tf_comparison")
BANDS = ("UVM2", "UVW1", "U", "B", "V")
GAMMA_K2_FWHM_FACTOR = 2.446386


def quantiles(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.quantile(values, [0.16, 0.5, 0.84]))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_evidence(text: str) -> float:
    return float(text.splitlines()[1].split()[1])


def posterior_quantities(sample: np.ndarray, transfer_function: str):
    amplitude0 = np.exp(sample[:, 4])
    amplitude1 = np.exp(sample[:, 7])
    width0 = np.exp(sample[:, 6])
    width1 = np.exp(sample[:, 9])
    shift = 2.0 if transfer_function == "gamma" else 0.0
    tau0 = sample[:, 5] + shift * width0
    tau1 = sample[:, 8] + shift * width1
    fraction0 = amplitude0 / (amplitude0 + amplitude1)
    ordered = tau0 < tau1
    weighted_centroid = fraction0 * tau0 + (1.0 - fraction0) * tau1
    return {
        "tau0": tau0[ordered],
        "tau1": tau1[ordered],
        "width0": width0[ordered],
        "width1": width1[ordered],
        "fraction0": fraction0[ordered],
        "weighted_centroid": weighted_centroid[ordered],
        "center1": sample[ordered, 8],
        "ordered_fraction": float(np.mean(ordered)),
    }


def write_band_component_comparison(archive: zipfile.ZipFile) -> None:
    rows = []
    for band in BANDS:
        model_samples = (
            ("Gamma", posterior_from_archive(archive, band), "gamma"),
            (
                "Gaussian",
                np.loadtxt(
                    gaussian_run(band) / "data" / "posterior_sample1d.txt_2"
                ),
                "gaussian",
            ),
        )
        for model, sample, transfer_function in model_samples:
            values = posterior_quantities(sample, transfer_function)
            f0_q = quantiles(values["fraction0"])
            tau0_q = quantiles(values["tau0"])
            tau1_q = quantiles(values["tau1"])
            weighted_q = quantiles(values["weighted_centroid"])
            rows.append(
                {
                    "band": band,
                    "model": model,
                    "n_all": len(sample),
                    "ordered_fraction": values["ordered_fraction"],
                    "f0_q16": f0_q[0],
                    "f0_median": f0_q[1],
                    "f0_q84": f0_q[2],
                    "tau0_q16_days": tau0_q[0],
                    "tau0_median_days": tau0_q[1],
                    "tau0_q84_days": tau0_q[2],
                    "tau1_q16_days": tau1_q[0],
                    "tau1_median_days": tau1_q[1],
                    "tau1_q84_days": tau1_q[2],
                    "weighted_tau_q16_days": weighted_q[0],
                    "weighted_tau_median_days": weighted_q[1],
                    "weighted_tau_q84_days": weighted_q[2],
                }
            )
    path = OUTPUT / "Mrk817_gamma_gaussian_band_f_centroid_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def posterior_from_archive(archive: zipfile.ZipFile, band: str) -> np.ndarray:
    member = (
        f"2comp/run_UVW2_to_{band}_2comp_gamma/data/"
        "posterior_sample1d.txt_2"
    )
    return np.loadtxt(io.BytesIO(archive.read(member)))


def gaussian_run(band: str) -> Path:
    return (
        ROOT
        / "gaussian2_uvw2_lag_m10_100"
        / "2comp"
        / f"run_UVW2_to_{band}_2comp_gaussian"
    )


def gamma_member(band: str, leaf: str) -> str:
    return f"2comp/run_UVW2_to_{band}_2comp_gamma/{leaf}"


def write_band_comparison(archive: zipfile.ZipFile) -> None:
    rows = []
    for band in BANDS:
        gamma_sample = posterior_from_archive(archive, band)
        gaussian_path = gaussian_run(band)
        gaussian_sample = np.loadtxt(
            gaussian_path / "data" / "posterior_sample1d.txt_2"
        )
        gamma = posterior_quantities(gamma_sample, "gamma")
        gaussian = posterior_quantities(gaussian_sample, "gaussian")
        gamma_q = quantiles(gamma["tau0"])
        gaussian_q = quantiles(gaussian["tau0"])
        gamma_evidence = parse_evidence(
            archive.read(gamma_member(band, "data/evidence.txt")).decode()
        )
        gaussian_evidence = parse_evidence(
            (gaussian_path / "data" / "evidence.txt").read_text()
        )
        gamma_param = archive.read(
            gamma_member(band, "param/param_input")
        ).decode()
        gaussian_param = (gaussian_path / "param" / "param_input").read_text()
        gamma_upper = float(
            next(
                line.split()[1]
                for line in gamma_param.splitlines()
                if line.startswith("LagLimitUpp")
            )
        )
        gaussian_upper = float(
            next(
                line.split()[1]
                for line in gaussian_param.splitlines()
                if line.startswith("LagLimitUpp")
            )
        )
        inputs_match = all(
            hashlib.sha256(
                archive.read(gamma_member(band, f"input/{name}"))
            ).hexdigest()
            == sha256(gaussian_path / "input" / name)
            for name in ("cont.txt", "line.txt")
        )
        rows.append(
            {
                "band": band,
                "gamma_tau0_q16_days": gamma_q[0],
                "gamma_tau0_median_days": gamma_q[1],
                "gamma_tau0_q84_days": gamma_q[2],
                "gaussian_tau0_q16_days": gaussian_q[0],
                "gaussian_tau0_median_days": gaussian_q[1],
                "gaussian_tau0_q84_days": gaussian_q[2],
                "median_difference_gamma_minus_gaussian_days": (
                    gamma_q[1] - gaussian_q[1]
                ),
                "gamma_ln_evidence": gamma_evidence,
                "gaussian_ln_evidence": gaussian_evidence,
                "delta_ln_evidence_gaussian_minus_gamma": (
                    gaussian_evidence - gamma_evidence
                ),
                "evidence_directly_comparable": gamma_upper == gaussian_upper,
                "inputs_match": inputs_match,
                "gamma_ordered_fraction": gamma["ordered_fraction"],
                "gaussian_ordered_fraction": gaussian["ordered_fraction"],
            }
        )
    path = OUTPUT / "Mrk817_U_gamma_diagnostic_band_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def interpolated_fit_metrics(run: Path, observed: np.ndarray) -> dict[str, float]:
    reconstruction = np.loadtxt(run / "data" / "pall.txt_2")
    response_reconstruction = reconstruction[len(reconstruction) // 2 :]
    prediction = np.interp(
        observed[:, 0], response_reconstruction[:, 0], response_reconstruction[:, 1]
    )
    residual = observed[:, 1] - prediction
    standardized = residual / observed[:, 2]
    return {
        "rmse_flux_units": float(np.sqrt(np.mean(residual**2))),
        "chi2_per_observation": float(np.mean(standardized**2)),
        "mean_standardized_residual": float(np.mean(standardized)),
        "median_absolute_standardized_residual": float(
            np.median(np.abs(standardized))
        ),
    }


def write_u_model_comparison() -> None:
    gamma_run = (
        ROOT
        / "gamma2_uvw2_lag_m10_100"
        / "2comp"
        / "run_UVW2_to_U_2comp_gamma"
    )
    gaussian_path = gaussian_run("U")
    observed = np.loadtxt(gamma_run / "input" / "line.txt")
    continuum_times = np.loadtxt(gamma_run / "input" / "cont.txt")[:, 0]
    response_times = observed[:, 0]
    nearest_offset = np.array(
        [
            continuum_times[np.argmin(np.abs(continuum_times - time))] - time
            for time in response_times
        ]
    )
    cadence = np.diff(np.sort(response_times))

    rows = []
    for name, run, transfer_function in (
        ("Gamma", gamma_run, "gamma"),
        ("Gaussian", gaussian_path, "gaussian"),
    ):
        sample = np.loadtxt(run / "data" / "posterior_sample1d.txt_2")
        values = posterior_quantities(sample, transfer_function)
        tau0 = quantiles(values["tau0"])
        tau1 = quantiles(values["tau1"])
        fraction0 = quantiles(values["fraction0"])
        width0 = quantiles(values["width0"])
        width1 = quantiles(values["width1"])
        fit = interpolated_fit_metrics(run, observed)
        rows.append(
            {
                "model": name,
                "tau0_q16_days": tau0[0],
                "tau0_median_days": tau0[1],
                "tau0_q84_days": tau0[2],
                "width0_median_days": width0[1],
                "tau1_median_days": tau1[1],
                "width1_median_days": width1[1],
                "f0_median": fraction0[1],
                "ln_evidence": parse_evidence(
                    (run / "data" / "evidence.txt").read_text()
                ),
                "corr_tau0_f0": float(
                    np.corrcoef(values["tau0"], values["fraction0"])[0, 1]
                ),
                "corr_tau0_tau1": float(
                    np.corrcoef(values["tau0"], values["tau1"])[0, 1]
                ),
                "corr_tau0_center1": float(
                    np.corrcoef(values["tau0"], values["center1"])[0, 1]
                ),
                "response_n": len(response_times),
                "response_median_cadence_days": float(np.median(cadence)),
                "median_nearest_driver_offset_days": float(
                    np.median(np.abs(nearest_offset))
                ),
                **fit,
            }
        )
    path = OUTPUT / "Mrk817_U_gamma_diagnostic_model_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_u_likelihood_profile() -> None:
    run = (
        ROOT
        / "gamma2_uvw2_lag_m10_100"
        / "2comp"
        / "run_UVW2_to_U_2comp_gamma"
        / "data"
    )
    sample = np.loadtxt(run / "posterior_sample1d.txt_2")
    log_likelihood = np.loadtxt(run / "posterior_sample_info1d.txt_2")
    tau0 = sample[:, 5] + 2.0 * np.exp(sample[:, 6])
    edges = np.array([-0.30, -0.15, 0.00, 0.10, 0.20, 0.30, 0.45, 0.70])
    global_max = float(np.max(log_likelihood))
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (tau0 >= low) & (tau0 < high)
        if not np.any(selected):
            continue
        rows.append(
            {
                "tau0_bin": f"[{low:.2f}, {high:.2f})",
                "tau0_midpoint_days": (low + high) / 2.0,
                "n_samples": int(np.sum(selected)),
                "posterior_fraction": float(np.mean(selected)),
                "median_delta_log_likelihood": float(
                    np.median(log_likelihood[selected]) - global_max
                ),
                "max_delta_log_likelihood": float(
                    np.max(log_likelihood[selected]) - global_max
                ),
            }
        )
    path = OUTPUT / "Mrk817_U_gamma_diagnostic_likelihood_profile.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_u_kernel_location_summary() -> None:
    run = (
        ROOT
        / "gamma2_uvw2_lag_m10_100"
        / "2comp"
        / "run_UVW2_to_U_2comp_gamma"
        / "data"
    )
    sample = np.loadtxt(run / "posterior_sample1d.txt_2")
    onset0 = sample[:, 5]
    width0 = np.exp(sample[:, 6])
    mode0 = onset0 + width0
    tau0 = onset0 + 2.0 * width0
    tau1 = sample[:, 8] + 2.0 * np.exp(sample[:, 9])
    ordered = tau0 < tau1

    rows = []
    for order, name, definition, values in (
        (1, "onset c0", "response onset", onset0[ordered]),
        (2, "mode c0+w0", "fixed-k=2 Gamma mode", mode0[ordered]),
        (3, "centroid c0+2w0", "fixed-k=2 Gamma mean", tau0[ordered]),
    ):
        q16, median, q84 = quantiles(values)
        rows.append(
            {
                "display_order": order,
                "metric": name,
                "q16_days": q16,
                "median_days": median,
                "q84_days": q84,
                "probability_negative": float(np.mean(values < 0.0)),
                "definition": definition,
            }
        )

    path = OUTPUT / "Mrk817_U_gamma_kernel_location_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_gamma_component1_dce_summary(archive: zipfile.ZipFile) -> None:
    """Summarize whether the delayed Gamma component is DCE-like by band.

    The ``broad_dce`` box is deliberately a loose diagnostic region rather than
    a physical prior: a near-zero onset, a several-day-or-broader scale, a
    7--40 day centroid, and non-negligible integrated response fraction.
    Fractions below use all posterior draws in the denominator and include the
    common component-ordering condition.
    """

    def pointwise_median_shape(
        amplitude: np.ndarray, onset: np.ndarray, scale: np.ndarray
    ) -> dict[str, float]:
        lower = max(-10.0, float(np.quantile(onset, 0.001) - 5.0))
        upper = min(
            180.0,
            float(
                np.quantile(onset + 2.0 * scale, 0.999)
                + 10.0 * np.quantile(scale, 0.90)
            ),
        )
        lag = np.linspace(lower, upper, 6001)
        median_response = np.empty_like(lag)
        for start in range(0, len(lag), 500):
            stop = min(start + 500, len(lag))
            offset = lag[None, start:stop] - onset[:, None]
            response = np.where(
                offset >= 0.0,
                amplitude[:, None]
                / scale[:, None] ** 2
                * offset
                * np.exp(-np.maximum(offset, 0.0) / scale[:, None]),
                0.0,
            )
            median_response[start:stop] = np.median(response, axis=0)

        peak_index = int(np.argmax(median_response))
        above_half_max = np.flatnonzero(
            median_response >= 0.5 * median_response[peak_index]
        )
        area = float(np.trapezoid(median_response, lag))
        cumulative_area = np.concatenate(
            [
                [0.0],
                np.cumsum(
                    0.5
                    * (median_response[1:] + median_response[:-1])
                    * np.diff(lag)
                ),
            ]
        ) / area
        area_quantiles = np.interp([0.16, 0.50, 0.84], cumulative_area, lag)
        return {
            "peak_days": float(lag[peak_index]),
            "fwhm_days": float(
                lag[above_half_max[-1]] - lag[above_half_max[0]]
            ),
            "area_q16_days": float(area_quantiles[0]),
            "area_median_days": float(area_quantiles[1]),
            "area_q84_days": float(area_quantiles[2]),
            "area_central68_width_days": float(
                area_quantiles[2] - area_quantiles[0]
            ),
        }

    rows = []
    for band_order, band in enumerate(BANDS, start=1):
        sample = posterior_from_archive(archive, band)
        log_likelihood = np.loadtxt(
            io.BytesIO(
                archive.read(
                    gamma_member(band, "data/posterior_sample_info1d.txt_2")
                )
            )
        )

        amplitude0 = np.exp(sample[:, 4])
        amplitude1 = np.exp(sample[:, 7])
        fraction1 = amplitude1 / (amplitude0 + amplitude1)
        onset0 = sample[:, 5]
        onset1 = sample[:, 8]
        width0 = np.exp(sample[:, 6])
        width1 = np.exp(sample[:, 9])
        tau0 = onset0 + 2.0 * width0
        tau1 = onset1 + 2.0 * width1
        mode1 = onset1 + width1
        ordered = tau0 < tau1

        centroid_7_30 = ordered & (tau1 >= 7.0) & (tau1 <= 30.0)
        centroid_7_40 = ordered & (tau1 >= 7.0) & (tau1 <= 40.0)
        broad_dce = (
            ordered
            & (onset1 >= -2.0)
            & (onset1 <= 10.0)
            & (width1 >= 3.0)
            & (width1 <= 20.0)
            & (tau1 >= 7.0)
            & (tau1 <= 40.0)
            & (fraction1 >= 0.02)
        )
        global_max = float(np.max(log_likelihood))

        def region_summary(selected: np.ndarray) -> tuple[float, float, float]:
            if not np.any(selected):
                return 0.0, float("nan"), float("nan")
            return (
                float(np.mean(selected)),
                float(np.max(log_likelihood[selected]) - global_max),
                float(np.median(log_likelihood[selected]) - global_max),
            )

        centroid_7_30_summary = region_summary(centroid_7_30)
        centroid_7_40_summary = region_summary(centroid_7_40)
        broad_dce_summary = region_summary(broad_dce)
        retained = ordered

        f1_q = quantiles(fraction1[retained])
        onset1_q = quantiles(onset1[retained])
        mode1_q = quantiles(mode1[retained])
        tau1_q = quantiles(tau1[retained])
        width1_q = quantiles(width1[retained])
        fwhm1_q = tuple(GAMMA_K2_FWHM_FACTOR * value for value in width1_q)
        median_shape = pointwise_median_shape(
            amplitude1[retained], onset1[retained], width1[retained]
        )
        parameter_lines = archive.read(
            gamma_member(band, "data/para_names_line.txt_2")
        ).decode()
        width1_parameter_line = next(
            line
            for line in parameter_lines.splitlines()
            if line.strip().startswith("9 1-th_component_sigma")
        )
        width1_prior_min = float(np.exp(float(width1_parameter_line.split()[3])))
        maximum_likelihood_index = int(np.argmax(log_likelihood))
        rows.append(
            {
                "band": band,
                "band_order": band_order,
                "n_all": len(sample),
                "n_ordered": int(np.sum(retained)),
                "ordered_fraction": float(np.mean(retained)),
                "f1_q16": f1_q[0],
                "f1_median": f1_q[1],
                "f1_q84": f1_q[2],
                "onset1_q16_days": onset1_q[0],
                "onset1_median_days": onset1_q[1],
                "onset1_q84_days": onset1_q[2],
                "mode1_q16_days": mode1_q[0],
                "mode1_median_days": mode1_q[1],
                "mode1_q84_days": mode1_q[2],
                "tau1_q16_days": tau1_q[0],
                "tau1_median_days": tau1_q[1],
                "tau1_q84_days": tau1_q[2],
                "width1_q16_days": width1_q[0],
                "width1_median_days": width1_q[1],
                "width1_q84_days": width1_q[2],
                "individual_kernel_fwhm1_q16_days": fwhm1_q[0],
                "individual_kernel_fwhm1_median_days": fwhm1_q[1],
                "individual_kernel_fwhm1_q84_days": fwhm1_q[2],
                "response_std1_median_days": float(np.sqrt(2.0) * width1_q[1]),
                "pointwise_median_kernel1_peak_days": median_shape["peak_days"],
                "pointwise_median_kernel1_fwhm_days": median_shape["fwhm_days"],
                "pointwise_median_kernel1_area_q16_days": median_shape[
                    "area_q16_days"
                ],
                "pointwise_median_kernel1_area_median_days": median_shape[
                    "area_median_days"
                ],
                "pointwise_median_kernel1_area_q84_days": median_shape[
                    "area_q84_days"
                ],
                "pointwise_median_kernel1_area_central68_width_days": median_shape[
                    "area_central68_width_days"
                ],
                "width1_prior_min_days": width1_prior_min,
                "fraction_width1_within_10pct_of_prior_min": float(
                    np.mean(width1[retained] < 1.10 * width1_prior_min)
                ),
                "maximum_likelihood_kernel1_fwhm_days": float(
                    GAMMA_K2_FWHM_FACTOR * width1[maximum_likelihood_index]
                ),
                "probability_onset1_nonnegative": float(
                    np.mean(onset1[retained] >= 0.0)
                ),
                "centroid_7_30_posterior_fraction": centroid_7_30_summary[0],
                "centroid_7_30_best_delta_log_likelihood": centroid_7_30_summary[1],
                "centroid_7_30_median_delta_log_likelihood": centroid_7_30_summary[2],
                "centroid_7_40_posterior_fraction": centroid_7_40_summary[0],
                "centroid_7_40_best_delta_log_likelihood": centroid_7_40_summary[1],
                "centroid_7_40_median_delta_log_likelihood": centroid_7_40_summary[2],
                "broad_dce_posterior_fraction": broad_dce_summary[0],
                "broad_dce_best_delta_log_likelihood": broad_dce_summary[1],
                "broad_dce_median_delta_log_likelihood": broad_dce_summary[2],
            }
        )

    path = OUTPUT / "Mrk817_gamma_component1_dce_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_v_delayed_transfer_curve(archive: zipfile.ZipFile) -> None:
    transfer = np.loadtxt(
        io.BytesIO(
            archive.read(
                gamma_member("V", "data/tranfunc_0_1.txt_2")
            )
        )
    )
    selected = (transfer[:, 0] >= 20.0) & (transfer[:, 0] <= 50.0)
    rows = []
    for lag, median, lower_error, upper_error in transfer[selected]:
        rows.append(
            {
                "lag_days": float(lag),
                "posterior_median_transfer": float(median),
                "posterior_q16_transfer": float(median - lower_error),
                "posterior_q84_transfer": float(median + upper_error),
            }
        )

    path = OUTPUT / "Mrk817_V_gamma_delayed_transfer_curve.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def sqlite_value(value: str):
    lowered = value.lower()
    if lowered == "true":
        return 1
    if lowered == "false":
        return 0
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def write_sqlite_snapshot() -> None:
    database = OUTPUT / "Mrk817_U_gamma_diagnostic.sqlite"
    table_files = {
        "band_comparison": OUTPUT / "Mrk817_U_gamma_diagnostic_band_comparison.csv",
        "u_model_comparison": OUTPUT
        / "Mrk817_U_gamma_diagnostic_model_comparison.csv",
        "u_likelihood_profile": OUTPUT
        / "Mrk817_U_gamma_diagnostic_likelihood_profile.csv",
        "u_kernel_location": OUTPUT
        / "Mrk817_U_gamma_kernel_location_summary.csv",
        "band_component_comparison": OUTPUT
        / "Mrk817_gamma_gaussian_band_f_centroid_comparison.csv",
        "gamma_component1_dce": OUTPUT
        / "Mrk817_gamma_component1_dce_summary.csv",
        "v_gamma_delayed_transfer_curve": OUTPUT
        / "Mrk817_V_gamma_delayed_transfer_curve.csv",
    }
    with sqlite3.connect(database) as connection:
        for table, path in table_files.items():
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                raw_rows = list(reader)
                columns = reader.fieldnames or []
            connection.execute(f'DROP TABLE IF EXISTS "{table}"')
            column_sql = ", ".join(f'"{column}"' for column in columns)
            connection.execute(f'CREATE TABLE "{table}" ({column_sql})')
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                [tuple(sqlite_value(row[column]) for column in columns) for row in raw_rows],
            )
            # Execute the same source query exposed in the report artifact.
            connection.execute(f'SELECT * FROM "{table}"').fetchall()
        connection.commit()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        write_band_comparison(archive)
        write_band_component_comparison(archive)
        write_gamma_component1_dce_summary(archive)
        write_v_delayed_transfer_curve(archive)
    write_u_model_comparison()
    write_u_likelihood_profile()
    write_u_kernel_location_summary()
    write_sqlite_snapshot()
    for path in sorted(OUTPUT.glob("Mrk817_U_gamma_diagnostic_*.csv")):
        print(path)
    print(OUTPUT / "Mrk817_gamma_gaussian_band_f_centroid_comparison.csv")
    print(OUTPUT / "Mrk817_gamma_component1_dce_summary.csv")
    print(OUTPUT / "Mrk817_V_gamma_delayed_transfer_curve.csv")
    print(OUTPUT / "Mrk817_U_gamma_kernel_location_summary.csv")
    print(OUTPUT / "Mrk817_U_gamma_diagnostic.sqlite")


if __name__ == "__main__":
    main()
