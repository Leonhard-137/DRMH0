#!/usr/bin/env python3
"""Prepare Mrk 817 fast residuals plus the original mean; never start a fit."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGES = ("mica_gaussian2_free", "mica_gamma2_free")
BANDS = ("UVW2", "UVM2", "UVW1", "U", "B", "V")


def resolve(base: Path, value: str) -> Path:
    return (base / Path(value).expanduser()).resolve()


def read_curve(path: Path) -> np.ndarray:
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 3 or len(data) < 2:
        raise ValueError(f"Expected at least two time/flux/error rows: {path}")
    if not np.isfinite(data).all() or np.any(data[:, 2] <= 0):
        raise ValueError(f"Nonfinite data or nonpositive errors: {path}")
    if np.any(np.diff(data[:, 0]) < 0):
        raise ValueError(f"Observation times must be sorted: {path}")
    return data


def gaussian_weights(times: np.ndarray, sigma_days: float) -> np.ndarray:
    """Normalized Gaussian running mean at the original (possibly irregular) epochs.

    Each observation receives its Gaussian time-distance weight. There is no
    inverse-variance weighting, interpolation, padding, or trend subtraction.
    Near the campaign boundaries the available weights are renormalized.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("times must be a finite, nonempty vector")
    if not np.isfinite(sigma_days) or sigma_days <= 0:
        raise ValueError("sigma_days must be finite and positive")
    delta = (times[:, None] - times[None, :]) / sigma_days
    weights = np.exp(-0.5 * delta**2)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def detrend_curve(data: np.ndarray, sigma_days: float,
                  error_mode: str = "original") -> tuple[np.ndarray, np.ndarray]:
    """Return original - Gaussian trend + original mean and its covariance.

    W estimates the slow trend; I-W retains the fast residual. The restored
    sample mean is a constant flux baseline for MICA normalization and leaves
    flux differences unchanged. Covariance includes its measurement noise via
    J_ij=1/N: C=(I-W+J) diag(error**2) (I-W+J).T.
    Original quoted errors remain the default MICA input; propagated mode uses
    the diagonal of C. MICA does not consume C's off-diagonal entries.
    """
    if error_mode not in {"original", "propagated"}:
        raise ValueError("error_mode must be original or propagated")
    weights = gaussian_weights(data[:, 0], sigma_days)
    transform = np.eye(len(data)) - weights + np.full_like(weights, 1.0 / len(data))
    noise_map = transform * data[:, 2][None, :]
    covariance = noise_map @ noise_map.T
    result = data.copy()
    result[:, 1] = data[:, 1] - weights @ data[:, 1] + np.mean(data[:, 1])
    if error_mode == "propagated":
        result[:, 2] = np.sqrt(np.diag(covariance))
    return result, covariance


def case_config(base: dict, spec: dict, experiment: dict) -> dict:
    keys = ("source", "reference_band", "driver_band", "redshift", "band_list",
            "response_bands", "bands", "lambda_tau", "fflux", "fflux_comp0")
    cfg = {key: deepcopy(base[key]) for key in keys if key in base}
    if cfg.get("source") != "Mrk817_a" or cfg.get("driver_band") != "UVW2":
        raise ValueError("This experiment expects source Mrk817_a and driver UVW2")
    if cfg.get("band_list") != list(BANDS) or cfg.get("response_bands") != list(BANDS[1:]):
        raise ValueError("This experiment requires UVW2 and all five specified responses")
    cfg["pipeline"] = {"ffa_mpi": 2}
    cfg["mica_stages"] = list(STAGES)
    for stage in STAGES:
        fit = deepcopy(base[stage])
        fit["prepared_dir"] = "prepared"
        fit["max_parallel"] = int(experiment.get("max_parallel", 1))
        fit["mpi_per_band"] = int(experiment.get("mpi_per_band", 2))
        if fit["max_parallel"] < 1 or fit["mpi_per_band"] < 1:
            raise ValueError("MICA process counts must be positive")
        fit["posterior_branches"] = {"enabled": False}
        if fit.get("ncomp") != 2 or fit.get("lag_prior") != [[0.0, 100.0], [0.0, 100.0]]:
            raise ValueError(f"Expected the agreed two-component 0–100 d prior: {stage}")
        cfg[stage] = fit
    for section in ("fflux", "fflux_comp0"):
        if section in cfg:
            cfg[section]["prepared_dir"] = "prepared"
            # data/original was already dereddened and host-subtracted upstream.
            cfg[section]["extinction"] = {"law": "F99", "ebv": 0.0, "rv": 3.1}
            cfg[section]["steps"] = ["prep", "dered", "fit", "summ", "fvg", "sedfit"]
            cfg[section].pop("host_sub", None)
    cfg["smoothing"] = {
        "method": "normalized_gaussian_at_observation_times",
        "sigma_days": float(spec["sigma_days"]),
        "smooth_driver": bool(spec["smooth_driver"]),
        "output": "highpass_residual_plus_original_mean",
        "flux_formula": "F_original - W @ F_original + mean(F_original)",
        "offset": "original_sample_mean_per_band",
        "error_mode": experiment.get("error_mode", "original"),
        "input_flux_state": "already_dereddened_and_host_subtracted",
        "noise_covariance_used_by_mica": "diagonal_only",
    }
    return cfg


def curve_bytes(data: np.ndarray) -> bytes:
    buffer = io.StringIO()
    np.savetxt(buffer, data, fmt="%.12e", header="time flux flux_error")
    return buffer.getvalue().encode()


def plot_inputs(case_dir: Path, curves: dict, outputs: dict, applied: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator

    with plt.style.context(ROOT / "src/image-creator.mplstyle"):
        fig, axes = plt.subplots(6, 1, figsize=(7.1, 8.8), sharex=True)
        try:
            for ax, band in zip(axes, BANDS):
                original, filtered = curves[band], outputs[band]
                t = original[:, 0] - 59000
                ax.errorbar(t, original[:, 1], yerr=original[:, 2], fmt="none",
                            ecolor="#B8B8B8", elinewidth=0.55, capsize=0,
                            alpha=0.55, zorder=1)
                ax.plot(t, original[:, 1], "o", ms=2.2, color="#222222", zorder=2)
                if applied[band]:
                    trend = original[:, 1] - filtered[:, 1] + np.mean(original[:, 1])
                    ax.plot(t, trend, color="#E69F00", lw=1.2, zorder=3)
                    ax.plot(t, filtered[:, 1], color="#1F4E9E", lw=1.2, zorder=4)
                ax.set_ylabel(band)
                ax.xaxis.set_minor_locator(AutoMinorLocator(4))
                ax.yaxis.set_minor_locator(AutoMinorLocator(4))
            axes[-1].set_xlabel("MJD − 59000 [day]")
            fig.supylabel("Flux [input units]", x=0.012, fontsize=10)
            fig.suptitle(f"Mrk 817 / {case_dir.name}", y=0.995)
            fig.legend(handles=[
                Line2D([], [], color="#222222", marker="o", ls="none", ms=3,
                       label="Original"),
                Line2D([], [], color="#E69F00", lw=1.2, label="Gaussian slow trend"),
                Line2D([], [], color="#1F4E9E", lw=1.2, label="Fast residual + original mean"),
            ], loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3, fontsize=8)
            fig.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.07, hspace=0.1)
            destination = case_dir / "figures"
            destination.mkdir(exist_ok=True)
            for extension in ("png", "pdf"):
                fig.savefig(destination / f"lightcurves.{extension}", dpi=300)
        finally:
            plt.close(fig)


def prepare(config_path: Path, selected: list[str] | None = None,
            output_dir: Path | None = None, make_plots: bool = True) -> list[Path]:
    config_path = config_path.resolve()
    experiment = yaml.safe_load(config_path.read_text())
    config_base = config_path.parent
    source_path = resolve(config_base, experiment["source_config"])
    base = yaml.safe_load(source_path.read_text())
    input_dir = resolve(config_base, experiment["input_dir"])
    root = output_dir.resolve() if output_dir else resolve(config_base, experiment["cases_dir"])
    definitions = experiment["cases"]
    names = list(dict.fromkeys(selected or definitions))
    unknown = set(names) - set(definitions)
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")
    curves = {band: read_curve(input_dir / f"Mrk817_a_{band}.txt") for band in BANDS}
    prepared = []
    for name in names:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid case name: {name}")
        spec = definitions[name]
        if not isinstance(spec.get("smooth_driver"), bool):
            raise ValueError(f"smooth_driver must be a boolean: {name}")
        cfg = case_config(base, spec, experiment)
        case_dir = root / name
        applied = {band: band != "UVW2" or spec["smooth_driver"] for band in BANDS}
        outputs, covariances, files = {}, {}, {}
        for band, original in curves.items():
            if applied[band]:
                curve, cov = detrend_curve(original, spec["sigma_days"], cfg["smoothing"]["error_mode"])
                files[f"prepared/Mrk817_a_{band}.txt"] = curve_bytes(curve)
            else:
                curve, cov = original.copy(), np.diag(original[:, 2]**2)
                files[f"prepared/Mrk817_a_{band}.txt"] = (input_dir / f"Mrk817_a_{band}.txt").read_bytes()
            outputs[band], covariances[band] = curve, cov
        files["config/source_config.yaml"] = yaml.safe_dump(cfg, sort_keys=False).encode()
        manifest = dict(
            case=name, source_config=str(source_path), source_inputs=str(input_dir),
            smoothing=cfg["smoothing"],
            detrended_bands=[band for band in BANDS if applied[band]],
            added_mean_by_band={band: float(np.mean(curves[band][:, 1]))
                                if applied[band] else 0.0 for band in BANDS},
            input_columns=["MJD", "flux", "flux_error"],
            weights="exp(-0.5*((ti-tj)/sigma_days)**2), normalized over observed j",
            edges="renormalize available observations; no padding or extrapolation",
            covariance="H @ diag(original_error**2) @ H.T, H=I-W+J, J_ij=1/N; saved in noise/",
            uncertainty_scope="Exploratory: MICA ignores smoothing-induced noise correlations.",
        )
        files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
        if case_dir.exists():
            for relative, content in files.items():
                path = case_dir / relative
                if not path.is_file() or path.read_bytes() != content:
                    raise FileExistsError(
                        f"Existing case differs or is incomplete: {path}; use a new --output-dir"
                    )
            print(f"Reuse identical inputs: {case_dir}", flush=True)
        else:
            for relative, content in files.items():
                path = case_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            print(f"Prepared: {case_dir}", flush=True)
        noise_dir = case_dir / "noise"
        noise_dir.mkdir(exist_ok=True)
        for band in BANDS:
            np.savez_compressed(noise_dir / f"{band}.npz", time=curves[band][:, 0],
                                covariance=covariances[band],
                                propagated_error=np.sqrt(np.diag(covariances[band])),
                                mica_error=outputs[band][:, 2])
        with (case_dir / "input_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "band", "detrended", "added_mean", "sigma_days", "n_points", "mjd_min", "mjd_max",
                "rms_original", "rms_input", "median_mica_error", "median_propagated_error"])
            writer.writeheader()
            for band in BANDS:
                original, curve = curves[band], outputs[band]
                writer.writerow(dict(
                    band=band, detrended=applied[band],
                    added_mean=float(np.mean(original[:, 1])) if applied[band] else 0.0,
                    sigma_days=spec["sigma_days"] if applied[band] else 0,
                    n_points=len(curve), mjd_min=curve[0, 0], mjd_max=curve[-1, 0],
                    rms_original=float(np.std(original[:, 1], ddof=1)),
                    rms_input=float(np.std(curve[:, 1], ddof=1)),
                    median_mica_error=float(np.median(curve[:, 2])),
                    median_propagated_error=float(np.median(np.sqrt(np.diag(covariances[band])))),
                ))
        if make_plots:
            plot_inputs(case_dir, curves, outputs, applied)
        prepared.append(case_dir)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "experiment.yaml")
    parser.add_argument("--case", action="append", help="one case; repeat to select several")
    parser.add_argument("--output-dir", type=Path, help="alternative cases directory")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    prepare(args.config, args.case, args.output_dir, not args.no_plots)


if __name__ == "__main__":
    main()
