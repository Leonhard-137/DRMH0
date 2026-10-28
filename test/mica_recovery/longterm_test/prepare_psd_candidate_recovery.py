#!/usr/bin/env python3
"""Prepare saved PSD-matched realizations for MICA; never launch a fit.

Import the six deliberately selected W2/U light curves exactly as saved. No
random realization is generated or replayed, and no RMS normalization is
performed. Restore observed flux units with the original per-band sample
means, then form raw and BOTH-band sigma=5/10/20/40 high-pass inputs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from astropy.timeseries import LombScargle
import yaml

from fit import build_plan, source_config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUT = HERE / "cases_psd_candidates"
EXAMPLES = HERE / "psd_match_817/local_peak_check/examples"
SOURCE_SUMMARY = EXAMPLES.parent / "summary.json"
ORIGINAL = ROOT / "test/mica_smooth/data/original"
CANDIDATES = ("C045", "C001", "C078", "C075", "C089", "C069")
BANDS = ("UVW2", "U")
PARAMETERS = ("tauI", "tauS", "AW", "AU", "BW", "BU")
VARIANTS = (("raw", None), *((f"all_s{s}", s) for s in (5, 10, 20, 40)))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def binned_psd(times, fractional):
    """Original 23-bin, unweighted floating-mean LS density; no interpolation."""
    times = np.asarray(times, dtype=float)
    span = float(np.ptp(times))
    frequency = np.arange(1/span, .3, 1/(5*span))
    power = LombScargle(times-times[0], fractional, fit_mean=True, normalization="psd").power(
        frequency, method="cython") * (2*span/len(times))
    edges = np.geomspace(frequency[0], frequency[-1]*(1+1e-10), 24)
    centers, values = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (frequency >= low) & (frequency < high)
        if not np.any(selected):
            raise ValueError("Empty periodogram bin")
        centers.append(float(np.exp(np.log(frequency[selected]).mean())))
        values.append(float(power[selected].mean()))
    return np.asarray(centers), np.asarray(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    candidates = list(dict.fromkeys(args.candidates))
    output = args.output_dir.resolve()
    config_path = output / "experiment.yaml"
    names = [f"{candidate}_{variant}" for candidate in candidates for variant, _ in VARIANTS]
    protected = [config_path, output/"manifest.json", output/"validation.json", output/"README.md"]
    if any(path.exists() for path in protected) or any((output/name).exists() for name in names):
        raise FileExistsError("Preserving existing candidate inputs, configuration, and results")

    summary = json.loads(SOURCE_SUMMARY.read_text())
    catalog = {row["candidate_id"]: row for row in summary["confirmation"]}
    originals = {b: np.loadtxt(ORIGINAL/f"Mrk817_a_{b}.txt") for b in BANDS}
    means = {b: float(originals[b][:, 1].mean()) for b in BANDS}
    spec = importlib.util.spec_from_file_location("mica_smooth_prepare", ROOT/"test/mica_smooth/prepare.py")
    smooth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smooth)

    historical = yaml.safe_load((HERE/"experiment_tau2_area9_shared.yaml").read_text())
    cfg = dict(
        experiment=dict(output_dir=".", simulation_source="LTSIM"),
        observations=dict(input_dir=str(ORIGINAL), input_source="Mrk817_a", driver_band="UVW2", response_band="U",
                          time_zero="original_MJD", flux_units="original observed band flux units",
                          amplitude_calibration="saved fractional realization times fixed observed sample mean",
                          noise="saved realization with original observed errors"),
        paired_ablation=dict(cases={name: dict(blr=True, shared_driver=True, shared_response=True) for name in names}),
        mica_fit=deepcopy(historical["mica_fit"]),
        generation=dict(inference_only=True, manifest=str(output/"manifest.json"),
                        preparation_script=str(Path(__file__).resolve()), source_summary=str(SOURCE_SUMMARY),
                        no_new_random_draw=True, no_rms_rescaling=True),
        detrending=dict(bands=list(BANDS), sigma_days=[5, 10, 20, 40], error_mode="original", inference_only=True,
                        formula="F_raw - normalized Gaussian trend(F_raw) + mean(F_raw)"),
        execution=dict(max_parallel_fits=8, mpi_per_fit=2, model_order=["gamma", "gaussian"], serial_postfit_figures=True),
    )
    native_config = source_config(cfg)
    for model in ("gamma", "gaussian"):
        native = native_config[f"mica_{model}2"]
        assert native["ncomp"] == 2 and native["max_num_saves"] == 10000 and native["ptol"] == .01
        assert native["mpi_per_band"] == 2
    assert cfg["mica_fit"] == historical["mica_fit"]
    kernel_source = json.loads((HERE/"psd_match_817/core_check.json").read_text())["metadata"]
    kernel_truth = dict(parameters=deepcopy(kernel_source["kernel_parameters"]),
        disk_normalization=kernel_source["disk_area_normalization"],
        disk_quadrature_areas_before_normalization=kernel_source["disk_quadrature_areas_before_normalization"],
        response_disk_fraction=.9, response_blr_fraction=.1, response_area_ratio=9.0,
        blr_shape=2, blr_onset_days=6.0, blr_scale_days=7.0, blr_centroid_days=20.0,
        blr_physical_std_days=float(np.sqrt(2)*7),
        reference="Physical I-to-W2/I-to-U kernels; not an injected single W2-to-U effective kernel")

    prepared, candidate_records, case_records, jobs = {}, {}, {}, []
    worst_raw_psd, worst_filtered_psd, worst_formula, worst_roundtrip = 0., 0., 0., 0.
    for candidate in candidates:
        record = catalog[candidate]
        curves_path, spectrum_path = EXAMPLES/f"{candidate}_curves.npz", EXAMPLES/f"{candidate}.npz"
        with np.load(curves_path) as handle:
            saved = {key: handle[key].copy() for key in handle.files}
        with np.load(spectrum_path) as handle:
            spectral = {key: handle[key].copy() for key in handle.files}
        parameters = {key: float(value) for key, value in zip(PARAMETERS, spectral["params"])}
        np.testing.assert_array_equal(spectral["params"], [record[key] for key in PARAMETERS])
        assert int(spectral["example_index0"]) == int(record["example_index0"])
        metadata = dict(candidate_id=candidate, role=record["role"], parameters=parameters,
            seed=int(spectral["seed"]), example_index0=int(spectral["example_index0"]),
            source_ensemble_size=int(spectral["nreal"]), source_curves=str(curves_path),
            source_psd=str(spectrum_path), observed_means=means,
            fractional_model="w=AW*(DW*I)+BW*S+nW; u=AU*(0.9*DU*I+0.1*B*I)+BU*S+nU",
            physical_model="F_W=mean_W*(1+w); F_U=mean_U*(1+u)",
            latent_processes=dict(I="unit stationary-variance OU", S="unit stationary-variance Matern3/2; covariance (1+|h|/tauS)*exp(-|h|/tauS)",
                                  shared_between_bands=True, I_and_S_independent=True),
            normalization="Import saved fractional curves; restore fixed observed band means; no new draw, seed change, RMS rescaling, clipping or added baseline",
            selected_realization_caveat="Deliberately selected low-mismatch joint local-peak example, not a typical or unbiased realization",
            kernel_truth=kernel_truth,
            truth_time_series_available=False,
            truth_time_series_note="Saved NPZ contains total noisy curves only; no component values are fabricated or replayed")
        candidate_records[candidate] = metadata
        raw_curves, fractional = {}, {}
        raw_checks = {}
        for bi, b in enumerate(BANDS):
            obs = originals[b]
            t, y, err = (saved[f"{b}_{key}"] for key in ("time", "fractional", "error"))
            np.testing.assert_array_equal(t, obs[:, 0])
            np.testing.assert_array_equal(err, obs[:, 2]/means[b])
            assert y.shape == (len(obs),) and np.isfinite(y).all()
            curve = np.column_stack([t, means[b]*(1+y), obs[:, 2]])
            restored = curve[:, 1]/means[b]-1
            roundtrip = float(np.max(np.abs(restored-y)))
            np.testing.assert_allclose(restored, y, rtol=0, atol=4e-16)
            frequency, psd = binned_psd(t, restored)
            np.testing.assert_allclose(frequency, spectral["frequency"][bi], rtol=2e-12, atol=1e-15)
            relative = float(np.max(np.abs(psd/spectral["example"][bi]-1)))
            if relative > 1e-9:
                raise ValueError(f"{candidate}/{b}: saved raw PSD mismatch {relative}")
            raw_checks[b] = dict(raw_psd_max_relative_error=relative, fractional_roundtrip_max_abs_error=roundtrip,
                                 samples=len(t), mean_flux=float(curve[:, 1].mean()), original_mean=means[b])
            worst_raw_psd, worst_roundtrip = max(worst_raw_psd, relative), max(worst_roundtrip, roundtrip)
            raw_curves[b], fractional[b] = curve, y
        metadata["raw_validation"] = raw_checks

        for variant, sigma in VARIANTS:
            name, case_dir = f"{candidate}_{variant}", output/f"{candidate}_{variant}"
            arrays, covariances, trends, validations, psds, frequencies = {}, {}, {}, {}, [], []
            saved_filtered = None
            if sigma in (20, 40):
                with np.load(EXAMPLES/f"{candidate}_s{sigma}.npz") as handle:
                    saved_filtered = {key: handle[key].copy() for key in handle.files}
                assert int(saved_filtered["seed"]) == metadata["seed"]
                assert int(saved_filtered["example_index0"]) == metadata["example_index0"]
                np.testing.assert_array_equal(saved_filtered["params"], spectral["params"])
            for bi, b in enumerate(BANDS):
                raw = raw_curves[b]
                if sigma is None:
                    curve = raw.copy()
                    formula_error, mean_error, processed = 0., 0., False
                else:
                    weights = smooth.gaussian_weights(raw[:, 0], sigma)
                    curve, covariances[b] = smooth.detrend_curve(raw, sigma, "original")
                    trend = weights @ raw[:, 1]
                    trends[b] = np.column_stack([raw[:, 0], trend])
                    exact = raw[:, 1]-trend+raw[:, 1].mean()
                    formula_error = float(np.max(np.abs(curve[:, 1]-exact)))
                    expected_fractional = fractional[b]-weights@fractional[b]+fractional[b].mean()
                    np.testing.assert_allclose(curve[:, 1], means[b]*(1+expected_fractional), rtol=1e-13, atol=1e-13)
                    # Row-normalized W need not preserve the unweighted sample
                    # mean. Validate the exact restored-mean formula, not a
                    # scientifically incorrect equality of before/after means.
                    expected_mean = 2*raw[:, 1].mean()-trend.mean()
                    mean_error = float(abs(curve[:, 1].mean()-expected_mean))
                    processed = True
                np.testing.assert_array_equal(curve[:, [0, 2]], raw[:, [0, 2]])
                if not np.isfinite(curve).all() or np.any(curve[:, 1] <= 0):
                    raise ValueError(f"{name}/{b}: nonfinite/nonpositive flux; no clipping or baseline change permitted")
                frequency, psd = binned_psd(curve[:, 0], curve[:, 1]/means[b]-1)
                saved_error = None
                if saved_filtered is not None:
                    np.testing.assert_allclose(frequency, saved_filtered["frequency"][bi], rtol=2e-12, atol=1e-15)
                    saved_error = float(np.max(np.abs(psd/saved_filtered["example"][bi]-1)))
                    if saved_error > 1e-9:
                        raise ValueError(f"{name}/{b}: saved filtered PSD mismatch {saved_error}")
                    worst_filtered_psd = max(worst_filtered_psd, saved_error)
                worst_formula = max(worst_formula, formula_error, mean_error)
                validations[b] = dict(processed=processed, epochs_unchanged=True, quoted_errors_unchanged=True,
                    input_positive=True, minimum_flux=float(curve[:, 1].min()),
                    formula_max_abs_error=formula_error, restored_mean_formula_abs_error=mean_error,
                    original_mean=float(raw[:, 1].mean()), output_mean=float(curve[:, 1].mean()),
                    saved_filtered_psd_max_relative_error=saved_error,
                    variability_rms=float(np.std(curve[:, 1])), fractional_rms=float(np.std(curve[:, 1]/means[b])))
                arrays[b] = curve
                frequencies.append(frequency)
                psds.append(psd)
            case_meta = dict(candidate_id=candidate, variant=variant, sigma_days=sigma, case=name, case_dir=str(case_dir),
                original_case=str(output/f"{candidate}_raw"), filtered_bands=[] if sigma is None else list(BANDS),
                input_formula="F=original_observed_mean*(1+saved_fractional)" if sigma is None else cfg["detrending"]["formula"],
                quoted_errors="original observed absolute errors", full_noise_covariance="saved after filtering; MICA consumes original diagonal errors",
                validation=validations)
            case_records[name] = case_meta
            prepared[name] = (arrays, covariances, trends, np.asarray(frequencies), np.asarray(psds))
            for model in ("gamma", "gaussian"):
                run_dir = case_dir/f"runs/mica/2comp/run_UVW2_to_U_2comp_{model}"
                jobs.append(dict(case=name, case_dir=str(case_dir), candidate_id=candidate, variant=variant,
                    model=model, stage=f"mica_{model}2", run_dir=str(run_dir),
                    state_file=str(run_dir/"run_mica_state.json"),
                    expected_results=str(case_dir/"results"/model)))

    validation = dict(status="pass", passed=True, scope="Input-only validation; no MICA launched", candidate_count=len(candidates),
        case_count=len(names), job_count=len(jobs), raw_psd_max_relative_error=worst_raw_psd,
        sigma20_40_saved_psd_max_relative_error=worst_filtered_psd,
        fractional_roundtrip_max_abs_error=worst_roundtrip, highpass_formula_max_abs_error=worst_formula,
        epochs_and_errors_unchanged=True, all_flux_positive=True, no_new_random_draw=True,
        no_rms_rescaling=True, no_clipping_or_baseline_adjustment=True,
        both_bands_filtered_in_all_sigma_cases=True, native_ncomp=2,
        prior_and_sampling_equal_to=str(HERE/"experiment_tau2_area9_shared.yaml"),
        max_num_saves=10000, ptol=.01, mpi_per_fit=2, max_parallel_fits=8,
        filter_mean_note="Each output adds the original sample mean after trend subtraction; irregular row-normalized weights can change the output sample mean; validated against exact formula")
    manifest = dict(schema=1, config=str(config_path), source_summary=str(SOURCE_SUMMARY),
        candidate_order=candidates, case_order=names, candidates=candidate_records,
        cases=case_records, jobs=jobs, validation=validation, launch_authorized_by_this_script=False)

    # Complete all scientific input checks before the first filesystem write.
    output.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# Inference-only import of saved selected PSD realizations.\n"+yaml.safe_dump(cfg, sort_keys=False))
    for name, (arrays, covariance, trends, frequency, psd) in prepared.items():
        case_dir = output/name
        for folder in ("input", "config", "diagnostics", "results"):
            (case_dir/folder).mkdir(parents=True, exist_ok=True)
        for b in BANDS:
            path = case_dir/f"input/LTSIM_{b}.txt"
            np.savetxt(path, arrays[b], fmt="%.17g", header="MJD flux flux_error; original observed band flux units")
            np.testing.assert_array_equal(np.loadtxt(path), arrays[b])
            if b in covariance:
                np.save(case_dir/f"input/{b}_filtered_noise_covariance.npy", covariance[b])
                np.savetxt(case_dir/f"input/{b}_removed_trend.txt", trends[b], fmt="%.17g", header="MJD Gaussian_trend")
        generation = deepcopy(candidate_records[case_records[name]["candidate_id"]])
        generation.update(sigma_days=case_records[name]["sigma_days"], filtered_bands=case_records[name]["filtered_bands"],
                          input_units="original observed flux units", input_formula=case_records[name]["input_formula"])
        write_json(case_dir/"config/generation.json", generation)
        write_json(case_dir/"provenance.json", case_records[name])
        (case_dir/"config/source_config.yaml").write_text("# Same native inference settings as prior two-component tests.\n"+
                                                         yaml.safe_dump(native_config, sort_keys=False))
        np.savez_compressed(case_dir/"diagnostics/input_psd.npz", frequency=frequency, psd=psd, bands=np.asarray(BANDS))
    plan = build_plan(config_path, suite="paired_ablation", model="both")
    if len(plan) != len(jobs):
        raise AssertionError("Native MICA plan count differs from manifest")
    expected_runs = {job["run_dir"] for job in jobs}
    assert {str(job["run_dir"]) for job in plan} == expected_runs
    validation["native_plan_validated"] = True
    validation["saved_text_roundtrip_exact"] = True
    write_json(output/"validation.json", validation)
    write_json(output/"manifest.json", manifest)
    (output/"README.md").write_text(
        "# Recovery fits for six saved PSD candidate realizations\n\n"
        "Each selected candidate contributes raw and BOTH-band sigma5/10/20/40 Gaussian high-pass inputs. "
        "These are the exact previously selected single realizations, imported without redrawing or RMS rescaling. "
        "Absolute flux is the saved fractional variation plus one, times the original observed band mean. "
        "The original MJD epochs and absolute measurement errors are retained.\n\n"
        "The raw candidate spectra are checked against saved examples. Sigma20/40 are additionally checked "
        "against their existing filtered spectra. Every filter is F-W@F+mean(F), applied independently to both "
        "actual band grids; row normalization need not preserve the output arithmetic mean. Full propagated "
        "noise covariance is saved, but native MICA continues to use original diagonal errors.\n\n"
        "Physical kernel truth and source seed/index are retained in generation.json. The selected NPZs only "
        "contain total noisy curves; component light curves are not fabricated. Hence existing plot_results.py "
        "produces inferred transfer functions, corner and light-curve plots without a false noiseless-truth overlay.\n\n"
        f"Candidates: {', '.join(candidates)}. Cases: {len(names)}. Fits: {len(jobs)}. Each fit has two components, "
        "max_num_saves=10000, ptol=0.01, two MPI ranks, original priors. The separate controller is limited to "
        "eight concurrent fits, prioritizes Gamma, and serializes post-fit plots. This preparation script launches "
        "no sampler. Existing inputs/results are never overwritten.\n")
    print(json.dumps(dict(config=str(config_path), manifest=str(output/"manifest.json"), validation=validation), indent=2))


if __name__ == "__main__":
    main()
