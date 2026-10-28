#!/usr/bin/env python3
"""Measure fixed-posterior MICA sensitivity to target-only signal vectors.

Uses three existing B-band joint posteriors and native MICA covariances. All
parameters, normalization, errors, and observing times remain fixed within a
draw. This is a linear-operator diagnostic, not a refit or new random simulation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGES = HERE / "outputs/component_lightcurves/reconstruction"
OUT = HERE / "outputs/independent_influence"
FIT_CASES = ["disk_only", "random_within", "random_beyond"]
PERIODS = [2.0, 5.0, 10.0, 30.0, 100.0, 300.0]
sys.path.insert(0, str(ROOT / "src"))


class NativeLC(ctypes.Structure):
    """MICA2-astro-e5c622b/src/allvars.h::LC, verified against native values."""
    _fields_ = [("n", ctypes.c_int), ("scale", ctypes.c_double),
                ("error_mean", ctypes.c_double),
                ("t", ctypes.POINTER(ctypes.c_double)),
                ("f", ctypes.POINTER(ctypes.c_double)),
                ("fe", ctypes.POINTER(ctypes.c_double))]


def centered(values: np.ndarray) -> np.ndarray:
    return values - np.mean(values, axis=0, keepdims=True)


def make_probes(times: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    vectors, metadata = [], []
    for case in ("random_within", "random_beyond"):
        info = json.loads((STAGES / case / "B/provenance.json").read_text())
        truth_path = Path(info["case_dir"]) / "truth_at_B_observations.csv"
        truth = np.genfromtxt(truth_path, delimiter=",", names=True)
        if not np.allclose(times, truth["observer_day"], atol=1e-7, rtol=0):
            raise ValueError("Probe and fit observing times differ")
        vectors.append(centered(truth["slow_variation"]))
        metadata.append({"probe": case + "_u", "kind": "known_independent_injection",
                         "period_days": None, "phase": None, "source": str(truth_path)})
    reference_rms = float(np.sqrt(np.mean(vectors[0] ** 2)))
    for period in PERIODS:
        phase = 2 * np.pi * (times - times.min()) / period
        for phase_name, values in (("sin", np.sin(phase)), ("cos", np.cos(phase))):
            values = centered(values)
            values *= reference_rms / np.sqrt(np.mean(values ** 2))
            vectors.append(values)
            metadata.append({"probe": f"{phase_name}_{period:g}d", "kind": "deterministic_operator_probe",
                             "period_days": period, "phase": phase_name, "source": None})
    return np.column_stack(vectors), metadata


def metrics(signal: np.ndarray, predicted: np.ndarray) -> dict:
    denominator = float(np.dot(signal, signal))
    norm = float(np.sqrt(np.mean(signal ** 2)))
    predicted_norm = float(np.sqrt(np.mean(predicted ** 2)))
    correlation = float(np.corrcoef(signal, predicted)[0, 1]) if predicted_norm > 1e-15 else None
    return {"beta": float(np.dot(predicted, signal) / denominator),
            "r": correlation, "rms_over_u": predicted_norm / norm}


def worker(fit_case: str) -> dict:
    from mpi4py import MPI
    import pymica
    from scipy.linalg import lu_factor, lu_solve
    from collect_mica_results import read_yaml
    from decompose_mica_branches import _native_mica_library, _set_native_string
    from run_mica import fit_config, setup_kwargs
    from reconstruct_component_means import flux_scales

    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Use exactly one MPI rank per diagnostic")
    started = time.monotonic()
    stage = STAGES / fit_case / "B"
    info = json.loads((stage / "provenance.json").read_text())
    destination = OUT / fit_case
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "summary.json").exists():
        raise FileExistsError("Preserve existing diagnostic results")
    posterior = np.loadtxt(stage / "data/posterior_sample1d.txt_2", ndmin=2)
    if posterior.shape[1] != 10 or not np.all(posterior[:, 5] <= posterior[:, 8]):
        raise ValueError("Expected complete centroid-ordered Gamma posterior")
    if not np.isfinite(posterior).all() or not np.all(posterior[:, [0, 3]] == 0):
        raise ValueError("Invalid posterior or unexpected free systematic errors")
    config = read_yaml(Path(info["case_dir"]) / "config/source_config.yaml")
    fit = fit_config(config, stage="mica_recovery")
    if fit.get("flag_trend", 0) != 0 or fit.get("flag_negative_resp", False):
        raise ValueError("Only constant baseline, positive-response fits are supported")
    data = [np.loadtxt(stage / "input" / name, ndmin=2) for name in ("cont.txt", "line.txt")]
    observation_t = np.ascontiguousarray(data[1][:, 0])
    probes, probe_metadata = make_probes(observation_t)
    ncon, ntarget = len(data[0]), len(data[1])
    ndata, nprobe = ncon + ntarget, probes.shape[1]
    scratch = Path(tempfile.mkdtemp(prefix="native_", dir=destination))
    previous_cwd = Path.cwd()
    initialized = line_initialized = False
    library = None
    try:
        os.chdir(scratch)
        model = pymica.gmodel()
        kwargs = setup_kwargs(fit, {"set1": data}, "B")
        model.setup(**kwargs)
        library = _native_mica_library(pymica)
        library.set_argv.argtypes = [ctypes.c_int] * 5
        library.dnest_line.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        library.dnest_line.restype = ctypes.c_double
        library.set_covar_Pmat_data_line.argtypes = [ctypes.c_void_p, ctypes.c_int]
        library.set_covar_Pmat_data_line.restype = None
        library.set_covar_Umat_line_single.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        library.set_covar_Umat_line_single.restype = None
        library.decompose_single_component.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        library.decompose_single_component.restype = None
        library.set_argv(0, 0, 0, 1, 0)
        if library.read_data() != 0:
            raise RuntimeError("MICA read_data failed")
        library.init()
        initialized = True
        if library.mc_line_init() != 0:
            raise RuntimeError("MICA mc_line_init failed")
        line_initialized = True
        ctypes.c_int.in_dll(library, "num_gaussian").value = 2
        ctypes.c_int.in_dll(library, "type_lag_prior_pr").value = 1
        _set_native_string(library, "postfix", "_2", 256)
        library.dnest_line(0, None)
        if ctypes.c_int.in_dll(library, "num_params").value != 10:
            raise ValueError("Native parameter layout mismatch")
        serialized = np.loadtxt(scratch / "data/data_input.txt", ndmin=2)
        curves = [serialized[:ncon], serialized[ncon:]]
        scales = flux_scales(curves)
        y = np.r_[curves[0][:, 1] / scales[0], curves[1][:, 1] / scales[1]]
        design = np.zeros((ndata, 2))
        design[:ncon, 0] = 1
        design[ncon:, 1] = 1
        injected = np.zeros((ndata, nprobe))
        injected[ncon:] = probes / scales[1]
        # Extra RHS columns validate full minus full-u for both actual u vectors.
        rhs = np.column_stack((injected, y, y - injected[:, 0], y - injected[:, 1], design))
        counts = np.array([0, ntarget], dtype=np.int32)
        counts_pointer = counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        times_pointer = observation_t.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        native_lc = ctypes.POINTER(NativeLC).in_dll(library, "alldata")[0]
        if native_lc.n != ndata:
            raise ValueError("Native LC layout differs from expected data count")
        native_y = np.ctypeslib.as_array(native_lc.f, shape=(ndata,))
        if not np.allclose(native_y, y, atol=2e-14, rtol=2e-14):
            raise ValueError("Native normalized data disagree with conditional-mean RHS")
        original_native_y = native_y.copy()
        covariance = np.ctypeslib.as_array(
            ctypes.POINTER(ctypes.c_double).in_dll(library, "PCmat"),
            shape=(ndata * ndata,)).reshape(ndata, ndata)
        cross = np.ctypeslib.as_array(
            ctypes.POINTER(ctypes.c_double).in_dll(library, "USmat"),
            shape=(ntarget * ndata,)).reshape(ntarget, ndata)
        mean_accum = np.zeros((2, ntarget, nprobe))
        full_accum = np.zeros((2, ntarget))
        betas = np.zeros((len(posterior), 2, nprobe))
        probe_norm2 = np.sum(probes ** 2, axis=0)
        linear_validation = []
        for draw, pm in enumerate(np.ascontiguousarray(posterior)):
            pm_pointer = pm.ctypes.data_as(ctypes.c_void_p)
            library.set_covar_Pmat_data_line(pm_pointer, 0)
            factor = lu_factor(covariance, check_finite=False)
            inverse_rhs = lu_solve(factor, rhs, check_finite=False)
            inverse_design = inverse_rhs[:, -2:]
            inverse_values = inverse_rhs[:, :-2]
            baseline = np.linalg.solve(design.T @ inverse_design, design.T @ inverse_values)
            alpha = inverse_values - inverse_design @ baseline
            areas = np.exp(pm[[4, 7]])
            area_weights = areas / areas.sum()
            for component in range(2):
                library.set_covar_Umat_line_single(pm_pointer, 0, counts_pointer, times_pointer, component)
                predicted = (cross @ alpha + area_weights[component] * baseline[1]) * scales[1]
                predicted = centered(predicted)
                if not np.isfinite(predicted).all():
                    raise ValueError(f"Nonfinite prediction in {fit_case}, draw {draw}")
                delta = predicted[:, :nprobe]
                mean_accum[component] += delta
                full_accum[component] += predicted[:, nprobe]
                betas[draw, component] = np.sum(delta * probes, axis=0) / probe_norm2
                if draw == 0:
                    # Independent native routine, with only the in-memory data
                    # vector changed. Neither C, normalization, nor theta changes.
                    native_results = []
                    try:
                        for vector in (original_native_y, original_native_y - injected[:, 0],
                                       original_native_y - injected[:, 1]):
                            native_y[:] = vector
                            fall, ferr, q = np.empty(ntarget), np.empty(ntarget), np.empty(2)
                            library.decompose_single_component(
                                pm_pointer, 0, counts_pointer, times_pointer,
                                fall.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                ferr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), component)
                            native_results.append(centered(fall * scales[1]))
                    finally:
                        native_y[:] = original_native_y
                    for p in range(2):
                        native_difference = native_results[0] - native_results[p + 1]
                        native_error = float(np.max(np.abs(native_difference - delta[:, p])))
                        rhs_error = float(np.max(np.abs(predicted[:, nprobe] - predicted[:, nprobe + p + 1] - delta[:, p])))
                        passed = native_error < 2e-9 and rhs_error < 2e-9
                        linear_validation.append({"draw": 0, "component": component,
                                                  "probe": probe_metadata[p]["probe"],
                                                  "native_full_minus_removed_max_abs_diff": native_error,
                                                  "same_rhs_max_abs_diff": rhs_error, "pass": passed})
                        if not passed:
                            raise ValueError(f"Native linearity validation failed: {linear_validation[-1]}")
            if draw == 0 or (draw + 1) % 100 == 0 or draw + 1 == len(posterior):
                print(f"INFLUENCE {fit_case} {draw + 1}/{len(posterior)} "
                      f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        mean_delta = mean_accum / len(posterior)
        full_mean = full_accum / len(posterior)
        existing = np.load(stage / "means.npz")
        positions = np.searchsorted(existing["t"], observation_t)
        if not np.array_equal(existing["t"][positions], observation_t):
            raise ValueError("Prior validated mean grid lacks exact observing times")
        full_validation = {}
        for component, name in enumerate(("short", "long")):
            reference = centered(existing[name][positions])
            difference = float(np.max(np.abs(full_mean[component] - reference)))
            full_validation[name + "_max_abs_diff"] = difference
            if difference > 2e-9:
                raise ValueError(f"Full posterior mean changed in influence calculation: {name} {difference}")
        rows = []
        for p, probe in enumerate(probe_metadata):
            signal = probes[:, p]
            short, long = mean_delta[0, :, p], mean_delta[1, :, p]
            total = short + long
            row = {"fit_case": fit_case, **probe,
                   "probe_rms": float(np.sqrt(np.mean(signal ** 2))),
                   "posterior_rows": len(posterior), "kish_ess": info["kish_ess"],
                   "residual_rms_over_u": float(np.linalg.norm(signal - total) / np.linalg.norm(signal))}
            for name, prediction in (("short", short), ("long", long), ("total", total)):
                for key, value in metrics(signal, prediction).items():
                    row[key + "_" + name] = value
            for component, name in enumerate(("short", "long")):
                q16, q50, q84 = np.quantile(betas[:, component, p], [0.16, 0.5, 0.84])
                row.update({"beta_" + name + "_q16": float(q16),
                            "beta_" + name + "_q50": float(q50),
                            "beta_" + name + "_q84": float(q84)})
            rows.append(row)
        summary = {"fit_case": fit_case, "band": "B", "source_provenance": info,
                   "normalized_data_scales": scales.tolist(),
                   "theta_held_fixed_within_draw": True, "posterior_reused_without_refitting": True,
                   "linear_validation": linear_validation, "full_mean_validation": full_validation,
                   "runtime_seconds": time.monotonic() - started,
                   "rows": rows,
                   "notes": [
                       "beta is an amplitude projection coefficient, not a variance or power fraction; it need not lie in [0,1].",
                       "The operator is applied to target-only vectors after removal of a constant at the observed B timestamps.",
                       "Posterior beta quantiles describe sensitivity conditional on the existing posterior, not the posterior after removing or injecting u.",
                       "Sin/cos inputs are deterministic operator probes at actual sampling, not new random simulations or inferred periodic components.",
                       "eta was not computed: native set_covar_Amat_line_single uses Sll_single for off-diagonals and Sll_single2 on its diagonal, so its full matrix was not assumed to be Kkk.",
                       "Only covariance-to-mean mappings are used for conclusions; no new sampling or native-source changes occurred.",
                   ]}
        np.savez_compressed(destination / "influence_curves.npz", t=observation_t, u=probes,
                            delta_short=mean_delta[0], delta_long=mean_delta[1],
                            delta_total=mean_delta.sum(axis=0), beta_draws=betas,
                            probe_names=np.asarray([probe["probe"] for probe in probe_metadata]))
        with (destination / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"fit_case": fit_case, "complete": True,
                          "runtime_seconds": summary["runtime_seconds"], "rows": rows[:2]}), flush=True)
        return summary
    finally:
        if line_initialized:
            library.mc_line_end()
        if initialized:
            library.end_run()
        os.chdir(previous_cwd)


def run_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    def launch(case: str) -> dict:
        if not (OUT / case / "summary.json").exists():
            command = [str(Path(sys.executable).parent / "mpiexec"), "-n", "1", sys.executable,
                       "-B", str(Path(__file__).resolve()), "--worker", case]
            with (OUT / f"{case}.log").open("x") as log:
                subprocess.run(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads((OUT / case / "summary.json").read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:
        summaries = list(pool.map(launch, FIT_CASES))
    rows = [row for summary in summaries for row in summary["rows"]]
    with (OUT / "all_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "all_results.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": True, "fit_cases": FIT_CASES, "n_diagnostics": len(rows)}))


def paired_effects() -> dict:
    """Compare the already fitted short curves; no native calls or refits."""
    def observed_short(case: str) -> tuple[np.ndarray, np.ndarray, dict]:
        stage = STAGES / case / "B"
        info = json.loads((stage / "provenance.json").read_text())
        times = np.loadtxt(stage / "input/line.txt", ndmin=2)[:, 0]
        with np.load(stage / "means.npz") as means:
            indices = np.searchsorted(means["t"], times)
            if not np.array_equal(means["t"][indices], times):
                raise ValueError("Stored means lack the exact paired observing times")
            values = centered(means["short"][indices])
        return times, values, info

    times0, short0, pure_info = observed_short("disk_only")
    pure_truth = np.genfromtxt(Path(pure_info["case_dir"]) / "truth_at_B_observations.csv",
                              delimiter=",", names=True)
    disk = centered(pure_truth["disk_variation"])
    disk_rms = float(np.sqrt(np.mean(disk ** 2)))
    rows = []
    for case in ("random_within", "random_beyond"):
        times, fitted_short, info = observed_short(case)
        if not np.array_equal(times, times0):
            raise ValueError("The paired fits do not share identical observing times")
        truth = np.genfromtxt(Path(info["case_dir"]) / "truth_at_B_observations.csv",
                             delimiter=",", names=True)
        if not np.allclose(truth["disk_variation"], pure_truth["disk_variation"], atol=1e-12, rtol=0):
            raise ValueError("Paired fits do not share the same disk realization")
        if not np.allclose(truth["noise"], pure_truth["noise"], atol=1e-12, rtol=0):
            raise ValueError("Paired fits do not share the same noise realization")
        u = centered(truth["slow_variation"])
        u_rms = float(np.sqrt(np.mean(u ** 2)))
        with np.load(OUT / case / "influence_curves.npz") as influence:
            if not np.array_equal(influence["t"], times):
                raise ValueError("Fixed-posterior influence observing times differ")
            probe_id = list(influence["probe_names"]).index(case + "_u")
            if not np.allclose(influence["u"][:, probe_id], u, atol=1e-14, rtol=0):
                raise ValueError("Fixed-posterior influence uses another u vector")
            direct = centered(influence["delta_short"][:, probe_id])
        total = fitted_short - short0
        remainder = total - direct
        row = {
            "fit_case": case, "band": "B", "observation_rows": len(times),
            "pure_posterior_rows": pure_info["posterior_rows"],
            "mixed_posterior_rows": info["posterior_rows"],
            "truth_disk_rms": disk_rms, "truth_u_rms": u_rms,
            "paired_observed_difference_minus_u_max_abs": float(np.max(np.abs(
                truth["observed_flux"] - pure_truth["observed_flux"] - truth["slow_variation"]))),
        }
        for name, vector in (("paired_end_to_end", total), ("fixed_posterior_direct", direct),
                             ("fitted_model_change_remainder", remainder)):
            values = metrics(u, vector)
            rms = float(np.sqrt(np.mean(vector ** 2)))
            row[name] = {"beta_vs_u": values["beta"], "r_vs_u": values["r"],
                         "rms": rms, "rms_over_u": rms / u_rms,
                         "rms_over_disk": rms / disk_rms}
        rows.append(row)
    result = {
        "definition": {
            "paired_end_to_end": "center(short_mixed_fit) - center(short_disk_only_fit)",
            "fixed_posterior_direct": "E_under_mixed_posterior[D U_short Q_C e_u] in input flux units",
            "fitted_model_change_remainder": "paired_end_to_end - fixed_posterior_direct",
            "centering": "Arithmetic mean over the identical actual B observation times",
        },
        "rows": rows,
        "notes": [
            "Uses existing mean and influence arrays only; no native reconstruction, fit, or posterior resampling.",
            "This is one paired realization and a difference of two independently sampled fits, including posterior numerical error.",
            "The remainder measures the change of fitted-model/posterior behavior on the baseline input; it can include normalization/serialization roundoff and is not a separately identified physical component.",
            "beta is an amplitude projection, not a variance or power fraction; none of these values is a population contamination rate.",
            "The fixed-posterior direct coefficient is not the total effect of adding u and refitting.",
        ],
    }
    destination = OUT / "paired_short_effects.json"
    if destination.exists():
        raise FileExistsError("Preserve existing paired-effect diagnostic")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=FIT_CASES)
    parser.add_argument("--paired-effects", action="store_true",
                        help="Postprocess the existing means/influence files without native calls")
    arguments = parser.parse_args()
    if arguments.paired_effects:
        paired_effects()
    elif arguments.worker:
        worker(arguments.worker)
    else:
        run_all()
