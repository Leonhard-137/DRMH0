#!/usr/bin/env python3
"""Exact MICA conditional component means, without predictive covariances.

This consumes the already centroid-ordered joint posterior in a staged run.
MICA supplies its native data and component cross covariance matrices. A
single dense LU solve per posterior row supplies the conditional means; no
sampling, posterior selection, or independent driver reconstruction occurs.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))


def native_block(path: Path, index: int = 1) -> np.ndarray:
    with path.open() as handle:
        next(handle)
        counts = [int(value) for value in next(handle).lstrip("# ").split(":")]
    values = np.loadtxt(path, ndmin=2)
    if sum(counts) != len(values):
        raise ValueError(f"Incomplete native output: {path}")
    return values[sum(counts[:index]):sum(counts[:index + 1])]


def flux_scales(curves: list[np.ndarray]) -> np.ndarray:
    """Match read.c normalization using data actually serialized by setup()."""
    means = np.array([float(np.mean(curve[:, 1])) for curve in curves])
    ranges = np.array([float(np.ptp(curve[:, 1])) / 2 for curve in curves])
    if np.any(means <= 0) or np.any(ranges <= 0):
        raise ValueError("Invalid native MICA normalization inputs")
    return ranges if np.any(means < ranges / 100) else means


def run_mean_reconstruction(stage_dir: str | Path) -> dict:
    """Write means.npz and mean_validation.json in one staged run.

    Run in a dedicated one-rank MPI process. Native setup products are written
    only under a fresh mean_only_native_*/ directory, preserving earlier runs.
    """
    from mpi4py import MPI
    import pymica
    from scipy.linalg import lu_factor, lu_solve
    from collect_mica_results import read_yaml
    from decompose_mica_branches import _native_mica_library, _set_native_string
    from run_mica import fit_config, setup_kwargs

    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Mean-only reconstruction requires exactly one MPI rank")
    started = time.monotonic()
    stage = Path(stage_dir).resolve()
    if (stage / "means.npz").exists() or (stage / "mean_validation.json").exists():
        raise FileExistsError(f"Preserve existing mean-only products: {stage}")
    info = json.loads((stage / "provenance.json").read_text())
    source = Path(info["case_dir"])
    config = read_yaml(source / "config" / "source_config.yaml")
    fit = fit_config(config, stage="mica_recovery")
    if fit.get("flag_trend", 0) != 0 or fit.get("flag_negative_resp", False):
        raise ValueError("This comparison requires constant baseline and positive Gamma responses")
    if fit["ncomp"] != 2 or fit["_native_type_tf"] != "gamma":
        raise ValueError("This comparison requires two identical Gamma components")
    fit["nd_rec"] = int(info.get("nd_rec", 600))
    posterior = np.loadtxt(stage / "data/posterior_sample1d.txt_2", ndmin=2)
    if (posterior.shape[1] != int(info["sample_columns"])
            or not np.isfinite(posterior).all()
            or not np.all(posterior[:, 5] <= posterior[:, 8])):
        raise ValueError("Invalid centroid-ordered joint posterior")
    inputs = [np.loadtxt(stage / "input" / name, ndmin=2)
              for name in ("cont.txt", "line.txt")]
    scratch = Path(tempfile.mkdtemp(prefix="mean_only_native_", dir=stage))
    previous_cwd = Path.cwd()
    initialized = False
    line_initialized = False
    library = None
    try:
        os.chdir(scratch)
        model = pymica.gmodel()
        kwargs = setup_kwargs(fit, {"set1": inputs}, info["band"])
        model.setup(**kwargs)
        library = _native_mica_library(pymica)
        library.set_argv.argtypes = [ctypes.c_int] * 5
        library.dnest_line.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        library.dnest_line.restype = ctypes.c_double
        library.set_covar_Pmat_data_line.argtypes = [ctypes.c_void_p, ctypes.c_int]
        library.set_covar_Pmat_data_line.restype = None
        library.set_covar_Umat_line_single.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ]
        library.set_covar_Umat_line_single.restype = None
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
        if ctypes.c_int.in_dll(library, "num_params").value != posterior.shape[1]:
            raise ValueError("MICA and posterior parameter layouts disagree")

        # Native setup rounds inputs to its data_input format. Use those actual
        # values, with the same per-curve scales, in the conditional mean solve.
        serialized = np.loadtxt(scratch / "data/data_input.txt", ndmin=2)
        ncon = len(inputs[0])
        curves = [serialized[:ncon], serialized[ncon:]]
        scales = flux_scales(curves)
        normalized_y = np.r_[curves[0][:, 1] / scales[0], curves[1][:, 1] / scales[1]]
        ndata = len(normalized_y)
        design = np.zeros((ndata, 2))
        design[:ncon, 0] = 1
        design[ncon:, 1] = 1
        rhs = np.column_stack((normalized_y, design))
        low = min(curve[0, 0] for curve in curves)
        high = max(curve[-1, 0] for curve in curves)
        extension = 0.05 * (high - low)
        native_200_t = np.linspace(low - extension, high + extension, min(ncon * 3, 200))
        native_t = np.linspace(low - extension, high + extension, min(ncon * 3, fit["nd_rec"]))
        obs_t = np.asarray(inputs[1][:, 0], dtype=np.float64)
        mean_step = float(info.get("mean_grid_step_days", 1.0))
        if not np.isfinite(mean_step) or mean_step <= 0:
            raise ValueError("Mean reconstruction grid step must be positive")
        plot_t = np.r_[np.arange(obs_t.min(), obs_t.max(), mean_step), obs_t.max()]
        have_native_components = any((stage / "data" / f"pline.txt_2_comp{i}").exists()
                                     for i in range(2))
        extra_native = native_t if have_native_components else np.empty(0)
        grid = np.unique(np.r_[extra_native, native_200_t, obs_t, plot_t])
        mean_sum = np.zeros((2, len(grid)))
        compensation = np.zeros_like(mean_sum)
        pc_pointer = ctypes.POINTER(ctypes.c_double).in_dll(library, "PCmat")
        cross_pointer = ctypes.POINTER(ctypes.c_double).in_dll(library, "USmat")
        covariance = np.ctypeslib.as_array(pc_pointer, shape=(ndata * ndata,)).reshape(ndata, ndata)
        # The native cross covariance buffer has nrec_max rows. Chunk arbitrary
        # grids rather than changing native allocation or imposing interpolation.
        chunk_size = min(512, ctypes.c_int.in_dll(library, "nrec_max").value)
        if chunk_size < 1:
            raise ValueError("MICA reconstruction buffer is empty")
        parameter_rows = np.ascontiguousarray(posterior, dtype=np.float64)
        for draw, pm in enumerate(parameter_rows):
            model_pointer = pm.ctypes.data_as(ctypes.c_void_p)
            library.set_covar_Pmat_data_line(model_pointer, 0)
            factor = lu_factor(covariance, overwrite_a=False, check_finite=False)
            solved = lu_solve(factor, rhs, check_finite=False)
            baseline = np.linalg.solve(design.T @ solved[:, 1:], design.T @ solved[:, 0])
            alpha = solved[:, 0] - solved[:, 1:] @ baseline
            areas = np.exp(pm[[4, 7]])
            weights = areas / areas.sum()
            for start in range(0, len(grid), chunk_size):
                stop = min(start + chunk_size, len(grid))
                times = np.ascontiguousarray(grid[start:stop], dtype=np.float64)
                counts = np.array([0, len(times)], dtype=np.int32)
                cross = np.ctypeslib.as_array(cross_pointer, shape=(len(times) * ndata,)).reshape(len(times), ndata)
                for component in range(2):
                    library.set_covar_Umat_line_single(
                        model_pointer, 0, counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        times.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), component,
                    )
                    values = (cross @ alpha + weights[component] * baseline[1]) * scales[1]
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite conditional mean at posterior draw {draw}")
                    # Compensated accumulation limits cancellation for highly
                    # anticorrelated components with individually large offsets.
                    corrected = values - compensation[component, start:stop]
                    updated = mean_sum[component, start:stop] + corrected
                    compensation[component, start:stop] = (updated - mean_sum[component, start:stop]) - corrected
                    mean_sum[component, start:stop] = updated
            if draw == 0 or (draw + 1) % 100 == 0 or draw + 1 == len(parameter_rows):
                print(f"MEAN {info['case']} {info['band']} {draw + 1}/{len(parameter_rows)}, "
                      f"{time.monotonic() - started:.1f}s", flush=True)
        means = mean_sum / len(parameter_rows)
        total = means.sum(axis=0)
        original = Path(info["run_dir"])
        reference_total = native_block(original / "data/pall.txt_2")
        at_old = np.searchsorted(grid, native_200_t)
        if len(reference_total) != len(native_200_t):
            raise ValueError("Original native total does not have the expected 200-node grid")
        maximum_difference = float(np.max(np.abs(total[at_old] - reference_total[:, 1])))
        validation = {
            "algorithm": "native_covariance_LU_conditional_mean",
            "case": info["case"], "band": info["band"],
            "posterior_rows": len(parameter_rows), "grid_rows": len(grid),
            "native_grid_rows": len(native_t), "observation_rows": len(obs_t),
            "plot_grid_rows": len(plot_t), "mean_grid_step_days": mean_step,
            "native_grid_included": have_native_components, "scales": scales.tolist(),
            "native_total_max_abs_diff": maximum_difference,
            "native_total_time_rounding_max_abs": float(np.max(np.abs(reference_total[:, 0] - native_200_t))),
            "native_total_pass": bool(np.allclose(total[at_old], reference_total[:, 1], atol=3e-6, rtol=1e-6)),
            "runtime_seconds": time.monotonic() - started,
            "notes": ["Every staged joint posterior row retained with equal weight.",
                      "No predicted-error or credible-interval estimate is produced.",
                      "Component baseline allocation follows native MICA area fractions.",
                      "Arbitrary evaluation times use native analytic covariance, without interpolation."],
        }
        for component in range(2):
            native_path = stage / "data" / f"pline.txt_2_comp{component}"
            if native_path.exists():
                reference = native_block(native_path)
                if len(reference) != len(native_t):
                    raise ValueError("Preserved native component has another reconstruction grid")
                values = means[component, np.searchsorted(grid, native_t)]
                validation[f"native_comp{component}_max_abs_diff"] = float(np.max(np.abs(values - reference[:, 1])))
                validation[f"native_comp{component}_pass"] = bool(np.allclose(values, reference[:, 1], atol=3e-6, rtol=1e-6))
        if not validation["native_total_pass"] or any(
                not value for key, value in validation.items() if key.startswith("native_comp") and key.endswith("_pass")):
            raise ValueError(f"Native-mean validation failed: {json.dumps(validation)}")
        np.savez_compressed(stage / "means.npz", t=grid, short=means[0], long=means[1], total=total,
                            native_200_t=native_200_t, native_t=extra_native, obs_t=obs_t, plot_t=plot_t)
        (stage / "mean_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps(validation), flush=True)
        return validation
    finally:
        if line_initialized:
            library.mc_line_end()
        if initialized:
            library.end_run()
        os.chdir(previous_cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_dir", type=Path)
    arguments = parser.parse_args()
    run_mean_reconstruction(arguments.stage_dir)
