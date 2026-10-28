#!/usr/bin/env python3
"""Three pre-specified sensitivity fits against the saved xi=10 baseline.

Only changes original errors, Ground_z wavelength, or Nt, one at a time.
The Nt experiment deliberately keeps xi and the discrete regularizer definition
unchanged; it is a grid sensitivity test, not a numerical convergence proof.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import gc
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from temperature_mapping import Inversion, DEFAULT, load_data

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def comparison(q, baseline, times):
    result = {}
    for label, mask in (
        ("full_time", np.ones(len(times), dtype=bool)),
        ("exclude_edge_20days", (times >= times.min()+20) & (times <= times.max()-20)),
    ):
        a = q[:, mask].ravel()
        b = baseline[:, mask].ravel()
        delta = a-b
        rms_baseline = float(np.sqrt(np.mean(b*b)))
        result[label] = {
            "n_time_bins": int(mask.sum()), "n_grid_cells": int(len(a)),
            "rms_difference": float(np.sqrt(np.mean(delta*delta))),
            "baseline_rms": rms_baseline,
            "rms_difference_over_baseline_rms": float(np.sqrt(np.mean(delta*delta))/rms_baseline),
            "pearson_correlation": float(np.corrcoef(a,b)[0,1]),
            "mean_difference": float(delta.mean()),
            "max_abs_difference": float(np.abs(delta).max()),
        }
    return result


def main():
    start = time.perf_counter()
    threadpool_limits(limits=1)
    config = json.loads((OUT/"config.json").read_text())
    baseline = np.load(OUT/"fits.npz")
    data, bands, bi, wave = load_data(ROOT/"derived/lightcurves.csv")
    t, flux, error = (data[k].astype(float) for k in ("thjd","flux","error"))
    original = data["error_original"].astype(float)
    for actual, expected, name in (
        (t, baseline["t"], "time"),
        (flux, baseline["flux"], "flux"),
        (error, baseline["error"], "error"),
        (np.array(bands), baseline["bands"], "band ordering"),
        (wave, baseline["wavelengths"], "wavelengths"),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError(f"Prepared data differ from baseline: {name}")
    params = {key: config[key] for key in DEFAULT}
    if params["nr"] != 50 or params["nt"] != 250:
        raise ValueError("Expected NR=50, Nt=250 baseline")
    qb = baseline["q_baseline"].copy()
    tp = baseline["tp"].copy()
    u = baseline["u"].copy()
    baseline_residual = flux-baseline["pred_baseline"]
    summary = {
        "baseline": {
            "xi": 10, "nr": params["nr"], "nt": params["nt"],
            "chi2_per_point": float(np.mean((baseline_residual/error)**2)),
            "scale99": float(np.quantile(np.abs(qb),.99)),
            "n_data": len(t),
            "source": "outputs/fits.npz q_baseline/pred_baseline; outputs/config.json",
        },
        "comparison": "Unweighted grid-cell RMS and flattened Pearson correlation; all 50 radial cells; variants linearly interpolated in observer time onto the saved baseline (u,tp) grid",
        "error_comparison_note": "chi2_per_point uses each variant's fitting errors; chi2_per_point_baseline_errors uses common baseline errors, allowing like-for-like residual comparison",
        "nt_caveat": "Nt=500 keeps xi=10 and the existing unscaled discrete regularizer exactly as defined. Its strength relative to the likelihood and its physical time-smoothing scale change with the grid. This is sensitivity evidence, not a convergence proof.",
        "fixed_parameters": params,
        "threads": {"OPENBLAS_NUM_THREADS": 1, "OMP_NUM_THREADS": 1, "MKL_NUM_THREADS": 1},
        "variants": {},
    }
    saved = {"u_baseline":u, "tp_baseline":tp, "q_baseline":qb}
    for name in ("original_errors", "ground_z_8897", "nt_500"):
        variant_start = time.perf_counter()
        p = params.copy()
        e = error.copy()
        w = wave.copy()
        if name == "original_errors":
            e = original.copy()
            change = "Use error_original instead of triplet-expanded errors"
        elif name == "ground_z_8897":
            w[bands.index("Ground_z")] = 8897.0
            change = "Ground_z observed wavelength 8700 -> 8897 Angstrom"
        else:
            p["nt"] = 500
            change = "Nt 250 -> 500; fixed xi=10, no regularizer recalibration"
        model = Inversion(t, bi, w, e, p)
        construction_seconds = time.perf_counter()-variant_start
        factor_seconds = model.factorize(10.0)
        solve_start = time.perf_counter()
        fit = model.solve(flux)
        solve_seconds = time.perf_counter()-solve_start
        if not np.array_equal(model.u, u):
            raise ValueError("Unexpected radial-grid difference")
        if np.array_equal(model.tp, tp):
            q_common = fit["q"].copy()
        else:
            q_common = np.vstack([np.interp(tp, model.tp, row) for row in fit["q"]])
        if not np.isfinite(q_common).all():
            raise ValueError("Invalid interpolated map")
        result = {
            "change": change, "xi": 10, "nr": p["nr"], "nt": p["nt"],
            "chi2_per_point": fit["chi2_per_point"],
            "chi2_per_point_baseline_errors": float(np.mean((fit["residual"]/error)**2)),
            "scale99": fit["scale99"],
            "scale99_on_baseline_grid": float(np.quantile(np.abs(q_common),.99)),
            "normal_equation_relative_residual": fit["normal_equation_relative_residual"],
            "map_comparison": comparison(q_common,qb,tp),
            "construction_seconds": construction_seconds,
            "factor_seconds": factor_seconds,
            "solve_seconds": solve_seconds,
            "elapsed_seconds": time.perf_counter()-variant_start,
        }
        summary["variants"][name] = result
        saved.update({f"q_{name}":fit["q"], f"q_common_{name}":q_common,
                      f"tp_{name}":model.tp, f"pred_{name}":fit["pred"],
                      f"error_{name}":e, f"wavelengths_{name}":w})
        print(name, json.dumps(result), flush=True)
        del model, fit
        gc.collect()
    summary["elapsed_seconds"] = time.perf_counter()-start
    np.savez_compressed(OUT/"sensitivity.npz",**saved)
    (OUT/"sensitivity.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(f"Total elapsed {summary['elapsed_seconds']:.3f} seconds",flush=True)


if __name__ == "__main__":
    main()

