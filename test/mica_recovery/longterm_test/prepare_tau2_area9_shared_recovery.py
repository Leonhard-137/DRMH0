#!/usr/bin/env python3
"""Prepare six matched direct shared-S controls; never launch a fit.

The saved S was drawn only at U epochs. Extend that SAME realization to W2
epochs using its conditional Gaussian distribution, retaining conditional
residual uncertainty. Components and measurements in the old experiment are
read-only, and this script refuses to overwrite its own output/configuration.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
import yaml

from prepare import _sample_psd_covariance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REFERENCE = HERE / "cases_tau2_area9/raw"
OUT = HERE / "cases_tau2_area9_shared"
CONFIG = HERE / "experiment_tau2_area9_shared.yaml"
BANDS = ("UVW2", "U")
EXTENSION_SEED = 2026100301
CASES = {"raw": (None, ()), **{f"all_s{s}": (s, BANDS) for s in (5, 10, 20, 40)},
         "u_s20": (20, ("U",))}
SCHEMA = ("time illumination_variation shared_long_variation shared_short_variation "
          "measurement_noise disk_illumination_variation blr_illumination_variation")


def center_covariance(matrix: np.ndarray) -> np.ndarray:
    """P_row K P_col, for P = identity - sample-mean operator."""
    return matrix - matrix.mean(axis=0)[None, :] - matrix.mean(axis=1)[:, None] + matrix.mean()


def extend_shared_slow(tw, tu, saved_u, tau, amplitude):
    """Conditional distribution of P_W S_W given the saved P_U S_U.

    Removing the final U datum only drops a redundant mean-zero constraint;
    no observational nugget/jitter or smoothing is added to S. W2 centering
    changes the constant zero point, not gain, lag, or its varying component.
    """
    def kernel(t1, t2):
        h = np.abs(t1[:, None] - t2[None, :]) / tau
        return amplitude**2 * (1 + h) * np.exp(-h)

    cyy = center_covariance(kernel(tu, tu))
    czy = center_covariance(kernel(tw, tu))
    czz = center_covariance(kernel(tw, tw))
    if abs(saved_u.mean()) > 1e-10:
        raise ValueError("Saved U slow process is not centered")
    y = saved_u - saved_u.mean()  # remove only text-rounding residue for the solve
    factor = cho_factor(cyy[:-1, :-1], lower=True, check_finite=True)
    weights = cho_solve(factor, y[:-1])
    cross = czy[:, :-1]
    mean = cross @ weights
    mean -= mean.mean()
    conditional = czz - cross @ cho_solve(factor, cross.T)
    conditional = (conditional + conditional.T) / 2
    before_center_min = float(eigh(conditional, eigvals_only=True)[0])
    # Enforce the analytically exact mean-zero W2 subspace. This removes
    # cancellation roundoff in the null direction, without adding jitter.
    conditional = center_covariance(conditional)
    conditional = (conditional + conditional.T) / 2
    residual, min_eigenvalue = _sample_psd_covariance(
        conditional, np.random.default_rng(EXTENSION_SEED))
    residual -= residual.mean()
    draw = mean + residual
    replay_error = float(np.max(np.abs(cyy[:, :-1] @ weights - y)))
    solve_error = float(np.max(np.abs(cyy[:-1, :-1] @ weights - y[:-1])))
    if replay_error > 1e-7:
        raise ValueError(f"Conditional reconstruction at U epochs is inaccurate: {replay_error}")
    diagnostics = dict(
        method="Conditional Gaussian extension of saved, centered U S to separately centered W2 S",
        covariance="K(t,t')=A_S^2*(1+abs(t-t')/tau_S)*exp(-abs(t-t')/tau_S)",
        centering="Cyy=P_U K_UU P_U; Czy=P_W K_WU P_U; Czz=P_W K_WW P_W; P=I-11^T/N",
        conditional_mean="Czy[:,keep] solve(Cyy[keep,keep], saved_U_S[keep]); keep=all U except final redundant datum",
        conditional_covariance="Czz-Czy[:,keep] solve(Cyy[keep,keep],Czy[:,keep].T)",
        component_zero_points="W2 and U are separately sample-centered; this changes only constant zero points, not common-process gain or lag",
        prior_tau_days=float(tau), prior_amplitude=float(amplitude),
        seed=EXTENSION_SEED, old_u_slow_unchanged=True, disk_blr_noise_unchanged=True,
        common_gain=1.0, relative_lag_days=0.0, disk_convolution=False,
        conditioning_nugget=0.0, saved_u_roundoff_mean_removed_for_conditioning=float(saved_u.mean()),
        conditioning_matrix_condition_number=float(np.linalg.cond(cyy[:-1, :-1])),
        conditioning_solve_max_abs_error=solve_error, conditioning_u_replay_max_abs_error=replay_error,
        covariance_min_eigenvalue_before_zero_mean_projection=before_center_min,
        conditional_min_eigenvalue=min_eigenvalue,
        conditional_expected_residual_rms=float(np.sqrt(np.trace(conditional) / len(tw))),
        conditional_realized_residual_rms=float(np.std(residual)),
        conditional_max_point_std=float(np.sqrt(np.diag(conditional).max())),
        slow_u_sample_rms=float(np.std(saved_u)), slow_w_sample_rms=float(np.std(draw)),
        note="The historical S had no saved off-grid values. This is a conditional extension retaining uncertainty, not recovery of unique historical hidden values.",
    )
    evidence = dict(tw=tw, tu=tu, observed_u=saved_u, draw_w=draw,
                    conditional_mean_w=mean, conditional_covariance_w=conditional,
                    conditional_residual_w=residual)
    return draw, diagnostics, evidence


def main():
    if OUT.exists() or CONFIG.exists():
        raise FileExistsError("Preserving existing shared-S experiment output/configuration")
    old_meta = json.loads((REFERENCE / "config/generation.json").read_text())
    p = deepcopy(old_meta["parameters"])
    data = {b: np.loadtxt(REFERENCE / f"input/LTSIM_{b}.txt") for b in BANDS}
    parts = {b: np.loadtxt(REFERENCE / f"components/{b}_components.txt") for b in BANDS}
    baseline = float(old_meta["baseline_flux"])
    assert baseline == 7.0 and old_meta["alpha"] == 1.45 and p["driver_tau_days"] == 2.0
    assert p["response_disk_gain"] == .787 and p["response_blr_gain"] == .787 / 9
    for b in BANDS:
        np.testing.assert_array_equal(data[b][:, 0], parts[b][:, 0])
        np.testing.assert_allclose(data[b][:, 1], baseline + parts[b][:, 1:5].sum(axis=1), atol=1e-12)
    assert np.all(parts["UVW2"][:, 2] == 0)
    old_parts = {b: c.copy() for b, c in parts.items()}
    slow_w, diagnostics, evidence = extend_shared_slow(
        parts["UVW2"][:, 0], parts["U"][:, 0], parts["U"][:, 2],
        p["independent_gp_tau_days"], p["independent_gp_amplitude"])
    parts["UVW2"][:, 2] = slow_w
    np.testing.assert_array_equal(parts["U"], old_parts["U"])
    np.testing.assert_array_equal(parts["UVW2"][:, [0, 1, 3, 4, 5, 6]],
                                  old_parts["UVW2"][:, [0, 1, 3, 4, 5, 6]])
    # Flags here describe the actual new saved experiment. Generic ablation
    # preparation is intentionally unsupported for this inference-only config.
    p.update(shared_driver=True, shared_response=True, unfiltered_response_gp=False,
             direct_shared_slow=True, direct_shared_slow_gain_w=1.0,
             direct_shared_slow_gain_u=1.0, direct_shared_slow_lag_days=0.0)
    spec = importlib.util.spec_from_file_location("smooth_prepare", ROOT / "test/mica_smooth/prepare.py")
    smooth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smooth)
    base_inputs = {b: np.column_stack([data[b][:, 0], baseline + parts[b][:, 1:5].sum(axis=1), data[b][:, 2]]) for b in BANDS}
    prepared, manifest_cases = {}, {}
    for name, (sigma, filtered_bands) in CASES.items():
        values, curves, covariance, trends = {}, {}, {}, {}
        for b in BANDS:
            c = parts[b].copy()
            if b in filtered_bands:
                w = smooth.gaussian_weights(c[:, 0], sigma)
                c[:, 1:] = parts[b][:, 1:] - w @ parts[b][:, 1:] + parts[b][:, 1:].mean(axis=0)
                curve, covariance[b] = smooth.detrend_curve(base_inputs[b], sigma, "original")
                trends[b] = np.column_stack([c[:, 0], w @ base_inputs[b][:, 1]])
            else:
                curve = base_inputs[b].copy()
            np.testing.assert_allclose(c[:, 1], c[:, 5] + c[:, 6], atol=1e-12)
            np.testing.assert_allclose(curve[:, 1], baseline + c[:, 1:5].sum(axis=1), atol=1e-12)
            np.testing.assert_array_equal(curve[:, [0, 2]], data[b][:, [0, 2]])
            if not np.isfinite(curve).all() or np.any(curve[:, 1] <= 0):
                raise ValueError(f"{name}/{b} requires a changed baseline; stop and report")
            # The only physical input change is S added to W2: all U inputs
            # match the prior paired experiment, including filtered controls.
            prior_case = "all_s20" if name == "u_s20" else name
            prior_u = np.loadtxt(REFERENCE.parent / prior_case / "input/LTSIM_U.txt")
            if b == "U":
                np.testing.assert_allclose(curve, prior_u, rtol=0, atol=1e-12)
            values[b], curves[b] = c, curve
        prepared[name] = (values, curves, covariance, trends)
        manifest_cases[name] = dict(sigma_days=sigma, filtered_bands=list(filtered_bands),
            baseline_flux=baseline, formula="original - Gaussian weighted trend + original mean" if sigma else "raw",
            driver_unchanged=("UVW2" not in filtered_bands),
            rms={b: {label: float(np.std(values[b][:, col])) for label, col in
                     (("disk", 5), ("BLR", 6), ("independent_S", 2), ("noise", 4))} for b in BANDS},
            minimum_flux={b: float(curves[b][:, 1].min()) for b in BANDS},
            quoted_errors="original", noise_covariance_used_by_mica="original diagonal errors; saved transformed covariance not supplied")
    np.testing.assert_array_equal(prepared["u_s20"][1]["UVW2"], prepared["raw"][1]["UVW2"])
    np.testing.assert_array_equal(prepared["u_s20"][1]["U"], prepared["all_s20"][1]["U"])
    meta = deepcopy(old_meta)
    meta.update(parameters=p, reference_case=str(REFERENCE),
        model="W2=7+alpha*DW*I+S+noise; U=7+alpha*(0.787*DU*I+(0.787/9)*B*I)+S+noise; S independent of I, same direct process at both bands, gain=1, lag=0, no convolution",
        shared_slow_extension=diagnostics,
        statistics={b: {"realized_rms": manifest_cases["raw"]["rms"][b]} for b in BANDS})
    cfg = yaml.safe_load((HERE / "experiment_tau2_area9.yaml").read_text())
    cfg["experiment"]["output_dir"] = OUT.name
    cfg["paired_ablation"]["cases"] = {name: dict(blr=True, shared_driver=True, shared_response=True) for name in CASES}
    cfg["generation"].update(input_case=str(OUT / "raw"), manifest=str(OUT / "manifest.json"),
        parameters=p, shared_slow_extension_seed=EXTENSION_SEED,
        model=meta["model"], inference_only=True,
        preparation_script=str(Path(__file__).resolve()))
    cfg["detrending"].update(input_case=str(OUT / "raw"),
        cases={name: dict(sigma_days=sigma, bands=list(bands)) for name, (sigma, bands) in CASES.items()})
    # All component/numerical/positivity validations precede first output write.
    OUT.mkdir()
    np.savez_compressed(OUT / "shared_slow_extension.npz", **evidence)
    manifest = dict(reference_case=str(REFERENCE), model=meta["model"],
        generation=meta, extension=diagnostics, cases=manifest_cases,
        psd_preflight_status="not_yet_run; generation does not authorize fitting")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, (values, curves, covariance, trends) in prepared.items():
        case = OUT / name
        for directory in ("input", "components", "config", "results"):
            (case / directory).mkdir(parents=True, exist_ok=True)
        for b in BANDS:
            np.savetxt(case / f"input/LTSIM_{b}.txt", curves[b], fmt="%.17g", header="time flux error")
            np.savetxt(case / f"components/{b}_components.txt", values[b], fmt="%.17g",
                       header=SCHEMA + "\nflux=7+sum(columns[1:5]); last two columns partition illumination")
            if b in covariance:
                np.save(case / f"input/{b}_filtered_noise_covariance.npy", covariance[b])
                np.savetxt(case / f"input/{b}_removed_trend.txt", trends[b], fmt="%.17g", header="time Gaussian_trend")
        (case / "config/generation.json").write_text(json.dumps(meta, indent=2) + "\n")
        provenance = dict(original_case=str(OUT / "raw"), reference_case=str(REFERENCE),
                          shared_slow_extension=str(OUT / "shared_slow_extension.npz"), **manifest_cases[name])
        (case / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    CONFIG.write_text("# Inference-only inputs: prepare_tau2_area9_shared_recovery.py; never generic prepare.py.\n"
                      + yaml.safe_dump(cfg, sort_keys=False))
    (OUT / "README.md").write_text(
        "# Tau2, area 9:1, same direct slow process in W2 and U\n\n" + meta["model"] + "\n\n"
        "Six matched inputs: raw; BOTH-band sigma5/10/20/40 detrending; U-only sigma20. "
        "Each receives two-Gaussian and two-Gamma fits (12 fits, 2 MPI ranks each). "
        "Priors and sampling retain max_num_saves=10000, ptol=0.01.\n\n"
        "Only W2 is given the additional S. U, disk, BLR, sampling, original errors and noise "
        "remain paired with cases_tau2_area9. S is independent of I and common to the two bands. "
        "Its original sample is available only at U epochs: a conditional Matérn3/2 GP draw extends "
        "the same sample to W2 epochs, retaining the residual covariance. Both bands' S are "
        "separately sample-centered (constant zero-point shifts). This does not introduce a "
        "relative lag, distinct random S, gain change, or disk convolution.\n\n"
        "Conditional evidence is in shared_slow_extension.npz and manifest.json. Baseline stays 7. "
        "Full filtered noise covariance is saved, while MICA retains original diagonal errors. "
        "Input generation does not run MICA: launch is conditional on a separate PSD preflight.\n")
    print(json.dumps(dict(output=str(OUT), config=str(CONFIG), baseline=baseline,
                          extension=diagnostics, cases=manifest_cases), indent=2))


if __name__ == "__main__":
    main()
