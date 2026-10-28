"""Assess weighted posterior stability without running or modifying a fit.

The immutable tolerances are in stability_gate_v1.json. A passing assessment is
an empirical stability check across independent runs and larger budgets, not a
proof of convergence, an iid effective sample size, or truth recovery.

Accepted run schema (DNest or dynesty): sampler_config.json containing mode,
parameter_names, priors, seed, budget; weighted_posterior.csv with named physical
parameter columns and weight; input_w2.txt; input_response.txt; run_input.json
with driver/response paths (or config.case). Dynesty config must set
sampler="dynesty", budget.nlive and a fixed stopping tolerance, e.g. dlogz.
DNest defaults to sampler="dnest" and budget.max_num_saves. Finished CSVs are
used directly; native equal-weight resamples are deliberately not consulted.

Example:
  python -B assess_stability.py --runs-root OUTPUT/phase1 --case shared \
      --mode shared --sampler dnest --out OUTPUT/phase1_stability.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations, product
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr


GATE_PATH = Path(__file__).with_name("stability_gate_v1.json")
DEFAULT_INPUTS = Path("/home/leonhard/sci001/output/mica_slowgp/small_cases")
PHYSICAL_NAMES = ("sigma_x", "tau_x", "A0", "mu0", "w0", "A1", "mu1", "w1")
SLOW_NAMES = ("qW", "qb", "Ts")


def read_json(path):
    return json.loads(Path(path).read_text())


def normalized_weights(weights):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Weights must be a finite, nonnegative, nonempty vector")
    if not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("Weights must have positive finite total")
    return w/w.sum()


def weighted_quantile(values, weights, probabilities):
    """Left-inverse empirical CDF: no interpolation across a probability atom."""
    x = np.asarray(values, dtype=float)
    w = normalized_weights(weights)
    if x.shape != w.shape or not np.isfinite(x).all():
        raise ValueError("Quantile inputs must have matching finite shapes")
    positive = w > 0
    order = np.argsort(x[positive], kind="stable")
    x, w = x[positive][order], w[positive][order]
    cumulative = np.cumsum(w)
    cumulative[-1] = 1.0
    index = np.searchsorted(cumulative, probabilities, side="left")
    return x[np.clip(index, 0, len(x)-1)]


def empirical_distances(x, wx, y, wy):
    """Exact weighted one-dimensional KS and W1, including ties correctly."""
    x, y = np.asarray(x), np.asarray(y)
    wx, wy = normalized_weights(wx), normalized_weights(wy)
    positive_x, positive_y = wx > 0, wy > 0
    x, wx, y, wy = x[positive_x], wx[positive_x], y[positive_y], wy[positive_y]
    ox, oy = np.argsort(x, kind="stable"), np.argsort(y, kind="stable")
    x, y = x[ox], y[oy]
    cx, cy = np.r_[0., np.cumsum(wx[ox])], np.r_[0., np.cumsum(wy[oy])]
    cx[-1], cy[-1] = 1., 1.
    locations = np.unique(np.r_[x, y])
    difference = np.abs(cx[np.searchsorted(x, locations, side="right")]
                        - cy[np.searchsorted(y, locations, side="right")])
    ks = float(difference.max(initial=0.))
    w1 = float(np.sum(np.diff(locations)*difference[:-1]))
    return ks, w1


def compare_scalar(x, wx, y, wy, gate):
    wx, wy = normalized_weights(wx), normalized_weights(wy)
    probabilities = gate["quantile_probabilities"]
    qa, qb = weighted_quantile(x, wx, probabilities), weighted_quantile(y, wy, probabilities)
    pooled_q = weighted_quantile(np.r_[x, y], np.r_[wx/2, wy/2], [.16, .84])
    scale = float(pooled_q[1]-pooled_q[0])
    ks, w1 = empirical_distances(x, wx, y, wy)
    mean_a, mean_b = float(np.dot(wx, x)), float(np.dot(wy, y))
    quantile_difference = float(np.max(np.abs(qa-qb)))
    mean_difference = abs(mean_a-mean_b)

    def standardize(value):
        # A degenerate central interval is not silently widened to make a pass.
        if scale > 0:
            return value/scale
        return 0. if value == 0 else math.inf

    normalized = dict(wasserstein1=standardize(w1),
                      quantile_displacement=standardize(quantile_difference),
                      mean_displacement=standardize(mean_difference))
    checks = {
        "weighted_ks": ks <= gate["maximum_weighted_ks"],
        "standardized_wasserstein1": normalized["wasserstein1"] <= gate["maximum_standardized_wasserstein1"],
        "standardized_quantile_displacement": normalized["quantile_displacement"] <= gate["maximum_standardized_quantile_displacement"],
        "standardized_mean_displacement": normalized["mean_displacement"] <= gate["maximum_standardized_mean_displacement"],
    }
    return dict(passed=all(checks.values()), checks=checks, weighted_ks=ks,
                wasserstein1=w1, standardized=normalized,
                pooled_q84_minus_q16=scale, quantiles_a=qa.tolist(),
                quantiles_b=qb.tolist(), mean_a=mean_a, mean_b=mean_b,
                degenerate_central_scale=(scale == 0))


def derived_parameters(columns):
    a0, a1 = columns["A0"], columns["A1"]
    m0, m1, w0, w1 = (columns[n] for n in ("mu0", "mu1", "w0", "w1"))
    area = a0+a1
    if np.any(area <= 0) or np.any(w0 <= 0) or np.any(w1 <= 0):
        raise ValueError("Response area and widths must be positive")
    center = (a0*m0+a1*m1)/area
    variance = (a0*(w0*w0+(m0-center)**2)+a1*(w1*w1+(m1-center)**2))/area
    negative = (a0*ndtr(-m0/w0)+a1*ndtr(-m1/w1))/area
    return dict(total_area=area, total_centroid=center, total_sd=np.sqrt(variance),
                total_negative_lag_fraction=negative)


def branch_probabilities(columns, weights):
    return {"mu0<0": float(np.dot(weights, columns["mu0"] < 0)),
            "A0<A1": float(np.dot(weights, columns["A0"] < columns["A1"])),
            "mu1<5": float(np.dot(weights, columns["mu1"] < 5))}


def transfer_grid(priors, gate):
    spec = gate["tf_grid"]
    maximum_width = max(priors[n]["bounds"][1] for n in ("w0", "w1"))
    minimum_width = min(priors[n]["bounds"][0] for n in ("w0", "w1"))
    if minimum_width <= 0:
        raise ValueError("Finite TF grid requires strictly positive prior width minima")
    low = min(priors[n]["bounds"][0] for n in ("mu0", "mu1"))
    high = max(priors[n]["bounds"][1] for n in ("mu0", "mu1"))
    low -= spec["prior_width_padding_sigma"]*maximum_width
    high += spec["prior_width_padding_sigma"]*maximum_width
    max_step = minimum_width*spec["maximum_step_over_minimum_prior_width"]
    points = max(spec["minimum_points"], int(np.ceil((high-low)/max_step))+1)
    if points > spec["maximum_points"]:
        raise ValueError(f"TF grid needs {points} points; refusing to underresolve narrow kernels")
    return np.linspace(low, high, points)


def transfer_summary(columns, weights, grid, gate):
    """Weighted function moments/quantiles in bounded column blocks, no resample."""
    positive = weights > 0
    weights = normalized_weights(weights[positive])
    columns = {name: x[positive] for name, x in columns.items()}
    n = len(weights)
    block_size = max(1, min(64, gate["tf_grid"]["maximum_working_matrix_entries"]//n))
    probabilities = gate["tf_quantiles"]
    mean = np.empty(len(grid))
    quantiles = np.empty((len(probabilities), len(grid)))
    for start in range(0, len(grid), block_size):
        stop = min(len(grid), start+block_size)
        t = grid[None, start:stop]
        value = np.zeros((n, stop-start))
        for suffix in ("0", "1"):
            amplitude = columns["A"+suffix][:, None]
            center = columns["mu"+suffix][:, None]
            width = columns["w"+suffix][:, None]
            value += amplitude/(math.sqrt(2*math.pi)*width)*np.exp(-.5*((t-center)/width)**2)
        mean[start:stop] = weights @ value
        order = np.argsort(value, axis=0, kind="stable")
        ordered_value = np.take_along_axis(value, order, axis=0)
        cumulative = np.cumsum(weights[order], axis=0)
        cumulative[-1] = 1.
        for j, probability in enumerate(probabilities):
            index = np.argmax(cumulative >= probability, axis=0)
            quantiles[j, start:stop] = ordered_value[index, np.arange(stop-start)]
    area = float(np.dot(weights, columns["A0"]+columns["A1"]))
    integrated_area = float(np.trapezoid(mean, grid))
    relative_error = abs(integrated_area-area)/area
    # This is numerical adequacy, not a stability tolerance that may be relaxed.
    if relative_error > 1e-6:
        raise ValueError(f"TF integration does not recover analytic area: {relative_error:g}")
    return dict(mean=mean, quantiles=quantiles, analytic_mean_area=area,
                relative_area_integration_error=relative_error)


@dataclass
class Run:
    path: Path
    case: str
    mode: str
    sampler: str
    seed: int
    effort: float
    config: dict
    columns: dict
    weights: np.ndarray
    input_arrays: tuple
    features: dict = field(default_factory=dict)
    transforms: dict = field(default_factory=dict)
    branches: dict = field(default_factory=dict)
    transfer: dict | None = None
    quality: dict = field(default_factory=dict)

    @property
    def label(self):
        return f"{self.case}/{self.mode}/{self.sampler}/seed={self.seed}/effort={self.effort:g}"


def run_identity(folder):
    config = read_json(folder/"sampler_config.json")
    sampler = str(config.get("sampler", config.get("sampler_type", "dnest"))).lower()
    if sampler in ("cdnest", "dnest4"):
        sampler = "dnest"
    run_input = read_json(folder/"run_input.json") if (folder/"run_input.json").exists() else {}
    case = config.get("case", run_input.get("case"))
    if case is None:
        case = Path(run_input["driver"]).parent.name
    return config, case, config["mode"], sampler


def read_run(folder, inputs_root, gate):
    config, case, mode, sampler = run_identity(folder)
    if sampler not in gate["effort_fields"]:
        raise ValueError(f"Unsupported sampler {sampler}")
    mode = config["mode"]
    if case not in gate["cases"] or mode not in gate["modes"]:
        raise ValueError(f"Unknown fixed-suite case/mode: {case}/{mode}")
    expected_names = list(PHYSICAL_NAMES + (() if mode == "none" else SLOW_NAMES))
    names = config["parameter_names"]
    if set(names) != set(expected_names) or len(names) != len(expected_names):
        raise ValueError("Physical parameter schema differs from the two-Gaussian model")
    if sampler == "dynesty" and not any(name in config["budget"] for name in ("dlogz", "dlogz_init")):
        raise ValueError("Dynesty budget must record its stopping tolerance")
    priors = read_json(inputs_root/"priors.json")
    expected_priors = {name: priors[name] for name in names}
    if config["priors"] != expected_priors:
        raise ValueError("Run priors differ from the original fixed small-case priors")
    actual_inputs = tuple(np.loadtxt(folder/name, ndmin=2)
                          for name in ("input_w2.txt", "input_response.txt"))
    expected_inputs = tuple(np.loadtxt(inputs_root/case/name, ndmin=2)
                            for name in ("w2.txt", "response.txt"))
    if any(a.shape != (42, 3) for a in actual_inputs):
        raise ValueError("This stability experiment requires the existing 42+42 input points")
    if any(not np.array_equal(a, b) for a, b in zip(actual_inputs, expected_inputs)):
        raise ValueError("Run input values differ from the original fixed simulation input")
    table = np.genfromtxt(folder/"weighted_posterior.csv", delimiter=",", names=True, ndmin=1)
    if table.dtype.names is None or not set(names+["weight"]) <= set(table.dtype.names):
        raise ValueError("Weighted posterior CSV is missing parameter or weight columns")
    columns = {name: np.asarray(table[name], dtype=float) for name in names}
    if any(not np.isfinite(value).all() for value in columns.values()):
        raise ValueError("Posterior parameters contain nonfinite values")
    weights = normalized_weights(table["weight"])
    positive = weights > 0
    columns = {name: value[positive] for name, value in columns.items()}
    weights = normalized_weights(weights[positive])
    for name, values in columns.items():
        low, high = expected_priors[name]["bounds"]
        if np.any(values < low) or np.any(values > high):
            raise ValueError(f"Posterior {name} lies outside its declared unchanged prior")
    if np.any(columns["mu0"] >= columns["mu1"]):
        raise ValueError("Ordered-center constraint violated")
    effort = float(config["budget"][gate["effort_fields"][sampler]])
    if not np.isfinite(effort) or effort <= 0:
        raise ValueError("Declared effort must be positive and finite")
    run = Run(folder, case, mode, sampler, int(config["seed"]), effort,
              config, columns, weights, actual_inputs)
    physical = {**columns, **derived_parameters(columns)}
    for name, values in physical.items():
        transform = "log" if name in gate["log_parameters"] or name in gate["derived_log_parameters"] else "linear"
        if transform == "log" and np.any(values <= 0):
            raise ValueError(f"Cannot test positive-scale parameter {name} on log scale")
        run.features[name] = np.log(values) if transform == "log" else values
        run.transforms[name] = transform
    run.branches = branch_probabilities(columns, weights)
    _, inverse = np.unique(np.column_stack([columns[name] for name in names]), axis=0, return_inverse=True)
    atom_weights = np.bincount(inverse, weights=weights)
    ess, max_weight = float(1/np.dot(weights, weights)), float(weights.max())
    run.quality = dict(positive_weight_rows=len(weights), weight_kish_ess=ess,
                       maximum_single_row_weight=max_weight,
                       unique_parameter_states=len(atom_weights),
                       combined_identical_state_weight_ess=float(1/np.dot(atom_weights, atom_weights)),
                       weight_ess_is_not_independent_sample_count=True,
                       passed=(ess >= gate["minimum_weight_kish_ess"]
                               and max_weight <= gate["maximum_single_row_weight"]))
    return run


def comparison(a, b, gate, kind):
    parameter_results = {name: compare_scalar(a.features[name], a.weights, b.features[name], b.weights, gate)
                         for name in a.features}
    for name, result in parameter_results.items():
        result["coordinate"] = a.transforms[name]
    branch_results = {name: dict(a=a.branches[name], b=b.branches[name],
                                absolute_difference=abs(a.branches[name]-b.branches[name]),
                                passed=abs(a.branches[name]-b.branches[name]) <= gate["maximum_branch_probability_displacement"])
                      for name in gate["branches"]}
    grid = transfer_grid(a.config["priors"], gate)
    for run in (a, b):
        if run.transfer is None:
            run.transfer = transfer_summary(run.columns, run.weights, grid, gate)
    normalization = .5*(a.transfer["analytic_mean_area"]+b.transfer["analytic_mean_area"])
    mean_difference = float(np.trapezoid(np.abs(a.transfer["mean"]-b.transfer["mean"]), grid)/normalization)
    quantile_difference = np.trapezoid(np.abs(a.transfer["quantiles"]-b.transfer["quantiles"]), grid, axis=1)/normalization
    tf_result = dict(mean_integrated_absolute_displacement=mean_difference,
                     quantile_integrated_absolute_displacements=quantile_difference.tolist(),
                     quantile_probabilities=gate["tf_quantiles"],
                     normalization_area=normalization, grid_points=len(grid),
                     grid_min=float(grid[0]), grid_max=float(grid[-1]),
                     grid_step=float(grid[1]-grid[0]),
                     passed=(mean_difference <= gate["maximum_tf_mean_integrated_absolute_displacement"]
                             and float(quantile_difference.max()) <= gate["maximum_tf_quantile_integrated_absolute_displacement"]))
    failed_parameters = [name for name, result in parameter_results.items() if not result["passed"]]
    failed_branches = [name for name, result in branch_results.items() if not result["passed"]]
    passed = not failed_parameters and not failed_branches and tf_result["passed"]
    return dict(kind=kind, run_a=str(a.path), run_b=str(b.path), seed_a=a.seed, seed_b=b.seed,
                effort_a=a.effort, effort_b=b.effort, passed=passed,
                failed_parameters=failed_parameters, failed_branches=failed_branches,
                parameters=parameter_results, branches=branch_results, total_tf=tf_result)


def algorithm_contract(config):
    # Recorded fields only: no claim that a missing source-version field verifies code identity.
    keys = ("center_prior", "parameterization", "proposal", "proposals", "proposal_config",
            "proposal_policy", "proposal_version", "sampler_version", "mpi_size")
    return {key: config[key] for key in keys if key in config}


def assess_group(runs, gate):
    by_seed = {}
    for run in runs:
        by_seed.setdefault(run.seed, []).append(run)
    issues = []
    if len(by_seed) < gate["minimum_independent_seeds"]:
        issues.append(f"Need >=3 independent seeds; found {len(by_seed)}")
    baseline = runs[0]
    effort_field = gate["effort_fields"][baseline.sampler]
    baseline_controls = {k: v for k, v in baseline.config["budget"].items() if k != effort_field}
    for run in runs[1:]:
        if run.config["priors"] != baseline.config["priors"]:
            issues.append("Prior configuration differs across runs")
        if algorithm_contract(run.config) != algorithm_contract(baseline.config):
            issues.append("Recorded algorithm/proposal settings differ across runs")
        controls = {k: v for k, v in run.config["budget"].items() if k != effort_field}
        if controls != baseline_controls:
            issues.append("Non-effort budget controls differ across runs")
    finals, extension_pairs = [], []
    for seed, seed_runs in sorted(by_seed.items()):
        ordered = sorted(seed_runs, key=lambda run: run.effort)
        if len({r.effort for r in ordered}) != len(ordered):
            issues.append(f"Seed {seed}: duplicate runs at the same effort; do not count as independent")
        if len(ordered) < gate["minimum_budgets_per_seed"]:
            issues.append(f"Seed {seed}: missing budget extension")
        finals.append(ordered[-1])
        if len(ordered) >= 2:
            low, high = ordered[-2:]
            if high.effort/low.effort < gate["minimum_effort_ratio"]:
                issues.append(f"Seed {seed}: latest effort ratio is below 2")
            effort_field = gate["effort_fields"][high.sampler]
            low_controls = {k: v for k, v in low.config["budget"].items() if k != effort_field}
            high_controls = {k: v for k, v in high.config["budget"].items() if k != effort_field}
            if low_controls != high_controls:
                issues.append(f"Seed {seed}: non-effort budget controls changed")
            extension_pairs.append((low, high))
    if gate["require_equal_final_effort"] and len({r.effort for r in finals}) != 1:
        issues.append("Final effort differs across independent seeds")
    # Compare even incomplete designs to expose the reason they cannot pass.
    pairs = [(a, b, "independent_seed_final_budget") for a, b in combinations(finals, 2)]
    pairs += [(a, b, "within_seed_budget_extension") for a, b in extension_pairs]
    comparisons = [comparison(a, b, gate, kind) for a, b, kind in pairs]
    assessed = {str(run.path): run for run in finals}
    assessed.update({str(run.path): run for pair in extension_pairs for run in pair})
    quality_pass = all(run.quality["passed"] for run in assessed.values())
    distribution_pass = bool(comparisons) and all(item["passed"] for item in comparisons)
    status = "stable_within_prespecified_tolerances" if not issues and quality_pass and distribution_pass else (
        "incomplete_design" if issues else "not_stable_within_prespecified_tolerances")
    summaries = []
    for run in sorted(runs, key=lambda r: (r.seed, r.effort)):
        summaries.append(dict(path=str(run.path), seed=run.seed, effort=run.effort,
                              quality=run.quality, branches=run.branches,
                              parameters={name: dict(coordinate=run.transforms[name],
                                  quantiles=weighted_quantile(x, run.weights, gate["quantile_probabilities"]).tolist(),
                                  mean=float(np.dot(run.weights, x)))
                                  for name, x in run.features.items()}))
    return dict(case=baseline.case, mode=baseline.mode, sampler=baseline.sampler,
                status=status, design_issues=list(dict.fromkeys(issues)),
                number_independent_seeds=len(by_seed), weight_resolution_passed=quality_pass,
                all_distribution_comparisons_passed=distribution_pass,
                runs=summaries, comparisons=comparisons), finals


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def self_test():
    gate = read_json(GATE_PATH)
    a, wa = np.array([0., 0., 1.]), np.array([.2, .3, .5])
    b, wb = np.array([0., 1.]), np.array([.5, .5])
    assert empirical_distances(a, wa, b, wb) == (0., 0.)
    np.testing.assert_array_equal(weighted_quantile(a, wa, [.5, .500001]), [0., 1.])
    assert empirical_distances(b, wb, b+1, wb) == (.5, 1.)
    assert compare_scalar(a, wa, b, wb, gate)["passed"]
    assert not compare_scalar(b, wb, b+1, wb, gate)["passed"]
    prior = {"mu0": {"bounds": [-3., 28.]}, "mu1": {"bounds": [-3., 28.]},
             "w0": {"bounds": [.1, 12.]}, "w1": {"bounds": [.1, 12.]}}
    grid = transfer_grid(prior, gate)
    assert np.diff(grid).max() <= .1/3+1e-12
    columns = dict(A0=np.array([1., 1.]), A1=np.array([.4, .4]),
                   mu0=np.array([1., 1.]), mu1=np.array([7., 7.]),
                   w0=np.array([.1, .1]), w1=np.array([2., 2.]))
    summary = transfer_summary(columns, wb, grid, gate)
    np.testing.assert_allclose(summary["quantiles"], np.tile(summary["mean"], (3, 1)), atol=1e-14)
    assert summary["relative_area_integration_error"] < 1e-10
    moment = derived_parameters(columns)
    np.testing.assert_allclose(moment["total_area"], 1.4)
    np.testing.assert_allclose(moment["total_centroid"], (1+2.8)/1.4)
    # Test the decision design independently of disk I/O or any real posterior.
    def synthetic_run(seed, effort, shift=0):
        n = 401
        weight = np.repeat(1/n, n)
        cols = {key: np.repeat(value[0], n) for key, value in columns.items()}
        cols["mu0"] = np.linspace(.7, 1.3, n)+shift
        config = dict(priors=prior, budget=dict(max_num_saves=effort, ptol=.01),
                      center_prior="ordered", mpi_size=1)
        run = Run(Path(f"synthetic_{seed}_{effort}"), "shared", "shared", "dnest",
                  seed, effort, config, cols, weight, ())
        run.features = {**cols, **derived_parameters(cols)}
        run.transforms = {key: "linear" for key in run.features}
        run.branches = branch_probabilities(cols, weight)
        run.quality = dict(passed=True)
        return run
    matched = [synthetic_run(seed, effort) for seed in (1, 2, 3) for effort in (400, 800)]
    assert assess_group(matched, gate)[0]["status"] == "stable_within_prespecified_tolerances"
    assert assess_group(matched[:-1], gate)[0]["status"] == "incomplete_design"
    shifted = [synthetic_run(seed, effort, 2. if seed == 3 else 0.)
               for seed in (1, 2, 3) for effort in (400, 800)]
    assert assess_group(shifted, gate)[0]["status"] == "not_stable_within_prespecified_tolerances"
    print("self-test passed: ties, ECDF quantiles, KS/W1, TF resolution/area, complete/incomplete/shifted 3x2 designs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, action="append", default=[])
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--inputs-root", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--mode", action="append", dest="modes")
    parser.add_argument("--sampler", action="append", choices=("dnest", "dynesty"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.out or not (args.runs_root or args.run_dir):
        parser.error("Assessment requires --out and --runs-root or --run-dir")
    if args.out.exists() and not args.overwrite:
        parser.error("Report exists; choose a new output path or explicitly use --overwrite")
    gate = read_json(GATE_PATH)
    cases, modes = args.cases or gate["cases"], args.modes or gate["modes"]
    if set(cases)-set(gate["cases"]) or set(modes)-set(gate["modes"]):
        parser.error("Unknown case/mode")
    folders = {folder.resolve() for folder in args.run_dir}
    for root in args.runs_root:
        folders.update(path.parent.resolve() for path in root.rglob("sampler_config.json"))
    runs, rejected = [], []
    for folder in sorted(folders):
        try:
            _, case, mode, sampler = run_identity(folder)
            if case not in cases or mode not in modes or (args.sampler and sampler not in args.sampler):
                continue
            run = read_run(folder, args.inputs_root, gate)
            runs.append(run)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            rejected.append(dict(path=str(folder), reason=str(exc)))
    samplers = args.sampler or sorted({run.sampler for run in runs}) or ["dnest"]
    groups, final_runs = [], {}
    for case, mode, sampler in product(cases, modes, samplers):
        selected = [run for run in runs if (run.case, run.mode, run.sampler) == (case, mode, sampler)]
        if not selected:
            groups.append(dict(case=case, mode=mode, sampler=sampler, status="missing_group"))
            continue
        try:
            group, finals = assess_group(selected, gate)
            groups.append(group)
            final_runs[(case, mode, sampler)] = finals
        except (ValueError, FloatingPointError) as exc:
            groups.append(dict(case=case, mode=mode, sampler=sampler, status="assessment_error", error=str(exc)))
    cross_sampler = []
    for case, mode in product(cases, modes):
        for sa, sb in combinations(samplers, 2):
            for a, b in product(final_runs.get((case, mode, sa), []), final_runs.get((case, mode, sb), [])):
                cross_sampler.append(comparison(a, b, gate, "independent_sampler_final_budget"))
    stable = bool(groups) and not rejected and all(
        group["status"] == "stable_within_prespecified_tolerances" for group in groups)
    if cross_sampler:
        stable = stable and all(result["passed"] for result in cross_sampler)
    full_scope = set(cases) == set(gate["cases"]) and set(modes) == set(gate["modes"])
    report = dict(gate=gate, assessed_at_utc=datetime.now(timezone.utc).isoformat(),
                  input_reference=str(args.inputs_root.resolve()),
                  scope=dict(cases=cases, modes=modes, samplers=samplers),
                  scoped_stability_gate_passed=stable,
                  full_four_case_three_model_goal_gate_passed=stable and full_scope,
                  rejected_runs=rejected, groups=groups,
                  cross_sampler_comparisons=cross_sampler,
                  uncertainty_statement="Independent-run spread and budget drift are reported directly. No iid error bars, Rhat, or calibrated block-bootstrap uncertainty is assigned to native weighted rows.",
                  limitation="Stable repeated posteriors may share a missed mode or model error; this gate does not certify truth identification or uncertainty coverage.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False)+"\n")
    print(f"Report: {args.out}")
    print(f"Scoped gate passed: {stable}; full 4x3 goal gate passed: {stable and full_scope}")
    for group in groups:
        failed_pairs = sum(not comparison["passed"] for comparison in group.get("comparisons", []))
        print(f"{group['case']}/{group['mode']}/{group['sampler']}: {group['status']}; failed comparisons={failed_pairs}")
        for issue in group.get("design_issues", []):
            print(f"  {issue}")
    for rejection in rejected:
        print(f"Rejected {rejection['path']}: {rejection['reason']}")


if __name__ == "__main__":
    main()
