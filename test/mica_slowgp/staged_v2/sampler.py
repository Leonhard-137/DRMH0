"""STAGED v2: bounded, proper-prior CDNest callbacks, not the live sampler.

The stored coordinates are physical parameters, not their logarithms.  CDNest
does the nested sampling and writes its native samples, levels and weights.
Fixed-budget runs are exploratory: this module never certifies convergence or
promotes their raw evidence estimates to model-selection results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import importlib
import json
from pathlib import Path
import time
from typing import Callable, Mapping

import numpy as np


BASE_NAMES = ("sigma_x", "tau_x", "A0", "mu0", "w0", "A1", "mu1", "w1")
SLOW_NAMES = ("qW", "qb", "Ts")
PROPOSAL_VERSION = "prior-preserving-component-crossing-ridges-v2-staged"


def parameter_names(mode: str) -> tuple[str, ...]:
    if mode not in ("none", "independent", "shared"):
        raise ValueError("mode must be none, independent or shared")
    return BASE_NAMES + (() if mode == "none" else SLOW_NAMES)


def default_priors(mode: str = "independent") -> dict:
    """Broad declared bounds; no empirical single-OU or disk-size prior."""
    bounds = {
        "sigma_x": (1e-3, 10.0), "tau_x": (0.1, 1e4),
        "A0": (1e-5, 10.0), "mu0": (0.0, 100.0), "w0": (0.03, 40.0),
        "A1": (1e-5, 10.0), "mu1": (0.0, 100.0), "w1": (0.03, 40.0),
        "qW": (1e-5, 10.0), "qb": (1e-5, 10.0), "Ts": (10.0, 1e3),
    }
    return {name: {"kind": "uniform" if name.startswith("mu") else "log_uniform",
                   "bounds": list(bounds[name])} for name in parameter_names(mode)}


def load_priors(value=None, mode: str = "independent") -> dict:
    """Read a complete or partial parameter mapping (or JSON with priors key)."""
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if value is not None and "priors" in value:
        value = value["priors"]
    result = default_priors(mode)
    if value is not None:
        unknown = set(value) - set(BASE_NAMES + SLOW_NAMES)
        if unknown:
            raise ValueError(f"Unknown prior parameters: {sorted(unknown)}")
        for name in parameter_names(mode):
            if name in value:
                result[name] = dict(value[name])
    for name, spec in result.items():
        kind = spec.get("kind", "log_uniform")
        if kind not in ("uniform", "log_uniform"):
            raise ValueError(f"Unsupported prior for {name}: {kind}")
        lo, hi = map(float, spec["bounds"])
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError(f"Finite increasing bounds required for {name}")
        if kind == "log_uniform" and lo <= 0:
            raise ValueError(f"Positive log-uniform bounds required for {name}")
        if name in ("sigma_x", "tau_x", "Ts") and lo <= 0:
            raise ValueError(f"Strictly positive lower bound required for {name}")
        if name in ("A0", "A1", "w0", "w1", "qW", "qb") and lo < 0:
            raise ValueError(f"Nonnegative bounds required for {name}")
        if name.startswith("mu") and kind != "uniform":
            raise ValueError("Ordered centers use a uniform joint prior")
        result[name] = {"kind": kind, "bounds": [lo, hi]}
    a, b = result["mu0"]["bounds"]
    c, d = result["mu1"]["bounds"]
    if min(b, d) <= a:
        raise ValueError("Ordered-center prior has zero area")
    return result


@dataclass(frozen=True)
class SamplerBudget:
    """Maximum saves gives a finite proposal budget, not a convergence rule."""
    max_num_saves: int = 1000
    num_particles: int = 2
    thread_steps_factor: float = 5.0
    new_level_interval_factor: float = 2.0
    save_interval_factor: float = 2.0
    max_num_levels: int = 0
    lam: float = 10.0
    beta: float = 100.0
    ptol: float = 0.1

    def __post_init__(self):
        for name in ("max_num_saves", "num_particles"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if int(self.max_num_levels) != self.max_num_levels or self.max_num_levels < 0:
            raise ValueError("max_num_levels must be a nonnegative integer")
        for name in ("thread_steps_factor", "new_level_interval_factor",
                     "save_interval_factor", "lam", "beta", "ptol"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


class DNestModel:
    """Native CDNest callback protocol with prior-preserving proposals.

    Centers have a normalized uniform density on their bounded rectangle
    restricted to mu0 < mu1.  This is *not* a prior fixing either component to
    an externally supplied disk/BLR timescale.  Log-scale random walks preserve
    their log-uniform prior. The staged slow ridge uses its nonzero log Hastings
    correction for the two uniform amplitude priors.
    """
    def __init__(self, log_likelihood: Callable[[dict], float], mode="independent",
                 priors=None, seed=12345, seed_native: Callable[[], None] | None = None):
        self.names = parameter_names(mode)
        self.mode = mode
        self.priors = load_priors(priors, mode)
        self.num_params = len(self.names)
        self.param_range = np.asarray([self.priors[n]["bounds"] for n in self.names])
        # Custom from_prior/perturb replace CDNest's generic prior routines.
        self.prior_type = ["Uniform"] * self.num_params
        self.prior_info = np.zeros_like(self.param_range)
        self._logscale = np.array([self.priors[n]["kind"] == "log_uniform" for n in self.names])
        self._bounds = self.param_range.copy()
        self._bounds[self._logscale] = np.log(self._bounds[self._logscale])
        self._likelihood = log_likelihood
        self._rng = np.random.default_rng(seed)
        self._seed_native = seed_native
        self._native_seeded = False
        self.likelihood_calls = 0
        self.invalid_likelihood_calls = 0
        self._exchangeable_components = all(
            self.priors[a] == self.priors[b]
            for a, b in (("A0", "A1"), ("mu0", "mu1"), ("w0", "w1")))
        self._scale_compensation_enabled = bool(self._logscale[[0, 2, 5]].all())
        self._ou_ridge_enabled = bool(self._logscale[[0, 1]].all())
        self._slow_ridge_enabled = (mode != "none"
            and self.priors["qW"]["kind"] == "uniform"
            and self.priors["qb"]["kind"] == "uniform"
            and self.priors["Ts"]["kind"] == "log_uniform")
        self._noncenter_indices = np.array([j for j in range(self.num_params) if j not in (3, 6)])
        self.proposal_counts = dict.fromkeys((
            "component_center_crossing", "scale_compensation", "joint_prior_coordinates",
            "center_prior_refresh", "ou_ridge", "slow_ridge", "single_coordinate"), 0)
        self.component_reorder_proposals = 0
        self.ridge_boundary_rejections = {"ou_ridge": 0, "slow_ridge": 0}

    def proposal_config(self):
        """Describe attempts; these are not accepted-move counts."""
        crossing = 0.25 if self._exchangeable_components else 0.0
        compensation = 0.10 if self._scale_compensation_enabled else 0.0
        ou_ridge = 0.10 if self._ou_ridge_enabled else 0.0
        slow_ridge = 0.10 if self._slow_ridge_enabled else 0.0
        return {
            "version": PROPOSAL_VERSION,
            "attempt_probabilities": {
                "component_center_crossing": crossing,
                "scale_compensation": compensation,
                "joint_prior_coordinates": 0.10,
                "center_prior_refresh": 0.05,
                "ou_ridge": ou_ridge,
                "slow_ridge": slow_ridge,
                "single_coordinate": 0.85-crossing-compensation-ou_ridge-slow_ridge},
            "component_priors_exchangeable": self._exchangeable_components,
            "component_center_refresh_probability_within_move": 0.20,
            "component_sort": "sort entire (A, mu, w) triples after a full-range center move",
            "scale_compensation": "symmetric wrapped log shift (+s, -s, -s) for (sigma_x, A0, A1); requires all three log-uniform",
            "joint_prior_coordinates": "random nonempty subset excluding centers; symmetric Gaussian vector with a common heavy-tail scale in wrapped unit-prior coordinates",
            "ou_ridge": "(sigma_x,tau_x) *= (exp(s),exp(2*s)); both priors log-uniform; logH=0",
            "slow_ridge": "(qW,qb,Ts) *= exp(s); uniform qW/qb and log-uniform Ts; logH=2*s",
            "ridge_step": "s = normal(0,1) * 10**uniform(-4,0.5), independent of state and symmetric",
            "ridge_boundary_policy": "reject entire proposal with -inf and leave coordinates unchanged; do not redraw s or wrap",
            "slow_kernel_convention": "(1+abs(dt)/Ts)*exp(-abs(dt)/Ts); derivative covariance q_i*q_j/Ts**2",
            "fallback": "disabled moves, including slow ridge in mode none, use the original single-coordinate move",
            "log_hastings_rules": {"ou_ridge": "0", "slow_ridge": "2*s", "other_moves": "0"},
            "counts_are_proposals_not_acceptances": True,
        }

    def _centers_from_prior(self):
        # Integrate allowed mu1 length over mu0, then invert the piecewise CDF.
        a, b = self.param_range[3]
        c, d = self.param_range[6]
        flat_end = min(b, c)
        flat_area = max(0.0, flat_end-a) * (d-c)
        slope_start, slope_end = max(a, c), min(b, d)
        slope_area = max(0.0, slope_end-slope_start) * (d-(slope_start+slope_end)/2)
        u = self._rng.random() * (flat_area+slope_area)
        if u < flat_area:
            x = a + u/(d-c)
        else:
            v = u-flat_area
            h = d-slope_start
            x = slope_start + 2*v/(h+np.sqrt(max(0.0, h*h-2*v)))
        y = self._rng.uniform(max(c, x), d)
        return x, y

    def from_prior(self):
        if not self._native_seeded and self._seed_native is not None:
            self._seed_native()
            self._native_seeded = True
        values = self._rng.uniform(self._bounds[:, 0], self._bounds[:, 1])
        values[self._logscale] = np.exp(values[self._logscale])
        values[3], values[6] = self._centers_from_prior()
        return np.asarray(values, dtype=np.float64)

    def _heavy_step(self):
        return self._rng.normal() * 10.0**self._rng.uniform(-4.0, 0.5)

    def _single_coordinate(self, coords):
        j = int(self._rng.integers(self.num_params))
        lo, hi = self._bounds[j]
        if j == 3:
            hi = min(hi, coords[6])
        elif j == 6:
            lo = max(lo, coords[3])
        if hi <= lo:
            return -np.inf
        value = np.log(coords[j]) if self._logscale[j] else coords[j]
        # Symmetric heavy-tailed proposal in the uniform prior coordinate.
        value = lo + (value-lo + (hi-lo)*self._heavy_step()) % (hi-lo)
        coords[j] = np.exp(value) if self._logscale[j] else value
        return 0.0

    def _component_center_crossing(self, coords):
        """Reversible on the ordered quotient of an exchangeable product prior.

        Choose a physical component uniformly, move its center with a symmetric
        full-interval kernel, and sort whole triples. The reverse selects that
        same physical component (possibly with the other label), with the same
        1/2 probability. The ordered-prior normalization is the constant 2 and
        both the product density and proposal density are exchange invariant.
        Therefore the prior-times-proposal ratio is one, including crossings.
        Nonexchangeable component priors must not use this move.
        """
        if not self._exchangeable_components:
            raise ValueError("Component crossing requires identical component priors")
        j = 3+3*int(self._rng.integers(2))
        lo, hi = self.param_range[j]
        if self._rng.random() < 0.20:
            coords[j] = self._rng.uniform(lo, hi)
        else:
            coords[j] = lo + (coords[j]-lo+(hi-lo)*self._heavy_step()) % (hi-lo)
        if coords[3] > coords[6]:
            first = coords[2:5].copy()
            coords[2:5] = coords[5:8]
            coords[5:8] = first
            self.component_reorder_proposals += 1
        return 0.0

    def _scale_compensation(self, coords):
        # Translation on a rectangular log-coordinate torus preserves volume.
        # Its negative is equally probable, even when individual bounds wrap.
        indices = np.array([0, 2, 5])
        lo, hi = self._bounds[indices].T
        step = np.min(hi-lo)*self._heavy_step()
        values = lo+(np.log(coords[indices])-lo+step*np.array([1.0, -1.0, -1.0])) % (hi-lo)
        coords[indices] = np.exp(values)
        return 0.0

    def _joint_prior_coordinates(self, coords):
        # The Jacobian of the prior CDF cancels the physical prior density;
        # subset choice is state independent and each wrapped vector is symmetric.
        count = int(self._rng.integers(1, len(self._noncenter_indices)+1))
        indices = self._rng.choice(self._noncenter_indices, size=count, replace=False)
        lo, hi = self._bounds[indices].T
        values = coords[indices].copy()
        logscale = self._logscale[indices]
        values[logscale] = np.log(values[logscale])
        step = self._rng.normal(size=count)*10.0**self._rng.uniform(-4.0, 0.5)
        values = lo+((values-lo)/(hi-lo)+step) % 1.0*(hi-lo)
        values[logscale] = np.exp(values[logscale])
        coords[indices] = values
        return 0.0

    def _ou_ridge(self, coords):
        """Symmetric log shift (s,2s); log-uniform prior cancels Jacobian e^(3s)."""
        if not self._ou_ridge_enabled:
            raise ValueError("OU ridge requires log-uniform sigma_x and tau_x")
        step = self._heavy_step()
        indices = np.array([0, 1])
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = coords[indices]*np.exp(step*np.array([1.0, 2.0]))
        lo, hi = self.param_range[indices].T
        if not np.isfinite(candidate).all() or np.any(candidate < lo) or np.any(candidate > hi):
            self.ridge_boundary_rejections["ou_ridge"] += 1
            return -np.inf
        coords[indices] = candidate
        return 0.0

    def _slow_ridge(self, coords):
        """Uniform qW/qb and log-uniform Ts give logH=(-s)+(3s)=2s.

        The same correction applies to independent/shared marginalized GPs.
        For the implemented Mat32 kernel the derivative covariance is
        q_i*q_j/Ts**2, and this approximate ridge keeps that quantity fixed.
        """
        if not self._slow_ridge_enabled:
            raise ValueError("Slow ridge requires uniform qW/qb and log-uniform Ts")
        step = self._heavy_step()
        indices = np.array([8, 9, 10])
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = coords[indices]*np.exp(step)
        lo, hi = self.param_range[indices].T
        if not np.isfinite(candidate).all() or np.any(candidate < lo) or np.any(candidate > hi):
            self.ridge_boundary_rejections["slow_ridge"] += 1
            return -np.inf
        coords[indices] = candidate
        return float(2.0*step)

    def perturb(self, coords):
        u = self._rng.random()
        if u < 0.25 and self._exchangeable_components:
            self.proposal_counts["component_center_crossing"] += 1
            return self._component_center_crossing(coords)
        if 0.25 <= u < 0.35 and self._scale_compensation_enabled:
            self.proposal_counts["scale_compensation"] += 1
            return self._scale_compensation(coords)
        if 0.35 <= u < 0.45:
            self.proposal_counts["joint_prior_coordinates"] += 1
            return self._joint_prior_coordinates(coords)
        if 0.45 <= u < 0.50:
            self.proposal_counts["center_prior_refresh"] += 1
            coords[3], coords[6] = self._centers_from_prior()
            return 0.0
        if 0.50 <= u < 0.60 and self._ou_ridge_enabled:
            self.proposal_counts["ou_ridge"] += 1
            return self._ou_ridge(coords)
        if 0.60 <= u < 0.70 and self._slow_ridge_enabled:
            self.proposal_counts["slow_ridge"] += 1
            return self._slow_ridge(coords)
        self.proposal_counts["single_coordinate"] += 1
        return self._single_coordinate(coords)

    def log_likelihood(self, coords):
        self.likelihood_calls += 1
        coords = np.asarray(coords)
        if (not np.isfinite(coords).all() or
            np.any(coords < self.param_range[:, 0]) or
            np.any(coords > self.param_range[:, 1]) or coords[3] >= coords[6]):
            self.invalid_likelihood_calls += 1
            return -np.inf
        value = float(self._likelihood(dict(zip(self.names, coords))))
        if not np.isfinite(value):
            self.invalid_likelihood_calls += 1
            return -np.inf
        return value


def _native_seed_callback(seed: int):
    """Reset the actual CDNest GSL RNG after it is allocated by setup().

    The public Python wrapper has no seed option and otherwise uses time(NULL).
    One irrelevant -DBL_MAX floor-level tiebreaker predates the first callback;
    its raw-file bytes remain time dependent. Particle trajectories are seeded
    here. Native equal-weight resampling has a separate time seed; our primary
    posterior product uses the deterministic weights. No library is modified.
    """
    extension = importlib.import_module("cydnest.cydnest")
    library = ctypes.CDLL(extension.__file__)
    pointer = ctypes.c_void_p.in_dll(library, "dnest_gsl_r")
    setter = library.gsl_rng_set
    setter.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    setter.restype = None
    def apply():
        if not pointer.value:
            raise RuntimeError("CDNest RNG is unavailable at from_prior callback")
        setter(pointer.value, ctypes.c_ulong(seed))
    return apply, str(extension.__file__)


def weighted_quantile(values, weights, probabilities=(0.1586552539, 0.5, 0.8413447461)):
    values, weights = np.asarray(values), np.asarray(weights)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[good], weights[good]
    if not len(values):
        raise ValueError("No positive posterior weight")
    order = np.argsort(values)
    return np.interp(probabilities, np.cumsum(weights[order])/np.sum(weights), values[order])


def read_weighted_posterior(output_dir, names):
    """Read C postprocessor schema: weights.txt = logX, logL, scaled weight.

    Do not call cydnest.postprocess(): its Python implementation overwrites
    weights.txt with a different one-column format. No equal-weight assumption
    is made about the raw DNest sample file.
    """
    folder = Path(output_dir)
    sample = np.loadtxt(folder/"sample.txt", ndmin=2)
    info = np.loadtxt(folder/"sample_info.txt", ndmin=2)
    weights = np.loadtxt(folder/"weights.txt", ndmin=2)
    if sample.shape[1] != len(names) or weights.shape[1] != 3:
        raise ValueError("Unexpected native CDNest sample/weight schema")
    if len(sample) != len(weights) or len(sample) != len(info):
        raise ValueError("Native CDNest output rows do not align")
    if not np.isfinite(sample).all() or not np.isfinite(weights[:, 1:]).all():
        raise ValueError("Non-finite native CDNest output")
    if np.any(weights[:, 2] < 0) or weights[:, 2].sum() <= 0:
        raise ValueError("Invalid native posterior weights")
    if not np.allclose(weights[:, 1], info[:, 1], rtol=1e-6, atol=1e-6):
        raise ValueError("Weight and sample-info likelihood columns do not align")
    # CDNest's C interpolation divides by N-1 for a level containing N points.
    # At N=1 logX is NaN, but the weight uses the two level boundaries and remains
    # valid. Recover only that known diagnostic-coordinate edge case, preserving
    # the C-computed weights and the unmodified original file.
    repaired = 0
    if not np.isfinite(weights[:, 0]).all():
        levels = np.loadtxt(folder/"levels.txt", ndmin=2)
        assigned = np.minimum(info[:, 0].astype(int), len(levels)-1)
        sandwich = np.maximum(assigned, np.searchsorted(levels[:, 1], info[:, 1], side="left")-1)
        for row in np.flatnonzero(~np.isfinite(weights[:, 0])):
            lev = sandwich[row]
            if np.count_nonzero(sandwich == lev) != 1:
                raise ValueError("Unexpected non-finite logX outside the native singleton-level edge case")
            left = -np.inf if lev == len(levels)-1 else levels[lev+1, 0]
            weights[row, 0] = np.logaddexp(levels[lev, 0], left)-np.log(2.0)
            repaired += 1
    w = weights[:, 2]/weights[:, 2].sum()
    return {"samples": sample, "weights": w, "log_likelihood": weights[:, 1],
            "log_prior_mass": weights[:, 0], "sample_info": info,
            "singleton_logx_repairs": repaired}


def summarize_run(output_dir, names, logz=None, metadata=None):
    folder = Path(output_dir)
    post = read_weighted_posterior(folder, names)
    sample, w = post["samples"], post["weights"]
    levels = np.loadtxt(folder/"levels.txt", ndmin=2)
    positive = w > 0
    summary = {name: dict(zip(("q16", "q50", "q84"),
                             map(float, weighted_quantile(sample[:, j], w))))
               for j, name in enumerate(names)}
    diagnostics = {
        "raw_sample_count": len(sample), "positive_weight_count": int(positive.sum()),
        "weight_ess": float(1/np.sum(w*w)),
        "weight_entropy_ess": float(np.exp(-np.sum(w[positive]*np.log(w[positive])))),
        "weight_ess_is_not_autocorrelation_ess": True,
        "native_singleton_logx_coordinates_repaired": post["singleton_logx_repairs"],
        "native_equal_weight_resample_is_time_seeded": True,
        "max_posterior_weight": float(w.max()), "number_levels": len(levels),
        "last_level_log_prior_mass": float(levels[-1, 0]),
        "last_level_threshold_log_likelihood": float(levels[-1, 1]),
        "max_sample_log_likelihood": float(np.max(post["log_likelihood"])),
        "posterior_weight_above_last_level": float(w[post["log_likelihood"] >= levels[-1, 1]].sum()),
        "raw_log_evidence": None if logz is None else float(logz),
        "convergence_status": "not_established_fixed_budget",
        "evidence_for_model_comparison": False,
        "caveat": "Fixed-budget exploratory run. Weight ESS and completed saves do not establish mixing, convergence, or reliable evidence. Offset priors must also be consistent for evidence comparison.",
    }
    if levels.shape[1] >= 5:
        tries = float(np.sum(levels[:, 4]))
        diagnostics["level_acceptance_fraction"] = float(np.sum(levels[:, 3])/tries) if tries else None
    if metadata:
        diagnostics.update(metadata)
    np.savetxt(folder/"weighted_posterior.csv",
               np.column_stack((sample, post["log_likelihood"], post["log_prior_mass"], w)),
               delimiter=",", header=",".join((*names, "log_likelihood", "log_prior_mass", "weight")),
               comments="")
    (folder/"posterior_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (folder/"diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
    return {"summary": summary, "diagnostics": diagnostics,
            "weighted_posterior": str(folder/"weighted_posterior.csv")}


def run_dnest(log_likelihood, output_dir, mode="independent", priors=None,
              budget: SamplerBudget | Mapping | None = None, seed=12345):
    """Run one genuine CDNest job; invoke in a fresh process per job.

    Each MPI rank must call this function. Outputs are written only by rank 0.
    The native wrapper contains global callbacks and is not thread-safe.
    """
    import cydnest
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if not isinstance(seed, (int, np.integer)) or seed < 0 or seed > 2**32-1:
        raise ValueError("seed must be a uint32 integer")
    budget = SamplerBudget() if budget is None else (SamplerBudget(**budget) if isinstance(budget, Mapping) else budget)
    folder = Path(output_dir).resolve()
    # cydnest.pyx uses 200-byte C arrays including filename suffixes.
    if len(str(folder).encode()) > 160:
        raise ValueError("CDNest output path exceeds safe native buffer length (160 bytes)")
    if (folder/"sample.txt").exists():
        raise FileExistsError(f"Refusing to overwrite existing DNest run: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    native_seed, library_path = _native_seed_callback(int(seed)+rank)
    model = DNestModel(log_likelihood, mode, priors, int(seed)+rank, native_seed)
    run_config = {"mode": mode, "parameter_names": model.names, "priors": model.priors,
                  "center_prior": "uniform on bounded rectangle conditioned on mu0 < mu1",
                  "proposals": model.proposal_config(),
                  "budget": asdict(budget), "seed": int(seed), "mpi_size": comm.Get_size(),
                  "native_library": library_path,
                  "seed_policy": "NumPy Generator seed+rank and native GSL reset at first from_prior; native equal-weight resampling and initial floor-level tiebreaker remain time dependent; weighted posterior is primary",
                  "likelihood_contract": "physical parameter dict; proper bounds handled by sampler; no empirical driver-only prior"}
    if rank == 0:
        (folder/"sampler_config.json").write_text(json.dumps(run_config, indent=2)+"\n")
    comm.Barrier()
    start = time.time()
    sampler = cydnest.sampler(model, sample_dir=str(folder), **asdict(budget))
    logz = sampler.run()
    elapsed = time.time()-start
    counts = comm.gather({"rank": rank, "likelihood_calls": model.likelihood_calls,
                          "invalid_likelihood_calls": model.invalid_likelihood_calls,
                          "proposal_counts": model.proposal_counts,
                          "component_reorder_proposals": model.component_reorder_proposals,
                          "ridge_boundary_rejections": model.ridge_boundary_rejections,
                          "native_seed_applied": model._native_seeded}, root=0)
    result = None
    if rank == 0:
        run_config["proposal_attempt_counts_by_rank"] = [
            {"rank": item["rank"], "proposal_counts": item["proposal_counts"],
             "component_reorder_proposals": item["component_reorder_proposals"],
             "ridge_boundary_rejections": item["ridge_boundary_rejections"]}
            for item in counts]
        (folder/"sampler_config.json").write_text(json.dumps(run_config, indent=2)+"\n")
        result = summarize_run(folder, model.names, logz,
                               {"elapsed_seconds": elapsed, "rank_diagnostics": counts,
                                "proposal_version": PROPOSAL_VERSION,
                                "proposal_config": model.proposal_config(),
                                "max_num_saves": budget.max_num_saves})
    return comm.bcast(result, root=0)
