"""Mrk 817 Fig. 14/15-style DRW + top-hat residual fits.

Uses the official JAVELIN covariance when the isolated task runtime is available,
with an independently checked marginal Gaussian likelihood and emcee 3.
Explicit priors/sampler differ from the unpublished original run configuration.
The marginal likelihood integrates a separate constant mean for each series.
"""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = ROOT / "outputs" / "lag_env" / "bin" / "python"
VENDOR = ROOT / "outputs" / "lag_vendor" / "javelin"
if __name__ == "__main__" and ENV.exists() and Path(sys.prefix).name != "lag_env":
    os.execv(str(ENV), [str(ENV), str(Path(__file__).resolve()), *sys.argv[1:]])
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.integrate import quad
from threadpoolctl import threadpool_limits

_SPEAR = None
_ACTIVE = None
BOUNDS = np.array([[np.log(.02), np.log(20.)], [np.log(1.), np.log(10000.)],
                   [-50., 80.], [.01, 40.], [.001, 1.5]])
NAMES = np.array(["log10_sigma", "log10_tau_days", "lag_days", "width_days", "scale"])
COMMIT = "ad4168bc987daa7e493ceb15340d422f5380449e"


def load_official():
    global _SPEAR
    if VENDOR.exists():
        sys.path.insert(0, str(VENDOR))
        from javelin.spear import spear
        _SPEAR = spear
        return True
    return False


def box_cross(dt, tau, width):
    d = np.abs(np.asarray(dt, dtype=float))
    a = d / tau
    z = width / tau
    outside = np.exp(-np.maximum(a-z/2., 0.)) * (-np.expm1(-z))/z
    inside = (-np.expm1(-(z/2.+a))-np.expm1(-(z/2.-a)))/z
    return np.where(d >= width/2., outside, inside)


def box_auto(dt, tau, width):
    d = np.abs(np.asarray(dt, dtype=float))
    a = d/tau
    z = width/tau
    outside = np.exp(-np.maximum(a-z, 0.)) * (-np.expm1(-z)/z)**2
    # Clipping the interior expression avoids overflow outside its domain.
    ai = np.minimum(a, z)
    inside = (2*(z-ai)+np.expm1(-z-ai)+np.expm1(-z+ai)
              -2*np.expm1(-ai))/z**2
    return np.where(d >= width, outside, inside)


def analytic_cov(t1, k1, t2, k2, p):
    sigma, tau = np.exp(p[:2])
    lag, width, scale = p[2:]
    dt = np.asarray(t1)[:, None]-np.asarray(t2)[None, :]
    k1, k2 = np.asarray(k1)[:, None], np.asarray(k2)[None, :]
    out = np.exp(-np.abs(dt)/tau)
    out = np.where((k1 == 0) & (k2 == 1),
                   scale*box_cross(dt+lag, tau, width), out)
    out = np.where((k1 == 1) & (k2 == 0),
                   scale*box_cross(dt-lag, tau, width), out)
    out = np.where((k1 == 1) & (k2 == 1),
                   scale**2*box_auto(dt, tau, width), out)
    return sigma**2*out


def covariance(t1, k1, t2, k2, p, symmetric=False):
    if _SPEAR is None:
        return analytic_cov(t1, k1, t2, k2, p)
    sigma, tau = np.exp(p[:2])
    lag, width, scale = p[2:]
    return np.asarray(_SPEAR(
        np.asarray(t1, float), np.asarray(t2, float),
        np.asarray(k1, int)+1, np.asarray(k2, int)+1,
        sigma, tau, np.array([0., lag]), np.array([0., width]),
        np.array([1., scale]), symm=symmetric), dtype=float)


class EchoGP:
    def __init__(self, t, driver, response, error):
        self.n = len(t)
        self.unit = max(float(np.std(driver)), float(np.median(error)), 1e-100)
        self.centers = np.array([np.mean(driver), np.mean(response)])
        self.time = np.r_[t, t].astype(float)
        self.kind = np.repeat([0, 1], self.n)
        self.y = np.r_[driver-self.centers[0], response-self.centers[1]]/self.unit
        self.error = np.tile(error/self.unit, 2)
        self.B = np.eye(2)[self.kind]
        self.rhs = np.column_stack([self.y, self.B])

    def factor(self, p):
        c = covariance(self.time, self.kind, self.time, self.kind, p, True)
        c.flat[::len(c)+1] += self.error**2
        cf = cho_factor(c, lower=True, check_finite=False)
        sol = cho_solve(cf, self.rhs, check_finite=False)
        ci_y, ci_b = sol[:, 0], sol[:, 1:]
        q = self.B.T @ ci_b
        qf = cho_factor(q, lower=True, check_finite=False)
        beta = cho_solve(qf, self.B.T @ ci_y, check_finite=False)
        alpha = ci_y-ci_b@beta
        logdet = 2*np.log(np.diag(cf[0])).sum()
        logdetq = 2*np.log(np.diag(qf[0])).sum()
        value = -.5*(self.y@alpha+logdet+logdetq+(len(self.y)-2)*np.log(2*np.pi))
        return value, cf, ci_b, qf, beta, alpha

    def __call__(self, p):
        if not np.all((p > BOUNDS[:, 0]) & (p < BOUNDS[:, 1])):
            return -np.inf
        try:
            return self.factor(p)[0]
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return -np.inf

    def predict(self, p, grid):
        _, cf, ci_b, qf, beta, alpha = self.factor(p)
        ts = np.tile(grid, 2)
        ks = np.repeat([0, 1], len(grid))
        bs = np.eye(2)[ks]
        cross = covariance(ts, ks, self.time, self.kind, p)
        means = bs@beta+cross@alpha
        cinv_cross = cho_solve(cf, cross.T, check_finite=False)
        sigma = np.exp(p[0])
        priorvar = np.where(ks == 0, sigma**2,
                            sigma**2*p[4]**2*float(box_auto(0., np.exp(p[1]), p[3])))
        h = bs-cross@ci_b
        variance = priorvar-np.sum(cross*cinv_cross.T, axis=1)
        variance += np.sum(h*cho_solve(qf, h.T, check_finite=False).T, axis=1)
        means = means*self.unit+self.centers[ks]
        return means.reshape(2, -1), np.maximum(variance, 0).reshape(2, -1)*self.unit**2


def worker(p):
    return _ACTIVE(p)


def physical_samples(internal, unit):
    result = np.array(internal, copy=True)
    result[..., 0] = (result[..., 0]+np.log(unit))/np.log(10)
    result[..., 1] /= np.log(10)
    return result


def selftest(out):
    errors = []
    for width in [.01, .5, 4., 30.]:
        for dt in [0., .2*width, width, 50.]:
            tau = 37.
            cross = quad(lambda u: np.exp(-abs(dt-u)/tau),
                         -width/2, width/2, points=[0.] if dt == 0 else None,
                         epsabs=1e-10)[0]/width
            auto = quad(lambda u: (width-abs(u))*np.exp(-abs(dt-u)/tau),
                        -width, width, points=[0.], epsabs=1e-10)[0]/width**2
            errors += [abs(float(box_cross(dt, tau, width))-cross),
                       abs(float(box_auto(dt, tau, width))-auto)]
    result = {"analytic_quadrature_max_absolute_error": max(errors)}
    assert max(errors) < 2e-7, result
    rng = np.random.default_rng(817)
    t = np.sort(rng.uniform(0, 230, 85))
    p = np.array([np.log(1.2), np.log(35.), 12.5, 4., .4])
    times = np.tile(t, 2)
    kinds = np.repeat([0, 1], len(t))
    ca = analytic_cov(times, kinds, times, kinds, p)
    if _SPEAR is not None:
        co = covariance(times, kinds, times, kinds, p, True)
        result["official_covariance_max_absolute_error"] = float(np.max(abs(ca-co)))
        np.testing.assert_allclose(ca, co, rtol=1e-7, atol=2e-8)
    y = rng.multivariate_normal(np.zeros(len(times)), ca+np.eye(len(times))*.04**2)
    gp = EchoGP(t, y[:len(t)], y[len(t):], np.full(len(t), .04))
    pn = p.copy()
    pn[0] -= np.log(gp.unit)
    if _SPEAR is not None:
        from javelin.zylc import LightCurve
        from javelin.lcmodel import lnlikefn_spear
        lc = LightCurve([np.column_stack([t, gp.y[:len(t)], gp.error[:len(t)]]),
                         np.column_stack([t, gp.y[len(t):], gp.error[len(t):]])])
        official = float(lnlikefn_spear(lc, np.exp(pn[0]), np.exp(pn[1]),
                                        np.array([pn[2]]),
                                        np.array([pn[3]]),
                                        np.array([pn[4]])))
        ours = gp(pn)+.5*(len(times)-2)*np.log(2*np.pi)
        result["official_loglike_absolute_error"] = abs(official-ours)
        assert abs(official-ours) < 1e-5, result
    fit = minimize(lambda x: -gp(x), pn+np.array([0., 0., 1., .3, .03]),
                   method="Nelder-Mead", options={"maxiter": 400})
    result["synthetic_true"] = physical_samples(pn, gp.unit).tolist()
    result["synthetic_recovered_map"] = physical_samples(fit.x, gp.unit).tolist()
    result["synthetic_lag_error_days"] = float(abs(fit.x[2]-p[2]))
    assert abs(fit.x[2]-p[2]) < 2., result
    (out/"lag_selftest.json").write_text(json.dumps(result, indent=2))
    print("SELFTEST", json.dumps(result), flush=True)
    return result


def run_case(name, t, driver, residual, error, args, out):
    import emcee
    from multiprocessing import get_context
    global _ACTIVE
    started = time.monotonic()
    gp = EchoGP(t, driver, residual, error)
    _ACTIVE = gp
    scale = np.clip(np.std(residual)/gp.unit, .03, 1.)
    seeds = [np.array([0., np.log(100.), lag, 2., scale])
             for lag in np.unique(np.r_[np.linspace(-48, 78, 19), 0., 21.])]
    ranked = sorted(seeds, key=gp, reverse=True)
    optima = []
    for seed in ranked[:4]:
        fit = minimize(lambda p: -gp(p), seed, method="Nelder-Mead",
                       options={"maxiter": 450, "xatol": 1e-4, "fatol": 1e-4})
        optima.append((gp(fit.x), fit.x))
    bestlp, best = max(optima, key=lambda pair: pair[0])
    tick = time.monotonic()
    for _ in range(10):
        gp(best)
    seconds_per_call = (time.monotonic()-tick)/10
    print(name, "MAP start", physical_samples(best, gp.unit), "logp", bestlp, flush=True)
    print(name, "likelihood_seconds_per_call", seconds_per_call, flush=True)
    rng = np.random.default_rng(args.seed+(1 if name == "baseline" else 2))
    starts = []
    while len(starts) < args.walkers:
        candidate = best+rng.normal(size=5)*[.025, .04, .10, .08, .004]
        if np.isfinite(gp(candidate)):
            starts.append(candidate)
    pool = get_context("fork").Pool(args.workers) if args.workers > 1 else None
    try:
        sampler = emcee.EnsembleSampler(args.walkers, 5, worker, pool=pool)
        sampler.random_state = np.random.RandomState(args.seed).get_state()
        state = sampler.run_mcmc(np.asarray(starts), args.burn, progress=False)
        sampler.reset()
        iterations = 0
        converged = False
        tau = np.full(5, np.nan)
        while iterations < args.max_steps:
            amount = min(250, args.max_steps-iterations)
            state = sampler.run_mcmc(state, amount, progress=False)
            iterations += amount
            if iterations >= 100:
                tau = sampler.get_autocorr_time(tol=0)
            print(name, "steps", iterations, "seconds", round(time.monotonic()-started, 1),
                  "max_tau", round(float(np.max(tau)), 1), flush=True)
            if iterations >= args.steps and np.all(iterations > 50*tau):
                converged = True
                break
        chain = sampler.get_chain()
        logp = sampler.get_log_prob()
        acceptance = sampler.acceptance_fraction
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    flat = chain.reshape(-1, 5)
    half = len(chain)//2
    split = np.concatenate([chain[:half], chain[-half:]], axis=1)
    within = np.mean(np.var(split, axis=0, ddof=1), axis=0)
    between = half*np.var(np.mean(split, axis=0), axis=0, ddof=1)
    split_rhat = np.sqrt(((half-1)*within/half+between/half)/within)
    samples = physical_samples(flat, gp.unit)
    quantiles = np.quantile(samples, [.16, .5, .84], axis=0)
    grid = np.linspace(t.min(), t.max(), 420)
    indices = rng.choice(len(flat), min(20, len(flat)), replace=False)
    predictions = [gp.predict(flat[i], grid) for i in indices]
    means = np.array([v[0] for v in predictions])
    variances = np.array([v[1] for v in predictions])
    pred_mean = means.mean(axis=0)
    pred_std = np.sqrt(np.maximum((variances+means**2).mean(axis=0)-pred_mean**2, 0.))
    summary = {
        "parameters": {str(k): {"median": float(q[1]), "minus": float(q[1]-q[0]),
                                "plus": float(q[2]-q[1])}
                       for k, q in zip(NAMES, quantiles.T)},
        "n_driver": len(t), "n_response": len(t), "flux_normalization": gp.unit,
        "iterations_after_burn": len(chain), "burn_iterations": args.burn,
        "walkers": args.walkers, "acceptance_mean": float(np.mean(acceptance)),
        "integrated_autocorrelation_iterations": tau.tolist(),
        "approximate_ensemble_ess": (len(flat)/tau).tolist(),
        "split_walker_rhat_diagnostic": split_rhat.tolist(),
        "rhat_caution": "Ensemble walkers are correlated, so this is a warning diagnostic, not independent-chain proof.",
        "likelihood_seconds_per_call": seconds_per_call,
        "converged_50tau": bool(converged),
        "local_optima": [{"logp": float(v), "parameters": physical_samples(p, gp.unit).tolist()}
                         for v, p in optima],
        "runtime_seconds": time.monotonic()-started,
    }
    data = {name+"_samples": samples, name+"_chain_internal": chain,
            name+"_log_prob": logp, name+"_prediction_time": grid,
            name+"_driver_mean": pred_mean[0], name+"_driver_std": pred_std[0],
            name+"_response_mean": pred_mean[1], name+"_response_std": pred_std[1]}
    np.savez_compressed(out/("lag_"+name+"_results.npz"), **data)
    (out/("lag_"+name+"_summary.json")).write_text(json.dumps(summary, indent=2))
    print("COMPLETE", name, json.dumps(summary["parameters"]), flush=True)
    return data, summary


def make_figures(out, pdf=None, figure_dir=None):
    """Append available Fig.14/15 analogues; safe to import in the astro env."""
    import matplotlib.pyplot as plt
    out = Path(out)
    figure_dir = Path(figure_dir) if figure_dir is not None else out
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = out/"lag_results.npz"
    if not path.exists():
        return []
    summary_path = out/"lag_summary.json"
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    files = []
    with np.load(path, allow_pickle=False) as a:
        for name, number, title in [
                ("blr_0p5", 14, "Injected BLR residual, f_BLR = 0.5"),
                ("baseline", 15, "Original-data residual")]:
            if name+"_samples" not in a:
                continue
            fig = plt.figure(figsize=(12.2, 5.0))
            gs = fig.add_gridspec(2, 4, width_ratios=[1.7, 1.7, 1., 1.],
                                  hspace=.42, wspace=.60)
            grid = a[name+"_prediction_time"]
            for row, obs, part, ylabel in [
                (0, a["driver_flux"], "driver", "Original Swift U flux"),
                (1, a[name+"_residual"], "response", "U residual flux")]:
                ax = fig.add_subplot(gs[row, :2])
                ax.errorbar(a["t"], obs*10., yerr=a["error"]*10., fmt=".", ms=2.4,
                            color=".2", ecolor=".65", elinewidth=.5, alpha=.65)
                mean = a[name+"_"+part+"_mean"]*10.
                std = a[name+"_"+part+"_std"]*10.
                ax.fill_between(grid, mean-std, mean+std, color="tab:red", alpha=.18)
                ax.plot(grid, mean, color="tab:red", lw=1.)
                ax.set_ylabel(ylabel + "\n" + r"[$10^{-16}$ erg s$^{-1}$ cm$^{-2}$ Angstrom$^{-1}$]",
                              fontsize=9)
                if row:
                    ax.set_xlabel("THJD (days)")
                else:
                    ax.set_title(title, fontsize=11)
                ax.tick_params(direction="in", top=True, right=True)
            labels = [r"$\log_{10}\sigma$ ($10^{-16}$ flux units)", r"$\log_{10}\tau$ (days)",
                      "Lag (days)", "Response scale"]
            samples = a[name+"_samples"]
            for k, (column, label) in enumerate(zip([0, 1, 2, 4], labels)):
                ax = fig.add_subplot(gs[k//2, 2+k%2])
                vals = samples[:, column]+(1. if column == 0 else 0.)
                ax.hist(vals, bins=45, color=".6", density=True)
                q = np.quantile(vals, [.16, .5, .84])
                for j, v in enumerate(q):
                    ax.axvline(v, color="tab:red", ls="-" if j == 1 else "--", lw=1)
                ax.set_xlabel(label, fontsize=9)
                ax.set_yticks([])
                ax.set_title(f"{q[1]:.3g} +{q[2]-q[1]:.2g} / -{q[1]-q[0]:.2g}", fontsize=9)
                ax.tick_params(direction="in", top=True)
            case_info = summaries.get("cases", {}).get(name, {})
            convergence = ("50-tau length criterion passed" if case_info.get("converged_50tau", False)
                           else "PRELIMINARY: 50-tau convergence criterion NOT met")
            fig.suptitle(f"Figure {number} analogue: JAVELIN kernel + emcee; reconstruction priors\n{convergence}",
                         fontsize=9, y=1.015)
            filename = figure_dir/f"fig{number}_residual_{name}.png"
            fig.savefig(filename, dpi=170, bbox_inches="tight")
            if pdf is not None:
                pdf.savefig(fig, bbox_inches="tight")
            files.append(str(filename))
            plt.close(fig)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"outputs"/"fits.npz")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs")
    parser.add_argument("--walkers", type=int, default=24)
    parser.add_argument("--burn", type=int, default=600)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8172024)
    parser.add_argument("--selftest-only", action="store_true")
    parser.add_argument("--skip-selftest", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Refresh existing case summaries/figures without fitting missing cases.")
    args = parser.parse_args()
    if args.summarize_only:
        args.reuse = True
    args.output.mkdir(parents=True, exist_ok=True)
    official = load_official()
    with threadpool_limits(limits=1):
        tests = None if args.skip_selftest else selftest(args.output)
        if args.selftest_only:
            return
        with np.load(args.input, allow_pickle=False) as a:
            selected = a["band"].astype(str) == "Swift_U"
            values = {k: a[k][selected].astype(float) for k in
                      ["t", "flux", "error", "pred_baseline", "pred_blr_0p5", "flux_blr_0p5"]}
        keep = np.logical_and.reduce([np.isfinite(v) for v in values.values()])
        keep &= values["error"] > 0
        order = np.argsort(values["t"][keep])
        values = {k: v[keep][order] for k, v in values.items()}
        t, driver, error = values["t"], values["flux"], values["error"]
        residuals = {"baseline": driver-values["pred_baseline"],
                     "blr_0p5": values["flux_blr_0p5"]-values["pred_blr_0p5"]}
        results = {"parameter_names": NAMES, "t": t, "driver_flux": driver, "error": error,
                   **{k+"_residual": v for k, v in residuals.items()}}
        summary = {
            "input": str(args.input.resolve()), "n_swift_u": len(t),
            "backend": "official JAVELIN covariance + SciPy marginal likelihood + emcee 3"
                       if official else "analytic DRW/top-hat covariance + emcee 3",
            "javelin_source": "https://github.com/nye17/javelin",
            "javelin_commit": COMMIT if official else None,
            "software_versions": {"python": sys.version.split()[0],
                                  **{k: version(k) for k in ["numpy", "scipy", "emcee"]}},
            "seed": args.seed,
            "sampler": "emcee 3 default StretchMove(a=2); four local optimization starts",
            "stored_flux_unit": "1e-15 erg/s/cm2/Angstrom; log10_sigma is in this unit",
            "plot_flux_unit": "1e-16 erg/s/cm2/Angstrom (stored flux multiplied by 10)",
            "selftest": tests,
            "prior": {"density": "uniform in ln(sigma/driver_std), ln(tau), lag, width, scale",
                      "bounds_internal": BOUNDS.tolist(),
                      "tau_lag_width_time_unit": "observed days",
                      "width_definition": "full width of centered top-hat",
                      "response_scale": "positive"},
            "method_notes": [
                "Separate constant flux means are analytically marginalized.",
                "Original Swift U drives both baseline and injected-residual fits.",
                "Residual observational errors reuse U errors; map uncertainty and shared-data covariance are not propagated.",
                "No double use of a continuum posterior as an additional prior.",
                "Exact original priors and sampler settings were not published; this is an explicit reconstruction.",
                "Near-zero lag of data-derived residuals is not an independent physical-component identification.",
                "Posterior predictive bands mix Gaussian conditional moments across 20 posterior draws."],
            "paper_reference": {"blr_0p5": {"lag_days": [20.1, .4, .4], "scale": [.52, .02, .01]},
                                "baseline": {"lag_days": [.05, .10, .15], "scale": [.20, .01, .01]}},
            "cases": {},
        }
        for name in ["blr_0p5", "baseline"]:
            cached = args.output/("lag_"+name+"_results.npz")
            cached_summary = args.output/("lag_"+name+"_summary.json")
            if args.summarize_only and not (cached.exists() and cached_summary.exists()):
                continue
            if args.reuse and cached.exists() and cached_summary.exists():
                with np.load(cached, allow_pickle=False) as a:
                    case_data = {k: a[k] for k in a.files}
                case_summary = json.loads(cached_summary.read_text())
            else:
                case_data, case_summary = run_case(name, t, driver, residuals[name],
                                                   error, args, args.output)
            internal = case_data[name+"_chain_internal"].reshape(-1, 5)
            span = BOUNDS[:, 1]-BOUNDS[:, 0]
            case_summary["prior_boundary_fraction_1percent"] = {
                str(k): {"lower": float(np.mean(internal[:, j] < BOUNDS[j, 0]+.01*span[j])),
                         "upper": float(np.mean(internal[:, j] > BOUNDS[j, 1]-.01*span[j]))}
                for j, k in enumerate(NAMES)}
            case_summary["stored_flux_unit"] = "1e-15 erg/s/cm2/Angstrom"
            (args.output/("lag_"+name+"_summary.json")).write_text(json.dumps(case_summary, indent=2))
            results.update(case_data)
            summary["cases"][name] = case_summary
            np.savez_compressed(args.output/"lag_results.npz", **results)
            (args.output/"lag_summary.json").write_text(json.dumps(summary, indent=2))
            make_figures(args.output)
    print("LAG_RESULTS", args.output/"lag_results.npz", flush=True)


if __name__ == "__main__":
    main()
