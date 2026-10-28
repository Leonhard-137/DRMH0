#!/usr/bin/env python3
"""PSD-only calibration of a fixed disk+BLR+shared-slow generative model.

Search and retain multiple candidate solutions; never invoke MICA.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from psd_match_forward import ForwardModel

HERE = Path(__file__).resolve().parent
OUT = HERE / 'psd_match_817'
NAMES = ['tauI', 'tauS', 'AW', 'AU', 'BW', 'BU']
BOUNDS = np.array([[.1, 100.], [5., 1000.], [.001, 1.5], [.001, 1.5],
                   [.0001, 3.], [.0001, 3.]])
LOG_BOUNDS = np.log10(BOUNDS)
SEARCH_SEED = 8171003
VALIDATION_SEED = 8171004
REFERENCE = np.array([2., 55.3735295468, .10, .10, .22, .22])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def status(stage, **extra):
    OUT.mkdir(exist_ok=True)
    value = dict(stage=stage, unix_time=time.time(), **extra)
    write_json(OUT / 'status.json', value)
    print(json.dumps(value), flush=True)


def score(logpsd, target, weights):
    return float(np.mean(((np.mean(logpsd, axis=0) - target) / weights)**2))


def calibration(model):
    logs = np.log(model.evaluate(REFERENCE))
    weights = np.maximum(np.std(logs, axis=0, ddof=1), .15)
    target = np.log(model.observed_psd)
    np.savez_compressed(OUT / 'fixed_metric.npz', weights=weights, target=target,
                        reference_params=REFERENCE)
    return target, weights


def search():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'search_complete.json').exists():
        print('Existing completed search retained; use validate for subsequent stages.', flush=True)
        return
    config = dict(
        parameter_names=NAMES, parameter_bounds=BOUNDS.tolist(),
        model='fractional W2=AW*(DW*I)+BW*S+nW; '
              'fractional U=AU*((0.9*DU+0.1*BLR)*I)+BU*S+nU',
        latent_processes='independent, unit stationary variance OU I and Matern32 S; '
                         'same I and same S in both bands; S is added directly',
        fixed_kernel='Historical Starkey disk; BLR gamma k=2 onset=6 d, theta=7 d; '
                     'U disk:BLR areas=9:1',
        observed_inputs=str(HERE.parents[1] / 'mica_smooth/data/original'),
        metric='mean squared difference of binned log PSD means, standardized '
               'by fixed reference-model log PSD standard deviations; not chi-square likelihood',
        search_realizations=64, rerank_realizations=256, validation_realizations=500,
        search_seed=SEARCH_SEED, validation_seed=VALIDATION_SEED,
        physical_selection='Deferred to user. No fast-driver or slow-dominance penalty; '
                           'bounds and boundary proximity reported.',
        selection_rule='Independent joint max-band PSD discrepancy <= its simulated '
                       '95th percentile. Conditional compatibility screen, not a posterior '
                       'or global post-selection p-value.',
        generated_at_unix=time.time())
    write_json(OUT / 'search_config.json', config)
    model = ForwardModel(seed=SEARCH_SEED, nreal=64)
    target, weights = calibration(model)
    evaluated, profile, counter = [], [], [0]
    started = time.monotonic()

    def objective(logparams):
        params = 10.**np.asarray(logparams)
        value = score(np.log(model.evaluate(params)), target, weights)
        counter[0] += 1
        evaluated.append([value, *params.tolist()])
        if counter[0] % 200 == 0:
            status('search', evaluations=counter[0], best=min(x[0] for x in evaluated),
                   elapsed_seconds=time.monotonic()-started)
        return value

    status('global_search', bounds=BOUNDS.tolist())
    result = differential_evolution(objective, LOG_BOUNDS, popsize=6, maxiter=55,
                                    tol=.015, seed=SEARCH_SEED, polish=False,
                                    updating='immediate', workers=1)
    global_candidates = [[float(e), *(10.**p).tolist()]
                         for e, p in zip(result.population_energies, result.population)]
    write_json(OUT / 'global_search.json', dict(
        success=bool(result.success), message=str(result.message), nfev=int(result.nfev),
        best=[float(result.fun), *(10.**result.x).tolist()], population=global_candidates))
    # Profiles retain different time-scale solutions instead of retaining only
    # one converged DE population. Optimize four amplitudes at each fixed pair.
    tau_i_grid = [.1, .25, .5, 1., 2., 4., 8., 16., 32., 64., 100.]
    tau_s_grid = [5., 10., 20., 40., 80., 160., 320., 640., 1000.]
    for i, ti in enumerate(tau_i_grid):
        for j, ts in enumerate(tau_s_grid):
            fixed = np.log10([ti, ts])
            # Materialize the cached simulation components only once per pair.
            model.components(ti, ts)
            def amp_objective(logamps):
                return objective(np.r_[fixed, logamps])
            fit = differential_evolution(amp_objective, LOG_BOUNDS[2:], popsize=5,
                                         maxiter=40, tol=.01, polish=False,
                                         seed=SEARCH_SEED + 100*i + j,
                                         x0=result.x[2:])
            row = [float(fit.fun), ti, ts, *(10.**fit.x).tolist()]
            profile.append(row)
            status('time_scale_profiles', completed=len(profile), total=len(tau_i_grid)*len(tau_s_grid),
                   tauI=ti, tauS=ts, score=fit.fun,
                   elapsed_seconds=time.monotonic()-started)
    pool = profile + global_candidates
    # Remove numerically identical parameter points, retaining the lower score.
    unique = {}
    for row in sorted(pool):
        key = tuple(np.round(np.log10(row[1:]), 4))
        unique.setdefault(key, row)
    candidates = list(unique.values())
    np.savetxt(OUT / 'search_evaluations.csv', np.asarray(evaluated), delimiter=',',
               header='score,'+','.join(NAMES), comments='')
    np.savetxt(OUT / 'candidate_pool.csv', np.asarray(candidates), delimiter=',',
               header='score,'+','.join(NAMES), comments='')
    write_json(OUT / 'search_complete.json', dict(
        evaluations=counter[0], candidates=len(candidates),
        elapsed_seconds=time.monotonic()-started, global_optimizer_converged=bool(result.success)))
    status('search_complete', evaluations=counter[0], candidates=len(candidates))


def candidate_diagnostics(params, logs, center, target, weights):
    # Same complete paired realizations calibrate the two-band max statistic.
    sim_by_band = np.mean(((logs-center)/weights)**2, axis=2)
    obs_by_band = np.mean(((target-center)/weights)**2, axis=1)
    typical = np.median(sim_by_band, axis=0)
    tsim = np.max(sim_by_band/typical, axis=1)
    tobs = float(np.max(obs_by_band/typical))
    q95 = float(np.quantile(tsim, .95))
    residual_dex = (target-center)/np.log(10)
    logp = np.log10(params)
    boundary = [NAMES[j] for j in range(6)
                if min(logp[j]-LOG_BOUNDS[j, 0], LOG_BOUNDS[j, 1]-logp[j])
                < .025*(LOG_BOUNDS[j, 1]-LOG_BOUNDS[j, 0])]
    x = 2*np.pi*params[1]/100.
    return dict(
        joint_discrepancy=tobs, joint_threshold95=q95,
        conditional_tail_fraction=float((1+np.count_nonzero(tsim>=tobs))/(len(tsim)+1)),
        compatible95=bool(tobs <= q95),
        median_abs_residual_dex_W2=float(np.median(np.abs(residual_dex[0]))),
        median_abs_residual_dex_U=float(np.median(np.abs(residual_dex[1]))),
        p90_abs_residual_dex_W2=float(np.quantile(np.abs(residual_dex[0]), .9)),
        p90_abs_residual_dex_U=float(np.quantile(np.abs(residual_dex[1]), .9)),
        I_variance_fraction_period_below10=float(1-2/np.pi*np.arctan(2*np.pi*params[0]/10)),
        S_variance_fraction_period_above100=float(2/np.pi*(np.arctan(x)+x/(1+x*x))),
        tau_ratio=float(params[1]/params[0]),
        stationary_amplitude_ratio_W2=float(params[4]/params[2]),
        stationary_amplitude_ratio_U=float(params[5]/params[3]),
        near_bounds=';'.join(boundary))


def validate():
    pool = np.loadtxt(OUT / 'candidate_pool.csv', delimiter=',', skiprows=1, ndmin=2)
    metric = np.load(OUT / 'fixed_metric.npz')
    target, weights = metric['target'], metric['weights']
    train = ForwardModel(seed=SEARCH_SEED, nreal=256)
    status('rerank_256', candidates=len(pool))
    reranked = []
    for i, row in enumerate(pool):
        params = row[1:]
        logs = np.log(train.evaluate(params))
        center = logs.mean(axis=0)
        reranked.append((score(logs, target, weights), params, center))
        if (i+1) % 10 == 0:
            status('rerank_256', completed=i+1, total=len(pool))
    reranked.sort(key=lambda x: x[0])
    del train
    model = ForwardModel(seed=VALIDATION_SEED, nreal=500)
    (OUT / 'validation').mkdir(exist_ok=True)
    rows = []
    for i, (distance, params, center) in enumerate(reranked):
        key = f'C{i+1:03d}'
        p = model.evaluate(params)
        logs = np.log(p)
        diagnostic = candidate_diagnostics(params, logs, center, target, weights)
        row = dict(candidate_id=key, score256=float(distance), **dict(zip(NAMES, params.tolist())), **diagnostic)
        q = np.quantile(p, [.025, .16, .5, .84, .975], axis=0)
        # Curves are sampled at the actual band epochs; no per-draw rescaling.
        curves = model.curves(params)
        for b, label in enumerate(('UVW2', 'U')):
            values = curves[label] if isinstance(curves, dict) else curves[b]
            rms = np.std(values, axis=1)
            row[f'fractional_rms_{label}_p05'] = float(np.quantile(rms, .05))
            row[f'fractional_rms_{label}_median'] = float(np.median(rms))
            row[f'fractional_rms_{label}_p95'] = float(np.quantile(rms, .95))
        np.savez_compressed(OUT / 'validation' / f'{key}.npz',
                            params=params, center256=center, mean_P=p.mean(axis=0),
                            quantiles_P=q, quantile_levels=[.025, .16, .5, .84, .975],
                            log_center500=logs.mean(axis=0))
        rows.append(row)
        if (i+1) % 10 == 0:
            status('validate_500', completed=i+1, total=len(pool),
                   compatible=sum(r['compatible95'] for r in rows))
    fieldnames = list(rows[0])
    for filename, selected in [('all_candidates.csv', rows),
                               ('compatible_candidates.csv', [r for r in rows if r['compatible95']])]:
        with (OUT / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(selected)
    write_json(OUT / 'candidate_details.json', rows)
    compatible = [r for r in rows if r['compatible95']]
    summary = dict(total_candidates=len(rows), compatible95=len(compatible),
                   best_psd=rows[0],
                   parameter_ranges={k: [min(r[k] for r in compatible), max(r[k] for r in compatible)]
                                     for k in NAMES} if compatible else {},
                   interpretation='Conditional PSD compatibility set within the documented search bounds; '
                                  'not a posterior, confidence region, or unique physical decomposition.')
    write_json(OUT / 'validation_summary.json', summary)
    status('validation_complete', total=len(rows), compatible95=len(compatible))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('search', 'validate', 'all'))
    args = parser.parse_args()
    if args.stage in ('search', 'all'):
        search()
    if args.stage in ('validate', 'all'):
        validate()
