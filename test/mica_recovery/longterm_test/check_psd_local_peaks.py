#!/usr/bin/env python3
"""Check a specified observed local PSD peak in fixed, previously searched models.

This is an exploratory replication frequency, not a significance test or fit.
No parameter optimization, MICA invocation, or historical-result mutation.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import json
import time
from pathlib import Path
import numpy as np
from psd_match_forward import ForwardModel, BANDS, PARAMETERS

BASE = Path(__file__).resolve().parent / 'psd_match_817'
OUT = BASE / 'local_peak_check'
SEED = 8171005
CHECK_SEED = 8171006
NREAL = 500
CHECK_NREAL = 2000
CORE = np.array([11, 12, 13])
LEFT = np.array([9, 10])
RIGHT = np.array([14, 15])
SEARCH = np.arange(10, 15)
TOL_DEX = .2


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(max(0, center-half)), float(min(1, center+half))]


def features(powers, frequency):
    """Powers: (realization, band, bin), with fixed log-linear side bands."""
    logp = np.log10(np.atleast_3d(powers))
    logf = np.log10(frequency)
    xl, xr = logf[:, LEFT].mean(axis=1), logf[:, RIGHT].mean(axis=1)
    yl, yr = logp[:, :, LEFT].mean(axis=2), logp[:, :, RIGHT].mean(axis=2)
    w = (logf[:, CORE]-xl[:, None])/(xr-xl)[:, None]
    background = yl[:, :, None]+(yr-yl)[:, :, None]*w[None]
    excess = logp[:, :, CORE]-background
    peak_offset = excess.argmax(axis=2)
    peak_bin = CORE[peak_offset]
    contrast = np.take_along_axis(excess, peak_offset[:, :, None], axis=2)[:, :, 0]
    peak_logpower = np.take_along_axis(logp, peak_bin[:, :, None], axis=2)[:, :, 0]
    raw_peak = SEARCH[logp[:, :, SEARCH].argmax(axis=2)]
    # Require an actual peak within the target interval, not just red power
    # decreasing monotonically through the left boundary of the interval.
    has_peak = (raw_peak >= CORE[0]) & (raw_peak <= CORE[-1])
    return dict(peak_bin=peak_bin, contrast_dex=contrast,
                peak_logpower=peak_logpower, broad_contrast_dex=excess.mean(axis=2),
                has_peak=has_peak)


def assess(powers, observed, frequency):
    obs = features(observed[None], frequency)
    sim = features(powers, frequency)
    position = np.abs(sim['peak_bin']-obs['peak_bin']) <= 1
    similar = (position & sim['has_peak'] &
               (np.abs(sim['contrast_dex']-obs['contrast_dex']) <= TOL_DEX) &
               (np.abs(sim['peak_logpower']-obs['peak_logpower']) <= TOL_DEX))
    strong = position & sim['has_peak'] & (sim['contrast_dex'] >= obs['contrast_dex'])
    aligned = np.abs(sim['peak_bin'][:, 0]-sim['peak_bin'][:, 1]) <= 1
    joint = similar.all(axis=1) & aligned
    joint_strong = strong.all(axis=1) & aligned
    global_rmse = np.sqrt(np.mean((np.log10(powers)-np.log10(observed))**2, axis=(1, 2)))
    local_rmse = np.sqrt(np.mean((np.log10(powers[:, :, 9:16])-
                                 np.log10(observed[:, 9:16]))**2, axis=(1, 2)))
    counts = dict(W2_similar_count=int(similar[:, 0].sum()),
                  U_similar_count=int(similar[:, 1].sum()),
                  joint_similar_count=int(joint.sum()),
                  joint_atleast_contrast_count=int(joint_strong.sum()))
    n = len(powers)
    metrics = dict(nreal=n, **counts,
                   joint_similar_rate=counts['joint_similar_count']/n,
                   joint_atleast_contrast_rate=counts['joint_atleast_contrast_count']/n,
                   joint_similar_mc95_low=wilson(counts['joint_similar_count'], n)[0],
                   joint_similar_mc95_high=wilson(counts['joint_similar_count'], n)[1])
    matching = np.flatnonzero(joint)
    chosen = int(matching[np.argmin(global_rmse[matching])]) if len(matching) else None
    metrics.update(example_index0=chosen,
                   example_global_rmse_dex=None if chosen is None else float(global_rmse[chosen]),
                   example_local_rmse_dex=None if chosen is None else float(local_rmse[chosen]))
    return metrics, dict(**sim, similar=similar, joint_similar=joint,
                          joint_atleast_contrast=joint_strong,
                          global_rmse_dex=global_rmse, local_rmse_dex=local_rmse)


def save_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    for folder in ('draws', 'examples', 'confirmation'):
        (OUT/folder).mkdir(exist_ok=True)
    candidates = [r for r in json.loads((BASE/'candidate_details.json').read_text()) if r['compatible95']]
    model = ForwardModel(seed=SEED, nreal=NREAL, tau_cache_size=3, bundle_cache_size=1)
    observed, frequency = model.obs, model.freq_bins
    obsfeature = features(observed[None], frequency)
    # Deterministic sanity checks for the new metric, not for the reused core.
    self_metrics, _ = assess(np.repeat(observed[None], 2, axis=0), observed, frequency)
    assert self_metrics['joint_similar_count'] == 2
    slope_only = (frequency/.03)**-1.4
    assert not features(slope_only[None], frequency)['has_peak'].any()
    shifted_metrics, _ = assess(10*observed[None], observed, frequency)
    assert shifted_metrics['joint_similar_count'] == 0
    invariant = features(10*observed[None], frequency)['contrast_dex']-obsfeature['contrast_dex']
    assert np.max(np.abs(invariant)) < 1e-12
    write_json(OUT/'definition.json', dict(
        purpose='Exploratory reproduction of the observed ~30 day local peak, conditional on fixed candidates.',
        original_candidate_count=len(candidates), seed=SEED, nreal=NREAL,
        confirmation_seed=CHECK_SEED, confirmation_nreal=CHECK_NREAL,
        peak_bins0=CORE.tolist(), side_bins0=[LEFT.tolist(), RIGHT.tolist()],
        frequency=frequency.tolist(), observed=observed.tolist(),
        observed_features={k:v[0].tolist() for k,v in obsfeature.items()},
        definition='Log-linear background joins the mean logP/logf of bins9-10 and bins14-15. '
                   'Peak is the maximum logP excess in bins11-13. Require raw local maximum '
                   'over bins10-14 in bins11-13, peak location within one bin of observation, '
                   'contrast and absolute peak logP each within 0.2dex of observation. '
                   'Both bands must pass in the same realization and peak positions differ by at most one bin.',
        strong_definition='Same position/local-maximum rules; both contrasts at least observed; absolute power unconstrained.',
        caveat='Feature and tolerance chosen after inspecting data. Fractions are model-conditional replication rates, '
               'not p-values, posterior probabilities, or physical rankings. Wilson ranges describe Monte Carlo count uncertainty.',
        test_results='Identity matches; monotone power law has no peak; a factor10 scaling fails absolute-power rule; contrast scale invariant.'))
    started = time.monotonic()
    rows = []
    for i, candidate in enumerate(candidates):
        key = candidate['candidate_id']
        params = [candidate[k] for k in PARAMETERS]
        power = model.evaluate(params)
        metrics, detail = assess(power, observed, frequency)
        row = dict(candidate_id=key, **{k:candidate[k] for k in PARAMETERS},
                   score256=candidate['score256'],
                   I_variance_fraction_period_below10=candidate['I_variance_fraction_period_below10'],
                   U_BLR_to_noise_rms=candidate['U_BLR_to_noise_rms'], **metrics)
        rows.append(row)
        np.savez_compressed(OUT/'draws'/f'{key}.npz', powers=power, params=params,
                            frequency=frequency, observed=observed, **detail)
        if (i+1)%10 == 0 or i+1 == len(candidates):
            progress=dict(stage='screen', completed=i+1, total=len(candidates),
                          any_joint=sum(r['joint_similar_count']>0 for r in rows),
                          elapsed_seconds=time.monotonic()-started)
            write_json(OUT/'status.json', progress)
            print(json.dumps(progress), flush=True)
    rows.sort(key=lambda r:(-r['joint_similar_count'], r['score256']))
    save_rows(OUT/'candidate_rates.csv', rows)
    # Confirmation uses new draws for six pre-specified categories of candidates.
    selection = []
    def add(row, role):
        if row['candidate_id'] not in {x['candidate_id'] for x in selection}:
            selection.append(dict(row, role=role))
    for row in rows[:3]: add(row, 'highest_screen_rate')
    fast = [r for r in rows if r['I_variance_fraction_period_below10'] >= .5]
    if fast: add(fast[0], 'fast_driver')
    add(next(r for r in rows if r['candidate_id']=='C001'), 'best_average_PSD')
    add(next(r for r in rows if r['candidate_id']=='C069'), 'previous_fast_example')
    del model
    check = ForwardModel(seed=CHECK_SEED, nreal=CHECK_NREAL, tau_cache_size=3, bundle_cache_size=1)
    confirmed = []
    for row in selection:
        key = row['candidate_id']
        params = [row[k] for k in PARAMETERS]
        power = check.evaluate(params)
        metrics, detail = assess(power, observed, frequency)
        new = {k:row[k] for k in ('candidate_id','role',*PARAMETERS,
               'I_variance_fraction_period_below10','U_BLR_to_noise_rms')}
        new.update(screen_count=row['joint_similar_count'], screen_nreal=NREAL, **metrics)
        confirmed.append(new)
        np.savez_compressed(OUT/'confirmation'/f'{key}.npz', powers=power, **detail)
        index=metrics['example_index0']
        if index is not None:
            np.savez_compressed(OUT/'examples'/f'{key}.npz', frequency=frequency,
                observed=observed, example=power[index], median=np.median(power,axis=0),
                first_draws=power[:8], params=params, example_index0=index,
                seed=CHECK_SEED, nreal=CHECK_NREAL)
            curves=check.curves(params)
            np.savez_compressed(OUT/'examples'/f'{key}_curves.npz',
                **{f'{b}_time':check.times[b]+check.time_zero for b in BANDS},
                **{f'{b}_fractional':curves[b][index] for b in BANDS},
                **{f'{b}_error':check.errors[b] for b in BANDS})
            # Same selected realization after filtering, never reselect per sigma.
            selected_curves={b:curves[b][index] for b in BANDS}
            for sigma in (20,40):
                filtered=check.periodograms(selected_curves,sigma=sigma)[0]
                np.savez_compressed(OUT/'examples'/f'{key}_s{sigma}.npz',
                    frequency=frequency, observed=check.observed(sigma=sigma),
                    example=filtered, params=params, example_index0=index, seed=CHECK_SEED)
        print(json.dumps(dict(stage='confirmation',candidate_id=key,**metrics)),flush=True)
    save_rows(OUT/'confirmation_rates.csv',confirmed)
    examples=[]
    def show(row,role):
        if row['example_index0'] is not None and row['candidate_id'] not in {x['candidate_id'] for x in examples}:
            examples.append(dict(row,role=role))
    show(confirmed[0],'highest_screen_rate')
    show(next(r for r in confirmed if r['candidate_id']=='C001'),'best_average_PSD')
    for r in confirmed:
        if r['I_variance_fraction_period_below10']>=.5:
            show(r,'fast_driver')
            break
    summary=dict(stage='complete',candidates=len(rows),nreal=NREAL,seed=SEED,
        with_joint_similar_peak=sum(r['joint_similar_count']>0 for r in rows),
        with_joint_atleast_contrast=sum(r['joint_atleast_contrast_count']>0 for r in rows),
        joint_similar_rate_range=[min(r['joint_similar_rate'] for r in rows),max(r['joint_similar_rate'] for r in rows)],
        top_screen=rows[:10],confirmation=confirmed,selected_examples=examples,
        elapsed_seconds=time.monotonic()-started,
        interpretation='Specified local features in paired realizations; not complete periodogram/lightcurve replication. '
            'Selected examples are chosen for low whole-spectrum RMSE among joint matches; they are not typical realizations. '
            'No model parameters were refit. No MICA launched.')
    write_json(OUT/'summary.json',summary)
    write_json(OUT/'status.json',dict(stage='complete',elapsed_seconds=time.monotonic()-started))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
