#!/usr/bin/env python3
"""Attach physical diagnostics and render a finite PSD-compatible solution set."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from psd_match_forward import ForwardModel, BANDS
from psd_match_physics import physics_metrics
from search_817_psd import OUT, NAMES, VALIDATION_SEED, write_json, status
from plot_style import COLORS, style_axis


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    plt.style.use(Path(__file__).resolve().parents[3] / 'src/image-creator.mplstyle')
    rows = json.loads((OUT / 'candidate_details.json').read_text())
    model = ForwardModel(seed=VALIDATION_SEED, nreal=500)
    metric = np.load(OUT / 'fixed_metric.npz')
    weights = metric['weights']
    (OUT / 'physics').mkdir(exist_ok=True)
    for i, row in enumerate(rows):
        params = np.array([row[k] for k in NAMES])
        physics = physics_metrics(model, params)
        write_json(OUT/'physics'/f"{row['candidate_id']}.json", physics)
        raw = physics['rms_by_sigma']['raw']
        row['W2_slow_to_response_rms'] = raw['UVW2']['slow_to_illumination']
        row['U_slow_to_response_rms'] = raw['U']['slow_to_illumination']
        row['U_BLR_to_noise_rms'] = raw['U']['blr_to_noise']
        for band in BANDS:
            row[f'{band}_expected_total_rms'] = raw[band]['total']
        if (i+1) % 15 == 0:
            status('physical_diagnostics', completed=i+1, total=len(rows))
    compatible = [r for r in rows if r['compatible95']]
    for filename, subset in [('all_candidates.csv', rows), ('compatible_candidates.csv', compatible)]:
        with (OUT/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(subset)
    write_json(OUT/'candidate_details.json', rows)

    # Representative choices expose parameter tradeoffs, not a physical ranking.
    base = compatible if compatible else rows
    selected = [min(base, key=lambda r: r['score256'])]
    criteria = [('fastest_driver', lambda r: (r['tauI'], r['score256'])),
                ('longest_slow', lambda r: (-r['tauS'], r['score256']))]
    reasons = ['smallest_PSD_distance']
    for label, key in criteria:
        ordered = sorted(base, key=key)
        pick = next((r for r in ordered if r['candidate_id'] not in {s['candidate_id'] for s in selected}), None)
        if pick is not None:
            selected.append(pick); reasons.append(label)
    write_json(OUT/'representatives.json', [dict(selection_reason=why, **r)
                                          for r, why in zip(selected, reasons)])
    palette = [COLORS['PRIMARY'], COLORS['ALTERNATIVE'], COLORS['SECONDARY']]
    linestyles = ['-', (0, (5, 2.2)), (0, (5, 1.8, 1.3, 1.8))]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.8), sharex=True, sharey=True)
    filtered_checks = []
    for row_index, sigma in enumerate((None, 20, 40)):
        observed = model.observed(sigma=sigma)
        for b, band in enumerate(BANDS):
            ax = axes[row_index, b]
            ax.loglog(model.freq_bins[b], observed[b], color=COLORS['INK'], lw=1.6, label='Mrk 817 observed')
            ax.set_title(f"{band}: " + ('original' if sigma is None else f'high-pass sigma={sigma} d'), loc='left')
            style_axis(ax, y_on_right=False)
        for k, row in enumerate(selected):
            params = [row[name] for name in NAMES]
            powers = model.evaluate(params, sigma=sigma)
            quantiles = np.quantile(powers, [.16, .5, .84], axis=0)
            np.savez_compressed(OUT/'validation'/f"{row['candidate_id']}_{'raw' if sigma is None else 's'+str(sigma)}.npz",
                                params=params, quantiles=quantiles, observed=observed, frequency=model.freq_bins)
            for b, band in enumerate(BANDS):
                ax = axes[row_index, b]
                ax.loglog(model.freq_bins[b], quantiles[1, b], color=palette[k], ls=linestyles[k], lw=1.7,
                          label=row['candidate_id'])
                if k == 0:
                    ax.fill_between(model.freq_bins[b], quantiles[0, b], quantiles[2, b],
                                    color=COLORS['UNCERTAINTY'], alpha=.32, linewidth=0,
                                    label=f"{row['candidate_id']} 68% realizations")
            logp = np.log(powers); center = logp.mean(axis=0)
            obs_score = np.mean(((np.log(observed)-center)/weights)**2, axis=1)
            sim_score = np.mean(((logp-center)/weights)**2, axis=2)
            filtered_checks.append(dict(candidate_id=row['candidate_id'], sigma=sigma,
                per_band_distance=obs_score.tolist(),
                per_band_tail_fraction=((1+(sim_score>=obs_score).sum(axis=0))/(len(sim_score)+1)).tolist(),
                note='Same data after filtering: descriptive check, not independent evidence or a selection criterion.'))
    for ax in axes[-1]: ax.set_xlabel(r'Frequency [day$^{-1}$]')
    for ax in axes[:, 0]: ax.set_ylabel(r'Fractional $P(f)$ [day]')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, 1))
    save(fig, 'candidate_psd_comparison')
    write_json(OUT/'filtered_checks.json', filtered_checks)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    passed = [r for r in rows if r['compatible95']]
    failed = [r for r in rows if not r['compatible95']]
    for ax, xkey, ykey in ((axes[0], 'tauI', 'tauS'),
                           (axes[1], 'I_variance_fraction_period_below10', 'U_BLR_to_noise_rms')):
        if failed:
            ax.scatter([r[xkey] for r in failed], [r[ykey] for r in failed],
                       color=COLORS['UNCERTAINTY'], marker='x', s=22, label='Outside PSD screen')
        if passed:
            values = [r['score256'] for r in passed]
            scatter = ax.scatter([r[xkey] for r in passed], [r[ykey] for r in passed],
                                 c=values, cmap='viridis', norm=LogNorm(min(values), max(values)),
                                 s=30, edgecolors='none', label='PSD-compatible candidates')
        ax.set_yscale('log')
        style_axis(ax, y_on_right=False)
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'Central DRW $\tau_I$ [day]')
    axes[0].set_ylabel(r'Slow process $\tau_S$ [day]')
    axes[1].set_xlabel(r'Central variance at periods $<10$ d')
    axes[1].set_ylabel('U BLR RMS / measurement-noise RMS')
    axes[1].axhline(1, color=COLORS['GUIDE'], lw=.9, zorder=0)
    axes[1].set_xlim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .91))
    if passed:
        fig.subplots_adjust(right=.89, wspace=.30)
        cbax = fig.add_axes([.91, .18, .015, .65])
        fig.colorbar(scatter, cax=cbax, label='PSD distance (256 realizations)')
    save(fig, 'solution_set')

    summary = dict(
        candidate_count=len(rows), compatible_count=len(compatible), representatives=selected,
        actual_observed_fractional_rms={b:float(np.std(model.observed_curves[b])) for b in BANDS},
        compatible_ranges={k:[min(r[k] for r in compatible),max(r[k] for r in compatible)]
                           for k in NAMES} if compatible else {},
        compatible_fast_majority=sum(r['I_variance_fraction_period_below10']>=.5 for r in compatible),
        compatible_blr_above_noise=sum(r['U_BLR_to_noise_rms']>=1 for r in compatible),
        compatible_both=sum(r['I_variance_fraction_period_below10']>=.5 and r['U_BLR_to_noise_rms']>=1
                            for r in compatible),
        selection='Finite candidate set passing a candidate-conditional 95% PSD discrepancy screen. '
                  'Not a posterior or confidence set. Physical choice remains with the user.')
    write_json(OUT/'final_summary.json', summary)
    (OUT/'methodology.txt').write_text('''Mrk817 W2/U PSD calibration, 2026-10-03

Input: actual extinction-corrected, host-subtracted original observations in
test/mica_smooth/data/original; W2 306 and U 316 epochs. Constant means fix
fractional units. Actual per-epoch errors supply independent Gaussian noise.
No MICA runs and no old experiment inputs or results are modified.

Model in fractional units:
W2 = AW DW*I + BW S + noise;
U = AU (0.9 DU + 0.1 BLR)*I + BU S + noise.
I is unit stationary-variance OU; S is independent unit stationary-variance
Matern32 with K=(1+abs(h)/tauS) exp(-abs(h)/tauS). Both bands share the same
I and same S, with no delay/convolution of S. Amplitudes are stationary latent
coefficients, not realized light-curve RMS. No per-realization RMS adjustment.
Disk kernels match historical test geometry and are unit-area normalized;
BLR retains Gamma k=2 onset=6 d, scale=7 d, centroid=20 d, SD=sqrt(2)*7 d.

Stationary I-response covariance is evaluated with the established Fourier
quadrature on a 13107.2 d covariance period, then sampled jointly at actual
epochs. S is drawn by an exact state-space transition at merged epochs. This
incorporates low-frequency variability in finite sampled windows without
generating an arbitrary short periodic record. No hidden jitter is added.

PSD estimator: floating-mean unweighted LombScargle times 2T/N; arithmetic
average in 23 logarithmic bins, 1/T to 0.3/day. All curves use the same original
band mean. Raw observed power includes measurement noise. This is a sampled
periodogram, not a deconvolved intrinsic PSD. Mathematically equivalent cached
LS projections and component quadratic forms accelerate amplitude searches.

Search: six-dimensional differential evolution followed by a two-time-scale
grid with four-amplitude differential evolution at each point; fixed 64 normal
draws, fixed reference-model log-PSD scatter weights, no physical preference
penalty. Candidate parameters are reranked using 256 draws with the same prefix.
The two raw PSDs are the sole fitting targets. Bounds, seeds and full tested
points are saved in search_config.json and search_evaluations.csv.

Independent screen: 500 new paired realizations for every retained candidate.
For each band compute standardized log-PSD distance to the 256-draw center;
divide by median simulated distance and take the maximum across bands. A
candidate passes if its observed maximum is below the simulated 95th quantile.
This prevents failure in one band being hidden by a good fit in the other.
This finite candidate-conditional screen is not a post-selection global
p-value, Bayesian posterior, or confidence region. Near-bound candidates are
flagged; the reported set is limited to the documented finite search.

The figures show three examples (best PSD distance, fastest central driver,
longest slow process) to expose degeneracy, not to pick the preferred physics.
Shading is the best-distance example's pointwise central 68% simulation range,
not parameter uncertainty or a simultaneous band. Sigma20/40 comparison uses
the actual row-normalized high-pass operator; it is a dependent consistency
check and did not influence search or candidate admission.

Physical diagnostics use sqrt(E[ddof0 sample variance]) at the actual epochs,
including mean subtraction and, where requested, the exact high-pass matrix.
U illumination includes the disk/BLR covariance. TF area is not equated to RMS.
Physical attributes are exported so the user can choose among compatible cases.
''')
    status('complete', candidates=len(rows), compatible=len(compatible),
           output=str(OUT), mica_launched=False)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
