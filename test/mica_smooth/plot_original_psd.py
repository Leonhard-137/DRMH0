#!/usr/bin/env python3
"""Plot unfiltered observed Mrk817_a light curves; do not run MICA or FFA."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/'mica_recovery/longterm_test'))
from plot_independent_slow_psd import expected_ls_density
from plot_style import style_axis, COLORS

OUT = HERE/'psd_original_observed'
BANDS = ('UVW2', 'UVM2', 'UVW1', 'U', 'B', 'V')


def bin_power(f, *powers):
    edges = np.geomspace(f[0], f[-1]*(1+1e-10), 24)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = (f >= lo) & (f < hi)
        if use.any():
            rows.append([np.exp(np.mean(np.log(f[use]))),
                         *(np.mean(p[use]) for p in powers), int(use.sum())])
    return np.asarray(rows)


def main():
    OUT.mkdir(exist_ok=True)
    plt.style.use(ROOT/'src/image-creator.mplstyle')
    spectra, summary = {}, []
    for band in BANDS:
        source = HERE/'data/original'/f'Mrk817_a_{band}.txt'
        data = np.loadtxt(source)
        assert data.shape[1] == 3 and np.isfinite(data).all()
        t = data[:, 0]-data[0, 0]
        assert np.all(np.diff(t) > 0) and np.all(data[:, 2] > 0)
        mean, span, n = float(np.mean(data[:, 1])), float(np.ptp(t)), len(t)
        assert mean > 0
        y = (data[:, 1]-mean)/mean
        f = np.arange(1/span, 0.30, 1/(5*span))
        scale = 2*span/n
        psd = LombScargle(t, y, fit_mean=True, normalization='psd').power(f, method='cython')*scale
        # Exact expectation of the same floating-mean LS statistic for
        # independent errors with the measured per-epoch variances.
        covariance = np.diag((data[:, 2]/mean)**2)
        noise = expected_ls_density(t, covariance, f, scale)
        assert np.all(np.isfinite(psd)) and np.all(psd > 0) and np.all(noise > 0)
        bins = bin_power(f, psd, noise, f*psd, f*noise)
        spectra[band] = (f, psd, noise, bins)
        np.savetxt(OUT/f'{band}_periodogram.csv', np.column_stack((f, psd, noise, f*psd, f*noise)),
                   delimiter=',', header='frequency_per_day,fractional_P,expected_noise_P,fractional_fP,expected_noise_fP', comments='')
        np.savetxt(OUT/f'{band}_binned.csv', bins, delimiter=',',
                   header='frequency_per_day,mean_P,mean_noise_P,mean_fP,mean_noise_fP,n_ordinates', comments='')
        summary.append(dict(band=band, source=str(source), n=n, span_days=span,
                            mjd_start=float(data[0, 0]), mjd_end=float(data[-1, 0]),
                            mean_flux=mean, fractional_sample_rms=float(np.std(y)),
                            median_cadence_days=float(np.median(np.diff(t))),
                            max_gap_days=float(np.max(np.diff(t)))))
    for kind in ('P', 'fP'):
        fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
        for ax, band in zip(axes.flat, BANDS):
            f, p, noise, bins = spectra[band]
            shown = p if kind == 'P' else f*p
            column = 1 if kind == 'P' else 3
            ax.loglog(f, shown, color=COLORS['INK'], lw=.5, alpha=.18, rasterized=True)
            ax.loglog(bins[:, 0], bins[:, column], color=COLORS['INK'], lw=1.7,
                      label='Original observed light curve')
            ax.loglog(bins[:, 0], bins[:, column+1], color=COLORS['UNCERTAINTY'], lw=1.4,
                      ls='--', label='Expected measurement noise')
            ax.set_title(band, loc='left')
            ax.set_xlim(min(v[0][0] for v in spectra.values()), .30)
            style_axis(ax, y_on_right=False)
        for ax in axes[-1]:
            ax.set_xlabel(r'Frequency [day$^{-1}$]')
        for ax in axes[:, 0]:
            ax.set_ylabel(r'Fractional $P(f)$ [day]' if kind == 'P' else r'Fractional $fP(f)$')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .965))
        name = 'mrk817_original_psd' if kind == 'P' else 'mrk817_original_fpsd'
        for extension in ('png', 'pdf'):
            fig.savefig(OUT/f'{name}.{extension}', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (OUT/'README.md').write_text('''# Observed Mrk817_a original PSD

Six observed bands: UVW2, UVM2, UVW1, U, B, V. Inputs are the unfiltered
test/mica_smooth/data/original/Mrk817_a_*.txt light curves, copied from
Mrk817_a/fflux/host_sub. These had extinction and host subtraction upstream;
"original" here means before Gaussian detrending, not uncalibrated photometry.
The full available interval of each saved band is used, with its own epochs.

Fractional light curve: (F - original mean)/original mean. Unweighted,
floating-mean Lomb--Scargle, normalization=psd, multiplied by 2T/N.
No interpolation, temporal smoothing/detrending, extra mean rescaling, or
noise subtraction is applied. The density normalization is approximate for
irregular sampling; these are observed-window periodograms, not intrinsic PSDs.

Grid: 1/T to 0.30/day, spacing 1/(5T). The ceiling is a display choice, not a
strict irregular-sampling Nyquist frequency. Faint lines are raw ordinates;
black lines are arithmetic mean power in 23 logarithmic frequency bins.
The companion fP plot bins f*P before averaging in the same bins.

Gray dashed curves are the exact expected floating-mean LS periodogram for
independent measurement errors with the original quoted per-epoch variances,
using the actual sample times and the same fractional normalization.
They are measurement-noise expectations, not detection thresholds, confidence
intervals or corrections for red-noise leakage. No PSD slope, break or physical
component is inferred. Fractional normalization differs from the absolute
flux-squared units used in the preceding shared-S simulation figure.

References: https://docs.astropy.org/en/stable/timeseries/lombscargle.html
''')
    print(json.dumps(summary, indent=2))
    print(OUT/'mrk817_original_psd.png')


if __name__ == '__main__':
    main()
