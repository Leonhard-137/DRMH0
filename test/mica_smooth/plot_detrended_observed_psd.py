#!/usr/bin/env python3
"""Compare real Mrk817_a PSDs from original and existing MICA high-pass inputs."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from astropy.timeseries import LombScargle
from plot_original_psd import HERE, ROOT, BANDS, bin_power, style_axis

OUT = HERE / 'psd_detrended_observed'
SERIES = (
    ('original', 'Original', '#222222', '-'),
    ('all_s20', r'High-pass $\sigma=20$ d', '#1F4E9E', '-'),
    ('all_s40', r'High-pass $\sigma=40$ d', '#C43C39', (0, (5.0, 2.2))),
)


def reciprocal(x):
    return 1.0 / np.maximum(np.asarray(x, dtype=float), 1e-12)


def main():
    OUT.mkdir(exist_ok=True)
    plt.style.use(ROOT / 'src/image-creator.mplstyle')
    spectra, summary = {}, []
    for band in BANDS:
        original_path = HERE / 'data/original' / f'Mrk817_a_{band}.txt'
        original = np.loadtxt(original_path)
        time = original[:, 0] - original[0, 0]
        span, count = float(np.ptp(time)), len(time)
        mean = float(original[:, 1].mean())
        assert np.isfinite(original).all() and mean > 0
        assert np.all(np.diff(time) > 0) and np.all(original[:, 2] > 0)
        frequency = np.arange(1 / span, .30, 1 / (5 * span))
        powers, source_paths, validation = [], {}, {}
        for case, _, _, _ in SERIES:
            source = original_path if case == 'original' else (
                HERE / 'cases_highpass' / case / 'prepared' / original_path.name)
            data = np.loadtxt(source)
            assert data.shape == original.shape and np.isfinite(data).all()
            assert np.array_equal(data[:, 0], original[:, 0])
            assert np.array_equal(data[:, 2], original[:, 2])
            if case != 'original':
                manifest = json.loads((source.parent.parent / 'manifest.json').read_text())
                setup = manifest['smoothing']
                assert setup['output'] == 'highpass_residual_plus_original_mean'
                assert setup['smooth_driver'] and band in manifest['detrended_bands']
                sigma = float(setup['sigma_days'])
                assert sigma == float(case.removeprefix('all_s'))
                weights = np.exp(-.5 * ((time[:, None] - time[None, :]) / sigma)**2)
                weights /= weights.sum(axis=1, keepdims=True)
                expected = original[:, 1] - weights @ original[:, 1] + mean
                error = float(np.max(np.abs(expected - data[:, 1])))
                assert np.allclose(expected, data[:, 1], rtol=1e-9, atol=1e-8)
                validation[case] = dict(sigma_days=sigma, max_input_difference=error)
            # The SAME original mean scales all three series. Floating mean
            # removes any residual constant introduced by the finite window.
            fractional = (data[:, 1] - mean) / mean
            power = LombScargle(time, fractional, fit_mean=True, normalization='psd').power(
                frequency, method='cython') * (2 * span / count)
            assert np.isfinite(power).all() and np.all(power > 0)
            powers.append(power)
            source_paths[case] = str(source)
        # Ensure the reference is numerically identical to the previous plot.
        previous = np.loadtxt(HERE / 'psd_original_observed' / f'{band}_periodogram.csv',
                              delimiter=',', skiprows=1)
        assert np.allclose(previous[:, 0], frequency, rtol=1e-12, atol=0)
        assert np.allclose(previous[:, 1], powers[0], rtol=1e-10, atol=1e-14)
        bins = bin_power(frequency, *powers, *(frequency * p for p in powers))
        spectra[band] = (frequency, powers, bins)
        np.savetxt(OUT / f'{band}_periodogram.csv',
                   np.column_stack((frequency, *powers, *(frequency * p for p in powers))),
                   delimiter=',', comments='',
                   header='frequency_per_day,P_original,P_s20,P_s40,fP_original,fP_s20,fP_s40')
        np.savetxt(OUT / f'{band}_binned.csv', bins, delimiter=',', comments='',
                   header='frequency_per_day,mean_P_original,mean_P_s20,mean_P_s40,'
                          'mean_fP_original,mean_fP_s20,mean_fP_s40,n_ordinates')
        low = frequency < .01
        middle = (frequency >= .03) & (frequency < .1)
        summary.append(dict(
            band=band, n=count, span_days=span, normalization_mean_flux=mean,
            source_paths=source_paths, validation=validation,
            low_frequency_power_ratio_below_001={
                case: float(powers[j][low].sum() / powers[0][low].sum())
                for j, (case, *_) in enumerate(SERIES) if j},
            power_ratio_003_to_01={
                case: float(powers[j][middle].sum() / powers[0][middle].sum())
                for j, (case, *_) in enumerate(SERIES) if j}))

    for kind in ('P', 'fP'):
        fig, axes = plt.subplots(3, 2, figsize=(10, 9.6), sharex=True, sharey=True)
        for ax, band in zip(axes.flat, BANDS):
            frequency, powers, bins = spectra[band]
            for j, (_, label, color, linestyle) in enumerate(SERIES):
                column = j + (1 if kind == 'P' else 4)
                ax.loglog(bins[:, 0], bins[:, column], color=color, ls=linestyle,
                          lw=1.8 if j else 1.45, label=label)
            ax.set_title(band, loc='left', fontsize=13)
            ax.set_xlim(min(v[0][0] for v in spectra.values()), .30)
            style_axis(ax, y_on_right=False)
        for ax in axes[0]:
            secondary = ax.secondary_xaxis('top', functions=(reciprocal, reciprocal))
            secondary.set_xticks([300, 100, 30, 10, 4], labels=['300', '100', '30', '10', '4'])
            secondary.xaxis.set_minor_locator(NullLocator())
            secondary.set_xlabel(r'Period $1/f$ [day]')
            ax.set_title(ax.get_title(loc='left'), loc='left', pad=43)
        for ax in axes[-1]:
            ax.set_xlabel(r'Frequency [day$^{-1}$]')
        for ax in axes[:, 0]:
            ax.set_ylabel(r'Fractional $P(f)$ [day]' if kind == 'P' else r'Fractional $fP(f)$')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=11)
        fig.tight_layout(rect=(0, .043, 1, 1))
        name = 'mrk817_original_vs_detrended_' + ('psd' if kind == 'P' else 'fpsd')
        for extension in ('png', 'pdf'):
            fig.savefig(OUT / f'{name}.{extension}', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUT / 'README.md').write_text('''# Mrk 817 observed PSD before and after Gaussian detrending

These are real observations, not recovery simulations. The original files in
../data/original are already extinction-corrected and host-subtracted. The two
filtered series are read directly from ../cases_highpass/all_s20/prepared and
../cases_highpass/all_s40/prepared, the actual saved MICA inputs. All six bands,
including UVW2, use F_highpass = F_original - W F_original + mean(F_original).
W is the row-normalized Gaussian matrix at each band's actual epochs. Its sigma
is 20 or 40 observer-frame days. No new filtered inputs or fits are produced.
Times, errors and the high-pass formula are checked against each original file.

Estimator matches ../plot_original_psd.py: unweighted floating-mean Lomb--Scargle,
normalization=psd, multiplied by 2T/N. Fractional flux always uses the ORIGINAL
mean for all three curves within each band; no unit-variance rescaling is used.
Each band's full saved window and own epochs are retained. No interpolation,
additional detrending, noise subtraction or fitted PSD model is used.
Frequency grid: 1/T to 0.30/day, spacing 1/(5T); 0.30/day is a plotting limit,
not an exact irregular-sampling Nyquist frequency. Curves show arithmetic mean
periodogram power in the same 23 logarithmic bins. The fP companion averages
f*P in each bin, rather than multiplying binned P by the bin-center frequency.
All panels within a figure share axis limits. Colors and styles distinguish
processing choices, not different physical components or a preferred model.

Raw and binned periodograms are saved as CSVs. The reference PSD is numerically
checked against the previously plotted original PSD files. The summary's low
frequency ratios are descriptive sums of periodogram power below 0.01/day over
the same sampled grid, not intrinsic variance fractions or significance tests.
These finite-window irregularly sampled periodograms are approximate fractional
spectral densities; leakage and frequency-bin correlations remain. There are
no uncertainty bands or measurement-noise curves in these comparison figures.
No claim of a statistically significant interband PSD difference is made.
''')
    print(json.dumps(summary, indent=2))
    print(OUT / 'mrk817_original_vs_detrended_psd.png')


if __name__ == '__main__':
    main()
