#!/usr/bin/env python3
"""PSD of the saved U-only independent GP and selected saved high-pass residuals.

Reads saved truth; never prepares inputs, draws a new realization, or runs MICA.
"""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import argparse
import json
import numpy as np
import yaml
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from plot_style import configure_style, style_axis, COLORS, SHORT_DASH, REFERENCE_DASH

HERE = Path(__file__).resolve().parent
CASE = HERE / 'cases/abl_disk_blr_uonly_gp_raw'
OUT = CASE / 'results/independent_slow_psd'
SIGMAS = (20, 40)


def binned(f, p):
    edges = np.geomspace(f[0], f[-1] * (1 + 1e-10), 30)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (f >= lo) & (f < hi)
        if mask.any():
            rows.append([np.exp(np.mean(np.log(f[mask]))), np.mean(p[mask]), mask.sum()])
    return np.asarray(rows)


def expected_ls_density(t, covariance, frequency, scale):
    """Exact expected floating-mean LS quadratic form for these sampled times."""
    angle = 2 * np.pi * np.outer(t - t.mean(), frequency)
    s, c = np.sin(angle), np.cos(angle)
    s -= s.mean(axis=0)
    c -= c.mean(axis=0)
    ss = np.sum(s*s, axis=0)
    cc = np.sum(c*c, axis=0)
    sc = np.sum(s*c, axis=0)
    ks, kc = covariance @ s, covariance @ c
    trace = (cc*np.sum(s*ks, axis=0) + ss*np.sum(c*kc, axis=0)
             - 2*sc*np.sum(s*kc, axis=0)) / (ss*cc - sc*sc)
    return 0.5 * scale * trace


def reciprocal(x):
    return 1 / np.maximum(np.asarray(x, dtype=float), 1e-15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sigmas', nargs='+', type=int, choices=(5, 10, 20, 40), default=list(SIGMAS))
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    sigmas = tuple(dict.fromkeys(args.sigmas))
    output = args.output_dir.resolve()
    config_path = HERE / 'experiment_uonly_gp_raw.yaml'
    config = yaml.safe_load(config_path.read_text())
    gp = config['paired_ablation']['unfiltered_response_gp']
    tau, amplitude = float(gp['tau_days']), float(gp['amplitude'])
    raw = np.loadtxt(CASE / 'components/U_components.txt')
    tw = np.loadtxt(CASE / 'components/UVW2_components.txt')
    assert np.all(tw[:, 2] == 0), 'This case must have S in U only'
    t, slow = raw[:, 0], raw[:, 2]
    n, span = len(t), float(np.ptp(t))
    assert np.all(np.diff(t) > 0)
    assert abs(float(slow.mean())) < 1e-10
    curves = {'original': slow}
    paths = {'original': str(CASE / 'components/U_components.txt')}
    checks = {}
    for sigma in sigmas:
        suite_dir = 'cases_uonly_gp_highpass_s5_s10' if sigma in (5, 10) else 'cases_uonly_gp_highpass'
        path = HERE / suite_dir / f'uonly_gp_raw_u_s{sigma}/components/U_components.txt'
        values = np.loadtxt(path)
        assert np.array_equal(t, values[:, 0])
        weight = np.exp(-0.5*((t[:, None]-t[None, :])/sigma)**2)
        weight /= weight.sum(axis=1, keepdims=True)
        expected = slow - weight @ slow + slow.mean()
        checks[str(sigma)] = float(np.max(np.abs(expected-values[:, 2])))
        assert checks[str(sigma)] < 1e-10
        curves[f'sigma{sigma}'] = values[:, 2]
        paths[f'sigma{sigma}'] = str(path)

    # Known covariance: C(h)=A^2(1+|h|/tau) exp(-|h|/tau).
    # Fourier convention exp(-2 pi i f t); one-sided PSD for f>0.
    theory_f = np.geomspace(1e-4, 0.30, 1600)
    theory = 8*amplitude**2*tau / (1+(2*np.pi*tau*theory_f)**2)**2
    filtered_theory = {s: theory * (-np.expm1(-2*np.pi**2*s*s*theory_f**2))**2 for s in sigmas}
    variance_check = quad(lambda x: 4/np.pi/(1+x*x)**2, 0, np.inf, epsabs=1e-11)[0]
    assert abs(variance_check-1) < 1e-9

    # Same estimator as the project's observed Mrk817 diagnostic: no interpolation,
    # unweighted floating-mean LS, psd normalization times 2T/N. This density
    # normalization is approximate for irregular sampling, not an intrinsic PSD.
    freq = np.arange(1/span, 0.30, 1/(5*span))
    scale = 2*span/n
    periodograms = {key: LombScargle(t, y, fit_mean=True, normalization='psd').power(
        freq, method='cython') * scale for key, y in curves.items()}
    h = np.abs(t[:, None]-t[None, :])/tau
    covariance = amplitude**2*(1+h)*np.exp(-h)
    expected_raw = expected_ls_density(t, covariance, freq, scale)
    # The estimator expectation includes the actual window and floating mean;
    # no random simulation or amplitude adjustment enters this curve.

    summary = {
        'input_config': str(config_path), 'component_column_zero_indexed': 2,
        'input_files': paths, 'component': 'A_S*S added directly to U; no measurement noise or disk convolution',
        'n': n, 'span_days': span, 'median_cadence_days': float(np.median(np.diff(t))),
        'tau_days': tau, 'ensemble_std': amplitude,
        'covariance': 'A^2*(1+abs(h)/tau)*exp(-abs(h)/tau)',
        'one_sided_psd': '8*A^2*tau/(1+(2*pi*tau*f)^2)^2',
        'frequency_unit': 'cycles per observer-frame day',
        'knee_frequency_per_day': 1/(2*np.pi*tau), 'knee_period_days': 2*np.pi*tau,
        'variance_per_log_frequency_peak_period_days': 2*np.pi*np.sqrt(3)*tau,
        'ensemble_variance_at_periods_above_days': {},
        'actual_sample_rms': {key: float(np.std(y)) for key, y in curves.items()},
        'actual_sample_rms_retained': {key: float(np.std(y)/np.std(slow)) for key, y in curves.items()},
        'actual_sample_variance_retained': {key: float(np.var(y)/np.var(slow)) for key, y in curves.items()},
        'selected_sigmas_days': list(sigmas),
        'ideal_infinite_stationary_filter_variance_retained': {},
        'filter_reproduction_max_abs_error': checks,
        'psd_integral_divided_by_A_squared': variance_check,
        'periodogram': 'Astropy unweighted floating-mean LS, psd times 2*T/N; approximate density for irregular times',
        'periodogram_frequency_min': float(freq[0]), 'periodogram_frequency_max': float(freq[-1]),
        'binning': '29 logarithmic frequency bins; arithmetic mean power; no independent-bin uncertainty claim',
        'expected_periodogram': '0.5*(2*T/N)*trace(P_f K); sine/cosine columns projected off the constant',
    }
    for period in (30, 50, 100, 200, 300, span):
        x = 2*np.pi*tau/period
        summary['ensemble_variance_at_periods_above_days'][str(period)] = float(2/np.pi*(np.arctan(x)+x/(1+x*x)))
    for sigma in sigmas:
        v = quad(lambda x: 4/np.pi/(1+x*x)**2*(-np.expm1(-0.5*(sigma*x/tau)**2))**2,
                 0, np.inf, epsabs=1e-11)[0]
        summary['ideal_infinite_stationary_filter_variance_retained'][str(sigma)] = float(v)

    configure_style()
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)
    theory_ax, sample_ax = axes
    theory_ax.loglog(theory_f, theory, color=COLORS['REFERENCE'], ls=REFERENCE_DASH,
                     lw=1.8, label='Injected GP')
    color = {5: COLORS['HIGHLIGHT'], 10: COLORS['PRIMARY'], 20: COLORS['ALTERNATIVE'], 40: COLORS['SECONDARY']}
    linestyle = {5: '-', 10: (0, (2.7, 1.2)), 20: SHORT_DASH, 40: (0, (5.0, 1.8, 1.3, 1.8))}
    for sigma in sigmas:
        theory_ax.loglog(theory_f, filtered_theory[sigma], color=color[sigma],
                         ls=linestyle[sigma], lw=1.6, label=rf'High-pass $\sigma={sigma}$ d')
    theory_ax.set_ylabel(r'One-sided PSD [(relative flux)$^2$ day]')
    theory_ax.set_title('(a) Specified stationary process', loc='left', fontsize=12, pad=10)
    theory_ax.set_ylim(1e-7, 30)
    theory_ax.legend(loc='upper right', frameon=False, fontsize=10)

    for key, p in periodograms.items():
        sigma = None if key == 'original' else int(key.replace('sigma', ''))
        col = COLORS['INK'] if sigma is None else color[sigma]
        line = '-' if sigma is None else linestyle[sigma]
        label = 'Saved independent signal' if sigma is None else rf'High-pass $\sigma={sigma}$ d'
        sample_ax.loglog(freq, p, color=col, lw=.5, alpha=.18, rasterized=True)
        bp = binned(freq, p)
        sample_ax.loglog(bp[:, 0], bp[:, 1], color=col, ls=line, lw=1.65, label=label)
    sample_ax.loglog(freq, expected_raw, color=COLORS['REFERENCE'], ls=REFERENCE_DASH,
                    lw=1.4, label='GP expectation with sampling')
    sample_ax.set_ylabel(r'Sampled PSD, approx. [(relative flux)$^2$ day]')
    sample_ax.set_title('(b) Actual injected realization', loc='left', fontsize=12, pad=10)
    sample_ax.set_xlabel(r'Frequency (day$^{-1}$)')
    sample_ax.set_ylim(1e-6, 8)
    sample_ax.legend(loc='lower left', frameon=False, fontsize=9.6)
    for ax in axes:
        ax.axvline(1/span, color=COLORS['GUIDE'], lw=.9, zorder=0)
        ax.set_xlim(theory_f[0], theory_f[-1])
        style_axis(ax, y_on_right=False)
    top = theory_ax.secondary_xaxis('top', functions=(reciprocal, reciprocal))
    top.set_xticks([10000, 1000, 300, 100, 30, 10, 4], labels=['10000', '1000', '300', '100', '30', '10', '4'])
    top.set_xlabel('Period 1/f (days)')
    top.tick_params(direction='in', which='both')
    fig.tight_layout(h_pad=1.4)

    output.mkdir(parents=True, exist_ok=True)
    for extension in ('jpg', 'pdf'):
        fig.savefig(output / f'independent_slow_psd.{extension}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    np.savetxt(output/'theoretical_psd.csv', np.column_stack([theory_f, theory, *[filtered_theory[s] for s in sigmas]]),
               delimiter=',', header=','.join(['frequency_per_day', 'raw_one_sided_psd']+[f'ideal_hp{s}_psd' for s in sigmas]), comments='')
    np.savetxt(output/'sampled_periodogram.csv', np.column_stack([freq, *periodograms.values(), expected_raw]),
               delimiter=',', header=','.join(['frequency_per_day', 'raw_ls_density']+[f'hp{s}_ls_density' for s in sigmas]+['expected_raw_ls_density']), comments='')
    np.savetxt(output/'saved_signal.csv', np.column_stack([t, *curves.values()]), delimiter=',',
               header=','.join(['time_day', 'raw_independent_signal']+[f'hp{s}_independent_signal' for s in sigmas]), comments='')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'README.md').write_text('''# Independent slow-signal PSD

Source: `cases/abl_disk_blr_uonly_gp_raw/components/U_components.txt`, zero-indexed column 2
(`shared_long_variation` is the legacy name; here it is the U-only independent GP).
Only the saved slow-signal contribution is plotted: no disk, BLR, or measurement-noise terms.
All selected sigma curves read the existing paired detrended component files. Direct
reapplication of their normalized Gaussian weights is checked against these files.
No new realization, input edit, MICA sampling, or parameter change was performed.

## Top panel: known ensemble PSD

The actual generator uses covariance C(h)=A^2(1+|h|/tau)exp(-|h|/tau), with
A=0.15396862368414638 and tau=55.3735295468 observer-frame days.
Under Fourier convention exp(-2*pi*i*f*t), the ONE-SIDED PSD for f>0 is
P(f)=8*A^2*tau/[1+(2*pi*tau*f)^2]^2; its integral equals A^2.
This is a smooth spectrum, not a narrow line at period=tau. The knee is defined
by 2*pi*tau*f=1 (one quarter of the zero-frequency plateau).
The ideal filtered spectra multiply this PSD by
[1-exp(-2*pi^2*sigma^2*f^2)]^2. They assume an infinitely long, continuously
sampled stationary signal. They are not expected to exactly reproduce the
finite-window residual realization in the lower panel.

## Bottom panel: the signal actually injected

The 324 saved U epochs are irregular. We use unweighted floating-mean
Astropy LombScargle(normalization='psd'), without interpolation, times 2*T/N.
This is the project's uniform-sampling-equivalent one-sided density normalization;
it is approximate for irregular sampling, and the plotted periodogram is NOT a
window-deconvolved intrinsic PSD. The frequency grid starts at 1/T and ends
at 0.30/day with spacing 1/(5*T); 0.30/day is a display ceiling, not a rigorous
irregular-sampling Nyquist frequency. Thin curves show individual ordinates;
thicker curves average power in 29 logarithmic bins. There are no slope fits,
periodicity claims, or independent-bin error estimates.

The blue dotted lower curve is the exact expectation of this same unfiltered
periodogram under the configured covariance and ACTUAL sample times, including
the floating mean. At each f, if P_f projects onto mean-removed sine/cosine columns,
it is (T/N)*trace(P_f K). It is not a fitted PSD or a fresh Monte Carlo draw.
Its difference from the top-panel intrinsic spectrum illustrates window and
sampling effects. The light-gray vertical line marks 1/T for T=458.511897 days.
No sampled periodogram is drawn below that line.

The source is sample-demeaned and has finite-realization RMS different from A.
Consequently the quoted ensemble long-period variance fractions and ideal
filter-retention fractions must not be confused with finite-sample RMS ratios.
PSD is in squared relative-flux units times days; no separate curve is normalized
to equal integrated power. High-frequency power in a sampled periodogram can
include leakage/aliasing of the much stronger low-frequency process.

Method reference: https://docs.astropy.org/en/stable/timeseries/lombscargle.html
Reproduce with the astro environment and `plot_independent_slow_psd.py`.
Select existing cases with `--sigmas 5 10 20 40` and specify `--output-dir`
to retain the earlier two-sigma figures. The selected sigmas are in summary.json.
Orange: sigma5; dark blue: sigma10; red: sigma20; green: sigma40.
The royal-blue dotted curves are theory/reference, and the black curve is the original sample.
''')
    print(json.dumps(summary, indent=2))
    print('FIGURE:', output / 'independent_slow_psd.jpg')


if __name__ == '__main__':
    main()
