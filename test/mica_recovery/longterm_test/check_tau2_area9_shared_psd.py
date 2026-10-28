#!/usr/bin/env python3
"""Read-only preflight of saved shared-S inputs; write PSD diagnostics, never fit.

Exit zero only when deterministic input/model/filter checks and broad sampled
spectral sanity checks pass. Scientific limitations remain explicit warnings.
The observed spectra are finite-window Lomb--Scargle spectra, not intrinsic PSDs.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
from pathlib import Path
import argparse
import json
import traceback
from datetime import datetime, timezone
import numpy as np
from astropy.timeseries import LombScargle
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import configure_style, style_axis, save_figure, COLORS, SHORT_DASH, REFERENCE_DASH
from plot_independent_slow_psd import expected_ls_density, binned

HERE = Path(__file__).resolve().parent
CASES = ('raw', 'all_s5', 'all_s10', 'all_s20', 'all_s40', 'u_s20')
SIGMAS = (5, 10, 20, 40)
BANDS = ('UVW2', 'U')
COLS = {'disk': 5, 'BLR': 6, 'slow': 2, 'noise': 4}
LABELS = {'raw': 'Raw', 'all_s5': r'Both $\sigma=5$ d', 'all_s10': r'Both $\sigma=10$ d',
          'all_s20': r'Both $\sigma=20$ d', 'all_s40': r'Both $\sigma=40$ d',
          'u_s20': r'U only $\sigma=20$ d'}
PALETTE = {'raw': COLORS['INK'], 'all_s5': COLORS['HIGHLIGHT'], 'all_s10': COLORS['PRIMARY'],
           'all_s20': COLORS['ALTERNATIVE'], 'all_s40': COLORS['SECONDARY']}
COMP_COLORS = {'total': COLORS['INK'], 'disk': COLORS['PRIMARY'], 'BLR': COLORS['ALTERNATIVE'],
               'slow': COLORS['SECONDARY'], 'noise': COLORS['UNCERTAINTY']}


def weights(t, sigma):
    w = np.exp(-0.5*((t[:, None]-t[None, :])/sigma)**2)
    return w / w.sum(axis=1, keepdims=True)


def error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float('inf')
    return float(np.max(np.abs(a-b)))


def power(t, y, f):
    if np.std(y) < 1e-15:
        return np.zeros_like(f)
    return LombScargle(t, y, fit_mean=True, normalization='psd').power(f, method='cython') * (2*np.ptp(t)/len(t))


def band_power(f, p, low=0.0, high=0.01):
    m = (f >= low) & (f <= high)
    return float(trapezoid(p[m], f[m]))


def write_report(out, report):
    report['status'] = 'fail' if report['blocking_anomalies'] else 'pass'
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE/'cases_tau2_area9_shared')
    args = parser.parse_args()
    root = args.root.resolve()
    out = root/'diagnostics/psd_preflight'
    # Absence of prepared inputs is an error, not permission to create them.
    if not (root/'raw/components/U_components.txt').is_file():
        raise FileNotFoundError(f'Prepared inputs are absent: {root}')
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'fail', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'root': str(root), 'blocking_anomalies': [], 'warnings': [], 'checks': {},
              'method': 'Actual saved curves: unweighted floating-mean LombScargle, normalization=psd times 2*T/N; no interpolation. Approximate density, not a window-deconvolved intrinsic PSD.',
              'frequency_unit': 'cycles per observer-frame day', 'low_band_upper_per_day': 0.01,
              'acceptance_rule': 'All listed deterministic input/model checks, low-frequency attenuation/order checks, and shared-S conditional/spectral consistency checks must pass. No fit-quality or intrinsic-PSD inference is asserted.'}

    def check(name, ok, evidence):
        report['checks'][name] = {'pass': bool(ok), 'evidence': evidence}
        if not ok:
            report['blocking_anomalies'].append(name)

    try:
        metadata = json.loads((root/'raw/config/generation.json').read_text())
        p = metadata['parameters']
        alpha, baseline = float(metadata['alpha']), float(metadata['baseline_flux'])
        tau_i, tau_s = float(p['driver_tau_days']), float(p['independent_gp_tau_days'])
        amp_s = float(p['independent_gp_amplitude'])
        check('approved_parameters', abs(tau_i-2)<1e-12 and abs(alpha-1.45)<1e-12 and
              abs(float(p['response_disk_gain'])/float(p['response_blr_gain'])-9)<1e-10 and
              abs(tau_s-55.3735295468)<1e-8 and abs(float(p['blr_theta_days'])-7)<1e-12 and
              abs(float(p['blr_onset_days'])-6)<1e-12,
              {'tau_I_days': tau_i, 'alpha': alpha, 'tau_S_days': tau_s, 'slow_ensemble_rms': amp_s,
               'U_disk_BLR_area_ratio': float(p['response_disk_gain'])/float(p['response_blr_gain']),
               'BLR_onset_days': p['blr_onset_days'], 'BLR_theta_days': p['blr_theta_days']})
        data, parts, spectra, operators = {}, {}, {}, {}
        old = HERE/'cases_tau2_area9/raw'
        for case in CASES:
            data[case], parts[case], spectra[case], operators[case] = {}, {}, {}, {}
            for band in BANDS:
                a = np.loadtxt(root/case/f'input/LTSIM_{band}.txt')
                c = np.loadtxt(root/case/f'components/{band}_components.txt')
                data[case][band], parts[case][band] = a, c
                check(f'{case}/{band}/valid_samples', a.shape[1]==3 and c.shape[1]==7 and len(a)==len(c) and
                      np.isfinite(a).all() and np.isfinite(c).all() and np.all(np.diff(a[:, 0])>0) and
                      np.all(a[:, 1:]>0) and error(a[:, 0], c[:, 0])<1e-10,
                      {'n': len(a), 'min_flux': float(a[:, 1].min()), 'min_error': float(a[:, 2].min())})
                check(f'{case}/{band}/component_sum', error(a[:, 1], baseline+c[:, 1:5].sum(axis=1))<1e-9 and
                      error(c[:, 1], c[:, 5]+c[:, 6])<1e-9,
                      {'flux_max_abs_error': error(a[:, 1], baseline+c[:, 1:5].sum(axis=1)),
                       'illumination_max_abs_error': error(c[:, 1], c[:, 5]+c[:, 6])})
                reference = np.loadtxt(old/f'components/{band}_components.txt')
                quoted = np.loadtxt(old/f'input/LTSIM_{band}.txt')
                check(f'{case}/{band}/times_and_errors', error(a[:, [0, 2]], quoted[:, [0, 2]])<1e-12,
                      {'max_abs_error': error(a[:, [0, 2]], quoted[:, [0, 2]])})
                if case == 'raw':
                    # W2 alone gains S; all existing non-S columns and full U are paired.
                    selected = [0, 1, 3, 4, 5, 6] if band == 'UVW2' else list(range(7))
                    e = error(c[:, selected], reference[:, selected])
                    check(f'raw/{band}/paired_old_realization', e<1e-10, {'max_abs_error': e})
                    op = np.eye(len(a))
                else:
                    sigma = 20 if case == 'u_s20' else int(case.removeprefix('all_s'))
                    filtered = not (case == 'u_s20' and band == 'UVW2')
                    op = np.eye(len(a))-weights(a[:, 0], sigma)+np.ones((len(a), len(a)))/len(a) if filtered else np.eye(len(a))
                    expected = parts['raw'][band].copy()
                    expected[:, 1:] = op @ expected[:, 1:]
                    e = error(c, expected)
                    expected_flux = op @ data['raw'][band][:, 1]
                    check(f'{case}/{band}/filter_operator', e<1e-9 and error(a[:, 1], expected_flux)<1e-9,
                          {'filtered': filtered, 'sigma_days': sigma, 'components_max_abs_error': e,
                           'flux_max_abs_error': error(a[:, 1], expected_flux)})
                operators[case][band] = op
        raw = parts['raw']
        check('slow_U_sample_rms', abs(float(np.std(raw['U'][:, 2]))-1.3)<1e-10,
              {'rms': float(np.std(raw['U'][:, 2])), 'target': 1.3})
        check('W2_slow_added', float(np.std(raw['UVW2'][:, 2]))>0.5,
              {'rms': float(np.std(raw['UVW2'][:, 2]))})

        # Evidence that W2 is an extension of the SAME S, not an independent draw.
        extension = np.load(root/'shared_slow_extension.npz')
        report['shared_extension_keys'] = list(extension.files)
        for key, expected in [('tw', raw['UVW2'][:, 0]), ('tu', raw['U'][:, 0]),
                              ('observed_u', raw['U'][:, 2]), ('draw_w', raw['UVW2'][:, 2])]:
            check(f'shared_extension/{key}', error(extension[key], expected)<1e-9,
                  {'max_abs_error': error(extension[key], expected)})
        resid = extension['conditional_residual_w']
        cond = extension['conditional_covariance_w']
        predicted_rms = float(np.sqrt(max(np.trace(cond)/len(cond), 0)))
        resid_rms = float(np.sqrt(np.mean(resid**2)))
        e = error(extension['draw_w'], extension['conditional_mean_w']+resid)
        check('shared_extension/conditional_draw', e<1e-9 and np.isfinite(cond).all() and
              resid_rms < 6*predicted_rms and predicted_rms < 0.01*np.std(raw['U'][:, 2]),
              {'draw_sum_max_abs_error': e, 'conditional_expected_rms': predicted_rms,
               'conditional_realized_rms': resid_rms,
               'threshold': 'draw residual RMS <6 times conditional expected RMS; conditional expected RMS <1% of U S RMS'})

        spans = {band: float(np.ptp(raw[band][:, 0])) for band in BANDS}
        span = min(spans.values())
        f = np.arange(1/span, 0.30, 1/(5*span))
        report['sampling'] = {band: {'n': len(raw[band]), 'span_days': spans[band],
                                     'median_gap_days': float(np.median(np.diff(raw[band][:, 0])))} for band in BANDS}
        report['frequency_grid'] = {'min': float(f[0]), 'max': float(f[-1]), 'step': float(f[1]-f[0]),
                                    'display_ceiling_not_Nyquist': True}
        report['actual_rms'], report['low_frequency_ls_integral'] = {}, {}
        for case in CASES:
            report['actual_rms'][case], report['low_frequency_ls_integral'][case] = {}, {}
            for band in BANDS:
                a, c = data[case][band], parts[case][band]
                curves = {'total': a[:, 1]-np.mean(a[:, 1]), **{key: c[:, col] for key, col in COLS.items()}}
                spectra[case][band] = {key: power(a[:, 0], y, f) for key, y in curves.items()}
                check(f'{case}/{band}/finite_spectrum', all(np.isfinite(y).all() and np.min(y)>=-1e-12 for y in spectra[case][band].values()),
                      {'estimator': 'floating-mean LombScargle', 'density_scale': 2*spans[band]/len(a)})
                report['actual_rms'][case][band] = {key: float(np.std(y)) for key, y in curves.items()}
                report['low_frequency_ls_integral'][case][band] = {key: band_power(f, y) for key, y in spectra[case][band].items()}
                np.savetxt(out/f'{case}_{band}_sampled_psd.csv', np.column_stack([f, *spectra[case][band].values()]),
                           delimiter=',', header='frequency_per_day,'+','.join(spectra[case][band]), comments='')

        expected_slow = {case: {} for case in CASES}
        for band in BANDS:
            t = raw[band][:, 0]
            h = np.abs(t[:, None]-t[None, :])/tau_s
            covariance = amp_s**2*(1+h)*np.exp(-h)
            for case in CASES:
                op = operators[case][band]
                expected_slow[case][band] = expected_ls_density(t, op@covariance@op.T, f, 2*spans[band]/len(t))
            integrals = [report['low_frequency_ls_integral'][f'all_s{s}'][band]['slow'] for s in SIGMAS]
            denominator = report['low_frequency_ls_integral']['raw'][band]['slow']
            ratio = np.array(integrals)/denominator
            check(f'{band}/slow_low_frequency_attenuation', np.all(ratio<1.02) and np.all(np.diff(ratio)>=-0.02),
                  {'sigma_days': SIGMAS, 'low_band_power_retained': ratio.tolist(),
                   'threshold': 'all <1.02 of raw; nondecreasing sigma ordering, tolerance 0.02 of raw'})
            check(f'{band}/strongest_filter_effective', ratio[0]<0.10,
                  {'sigma5_low_band_power_retained': float(ratio[0]), 'threshold': '<0.10 of raw below 0.01/day'})
        low = report['low_frequency_ls_integral']['raw']
        actual_ratio = low['UVW2']['slow']/low['U']['slow']
        expectation_ratio = band_power(f, expected_slow['raw']['UVW2'])/band_power(f, expected_slow['raw']['U'])
        check('shared_S_low_frequency_spectral_consistency', 0.5<actual_ratio/expectation_ratio<2,
              {'W2_U_observed_low_band_ratio': actual_ratio, 'same_process_window_expectation_ratio': expectation_ratio,
               'observed_over_expectation_ratio': actual_ratio/expectation_ratio,
               'threshold': 'observed W2/U low-band ratio within factor 2 of identical-process, band-specific window expectation; exact shared-process construction is checked separately'})
        for band, reference_case in [('UVW2', 'raw'), ('U', 'all_s20')]:
            e = error(spectra['u_s20'][band]['total'], spectra[reference_case][band]['total'])
            check(f'u_s20/{band}/paired_spectrum', e<1e-8, {'reference_case': reference_case, 'max_abs_error': e})
        report['warnings'] += [
            'The 458-day observation window is shorter than the S fP(f) peak period (~603 days); this peak cannot be resolved from these saved light curves.',
            'LombScargle density and its frequency-band integrals are descriptive, window-dependent quantities for irregular sampling; they are not intrinsic spectral variance fractions. High-frequency leakage of S is not by itself evidence of a faulty generator.',
            'tau_I=2 days implies only ~43% of theoretical driver variance at periods <10 days. This is the approved model, not an unexpected PSD failure.',
            'The unchanged W2/U observation times differ. Identical sigma therefore gives slightly different discrete filters; filtered S is not exactly identical at different epochs.',
            'High-pass filtering changes the driver covariance away from a DRW and correlates errors. Native MICA still uses its existing DRW model and quoted diagonal errors; this preflight does not validate that likelihood approximation.',
            'u_s20 intentionally filters only U. Its effective W2-to-U kernel differs from the unfiltered physical response; it is a control, not a direct physical-kernel recovery measurement.'
        ]
        for case in CASES:
            r = report['actual_rms'][case]['U']
            if r['BLR']<r['noise']:
                report['warnings'].append(f"{case}: U BLR RMS {r['BLR']:.6g} is below realized noise RMS {r['noise']:.6g}; weak BLR identifiability is an expected scientific limitation, not a PSD-generation anomaly.")
            if r['slow']>r['BLR']:
                report['warnings'].append(f"{case}: residual U S/BLR RMS ratio={r['slow']/r['BLR']:.3g}; attenuation need not yield successful BLR recovery.")

        configure_style()
        theory_f = np.geomspace(1e-4, 1, 1200)
        theory_i = 4*alpha**2*tau_i/(1+(2*np.pi*tau_i*theory_f)**2)
        theory_s = 8*amp_s**2*tau_s/(1+(2*np.pi*tau_s*theory_f)**2)**2
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        for ax, multiply, ylabel in [(axes[0, 0], 1, r'Intrinsic PSD [flux$^2$ day]'),
                                     (axes[0, 1], theory_f, r'$fP(f)$ [flux$^2$]')]:
            ax.loglog(theory_f, multiply*theory_i, color=COLORS['PRIMARY'], lw=1.7, label=r'Central $\alpha I$')
            ax.loglog(theory_f, multiply*theory_s, color=COLORS['SECONDARY'], lw=1.7, label='Independent S')
            ax.axvline(1/span, color=COLORS['GUIDE'], lw=0.8)
            ax.set_ylabel(ylabel); ax.legend(frameon=False)
        for j, band in enumerate(BANDS):
            ax = axes[1, j]
            for key, pp in spectra['raw'][band].items():
                if np.max(pp)<=0: continue
                bb = binned(f, pp)
                ax.loglog(bb[:, 0], bb[:, 1], color=COMP_COLORS[key], lw=1.6 if key!='noise' else 1.1, label=key)
            ax.set_title('W2' if band=='UVW2' else 'U', loc='left')
            ax.set_ylabel(r'Sampled PSD [flux$^2$ day]'); ax.legend(frameon=False, fontsize=10)
        for ax in axes.flat:
            ax.set_xlabel(r'Frequency [day$^{-1}$]'); style_axis(ax, y_on_right=False)
        fig.tight_layout(); save_figure(fig, out/'raw_components_psd')

        fig, axes = plt.subplots(6, 2, figsize=(11, 19), sharex=True, sharey=True)
        for row, case in enumerate(CASES):
            for col, band in enumerate(BANDS):
                ax = axes[row, col]
                for key, pp in spectra[case][band].items():
                    if np.max(pp)<=0: continue
                    bb = binned(f, pp)
                    ax.loglog(bb[:, 0], bb[:, 1], color=COMP_COLORS[key], lw=1.35, label=key)
                ax.set_title(LABELS[case]+(' / W2' if band=='UVW2' else ' / U'), loc='left', fontsize=12)
                ax.set_ylabel(r'PSD [flux$^2$ day]')
                ax.set_ylim(1e-6, 3000); ax.set_xlim(f[0], f[-1])
                style_axis(ax, y_on_right=False)
        axes[0, 0].legend(frameon=False, fontsize=10, ncol=2)
        axes[0, 1].legend(frameon=False, fontsize=10, ncol=2)
        for ax in axes[-1]: ax.set_xlabel(r'Frequency [day$^{-1}$]')
        fig.tight_layout(); save_figure(fig, out/'six_case_component_psd')

        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
        for col, band in enumerate(BANDS):
            for case in CASES[:-1]:
                for row in (0, 1):
                    pp = spectra[case][band]['slow'] if row==0 else expected_slow[case][band]
                    bb = binned(f, pp)
                    axes[row, col].loglog(bb[:, 0], bb[:, 1], color=PALETTE[case], lw=1.6, label=LABELS[case])
            for row in (0, 1):
                ax = axes[row, col]
                ax.set_title(('Saved S' if row==0 else 'Same-window GP expectation')+(' / W2' if band=='UVW2' else ' / U'), loc='left', fontsize=12)
                ax.set_ylabel(r'Sampled PSD [flux$^2$ day]'); style_axis(ax, y_on_right=False)
                ax.set_xlim(f[0], f[-1]); ax.set_ylim(1e-6, 3000)
        axes[0, 0].legend(frameon=False, fontsize=10)
        for ax in axes[-1]: ax.set_xlabel(r'Frequency [day$^{-1}$]')
        fig.tight_layout(); save_figure(fig, out/'slow_filter_psd')
        np.savetxt(out/'intrinsic_psd.csv', np.column_stack([theory_f, theory_i, theory_s]), delimiter=',',
                   header='frequency_per_day,alphaI_intrinsic_one_sided_psd,S_intrinsic_one_sided_psd', comments='')
        for band in BANDS:
            np.savetxt(out/f'{band}_expected_slow_sampled_psd.csv', np.column_stack([f, *[expected_slow[case][band] for case in CASES]]),
                       delimiter=',', header='frequency_per_day,'+','.join(CASES), comments='')
        report['figures'] = [str(out/f'{name}.jpg') for name in ('raw_components_psd', 'six_case_component_psd', 'slow_filter_psd')]
        report['theory'] = {'central_DRW_one_sided_PSD': '4 alpha^2 tau_I / (1+(2 pi tau_I f)^2)',
                            'slow_Matern32_one_sided_PSD': '8 A_S^2 tau_S / (1+(2 pi tau_S f)^2)^2',
                            'central_fP_peak_period_days': 2*np.pi*tau_i, 'slow_fP_peak_period_days': 2*np.pi*np.sqrt(3)*tau_s,
                            'central_variance_fraction_period_below_10d': 1-2/np.pi*np.arctan(2*np.pi*tau_i/10)}
        write_report(out, report)
        lines = ['# Shared-S PSD preflight', '', f"Status: **{report['status'].upper()}**", '',
                 'Input is never edited and this program does not run MICA.', '',
                 '## What is plotted', '',
                 '- `raw_components_psd`: theoretical central/slow PSD and fP, plus raw saved W2/U component periodograms.',
                 '- `six_case_component_psd`: all six saved cases, same axes, full light curve and known disk/BLR/S/noise components.',
                 '- `slow_filter_psd`: saved S spectra and exact expected LS spectra under the specified covariance at each actual sample window after the actual discrete filter.', '',
                 'PSD uses unweighted floating-mean LombScargle, multiplied by 2T/N. Logarithmic bins show arithmetic mean periodogram power. No interpolation, no separate unit-power normalization, no line search, and no fitted spectral slope is used. Upper frequency 0.30/day is a display ceiling, not an irregular-sampling Nyquist frequency. Component PSDs do not add exactly in one realization because cross terms exist.', '',
                 'Expected LS spectrum is (T/N) trace(P_f K_filtered), with P_f the projector onto centered sine/cosine columns; K_filtered=F K F^T. This explicitly retains the observing window and finite-sample filter effects. Intrinsic fP peaks outside the observing baseline cannot be read back as resolved measured peaks.', '',
                 'Hard gates: all inputs finite and positive; approved parameters; original signal/noise/times/errors preserved; exact component sum and actual filter operator; shared conditional-GP extension; valid finite spectra; <0.01/day S power attenuated and correctly ordered with sigma; sigma5 leaves <10% of raw low-band power; raw W/U low-band S power ratio within factor two of the same-process window expectation. Numerical tolerances and all measured values are in report.json. Scientific modeling limitations below are explicitly not classified as unexpected generation defects.', '',
                 '## Warnings', '']+[f'- {w}' for w in report['warnings']]+['', '## Blocking anomalies', '']+[f'- {w}' for w in report['blocking_anomalies']]
        (out/'README.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps({'status': report['status'], 'report': str(out/'report.json'), 'checks': len(report['checks']),
                          'blocking_anomalies': report['blocking_anomalies'], 'warnings': len(report['warnings'])}, indent=2))
        return 1 if report['blocking_anomalies'] else 0
    except Exception as exc:
        report['blocking_anomalies'].append(f'{type(exc).__name__}: {exc}')
        report['exception_traceback'] = traceback.format_exc()
        write_report(out, report)
        print(report['exception_traceback'])
        print('FAILED:', out/'report.json')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
