#!/usr/bin/env python3
"""Plot saved U high-pass inputs against the matched U without independent S.

Example: python plot_highpass_vs_clean_u.py --sigmas 10 20 40
All figures use common limits determined from the sigma5/10/20/40 inputs.
No fitting, random draws, interpolation or independent amplitude rescaling.
"""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_style import COLORS, configure_style, save_figure, style_axis

HERE = Path(__file__).resolve().parent
RAW = HERE / 'cases/abl_disk_blr_uonly_gp_raw'
CONTROL = HERE / 'cases/abl_disk_blr'


def rms(y):
    return float(np.std(y, ddof=0))


def case_path(sigma):
    directory = 'cases_uonly_gp_highpass_s5_s10' if sigma in (5, 10) else 'cases_uonly_gp_highpass'
    return HERE / directory / f'uonly_gp_raw_u_s{sigma}'


def limits(arrays, padding):
    low = min(float(np.min(y)) for y in arrays)
    high = max(float(np.max(y)) for y in arrays)
    return [low - padding * (high - low), high + padding * (high - low)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sigmas', type=int, nargs='+', choices=(5, 10, 20, 40), default=[10, 20, 40])
    args = parser.parse_args()
    raw = np.loadtxt(RAW / 'input/LTSIM_U.txt')
    parts = np.loadtxt(RAW / 'components/U_components.txt')
    control = np.loadtxt(CONTROL / 'input/LTSIM_U.txt')
    control_parts = np.loadtxt(CONTROL / 'components/U_components.txt')
    assert np.array_equal(raw[:, 0], parts[:, 0])
    assert np.array_equal(raw[:, [0, 2]], control[:, [0, 2]])
    assert np.array_equal(parts[:, [0, 1, 3, 4, 5, 6]], control_parts[:, [0, 1, 3, 4, 5, 6]])
    assert np.all(control_parts[:, 2] == 0)
    clean = raw[:, 1] - parts[:, 2]
    baseline_error = float(np.max(np.abs(clean - control[:, 1])))
    assert baseline_error < 1e-9
    curves = {}
    for sigma in (5, 10, 20, 40):
        case = case_path(sigma)
        data = np.loadtxt(case / 'input/LTSIM_U.txt')
        filtered_parts = np.loadtxt(case / 'components/U_components.txt')
        assert np.array_equal(raw[:, [0, 2]], data[:, [0, 2]])
        assert np.array_equal(raw[:, 0], filtered_parts[:, 0])
        identity_error = float(np.max(np.abs(1 + filtered_parts[:, 1:5].sum(axis=1) - data[:, 1])))
        assert identity_error < 1e-9
        curves[sigma] = {'case': case, 'input': data[:, 1], 'parts': filtered_parts,
                         'delta': data[:, 1] - clean, 'identity_error': identity_error}
    ylimits = limits([clean] + [c['input'] for c in curves.values()], 0.20)
    dlimits = limits([c['delta'] for c in curves.values()] + [c['parts'][:, 2] for c in curves.values()], 0.25)
    configure_style()
    time = raw[:, 0]
    for sigma, curve in curves.items():
        component_stats = {}
        for name, column in [('disk', 5), ('BLR', 6), ('independent_S', 2), ('noise', 4), ('disk_plus_BLR', 1)]:
            before, after = rms(parts[:, column]), rms(curve['parts'][:, column])
            component_stats[name] = {'rms_before': before, 'rms_after': after, 'rms_retained': after / before}
        summary = {
            'sigma_days': sigma,
            'sources': {'raw': str(RAW), 'filtered': str(curve['case']), 'matched_no_S_control': str(CONTROL)},
            'comparison': 'Saved U high-pass MICA input versus U_raw minus the exact saved independent S; same disk/BLR/noise realization.',
            'normalization': 'Original saved relative flux. No independent amplitude rescaling or recentering.',
            'rms_definition': 'Unweighted standard deviation ddof=0 over saved epochs; includes noise where present.',
            'n_epochs': len(time),
            'control_identity_max_abs_error': baseline_error,
            'filtered_component_sum_max_abs_error': curve['identity_error'],
            'component_statistics': component_stats,
            'U_no_S_rms': rms(clean), 'U_filtered_rms': rms(curve['input']),
            'pearson_r_filtered_vs_no_S': float(np.corrcoef(curve['input'], clean)[0, 1]),
            'difference_rms': rms(curve['delta']),
            'difference_rmse': float(np.sqrt(np.mean(curve['delta'] ** 2))),
            'shared_flux_limits': ylimits, 'shared_difference_limits': dlimits,
        }
        print(json.dumps({'sigma': sigma, 'r': summary['pearson_r_filtered_vs_no_S'],
                          'retained': {k: v['rms_retained'] for k, v in component_stats.items()}}))
        if sigma not in args.sigmas:
            continue
        out = curve['case'] / 'results/comparison_no_independent_slow'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        np.savetxt(out / 'light_curve_comparison.csv', np.column_stack([
            time, raw[:, 1], clean, curve['input'], curve['delta'], parts[:, 2], curve['parts'][:, 2],
            parts[:, 5], curve['parts'][:, 5], parts[:, 6], curve['parts'][:, 6],
            parts[:, 4], curve['parts'][:, 4], raw[:, 2],
        ]), delimiter=',', comments='', fmt='%.14g', header=(
            'time,U_original,U_no_S,U_filtered,U_filtered_minus_U_no_S,S_original,S_filtered,'
            'disk_original,disk_filtered,BLR_original,BLR_filtered,noise_original,noise_filtered,quoted_flux_error'))
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                                 gridspec_kw={'height_ratios': [2.1, 1.0]})
        axes[0].plot(time, clean, color=COLORS['INK'], lw=.95, marker='o', ms=1.9, mew=0,
                     label=r'$U_{\rm no\,S}$: disk + BLR + noise')
        axes[0].plot(time, curve['input'], color=COLORS['ALTERNATIVE'], lw=.95, marker='o', ms=1.9, mew=0,
                     label=rf'$U_{{{sigma}}}$: actual input after $\sigma={sigma}$ d detrending')
        axes[0].axhline(1, color=COLORS['GUIDE'], lw=.9, zorder=0)
        axes[0].set_ylabel('Relative U flux')
        axes[0].set_title(rf'(a) $\sigma={sigma}$ d', loc='left', fontsize=12, pad=5)
        axes[0].legend(loc='upper left', bbox_to_anchor=(0, 1.01), frameon=False, fontsize=10)
        axes[0].set_ylim(ylimits)
        axes[1].axhline(0, color=COLORS['GUIDE'], lw=.9, zorder=0)
        axes[1].plot(time, curve['delta'], color=COLORS['INK'], lw=1,
                     label=rf'$U_{{{sigma}}}-U_{{\rm no\,S}}$')
        axes[1].plot(time, curve['parts'][:, 2], color=COLORS['SECONDARY'], lw=1.3,
                     ls=(0, (5.0, 1.8, 1.3, 1.8)), label=rf'Remaining independent $S_{{{sigma}}}$')
        axes[1].set_ylim(dlimits)
        axes[1].set_ylabel('Flux difference')
        axes[1].set_title('(b)', loc='left', fontsize=12, pad=5)
        axes[1].set_xlabel('Time since MJD 59176.004915 (days)')
        axes[1].legend(loc='upper left', frameon=False, fontsize=10, ncol=2)
        for ax in axes:
            style_axis(ax, y_on_right=False)
            ax.set_xlim(time[0] - 3, time[-1] + 3)
        fig.subplots_adjust(hspace=.16, left=.10, right=.98, bottom=.10, top=.95)
        save_figure(fig, out / f'sigma{sigma}_vs_no_independent_slow')
        print('Figure:', out / f'sigma{sigma}_vs_no_independent_slow.jpg')


if __name__ == '__main__':
    main()
