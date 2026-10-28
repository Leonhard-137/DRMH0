#!/usr/bin/env python3
"""Compare the saved sigma5 MICA input with the matched no-independent-S U.

Read-only with respect to simulation inputs and MICA outputs. No random draws,
refitting, interpolation, amplitude rescaling, or new detrending is performed.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_style import COLORS, configure_style, save_figure, style_axis

HERE = Path(__file__).resolve().parent
RAW = HERE / 'cases/abl_disk_blr_uonly_gp_raw'
FILTERED = HERE / 'cases_uonly_gp_highpass_s5_s10/uonly_gp_raw_u_s5'
CONTROL = HERE / 'cases/abl_disk_blr'
OUT = FILTERED / 'results/comparison_no_independent_slow'


def rms(values):
    """Unweighted population standard deviation over the actual saved epochs."""
    return float(np.std(values, ddof=0))


def main():
    raw = np.loadtxt(RAW / 'input/LTSIM_U.txt')
    filtered = np.loadtxt(FILTERED / 'input/LTSIM_U.txt')
    control = np.loadtxt(CONTROL / 'input/LTSIM_U.txt')
    parts = np.loadtxt(RAW / 'components/U_components.txt')
    filtered_parts = np.loadtxt(FILTERED / 'components/U_components.txt')
    control_parts = np.loadtxt(CONTROL / 'components/U_components.txt')
    for array in (filtered, control, parts, filtered_parts, control_parts):
        assert np.array_equal(raw[:, 0], array[:, 0])
    assert np.array_equal(raw[:, 2], filtered[:, 2])
    assert np.array_equal(raw[:, 2], control[:, 2])
    assert np.array_equal(parts[:, [1, 3, 4, 5, 6]], control_parts[:, [1, 3, 4, 5, 6]])
    assert np.all(control_parts[:, 2] == 0)
    clean = raw[:, 1] - parts[:, 2]
    clean_filtered = filtered[:, 1] - filtered_parts[:, 2]
    delta = filtered[:, 1] - clean
    input_identity = float(np.max(np.abs(1 + filtered_parts[:, 1:5].sum(axis=1) - filtered[:, 1])))
    control_identity = float(np.max(np.abs(clean - control[:, 1])))
    assert input_identity < 1e-9 and control_identity < 1e-9

    signals = {
        'U_no_S_unfiltered': clean,
        'U_sigma5_actual_input': filtered[:, 1],
        'U_no_S_sigma5': clean_filtered,
        'difference_sigma5_minus_no_S': delta,
        'U_no_S_noiseless': 1 + parts[:, 1],
        'U_sigma5_noiseless': 1 + filtered_parts[:, 1] + filtered_parts[:, 2],
    }
    component_stats = {}
    for name, column in [('disk', 5), ('BLR', 6), ('independent_S', 2), ('noise', 4), ('disk_plus_BLR', 1)]:
        before, after = rms(parts[:, column]), rms(filtered_parts[:, column])
        component_stats[name] = {'rms_before': before, 'rms_after': after, 'rms_retained': after / before}
    summary = {
        'sources': {'raw': str(RAW), 'sigma5': str(FILTERED), 'matched_control': str(CONTROL)},
        'comparison': 'Actual saved sigma5 U input versus the same U with only the injected independent S removed; W2 unchanged.',
        'same_realization': 'Disk, BLR and measurement-noise realizations, epochs and quoted errors are identical before filtering.',
        'clean_definition': 'U_no_S = U_raw - saved independent_S = matched abl_disk_blr U',
        'filter_definition': 'H5(x) = x - W5 @ x + mean(x); saved inputs are read directly.',
        'rms_definition': 'Unweighted standard deviation, ddof=0, at all 324 U epochs. No error subtraction; no amplitude normalization.',
        'n_epochs': len(raw),
        'max_abs_clean_vs_existing_control': control_identity,
        'max_abs_filtered_component_sum_vs_input': input_identity,
        'signal_statistics': {name: {'mean': float(np.mean(values)), 'rms': rms(values)} for name, values in signals.items()},
        'component_statistics': component_stats,
        'pearson_r_sigma5_vs_no_S_unfiltered': float(np.corrcoef(filtered[:, 1], clean)[0, 1]),
        'pearson_r_sigma5_vs_no_S_sigma5': float(np.corrcoef(filtered[:, 1], clean_filtered)[0, 1]),
        'rms_ratio_sigma5_vs_no_S_unfiltered': rms(filtered[:, 1]) / rms(clean),
        'difference_rmse': float(np.sqrt(np.mean(delta ** 2))),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT / 'light_curve_comparison.csv', np.column_stack([
        raw[:, 0], raw[:, 1], clean, filtered[:, 1], clean_filtered, delta,
        parts[:, 2], filtered_parts[:, 2], parts[:, 5], filtered_parts[:, 5],
        parts[:, 6], filtered_parts[:, 6], parts[:, 4], filtered_parts[:, 4], raw[:, 2],
    ]), delimiter=',', comments='', fmt='%.14g', header=(
        'time,U_original,U_no_S,U_sigma5,U_no_S_sigma5,U_sigma5_minus_U_no_S,'
        'S_original,S_sigma5,disk_original,disk_sigma5,BLR_original,BLR_sigma5,'
        'noise_original,noise_sigma5,quoted_flux_error'))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    configure_style()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                             gridspec_kw={'height_ratios': [2.1, 1.0]})
    time = raw[:, 0]
    # Both curves are saved sampled light curves, not inferred model functions.
    axes[0].plot(time, clean, color=COLORS['INK'], lw=0.95,
                 marker='o', ms=1.9, mew=0, label=r'$U_{\rm no\,S}$: disk + BLR + noise')
    axes[0].plot(time, filtered[:, 1], color=COLORS['ALTERNATIVE'], lw=0.95,
                 marker='o', ms=1.9, mew=0, label=r'$U_5$: actual input after $\sigma=5$ d detrending')
    axes[0].axhline(1, color=COLORS['GUIDE'], lw=0.9, zorder=0)
    axes[0].set_ylabel('Relative U flux')
    axes[0].set_title('(a)', loc='left', fontsize=12, pad=5)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, 1.01), frameon=False, fontsize=10)
    axes[0].margins(y=0.20)

    axes[1].axhline(0, color=COLORS['GUIDE'], lw=0.9, zorder=0)
    axes[1].plot(time, delta, color=COLORS['INK'], lw=1.0,
                 label=r'$U_5-U_{\rm no\,S}$')
    axes[1].plot(time, filtered_parts[:, 2], color=COLORS['SECONDARY'], lw=1.3,
                 ls=(0, (5.0, 1.8, 1.3, 1.8)), label=r'Remaining independent $S_5$')
    axes[1].set_ylabel('Flux difference')
    axes[1].set_title('(b)', loc='left', fontsize=12, pad=5)
    axes[1].set_xlabel('Time since MJD 59176.004915 (days)')
    axes[1].legend(loc='upper left', frameon=False, fontsize=10, ncol=2)
    axes[1].margins(y=0.25)
    for ax in axes:
        style_axis(ax, y_on_right=False)
        ax.set_xlim(time[0] - 3, time[-1] + 3)
    fig.subplots_adjust(hspace=0.16, left=0.10, right=0.98, bottom=0.10, top=0.95)
    save_figure(fig, OUT / 'sigma5_vs_no_independent_slow')
    (OUT / 'README.md').write_text('''# Sigma5 U compared with the matched U without independent S

The black curve in panel (a) is original U minus its saved independent slow
component. It equals the existing abl_disk_blr input to text-file precision.
The red curve is the actual sigma5 input already used for both MICA models.
Both share the same underlying disk, BLR and measurement-noise realization.
No curves are independently rescaled or recentered for this figure.

Panel (b) shows red minus black and the residual independent slow component
for scale. The difference contains removed disk/BLR variability, the change
in the same measurement-noise realization, and residual S. It is not a new
independent-noise residual, so no reduced chi-squared is calculated.

RMS means the unweighted standard deviation at the saved observation epochs,
not mean flux, transfer-function area or an error-corrected variance estimate.
Component columns permit a direct calculation of retained disk, BLR, S and
noise RMS. No fitting or input modifications are performed by this script.

Reproduce with ../../../../plot_sigma5_vs_clean_u.py using the astro Python.
''')
    print(json.dumps(summary, indent=2))
    print('Figure:', OUT / 'sigma5_vs_no_independent_slow.jpg')


if __name__ == '__main__':
    main()
