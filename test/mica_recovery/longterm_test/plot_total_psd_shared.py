#!/usr/bin/env python3
"""Plot the already measured total-light-curve PSDs of the shared-S experiment."""
from pathlib import Path
import os

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_style import configure_style, style_axis, save_figure, COLORS
from plot_independent_slow_psd import binned

HERE = Path(__file__).resolve().parent
OUT = HERE / 'cases_tau2_area9_shared/diagnostics/psd_preflight'
CASES = ('raw', 'all_s5', 'all_s10', 'all_s20', 'all_s40')
LABELS = ('Raw', r'Both $\sigma=5$ d', r'Both $\sigma=10$ d',
          r'Both $\sigma=20$ d', r'Both $\sigma=40$ d')
PALETTE = tuple(COLORS[k] for k in ('INK', 'HIGHLIGHT', 'PRIMARY', 'ALTERNATIVE', 'SECONDARY'))


def main():
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True, sharey='row')
    rows = []
    for col, band in enumerate(('UVW2', 'U')):
        for case, label, color in zip(CASES, LABELS, PALETTE):
            path = OUT / f'{case}_{band}_sampled_psd.csv'
            data = np.genfromtxt(path, delimiter=',', names=True)
            f, p = data['frequency_per_day'], data['total']
            assert np.all(np.isfinite(p)) and np.all(p > 0), path
            # Reuse the original frequency grid, density and logarithmic bins.
            psd_bin = binned(f, p)
            fp_bin = binned(f, f*p)
            axes[0, col].loglog(psd_bin[:, 0], psd_bin[:, 1], color=color,
                                lw=1.7, label=label, zorder=3 if case == 'raw' else 2)
            axes[1, col].loglog(fp_bin[:, 0], fp_bin[:, 1], color=color, lw=1.7)
            rows.extend((band, case, *values) for values in
                        zip(psd_bin[:, 0], psd_bin[:, 1], fp_bin[:, 1], psd_bin[:, 2]))
        axes[0, col].set_title(('W2' if band == 'UVW2' else 'U')+' total light curve', loc='left')
        axes[1, col].set_xlabel(r'Frequency [day$^{-1}$]')
        for ax in axes[:, col]:
            ax.set_xlim(f[0], f[-1])
            style_axis(ax, y_on_right=False)
    axes[0, 0].set_ylabel(r'Sampled $P(f)$ [flux$^2$ day]')
    axes[1, 0].set_ylabel(r'Sampled $fP(f)$ [flux$^2$]')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=11, ncol=5,
               loc='upper center', bbox_to_anchor=(0.52, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, OUT/'total_lightcurve_psd')
    with (OUT/'total_lightcurve_psd_binned.csv').open('w') as handle:
        handle.write('band,case,frequency_per_day,mean_P,mean_fP,n_ordinates\n')
        for band, case, f, p, fp, n in rows:
            handle.write(f'{band},{case},{f:.17g},{p:.17g},{fp:.17g},{int(n)}\n')
    (OUT/'total_lightcurve_psd.md').write_text('''# Total light-curve PSD comparison

Shared-S tau2, area9 simulation: W2 = disk + S + noise; U = disk + BLR + S + noise.
Each total periodogram is calculated directly from its saved input light curve;
component periodograms are not summed. The constant baseline is removed by the
floating-mean Lomb--Scargle fit.

This plot reads the existing `*_sampled_psd.csv` total columns. Unweighted
floating-mean Lomb--Scargle with normalization=psd, multiplied by 2T/N; no
interpolation, no separate variance normalization, and no noise subtraction.
These are finite-window spectra, not window-deconvolved intrinsic PSDs.

Top: arithmetic mean P(f) in the original 29 logarithmic frequency bins.
Bottom: arithmetic mean f*P(f) in exactly the same bins. fP indicates power per
log-frequency interval, subject to the sampled-periodogram limitation above.

Curves show raw and both-band Gaussian detrending at sigma=5/10/20/40 days.
The sixth case u_s20 overlaps raw exactly in W2 and all_s20 exactly in U, so no
duplicate curve is drawn. Filter formula: original minus Gaussian trend plus
original mean. This plotting script does not alter inputs or fitting jobs.
''')
    print(OUT/'total_lightcurve_psd.jpg')


if __name__ == '__main__':
    main()
