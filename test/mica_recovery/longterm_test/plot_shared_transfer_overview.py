#!/usr/bin/env python3
"""Display the completed six-case shared-S posteriors; never launch fits."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit import build_plan
from plot_results import load_run, transfer_grid, transfer_quantiles
from plot_style import configure_style, style_axis, save_figure, COLORS, SHORT_DASH

HERE = Path(__file__).resolve().parent
CONFIG = HERE/'experiment_tau2_area9_shared.yaml'
ROOT = HERE/'cases_tau2_area9_shared'
OUT = ROOT/'diagnostics/transfer_overview'
LABELS = {'raw': 'Raw', 'all_s5': r'Both $\sigma=5$ d',
          'all_s10': r'Both $\sigma=10$ d', 'all_s20': r'Both $\sigma=20$ d',
          'all_s40': r'Both $\sigma=40$ d', 'u_s20': r'U only $\sigma=20$ d'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    for family in ('gamma', 'gaussian'):
        plan = build_plan(CONFIG, suite='paired_ablation', model=family)
        fig, axes = plt.subplots(3, 2, figsize=(11.5, 10))
        for ax, job in zip(axes.flat, plan):
            run = load_run(job)
            summary = json.loads((run['case']/f'results/{family}/summary.json').read_text())
            limits = summary['transfer']['lag_limits_days']
            lag = transfer_grid(run['centers'], run['widths'], family, limits)
            total, parts = transfer_quantiles(lag, run['amp'], run['centers'], run['widths'], family)
            positive = lambda y: np.ma.masked_less_equal(y, 0)
            ax.fill_between(lag, positive(total[:, 0]), positive(total[:, 2]),
                            color=COLORS['UNCERTAINTY'], alpha=.40, lw=0,
                            label='Total 68.3% interval')
            ax.plot(lag, positive(total[:, 1]), color=COLORS['INK'], lw=1.7, label='Total median')
            for j, color in enumerate((COLORS['PRIMARY'], COLORS['ALTERNATIVE'])):
                ax.plot(lag, positive(parts[j]), color=color, lw=1.35,
                        ls=SHORT_DASH, label=f'C{j} median')
            ax.axvline(0, color=COLORS['GUIDE'], lw=.8, zorder=0)
            maximum = float(np.max(total[:, 2]))
            ax.set_yscale('log')
            ax.set_ylim(maximum*1e-6, maximum*1.3)
            ax.set_xlim(*limits)
            ax.set_title(LABELS[run['case'].name], loc='left', fontsize=13)
            ax.set_xlabel(r'$\tau$ [days]')
            ax.set_ylabel(r'Response [day$^{-1}$]')
            style_axis(ax, y_on_right=False)
            print(f"plotted {run['case'].name}/{family}", flush=True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .963))
        save_figure(fig, OUT/f'{family}_six_cases')
    (OUT/'README.md').write_text('''# Completed shared-S recovery transfer functions

Six-case order: raw, both sigma5, both sigma10, both sigma20, both sigma40,
U-only sigma20. Both models use all existing posterior draws, reordered by
centroid within each joint draw: C0 has smaller centroid; C1 has larger.
Black: pointwise total median, after summing both components within each draw.
Gray: total pointwise 15.85--84.15 percentiles. Blue/red: component pointwise
medians; their sum need not equal the total pointwise median.

Log response axes show weak long tails. Each panel uses the same full lag
limits as its existing case-local transfer_function plot (central 99.9% mass
of the posterior-averaged unit-area total response); axes differ by panel.
Native normalized-flux response amplitudes are retained without rescaling.
All amplitudes are constrained positive. Response at negative lag is not
negative response amplitude. No physical input kernel is overlaid because
the central-source response and observed W2-to-U predictor are distinct.

Existing original linear/log figures remain in CASE/results/MODEL/.
u_s20/Gamma has only 45 posterior draws (ESS about 45.9); this plot does not
establish convergence. No fitting, input regeneration or prior changes occur.
''')
    print(OUT, flush=True)


if __name__ == '__main__':
    main()
