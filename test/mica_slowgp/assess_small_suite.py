"""Assess saved weighted posterior and joint reconstruction; no refitting."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import mica_slowgp
from mica_slowgp.sampler import parameter_names, read_weighted_posterior, weighted_quantile
from mica_slowgp.cli import reconstruct


CASES = ['clean', 'response_only', 'independent', 'shared']
MODES = ['none', 'independent', 'shared']
COLORS = {'none': '#C43C39', 'independent': '#1F4E9E', 'shared': '#009E73'}


def summarize(root):
    records = []
    for case in CASES:
        truth_path = root / 'small_cases' / case
        truth = json.loads((truth_path / 'truth.json').read_text())
        truth_lc = np.load(truth_path / 'truth.npz')
        for mode in MODES:
            folder = root / 'runs' / f'{case}_{mode}'
            names = parameter_names(mode)
            post = read_weighted_posterior(folder, names)
            diag = json.loads((folder / 'diagnostics.json').read_text())
            if not (folder / 'joint_reconstruction.npz').exists():
                reconstruct(folder, draws=64, points=70, seed=20261023)
            curves = np.load(folder / 'joint_reconstruction.npz')
            p = [0.025, 0.1586552539, 0.5, 0.8413447461, 0.975]
            lag = weighted_quantile(post['samples'][:, names.index('mu0')], post['weights'], p)
            lng = weighted_quantile(post['samples'][:, names.index('mu1')], post['weights'], p)
            record = dict(case=case, mode=mode, mu0_true=truth['mu0'],
                          mu0_q025=lag[0], mu0_q16=lag[1], mu0_q50=lag[2], mu0_q84=lag[3], mu0_q975=lag[4],
                          mu1_q50=lng[2], mu0_in68=bool(lag[1] <= truth['mu0'] <= lag[3]),
                          mu0_in95=bool(lag[0] <= truth['mu0'] <= lag[4]),
                          weight_ess=diag['weight_ess'], max_weight=diag['max_posterior_weight'],
                          wall_seconds=diag['elapsed_seconds'], convergence=diag['convergence_status'],
                          top_level_weight=diag['posterior_weight_above_last_level'])
            for comp in ['x', 'disk', 'blr', 'slow_w', 'slow_b']:
                expected = np.interp(curves['time'], truth_lc['time'], truth_lc[comp])
                centered_draws = curves[comp] - curves[comp].mean(axis=1, keepdims=True)
                predicted = np.median(centered_draws, axis=0)
                expected -= expected.mean()
                predicted -= predicted.mean()
                rms = float(np.sqrt(np.mean((predicted - expected)**2)))
                record[comp + '_centered_rmse'] = rms
                record[comp + '_centered_nrmse'] = rms/np.std(expected) if np.std(expected) > 1e-10 else None
            ratio = np.std(curves['disk'], axis=1)/np.std(curves['x'], axis=1)
            ratio_truth = np.std(np.interp(curves['time'], truth_lc['time'], truth_lc['disk'])) / np.std(np.interp(curves['time'], truth_lc['time'], truth_lc['x']))
            record.update(short_to_x_rms_true=float(ratio_truth),
                          short_to_x_rms_median=float(np.median(ratio)))
            records.append(record)
    with (root / 'recovery_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    (root / 'recovery_summary.json').write_text(json.dumps(records, indent=2))
    return records


def plots(root, records):
    style = Path(mica_slowgp.__file__).with_name('image-creator.mplstyle')
    with plt.style.context(style):
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.8), sharex=True, sharey=True)
        for ax, case in zip(axes.flat, CASES):
            selected = [r for r in records if r['case'] == case]
            for j, r in enumerate(selected):
                ax.plot([r['mu0_q025'], r['mu0_q975']], [j, j], color='#B8B8B8', lw=1.0)
                ax.plot([r['mu0_q16'], r['mu0_q84']], [j, j], color=COLORS[r['mode']], lw=3)
                ax.plot(r['mu0_q50'], j, marker='o', color=COLORS[r['mode']], ms=5)
            ax.axvline(2.5, color='#4169E1', linestyle=(0, (1.2, 1.8)), lw=1.25)
            ax.set_title(case.replace('_', ' ')); ax.set_yticks(range(3), MODES)
            ax.set_ylim(2.6, -0.6); ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlabel(r'Short-component centroid $\mu_0$ (d)')
        fig.tight_layout()
        fig.savefig(root / 'lag_recovery.pdf'); fig.savefig(root / 'lag_recovery.png', dpi=300)
        plt.close(fig)

        for case, mode in [('independent', 'independent'), ('shared', 'shared')]:
            curves = np.load(root / 'runs' / f'{case}_{mode}' / 'joint_reconstruction.npz')
            truth = np.load(root / 'small_cases' / case / 'truth.npz')
            fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
            for ax, comp in zip(axes, ['x', 'disk', 'blr', 'slow_w', 'slow_b']):
                # Constant modes are unidentifiable separately from the offsets;
                # compare centered trajectories for the variation-recovery test.
                draw = curves[comp] - curves[comp].mean(axis=1, keepdims=True)
                q16, median, q84 = np.quantile(draw, [0.1586552539, 0.5, 0.8413447461], axis=0)
                target = np.interp(curves['time'], truth['time'], truth[comp]); target -= target.mean()
                ax.fill_between(curves['time'], q16, q84, color='#B8B8B8', alpha=.32, lw=0)
                ax.plot(curves['time'], median, color='#1F4E9E', lw=1.7, label='Posterior median')
                ax.plot(curves['time'], target, color='#4169E1', ls=(0, (1.2, 1.8)), lw=1.25, label='Injected truth')
                ax.set_ylabel({'x':'Common W2', 'disk':'Short response', 'blr':'Long response',
                               'slow_w':'W2 slow', 'slow_b':'Response slow'}[comp])
                ax.xaxis.set_minor_locator(AutoMinorLocator(4)); ax.yaxis.set_minor_locator(AutoMinorLocator(4))
            axes[0].legend(loc='upper right', ncol=2, frameon=False)
            axes[-1].set_xlabel('Time (d)'); fig.tight_layout()
            fig.savefig(root / f'{case}_components.pdf'); fig.savefig(root / f'{case}_components.png', dpi=300)
            plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); records = summarize(args.root); plots(args.root, records)
    for r in records:
        print(r['case'], r['mode'], 'lag', round(r['mu0_q50'], 3),
              '68%', [round(r['mu0_q16'], 3), round(r['mu0_q84'], 3)],
              'short_NRMSE', round(r['disk_centered_nrmse'], 3),
              'weight_ESS', round(r['weight_ess'], 1), flush=True)
