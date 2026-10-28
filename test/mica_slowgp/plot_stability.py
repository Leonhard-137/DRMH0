#!/usr/bin/env python3
"""Read completed weighted posteriors and plot the fixed two-row comparison.

Example (six runs per row: three seeds, two budgets):
  python -B plot_stability.py --runs-root OUTPUT/stability_v1 --sampler dynesty \
      --budgets 400 800 --seeds 101 102 103 --out OUTPUT/stability_comparison

Missing runs block a formal figure. --allow-incomplete explicitly makes a
temporary figure whose title contains 'incomplete'. No fit, posterior
resampling, CDF smoothing, truth lookup, or stability-gate modification occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from assess_stability import (
    DEFAULT_INPUTS, GATE_PATH, normalized_weights, read_json, read_run,
    run_identity, transfer_grid,
)


DEFAULT_STYLE = Path('/home/leonhard/software/MICA2-slowgp/src/mica_slowgp/image-creator.mplstyle')
ROWS = (('independent', 'independent'), ('shared', 'shared'))
COLORS = ('#1F4E9E', '#C43C39', '#E69F00')
# Requested semantics: color identifies seed, line style identifies budget.
# This overrides the generic style's model-role interpretation of line styles.
BUDGET_STYLES = ((0, (5.0, 2.2)), '-')


def weighted_ecdf(values, weights):
    """Exact right-continuous weighted ECDF, with tied atoms combined."""
    values, weights = np.asarray(values, dtype=float), normalized_weights(weights)
    if values.shape != weights.shape or not np.isfinite(values).all():
        raise ValueError('ECDF values and weights must have matching finite shapes')
    positive = weights > 0
    order = np.argsort(values[positive], kind='stable')
    values, weights = values[positive][order], weights[positive][order]
    locations, first = np.unique(values, return_index=True)
    masses = np.add.reduceat(weights, first)
    cumulative = np.cumsum(masses)
    cumulative[-1] = 1.0
    return locations, cumulative


def total_tf_mean(columns, weights, grid, maximum_entries=1_000_000):
    """All-weight posterior mean of A0*G0+A1*G1; no sample thinning.

    Exactly identical transfer-function parameter rows are combined by adding
    their weights. This changes neither the estimator nor its normalization.
    Work matrices are bounded in size, following the existing TF-grid policy.
    """
    weights = normalized_weights(weights)
    names = ('A0', 'mu0', 'w0', 'A1', 'mu1', 'w1')
    parameters = np.column_stack([columns[name] for name in names])
    if parameters.shape[0] != len(weights) or not np.isfinite(parameters).all():
        raise ValueError('Invalid transfer-function posterior columns')
    if np.any(parameters[:, [2, 5]] <= 0) or np.any(parameters[:, [0, 3]] < 0):
        raise ValueError('TF amplitudes must be nonnegative and Gaussian widths positive')
    positive = weights > 0
    parameters, inverse = np.unique(parameters[positive], axis=0, return_inverse=True)
    weights = np.bincount(inverse, weights=weights[positive])
    grid = np.asarray(grid, dtype=float)
    mean = np.zeros(len(grid))
    block = max(1, min(128, int(maximum_entries)//len(weights)))
    root_two_pi = np.sqrt(2*np.pi)
    for start in range(0, len(grid), block):
        stop = min(start+block, len(grid))
        times = grid[None, start:stop]
        values = np.zeros((len(weights), stop-start))
        for offset in (0, 3):
            amplitude = parameters[:, offset, None]
            center = parameters[:, offset+1, None]
            width = parameters[:, offset+2, None]
            values += amplitude/(root_two_pi*width)*np.exp(-0.5*((times-center)/width)**2)
        mean[start:stop] = weights @ values
    return mean


def tf_display_range(priors):
    """Display-only crop: center prior bounds padded by one largest width.

    The full transfer calculation still uses the unchanged eight-width grid.
    """
    width=max(priors[name]['bounds'][1] for name in ('w0','w1'))
    return (min(priors[name]['bounds'][0] for name in ('mu0','mu1'))-width,
            max(priors[name]['bounds'][1] for name in ('mu0','mu1'))+width)


def collect_runs(runs_root, sampler, budgets, seeds, inputs_root, gate):
    """Select by metadata only; never choose a run by its posterior or truth."""
    candidates = {}
    for config_path in sorted(Path(runs_root).rglob('sampler_config.json')):
        folder = config_path.parent
        config, case, mode, actual_sampler = run_identity(folder)
        if (case, mode) not in ROWS or actual_sampler != sampler:
            continue
        effort = float(config['budget'][gate['effort_fields'][sampler]])
        if effort not in budgets:
            continue
        seed = int(config['seed'])
        if seeds is not None and seed not in seeds:
            continue
        key = (case, mode, seed, effort)
        if key in candidates:
            raise ValueError(f'Duplicate run identity {key}: {candidates[key]} and {folder}; choose a narrower runs root')
        candidates[key] = folder
    inferred = seeds is None
    if inferred:
        seeds = sorted({key[2] for key in candidates})
        if len(seeds) > 3:
            raise ValueError('More than three seed identities found; specify --seeds explicitly')
    seeds = list(seeds)
    missing = []
    if len(seeds) != 3:
        missing.append({'reason': f'Expected three seed identities; found {len(seeds)}', 'seeds': seeds})
    available = {}
    for case, mode in ROWS:
        for seed in seeds:
            for effort in budgets:
                key = (case, mode, seed, effort)
                folder = candidates.get(key)
                absent = ['run directory'] if folder is None else [
                    name for name in ('weighted_posterior.csv', 'posterior_summary.json', 'diagnostics.json')
                    if not (folder/name).is_file()]
                if absent:
                    missing.append({'case': case, 'mode': mode, 'seed': seed, 'budget': effort,
                                    'path': None if folder is None else str(folder), 'missing': absent})
                else:
                    available[key] = read_run(folder, inputs_root, gate)
    # Enforce equal input/prior and non-effort controls within each displayed row.
    for case, mode in ROWS:
        group = [run for key, run in available.items() if key[:2] == (case, mode)]
        if not group:
            continue
        effort_field = gate['effort_fields'][sampler]
        baseline = {k:v for k,v in group[0].config['budget'].items() if k != effort_field}
        for run in group[1:]:
            controls = {k:v for k,v in run.config['budget'].items() if k != effort_field}
            if controls != baseline:
                raise ValueError(f'Non-budget controls differ in {case}/{mode}; refusing to label this a budget comparison')
    return available, seeds, missing, inferred


def render(available, seeds, budgets, sampler, style, gate, incomplete):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator

    with plt.style.context(style):
        fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.3))
        fig.subplots_adjust(left=.085, right=.985, bottom=.10, top=.84,
                            wspace=.28, hspace=.33)
        title = f'{sampler}: two-band posterior comparison'
        if incomplete:
            title += ' (incomplete)'
        fig.suptitle(title, fontsize=12, y=.992)
        handles = [Line2D([], [], color=COLORS[i], lw=1.5, label=f'Seed {seed}')
                   for i, seed in enumerate(seeds)]
        budget_name = r'$N_{\rm live}$' if sampler == 'dynesty' else 'Saves'
        handles += [Line2D([], [], color='#222222', lw=1.5, linestyle=BUDGET_STYLES[j],
                           label=f'{budget_name} {budget:g}') for j, budget in enumerate(budgets)]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53,.957),
                   ncol=min(5,len(handles)), frameon=False, columnspacing=1.7)
        for row, (case, mode) in enumerate(ROWS):
            row_runs = [run for key, run in available.items() if key[:2] == (case,mode)]
            grid = transfer_grid(row_runs[0].config['priors'], gate) if row_runs else None
            for i, seed in enumerate(seeds):
                for j, budget in enumerate(budgets):
                    run = available.get((case, mode, seed, budget))
                    if run is None:
                        continue
                    kwargs = dict(color=COLORS[i], linestyle=BUDGET_STYLES[j], linewidth=1.35)
                    for column, name in enumerate(('mu0', 'mu1')):
                        x, y = weighted_ecdf(run.columns[name], run.weights)
                        lo, hi = run.config['priors'][name]['bounds']
                        # Draw the actual ECDF atoms with a post-step; no spline,
                        # KDE, probability interpolation, or posterior resample.
                        axes[row,column].step(np.r_[lo,x,hi], np.r_[0.,y,1.], where='post', **kwargs)
                        axes[row,column].set_xlim(lo,hi)
                    mean = total_tf_mean(run.columns, run.weights, grid,
                                         gate['tf_grid']['maximum_working_matrix_entries'])
                    axes[row,2].plot(grid,mean,**kwargs)
            for column, name in enumerate(('mu0','mu1')):
                axes[row,column].set_ylim(0.,1.)
                axes[row,column].set_xlabel(r'$\mu_'+str(column)+r'$ (day)')
                axes[row,column].set_yticks([0.,.25,.5,.75,1.])
            axes[row,0].set_ylabel(f'{case.capitalize()} / {mode}\nWeighted ECDF')
            axes[row,2].set_xlabel(r'$\tau$ (day)')
            axes[row,2].set_ylabel(r'$\langle\Psi_{\rm total}(\tau)\rangle$ (day$^{-1}$)')
            axes[row,2].set_ylim(bottom=0.)
            if grid is not None:
                axes[row,2].set_xlim(*tf_display_range(row_runs[0].config['priors']))
            for ax in axes[row]:
                ax.xaxis.set_minor_locator(AutoMinorLocator(4))
                ax.yaxis.set_minor_locator(AutoMinorLocator(4))
            if row == 0:
                for ax, heading in zip(axes[row], (r'$\mu_0$ weighted ECDF',r'$\mu_1$ weighted ECDF','Posterior mean total TF')):
                    ax.set_title(heading, fontsize=11)
        return fig


def self_test():
    x,y=weighted_ecdf([2.,1.,2.,0.],[.2,.3,.5,0.])
    assert np.array_equal(x,[1.,2.]) and np.allclose(y,[.3,1.])
    grid=np.array([-1.,0.,1.])
    cols={name:np.array([value,value]) for name,value in
          dict(A0=1.,mu0=0.,w0=1.,A1=.5,mu1=0.,w1=2.).items()}
    actual=total_tf_mean(cols,[.25,.75],grid,maximum_entries=4)
    expected=np.exp(-grid**2/2)/np.sqrt(2*np.pi)+.5*np.exp(-grid**2/8)/(2*np.sqrt(2*np.pi))
    assert np.allclose(actual,expected,rtol=1e-14,atol=0.)
    # Tiny synthetic render stays in memory: it is not a scientific figure.
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    priors={'mu0':{'bounds':[-3.,28.]},'mu1':{'bounds':[-3.,28.]},
            'w0':{'bounds':[.1,12.]},'w1':{'bounds':[.1,12.]}}
    run=SimpleNamespace(columns=cols,weights=np.array([.25,.75]),config={'priors':priors})
    dummy={(case,mode,seed,budget):run for case,mode in ROWS
           for seed in (1,2,3) for budget in (400.,800.)}
    fig=render(dummy,[1,2,3],[400.,800.],'dynesty',DEFAULT_STYLE,read_json(GATE_PATH),True)
    fig.canvas.draw()
    assert len(fig.axes)==6 and all(len(ax.lines)==6 for ax in fig.axes)
    assert 'incomplete' in fig._suptitle.get_text()
    plt.close(fig)
    print(json.dumps({'self_test':'passed','checks':['weighted ECDF ties/zero weight',
          'all-weight Gaussian TF mean/duplicate aggregation','in-memory 2x3 renderer and incomplete title']}))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root',type=Path)
    parser.add_argument('--sampler',choices=('dnest','dynesty'),default='dynesty')
    parser.add_argument('--budgets',type=float,nargs=2,metavar=('LOW','HIGH'))
    parser.add_argument('--seeds',type=int,nargs=3,metavar=('S1','S2','S3'))
    parser.add_argument('--out',type=Path,help='Output stem; PDF, 300-dpi PNG, and JSON metadata are written')
    parser.add_argument('--inputs-root',type=Path,default=DEFAULT_INPUTS)
    parser.add_argument('--style',type=Path,default=DEFAULT_STYLE)
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--check-only',action='store_true',help='Report completeness without plotting or writing files')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv)
    if args.self_test:
        self_test();return 0
    if args.runs_root is None or args.budgets is None or (args.out is None and not args.check_only):
        parser.error('--runs-root, --budgets, and --out (unless --check-only) are required')
    budgets=sorted(args.budgets)
    if not np.isfinite(budgets).all() or budgets[0]<=0 or budgets[0]==budgets[1]:
        parser.error('provide two distinct positive finite budgets')
    if args.seeds is not None and len(set(args.seeds))!=3:
        parser.error('--seeds must contain three distinct seed values')
    gate=read_json(GATE_PATH)
    available,seeds,missing,inferred=collect_runs(args.runs_root,args.sampler,budgets,args.seeds,args.inputs_root,gate)
    metadata=dict(complete=not missing,sampler=args.sampler,seeds=seeds,seeds_inferred=inferred,
                  budgets=budgets,expected_runs=12,available_runs=len(available),missing=missing,
                  selection='Fixed independent/independent and shared/shared rows; metadata only, no truth read',
                  weighted_ecdf='Exact right-continuous empirical steps; tied atoms combined',
                  total_tf='All-weight posterior mean of both unit-area Gaussian kernels times A0/A1',
                  tf_grid='Unchanged stability_gate_v1 prior-padded grid; no posterior sample thinning',
                  style=str(args.style.resolve()),run_paths=[str(run.path) for run in available.values()],
                  evidence_boundary='Visual posterior comparison; completeness does not assert stability-gate passage or convergence')
    metadata['display_crop']={}
    metadata['full_integration_range']={}
    for case,mode in ROWS:
        row_runs=[run for key,run in available.items() if key[:2]==(case,mode)]
        if row_runs:
            priors=row_runs[0].config['priors']
            full_grid=transfer_grid(priors,gate)
            metadata['display_crop'][f'{case}/{mode}']=list(tf_display_range(priors))
            metadata['full_integration_range'][f'{case}/{mode}']=[float(full_grid[0]),float(full_grid[-1])]
    if args.check_only:
        print(json.dumps(metadata,indent=2));return 0 if not missing else 2
    if missing and not args.allow_incomplete:
        print(json.dumps(metadata,indent=2))
        parser.error('Incomplete required run set; refusing a formal figure. Use --allow-incomplete only for a temporary figure.')
    if not available:
        parser.error('No completed matching posterior is available; no data figure can be made')
    if not args.style.is_file():
        parser.error(f'Project style file is missing: {args.style}')
    fig=render(available,seeds,budgets,args.sampler,args.style,gate,bool(missing))
    stem=args.out.with_suffix('') if args.out.suffix.lower() in ('.pdf','.png') else args.out
    stem.parent.mkdir(parents=True,exist_ok=True)
    paths={kind:str(stem.with_name(stem.name+suffix)) for kind,suffix in
           [('pdf','.pdf'),('png','.png'),('metadata','.json')]}
    fig.savefig(paths['pdf'])
    fig.savefig(paths['png'],dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)
    metadata['outputs']=paths
    Path(paths['metadata']).write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
