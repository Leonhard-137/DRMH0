#!/usr/bin/env python3
"""Prepare the fixed six-job Mrk142 first pass; never launch a fit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path('/home/leonhard/sci001')
SOURCE = Path('/home/leonhard/software/MICA2-slowgp/archive/stability-v3-stratified-fix1/src')
PYTHON = Path('/home/leonhard/miniforge3/envs/mica2-slowgp/bin/python')
HERE = PROJECT / 'test/mica_slowgp/mrk142_firstpass'
DEFAULT_ROOT = PROJECT / 'output/mica_slowgp/mrk142_firstpass'
BANDS = ('W2', 'M2', 'W1', 'U', 'B', 'V')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def normalize(array):
    """Keep observed times; divide both flux and errors by the positive mean."""
    array = np.asarray(array, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) < 3:
        raise ValueError('Expected at least three time/flux/error rows')
    if not np.isfinite(array).all() or np.any(array[:, 2] <= 0):
        raise ValueError('Nonfinite data or nonpositive measurement error')
    scale = float(np.mean(array[:, 1]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Mean-flux normalization requires a positive mean')
    result = array.copy()
    result[:, 1:] /= scale
    return result, scale


def physical_gains(amplitude, driver_scale, response_scale):
    return np.asarray(amplitude) * response_scale / driver_scale


def width_bounds(path):
    found = {}
    for line in path.read_text().splitlines():
        words = line.split()
        if len(words) >= 5 and words[1] in ('0-th_component_sigma', '1-th_component_sigma'):
            if words[2] != 'LOG':
                raise ValueError(f'Expected native LOG sigma bounds in {path}')
            found[words[1]] = np.exp([float(words[3]), float(words[4])]).tolist()
    if len(found) != 2 or found['0-th_component_sigma'] != found['1-th_component_sigma']:
        raise ValueError(f'Both exchangeable width priors are required in {path}')
    return found['0-th_component_sigma']


def make_priors(width):
    def prior(kind, low, high):
        return dict(kind=kind, bounds=[low, high])
    p = dict(sigma_x=prior('log_uniform', 1e-3, 10.0),
             tau_x=prior('log_uniform', 0.1, 1e4),
             qW=prior('uniform', 0.0, 2.5), qb=prior('uniform', 0.0, 2.5),
             Ts=prior('log_uniform', 5.0, 1000.0))
    for index in (0, 1):
        p[f'A{index}'] = prior('log_uniform', 1e-5, 10.0)
        p[f'mu{index}'] = prior('uniform', 0.0, 100.0)
        p[f'w{index}'] = prior('log_uniform', *width)
    return p


def prepare(root=DEFAULT_ROOT, project=PROJECT, source=SOURCE, python=PYTHON):
    root, project, source, python = map(Path, (root, project, source, python))
    if root.exists():
        raise FileExistsError(f'Choose a new output root; refusing to replace {root}')
    baseline = project / 'Mrk142/runs/mica/gaussian/free_raw/2comp'
    data = {}
    for band in BANDS:
        original = project / 'Mrk142/prepared' / f'Mrk142_{band}.txt'
        array, scale = normalize(np.loadtxt(original, ndmin=2))
        data[band] = (original, array, scale)
    widths = {band: width_bounds(baseline / f'run_W2_to_{band}_2comp_gaussian/data/para_names_line.txt_2')
              for band in BANDS[1:]}
    if not (source / 'mica_slowgp/stratified.py').is_file() or not python.is_file():
        raise FileNotFoundError('Frozen isolated implementation and interpreter are required')
    # Validate all data and native prior inputs before creating the new root.
    root.mkdir(parents=True, exist_ok=False)
    for directory in ('inputs', 'priors', 'runs'):
        (root / directory).mkdir()
    normalization = {}
    for band, (original, array, scale) in data.items():
        path = root / 'inputs' / f'Mrk142_{band}_mean_normalized.txt'
        np.savetxt(path, array, fmt='%.17g', header='MJD flux_over_raw_mean error_over_raw_mean')
        normalization[band] = dict(scale=scale, n=len(array), input=str(original), normalized_path=str(path),
            input_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
            time_min=float(array[:, 0].min()), time_max=float(array[:, 0].max()))
    budget = dict(nlive=400, dlogz=0.05, bound='multi', sample='rslice', walks=50,
                  bootstrap=0, checkpoint_every=30.0, slices=14, update_interval=50.0, facc=0.5)
    budget_path = root / 'budget.json'
    write_json(budget_path, budget)
    priors_by_band = {}
    for band, width in widths.items():
        path = root / 'priors' / f'{band}.json'
        write_json(path, make_priors(width))
        priors_by_band[band] = path
    specs = [(b, 'primary', 14201 + i) for i, b in enumerate(BANDS[1:])] + [('U', 'repeat', 14213)]
    jobs = []
    for band, role, seed in specs:
        identifier = f'{band}_{role}_{seed}'
        out = root / 'runs' / identifier
        driver, response = normalization['W2']['normalized_path'], normalization[band]['normalized_path']
        command = [str(python), '-B', '-m', 'mica_slowgp.cli', 'fit', '--driver', driver,
                   '--response', response, '--mode', 'shared', '--priors', str(priors_by_band[band]),
                   '--out', str(out), '--seed', str(seed), '--sampler', 'dynesty', '--likelihood', 'fast',
                   '--budget', str(budget_path), '--center-strata', '0', '5', '20', '100']
        jobs.append(dict(id=identifier, band=band, role=role, seed=seed, out_dir=str(out),
                         driver=driver, response=response, priors=str(priors_by_band[band]), command=command))
    environment = {name: '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    environment.update(PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1', UCX_TLS='self', UCX_NET_DEVICES='lo')
    manifest = dict(schema='mrk142-firstpass-v1', created_at_utc=utcnow(), root=str(root), jobs=jobs,
        normalization=normalization, baseline_root=str(baseline),
        normalization_definition='normalized flux and error = original values / each-band raw arithmetic mean; times unchanged',
        physical_conversion=dict(gain='A_physical = A_normalized * response_scale / W2_scale',
             driver_sd='sigma_x_physical and qW_physical = normalized value * W2_scale',
             response_slow_sd='qb_physical = qb_normalized * response_scale',
             curves='Multiply each reconstructed curve and offset by its own band scale',
             input_flux_unit='Project convention: 1e-15 erg s^-1 cm^-2 Angstrom^-1'),
        run_settings=dict(mode='shared', max_parallel=6, global_fit_deadline_seconds=21600,
             interrupt_grace_seconds=120, terminate_grace_seconds=30, poll_seconds=10,
             monitor_interval_seconds=1800, source=str(source), python=str(python), cwd=str(project),
             environment=environment, budget=budget, center_strata=[0.0, 5.0, 20.0, 100.0],
             stratum_prior_masses=[0.0025, 0.0375, 0.96], nlive_definition='400 independently in each of three strata',
             postprocess_outside_fit_deadline=True, automatic_resume=False, automatic_budget_increase=False,
             checkpoint_policy='Preserve the latest periodic dynesty checkpoint; signal interruption need not save a newer checkpoint'),
        postprocess_command=[str(python), '-B', str(project / 'test/mica_slowgp/mrk142_firstpass/finish.py'), '--root', str(root)])
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'status.json', dict(schema='mrk142-supervisor-status-v1', state='prepared', updated_at_utc=utcnow(),
                jobs=[dict(id=j['id'], state='prepared') for j in jobs]))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = prepare(args.root)
    print(json.dumps(dict(manifest=str(args.root / 'manifest.json'), jobs=len(result['jobs']), launched=False)))


if __name__ == '__main__':
    main()
