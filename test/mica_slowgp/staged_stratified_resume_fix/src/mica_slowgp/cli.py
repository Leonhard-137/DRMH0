"""Explicit command line for the isolated, two-band experimental version."""
from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .core import PairData, condition
from .sampler import SamplerBudget, parameter_names, read_weighted_posterior, run_dnest
from .nested import load_saved_posterior, run_nested, NestedBudget


def read_data(driver, response):
    arrays = [np.loadtxt(p, ndmin=2) for p in (driver, response)]
    if any(a.shape[1] != 3 for a in arrays):
        raise ValueError('Each input must have three columns: time, flux, standard error')
    return PairData(*arrays[0].T, *arrays[1].T)


def reconstruct(folder: Path, draws=64, points=100, seed=321):
    """Draw hyperparameters with their native weights, then joint GP+offsets.

    Component curves share their conditional covariance. Both fitted offsets
    are sampled in the same Gaussian draw; totals are exact component sums.
    """
    config = json.loads((folder / 'sampler_config.json').read_text())
    mode = config['mode']
    names = parameter_names(mode)
    post = load_saved_posterior(folder, names)
    data = read_data(folder / 'input_w2.txt', folder / 'input_response.txt')
    times = np.linspace(max(data.tw.min(), data.tb.min()), min(data.tw.max(), data.tb.max()), points)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(post['samples']), size=draws, p=post['weights'])
    components = ['x', 'disk', 'blr', 'slow_w', 'slow_b']
    targets = {name: (name, times) for name in components}
    curves = {name: [] for name in components + ['w', 'b']}
    offsets = []
    min_eigenvalue = 0.0
    max_psd_tolerance = 0.0
    start = time.perf_counter()
    for index in indices:
        theta = dict(zip(names, post['samples'][index]))
        pred = condition(data, theta, targets, mode=mode)
        mean = np.r_[pred.mean, pred.offset_mean]
        cov = np.block([[pred.covariance, pred.offset_cross_covariance],
                        [pred.offset_cross_covariance.T, pred.offset_covariance]])
        values, vectors = np.linalg.eigh((cov + cov.T) / 2)
        min_eigenvalue = min(min_eigenvalue, float(values.min()))
        scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
        tolerance = 64 * len(values) * np.finfo(float).eps * scale
        max_psd_tolerance = max(max_psd_tolerance, tolerance)
        if values.min() < -tolerance:
            raise ArithmeticError(f'Joint reconstruction covariance is not PSD: {values.min()}')
        # Round-off zero modes occur for absent or exactly shared components.
        # Only the draw square-root clips these negligible negative modes.
        z = mean + vectors @ (np.sqrt(np.maximum(values, 0)) * rng.normal(size=len(values)))
        for name in components:
            curves[name].append(z[pred.slices[name]])
        offsets.append(z[-2:])
        curves['w'].append(z[-2] + curves['x'][-1] + curves['slow_w'][-1])
        curves['b'].append(z[-1] + curves['disk'][-1] + curves['blr'][-1] + curves['slow_b'][-1])
    arrays = {key: np.asarray(value) for key, value in curves.items()}
    np.savez_compressed(folder / 'joint_reconstruction.npz', time=times,
                        parameter_indices=indices, offsets=np.asarray(offsets), **arrays)
    summary = {'draws': draws, 'seed': seed, 'points': points,
               'min_joint_covariance_eigenvalue': min_eigenvalue,
               'max_roundoff_psd_tolerance': max_psd_tolerance,
               'elapsed_seconds': time.perf_counter() - start,
               'intervals': 'pointwise posterior including parameter, GP, and offset uncertainty',
               'caveat': 'Finite-budget sampler uncertainty remains; disk/blr are short/long labels, not established physical identities.'}
    (folder / 'reconstruction_metadata.json').write_text(json.dumps(summary, indent=2))
    for name, values in arrays.items():
        quant = np.quantile(values, [0.1586552539, 0.5, 0.8413447461], axis=0)
        np.savetxt(folder / f'{name}_reconstruction.txt', np.column_stack([times, *quant]),
                   header='time q16 q50 q84')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit', help='Fit one W2-response pair with a selected nested sampler')
    fit.add_argument('--driver', type=Path, required=True)
    fit.add_argument('--response', type=Path, required=True)
    fit.add_argument('--mode', choices=['none', 'independent', 'shared'], required=True)
    fit.add_argument('--priors', type=Path, required=True)
    fit.add_argument('--out', type=Path, required=True)
    fit.add_argument('--seed', type=int, default=12345)
    fit.add_argument('--sampler', choices=['dnest', 'dynesty'], default='dnest')
    fit.add_argument('--likelihood', choices=['reference', 'fast'], default='reference')
    fit.add_argument('--resume', action='store_true', help='Resume a dynesty checkpoint with identical configuration')
    fit.add_argument('--budget', type=Path, help='JSON fields of the selected sampler budget')
    fit.add_argument('--center-strata', type=float, nargs=4, metavar=('A','EDGE1','EDGE2','B'),
                     help='Integrate all three sorted-mu1 strata using unchanged full priors (dynesty only)')
    fit.add_argument('--saves', type=int, default=1000)
    fit.add_argument('--draws', type=int, default=0, help='Optional joint reconstruction draws after fit')
    rec = sub.add_parser('reconstruct', help='Joint posterior component reconstruction')
    rec.add_argument('--out', type=Path, required=True)
    rec.add_argument('--draws', type=int, default=64)
    rec.add_argument('--points', type=int, default=100)
    rec.add_argument('--seed', type=int, default=321)
    args = parser.parse_args(argv)
    # Explicitly reject accidental multi-rank invocation before any writes.
    os.environ.setdefault('UCX_TLS', 'self')
    os.environ.setdefault('UCX_NET_DEVICES', 'lo')
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError('This experimental CLI supports one MPI rank per pair; run separate pairs as separate jobs')
    if args.command == 'reconstruct':
        print(json.dumps(reconstruct(args.out, args.draws, args.points, args.seed), indent=2))
        return
    from .core import log_likelihood
    data = read_data(args.driver, args.response)
    if args.center_strata is not None:
        if args.sampler != 'dynesty':
            raise ValueError('--center-strata requires --sampler dynesty')
        from .stratified import run_stratified
        budget = NestedBudget(**json.loads(args.budget.read_text())) if args.budget else NestedBudget()
        likelihood = partial(log_likelihood, data, mode=args.mode)
        if args.likelihood == 'fast':
            from .fast import FastPairLikelihood
            likelihood = FastPairLikelihood(data, mode=args.mode)
        metadata = dict(driver=str(args.driver.resolve()),response=str(args.response.resolve()),
            interpreter=sys.executable,mode=args.mode,sampler=args.sampler,likelihood=args.likelihood,
            likelihood_backend=getattr(likelihood,'backend','reference'),
            points=[len(data.tw),len(data.tb)],center_strata=args.center_strata,
            flux_units='same user-supplied units throughout; no normalization or host subtraction')
        result = run_stratified(likelihood,args.out,mode=args.mode,priors=args.priors,
            budget=budget,seed=args.seed,center_strata=args.center_strata,resume=args.resume,
            input_files=(args.driver,args.response),run_input=metadata)
        print(json.dumps({'posterior':result['summary'],'diagnostics':result['diagnostics']},indent=2))
        if args.draws:
            print(json.dumps(reconstruct(args.out,args.draws,seed=args.seed+1),indent=2))
        return
    if (args.out / 'weighted_posterior.csv').exists() or ((args.out / 'sample.txt').exists() and not args.resume):
        raise FileExistsError(f'Existing run: {args.out}; choose a new output directory')
    if args.resume and args.sampler != 'dynesty':
        raise ValueError('Resume is implemented only for dynesty checkpoints')
    if args.resume:
        if not (args.out / 'dynesty.checkpoint').is_file():
            raise FileNotFoundError('Resume requires an existing dynesty checkpoint')
        for source, name in ((args.driver, 'input_w2.txt'), (args.response, 'input_response.txt')):
            saved = args.out / name
            if not saved.is_file() or source.read_bytes() != saved.read_bytes():
                raise ValueError(f'Resume input differs from the saved run: {name}')
        previous_input = json.loads((args.out / 'run_input.json').read_text())
        for name, value in [('mode', args.mode), ('sampler', args.sampler), ('likelihood', args.likelihood)]:
            if previous_input[name] != value:
                raise ValueError(f'Resume requires the same {name}')
    elif (args.out / 'sampler_config.json').exists():
        raise FileExistsError(f'An unfinished run exists: {args.out}; choose a new output directory or resume its checkpoint')
    args.out.mkdir(parents=True, exist_ok=True)
    for source, name in ((args.driver, 'input_w2.txt'), (args.response, 'input_response.txt')):
        (args.out / name).write_bytes(source.read_bytes())
    if args.sampler == 'dynesty':
        budget = NestedBudget(**json.loads(args.budget.read_text())) if args.budget else NestedBudget()
    else:
        budget = SamplerBudget(**json.loads(args.budget.read_text())) if args.budget else SamplerBudget(max_num_saves=args.saves)
    likelihood = partial(log_likelihood, data, mode=args.mode)
    if args.likelihood == 'fast':
        from .fast import FastPairLikelihood
        likelihood = FastPairLikelihood(data, mode=args.mode)
    (args.out / 'run_input.json').write_text(json.dumps({
        'driver': str(args.driver.resolve()), 'response': str(args.response.resolve()),
        'interpreter': sys.executable, 'mode': args.mode,
        'sampler': args.sampler, 'likelihood': args.likelihood,
        'likelihood_backend': getattr(likelihood, 'backend', 'reference'),
        'points': [len(data.tw), len(data.tb)],
        'flux_units': 'same user-supplied units throughout; no normalization or host subtraction',
    }, indent=2))
    if args.sampler == 'dynesty':
        result = run_nested(likelihood, args.out, mode=args.mode, priors=args.priors,
                            budget=budget, seed=args.seed, resume=args.resume)
    else:
        result = run_dnest(likelihood, args.out, mode=args.mode, priors=args.priors,
                           budget=budget, seed=args.seed)
    print(json.dumps({'posterior': result['summary'], 'diagnostics': result['diagnostics']}, indent=2))
    if args.draws:
        print(json.dumps(reconstruct(args.out, args.draws, seed=args.seed + 1), indent=2))


if __name__ == '__main__':
    main()
