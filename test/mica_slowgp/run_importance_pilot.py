#!/usr/bin/env python3
"""Bounded independent importance-sampling pilot for the unchanged pair model.

Proposals must already be frozen. This script never trains on evaluation
samples, clips weights, resumes unknown output, or declares a stability pass.
All three conditional center strata retain their exact original prior mass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import expit, logsumexp


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def longdouble_transform(z, transform):
    """Independent precision check of the same continuous conditional map."""
    z = np.asarray(z, dtype=np.longdouble)
    u = np.empty_like(z)
    positive = z >= 0
    with np.errstate(under='ignore'):
        u[positive] = 1/(1+np.exp(-z[positive]))
        ez = np.exp(z[~positive])
        u[~positive] = ez/(1+ez)
    bounds = np.asarray(transform.bounds, dtype=np.longdouble)
    linear = bounds.copy()
    linear[transform.logscale] = np.log(linear[transform.logscale])
    x = linear[:,0]+u*(linear[:,1]-linear[:,0])
    x[transform.logscale] = np.exp(x[transform.logscale])
    a = bounds[3,0]
    lo,hi = np.asarray(transform.center_interval,dtype=np.longdouble)
    x[6] = a+np.sqrt((lo-a)**2+u[6]*((hi-a)**2-(lo-a)**2))
    x[3] = a+u[3]*(x[6]-a)
    return np.asarray(x,dtype=float)


def capacity_check():
    """Count actual processes; an external live fit is not a queued launch."""
    active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().decode().split('\0')
            native = any(argv[i:i+3] == ['-m', 'mica_slowgp.cli', 'fit']
                         for i in range(len(argv)-2))
            pilot = any(Path(arg).name == Path(__file__).name for arg in argv if arg)
            if native or pilot:
                active.append(dict(pid=int(entry.name), argv=argv))
        except (OSError, UnicodeError):
            continue
    if len(active) >= 4:
        raise RuntimeError('All four scientific fit slots are occupied')
    registry = Path('/home/leonhard/sci001/output/mica_slowgp/_stability_scheduler/registry.json')
    if registry.exists():
        state = json.loads(registry.read_text())
        manifest = json.loads(Path(state['manifest']).read_text())
        if any(job['status'] in ('planned', 'starting') for job in manifest['jobs']):
            raise RuntimeError('An existing scheduler still has queued work')
    return active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--proposal-module-root', type=Path, required=True)
    parser.add_argument('--proposal', type=Path, action='append', required=True)
    parser.add_argument('--inputs-root', type=Path, required=True)
    parser.add_argument('--case', choices=('clean', 'response_only', 'independent', 'shared'), required=True)
    parser.add_argument('--mode', choices=('none', 'independent', 'shared'), required=True)
    parser.add_argument('--draws-per-stratum', type=int, default=16384)
    parser.add_argument('--seed', type=int, default=82103)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if len(args.proposal) != 3 or args.draws_per_stratum < 2:
        raise ValueError('Three frozen proposals and at least two draws per stratum are required')
    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(args.proposal_module_root.resolve()))
    from importance import FrozenProposal
    from mica_slowgp.core import PairData
    from mica_slowgp.fast import FastPairLikelihood
    from mica_slowgp.nested import PriorTransform
    from mica_slowgp.stratified import strata_definition
    sys.path.insert(0, str(Path(__file__).parent))
    import assess_stability as assess

    active = capacity_check()
    args.out.mkdir(parents=True, exist_ok=False)
    paths = [args.inputs_root / args.case / name for name in ('w2.txt', 'response.txt')]
    priors_path = args.inputs_root / 'priors.json'
    arrays = [np.loadtxt(path, ndmin=2) for path in paths]
    if any(array.shape != (42, 3) for array in arrays):
        raise ValueError('The frozen small-case contract requires 42 points per band')
    data = PairData(*arrays[0].T, *arrays[1].T)
    likelihood = FastPairLikelihood(data, mode=args.mode, backend='compiled')
    transform, edges, masses = strata_definition(args.mode, priors_path, [-3, 5, 20, 28])
    proposals = [FrozenProposal.load(path) for path in args.proposal]
    child_seeds = [int(s.generate_state(1, dtype=np.uint32)[0])
                   for s in np.random.SeedSequence(args.seed).spawn(3)]
    config = dict(sampler='frozen_importance', scope='pilot only; no formal stability claim',
        case=args.case, mode=args.mode, seed=args.seed, child_seeds=child_seeds,
        budget=dict(draws_per_stratum=args.draws_per_stratum),
        parameter_names=list(transform.names), priors=transform.priors,
        center_strata=edges, stratum_prior_masses=masses,
        proposal_files=[dict(path=str(p.resolve()), sha256=digest(p)) for p in args.proposal],
        proposal_module=dict(path=str((args.proposal_module_root/'importance.py').resolve()),
                             sha256=digest(args.proposal_module_root/'importance.py')),
        source_root=str(args.source_root.resolve()), driver=str(paths[0]), response=str(paths[1]),
        inputs=[dict(path=str(p), sha256=digest(p)) for p in paths+[priors_path]],
        active_fits_before_launch=active, pid=os.getpid(),
        created_at=datetime.now(timezone.utc).isoformat(),
        evaluation_independent_of_training=True, weights_clipped=False,
        saturated_unit_policy='natural float64 expit, continuous physical boundary target; densities evaluated at original z')
    write_json(args.out/'sampler_config.json', config)
    for path, name in zip(paths, ('input_w2.txt', 'input_response.txt')):
        (args.out/name).write_bytes(path.read_bytes())
    write_json(args.out/'run_input.json', dict(driver=str(paths[0]), response=str(paths[1]),
        points=[42,42], mode=args.mode, sampler='frozen_importance'))
    started = time.perf_counter()
    rows, layers = [], []
    for index, (proposal, lo, hi, mass, child_seed) in enumerate(zip(
            proposals, edges[:-1], edges[1:], masses, child_seeds)):
        layer_start = time.perf_counter()
        z, component_ids = proposal.sample_z(args.draws_per_stratum,
            np.random.default_rng(child_seed), return_components=True)
        if z.shape != (args.draws_per_stratum, len(transform.names)) or not np.isfinite(z).all():
            raise ValueError('Invalid proposal sample shape or nonfinite z')
        u = expit(z)
        logp = np.sum(-np.logaddexp(0., -z)-np.logaddexp(0., z), axis=1)
        logq = proposal.logpdf_z(z)
        conditional_transform = PriorTransform(args.mode, priors_path, (lo, hi))
        physical = np.array([conditional_transform(point) for point in u])
        logl = np.array([likelihood(dict(zip(transform.names, point))) for point in physical])
        if not np.isfinite(logl).all() or not np.isfinite(logq).all():
            raise ValueError('Nonfinite likelihood/proposal: inspect rather than silently discard evaluation rows')
        logw = logl+logp-logq
        logz = float(logsumexp(logw)-np.log(len(logw)))
        normalized = np.exp(logw-logsumexp(logw))
        saturated = np.any((u == 0) | (u == 1), axis=1)
        defensive = component_ids == len(proposal.means)
        audit_indices = np.unique(np.r_[np.flatnonzero(saturated),np.arange(min(8,len(z)))])
        precise_logl = np.array([likelihood(dict(zip(transform.names,
            longdouble_transform(z[i],conditional_transform)))) for i in audit_indices])
        precision_delta = precise_logl-logl[audit_indices]
        if not np.isfinite(precise_logl).all() or np.max(np.abs(precision_delta)) > 1e-7:
            raise ValueError('Extended-precision transform discrepancy needs investigation')
        relative_integral_delta = float(np.dot(normalized[audit_indices],np.expm1(precision_delta)))
        a = np.exp(logw-np.max(logw))
        relative_se = float(np.std(a, ddof=1)/np.sqrt(len(a))/np.mean(a))
        layer = dict(index=index, seed=child_seed, count=len(logw), logz=logz,
            relative_evidence_mcse=relative_se, weight_kish=float(1/np.dot(normalized,normalized)),
            maximum_row_weight=float(normalized.max()),
            top_one_percent_row_weight=float(np.sort(normalized)[-max(1,int(np.ceil(.01*len(normalized)))):].sum()),
            defensive_count=int(defensive.sum()), defensive_posterior_weight=float(normalized[defensive].sum()),
            saturated_count=int(saturated.sum()), saturated_posterior_weight=float(normalized[saturated].sum()),
            saturated_maximum_row_weight=float(normalized[saturated].max(initial=0.)),
            precision_audit_calls=len(audit_indices),
            precision_audit_maximum_logl_delta=float(np.max(np.abs(precision_delta))),
            precision_audit_relative_integral_delta=relative_integral_delta,
            seconds=time.perf_counter()-layer_start)
        layers.append(layer)
        np.savez_compressed(args.out/f'stratum_{index:02d}.npz', z=z, samples=physical,
            logl=logl, logp_z=logp, logq_z=logq, logweight=logw, saturated=saturated,
            component_id=component_ids, precision_audit_indices=audit_indices,
            precision_audit_logl_delta=precision_delta)
        rows.append(dict(samples=physical, logl=logl, logw=logw, saturated=saturated))
        print(json.dumps(layer), flush=True)

    all_log_contribution = np.concatenate([np.log(mass)+r['logw']-np.log(len(r['logw']))
                                          for mass,r in zip(masses,rows)])
    global_logz = float(logsumexp(all_log_contribution))
    weights = np.exp(all_log_contribution-global_logz)
    samples = np.concatenate([r['samples'] for r in rows])
    logl = np.concatenate([r['logl'] for r in rows])
    ids = np.concatenate([np.full(len(r['logw']), j, dtype=np.int64) for j,r in enumerate(rows)])
    scale_logw = max(float(r['logw'].max()) for r in rows)
    scaled = [np.exp(r['logw']-scale_logw) for r in rows]
    scaled_z = sum(mass*a.mean() for mass,a in zip(masses,scaled))
    var_z = sum(mass**2*np.var(a,ddof=1)/len(a) for mass,a in zip(masses,scaled))
    columns = dict(zip(transform.names,samples.T))
    columns.update(assess.derived_parameters(columns))
    gate = assess.read_json(assess.GATE_PATH)
    summaries = {}
    def ratio_mcse(values):
        mean = float(np.dot(weights,values))
        variance = sum(mass**2*np.var(a*(values[ids==j]-mean),ddof=1)/len(a)
                       for j,(mass,a) in enumerate(zip(masses,scaled)))/scaled_z**2
        return mean,float(np.sqrt(variance))

    for name, values in columns.items():
        mean,mcse = ratio_mcse(values)
        summaries[name] = dict(mean=mean, mean_mcse=mcse,
            quantiles=assess.weighted_quantile(values,weights,gate['quantile_probabilities']).tolist())
    branch_rows = {'mu0<0':columns['mu0']<0,'A0<A1':columns['A0']<columns['A1'],'mu1<5':columns['mu1']<5}
    branch_diagnostics = {name:dict(zip(('probability','mcse'),ratio_mcse(values)))
                          for name,values in branch_rows.items()}
    layer_probabilities = np.array([weights[ids==j].sum() for j in range(3)])
    layer_jacobian = (np.eye(3)-layer_probabilities[:,None])*np.asarray(masses)[None,:]/scaled_z
    layer_var_z = np.array([np.var(a,ddof=1)/len(a) for a in scaled])
    layer_covariance = (layer_jacobian*layer_var_z[None,:])@layer_jacobian.T
    report = dict(scope='one pilot; insufficient for the frozen stability gate',
        sampler='frozen_importance', layers=layers, logz=global_logz,
        relative_evidence_mcse=float(np.sqrt(var_z)/scaled_z),
        stratum_posterior_mass=layer_probabilities.tolist(),
        stratum_posterior_mass_covariance=layer_covariance.tolist(),
        weight_kish=float(1/np.dot(weights,weights)), maximum_row_weight=float(weights.max()),
        top_one_percent_row_weight=float(np.sort(weights)[-max(1,int(np.ceil(.01*len(weights)))):].sum()),
        summaries=summaries, branches=branch_diagnostics,
        scientific_likelihood_evaluations=len(weights),
        precision_validation_likelihood_evaluations=sum(layer['precision_audit_calls'] for layer in layers),
        seconds=time.perf_counter()-started,
        self_normalized_moments_are_not_unbiased=True,
        mcse_is_conditional_on_frozen_proposals=True,
        kish_does_not_prove_missed_mode_coverage=True)
    np.savez_compressed(args.out/'importance_results.npz',samples=samples,logl=logl,
        original_prior_log_contribution=all_log_contribution,weights=weights,stratum_id=ids)
    np.savetxt(args.out/'weighted_posterior.csv',np.column_stack((samples,logl,ids,all_log_contribution,weights)),
        delimiter=',',header=','.join((*transform.names,'log_likelihood','stratum_id','log_contribution','weight')),comments='')
    write_json(args.out/'diagnostics.json',report)
    write_json(args.out/'completed.json',dict(sampler='frozen_importance',pilot=True))
    print(json.dumps({k:report[k] for k in ('logz','relative_evidence_mcse','stratum_posterior_mass',
        'weight_kish','maximum_row_weight','scientific_likelihood_evaluations','seconds')}),flush=True)


if __name__ == '__main__':
    main()
