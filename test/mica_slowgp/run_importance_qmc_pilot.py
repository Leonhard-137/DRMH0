#!/usr/bin/env python3
"""Frozen-proposal randomized-QMC pilot for one W2/response pair.

Rows are component-stratified quadrature contributions. Uncertainty comes from
independent scramble integral vectors, never from an iid-row variance formula.
This driver neither trains proposals nor declares a formal stability pass.
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
from scipy.special import expit


PILOT_FILENAMES = {'run_importance_pilot.py', 'run_importance_qmc_pilot.py'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path))


def write_json(path, value):
    from importance import json_safe
    with Path(path).open('x') as stream:
        json.dump(json_safe(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def longdouble_transform(z, transform):
    """Same independent continuous-map precision audit as the iid pilot."""
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
    active = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry/'cmdline').read_bytes().decode().split('\0')
            native = any(argv[i:i+3] == ['-m','mica_slowgp.cli','fit']
                         for i in range(len(argv)-2))
            pilot = any(Path(arg).name in PILOT_FILENAMES for arg in argv if arg)
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
        if any(job['status'] in ('planned','starting') for job in manifest['jobs']):
            raise RuntimeError('An existing scheduler still has queued work')
    return active


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source-root', type=Path, required=True)
    result.add_argument('--proposal-module-root', type=Path, required=True)
    result.add_argument('--proposal', type=Path, action='append', required=True)
    result.add_argument('--inputs-root', type=Path, required=True)
    result.add_argument('--case', choices=('clean','response_only','independent','shared'), required=True)
    result.add_argument('--mode', choices=('none','independent','shared'), required=True)
    result.add_argument('--target-n', type=int, default=2048)
    result.add_argument('--scrambles', type=int, default=8)
    result.add_argument('--seed', type=int, default=82137)
    result.add_argument('--out', type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if len(args.proposal) != 3 or args.target_n < 1 or args.scrambles < 2 or args.seed < 0:
        raise ValueError('Require three proposals, positive target-n, R>=2 and nonnegative seed')
    sys.path.insert(0,str(args.source_root.resolve()))
    sys.path.insert(0,str(args.proposal_module_root.resolve()))
    import importance
    import qmc_importance
    from importance import FrozenProposal, log_prior_z
    from qmc_importance import sample_components, summarize_scrambles, combine_strata
    from mica_slowgp.core import PairData
    from mica_slowgp.fast import FastPairLikelihood
    from mica_slowgp.nested import PriorTransform
    from mica_slowgp.stratified import strata_definition
    sys.path.insert(0,str(Path(__file__).parent))
    import assess_stability as assess

    active = capacity_check()
    if args.out.exists():
        raise FileExistsError(args.out)
    paths = [args.inputs_root/args.case/name for name in ('w2.txt','response.txt')]
    priors_path = args.inputs_root/'priors.json'
    arrays = [np.loadtxt(path,ndmin=2) for path in paths]
    if any(array.shape != (42,3) for array in arrays):
        raise ValueError('The small-case contract requires 42 points in each of two bands')
    data = PairData(*arrays[0].T,*arrays[1].T)
    transform, edges, masses = strata_definition(args.mode,priors_path,[-3,5,20,28])
    proposals = [FrozenProposal.load(path) for path in args.proposal]
    if any(p.dimension != len(transform.names) for p in proposals):
        raise ValueError('Proposal dimension does not match the physical parameter vector')
    gate = assess.read_json(assess.GATE_PATH)
    branch_names = ['mu0<0','A0<A1','mu1<5']
    if gate['branches'] != branch_names:
        raise ValueError('Frozen branch definitions changed')
    args.out.mkdir(parents=True,exist_ok=False)
    config = dict(sampler='frozen_importance_qmc',algorithm=qmc_importance.SCHEME,
        scope='single randomized-QMC pilot; no formal stability claim',status='running',
        case=args.case,mode=args.mode,seed=args.seed,pid=os.getpid(),
        budget=dict(target_N=args.target_n,scrambles=args.scrambles),
        parameter_names=list(transform.names),priors=transform.priors,
        center_strata=edges,stratum_prior_masses=masses,
        driver=file_identity(__file__),qmc_module=file_identity(qmc_importance.__file__),
        proposal_module=file_identity(importance.__file__),
        proposal_files=[file_identity(p) for p in args.proposal],
        assessment_module=file_identity(assess.__file__),gate=file_identity(assess.GATE_PATH),
        quantile_probabilities=gate['quantile_probabilities'],
        source_root=str(args.source_root.resolve()),inputs=[file_identity(p) for p in paths+[priors_path]],
        active_fits_before_launch=active,created_at=datetime.now(timezone.utc).isoformat(),
        seed_rule='SeedSequence(root seed, spawn_key=(stratum_id,scramble_id,component_id))',
        layer_designs=[],scientific_likelihood_evaluations=0,
        precision_validation_likelihood_evaluations=0,
        endpoint_policy='Sobol endpoints fail visibly; finite z expit saturation uses continuous closed-unit target; original z densities retained',
        closed_unit_target_allowed=True,weights_clipped=False,training_rows_used_as_evaluation=False,
        row_weight_rule='logL+logpi_z-logq_z+logalpha_k-logNk-logR; global adds logrho_s',
        error_method='independent scramble integral covariance and posterior-ratio delta method; no iid row MCSE')
    write_json(args.out/'sampler_config.json',config)
    for p,name in zip(paths,('input_w2.txt','input_response.txt')):
        (args.out/name).write_bytes(p.read_bytes())
    (args.out/'input_priors.json').write_bytes(priors_path.read_bytes())
    write_json(args.out/'run_input.json',dict(driver=str(paths[0]),response=str(paths[1]),
        mode=args.mode,sampler=config['sampler'],points=[42,42]))
    likelihood = FastPairLikelihood(data,mode=args.mode,backend='compiled')
    started = time.perf_counter()
    evaluations, layers = [], []
    for index,(proposal,lo,hi) in enumerate(zip(proposals,edges[:-1],edges[1:])):
        layer_start = time.perf_counter()
        e = sample_components(proposal,seed=args.seed,stratum_id=index,
                              target_N=args.target_n,scrambles=args.scrambles)
        z = e['z']; u = expit(z)
        conditional = PriorTransform(args.mode,priors_path,(lo,hi))
        physical = np.array([conditional(point) for point in u])
        logp = log_prior_z(z)
        logq = proposal.logpdf_z(z)
        logl = np.array([likelihood(dict(zip(transform.names,point))) for point in physical])
        if not np.isfinite(logl).all() or not np.isfinite(logq).all() or not np.isfinite(logp).all():
            raise ValueError('Nonfinite target/density; investigate rather than discard quadrature rows')
        logimportance = logl+logp-logq
        logweight = logimportance+e['log_quadrature_factor']
        saturated = np.any((u==0)|(u==1),axis=1)
        columns = dict(zip(transform.names,physical.T))
        columns.update(assess.derived_parameters(columns))
        observables = dict(columns)
        observables.update({'mu0<0':columns['mu0']<0,
                            'A0<A1':columns['A0']<columns['A1'],
                            'mu1<5':columns['mu1']<5})
        summary,weights = summarize_scrambles(logweight,e['scramble_id'],args.scrambles,
                                               observables,saturated)
        audit_indices = np.unique(np.r_[np.flatnonzero(saturated),np.arange(min(8,len(z)))])
        precise = np.array([likelihood(dict(zip(transform.names,longdouble_transform(z[i],conditional))))
                            for i in audit_indices])
        delta = precise-logl[audit_indices]
        if not np.isfinite(precise).all() or np.max(abs(delta)) > 1e-7:
            raise ValueError('Extended-precision transform discrepancy needs investigation')
        defensive = e['component_id']==len(proposal.means)
        summary.update(index=index,likelihood_calls=len(z),target_N=args.target_n,
            rows_per_scramble=e['rows_per_scramble'],component_counts_per_scramble=e['component_counts_per_scramble'],
            component_full_probabilities=e['component_probabilities'],
            defensive_count=int(defensive.sum()),defensive_posterior_weight=float(weights[defensive].sum()),
            saturated_maximum_row_weight=float(weights[saturated].max(initial=0.)),
            top_one_percent_row_weight=float(np.sort(weights)[-max(1,int(np.ceil(.01*len(weights)))):].sum()),
            precision_audit_calls=len(audit_indices),precision_audit_maximum_logl_delta=float(max(abs(delta))),
            precision_audit_relative_integral_delta=float(weights[audit_indices]@np.expm1(delta)),
            seconds=time.perf_counter()-layer_start)
        e.update(samples=physical,logl=logl,logp_z=logp,logq_z=logq,
            logimportance=logimportance,logweight=logweight,saturated=saturated,
            observables=observables,diagnostics=summary)
        np.savez_compressed(args.out/f'stratum_{index:02d}.npz',
            z=z,u=u,samples=physical,logl=logl,logp_z=logp,logq_z=logq,
            logimportance=logimportance,logweight=logweight,
            log_quadrature_factor=e['log_quadrature_factor'],quadrature_factor=np.exp(e['log_quadrature_factor']),
            component_id=e['component_id'],scramble_id=e['scramble_id'],stratum_id=np.full(len(z),index,dtype=int),
            component_counts_per_scramble=e['component_counts_per_scramble'],
            component_full_probabilities=e['component_probabilities'],saturated=saturated,
            precision_audit_indices=audit_indices,precision_audit_logl_delta=delta,
            scramble_integrals_scaled=summary['scramble_integrals_scaled'],
            scramble_integral_log_scale=summary['scramble_integral_log_scale'],
            scramble_integral_column_order=np.array(summary['scramble_integral_column_order']))
        config['layer_designs'].append(dict(stratum_id=index,scrambles=args.scrambles,
            rows=len(z),rows_per_scramble=e['rows_per_scramble'],
            component_counts_per_scramble=e['component_counts_per_scramble'],
            component_full_probabilities=e['component_probabilities'],seed_hierarchy=e['seed_hierarchy']))
        config['scientific_likelihood_evaluations'] += len(z)
        config['precision_validation_likelihood_evaluations'] += len(audit_indices)
        evaluations.append(e);layers.append(summary)
        print(json.dumps(dict(stratum=index,rows=len(z),logz=summary['log_evidence'],
            relative_mcse=summary['relative_evidence_standard_error'],seconds=summary['seconds'])),flush=True)

    combined = combine_strata(evaluations,masses)
    overall = combined['diagnostics']; weights = combined['weights']
    samples = np.concatenate([e['samples'] for e in evaluations])
    parameter_names = list(transform.names)+gate['derived_tf_parameters']
    summaries = {}
    for name in parameter_names:
        item = overall['observables'][name]
        summaries[name] = dict(mean=item['posterior_mean'],mean_mcse=item['ratio_delta_standard_error'],
            quantiles=assess.weighted_quantile(combined['observables'][name],weights,gate['quantile_probabilities']))
    branches = {name:dict(probability=overall['observables'][name]['posterior_mean'],
        mcse=overall['observables'][name]['ratio_delta_standard_error']) for name in branch_names}
    mass_keys = [f'__stratum_mass_{j}' for j in range(3)]
    observable_order = overall['scramble_integral_column_order'][1:]
    mass_indices = [observable_order.index(name) for name in mass_keys]
    mass_covariance = overall['observable_ratio_covariance'][np.ix_(mass_indices,mass_indices)]
    report = dict(scope='one QMC pilot; insufficient for the frozen stability gate',
        sampler=config['sampler'],algorithm=qmc_importance.SCHEME,layers=layers,
        logz=overall['log_evidence'],relative_evidence_mcse=overall['relative_evidence_standard_error'],
        summaries=summaries,branches=branches,
        stratum_posterior_mass=[overall['observables'][name]['posterior_mean'] for name in mass_keys],
        stratum_posterior_mass_covariance=mass_covariance,
        weight_kish_descriptive_only=overall['weight_ess_descriptive_only'],
        maximum_row_weight=float(weights.max()),
        top_one_percent_row_weight=float(np.sort(weights)[-max(1,int(np.ceil(.01*len(weights)))):].sum()),
        scramble_covariance_diagnostics=overall,
        scientific_likelihood_evaluations=config['scientific_likelihood_evaluations'],
        precision_validation_likelihood_evaluations=config['precision_validation_likelihood_evaluations'],
        seconds=time.perf_counter()-started,iid_row_standard_error_used=False,
        posterior_ratio_unbiased=False,error_caveat=overall['error_caveat'])
    cat = lambda name: np.concatenate([e[name] for e in evaluations])
    logfactor = cat('log_quadrature_factor')
    global_factor = logfactor+np.log(np.asarray(masses))[combined['stratum_id']]
    np.savez_compressed(args.out/'importance_qmc_results.npz',z=cat('z'),samples=samples,
        logl=cat('logl'),logp_z=cat('logp_z'),logq_z=cat('logq_z'),logimportance=cat('logimportance'),
        logweight=combined['logweight'],weights=weights,
        log_quadrature_factor=logfactor,quadrature_factor=np.exp(logfactor),
        log_global_quadrature_factor=global_factor,global_quadrature_factor=np.exp(global_factor),
        component_id=cat('component_id'),scramble_id=combined['scramble_id'],stratum_id=combined['stratum_id'],
        saturated=combined['saturated'],scramble_integrals_scaled=overall['scramble_integrals_scaled'],
        scramble_integral_log_scale=overall['scramble_integral_log_scale'],
        scramble_integral_mean_covariance_scaled=overall['scramble_integral_mean_covariance_scaled'],
        scramble_integral_column_order=np.array(overall['scramble_integral_column_order']),
        observable_ratio_covariance=overall['observable_ratio_covariance'])
    np.savetxt(args.out/'weighted_posterior.csv',np.column_stack((samples,cat('logl'),
        combined['stratum_id'],cat('component_id'),combined['scramble_id'],cat('logimportance'),
        logfactor,combined['logweight'],weights)),delimiter=',',
        header=','.join((*transform.names,'log_likelihood','stratum_id','component_id','scramble_id',
                        'logimportance','log_quadrature_factor','logweight','weight')),comments='')
    write_json(args.out/'diagnostics.json',report)
    config.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat())
    final_config = args.out/'sampler_config.final.json'
    write_json(final_config,config)
    os.replace(final_config,args.out/'sampler_config.json')
    write_json(args.out/'completed.json',dict(sampler=config['sampler'],algorithm=qmc_importance.SCHEME,
                                             pilot=True,scrambles=args.scrambles))
    print(json.dumps(dict(logz=report['logz'],relative_evidence_mcse=report['relative_evidence_mcse'],
        scientific_likelihood_evaluations=report['scientific_likelihood_evaluations'],
        precision_validation_likelihood_evaluations=report['precision_validation_likelihood_evaluations'],
        seconds=report['seconds'])),flush=True)


if __name__ == '__main__':
    main()
