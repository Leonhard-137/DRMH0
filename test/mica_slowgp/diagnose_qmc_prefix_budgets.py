#!/usr/bin/env python3
"""Read-only, prespecified same-scramble QMC prefix-budget diagnostic.

Never imports a GP implementation or evaluates a scientific likelihood. The
self-test uses a new analytic toy only and never opens the source run.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import expit

DEFAULT_PLAN = Path(__file__).with_name('qmc_prefix_budget_plan.json')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def write_json(path, value):
    from importance import json_safe
    with Path(path).open('x') as stream:
        stream.write(json.dumps(json_safe(value),indent=2,allow_nan=False)+'\n')


def modules(plan):
    folder = Path(plan['frozen_module_directory'])
    for name,expected in plan['module_sha256'].items():
        if sha(folder/name) != expected:
            raise ValueError(f'Frozen module identity mismatch: {name}')
    sys.path.insert(0,str(folder))
    import importance
    import qmc_importance
    import assess_stability
    for module in (importance,qmc_importance,assess_stability):
        if Path(module.__file__).resolve().parent != folder.resolve():
            raise ValueError('A different module was already imported')
    return importance,qmc_importance,assess_stability


def prefix_indices(component_id, scramble_id, small_counts, large_counts, R):
    """Take stable within-group prefixes; do not sort by weights/parameters."""
    component_id = np.asarray(component_id)
    scramble_id = np.asarray(scramble_id)
    small_counts,large_counts = np.asarray(small_counts),np.asarray(large_counts)
    if (component_id.shape != scramble_id.shape or small_counts.shape != large_counts.shape
            or np.any(small_counts < 1) or np.any(small_counts > large_counts)
            or len(component_id) != R*int(large_counts.sum())):
        raise ValueError('Invalid source/prefix dimensions')
    if not np.array_equal(np.unique(component_id),np.arange(len(large_counts))):
        raise ValueError('Missing/unknown source component')
    if not np.array_equal(np.unique(scramble_id),np.arange(R)):
        raise ValueError('Missing/unknown source scramble')
    selected = []
    for r in range(R):
        for k,(small,large) in enumerate(zip(small_counts,large_counts)):
            group = np.flatnonzero((scramble_id==r)&(component_id==k))
            if len(group) != large:
                raise ValueError('Source component/scramble count mismatch')
            selected.append(group[:small])
    indices = np.sort(np.concatenate(selected))
    if len(indices) != R*int(small_counts.sum()):
        raise AssertionError('Wrong prefix size')
    return indices


def component_factors(alpha, counts, component_id, R):
    return np.log(alpha[component_id])-np.log(counts[component_id])-np.log(R)


def observables(samples, names, assess):
    values = dict(zip(names,samples.T))
    values.update(assess.derived_parameters(values))
    values.update({'mu0<0':values['mu0']<0,'A0<A1':values['A0']<values['A1'],
                   'mu1<5':values['mu1']<5})
    return values


def self_test(plan, qmc, importance, out):
    """Analytic toy only: same streams, nested Nk, and recomputed factors."""
    p = importance.FrozenProposal(np.array([.4,.55]),np.array([[0.,0.],[1.,-1.]]),
        np.array([np.eye(2),np.eye(2)*1.3]),df=7.,epsilon=.05)
    R=8;seed=905101;budgets=(16,64,256);checks=[]
    for s in range(3):
        large=qmc.sample_components(p,seed=seed,stratum_id=s,target_N=budgets[-1],scrambles=R)
        alpha,big_counts,_=qmc.component_budget(p,budgets[-1])
        for target in budgets:
            _,counts,_=qmc.component_budget(p,target)
            indices=prefix_indices(large['component_id'],large['scramble_id'],counts,big_counts,R)
            direct=qmc.sample_components(p,seed=seed,stratum_id=s,target_N=target,scrambles=R)
            z=large['z'][indices]
            np.testing.assert_array_equal(z,direct['z'])
            np.testing.assert_array_equal(large['component_id'][indices],direct['component_id'])
            np.testing.assert_array_equal(large['scramble_id'][indices],direct['scramble_id'])
            factor=component_factors(alpha,counts,direct['component_id'],R)
            np.testing.assert_array_equal(factor,direct['log_quadrature_factor'])
            for r in range(R):
                np.testing.assert_allclose(np.exp(factor[direct['scramble_id']==r]).sum(),1/R,atol=2e-16,rtol=0)
            u=expit(z);logL=-np.sum((u-.3)**2,axis=1)
            logimportance=logL+importance.log_prior_z(z)-p.logpdf_z(z)
            logweight=logimportance+factor
            obs={'h':u[:,0],'branch':u[:,1]>.5}
            summary,w=qmc.summarize_scrambles(logweight,direct['scramble_id'],R,obs)
            direct_summary,direct_w=qmc.summarize_scrambles(
                logL+importance.log_prior_z(direct['z'])-p.logpdf_z(direct['z'])+direct['log_quadrature_factor'],
                direct['scramble_id'],R,obs)
            np.testing.assert_array_equal(w,direct_w)
            np.testing.assert_array_equal(summary['scramble_integrals_scaled'],direct_summary['scramble_integrals_scaled'])
            checks.append(dict(stratum=s,target_N=target,rows=len(z),z_bitwise_equal=True,
                weights_bitwise_equal=True,quadrature_factor_exact=True,
                scramble_integrals_bitwise_equal=True))
    expected=[]
    folder=Path(plan['frozen_module_directory'])
    for target in plan['target_Ns']:
        counts=[]
        for s in range(3):
            proposal=importance.FrozenProposal.load(folder/f'stratum_{s:02d}.npz')
            _,nk,_=qmc.component_budget(proposal,target)
            counts.append(int(nk.sum())*plan['scrambles'])
        expected.append(dict(target_N=target,layer_rows=counts,total_rows=sum(counts)))
    assert expected[-1]['total_rows']==plan['source_evaluation_rows']
    result=dict(passed=True,scope='analytic toy and frozen proposal budget arithmetic only',
        analytic_seed=seed,scrambles=R,toy_targets=list(budgets),checks=checks,
        planned_scientific_prefix_counts=expected,source_run_opened=False,GP_calls=0,
        no_scientific_evaluation=True,plan_id=plan['plan_id'])
    write_json(out,result)
    print(json.dumps(dict(passed=True,toy_comparisons=len(checks),report=str(out),
                          source_run_opened=False,GP_calls=0,planned_counts=expected)))


def evaluate(plan, plan_path, qmc, importance, assess, out):
    source=Path(plan['source_run'])
    # These guards precede every scientific NPZ/CSV/diagnostics read.
    if not (source/'completed.json').is_file():
        raise RuntimeError('Source run has not completed; no scientific raw data opened')
    config=read(source/'sampler_config.json')
    if Path('/proc',str(config['pid'])).exists() or config.get('status')!='complete':
        raise RuntimeError('Source recorded PID has not exited; no scientific raw data opened')
    if out.exists(): raise FileExistsError(out)
    if (config['seed']!=plan['root_seed'] or config['budget']['target_N']!=plan['source_target_N']
            or config['budget']['scrambles']!=plan['scrambles']
            or config['case']!=plan['case'] or config['mode']!=plan['mode']):
        raise ValueError('Source run differs from the predeclared plan')
    R=plan['scrambles'];names=config['parameter_names']
    assert len(names)==11 and config['scientific_likelihood_evaluations']==plan['source_evaluation_rows']
    gate=assess.read_json(assess.GATE_PATH)
    assert gate['quantile_probabilities']==plan['quantile_probabilities']
    masses=np.asarray(plan['prior_mass_numerators'],dtype=float)/plan['prior_mass_denominator']
    np.testing.assert_array_equal(config['stratum_prior_masses'],masses)
    assert config['center_strata']==plan['center_edges']
    proposals=[]
    for item in config['proposal_files']:
        if sha(item['path'])!=item['sha256']: raise ValueError('Source proposal identity changed')
        proposals.append(importance.FrozenProposal.load(item['path']))
    raw_paths=[source/f'stratum_{s:02d}.npz' for s in range(3)]
    watched=raw_paths+[source/'sampler_config.json',source/'diagnostics.json',source/'completed.json']
    before={str(p):sha(p) for p in watched}
    raw=[]
    for s,p in enumerate(raw_paths):
        with np.load(p,allow_pickle=False) as a:
            # Retain only columns needed to recompute statistics; original arrays are untouched.
            row={k:a[k].copy() for k in ('samples','logl','logp_z','logq_z','logimportance',
                'logweight','component_id','scramble_id','saturated')}
        np.testing.assert_array_equal(row['logimportance'],row['logl']+row['logp_z']-row['logq_z'])
        alpha,big_counts,_=qmc.component_budget(proposals[s],plan['source_target_N'])
        np.testing.assert_array_equal(config['layer_designs'][s]['component_counts_per_scramble'],big_counts)
        assert len(row['samples'])==R*int(big_counts.sum())
        np.testing.assert_array_equal(row['logweight'],row['logimportance']+
            component_factors(alpha,big_counts,row['component_id'],R))
        raw.append(row)
    source_report=read(source/'diagnostics.json')
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();reports=[]
    for target in plan['target_Ns']:
        evaluations=[];selected_by_layer={};designs=[]
        for s,(row,proposal) in enumerate(zip(raw,proposals)):
            alpha,nk,_=qmc.component_budget(proposal,target)
            _,big,_=qmc.component_budget(proposal,plan['source_target_N'])
            ix=prefix_indices(row['component_id'],row['scramble_id'],nk,big,R)
            selected_by_layer[f'stratum_{s:02d}_indices']=ix
            if target==plan['source_target_N']: np.testing.assert_array_equal(ix,np.arange(len(row['samples'])))
            factor=component_factors(alpha,nk,row['component_id'][ix],R)
            logweight=row['logimportance'][ix]+factor
            obs=observables(row['samples'][ix],names,assess)
            diag,w=qmc.summarize_scrambles(logweight,row['scramble_id'][ix],R,obs,row['saturated'][ix])
            diag.update(likelihood_calls=0,reused_source_rows=len(ix))
            evaluations.append(dict(seed=plan['root_seed'],stratum_id=s,scrambles=R,
                logweight=logweight,scramble_id=row['scramble_id'][ix],saturated=row['saturated'][ix],
                observables=obs,diagnostics=diag))
            designs.append(dict(stratum=s,rows=len(ix),Nk=nk,component_full_probabilities=alpha,
                rows_per_scramble=int(nk.sum()),seed_hierarchy='unchanged source streams; stable prefixes only'))
        merged=qmc.combine_strata(evaluations,masses)
        d=merged['diagnostics'];weights=merged['weights'];summaries={}
        for name in list(names)+gate['derived_tf_parameters']+gate['branches']:
            item=d['observables'][name]
            summaries[name]=dict(mean=item['posterior_mean'],mcse=item['ratio_delta_standard_error'],
                quantiles=assess.weighted_quantile(merged['observables'][name],weights,plan['quantile_probabilities']))
        mass_names=[f'__stratum_mass_{s}' for s in range(3)]
        order=d['scramble_integral_column_order'][1:]
        mass_ix=[order.index(name) for name in mass_names]
        report=dict(target_N=target,root_seed=plan['root_seed'],scrambles=R,
            actual_rows=sum(x['rows'] for x in designs),layer_designs=designs,
            quantile_probabilities=plan['quantile_probabilities'],summaries=summaries,
            logz=d['log_evidence'],relative_evidence_mcse=d['relative_evidence_standard_error'],
            stratum_posterior_mass=[d['observables'][name]['posterior_mean'] for name in mass_names],
            stratum_posterior_mass_covariance=d['observable_ratio_covariance'][np.ix_(mass_ix,mass_ix)],
            scramble_covariance_diagnostics=d,maximum_row_weight=float(weights.max()),
            weight_kish_descriptive_only=d['weight_ess_descriptive_only'],new_GP_calls=0,
            same_seed_correlated_budgets=True,independent_replicates=False,formal_stability_gate=False)
        if target==plan['source_target_N']:
            np.testing.assert_allclose([report['logz'],report['relative_evidence_mcse']],
                [source_report['logz'],source_report['relative_evidence_mcse']],rtol=2e-12,atol=1e-14)
            for name in list(names)+gate['derived_tf_parameters']:
                np.testing.assert_allclose([summaries[name]['mean'],summaries[name]['mcse']],
                    [source_report['summaries'][name]['mean'],source_report['summaries'][name]['mean_mcse']],
                    rtol=2e-12,atol=1e-14)
                np.testing.assert_array_equal(summaries[name]['quantiles'],source_report['summaries'][name]['quantiles'])
            for name in gate['branches']:
                np.testing.assert_allclose([summaries[name]['mean'],summaries[name]['mcse']],
                    [source_report['branches'][name]['probability'],source_report['branches'][name]['mcse']],
                    rtol=2e-12,atol=1e-14)
            report['full_budget_reproduces_source_report']=True
        write_json(out/f'target_{target}.json',report)
        np.savez_compressed(out/f'target_{target}_prefix.npz',**selected_by_layer,
            global_logweight=merged['logweight'],global_weights=weights,
            stratum_id=merged['stratum_id'],scramble_id=merged['scramble_id'],
            scramble_integrals_scaled=d['scramble_integrals_scaled'],
            scramble_integral_column_order=np.array(d['scramble_integral_column_order']),
            scramble_integral_log_scale=d['scramble_integral_log_scale'])
        reports.append(report)
        print(json.dumps(dict(target_N=target,rows=report['actual_rows'],
            negative_lag_fraction=importance.json_safe(summaries['total_negative_lag_fraction']))),flush=True)
    unchanged={p:sha(p)==h for p,h in before.items()}
    if not all(unchanged.values()): raise AssertionError('Source artifact changed during diagnostic')
    write_json(out/'manifest.json',dict(plan_id=plan['plan_id'],plan_path=str(plan_path),plan_sha256=sha(plan_path),
        diagnostic_script_sha256=sha(__file__),created_at=datetime.now(timezone.utc).isoformat(),
        source_run=str(source),source_sha256=before,source_files_unchanged=unchanged,
        target_Ns=plan['target_Ns'],scrambles=R,root_seed=plan['root_seed'],
        actual_rows=[x['actual_rows'] for x in reports],new_GP_calls=0,
        diagnostic_seconds=time.perf_counter()-started,same_seed_correlated_budgets=True,
        independent_replicates=False,formal_stability_gate=False,limitations=plan['limitations']))
    write_json(out/'completed.json',dict(plan_id=plan['plan_id'],read_only_prefix_diagnostic=True,new_GP_calls=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=DEFAULT_PLAN)
    p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--self-test-out',type=Path,default=Path(__file__).with_name('qmc_prefix_budget_toy.json'))
    args=p.parse_args();plan=read(args.plan)
    importance,qmc,assess=modules(plan)
    if args.self_test:
        self_test(plan,qmc,importance,args.self_test_out)
    else:
        evaluate(plan,args.plan,qmc,importance,assess,args.out or Path(plan['output_directory']))


if __name__=='__main__': main()
