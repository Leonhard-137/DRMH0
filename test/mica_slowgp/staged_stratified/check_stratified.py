"""Integration tests using analytic functions only, never the GP likelihood."""
from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'src'))
from mica_slowgp import stratified as st
from mica_slowgp.nested import NestedBudget, PriorTransform, load_saved_posterior
from dynesty.sampler import Sampler

INPUTS=Path('/home/leonhard/sci001/output/mica_slowgp/small_cases')
PRIORS=INPUTS/'priors.json'
EDGES=[-3,5,20,28]


class AnalyticLikelihood:
    def __init__(self, scale=1.):
        self.scale=scale

    def __call__(self,p):
        x=(p['mu0']+3)/31; y=(p['mu1']+3)/31
        return np.log(self.scale*(1+x+2*y+3*x*y))


def analytic_pair_likelihood(data,p,mode):
    # Test CLI argument plumbing without evaluating the GP model at all.
    return AnalyticLikelihood()(p)


class PlannedPause(Exception):
    pass


def expect_error(function,types=(ValueError,FileExistsError)):
    try:
        function()
    except types:
        return
    raise AssertionError('Expected an explicit rejection')


def snapshot(folder):
    return {str(p.relative_to(folder)):(p.stat().st_mtime_ns,p.read_bytes())
            for p in folder.rglob('*') if p.is_file()}


def main():
    report={'scope':'analytic toys only; no GP likelihood or scientific fit',
            'source':str(HERE/'src/mica_slowgp')}
    transform,edges,masses=st.strata_definition('shared',PRIORS,EDGES)
    assert np.allclose(masses,np.array([64,465,432])/961,rtol=0,atol=1e-15)
    original=json.loads(PRIORS.read_text())
    for changed in ([0,5,20,28],[-3,20,5,28],[-3,5,28]):
        expect_error(lambda:st.strata_definition('shared',PRIORS,changed))
    asymmetric=json.loads(json.dumps(original)); asymmetric['A1']['bounds'][1]=3.
    expect_error(lambda:st.strata_definition('shared',asymmetric,EDGES))
    nonuniform=json.loads(json.dumps(original)); nonuniform['mu0']['kind']='log_uniform'
    nonuniform['mu0']['bounds']=[1,28];nonuniform['mu1']=nonuniform['mu0'].copy()
    expect_error(lambda:st.strata_definition('shared',nonuniform,[1,5,20,28]))

    z,w=leggauss(128); z=(z+1)/2; w=w/2
    uv=np.stack(np.meshgrid(z,z,indexing='ij'),axis=-1).reshape(-1,2)
    quadrature=np.outer(w,w).ravel()
    prior_moment=np.zeros(2); evidence=0.; numerator=np.zeros(2)
    for j in range(3):
        conditional=PriorTransform('shared',PRIORS,EDGES[j:j+2])
        points=[]
        for u,v in uv:
            cube=np.full(11,.5);cube[6]=u;cube[3]=v
            p=conditional(cube)
            points.append((p[[3,6]]+3)/31)
        xy=np.array(points);ell=1+xy[:,0]+2*xy[:,1]+3*xy[:,0]*xy[:,1]
        prior_moment+=masses[j]*(quadrature@xy)
        evidence+=masses[j]*(quadrature@ell)
        numerator+=masses[j]*(quadrature@(ell[:,None]*xy))
    errors=dict(prior=float(np.max(abs(prior_moment-[1/3,2/3]))),
                evidence=float(abs(evidence-41/12)),
                posterior=float(np.max(abs(numerator/evidence-[84/205,151/205]))))
    assert max(errors.values())<3e-8
    report['conditional_transform_quadrature_errors']=errors

    # The ordinary path must retain exactly the frozen transform.
    name='mica_slowgp._frozen_nested_test'
    source=Path('/home/leonhard/software/MICA2-slowgp/archive/stability-v2-rslice/src/mica_slowgp/nested.py')
    spec=importlib.util.spec_from_file_location(name,source)
    old=importlib.util.module_from_spec(spec);sys.modules[name]=old;spec.loader.exec_module(old)
    for mode in ('none','independent','shared'):
        a,b=old.PriorTransform(mode,PRIORS),PriorTransform(mode,PRIORS)
        for cube in np.random.default_rng(123).random((100,len(a.names))):
            assert np.array_equal(a(cube),b(cube))
    report['ordinary_prior_transform_300_points']='identical to frozen v2'

    budget=NestedBudget(nlive=50,sample='rslice',slices=3,update_interval=5.0,dlogz=.15)
    shared_args=dict(mode='shared',priors=PRIORS,budget=budget,seed=29531,
                     center_strata=EDGES,
                     input_files=(INPUTS/'shared/w2.txt',INPUTS/'shared/response.txt'),
                     run_input={'driver':str(INPUTS/'shared/w2.txt'),
                                'response':str(INPUTS/'shared/response.txt'),'case':'shared'})
    with tempfile.TemporaryDirectory(prefix='mica-stratified-check-') as temporary:
        root=Path(temporary);full=root/'full';resumed=root/'resumed'
        # Match scheduler-created log/state files. Their contents must survive.
        full.mkdir();(full/'fit.log').write_text('toy scheduler log\n')
        (full/'scheduler_state.json').write_text('{"status":"running"}\n')
        first=st.run_stratified(AnalyticLikelihood(),full,**shared_args)
        assert (full/'fit.log').read_text()=='toy scheduler log\n'
        assert (full/'scheduler_state.json').read_text()=='{"status":"running"}\n'

        native_run=Sampler.run_nested;child_run=st.run_nested
        def pause_native(sampler,**kwargs):
            kwargs.update(maxiter=9,add_live=False)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                native_run(sampler,**kwargs)
            raise PlannedPause()
        def pause_second_child(*args,**kwargs):
            if Path(args[1]).name=='stratum_01':
                with patch.object(Sampler,'run_nested',pause_native):
                    return child_run(*args,**kwargs)
            return child_run(*args,**kwargs)
        with patch.object(st,'run_nested',pause_second_child):
            expect_error(lambda:st.run_stratified(AnalyticLikelihood(),resumed,**shared_args),(PlannedPause,))
        completed_child=snapshot(resumed/'strata/stratum_00')
        before=snapshot(resumed)
        expect_error(lambda:st.run_stratified(AnalyticLikelihood(),resumed,**{**shared_args,'seed':5},resume=True))
        expect_error(lambda:st.run_stratified(AnalyticLikelihood(2.),resumed,**shared_args,resume=True))
        expect_error(lambda:st.run_stratified(AnalyticLikelihood(),resumed,**{**shared_args,'center_strata':[-3,6,20,28]},resume=True))
        assert snapshot(resumed)==before
        child_config=resumed/'strata/stratum_00/sampler_config.json'
        saved=child_config.read_bytes();changed=json.loads(saved);changed['seed']+=1
        child_config.write_text(json.dumps(changed))
        expect_error(lambda:st.run_stratified(AnalyticLikelihood(),resumed,**shared_args,resume=True))
        child_config.write_bytes(saved)
        # Record the new mtime after this intentional test-only tampering.
        completed_child=snapshot(resumed/'strata/stratum_00')
        second=st.run_stratified(AnalyticLikelihood(),resumed,**shared_args,resume=True)
        assert snapshot(resumed/'strata/stratum_00')==completed_child
        a=load_saved_posterior(full,transform.names);b=load_saved_posterior(resumed,transform.names)
        for key in ('samples','stratum_id','log_likelihood','conditional_log_prior_mass'):
            assert np.array_equal(a[key],b[key]),key
        assert np.allclose(a['weights'],b['weights'],rtol=1e-12,atol=1e-15)
        assert 'log_prior_mass' not in a
        d=second['diagnostics']
        assert d['exact_likelihood_calls_total']==sum(x['exact_likelihood_calls_total'] for x in d['strata'])
        assert d['exact_calls_this_invocation']<d['exact_likelihood_calls_total']
        assert d['strata'][0]['reused_completed']
        complete_snapshot=snapshot(resumed)
        reused=st.run_stratified(AnalyticLikelihood(),resumed,**shared_args,resume=True)
        assert reused['reused_completed'] and snapshot(resumed)==complete_snapshot

        # Existing gate reader consumes the physical/weight schema unchanged.
        sys.path.insert(0,str(HERE.parent))
        import assess_stability
        gate=json.loads((HERE.parent/'stability_gate_v1.json').read_text())
        run=assess_stability.read_run(full,INPUTS,gate)
        assert run.config['integration_scheme']==st.SCHEME
        assert st.SCHEME in assess_stability.algorithm_contract(run.config)['parameterization']
        report['existing_gate_reader']='schema accepted; no claim of gate passage or scientific fit'
        report['end_to_end']={'continuous_equals_resumed':True,'completed_child_unchanged':True,
            'completed_parent_read_only':True,'wrong_seed_callback_edges_child_seed_rejected':True,
            'logZ':first['diagnostics']['raw_log_evidence'],
            'analytic_logZ':float(np.log(41/12)),
            'delta_method_error':first['diagnostics']['raw_log_evidence_error'],
            'posterior_stratum_mass':[x['posterior_mass'] for x in d['strata']],
            'exact_total_calls':d['exact_likelihood_calls_total'],
            'resumed_invocation_calls':d['exact_calls_this_invocation']}

        # Unknown scientific output is rejected, not replaced.
        unknown=root/'unknown';unknown.mkdir();(unknown/'posterior.txt').write_text('do not replace')
        expect_error(lambda:st.run_stratified(AnalyticLikelihood(),unknown,**shared_args))
        assert (unknown/'posterior.txt').read_text()=='do not replace'
        partial=root/'partial';partial.mkdir();(partial/'weighted_posterior.csv').write_text('incomplete')
        expect_error(lambda:st.run_stratified(AnalyticLikelihood(),partial,**shared_args))

        # Exercise actual CLI routing with a patched, named analytic callback.
        from mica_slowgp import cli,core
        budget_file=root/'budget.json';budget_file.write_text(json.dumps(dict(nlive=50,sample='rslice',slices=3,dlogz=.3)))
        cli_folder=root/'cli'
        with patch.object(core,'log_likelihood',analytic_pair_likelihood), redirect_stdout(io.StringIO()):
            cli.main(['fit','--driver',str(INPUTS/'shared/w2.txt'),'--response',str(INPUTS/'shared/response.txt'),
                '--mode','shared','--priors',str(PRIORS),'--out',str(cli_folder),'--sampler','dynesty',
                '--budget',str(budget_file),'--center-strata','-3','5','20','28'])
        assert (cli_folder/'completed.json').is_file()
        report['cli_route']='passed with analytic callback; no GP calls'
    report['passed']=True
    (HERE/'check_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
