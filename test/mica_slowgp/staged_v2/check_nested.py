"""Small analytic toys for staged nested.py; never runs the GP likelihood.

The staged file is loaded under its eventual canonical module name only in
this process, so native pickle restore exercises real legacy class migration.
All sampler outputs live in automatically cleaned temporary directories.
"""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import warnings

import dynesty
import numpy as np
from dynesty.sampler import Sampler


HERE = Path(__file__).resolve().parent
LIVE = Path('/home/leonhard/software/MICA2-slowgp/src/mica_slowgp/nested.py')
LEGACY_CONFIG = Path('/home/leonhard/sci001/output/mica_slowgp/stability_v1/dynesty/shared_shared/budget_400/seed_71231/sampler_config.json')


class Toy2D:
    def __init__(self):
        self.body_calls = 0

    def __call__(self, p):
        self.body_calls += 1
        return -0.5*((p['a']/0.06)**2 + ((p['b']-2*p['a'])/0.3)**2)


class ToyPhysicalNames:
    """Analytic toy using the adapter's parameter names, not a scientific fit."""
    def __init__(self):
        self.body_calls = 0

    def __call__(self, p):
        self.body_calls += 1
        return -0.5*(((p['mu0']+0.3)/0.15)**2
                     + ((p['mu1']-0.5-0.5*p['mu0'])/0.2)**2)


def prior2d(u):
    return 2*u-1


def toy_priors():
    pairs = dict(sigma_x=(0.5,2), tau_x=(1,3), A0=(0.1,1), mu0=(-2,2),
                 w0=(0.1,1), A1=(0.1,1), mu1=(-2,2), w1=(0.1,1),
                 qW=(0,1), qb=(0,1), Ts=(2,5))
    return {key: dict(kind='uniform', bounds=list(value)) for key,value in pairs.items()}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_staged():
    import mica_slowgp
    module = load_module('mica_slowgp.nested', HERE/'nested.py')
    mica_slowgp.nested = module
    return module


class PlannedPause(Exception):
    pass


def paused_adapter_run(module, folder, budget, *, resume=False):
    native_run = Sampler.run_nested

    def bounded_native_run(sampler, **kwargs):
        kwargs.update(maxiter=29, add_live=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            native_run(sampler, **kwargs)
        raise PlannedPause('Saved analytic-toy checkpoint after 30 iterations')

    with patch.object(Sampler, 'run_nested', bounded_native_run):
        try:
            module.run_nested(ToyPhysicalNames(), folder, mode='shared',
                              priors=toy_priors(), budget=budget, seed=1729,
                              resume=resume)
        except PlannedPause:
            pass
        else:
            raise AssertionError('Toy pause was not reached')
    assert not (folder/'weighted_posterior.csv').exists()


def finish_adapter(module, folder, budget, *, resume):
    return module.run_nested(ToyPhysicalNames(), folder, mode='shared',
                             priors=toy_priors(), budget=budget, seed=1729,
                             resume=resume)


def test_counter_trajectory(module, folder, kind):
    def make():
        return dynesty.NestedSampler(module.VectorLikelihood(Toy2D(), ('a','b')),
            prior2d, 2, nlive=30, bound='single', sample=kind,
            **({'walks':10, 'facc':0.5} if kind=='rwalk' else {'slices':4}),
            update_interval=5.0, bootstrap=0,
            first_update={'min_ncall':0,'min_eff':100.},
            rstate=np.random.default_rng(1729))

    def advance(sampler, n, resume=False):
        generator = sampler.sample(dlogz=0, add_live=False, resume=resume)
        for _ in itertools.islice(generator, n):
            pass
        generator.close()

    full, split = make(), make()
    advance(full,160)
    advance(split,60)
    before = module._counter_for_sampler(split).exact_calls
    checkpoint = str(folder/f'{kind}.checkpoint')
    split.save(checkpoint)
    resumed = dynesty.NestedSampler.restore(checkpoint)
    counter = module._counter_for_sampler(resumed)
    assert counter.exact_calls == before
    advance(resumed,100,True)
    assert counter.exact_calls == counter.callback.body_calls
    assert counter.exact_calls == module._counter_for_sampler(full).exact_calls
    assert module._counter_for_sampler(split).exact_calls == before
    for key in ('samples','logl','ncall'):
        assert np.array_equal(full.results[key],resumed.results[key]),key
    before_live = counter.exact_calls
    result_count_before = int(sum(resumed.results.ncall))
    list(resumed.add_live_points())
    assert counter.exact_calls == before_live
    assert sum(resumed.results.ncall) == result_count_before+30
    return dict(checkpoint_exact=before, resumed_increment=counter.exact_calls-before,
                exact_total=counter.exact_calls, reported_ncall=resumed.ncall,
                continuous_equals_resumed=True, final_live_actual_calls=0)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--legacy-prepare':
        from mica_slowgp import nested
        assert Path(nested.__file__).resolve()==LIVE
        paused_adapter_run(nested,Path(sys.argv[2]),dict(nlive=50,walks=5,dlogz=0.4))
        return

    module = load_staged()
    report = dict(staged_module=str(HERE/'nested.py'), dynesty_version=dynesty.__version__)
    old = json.loads(LEGACY_CONFIG.read_text())
    original = json.loads(json.dumps(old))
    expanded = {**old,'budget':{**old['budget'],'slices':None,'update_interval':None,'facc':0.5}}
    assert module._normalized_resume_config(old)==module._normalized_resume_config(expanded)
    assert old==original
    integer = {**expanded,'budget':{**expanded['budget'],'update_interval':50}}
    floating = {**expanded,'budget':{**expanded['budget'],'update_interval':50.0}}
    assert module._normalized_resume_config(integer)!=module._normalized_resume_config(floating)
    for changes in ({'slices':0},{'slices':True},{'slices':2.5},
                    {'update_interval':0},{'update_interval':True},
                    {'update_interval':float('nan')},{'facc':0},{'facc':1.1}):
        try:
            module.NestedBudget(**changes)
        except ValueError:
            pass
        else:
            raise AssertionError(changes)
    report['config_legacy_defaults_and_interval_type'] = 'passed'

    kwargs = module._sampler_kwargs(module.NestedBudget(),11,1729)
    assert kwargs['walks']==50 and kwargs['facc']==0.5
    assert 'slices' not in kwargs and 'update_interval' not in kwargs
    kwargs = module._sampler_kwargs(module.NestedBudget(sample='rslice'),11,1729)
    assert kwargs['slices']==14 and 'walks' not in kwargs and 'facc' not in kwargs
    report['applicable_kwargs'] = 'passed'

    # Exercise the public constructor, not an inferred class default. In
    # dynesty 3.1.0 auto ignores external walks, so old auto+walks50 resolves
    # to ndim+20=31 in 11D. Neither constructor advances any nested iteration.
    transform=module.PriorTransform('shared',toy_priors())
    old_auto=dynesty.NestedSampler(
        module.VectorLikelihood(ToyPhysicalNames(),transform.names),transform,11,
        nlive=50,bound='multi',sample='auto',walks=50,bootstrap=0,
        rstate=np.random.default_rng(1729))
    new_auto=dynesty.NestedSampler(
        module.VectorLikelihood(ToyPhysicalNames(),transform.names),transform,11,
        **module._sampler_kwargs(module.NestedBudget(nlive=50,sample='auto'),11,1729))
    old_effective=module._effective_sampler_settings(old_auto)
    new_effective=module._effective_sampler_settings(new_auto)
    assert old_effective==new_effective
    assert new_effective['tuning']['walks']==31
    assert np.array_equal(old_auto.live_u,new_auto.live_u)
    assert np.array_equal(old_auto.live_logl,new_auto.live_logl)
    report['explicit_auto_preserves_old_effective_settings']=new_effective

    legacy = load_module('mica_slowgp._legacy_nested_test',LIVE)
    for mode in ('none','independent','shared'):
        a,b = legacy.PriorTransform(mode),module.PriorTransform(mode)
        for u in np.random.default_rng(1729).random((100,len(a.names))):
            assert np.array_equal(a(u),b(u))
    report['unchanged_prior_transform_300_points'] = 'passed'

    # Read an actual finished legacy checkpoint; do not advance or save it.
    actual = dynesty.NestedSampler.restore(str(LEGACY_CONFIG.with_name('dynesty.checkpoint')))
    actual_counter = module._counter_for_sampler(actual)
    assert actual_counter.exact_calls==0 and not actual_counter.counted_from_start
    report['actual_legacy_checkpoint_read_only'] = module._effective_sampler_settings(actual)

    with tempfile.TemporaryDirectory(prefix='mica-staged-v2-check-') as temporary:
        folder=Path(temporary)
        report['trajectory_toys']={kind:test_counter_trajectory(module,folder,kind)
                                   for kind in ('rwalk','rslice')}
        report['adapter_toys']={}
        for kind in ('rwalk','rslice'):
            budget=module.NestedBudget(nlive=50,walks=5,sample=kind,slices=3,
                                      update_interval=5.0,dlogz=0.4)
            full_folder=folder/f'{kind}-full'
            split_folder=folder/f'{kind}-split'
            full=finish_adapter(module,full_folder,budget,resume=False)
            paused_adapter_run(module,split_folder,budget)
            saved=dynesty.NestedSampler.restore(str(split_folder/'dynesty.checkpoint'))
            before=module._counter_for_sampler(saved).exact_calls
            split=finish_adapter(module,split_folder,budget,resume=True)
            a=np.genfromtxt(full_folder/'weighted_posterior.csv',delimiter=',',skip_header=1)
            b=np.genfromtxt(split_folder/'weighted_posterior.csv',delimiter=',',skip_header=1)
            assert np.allclose(a,b,rtol=1e-12,atol=1e-12)
            for run,base,path in ((full,0,full_folder),(split,before,split_folder)):
                diagnostic=run['diagnostics']
                sampler=dynesty.NestedSampler.restore(str(path/'dynesty.checkpoint'))
                counter=module._counter_for_sampler(sampler)
                assert counter.exact_calls==counter.callback.body_calls
                assert diagnostic['exact_likelihood_calls_total']==counter.exact_calls
                assert diagnostic['exact_calls_this_invocation']==counter.exact_calls-base
                assert diagnostic['sampler_effective']['bound_update_interval_reported_calls']==250
                module.load_saved_posterior(path,module.parameter_names('shared'))
            report['adapter_toys'][kind]=dict(
                exact_total=split['diagnostics']['exact_likelihood_calls_total'],
                checkpoint_exact=before,
                resumed_invocation=split['diagnostics']['exact_calls_this_invocation'],
                reported_ncall=split['diagnostics']['dynesty_reported_ncall'],
                continuous_equals_resumed=True)

        legacy_folder=folder/'legacy'
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),
                        '--legacy-prepare',str(legacy_folder)],check=True,
                       capture_output=True,text=True)
        config_before=(legacy_folder/'sampler_config.json').read_bytes()
        saved=dynesty.NestedSampler.restore(str(legacy_folder/'dynesty.checkpoint'))
        counter=module._counter_for_sampler(saved)
        prefix=counter.callback.body_calls
        assert counter.exact_calls==0 and not counter.counted_from_start
        budget=dict(nlive=50,walks=5,dlogz=0.4)
        # Two successive restores prove the unknown-prefix flag and measured
        # suffix survive checkpointing, rather than resetting on every resume.
        paused_adapter_run(module,legacy_folder,budget,resume=True)
        saved=dynesty.NestedSampler.restore(str(legacy_folder/'dynesty.checkpoint'))
        before=module._counter_for_sampler(saved).exact_calls
        assert before>0
        completed=finish_adapter(module,legacy_folder,budget,resume=True)
        diagnostic=completed['diagnostics']
        saved=dynesty.NestedSampler.restore(str(legacy_folder/'dynesty.checkpoint'))
        counter=module._counter_for_sampler(saved)
        assert diagnostic['exact_likelihood_calls_total'] is None
        assert not diagnostic['exact_count_includes_full_history']
        assert counter.exact_calls==counter.callback.body_calls-prefix
        assert diagnostic['exact_calls_this_invocation']==counter.exact_calls-before
        assert (legacy_folder/'sampler_config.json').read_bytes()==config_before
        report['legacy_resume_twice']=dict(known_toy_prefix=prefix,
            measured_suffix=counter.exact_calls, second_resume_increment=counter.exact_calls-before,
            exact_total=None, original_config_unchanged=True)

    report['passed']=True
    report['scope']='analytic toys only; no GP likelihood or scientific fit; temporary sampler outputs cleaned'
    (HERE/'check_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
