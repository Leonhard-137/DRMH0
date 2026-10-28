"""Exercise the actual staged callbacks against exact toy targets, without fits."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kstest


STAGED = Path(__file__).with_name('sampler.py').resolve()
spec = importlib.util.spec_from_file_location('mica_slowgp_staged_v2_sampler', STAGED)
sampler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sampler
spec.loader.exec_module(sampler)


def unit_coordinates(values, model):
    u = values.copy()
    u[:, model._logscale] = np.log(u[:, model._logscale])
    u = (u-model._bounds[:, 0])/np.diff(model._bounds, axis=1)[:, 0]
    x, y = u[:, 3].copy(), u[:, 6].copy()
    u[:, 3] = 1-(1-x)**2
    u[:, 6] = (y-x)/(1-x)
    return u


def physical_coordinates(unit, model):
    u = unit.copy()
    first = 1-np.sqrt(1-u[:, 3])
    u[:, 6] = first+(1-first)*u[:, 6]
    u[:, 3] = first
    x = model._bounds[:, 0]+u*np.diff(model._bounds, axis=1)[:, 0]
    x[:, model._logscale] = np.exp(x[:, model._logscale])
    return x


def check_target(mode, move, target, priors, n, seed):
    model = sampler.DNestModel(lambda p: 0., mode=mode, priors=priors, seed=seed)
    rng = np.random.default_rng(seed+100)
    unit = rng.random((n, model.num_params))
    tilt = np.linspace(-1.1, 1.5, model.num_params)
    if target == 'tilted':
        unit = np.log1p(unit*np.expm1(tilt))/tilt
    x = physical_coordinates(unit, model)
    y = x.copy()
    callback = model.perturb if move == 'mixture' else getattr(model, '_'+move)
    logh = np.array([callback(row) for row in y])
    assert np.all(np.isfinite(logh) | np.isneginf(logh))
    # The staged boundary policy must leave all coordinates unchanged.
    assert np.array_equal(y[np.isneginf(logh)], x[np.isneginf(logh)])
    uy = unit_coordinates(y, model)
    logratio = logh.copy()
    if target == 'tilted':
        logratio += (uy-unit)@tilt
    accepted = np.log(rng.random(n)) < np.minimum(0., logratio)
    out = unit.copy()
    out[accepted] = uy[accepted]
    reference = unit.copy()
    if target == 'tilted':
        out = np.expm1(tilt*out)/np.expm1(tilt)
        reference = np.expm1(tilt*reference)/np.expm1(tilt)
    mean_z = abs(out.mean(axis=0)-.5)*np.sqrt(12*n)
    ks = [kstest(out[:, j], 'uniform').statistic for j in range(model.num_params)]
    f, g = reference-.5, out-.5
    asymmetry_z=[]
    for i in range(model.num_params):
        for j in range(i+1, model.num_params):
            term=f[:, i]*g[:, j]-f[:, j]*g[:, i]
            se=term.std(ddof=1)/np.sqrt(n)
            asymmetry_z.append(float(abs(term.mean())/se) if se else 0.)
    count_z = {}
    if move == 'mixture':
        assert sum(model.proposal_counts.values()) == n
        for key,p in model.proposal_config()['attempt_probabilities'].items():
            count=model.proposal_counts[key]
            if p == 0:
                assert count == 0
            else:
                count_z[key]=abs(count-n*p)/np.sqrt(n*p*(1-p))
                assert count_z[key] < 6
        if mode == 'none': assert model.proposal_counts['slow_ridge'] == 0
        else: assert model.proposal_counts['slow_ridge'] > 0
    result=dict(mode=mode, move=move, target=target, n=n,
                acceptance_fraction=float(accepted.mean()),
                negative_infinite_logh_count=int(np.isneginf(logh).sum()),
                nonzero_finite_logh_count=int(np.count_nonzero(np.isfinite(logh)&(logh!=0))),
                maximum_uniform_mean_z=float(max(mean_z)), maximum_ks_D=float(max(ks)),
                maximum_detailed_balance_moment_z=max(asymmetry_z),
                proposal_config=model.proposal_config(), proposal_counts=model.proposal_counts,
                ridge_boundary_rejections=model.ridge_boundary_rejections,
                proposal_frequency_z=count_z)
    result['passed']=bool(max(mean_z)<6 and max(ks)<.015 and max(asymmetry_z)<6)
    assert result['passed'], result
    return result


def exact_callback_checks(priors):
    outcomes=[]
    for mode in ('none','independent','shared'):
        model=sampler.DNestModel(lambda p:0., mode=mode, priors=priors, seed=8)
        original=model.from_prior()
        original[0:2]=[1.,5.]
        if mode != 'none': original[8:11]=[.6,.9,30.]
        cases=[('ou_ridge', [0,1], [1.,2.], 0.)]
        if mode != 'none': cases.append(('slow_ridge',[8,9,10],[1.,1.,1.],2.))
        for name,indices,powers,coefficient in cases:
            step=.1; trial=original.copy();model._heavy_step=lambda:step
            logh=getattr(model,'_'+name)(trial)
            expected=original.copy();expected[indices]*=np.exp(step*np.array(powers))
            assert np.array_equal(trial,expected)
            assert logh==coefficient*step
            step=-.1
            reverse=getattr(model,'_'+name)(trial)
            assert np.allclose(trial,original,rtol=4e-16,atol=0)
            assert reverse==-logh
            for extreme in (1000.,-1000.):
                step=extreme; trial=original.copy()
                assert np.isneginf(getattr(model,'_'+name)(trial))
                assert np.array_equal(trial,original)
            outcomes.append({'mode':mode,'move':name,'reverse_and_exact_logH_passed':True,
                             'overflow_and_boundary_reject_without_mutation':True})
    # Prior types are part of the formula; default logarithmic q disables 2s move.
    default=sampler.DNestModel(lambda p:0.,mode='independent')
    assert default.proposal_config()['attempt_probabilities']['slow_ridge']==0
    return outcomes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--priors',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n',type=int,default=40000)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result={'staged_source':str(STAGED),'proposal_version':sampler.PROPOSAL_VERSION,
            'prior_source':str(args.priors.resolve()),'checks':[],
            'exact_callback_checks':exact_callback_checks(args.priors),
            'scope':'Actual staged callback toy checks only; no native or scientific fit'}
    i=0
    for mode,moves in [('none',['ou_ridge','mixture']),
                       ('independent',['slow_ridge','mixture']),
                       ('shared',['slow_ridge','mixture'])]:
        for move in moves:
            for target in ('prior','tilted'):
                result['checks'].append(check_target(mode,move,target,args.priors,args.n,2026102300+i))
                i+=1
    live=Path('/home/leonhard/software/MICA2-slowgp/src/mica_slowgp/sampler.py')
    frozen=Path('/home/leonhard/software/MICA2-slowgp/archive/stability-v1/src/mica_slowgp/sampler.py')
    result['live_sampler_identical_to_frozen_v1']=live.read_bytes()==frozen.read_bytes()
    assert result['live_sampler_identical_to_frozen_v1']
    result['passed']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'checks':len(result['checks']),
                      'version':sampler.PROPOSAL_VERSION,
                      'max_mean_z':max(x['maximum_uniform_mean_z'] for x in result['checks']),
                      'max_ks_D':max(x['maximum_ks_D'] for x in result['checks']),
                      'max_balance_z':max(x['maximum_detailed_balance_moment_z'] for x in result['checks']),
                      'live_sampler_unchanged':True,'output':str(args.output)},indent=2))


if __name__=='__main__':main()
