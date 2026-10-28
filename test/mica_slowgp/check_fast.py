#!/usr/bin/env python3
"""Reproducible reference/fast numerical checks and same-process timings."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import scipy

SOURCE = Path('/home/leonhard/software/MICA2-slowgp/src')
sys.path.insert(0, str(SOURCE))
from mica_slowgp.core import PairData, build_covariance, gaussian_ou_covariance, log_likelihood
from mica_slowgp.fast import FastPairLikelihood
from mica_slowgp import _fast_kernel
from mica_slowgp.sampler import default_priors


def sample_parameters(rng, priors, count):
    rows = []
    for _ in range(count):
        p = {}
        for name, spec in priors.items():
            lo, hi = spec['bounds']
            p[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))) if spec['kind']=='log_uniform'
                            else rng.uniform(lo, hi))
        p['mu0'], p['mu1'] = sorted((p['mu0'], p['mu1']))
        rows.append(p)
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--random-count',type=int,default=1000)
    parser.add_argument('--timing-repeats',type=int,default=100)
    args=parser.parse_args(argv)
    root=Path('/home/leonhard/sci001/output/mica_slowgp/small_cases')
    rng=np.random.default_rng(2026102203)
    priors=json.loads((root/'priors.json').read_text())
    rows=sample_parameters(rng,priors,args.random_count)
    wide_rows=sample_parameters(rng,default_priors(),128)
    base=dict(sigma_x=.8,tau_x=12.,A0=.9,mu0=1.2,w0=.7,A1=.3,mu1=9.,w1=4.,qW=.45,qb=.8,Ts=30.)
    edges=[]
    for change in [dict(A0=0.),dict(A1=0.),dict(A0=0.,A1=0.),dict(w0=0.,w1=0.),
                   dict(mu0=2.,mu1=2.),dict(qW=0.,qb=0.),dict(qW=0.),dict(qb=0.),
                   dict(w0=1e-12,w1=1e-12),dict(w0=1e5,w1=1e5,tau_x=1e-6),
                   dict(tau_x=1e8),dict(Ts=1e-8),dict(Ts=1e12)]:
        edges.append(base|change)
    # Every declared small-test prior boundary is tested independently.
    for name,spec in priors.items():
        for value in spec['bounds']:
            p=base|{name:value}
            p['mu0'],p['mu1']=sorted((p['mu0'],p['mu1']))
            edges.append(p)
    report={'seed':2026102203,'source':str(SOURCE),'interpreter':sys.executable,
            'numpy':np.__version__,'scipy':scipy.__version__,'strict_loglike_absolute_tolerance':1e-8,
            'random_parameter_sets':len(rows),'wide_default_prior_sets':len(wide_rows),
            'boundary_parameter_sets':len(edges),'cases':[],'failures':[]}
    all_rows=rows+wide_rows+edges
    for case in ('independent','shared'):
        w=np.loadtxt(root/case/'w2.txt'); b=np.loadtxt(root/case/'response.txt')
        data=PairData(*w.T,*b.T)
        for mode in ('none','independent','shared'):
            candidates={backend:FastPairLikelihood(data,mode,backend) for backend in ('python','compiled')}
            stats={key:{'max_abs_loglike_difference':0.,'max_abs_covariance_difference':0.,'failures':0}
                   for key in candidates}
            for index,p in enumerate(all_rows):
                ref=log_likelihood(data,p,mode)
                # All covariance comparisons are independent of factorization.
                reference_cov=build_covariance(data,p,mode)
                for key,fn in candidates.items():
                    actual=fn(p)
                    difference=abs(actual-ref) if np.isfinite(actual) and np.isfinite(ref) else (0. if actual==ref else np.inf)
                    covariance_difference=float(np.max(np.abs(fn.covariance(p)-reference_cov)))
                    stats[key]['max_abs_loglike_difference']=max(stats[key]['max_abs_loglike_difference'],difference)
                    stats[key]['max_abs_covariance_difference']=max(stats[key]['max_abs_covariance_difference'],covariance_difference)
                    if difference>1e-8:
                        stats[key]['failures']+=1
                        if len(report['failures'])<12:
                            report['failures'].append({'case':case,'mode':mode,'backend':key,'index':index,
                                                       'reference':ref,'actual':actual,'difference':difference,'parameters':p})
            timing={}
            funcs={'core':lambda p:log_likelihood(data,p,mode)}|candidates
            for key,fn in funcs.items():
                for _ in range(5):fn(base)
                elapsed=[]
                for _ in range(args.timing_repeats):
                    start=time.perf_counter();fn(base);elapsed.append(time.perf_counter()-start)
                timing[key]={'median_ms':float(1000*np.median(elapsed)),
                             'p10_ms':float(1000*np.quantile(elapsed,.1)),
                             'p90_ms':float(1000*np.quantile(elapsed,.9))}
            report['cases'].append({'case':case,'mode':mode,'observations':len(data.y),
                                    'comparisons_per_backend':len(all_rows),'accuracy':stats,'timing':timing})
    delta=np.ascontiguousarray(np.array([-1e5,-100.,-1.,-.001,0.,.001,1.,100.,1e5]))
    kernels=[]
    for width,tau in [(0.,1.),(5e-324,1.),(1e-12,1.),(1.,1e-12),(1e5,1e-3),(1e308,1e-10),(1.,1e308)]:
        ref=gaussian_ou_covariance(delta,width,tau)
        actual=_fast_kernel.gaussian_ou_values(delta,width,tau)
        positive=ref>np.finfo(float).tiny
        passed=bool(np.allclose(actual,ref,rtol=2e-13,atol=0.,equal_nan=False))
        kernels.append({'width':width,'tau':tau,'passed':passed,
                        'max_absolute_difference':float(np.max(abs(actual-ref))),
                        'max_relative_difference_nonsubnormal':float(np.max(abs(actual[positive]/ref[positive]-1))) if positive.any() else 0.})
    report['extreme_kernel_checks']=kernels
    report['all_passed']=not report['failures'] and all(x['passed'] for x in kernels)
    report['evidence_boundary']='Numerical equivalence over these inputs and timings only; no proof of posterior convergence or physical recovery.'
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'all_passed':report['all_passed'],'report':str(args.report),
                      'failures':report['failures'],
                      'cases':[{'case':x['case'],'mode':x['mode'],'accuracy':x['accuracy'],'timing':x['timing']} for x in report['cases']],
                      'extreme_kernel_checks':kernels},indent=2,allow_nan=False))
    return 0 if report['all_passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
