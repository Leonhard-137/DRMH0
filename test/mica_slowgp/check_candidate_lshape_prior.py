"""Independent prior/analytic integration checks; no GP or sampler runs."""
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import kstest

from candidate_lshape_prior import LShapePriorTransform, lshape_centers


BASE=Path('/home/leonhard/sci001/output/mica_slowgp/sampler_smoke/candidate_lshape_prior')
PRIOR=Path('/home/leonhard/sci001/output/mica_slowgp/small_cases/priors.json')


def conditional_prior_cdf(x, transform):
    z=x.copy()
    z[:,transform.logscale]=np.log(z[:,transform.logscale])
    z=(z-transform.uniform_bounds[:,0])/np.diff(transform.uniform_bounds,axis=1)[:,0]
    a=transform.bounds[3,0];lo,hi=transform.center_interval
    z[:,3]=(x[:,3]-a)/(x[:,6]-a)
    z[:,6]=((x[:,6]-a)**2-(lo-a)**2)/((hi-lo)*(hi+lo-2*a))
    return z


if __name__=='__main__':
    start=time.perf_counter()
    priors=json.loads(PRIOR.read_text())
    rng=np.random.default_rng(183061)
    n=120000
    thresholds={'mean_zscore':6.0,'ks_distance':.01,'absolute_correlation':.02,
                'analytic_integration_zscore':6.0,'finite_difference_jacobian_relative_error':1e-8}
    checks={};details=[]
    tilt=np.array([.5,-.4,.6,-.3,.2,-.7,1.1,.8,-.5,.4,-.6])
    exact_z=float(np.prod(np.expm1(tilt)/tilt))
    exact_mean=np.exp(tilt)/np.expm1(tilt)-1/tilt
    masses=[]
    for j,(lo,hi) in enumerate(((-3.,5.),(5.,20.),(20.,28.))):
        t=LShapePriorTransform(priors,(lo,hi))
        a=t.bounds[3,0]
        u=rng.random((n,11));x=t(u)
        z=conditional_prior_cdf(x,t)
        masses.append(t.center_prior_mass)
        mean_zscore=(z.mean(axis=0)-.5)*np.sqrt(12*n)
        ks=np.array([kstest(z[:,i],'uniform').statistic for i in range(11)])
        corr=np.corrcoef(z,rowvar=False);np.fill_diagonal(corr,0)
        checks[f'stratum_{j}_support']=bool(np.all(x[:,3] < x[:,6]) and np.all(x[:,3]>=a) and np.all(x[:,6]>=lo) and np.all(x[:,6]<=hi))
        checks[f'stratum_{j}_prior_marginals']=bool(np.max(np.abs(mean_zscore))<thresholds['mean_zscore'] and ks.max()<thresholds['ks_distance'])
        checks[f'stratum_{j}_prior_joint_independence']=bool(np.max(np.abs(corr))<thresholds['absolute_correlation'])
        # Analytic, nonflat target on all eleven conditional prior-CDF axes.
        likelihood=np.exp(z@tilt);weight=likelihood/likelihood.sum()
        integral=float(likelihood.mean());integral_se=float(likelihood.std(ddof=1)/np.sqrt(n))
        tilted_mean=weight@z
        mean_se=np.sqrt(np.sum((weight[:,None]*(z-tilted_mean))**2,axis=0))
        integration_zscore=abs(integral-exact_z)/integral_se
        tilted_mean_zscore=(tilted_mean-exact_mean)/mean_se
        checks[f'stratum_{j}_nonflat_analytic_toy']=bool(integration_zscore<thresholds['analytic_integration_zscore'] and np.max(np.abs(tilted_mean_zscore))<thresholds['analytic_integration_zscore'])
        # Check branch/cube endpoints without shifting measure-zero ties.
        p=(lo-a)/(hi+lo-2*a)
        axis=np.unique(np.array([0.,1.,p,np.nextafter(p,0),np.nextafter(p,1)]))
        axis=axis[(axis>=0)&(axis<=1)]
        uu,vv=np.meshgrid(axis,[0.,.5,1.],indexing='ij')
        c0,c1=lshape_centers(uu,vv,a,lo,hi)
        checks[f'stratum_{j}_endpoints']=bool(np.all(c0>=a-1e-13) and np.all(c1>=a-1e-13) and np.all(c0<=hi+1e-13) and np.all(c1<=hi+1e-13) and np.all(np.maximum(c0,c1)>=lo-1e-13))
        boundary_gap=np.max(np.abs(np.array(lshape_centers(np.nextafter(p,0),.5,a,lo,hi))-np.array(lshape_centers(p,.5,a,lo,hi)))) if p>0 else 0.
        # Jacobian is Delta in either open branch; test away from seams.
        area=(hi-lo)*(hi+lo-2*a);det_errors=[]
        for u0 in ([p/2] if p>0 else [])+[(1+p)/2]:
            u1=.37;eps=1e-6
            d0=(np.array(lshape_centers(u0+eps,u1,a,lo,hi))-np.array(lshape_centers(u0-eps,u1,a,lo,hi)))/(2*eps)
            d1=(np.array(lshape_centers(u0,u1+eps,a,lo,hi))-np.array(lshape_centers(u0,u1-eps,a,lo,hi)))/(2*eps)
            det_errors.append(abs(np.linalg.det(np.column_stack((d0,d1)))/area-1))
        checks[f'stratum_{j}_unordered_jacobian']=max(det_errors)<thresholds['finite_difference_jacobian_relative_error']
        # Explicit full-triple crossing with distinct amplitude/width markers.
        marker=np.full(11,.5);marker[[2,3,4,5,6,7]]=[.2,.9,.3,.8,.1,.7]
        marked=t(marker)
        bounds=t.uniform_bounds
        raw=bounds[:,0]+marker*np.diff(bounds,axis=1)[:,0];raw[t.logscale]=np.exp(raw[t.logscale])
        raw[3],raw[6]=lshape_centers(marker[3],marker[6],a,lo,hi)
        checks[f'stratum_{j}_whole_triple_sort']=bool(raw[3]>raw[6] and np.array_equal(marked[2:5],raw[5:8]) and np.array_equal(marked[5:8],raw[2:5]))
        details.append({'stratum':[lo,hi],'prior_mass':t.center_prior_mass,'sample_count':n,
            'max_prior_mean_zscore':float(np.max(np.abs(mean_zscore))),'max_ks':float(ks.max()),'max_absolute_correlation':float(np.max(np.abs(corr))),
            'nonflat_exact_evidence':exact_z,'nonflat_mc_evidence':integral,'evidence_zscore':float(integration_zscore),'max_nonflat_mean_zscore':float(np.max(np.abs(tilted_mean_zscore))),
            'max_jacobian_relative_error':float(max(det_errors)),'seam_jump_example_at_u1_half':float(boundary_gap)})
        if lo==a:
            # Original v2 unit-cube mapping on the reduced square [a,hi]^2.
            expected=t.uniform_bounds[:,0]+u*np.diff(t.uniform_bounds,axis=1)[:,0]
            expected[:,t.logscale]=np.exp(expected[:,t.logscale])
            expected[:,3]=a+(hi-a)*u[:,3];expected[:,6]=a+(hi-a)*u[:,6]
            exchange=expected[:,3]>expected[:,6]
            tmp=expected[exchange,2:5].copy();expected[exchange,2:5]=expected[exchange,5:8];expected[exchange,5:8]=tmp
            checks['lower_stratum_pointwise_identical_to_original_square_then_sort']=np.array_equal(expected,x)
    checks['original_three_stratum_masses_preserved']=np.allclose(masses,np.array([64,465,432])/961,rtol=0,atol=2e-16)
    checks['original_three_masses_sum_to_one']=abs(sum(masses)-1)<2e-16
    rejected={}
    for name,interval in [('empty',(5,5)),('below_prior',(-4,5)),('above_prior',(20,29))]:
        try:LShapePriorTransform(priors,interval)
        except ValueError:rejected[name]=True
        else:rejected[name]=False
    changed={k:dict(v) for k,v in priors.items()};changed['A1']={'kind':'log_uniform','bounds':[.03,2.5]}
    try:LShapePriorTransform(changed,(-3,5))
    except ValueError:rejected['nonexchangeable_amplitudes']=True
    else:rejected['nonexchangeable_amplitudes']=False
    checks['invalid_domains_and_nonexchangeable_prior_rejected']=all(rejected.values())
    # Both eight- and eleven-dimensional modes use this same center law.
    for mode in ('none','independent','shared'):
        t=LShapePriorTransform(priors,(-3,5),mode)
        sample=t(np.full(len(t.names),.4))
        checks[f'{mode}_shape']=sample.shape==(8 if mode=='none' else 11,)
    report={'passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'thresholds':thresholds,'strata':details,
        'invalid_input_checks':rejected,'elapsed_seconds':time.perf_counter()-start,
        'candidate_adopted':False,'gp_likelihood_calls':0,'scientific_sampler_runs':0,
        'limitations':['The equivalent prior does not imply better mixing.','For lo>a, the piecewise inverse map has a center-coordinate jump across its seam; prior measure remains correct.','Exact cube endpoints may yield equal centers on zero-measure sets; these are not moved or assigned artificial prior mass.','Nonexchangeable component priors are unsupported.']}
    BASE.mkdir(parents=True,exist_ok=True)
    (BASE/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
