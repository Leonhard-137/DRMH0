"""Analytic and density tests only; imports no GP or science sampler."""
from pathlib import Path
import json
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,ndtri,logsumexp,expit
from scipy.stats import multivariate_t

from importance import (FrozenProposal,fit_gaussian_mixture,freeze_student_mixture,
    logit_open,log_prior_z,unit_logs,evaluate_u,summarize_logweights,combine_strata,json_safe)

BASE=Path(__file__).resolve().parent/'artifacts'


if __name__=='__main__':
    started=time.perf_counter();BASE.mkdir(parents=True,exist_ok=True)
    checks={};results={};rng=np.random.default_rng(851031)
    train_uniform=rng.random((5000,2));fit=fit_gaussian_mixture(logit_open(train_uniform),np.ones(5000),seed=851032,max_iter=40)
    proposal=freeze_student_mixture([fit],[.95],metadata={'purpose':'constant analytic toy','training_seed':851031})
    testz=rng.normal(size=(200,2))*5
    independent=np.column_stack([np.log(w)+multivariate_t.logpdf(testz,loc=m,shape=s,df=proposal.df)
        for w,m,s in zip(proposal.component_weights,proposal.means,proposal.scales)]+[np.log(proposal.epsilon)+log_prior_z(testz)])
    density_error=float(np.max(np.abs(proposal.logpdf_z(testz)-logsumexp(independent,axis=1))))
    checks['normalized_student_density_matches_scipy']=density_error<2e-12
    one=FrozenProposal(np.array([.95]),np.array([[.3]]),np.array([[[1.7]]]),7,.05)
    normalization=quad(lambda x:np.exp(one.logpdf_z(np.array([x]))),-np.inf,np.inf,epsabs=1e-10)[0]
    checks['full_defensive_mixture_integrates_to_one']=abs(normalization-1)<1e-9
    draws=proposal.sample_z(100000,np.random.default_rng(851033))
    exact_mean=proposal.component_weights@proposal.means
    exact_second=np.einsum('k,kij->ij',proposal.component_weights,proposal.scales*proposal.df/(proposal.df-2)+np.einsum('ki,kj->kij',proposal.means,proposal.means))+proposal.epsilon*np.pi**2/3*np.eye(2)
    moments=np.column_stack((draws,draws[:,0]**2,draws[:,0]*draws[:,1],draws[:,1]**2))
    exact_moments=np.r_[exact_mean,exact_second[0,0],exact_second[0,1],exact_second[1,1]]
    moment_z=(moments.mean(axis=0)-exact_moments)/(moments.std(axis=0,ddof=1)/np.sqrt(len(draws)))
    checks['student_sampler_mean_and_covariance_conversion']=np.max(np.abs(moment_z))<6
    constant=evaluate_u(lambda u:0.,proposal,50000,851034,allow_closed_unit=True,
        observable_functions={'u0':lambda u:u[0],'u0_squared':lambda u:u[0]**2})
    cd=constant['diagnostics'];cz=np.exp(cd['log_evidence']);cse=np.exp(cd['log_evidence_standard_error'])
    checks['constant_target_known_evidence']=abs(cz-1)<6*cse
    checks['defensive_density_bound']=np.max(constant['logp_z']-constant['logq_z'])<=-np.log(proposal.epsilon)+2e-14
    checks['constant_target_known_moments']=all(abs(cd['observables'][name]['posterior_mean']-value)<6*cd['observables'][name]['ratio_delta_standard_error'] for name,value in [('u0',.5),('u0_squared',1/3)])
    results['constant']={'estimate':cz,'standard_error':cse,'diagnostics':cd}

    # Exact two-mode bounded-unit target and separate exact training draws.
    means=np.array([[.2,.25],[.8,.75]]);widths=np.array([[.06,.10],[.08,.07]]);mix=np.array([.35,.65])
    lower=ndtr(-means/widths);upper=ndtr((1-means)/widths)
    component_z=np.prod(upper-lower,axis=1);true_z=float(mix@component_z)
    component_probability=mix*component_z/true_z
    training_rng=np.random.default_rng(852001);labels=training_rng.choice(2,size=6000,p=component_probability)
    uu=training_rng.random((6000,2));train=means[labels]+widths[labels]*ndtri(lower[labels]+uu*(upper[labels]-lower[labels]))
    base=fit_gaussian_mixture(logit_open(train),np.ones(6000),seed=852002,max_iter=40)
    tail=fit_gaussian_mixture(logit_open(train),(train[:,0]<.5).astype(float),seed=852003,max_iter=40)
    bimodal=freeze_student_mixture([base,tail],[.475,.475],metadata={'purpose':'two-mode analytic toy','training_seed':852001})
    def target(u):
        return float(logsumexp(np.log(mix)-np.log(2*np.pi)-np.log(widths).sum(axis=1)-.5*np.sum(((u-means)/widths)**2,axis=1)))
    evaluated=evaluate_u(target,bimodal,60000,852004,allow_closed_unit=True,
        observable_functions={'left_half':lambda u:float(u[0]<.5)})
    bd=evaluated['diagnostics'];estimate=np.exp(bd['log_evidence']);se=np.exp(bd['log_evidence_standard_error'])
    exact_tail=float(np.sum(mix*(ndtr((.5-means[:,0])/widths[:,0])-lower[:,0])*(upper[:,1]-lower[:,1]))/true_z)
    checks['bimodal_target_known_evidence']=abs(estimate-true_z)<6*se
    checks['bimodal_target_known_tail_mass']=abs(bd['observables']['left_half']['posterior_mean']-exact_tail)<6*bd['observables']['left_half']['ratio_delta_standard_error']
    results['bimodal']={'exact_evidence':true_z,'estimate':estimate,'standard_error':se,'exact_tail_mass':exact_tail,'diagnostics':bd}
    bimodal.save(BASE/'toy_proposal.npz');restored=FrozenProposal.load(BASE/'toy_proposal.npz')
    checks['frozen_roundtrip_logpdf_and_draws_exact']=np.array_equal(restored.logpdf_z(testz),bimodal.logpdf_z(testz)) and np.array_equal(restored.sample_z(1000,np.random.default_rng(90)),bimodal.sample_z(1000,np.random.default_rng(90)))
    try:restored.means[0,0]=1
    except ValueError:checks['frozen_distribution_arrays_read_only']=True
    else:checks['frozen_distribution_arrays_read_only']=False

    # Unequal N_s plus zeros test the required p_s/N_s per-row factor.
    masses=np.array([64,465,432])/961
    synthetic=[{'logw':np.log(np.array([1.,3.])), 'observables':{'f':np.array([0.,1.])}},
               {'logw':np.array([-np.inf,np.log(4),np.log(4),-np.inf]), 'observables':{'f':np.array([0.,1.,0.,1.])}},
               {'logw':np.log(np.array([3.,3.,3.])), 'observables':{'f':np.array([.2,.4,.6])}}]
    combined=combine_strata(synthetic,masses)
    expected_rows=np.concatenate([m*np.exp(e['logw'])/len(e['logw']) for m,e in zip(masses,synthetic)])
    checks['unequal_stratum_sizes_exact_mass_merge']=np.allclose(combined['weights'],expected_rows/expected_rows.sum(),rtol=1e-14) and abs(np.exp(combined['log_evidence'])-expected_rows.sum())<1e-14
    checks['zero_weight_rows_and_all_strata_retained']=len(combined['weights'])==9 and (combined['weights']==0).sum()==2 and set(combined['stratum_id'])=={0,1,2}
    exact_f=float(expected_rows@np.concatenate([e['observables']['f'] for e in synthetic])/expected_rows.sum())
    checks['stratified_observable_alignment']=abs(combined['observable_summaries']['f']['posterior_mean']-exact_f)<1e-14

    # Explicit extreme coordinates: natural expit may hit closed endpoints;
    # target L(u)=1+u0*u1 is continuous, bounded, and evaluated there normally.
    extremes=np.array([[-1000.,1000.],[1000.,-1000.],[1000.,1000.],[-1000.,-1000.]])
    logs=unit_logs(extremes);p=log_prior_z(extremes);q=bimodal.logpdf_z(extremes);closed=expit(extremes)
    ll=np.log1p(closed[:,0]*closed[:,1]);logw=ll+p-q
    checks['extreme_logit_jacobian_and_full_density_finite']=np.isfinite(p).all() and np.isfinite(q).all() and np.isfinite(logw).all() and np.allclose(p,-2000.)
    checks['closed_endpoint_continuous_target_not_dropped']=np.array_equal(ll,np.array([0.,0.,np.log(2.),0.]))
    # Force only sampling output in a toy wrapper to exercise evaluation policy.
    class FixedDrawToy:
        def sample_z(self,n,rng,return_components=False):return (extremes.copy(),np.zeros(4,dtype=int)) if return_components else extremes.copy()
        def logpdf_z(self,z):return bimodal.logpdf_z(z)
    try:evaluate_u(lambda u:np.log1p(u[0]*u[1]),FixedDrawToy(),4,1)
    except FloatingPointError:checks['strict_mode_rejects_saturation_before_target']=True
    else:checks['strict_mode_rejects_saturation_before_target']=False
    extreme_eval=evaluate_u(lambda u:np.log1p(u[0]*u[1]),FixedDrawToy(),4,1,allow_closed_unit=True)
    checks['closed_mode_reports_saturation_and_keeps_rows']=extreme_eval['diagnostics']['saturated_rows']==4 and len(extreme_eval['logw'])==4 and np.array_equal(extreme_eval['logw'],logw)
    results.update(density_log_error=density_error,one_dimensional_normalization=normalization,
        max_sampler_moment_zscore=float(np.max(np.abs(moment_z))),extreme_logweights=logw.tolist())
    report={'passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()},'results':results,
        'predeclared_numerical_tolerance':'6 iid empirical standard errors for stochastic moments/evidence; density and algebraic checks use stated roundoff tolerances',
        'elapsed_seconds':time.perf_counter()-started,'gp_likelihood_calls':0,'scientific_fits_started':0,
        'training_evaluation_separate':True,'evidence_boundary':'Analytic numerical smoke checks; no scientific convergence claim.'}
    report['json_log_zero_convention']='A null logarithm of a zero contribution represents -infinity; computational arrays are unchanged.'
    (BASE/'toy_checks.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(json_safe(report),indent=2,allow_nan=False))
