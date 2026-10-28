"""Exact restricted FFA likelihood, existing lag posterior, one distance per AGN.
This is staged inference, not a full joint MICA/lightcurve-to-cosmology model.
"""
from pathlib import Path
import sys, json, time, argparse
import numpy as np
import pandas as pd
import yaml, emcee
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import skew
from ffa.ffa import build_model
from .kernel import FastFFA
from .config import RunConfig
PROBS=np.array([ndtr(-1),.5,ndtr(1)])
SEEDS=[940401,940417]

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def quant(prefix,x):
    return dict(zip([prefix+'_q16',prefix+'_q50',prefix+'_q84'],map(float,np.quantile(x,PROBS))))

def ensemble(logp,initial,bounds,folder,tag,seed):
    rng=np.random.default_rng(seed); nw=32; ndim=len(initial); burn=1500
    width=np.array([b[1]-b[0] for b in bounds]); perturb=np.minimum(width*.01,.05)
    pos=initial+rng.normal(size=(nw,ndim))*perturb
    for i,(lo,hi) in enumerate(bounds):pos[:,i]=np.clip(pos[:,i],lo+width[i]*1e-5,hi-width[i]*1e-5)
    sampler=emcee.EnsembleSampler(nw,ndim,logp)
    sampler.random_state=np.random.RandomState(seed).get_state()
    state=pos
    for target in [6000,10000,16000,24000,32000]:
        state=sampler.run_mcmc(state,target-sampler.iteration,progress=False)
        tau=sampler.get_autocorr_time(discard=burn,tol=0)
        kept=sampler.iteration-burn;ess=nw*kept/tau
        ok=bool(np.isfinite(tau).all() and np.min(kept/tau)>50 and np.min(ess)>1500)
        diag=dict(tag=tag,seed=seed,steps=sampler.iteration,burn=burn,tau=tau.tolist(),ESS=ess.tolist(),acceptance=float(sampler.acceptance_fraction.mean()),converged=ok)
        dump(folder/(tag+'_diagnostics.json'),diag)
        print(json.dumps(diag),flush=True)
        if ok:break
    np.savez_compressed(folder/(tag+'_chain.npz'),chain=sampler.get_chain(discard=burn),log_prob=sampler.get_log_prob(discard=burn))
    if not ok:raise RuntimeError('MCMC not converged: '+tag)
    return sampler.get_chain(discard=burn,flat=True),diag

def source_run(source, settings: RunConfig):
    ROOT, OUT = settings.project_root, settings.output_dir
    SOURCES, REL = settings.sources, settings.result_subdir
    start=time.time();folder=OUT/'sources'/source;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'status.json').exists() and json.loads((folder/'status.json').read_text()).get('state')=='complete':
        validate_cached_source(source, settings)
        print(source+' already complete');return
    dump(folder/'status.json',dict(state='running',source=source))
    base=ROOT/source/REL;ffroot=base/'fflux'
    cf=yaml.safe_load((ffroot/'run.yaml').read_text());ff=cf['fflux'];fit=ff['fit'];sed=ff['sed']
    sc=yaml.safe_load((ROOT/source/'config/source_config.yaml').read_text());bands=list(ff['bands']);waves=np.array([ff['wavelengths'][b] for b in bands],float)
    assert len(bands)==6 and len(set(bands))==6
    ref=str(sed['gal_ref']);ri=bands.index(ref);wave0=waves[ri]
    shift=fit.get('time_shifts');shift={int(k):v for k,v in shift.items()} if shift else None
    model=build_model(filenames=[f'{source}_{b}_comp0.txt' for b in bands],data_dir=ffroot/'dered',time_shifts=shift,tau_min=float(fit['tau_min']),R_min=float(fit['r_min']),R_max=float(fit['r_max']))
    fast=FastFFA(model, library_path=OUT/'build/libffa_kernel.so'); logq=(-7/3)*np.log(waves/wave0)
    amin=float(np.max(model.param_range[1:,0]-logq));amax=float(np.min(model.param_range[1:,1]-logq))
    bounds=[tuple(model.param_range[0]),(amin,amax)]
    def expand(theta):return np.r_[theta[0],theta[1]+logq]
    def lp(theta):
        if any(not(lo<float(x)<hi) for x,(lo,hi) in zip(theta,bounds)):return -np.inf
        return fast.loglike(expand(theta))
    old=np.loadtxt(ffroot/'fit/posterior_sample.txt',ndmin=2)[:,:7]
    initial=np.array([np.mean(old[:,0]),np.mean(old[:,1:]-logq)])
    initial=np.clip(initial,np.array(bounds)[:,0]+1e-5,np.array(bounds)[:,1]-1e-5)
    obj=lambda t:-fast.loglike(expand(t))
    opt=minimize(obj,initial,method='Powell',bounds=bounds,options=dict(xtol=1e-8,ftol=1e-10,maxiter=600))
    theta=np.asarray(opt.x); shared_ll=-float(opt.fun)
    full_bounds=[tuple(x) for x in model.param_range]
    free_candidates=[expand(theta),np.mean(old,axis=0)]
    free_solutions=[minimize(lambda t:-fast.loglike(t),x,method='L-BFGS-B',bounds=full_bounds,options=dict(maxiter=800,ftol=1e-12)) for x in free_candidates]
    free=max([(-float(sol.fun),sol.x,bool(sol.success)) for sol in free_solutions]+[(shared_ll,expand(theta),False)],key=lambda a:a[0])
    comparison=dict(shared_optimizer_success=bool(opt.success),shared_optimizer_message=str(opt.message),shared_loglike=shared_ll,free_loglike=free[0],delta_minus2loglike=max(0.,2*(free[0]-shared_ll)),constraints=5,shared_map=theta.tolist(),free_map=free[1].tolist(),free_optimizer_success=free[2],likelihood='original FFA GP, offsets marginalized with both determinant terms')
    dump(folder/'model_comparison.json',comparison)
    reps=[];diags=[]
    si=SOURCES.index(source)
    for i,seed in enumerate(SEEDS):
        ch,di=ensemble(lp,theta,bounds,folder,f'replica_{i+1}',seed+si*101)
        reps.append(ch);diags.append(di)
    means=np.array([np.quantile(a,PROBS,axis=0) for a in reps])
    frac=np.max(abs(means[0]-means[1])/(means.mean(axis=0)[2]-means.mean(axis=0)[0]),axis=0)
    if np.max(frac)>.1:raise RuntimeError(f'replica discrepancy {source}: {frac}')
    allchain=np.concatenate(reps)
    rng=np.random.default_rng(940499+si)
    # Fixed posterior-mean driving reconstruction, as in the existing FFA pipeline.
    representative=expand(np.mean(allchain,axis=0))
    tg,xg=fast.reconstruct(representative,ngrid=500)
    assert sed['x_bright']=='max' and sed['x_faint']=='min'
    xb=float(np.max(xg));xf=float(np.min(xg));assert xb>xf
    np.savetxt(folder/'reconstructed_X.txt',np.c_[tg,xg],header='time X_t')
    selected=allchain[rng.choice(len(allchain),12000,replace=False)]
    aref=np.empty(len(selected))
    for k,row in enumerate(selected):
        _,amean,acov,_=fast.evaluate(expand(row))
        aref[k]=rng.normal(amean[ri],np.sqrt(acov[ri,ri]))
    rms=np.exp(selected[:,1]);bright=aref+rms*xb;faint=aref+rms*xf;epsilon=faint/bright
    lag=pd.read_csv(base/'lag/absolute_lag_posterior_b4_3.txt',sep=r'\s+')
    tau_all=lag.tau0.to_numpy();assert np.allclose(lag.beta,4/3)
    tau=rng.choice(tau_all,len(selected),replace=True)
    oldrun=pd.read_csv(base/'distance/standard/distance_run_summary.csv').iloc[0]
    assert np.isclose(oldrun.inclination_deg,45) and np.isclose(oldrun.disk_b,.75)
    K=float(oldrun.K_b_Mpc);z=float(sc['redshift'])
    valid=np.isfinite(bright)&np.isfinite(faint)&(bright>faint)&(faint>0)&(tau>0)&(epsilon>0)&(epsilon<1)
    if np.sum(valid)<2000:raise RuntimeError('too few physical distance samples '+source)
    fnu=bright[valid]*1e-15*wave0**2/2.99792458e18*1e23
    ep=epsilon[valid]
    Dsaved=K*tau[valid]*(wave0/1e4)**(-1.5)*np.sqrt(np.cos(np.pi/4)/fnu)*(1-ep)/(1-ep**1.5)
    DL=Dsaved*(1+z)
    assert np.isfinite(DL).all() and (DL>0).all()
    pd.DataFrame(dict(tau0=tau[valid],tau_DRW=np.exp(selected[valid,0]),A=rms[valid],A_ref=aref[valid],epsilon=ep,F_bright_ref=bright[valid],D_saved=Dsaved,D_L=DL)).to_csv(folder/'distance_posterior.csv',index=False)
    # Same common-SED parameters must give exactly the same distance at every band.
    consistency=[]
    for j,wave in enumerate(waves):
        fb=bright[valid]*np.exp(logq[j]);fn=fb*1e-15*wave**2/2.99792458e18*1e23
        db=K*tau[valid]*(wave/wave0)**(4/3)*(wave/1e4)**(-1.5)*np.sqrt(np.cos(np.pi/4)/fn)*(1-ep)/(1-ep**1.5)*(1+z)
        consistency.append(float(np.max(abs(db/DL-1))))
    assert max(consistency)<1e-10
    raw=pd.read_csv(ffroot/'analysis'/f'{source}_comp0_6band_flux_posterior.txt',sep=r'\s+')
    br=[]
    for j,b in enumerate(bands):
        br.append(dict(source=source,band=b,wavelength=waves[j],**quant('R_free',raw['R_'+b]),**quant('R_shared',np.exp(allchain[:,1]+logq[j]))))
    pd.DataFrame(br).to_csv(folder/'band_comparison.csv',index=False)
    fitlag=pd.read_csv(base/'lag/fit_summary.csv');fl=fitlag.loc[fitlag.fit_model.astype(str).eq('4/3')].iloc[0]
    boundary={}
    for i,name in enumerate(['tau_DRW','A']):
        lo,hi=bounds[i];v=allchain[:,i]
        boundary[name+'_lower_5pct']=float(np.mean(v<lo+.05*(hi-lo)))
        boundary[name+'_upper_5pct']=float(np.mean(v>hi-.05*(hi-lo)))
    summary=dict(source=source,z=z,reference_band=ref,n_bands=6,**quant('D',DL),D_mean=float(np.mean(DL)),D_std=float(np.std(DL,ddof=1)),D_skew=float(skew(DL)),**quant('tau0',tau[valid]),**quant('A',rms[valid]),**quant('epsilon',ep),**quant('tau_DRW',np.exp(selected[valid,0])),sed_delta_chi2=comparison['delta_minus2loglike'],sed_constraints=5,lag_chi2=float(fl.chi2),lag_dof=int(fl.dof),physical_fraction=float(valid.mean()),mcmc_converged=True,**boundary)
    pd.DataFrame([summary]).to_csv(folder/'summary.csv',index=False)
    config=dict(source=source,bands=bands,wavelengths=waves.tolist(),reference_band=ref,source_redshift=z,likelihood='Exact original FFA GP marginalized-offset likelihood on saved comp0 dered lightcurves',parameters=['log_tau_DRW','log_R_ref'],prior_bounds=bounds,priors='uniform in original log coordinates; shared R interval is intersection of original per-band bounds',SED='R_lambda=R_ref*(lambda/lambda_ref)^(-7/3)',disk_beta=4/3,inclination_deg=45,nuclear_extinction='unchanged; no added correction',lag_input=str(base/'lag/absolute_lag_posterior_b4_3.txt'),lag_inference='existing fixed-beta posterior propagation, used once per source',lag_FFA_joint='independent-stage approximation; no fabricated row pairing',X_extrema='one posterior-mean reconstruction, matching upstream convention',x_bright=xb,x_faint=xf,offset_prior='existing flat A_b, marginalized then conditionally sampled',distance_kind='retain D_saved convention; DL=(1+z)D_saved',mcmc=diags)
    dump(folder/'config.json',config)
    dump(folder/'validation.json',dict(replica_quantile_fraction=frac.tolist(),all_band_distance_max_relative_difference=max(consistency),boundary=boundary,raw_FFA_point_count=model.ndall,physical_distance_draws=int(valid.sum()),physical_fraction=float(valid.mean()),lag_positive_fraction=float(np.mean(tau_all>0)),no_error_inflation=True,gaussian_posterior_extrapolation_used=False,free_shared_comparison=comparison))
    dump(folder/'status.json',dict(state='complete',source=source,elapsed_seconds=time.time()-start,summary=summary))
    print(json.dumps(summary),flush=True)

def finalize(settings: RunConfig):
    OUT, SOURCES = settings.output_dir, settings.sources
    if len(SOURCES) < 2:
        raise ValueError("Joint H0 requires at least two distinct AGN")
    rows=[];bandrows=[]
    for source in SOURCES:
        validate_cached_source(source,settings)
        p=OUT/'sources'/source
        assert json.loads((p/'status.json').read_text())['state']=='complete'
        rows.append(pd.read_csv(p/'summary.csv'));bandrows.append(pd.read_csv(p/'band_comparison.csv'))
    d=pd.concat(rows,ignore_index=True)
    review_path=OUT/'independent_review.json'
    if review_path.exists():
        review=json.loads(review_path.read_text())
        values={r['source']:r['shared_reoptimization']['review_delta_chi2_using_best_shared'] for r in review['sources']}
        d['sed_delta_chi2_reviewed']=d.source.map(values)
    d.to_csv(OUT/'source_distance_summary.csv',index=False)
    pd.concat(bandrows,ignore_index=True).to_csv(OUT/'band_comparison.csv',index=False)
    K=np.array([299792.458*(1+z)*quad(lambda x:1/np.sqrt(.3*(1+x)**3+.7),0,z)[0] for z in d.z])
    obs=d.D_mean.to_numpy();sd=d.D_std.to_numpy()
    def lp(theta):
        h=float(theta[0])
        if not 1<h<800:return -np.inf
        return float(-.5*np.sum(((obs-K/h)/sd)**2+np.log(2*np.pi*sd**2)))
    ah=np.sum(K*obs/sd**2)/np.sum((K/sd)**2);hhat=1/ah
    reps=[];di=[]
    for i,seed in enumerate(SEEDS):
        ch,dg=ensemble(lp,np.array([hhat]),[(1.,800.)],OUT,f'h0_replica_{i+1}',seed+5000);reps.append(ch[:,0]);di.append(dg)
    H=np.concatenate(reps);qs=[np.quantile(a,PROBS) for a in reps];frac=float(np.max(abs(qs[0]-qs[1]))/np.diff(np.quantile(H,PROBS[[0,2]]))[0])
    if frac>.1:raise RuntimeError('H0 replicate disagreement')
    pd.DataFrame({'H0':H[::8]}).to_csv(OUT/'h0_posterior.csv',index=False)
    chi=float(np.sum(((obs-K/hhat)/sd)**2)); dof=len(d)-1
    hsum=dict(**quant('h0',H),h0_mle=hhat,chi2=chi,dof=dof,reduced_chi2=chi/dof,n_sources=len(d),n_source_distance_constraints=len(d),f0_fixed=0.,mcmc_converged=True,replica_quantile_fraction=frac)
    pd.DataFrame([hsum]).to_csv(OUT/'h0_summary.csv',index=False)
    d['D_model_at_H0_mle']=K/hhat;d['Hubble_residual_sigma']=(obs-K/hhat)/sd;d['Hubble_chi2_contribution']=d.Hubble_residual_sigma**2
    d.to_csv(OUT/'fit_quality.csv',index=False)
    # Leave-one-source sensitivity uses the analytic MLE of the same Gaussian likelihood.
    sensitivity=[]
    for i,s in enumerate(d.source):
        m=np.arange(len(d))!=i;a=np.sum(K[m]*obs[m]/sd[m]**2)/np.sum((K[m]/sd[m])**2)
        sensitivity.append(dict(omitted_source=s,H0_MLE=1/a))
    pd.DataFrame(sensitivity).to_csv(OUT/'leave_one_source_out.csv',index=False)
    dump(OUT/'validation.json',dict(state='numerically_complete',source_statuses='all complete',f0=0,peculiar_velocity=0,number_of_H0_likelihood_terms=len(d),H0_prior=[1,800],MICA_rerun=False,SED_inference='exact restricted FFA likelihood',H0_inference='two-stage Gaussian linear-distance summary approximation',Gaussian_D_covariance_inversion=False,shared_FFA_parameters_counted_once=True,kernel_provenance='verified scalar FFA implementation; see src/shared_sed_hubble/README.md',h0=hsum,visual_review='pending'))
    print(json.dumps(hsum,indent=2),flush=True)

def validate_cached_source(source, settings):
    """Reject cached products from a different source/model/input branch."""
    folder=settings.output_dir/'sources'/source
    status=json.loads((folder/'status.json').read_text())
    config=json.loads((folder/'config.json').read_text())
    expected=settings.project_root/source/settings.result_subdir/'lag/absolute_lag_posterior_b4_3.txt'
    if status.get('state')!='complete' or status.get('source')!=source:
        raise ValueError(f'Incomplete or mismatched cached source: {source}')
    if (not np.isclose(config['disk_beta'],4/3)
            or not np.isclose(config['inclination_deg'],45)
            or Path(config['lag_input']).resolve()!=expected.resolve()
            or config['SED']!='R_lambda=R_ref*(lambda/lambda_ref)^(-7/3)'):
        raise ValueError(f'Cached model/branch mismatch: {source}; use a separate output directory')


def validate_results(settings):
    """Independent posterior/grid and distance-unit regression on saved results."""
    from scipy.integrate import cumulative_trapezoid
    OUT=settings.output_dir
    d=pd.read_csv(OUT/'source_distance_summary.csv')
    if tuple(d.source)!=settings.sources:
        raise ValueError('Summary sources differ from configured sources')
    h=pd.read_csv(OUT/'h0_summary.csv').iloc[0]
    if int(h.n_source_distance_constraints)!=len(d) or float(h.f0_fixed)!=0:
        raise ValueError('Wrong number of source constraints or nonzero f0')
    ess=[];replica=[];banddiff=[];formula_diff=[]
    for source in settings.sources:
        validate_cached_source(source,settings)
        folder=OUT/'sources'/source
        v=json.loads((folder/'validation.json').read_text())
        banddiff.append(v['all_band_distance_max_relative_difference'])
        replica.extend(v['replica_quantile_fraction'])
        for f in folder.glob('replica_*_diagnostics.json'):
            di=json.loads(f.read_text())
            if not di['converged']:raise ValueError(f'Unconverged source chain: {f}')
            ess.extend(di['ESS'])
        c=json.loads((folder/'config.json').read_text())
        samples=pd.read_csv(folder/'distance_posterior.csv')
        base=settings.project_root/source/settings.result_subdir
        k=float(pd.read_csv(base/'distance/standard/distance_run_summary.csv').iloc[0].K_b_Mpc)
        wave=c['wavelengths'][c['bands'].index(c['reference_band'])]
        ep=samples.epsilon.to_numpy()
        fn=samples.F_bright_ref.to_numpy()*1e-15*wave**2/2.99792458e18*1e23
        direct=k*samples.tau0.to_numpy()*(wave/1e4)**(-1.5)*np.sqrt(np.cos(np.pi/4)/fn)*(1-ep)/(1-ep**1.5)*(1+c['source_redshift'])
        formula_diff.append(float(np.max(abs(direct/samples.D_L.to_numpy()-1))))
    scale=np.array([299792.458*(1+z)*quad(lambda x:1/np.sqrt(.3*(1+x)**3+.7),0,z)[0] for z in d.z])
    grid=np.linspace(1,800,200001)
    ll=-.5*np.sum(((d.D_mean.to_numpy()[:,None]-scale[:,None]/grid)/d.D_std.to_numpy()[:,None])**2,axis=0)
    cdf=cumulative_trapezoid(np.exp(ll-ll.max()),grid,initial=0);cdf/=cdf[-1]
    q=np.interp(PROBS,cdf,grid);mc=h[['h0_q16','h0_q50','h0_q84']].to_numpy(float)
    fraction=float(np.max(abs(q-mc))/(q[2]-q[0]))
    if fraction>.03 or min(ess)<1500 or max(banddiff)>1e-10 or max(formula_diff)>1e-10:
        raise ValueError('Numerical validation failed; inspect saved diagnostics')
    report=dict(state='passed',H0_independent_grid_quantiles=q.tolist(),H0_MCMC_quantiles=mc.tolist(),
                H0_max_absolute_quantile_difference=float(abs(q-mc).max()),minimum_source_parameter_ESS=min(ess),
                maximum_source_replica_quantile_difference_as_interval_fraction=max(replica),
                maximum_all_band_distance_relative_difference=max(banddiff),maximum_distance_formula_relative_difference=max(formula_diff),
                number_of_H0_likelihood_terms=len(d),physical_fraction_min=float(d.physical_fraction.min()))
    dump(OUT/'numerical_validation.json',report)
    return report
