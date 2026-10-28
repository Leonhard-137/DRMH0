"""Frozen defensive importance proposal candidate; no GP/sampler imports.

All Student-t component weights are FULL mixture probabilities: they sum to
1-epsilon. The remaining epsilon multiplies the exact logistic prior density.
Training is approximate; independent importance evaluation uses exact q_z.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import expit, gammaln, logsumexp


def logit_open(u):
    u=np.asarray(u,dtype=float)
    if not np.isfinite(u).all() or np.any(u<=0) or np.any(u>=1):
        raise ValueError('Training coordinates must be strictly inside (0,1); no clipping is performed')
    return np.log(u)-np.log1p(-u)


def json_safe(value):
    """Report-only serialization: mathematical nonfinite values become null.

    This does not alter any sampling, density, or importance-weight arrays.
    A null zero-contribution log represents log(0)=-infinity.
    """
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    return value


def unit_logs(z):
    z=np.asarray(z,dtype=float)
    if not np.isfinite(z).all():
        raise ValueError('Finite z required')
    return -np.logaddexp(0.,-z),-np.logaddexp(0.,z)


def log_prior_z(z):
    logu,log1mu=unit_logs(z)
    return (logu+log1mu).sum(axis=-1)


def conditional_prior_coordinates(samples,names,priors,center_interval):
    """Inverse of current sorted-center conditional prior, without relabeling."""
    x=np.asarray(samples,dtype=float)
    bounds=np.array([priors[name]['bounds'] for name in names],dtype=float)
    logs=np.array([priors[name]['kind']=='log_uniform' for name in names])
    y=x.copy();transformed=bounds.copy()
    y[:,logs]=np.log(y[:,logs]);transformed[logs]=np.log(transformed[logs])
    u=(y-transformed[:,0])/np.diff(transformed,axis=1)[:,0]
    j0,j1=names.index('mu0'),names.index('mu1')
    a=bounds[j0,0];lo,hi=map(float,center_interval)
    u[:,j0]=(x[:,j0]-a)/(x[:,j1]-a)
    u[:,j1]=(x[:,j1]-lo)*(x[:,j1]+lo-2*a)/((hi-lo)*(hi+lo-2*a))
    return u


def _gaussian_logpdf(x,means,covariances):
    result=np.empty((len(x),len(means)))
    d=x.shape[1]
    for k,(mean,cov) in enumerate(zip(means,covariances)):
        chol=np.linalg.cholesky(cov)
        residual=solve_triangular(chol,(x-mean).T,lower=True,check_finite=False).T
        result[:,k]=-.5*(d*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+np.sum(residual**2,axis=1))
    return result


def fit_gaussian_mixture(z,weights,n_components=6,seed=12345,max_iter=60,
                         tolerance=1e-5,regularization=1e-4):
    """Weighted EM in standardized z coordinates; this trains q, not a target.

    The covariance ridge only regularizes the proposal. No target prior or
    likelihood is modified. All positive training weights are retained.
    """
    started=time.perf_counter();z=np.asarray(z,dtype=float);w=np.asarray(weights,dtype=float)
    if z.ndim!=2 or w.shape!=(len(z),) or not np.isfinite(z).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Invalid weighted training sample')
    keep=w>0;z=z[keep];w=w[keep]/w[keep].sum()
    if not 1<=n_components<=len(z) or regularization<=0 or max_iter<1:
        raise ValueError('Invalid mixture settings')
    rng=np.random.default_rng(seed);d=z.shape[1]
    origin=w@z;scale=np.sqrt(w@((z-origin)**2))
    if np.any(scale<=0):
        raise ValueError('Degenerate weighted training coordinate')
    x=(z-origin)/scale
    means=[x[rng.choice(len(x),p=w)].copy()]
    distance=np.sum((x-means[0])**2,axis=1)
    for k in range(1,n_components):
        probability=w*distance
        probability=probability/probability.sum() if probability.sum()>0 else w
        means.append(x[rng.choice(len(x),p=probability)].copy())
        distance=np.minimum(distance,np.sum((x-means[-1])**2,axis=1))
    means=np.asarray(means)
    global_cov=(x*w[:,None]).T@x+regularization*np.eye(d)
    covariances=np.repeat(global_cov[None,:,:],n_components,axis=0)
    proportions=np.full(n_components,1/n_components)
    history=[];rescued=0;converged=False
    for iteration in range(max_iter):
        logresp=_gaussian_logpdf(x,means,covariances)+np.log(proportions)
        normalizer=logsumexp(logresp,axis=1)
        objective=float(w@normalizer);history.append(objective)
        if len(history)>1 and abs(history[-1]-history[-2])<=tolerance*(1+abs(history[-2])):
            converged=True
            break
        responsibilities=np.exp(logresp-normalizer[:,None])*w[:,None]
        masses=responsibilities.sum(axis=0)
        for k in range(n_components):
            if masses[k]<1e-10:
                means[k]=x[rng.choice(len(x),p=w)];covariances[k]=global_cov;masses[k]=1e-10;rescued+=1
            else:
                means[k]=(responsibilities[:,k]@x)/masses[k]
                residual=x-means[k]
                covariances[k]=(residual*responsibilities[:,k,None]).T@residual/masses[k]+regularization*np.eye(d)
        proportions=masses/masses.sum()
    final_logresp=_gaussian_logpdf(x,means,covariances)+np.log(proportions)
    final_normalizer=logsumexp(final_logresp,axis=1)
    final_responsibilities=np.exp(final_logresp-final_normalizer[:,None])*w[:,None]
    final_component_ess=final_responsibilities.sum(axis=0)**2/np.sum(final_responsibilities**2,axis=0)
    means=origin+means*scale
    covariances=covariances*scale[None,:,None]*scale[None,None,:]
    return {'weights':proportions,'means':means,'covariances':covariances,
            'metadata':{'seed':int(seed),'n_components':n_components,'training_rows':len(x),
                'training_weight_ess':float(1/np.sum(w*w)),'iterations':len(history),'max_iter':max_iter,
                'tolerance':tolerance,'converged_by_tolerance':converged,'hit_maxiter':not converged and len(history)==max_iter,
                'final_weighted_logpdf_objective_standardized':float(w@final_normalizer),
                'final_component_responsibility_weight_ess':final_component_ess.tolist(),
                'final_component_probabilities':proportions.tolist(),
                'weighted_logpdf_objective_standardized':history,'rescued_empty_components':rescued,
                'standardized_covariance_regularization':regularization,'elapsed_seconds':time.perf_counter()-started}}


@dataclass(frozen=True)
class FrozenProposal:
    component_weights: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    df: float=7.0
    epsilon: float=.05
    metadata: dict|None=None

    def __post_init__(self):
        w=np.array(self.component_weights,dtype=float,copy=True)
        means=np.array(self.means,dtype=float,copy=True);scales=np.array(self.scales,dtype=float,copy=True)
        if (means.ndim!=2 or w.shape!=(len(means),) or scales.shape!=(len(means),means.shape[1],means.shape[1])
                or not all(np.isfinite(v).all() for v in (w,means,scales)) or np.any(w<=0)
                or not 0<self.epsilon<1 or not self.df>2 or not np.isfinite(self.df)
                or not np.isclose(w.sum()+self.epsilon,1,rtol=0,atol=1e-12)):
            raise ValueError('Invalid full-mixture probabilities or Student-t parameters')
        if not np.allclose(scales,np.swapaxes(scales,-1,-2),rtol=1e-12,atol=1e-14):
            raise ValueError('Student-t scale matrices must be symmetric')
        cholesky=np.linalg.cholesky(scales)
        for name,array in [('component_weights',w),('means',means),('scales',scales),('_cholesky',cholesky)]:
            array.setflags(write=False);object.__setattr__(self,name,array)
        object.__setattr__(self,'metadata',json.loads(json.dumps(self.metadata or {})))

    @property
    def dimension(self):return self.means.shape[1]

    def logpdf_z(self,z):
        z=np.asarray(z,dtype=float);one=z.ndim==1;z=np.atleast_2d(z)
        if z.shape[1]!=self.dimension or not np.isfinite(z).all():raise ValueError('Invalid z')
        d=self.dimension;columns=[]
        constant=gammaln((self.df+d)/2)-gammaln(self.df/2)-d/2*np.log(self.df*np.pi)
        for mass,mean,chol in zip(self.component_weights,self.means,self._cholesky):
            whitened=solve_triangular(chol,(z-mean).T,lower=True,check_finite=False).T
            largest=np.max(np.abs(whitened),axis=1)
            safe=np.where(largest==0,1,largest)
            with np.errstate(divide='ignore'):
                logmahal=2*np.log(largest)+np.log(np.sum((whitened/safe[:,None])**2,axis=1))
            logpdf=constant-np.log(np.diag(chol)).sum()-(self.df+d)/2*np.logaddexp(0,logmahal-np.log(self.df))
            columns.append(np.log(mass)+logpdf)
        columns.append(np.log(self.epsilon)+log_prior_z(z))
        result=logsumexp(np.column_stack(columns),axis=1)
        return float(result[0]) if one else result

    def sample_z(self,n,rng,*,return_components=False):
        if not isinstance(n,(int,np.integer)) or n<1:raise ValueError('n must be positive integer')
        categories=rng.choice(len(self.means)+1,size=n,p=np.r_[self.component_weights,self.epsilon])
        z=np.empty((n,self.dimension))
        for k,(mean,chol) in enumerate(zip(self.means,self._cholesky)):
            take=categories==k;count=int(take.sum())
            if count:
                normal=rng.normal(size=(count,self.dimension))@chol.T
                z[take]=mean+normal/np.sqrt(rng.chisquare(self.df,size=count)/self.df)[:,None]
        take=categories==len(self.means);count=int(take.sum())
        if count:
            # Open 52-bit midpoint uniforms: no rejection and no clipping.
            # Floating PRNG discretization applies to all draws; density is
            # the continuous logistic law, as for ordinary Monte Carlo RNGs.
            u=(rng.integers(0,2**52,size=(count,self.dimension),dtype=np.uint64).astype(float)+.5)/2**52
            z[take]=np.log(u)-np.log1p(-u)
        if not np.isfinite(z).all():raise FloatingPointError('Nonfinite proposal draw; no resampling performed')
        return (z,categories) if return_components else z

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        if path.suffix!='.npz':raise ValueError('Frozen proposal must use .npz')
        record={'format':'frozen_defensive_student_mixture_v1','df':self.df,'epsilon':self.epsilon,
                'component_weight_convention':'full probabilities; sum plus epsilon is one',
                'scale_convention':'multivariate Student-t shape S; covariance=df/(df-2)*S',
                'metadata':self.metadata}
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,component_weights=self.component_weights,means=self.means,scales=self.scales,record_json=np.array(json.dumps(record)))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as stored:
            record=json.loads(str(stored['record_json']))
            if record['format']!='frozen_defensive_student_mixture_v1':raise ValueError('Unknown proposal format')
            return cls(stored['component_weights'],stored['means'],stored['scales'],record['df'],record['epsilon'],record['metadata'])


def freeze_student_mixture(fits,block_weights,epsilon=.05,df=7.,inflation=1.2,metadata=None):
    """block_weights are full probabilities, e.g. (.475,.475), not (.5,.5)."""
    block_weights=np.asarray(block_weights,dtype=float)
    if len(fits)!=len(block_weights) or np.any(block_weights<=0) or not np.isclose(block_weights.sum()+epsilon,1,rtol=0,atol=1e-12) or df<=2 or inflation<=0:
        raise ValueError('Invalid full block weights or covariance conversion')
    weights=np.concatenate([mass*fit['weights'] for mass,fit in zip(block_weights,fits)])
    means=np.concatenate([fit['means'] for fit in fits])
    scales=np.concatenate([fit['covariances'] for fit in fits])*inflation**2*(df-2)/df
    meta={'training_blocks':[fit['metadata'] for fit in fits],'block_full_probabilities':block_weights.tolist(),
          'inflation_is_standard_deviation_multiplier':inflation,'training_evaluation_separate':True,**(metadata or {})}
    return FrozenProposal(weights,means,scales,df,epsilon,meta)


def summarize_logweights(logw,observables=None,saturated=None):
    """Unbiased mean-weight estimate and iid MC SE; ratio estimates are biased."""
    logw=np.asarray(logw,dtype=float);n=len(logw)
    if n<2 or np.isnan(logw).any() or np.isposinf(logw).any() or not np.isfinite(logw).any():raise ValueError('Need >=2 valid rows and positive total importance weight')
    largest=float(np.max(logw));scaled=np.exp(logw-largest)
    mean=float(scaled.mean());variance=float(scaled.var(ddof=1))
    logz=largest+np.log(mean)
    logse=largest+.5*np.log(variance/n) if variance>0 else -np.inf
    weights=scaled/scaled.sum();ess=float(1/np.sum(weights**2))
    result={'N':n,'log_evidence':float(logz),'log_evidence_standard_error':float(logse),
        'relative_evidence_standard_error':float(np.sqrt(variance/n)/mean),'weight_ess':ess,
        'maximum_normalized_weight':float(weights.max()),'zero_weight_rows':int(np.isneginf(logw).sum()),
        'all_rows_retained':True,'weights_clipped':False,'standard_error_assumption':'independent fresh draws from the frozen full proposal; finite second moment; empirical SE cannot detect unseen rare weights',
        'posterior_ratio_unbiased':False,'observables':{}}
    if observables:
        for name,values in observables.items():
            values=np.asarray(values,dtype=float)
            if values.shape!=(n,) or not np.isfinite(values).all():raise ValueError('Invalid observable rows')
            estimate=float(weights@values)
            ratio_se=float(np.sqrt(n/(n-1)*np.sum((weights*(values-estimate))**2)))
            numerator=scaled*values
            signed_mean=float(numerator.mean());signed_se=float(numerator.std(ddof=1)/np.sqrt(n))
            result['observables'][name]={'posterior_mean':estimate,'ratio_delta_standard_error':ratio_se,
                'unnormalized_mean_scaled_by_exp_max_logweight':signed_mean,
                'unnormalized_standard_error_scaled_by_exp_max_logweight':signed_se,
                'max_logweight_scale':largest}
    if saturated is not None:
        saturated=np.asarray(saturated,dtype=bool)
        result['saturated_rows']=int(saturated.sum())
        result['saturated_normalized_weight_mass']=float(weights[saturated].sum())
        result['saturated_log_evidence_contribution']=float(logsumexp(logw[saturated])-np.log(n)) if saturated.any() else -np.inf
    return result,weights


def evaluate_u(log_likelihood_u,proposal,n,seed,*,allow_closed_unit=False,observable_functions=None):
    """Single-row callback; every fresh iid draw is retained, including L=0.

    allow_closed_unit explicitly uses the target's continuous extension at
    natural float64 expit saturation. z/logp/logq are never clipped. Strict
    mode raises on saturation before evaluating any target callback.
    """
    started=time.perf_counter();z,categories=proposal.sample_z(n,np.random.default_rng(seed),return_components=True)
    logu,log1mu=unit_logs(z);u=expit(z);saturated=np.any((u==0)|(u==1),axis=1)
    if saturated.any() and not allow_closed_unit:raise FloatingPointError('expit saturated; use an explicitly supported continuous closed-unit target, not clipping or resampling')
    logp=log_prior_z(z);logq=proposal.logpdf_z(z)
    logl=np.array([log_likelihood_u(row) for row in u],dtype=float)
    if np.isnan(logl).any() or np.isposinf(logl).any():raise ValueError('Target returned NaN or +inf; no weights dropped')
    logw=logl+logp-logq
    observables={name:np.array([func(row) for row in u]) for name,func in (observable_functions or {}).items()}
    diagnostics,weights=summarize_logweights(logw,observables,saturated)
    diagnostics.update(seed=int(seed),elapsed_seconds=time.perf_counter()-started,allow_closed_unit=bool(allow_closed_unit),
        evaluation_target='provided closed-unit continuous extension' if allow_closed_unit else 'strict open-unit callback',
        likelihood_calls=n,training_rows_used_as_evaluation=False)
    return {'z':z,'u':u,'logu':logu,'log1mu':log1mu,'logp_z':logp,'logq_z':logq,'logl':logl,'logw':logw,
            'weights':weights,'component_id':categories,'saturated':saturated,'observables':observables,'diagnostics':diagnostics}


def combine_strata(evaluations,prior_masses):
    """Each row contributes p_s*w_si/N_s, including all zero-weight rows."""
    masses=np.asarray(prior_masses,dtype=float)
    if masses.shape!=(len(evaluations),) or np.any(masses<=0) or not np.isclose(masses.sum(),1,rtol=0,atol=1e-12):raise ValueError('Invalid complete prior partition')
    logrows=np.concatenate([e['logw']+np.log(m)-np.log(len(e['logw'])) for e,m in zip(evaluations,masses)])
    ids=np.concatenate([np.full(len(e['logw']),j,dtype=int) for j,e in enumerate(evaluations)])
    logz=float(logsumexp(logrows));weights=np.exp(logrows-logz)
    summaries=[summarize_logweights(e['logw'])[0] for e in evaluations]
    logse=.5*logsumexp([2*(np.log(m)+s['log_evidence_standard_error']) for m,s in zip(masses,summaries)])
    stratum_mass=np.array([weights[ids==j].sum() for j in range(len(masses))])
    observable_names=set(evaluations[0].get('observables',{}))
    if any(set(e.get('observables',{}))!=observable_names for e in evaluations):raise ValueError('Strata must use the same observable columns')
    observable_rows={};observable_summaries={}
    for name in observable_names:
        values=np.concatenate([e['observables'][name] for e in evaluations])
        if values.shape!=weights.shape or not np.isfinite(values).all():raise ValueError('Invalid merged observable rows')
        estimate=float(weights@values);variance=0.
        for j in range(len(masses)):
            row=weights[ids==j]*(values[ids==j]-estimate);n=len(row)
            variance+=n/(n-1)*np.sum((row-row.mean())**2)
        observable_rows[name]=values
        observable_summaries[name]={'posterior_mean':estimate,'ratio_delta_standard_error':float(np.sqrt(variance)),
            'ratio_standard_error_method':'sum of independent stratum variances; full N_s includes zero weights; empirical estimate'}
    return {'log_evidence':logz,'log_evidence_standard_error':float(logse),'weights':weights,
            'original_prior_logweight':logrows,'stratum_id':ids,'stratum_posterior_mass':stratum_mass,
            'Ns':[len(e['logw']) for e in evaluations],'all_strata_retained':True,'weight_rule':'log p_s + log w_si - log N_s',
            'observables':observable_rows,'observable_summaries':observable_summaries}
