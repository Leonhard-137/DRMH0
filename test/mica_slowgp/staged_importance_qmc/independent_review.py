"""Independent algebra/numerical review; synthetic callbacks only, no GP imports."""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="1"
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import numpy as np
from scipy.special import ndtr,logsumexp
from scipy.stats import chi2,multivariate_t,logistic
import qmc_importance as q
from importance import FrozenProposal,logit_open

ROOT=Path(__file__).resolve().parent
results=[]
def check(name,fn):
    detail=fn()
    results.append(dict(name=name,passed=True,detail=detail))

def make_rows(Z,Y,R=4):
    logweight=[];ids=[];f=[]
    for r,(z,y) in enumerate(zip(Z,Y)):
        for fraction in (.2,.8,0.):
            logweight.append(math.log(z*fraction/R) if fraction else -np.inf)
            ids.append(r);f.append(y/z)
    return np.array(logweight),np.array(ids),np.array(f)

Z=np.array([1.,2.,3.,4.]);Y=np.array([1.,2.,6.,8.]);R=4
lw,ids,f=make_rows(Z,Y)

def hand_covariance():
    s,w=q.summarize_scrambles(lw,ids,R,dict(f=f,g=2*f+1))
    expected_ratio=17/10;expected_var=47/750
    np.testing.assert_allclose(math.exp(s['log_evidence']),2.5,atol=1e-14)
    np.testing.assert_allclose(math.exp(s['log_evidence_standard_error']),math.sqrt(5/12),atol=1e-14)
    np.testing.assert_allclose(s['observables']['f']['posterior_mean'],expected_ratio,atol=1e-14)
    np.testing.assert_allclose(s['observables']['f']['ratio_delta_standard_error']**2,expected_var,atol=1e-14)
    np.testing.assert_allclose(s['observable_ratio_covariance'],expected_var*np.array([[1.,2.],[2.,4.]]),atol=1e-14)
    np.testing.assert_allclose(w.sum(),1,atol=1e-14)
    assert s['zero_weight_rows_retained']==4 and s['iid_row_standard_error_used'] is False
    return dict(evidence=2.5,evidence_variance=5/12,ratio=expected_ratio,ratio_variance=expected_var,
        expected_covariance_identity='Cov(Zmean,Ymean)=25/24; Var(Ymean)=131/48; includes ratio numerator-denominator covariance')
check('hand_calculated_scramble_covariance_and_ratio',hand_covariance)

def split_rows():
    baseline,_=q.summarize_scrambles(lw,ids,R,dict(f=f))
    copy=17
    s,_=q.summarize_scrambles(np.repeat(lw-math.log(copy),copy),np.repeat(ids,copy),R,dict(f=np.repeat(f,copy)))
    for field in ['log_evidence','relative_evidence_standard_error']:
        np.testing.assert_allclose(s[field],baseline[field],rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(s['observables']['f']['ratio_delta_standard_error'],baseline['observables']['f']['ratio_delta_standard_error'],rtol=1e-13,atol=1e-14)
    return dict(row_multiplier=copy,MCSE_unchanged=True,meaning='Rows inside a scramble do not become independent replicates when split.')
check('splitting_rows_does_not_reduce_MCSE',split_rows)

proposal=FrozenProposal(np.array([.65,.30]),np.array([[.3,-.2],[-.5,.7]]),
    np.array([[[4.,1.2],[1.2,2.25]],[[.4,.1],[.1,.8]]]),df=7.,epsilon=.05)

def student_mapping():
    L21=.6;L22=math.sqrt(2.25-L21**2)
    u=np.array([[ndtr(1.),ndtr(-1.),chi2.cdf(7.,7.)]])
    expected=np.array([.3+2.,-.2+L21-L22])
    np.testing.assert_allclose(q.component_map(u,proposal,0)[0],expected,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(q.component_map(np.array([[.2,.8]]),proposal,2),[[-math.log(4),math.log(4)]],atol=1e-14)
    return dict(expected_point=expected.tolist(),scale_matrix_is_t_shape=True,
        mapping='z=mu+L*N/sqrt(chi2(df)/df); Cov(z)=df/(df-2)*L L^T')
check('Student_t_df_shape_mapping_and_defensive_logistic',student_mapping)

def full_density():
    points=np.array([[0.,0.],[-3.,2.],[.3,-.7],[8.,-10.]])
    terms=[math.log(a)+multivariate_t.logpdf(points,loc=m,shape=S,df=proposal.df) for a,m,S in zip(proposal.component_weights,proposal.means,proposal.scales)]
    terms.append(math.log(proposal.epsilon)+logistic.logpdf(points).sum(axis=1))
    expected=logsumexp(np.column_stack(terms),axis=1)
    delta=np.max(np.abs(proposal.logpdf_z(points)-expected))
    assert delta<1e-12
    return dict(max_logdensity_difference=float(delta),denominator='sum of all full alpha_k q_k including defensive prior')
check('full_mixture_density_against_scipy_distributions',full_density)

small=FrozenProposal(np.array([.6,.35]),np.array([[.3],[-.4]]),np.array([[[.16]],[[.25]]]),df=7.,epsilon=.05)
def exact_component_integral():
    # Target density is exactly the FULL proposal; every importance ratio is one.
    def synthetic_loglike(u):
        z=logit_open(u)
        return small.logpdf_z(z)-float(logistic.logpdf(z).sum())
    e=q.evaluate_u(synthetic_loglike,small,seed=91239,stratum_id=0,target_N=64,scrambles=4,
        observable_functions={'constant':lambda u:3.})
    assert np.max(np.abs(e['logimportance']))<1e-11
    np.testing.assert_allclose(math.exp(e['diagnostics']['log_evidence']),1.,atol=1e-12)
    np.testing.assert_allclose(e['diagnostics']['observables']['constant']['posterior_mean'],3.,atol=1e-12)
    assert e['diagnostics']['relative_evidence_standard_error']<1e-12
    for r in range(4):
        take=e['scramble_id']==r
        np.testing.assert_allclose(np.exp(e['log_quadrature_factor'][take]).sum(),.25,atol=1e-14)
    return dict(rows=len(e['z']),evidence=math.exp(e['diagnostics']['log_evidence']),
        standard_error=e['diagnostics']['relative_evidence_standard_error'],scientific_GP_evaluations=0)
check('alpha_over_N_over_R_and_full_q_cancel_exactly',exact_component_integral)

def endpoint_and_seed():
    failed=[]
    for endpoint in (0.,1.):
        try:q.component_map(np.array([[endpoint,.3,.7]]),proposal,0)
        except FloatingPointError:failed.append(endpoint)
    assert failed==[0.,1.]
    a=q.sample_components(small,seed=91311,stratum_id=0,target_N=32,scrambles=4)
    b=q.sample_components(small,seed=91311,stratum_id=0,target_N=32,scrambles=4)
    c=q.sample_components(small,seed=91311,stratum_id=1,target_N=32,scrambles=4)
    assert np.array_equal(a['z'],b['z']) and not np.array_equal(a['z'],c['z'])
    keys=[tuple(s['spawn_key']) for s in a['seed_hierarchy']]
    assert len(set(keys))==len(keys)
    return dict(endpoints_rejected=failed,reproducible_same_root=True,different_stratum_not_reused=True,
        unique_component_scramble_keys=len(keys),qualification='PRNG stream construction and reproducibility checked, not a statistical proof of independence.')
check('fail_fast_endpoints_and_nonreused_seed_hierarchy',endpoint_and_seed)

def combine_known_vectors():
    Za=np.array([1.,2.,3.,4.]);Ya=np.array([1.,2.,6.,8.])
    Zb=np.array([4.,3.,2.,1.]);Yb=np.array([2.,1.,1.,.5]);rho=np.array([.25,.75])
    es=[]
    for j,(z,y) in enumerate(((Za,Ya),(Zb,Yb))):
        l,ii,ff=make_rows(z,y)
        es.append(dict(logweight=l,scramble_id=ii,scrambles=R,seed=119,stratum_id=j,
            observables={'f':ff},saturated=np.zeros(len(l),dtype=bool),diagnostics={'likelihood_calls':len(l)}))
    joined=q.combine_strata(es,rho);ss=joined['diagnostics']
    zr=rho[0]*Za+rho[1]*Zb;yr=rho[0]*Ya+rho[1]*Yb
    zmean=sum(zr)/R;ymean=sum(yr)/R;ratio=ymean/zmean
    expected_var=sum((float(y)-ratio*float(z))**2 for z,y in zip(zr,yr))/(R*(R-1)*zmean*zmean)
    np.testing.assert_allclose(math.exp(ss['log_evidence']),zmean,atol=1e-14)
    np.testing.assert_allclose(ss['observables']['f']['posterior_mean'],ratio,atol=1e-14)
    np.testing.assert_allclose(ss['observables']['f']['ratio_delta_standard_error']**2,expected_var,atol=1e-14)
    np.testing.assert_allclose(ss['observables']['__stratum_mass_0']['posterior_mean'],rho[0]*Za.mean()/zmean,atol=1e-14)
    return dict(evidence=zmean,ratio=ratio,ratio_variance=expected_var,
        meaning='rho inserted once before corresponding independent r vectors are summed; all layers retained')
check('rho_combination_of_hand_specified_scramble_vectors',combine_known_vectors)

budgets=[]
for j in range(3):
    path=Path('/home/leonhard/sci001/test/mica_slowgp/staged_importance_regularized/artifacts/regularized')/f'stratum_{j:02d}.npz'
    prop=FrozenProposal.load(path)
    alpha,counts,powers=q.component_budget(prop,target_N=2048)
    assert np.all(counts>=8) and np.all(counts>=2048*alpha)
    for count,power in zip(counts,powers):assert int(count)==1<<int(power)
    budgets.append(dict(stratum_id=j,proposal_path=str(path),component_probabilities=alpha.tolist(),
        component_counts_per_scramble=counts.tolist(),rows_per_scramble=int(counts.sum()),
        scrambles=8,total_rows=int(counts.sum()*8)))
report=dict(created_utc=datetime.now(timezone.utc).isoformat(),reviewer_scope='Independent read-only mathematical review and synthetic tests; no GP likelihood evaluated.',
    source_files={str(ROOT/name):hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('qmc_importance.py','importance.py')},
    tests=results,tests_passed=len(results),tests_failed=0,actual_budgets=budgets,
    total_scientific_rows_planned=sum(b['total_rows'] for b in budgets),
    blocking_findings=[],conclusion='Mathematical implementation is suitable for one new independent scientific pilot, subject to recording the frozen proposal/settings and checking actual wrapper outputs. This is not a formal stability-gate pass.',
    cautions=['Eight independent scramble integrals provide only seven covariance degrees of freedom; reported randomization error can be noisy.',
        'Ratio estimates are self-normalized and delta-method errors are approximate; missing rare contributions can make apparent precision misleading.',
        'Target_N is nominal allocation, not actual row budget; power-of-two rounding and minimum eight per component increase total calls.',
        'Natural expit saturation must use the explicitly recorded policy; raw Sobol 0/1 endpoints fail without reseeding or replacement.',
        'Sample rows within each scramble are not iid; Kish is descriptive only.'],
    scientific_likelihood_calls_in_review=0,source_edited=False)
output=ROOT/'independent_review.json';assert not output.exists();output.write_text(json.dumps(report,indent=2))
print(json.dumps({'tests_passed':len(results),'actual_budgets':budgets,'total_rows':report['total_scientific_rows_planned'],'output':str(output)},indent=2))
