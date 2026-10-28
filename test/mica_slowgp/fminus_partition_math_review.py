"""Mathematical toy only: no GP, scientific sampler, model edit, or driver."""
from pathlib import Path
from datetime import datetime,timezone
import json, math
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.integrate import quad

EDGES=np.array([0.,.001,.03,1.]);A=-3.
OUT=Path('/home/leonhard/sci001/output/mica_slowgp/fminus_partition_math_review.json')

def fminus(mu0,p):
    return (p['A0']*ndtr(-np.asarray(mu0)/p['w0'])+p['A1']*ndtr(-p['mu1']/p['w1']))/(p['A0']+p['A1'])

def intervals(p):
    a=A;b=p['mu1'];delta=b-a
    if delta<0:raise ValueError('Outside ordered center domain')
    if delta==0:
        # A closed-boundary convention only; no positive-dimensional inverse exists.
        weights=np.zeros(3);j=min(2,max(0,int(np.searchsorted(EDGES,float(fminus(a,p)),side='right')-1)));weights[j]=1.
        return np.full((3,2),a),weights,True
    lowf=float(fminus(b,p));highf=float(fminus(a,p));base=p['A1']*ndtr(-b/p['w1']);total=p['A0']+p['A1']
    cuts=[]
    for edge in EDGES:
        if edge<=lowf:cut=b
        elif edge>=highf:cut=a
        else:
            probability=(edge*total-base)/p['A0']
            # Extended analytic inverse: values outside (0,1) map to +/- infinity.
            cut=b if probability<=0 else a if probability>=1 else -p['w0']*ndtri(probability)
            cut=min(b,max(a,float(cut)))
        cuts.append(cut)
    cuts=np.array(cuts)
    bounds=np.column_stack((cuts[1:],cuts[:-1]))
    lengths=bounds[:,1]-bounds[:,0]
    if np.any(lengths<0):raise ArithmeticError('Nonmonotone partition')
    return bounds,lengths/delta,False

def primitive(x,w):
    return x*ndtr(-x/w)-w*np.exp(-.5*(x/w)**2)/math.sqrt(2*math.pi)

def analytic_conditional_mean(p):
    b=p['mu1'];d=b-A
    return (p['A0']*(primitive(b,p['w0'])-primitive(A,p['w0']))/d+p['A1']*ndtr(-b/p['w1']))/(p['A0']+p['A1'])

rng=np.random.default_rng(932041)
centers=[(-3.,5.),(5.,20.),(20.,28.)]
rhos=[((hi-A)**2-(lo-A)**2)/(28-A)**2 for lo,hi in centers]
checks=[];worst_mass=0.;worst_inverse=0.;inactive=[0,0,0];outer_sums=[]
for s,(lo,hi) in enumerate(centers):
    total=0.
    for _ in range(2000):
        u=rng.random(5)
        p=dict(mu1=A+np.sqrt((lo-A)**2+u[0]*((hi-A)**2-(lo-A)**2)),
            A0=np.exp(np.log(.02)+u[1]*np.log(2.5/.02)),A1=np.exp(np.log(.02)+u[2]*np.log(2.5/.02)),
            w0=np.exp(np.log(.1)+u[3]*np.log(12/.1)),w1=np.exp(np.log(.1)+u[4]*np.log(12/.1)))
        bounds,J,deg=intervals(p);assert not deg
        worst_mass=max(worst_mass,abs(J.sum()-1.));total+=J.sum()
        assert np.all(bounds>=A-1e-14) and np.all(bounds<=p['mu1']+1e-14)
        for j,((left,right),factor) in enumerate(zip(bounds,J)):
            if factor==0:inactive[j]+=1;continue
            for unit in (.17,.53,.89):
                value=float(fminus(left+unit*(right-left),p))
                assert EDGES[j]-1e-12<=value<=EDGES[j+1]+1e-12
        old_mu=A+rng.random()*(p['mu1']-A);value=float(fminus(old_mu,p))
        j=min(2,max(0,int(np.searchsorted(EDGES,value,side='right')-1)))
        left,right=bounds[j];assert right>left
        inverse=(old_mu-left)/(right-left);assert -1e-12<=inverse<=1+1e-12
        recovered=left+inverse*(right-left);worst_inverse=max(worst_inverse,abs(recovered-old_mu))
    outer_sums.append(total/2000)
assert worst_mass<1e-12 and worst_inverse<1e-11
checks.append(dict(name='6000_random_original_prior_outer_points',passed=True,
    maximum_sum_J_error=worst_mass,maximum_inverse_roundtrip_error=worst_inverse,
    inactive_bin_counts=inactive,full_prior_constant_L_integral=sum(r*x for r,x in zip(rhos,outer_sums)),
    note='Random outer points verify a pointwise partition identity; they are not scientific posterior draws.'))

cases=[dict(A0=.7058303474341876,A1=.6015591452597996,mu1=8.507394269372075,w0=.10095940208923783,w1=5.597562542511835),
 dict(A0=.9820171356519667,A1=.6710635746977218,mu1=6.297793925824859,w0=.4897432936092447,w1=6.734561815676229),
 dict(A0=.8233736087744109,A1=1.833399468418726,mu1=6.129197439613646,w0=.29098638059482695,w1=7.854827141507069),
 dict(A0=1.,A1=.5,mu1=8.,w0=1.,w1=1.5),
 dict(A0=.02,A1=2.5,mu1=28.,w0=.1,w1=.1),
 dict(A0=2.5,A1=.02,mu1=20.,w0=12.,w1=12.),
 dict(A0=.2,A1=.7,mu1=-2.,w0=.5,w1=.7),
 dict(A0=.2,A1=.7,mu1=0.,w0=2.,w1=5.)]
case_reports=[]
for p in cases:
    bounds,J,deg=intervals(p);num=0.;pieces=[]
    for (left,right),factor in zip(bounds,J):
        contribution=0. if factor==0 else factor*quad(lambda u:float(fminus(left+u*(right-left),p)),0.,1.,epsabs=1e-12,epsrel=1e-11)[0]
        pieces.append(contribution);num+=contribution
    exact=float(analytic_conditional_mean(p));error=abs(exact-num)
    assert abs(J.sum()-1.)<1e-12 and error<2e-10
    case_reports.append(dict(parameters=p,bounds=bounds.tolist(),J=J.tolist(),
        constant_L_integral=float(J.sum()),analytic_f_mean=exact,partitioned_quadrature_f_mean=num,
        absolute_error=error,f_numerator_integral_by_bin=pieces))
checks.append(dict(name='analytic_f_integral_vs_independent_adaptive_quadrature',passed=True,cases=case_reports))

edge_reports=[]
for b in [A,np.nextafter(A,np.inf),A+1e-12]:
    for width in [.1,12.]:
        p=dict(A0=.7,A1=.3,mu1=b,w0=width,w1=width)
        bounds,J,deg=intervals(p);assert np.isfinite(bounds).all() and np.isfinite(J).all() and abs(J.sum()-1)<1e-12
        edge_reports.append(dict(mu1=b,width=width,degenerate=deg,J=J.tolist()))
checks.append(dict(name='degenerate_center_and_near_endpoint_partition',passed=True,cases=edge_reports,
    qualification='At mu1=a the cube face has zero original measure; the defined closed extension assigns J=1 to its containing f bin. No inverse or density claim is made on this degenerate face.'))

p=cases[3];bounds,J,_=intervals(p)
likelihood=lambda mu:math.exp(-.5*((mu-2.2)/.4)**2)
original=quad(lambda mu:likelihood(mu)/(p['mu1']-A),A,p['mu1'],epsabs=1e-12)[0]
partitioned=sum(factor*quad(lambda u:likelihood(left+u*(right-left)),0,1,epsabs=1e-12)[0]
                for (left,right),factor in zip(bounds,J) if factor>0)
assert abs(original-partitioned)<1e-11
checks.append(dict(name='nonconstant_synthetic_likelihood_sum_identity',passed=True,
    original_integral=original,partitioned_integral=partitioned,
    GP_likelihood_calls=0,note='A one-dimensional synthetic Gaussian function only; no physical model evaluation.'))

p=cases[0];mu=2.218683910176357
first=p['A0']*ndtr(-mu/p['w0'])/(p['A0']+p['A1']);second=p['A1']*ndtr(-p['mu1']/p['w1'])/(p['A0']+p['A1'])
derivative=-p['A0']/(p['A0']+p['A1'])/p['w0']*math.exp(-.5*(mu/p['w0'])**2)/math.sqrt(2*math.pi)
report=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Mathematical partition and inexpensive toys only; original model/prior unchanged; no scientific driver or GP call.',
    edges=EDGES.tolist(),original_mu0_domain=[A,'mu1'],center_strata=centers,unchanged_center_prior_masses=rhos,
    identities=dict(derivative='df/dmu0 = -A0*phi(mu0/w0)/((A0+A1)*w0) < 0',
        root='mu0=-w0*ndtri((b*(A0+A1)-A1*Phi(-mu1/w1))/A0), extended to +/-infinity outside [0,1], then intersect original interval',
        local_target='L(theta(u,j))*J_j, J_j=length_j/(mu1-a); sum_j integral L*J_j recovers original conditional prior integral',
        combination='For each original center s: rho_s*sum_j Z_sj. No additional 1/number_of_f_bins or assumed global rho_j.',
        inverse_training='Classify each existing physical sample once; u3_new=(mu0-lower_j)/length_j. Original posterior row weights stay unchanged and may be renormalized within the chart for proposal training; do not multiply J again.'),
    checks=checks,checks_passed=len(checks),checks_failed=0,
    known_pilot_top_variance_point=dict(parameters=p,mu0=mu,f_first_component=float(first),f_second_component=float(second),
        derivative_mu0=float(derivative),partition=case_reports[0],
        interpretation='Here f is almost entirely the broad second component and is essentially flat in mu0 over the observed narrow-first-kernel region; changing mu0 coordinates does not automatically cover rare other-parameter combinations.'),
    conclusion='Mathematically valid full-prior decomposition with correct local Jacobian, but no evidence yet that it improves efficiency or solves the scientific tail problem.',
    remaining_issues=['Each f chart can be locally empty: evaluate no scientific L when J=0, retain a zero contribution and the declared quadrature accounting.',
        'The feasible set and clipped inverse boundaries depend on other coordinates; zero-support regions and nonsmooth transitions can impair proposals or QMC smoothness.',
        'f can be numerically flat in mu0 although its analytic derivative is strictly negative; classify/use the physical mu0 inverse, not an inversion of rounded observed f.',
        'Original posterior samples may scarcely cover an f chart or its important joint subregions; the inverse transform does not create missing information.',
        'The f bins cover all f values but do not guarantee adequate coverage of high-likelihood narrow regions in A0,A1,w0,w1,mu1 or GP parameters.',
        'No variance-reduction theorem is asserted for arbitrary proposals, allocations, ratios or this finite budget.'],
    scientific_likelihood_calls=0,scientific_driver_written=False)
assert not OUT.exists();OUT.write_text(json.dumps(report,indent=2))
print(json.dumps(dict(output=str(OUT),checks_passed=len(checks),maximum_partition_error=worst_mass,
    maximum_inverse_error=worst_inverse,top_point_first_component=float(first),top_point_second_component=float(second),
    top_point_derivative=float(derivative),top_point_J=case_reports[0]['J'],
    maximum_f_integral_error=max(x['absolute_error'] for x in case_reports)),indent=2))
