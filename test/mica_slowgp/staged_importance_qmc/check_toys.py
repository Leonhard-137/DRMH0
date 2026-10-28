"""Fixed analytic and mapping checks; no scientific likelihood calls."""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
from scipy.special import ndtr, ndtri, expit
from scipy.stats import chi2, multivariate_t
from scipy.special import logsumexp

from importance import FrozenProposal, json_safe, log_prior_z
from qmc_importance import (SCHEME, DEFAULT_SETTINGS, component_budget, component_map,
    sample_components, summarize_scrambles, evaluate_u, combine_strata)

HERE = Path(__file__).resolve().parent
OUT = HERE / 'artifacts'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toy_proposal():
    return FrozenProposal(np.array([.62, .33]),
        np.array([[-1.0986122886681098, 0.], [1.0986122886681098, 0.]]),
        np.array([[[.6, .1], [.1, 3.2]], [[.8, -.15], [-.15, 3.5]]]), 7., .05,
        {'purpose': 'fixed analytic toy only'})


def bimodal_logl(u):
    x = u[0]
    return float(np.logaddexp(np.log(.65 / (.05 * np.sqrt(2*np.pi))) - .5*((x-.25)/.05)**2,
                             np.log(.35 / (.07 * np.sqrt(2*np.pi))) - .5*((x-.75)/.07)**2))


def main():
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError('Toy artifacts already exist; no overwrite')
    OUT.mkdir()
    proposal = toy_proposal()
    checks = {}
    probabilities, counts, powers = component_budget(proposal)
    checks['unequal_counts_are_fixed_powers_of_two'] = bool(np.array_equal(counts, [2048, 1024, 128]))
    uniform = np.array([[.1, .2, .3], [.25, .5, .75], [.9, .8, .7]])
    mapped = component_map(uniform, proposal, 0)
    expected = (proposal.means[0] + (ndtri(uniform[:, :2]) @ np.linalg.cholesky(proposal.scales[0]).T)
                * np.sqrt(proposal.df / chi2.ppf(uniform[:, 2], proposal.df))[:, None])
    mapping_error = float(np.max(np.abs(mapped - expected)))
    checks['student_inverse_normal_chisquare_mapping'] = mapping_error < 1e-13
    defensive = component_map(uniform[:, :2], proposal, 2)
    checks['defensive_logit_roundtrip'] = bool(np.max(np.abs(expit(defensive) - uniform[:, :2])) < 1e-15)
    columns = [np.log(w) + multivariate_t.logpdf(mapped, loc=m, shape=s, df=proposal.df)
               for w, m, s in zip(proposal.component_weights, proposal.means, proposal.scales)]
    columns.append(np.log(proposal.epsilon) + log_prior_z(mapped))
    density_error = float(np.max(np.abs(logsumexp(np.column_stack(columns), axis=1) - proposal.logpdf_z(mapped))))
    checks['full_q_density_matches_scipy'] = density_error < 1e-12
    for endpoint in (0., 1.):
        bad = uniform.copy(); bad[0, 0] = endpoint
        try:
            component_map(bad, proposal, 0)
        except FloatingPointError:
            checks[f'endpoint_{int(endpoint)}_explicitly_fails'] = True
        else:
            checks[f'endpoint_{int(endpoint)}_explicitly_fails'] = False
    # Exact constant integrand h=L*pi/q=1: unequal N_k must not alter any
    # component mass. This algebraic test is separate from L=1 below.
    exact = sample_components(proposal, seed=890101)
    exact_summary, exact_weights = summarize_scrambles(exact['log_quadrature_factor'],
        exact['scramble_id'], 8)
    sums = np.array([exact_weights[exact['component_id'] == k].sum() for k in range(3)])
    checks['constant_integrand_evidence_exact_with_unequal_Nk'] = abs(exact_summary['log_evidence']) < 1e-14
    checks['component_masses_preserved_with_unequal_Nk'] = bool(np.max(np.abs(sums-probabilities)) < 1e-14)
    second = sample_components(proposal, seed=890101)
    third = sample_components(proposal, seed=890102)
    checks['same_seed_deterministic'] = bool(np.array_equal(exact['z'], second['z']))
    checks['independent_seed_changes_points'] = not np.array_equal(exact['z'], third['z'])
    checks['defensive_component_included'] = bool(np.sum(exact['component_id'] == 2) == 8*128)

    # Constant likelihood under the unit-cube prior has Z=1 and E[u0]=1/2.
    constant = evaluate_u(lambda u: 0., proposal, seed=891101,
        allow_closed_unit=True, observable_functions={'u0': lambda u: u[0]})
    cs = constant['diagnostics']; zc = np.exp(cs['log_evidence'])
    sec = np.exp(cs['log_evidence_standard_error'])
    checks['constant_likelihood_within_6_scramble_SE'] = abs(zc-1) <= 6*sec + 1e-12
    cu = cs['observables']['u0']
    checks['constant_likelihood_u_mean_within_6_SE'] = abs(cu['posterior_mean']-.5) <= 6*cu['ratio_delta_standard_error'] + 1e-12

    # True evidence and tail probability from normal CDF differences.
    ztrue = float(.65*(ndtr((1-.25)/.05)-ndtr(-.25/.05))
                  + .35*(ndtr((1-.75)/.07)-ndtr(-.75/.07)))
    tail_numerator = float(.65*(ndtr((1-.25)/.05)-ndtr((.5-.25)/.05))
                           + .35*(ndtr((1-.75)/.07)-ndtr((.5-.75)/.07)))
    tail_true = tail_numerator/ztrue
    experiments = []
    for j in range(12):
        evaluation = evaluate_u(bimodal_logl, proposal, seed=892000+j, allow_closed_unit=True,
                                observable_functions={'tail': lambda u: float(u[0]>.5)})
        report = evaluation['diagnostics']; tail = report['observables']['tail']
        experiments.append({'seed': 892000+j, 'evidence': float(np.exp(report['log_evidence'])),
            'evidence_SE': float(np.exp(report['log_evidence_standard_error'])),
            'tail': tail['posterior_mean'], 'tail_SE': tail['ratio_delta_standard_error'],
            'scrambles': 8, 'likelihood_calls': report['likelihood_calls'],
            'saturated_rows': report['saturated_rows']})
        if j == 0:
            np.savez_compressed(OUT/'bimodal_primary_rows.npz', **{k: evaluation[k] for k in
                ('z','u','component_id','scramble_id','logp_z','logq_z','logl','logimportance',
                 'log_quadrature_factor','logweight','weights','saturated')},
                tail=evaluation['observables']['tail'])
            primary_summary = report
    errors = {}
    for name, truth, se_key in (('evidence', ztrue, 'evidence_SE'), ('tail', tail_true, 'tail_SE')):
        estimates = np.array([e[name] for e in experiments])
        ses = np.array([e[se_key] for e in experiments])
        expected_var = float(np.mean(ses**2)); observed_var = float(estimates.var(ddof=1))
        pooled_se = float(np.sqrt(np.sum(ses**2))/len(experiments))
        errors[name] = {'truth': truth, 'experiment_mean': float(estimates.mean()),
            'mean_reported_SE': float(ses.mean()), 'outer_SD': float(np.sqrt(observed_var)),
            'outer_variance_over_mean_reported_variance': observed_var/expected_var,
            'count_within_3_reported_SE': int(np.sum(np.abs(estimates-truth) <= 3*ses)),
            'experiments': len(experiments), 'pooled_SE': pooled_se}
        checks[f'{name}_pooled_bias_within_6_SE'] = abs(estimates.mean()-truth) <= 6*pooled_se + 1e-12
    # No acceptance gate based on outer variance ratio is introduced: 12 tests
    # only describe the finite-scramble SE calibration.

    # Exact complete-strata algebra, with unequal component budgets and rho.
    fake_children = []
    rho = np.array([64,465,432])/961
    constants = np.array([.5, 2., 4.])
    observable = np.array([.1, .4, .8])
    for i in range(3):
        child = sample_components(proposal, seed=893101, stratum_id=i)
        n = len(child['z'])
        child.update(logweight=child['log_quadrature_factor']+np.log(constants[i]),
                     saturated=np.zeros(n,dtype=bool), observables={'f':np.full(n,observable[i])},
                     diagnostics={'likelihood_calls':0})
        fake_children.append(child)
    combined = combine_strata(fake_children,rho)
    z_combined = float(rho@constants); ratio_combined = float((rho*constants)@observable/z_combined)
    checks['complete_strata_evidence_exact'] = abs(combined['diagnostics']['log_evidence']-np.log(z_combined))<1e-14
    checks['complete_strata_ratio_exact'] = abs(combined['diagnostics']['observables']['f']['posterior_mean']-ratio_combined)<1e-14
    checks['complete_strata_row_weights_sum_one'] = abs(combined['weights'].sum()-1)<1e-14
    checks['row_logweight_identity'] = bool(np.allclose(evaluation['logweight'], evaluation['logl']+
        evaluation['logp_z']-evaluation['logq_z']+evaluation['log_quadrature_factor'],rtol=0,atol=1e-14))
    checks['no_iid_row_MCSE'] = all(not x['iid_row_standard_error_used'] for x in (cs,primary_summary,combined['diagnostics']))

    # Planning only: exact budget for the actual frozen regularized proposals.
    production = HERE.parent/'staged_importance_regularized/artifacts/regularized'
    budget = []
    for i in range(3):
        path = production/f'stratum_{i:02d}.npz'
        p = FrozenProposal.load(path); alpha,nk,_ = component_budget(p)
        budget.append({'stratum_id':i,'proposal_path':str(path),'proposal_sha256':digest(path),
            'dimension':p.dimension,'component_full_probabilities':alpha,'Nk':nk,
            'rows_per_scramble':int(nk.sum()),'scrambles':8,'total_rows':int(8*nk.sum()),
            'GP_calls_performed':0})
    result = {'completed':True,'scheme':SCHEME,'settings':DEFAULT_SETTINGS,'checks':checks,
        'all_checks_passed':bool(all(checks.values())),
        'mapping_max_absolute_error':mapping_error,'density_max_absolute_log_error':density_error,
        'constant_integrand':exact_summary,'constant_likelihood':cs,
        'bimodal_truth':{'evidence':ztrue,'tail_posterior_mean':tail_true},
        'bimodal_primary_summary':primary_summary,'outer_independent_experiments':experiments,
        'outer_error_checks':errors,'complete_strata_toy':combined['diagnostics'],
        'regularized_proposal_actual_budget':budget,
        'regularized_proposal_all_strata_total_rows':sum(x['total_rows'] for x in budget),
        'toy_likelihood_calls':constant['diagnostics']['likelihood_calls']+sum(e['likelihood_calls'] for e in experiments),
        'scientific_GP_likelihood_calls':0,'scientific_fits_started':0,
        'elapsed_seconds':time.perf_counter()-started,
        'evidence_boundary':'Analytic toys and numerical mapping only; small-R uncertainty calibration is descriptive; no scientific precision or stability claim.'}
    (OUT/'toy_report.json').write_text(json.dumps(json_safe(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(json_safe({k:result[k] for k in ('all_checks_passed','checks','outer_error_checks',
        'regularized_proposal_all_strata_total_rows','toy_likelihood_calls','elapsed_seconds')}),ensure_ascii=False))
    if not all(checks.values()):
        raise AssertionError('Fixed toy check failed; inspect retained report')


if __name__=='__main__':
    main()
