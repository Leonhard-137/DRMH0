"""Integrate fixed sorted-center strata without changing the full prior.

Each child uses a normalized conditional prior. Its raw evidence weights
receive exactly one factor of the original prior probability of that stratum.
There is no global nested-sampling trajectory for the concatenated samples.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.special import logsumexp

from .nested import (NestedBudget, PriorTransform, _make_run_config,
                     _normalized_resume_config, load_saved_posterior, run_nested)
from .sampler import weighted_quantile


SCHEME = 'ordered_mu1_stratified_v1'
CHILD_FINAL = ('weighted_posterior.csv', 'nested_results.npz',
               'posterior_summary.json', 'diagnostics.json')
PARENT_FINAL = ('weighted_posterior.csv', 'stratified_results.npz',
                'posterior_summary.json', 'diagnostics.json')
SCHEDULER_FILES = {'fit.log', 'scheduler_state.json'}


def strata_definition(mode, priors, edges):
    transform = PriorTransform(mode, priors)
    edges = np.asarray(edges, dtype=float)
    a, b = transform.bounds[3]
    if (edges.ndim != 1 or len(edges) != 4 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0) or edges[0] != a or edges[-1] != b):
        raise ValueError('center_strata must provide four increasing edges covering the complete original center interval')
    # This also checks exchangeability and uniform, identical center bounds.
    masses = [PriorTransform(mode, transform.priors, (lo, hi)).center_prior_mass
              for lo, hi in zip(edges[:-1], edges[1:])]
    if not np.isclose(sum(masses), 1, rtol=0, atol=8*np.finfo(float).eps):
        raise ValueError('Strata do not partition the complete original prior')
    return transform, edges.tolist(), masses


def _child_state(folder):
    if not folder.exists():
        return 'missing'
    allowed = set(CHILD_FINAL) | {'sampler_config.json', 'dynesty.checkpoint'}
    unknown = {p.name for p in folder.iterdir()} - allowed
    if unknown:
        raise ValueError(f'Unrecognized child state in {folder}: {sorted(unknown)}')
    final = [(folder/name).is_file() for name in CHILD_FINAL]
    base = all((folder/name).is_file() for name in ('sampler_config.json', 'dynesty.checkpoint'))
    if all(final) and base:
        return 'complete'
    if not any(final) and base:
        return 'checkpoint'
    raise ValueError(f'Incomplete or ambiguous child state in {folder}; refusing to overwrite')


def _load_child(folder, expected):
    actual = json.loads((folder/'sampler_config.json').read_text())
    if _normalized_resume_config(actual) != _normalized_resume_config(expected):
        raise ValueError(f'Child configuration mismatch: {folder}')
    names = expected['parameter_names']
    with np.load(folder/'nested_results.npz', allow_pickle=False) as raw:
        result = {name: raw[name].copy() for name in
                  ('samples', 'logl', 'logwt', 'logvol', 'logz', 'logzerr', 'ncall')}
    count = len(result['logwt'])
    if (count == 0 or result['samples'].shape != (count, len(names))
            or any(result[key].shape != (count,) for key in ('logl','logvol','logwt','logz','logzerr','ncall'))):
        raise ValueError(f'Unaligned child arrays: {folder}')
    if (not all(np.isfinite(result[key]).all() for key in ('samples','logl','logvol'))
            or np.isnan(result['logwt']).any() or np.isposinf(result['logwt']).any()
            or not np.isfinite(result['logz'][-1]) or not np.isfinite(result['logzerr'][-1])
            or result['logzerr'][-1] < 0):
        raise ValueError(f'Invalid child numerical output: {folder}')
    logz = float(logsumexp(result['logwt']))
    if not np.isclose(logz, result['logz'][-1], rtol=0, atol=1e-8):
        raise ValueError(f'Child evidence and raw weights disagree: {folder}')
    lo, hi = expected['center_interval']
    centers = result['samples'][:, [names.index('mu0'), names.index('mu1')]]
    if np.any(centers[:,0] > centers[:,1]) or np.any(centers[:,1] < lo) or np.any(centers[:,1] > hi):
        raise ValueError(f'Child center interval violated: {folder}')
    for index, name in enumerate(names):
        low, high = expected['priors'][name]['bounds']
        if np.any(result['samples'][:,index] < low) or np.any(result['samples'][:,index] > high):
            raise ValueError(f'Child sample outside original prior: {folder}/{name}')
    table = load_saved_posterior(folder, names)
    weights = np.exp(result['logwt']-logz)
    if (not np.array_equal(table['samples'], result['samples'])
            or not np.array_equal(table['log_likelihood'], result['logl'])
            or not np.array_equal(table['log_prior_mass'], result['logvol'])
            or not np.allclose(table['weights'], weights, rtol=1e-12, atol=1e-15)):
        raise ValueError(f'Child weighted CSV disagrees with raw arrays: {folder}')
    result['diagnostics'] = json.loads((folder/'diagnostics.json').read_text())
    result['summary'] = json.loads((folder/'posterior_summary.json').read_text())
    diagnostic = result['diagnostics']
    if (not np.isclose(diagnostic['raw_log_evidence'], logz, rtol=0, atol=1e-8)
            or diagnostic['raw_sample_count'] != count):
        raise ValueError(f'Child diagnostics disagree with raw arrays: {folder}')
    return result


def merge_children(children, prior_masses):
    """Combine aligned conditional quadratures; input order defines stratum_id."""
    masses = np.asarray(prior_masses, dtype=float)
    if (masses.shape != (len(children),) or np.any(masses <= 0)
            or not np.isfinite(masses).all() or not np.isclose(masses.sum(),1,rtol=0,atol=1e-14)):
        raise ValueError('Invalid original-prior stratum masses')
    conditional_logz = np.array([logsumexp(child['logwt']) for child in children])
    errors = np.array([child['logzerr'][-1] for child in children])
    contribution = np.log(masses)+conditional_logz
    logz = float(logsumexp(contribution))
    posterior_mass = np.exp(contribution-logz)
    original_logwt = np.concatenate([np.log(mass)+child['logwt'] for mass,child in zip(masses,children)])
    weights = np.exp(original_logwt-logsumexp(original_logwt))
    ids = np.concatenate([np.full(len(child['logwt']),j,dtype=np.int64) for j,child in enumerate(children)])
    jacobian = np.diag(posterior_mass)-np.outer(posterior_mass,posterior_mass)
    covariance = jacobian@np.diag(errors**2)@jacobian.T
    return dict(samples=np.concatenate([child['samples'] for child in children]),
                logl=np.concatenate([child['logl'] for child in children]),
                conditional_logvol=np.concatenate([child['logvol'] for child in children]),
                conditional_logwt=np.concatenate([child['logwt'] for child in children]),
                original_prior_logwt=original_logwt, weights=weights, stratum_id=ids,
                stratum_prior_mass=masses, stratum_posterior_mass=posterior_mass,
                conditional_logz=conditional_logz, conditional_logzerr=errors,
                stratum_logz_contribution=contribution, logz=logz,
                logzerr=float(np.sqrt(np.sum((posterior_mass*errors)**2))),
                stratum_posterior_mass_covariance=covariance)


def _completed_parent(folder, config):
    if not all((folder/name).is_file() for name in PARENT_FINAL):
        raise ValueError('Completion marker exists but parent outputs are missing')
    marker = json.loads((folder/'completed.json').read_text())
    if marker != dict(integration_scheme=SCHEME, number_of_strata=len(config['child_seeds'])):
        raise ValueError('Invalid stratified completion marker')
    posterior = load_saved_posterior(folder, config['parameter_names'])
    with np.load(folder/'stratified_results.npz',allow_pickle=False) as raw:
        if (not np.array_equal(posterior['samples'],raw['samples'])
                or not np.array_equal(posterior['stratum_id'],raw['stratum_id'])
                or not np.array_equal(posterior['log_likelihood'],raw['logl'])
                or not np.array_equal(posterior['conditional_log_prior_mass'],raw['conditional_logvol'])
                or not np.allclose(posterior['weights'],raw['weights'],rtol=1e-12,atol=1e-15)):
            raise ValueError('Completed parent raw and weighted samples disagree')
    return dict(summary=json.loads((folder/'posterior_summary.json').read_text()),
                diagnostics=json.loads((folder/'diagnostics.json').read_text()),
                weighted_posterior=str(folder/'weighted_posterior.csv'), reused_completed=True)


def run_stratified(log_likelihood, output_dir, mode='independent', priors=None,
                   budget=None, seed=12345, center_strata=(-3,5,20,28),
                   resume=False, input_files=None, run_input=None):
    """Sequential independent strata, strict state checking, exact mass merge.

    input_files optionally supplies the original driver/response paths for the
    parent gate/reconstruction interface. No child data normalization occurs.
    Completed children are read only. Unknown partial output is an error.
    """
    import dynesty
    start = time.perf_counter()
    budget = NestedBudget() if budget is None else (NestedBudget(**budget) if isinstance(budget,dict) else budget)
    if not isinstance(seed,(int,np.integer)) or isinstance(seed,bool) or not 0 <= seed <= 2**32-1:
        raise ValueError('seed must be a uint32 integer')
    transform,edges,masses = strata_definition(mode,priors,center_strata)
    payload = pickle.dumps(log_likelihood,protocol=pickle.HIGHEST_PROTOCOL)
    child_seeds = [int(child.generate_state(1,dtype=np.uint32)[0])
                   for child in np.random.SeedSequence(int(seed)).spawn(len(masses))]
    if len(set(child_seeds)) != len(child_seeds):
        raise ValueError('Derived child seed collision')
    if input_files is not None and len(input_files) != 2:
        raise ValueError('input_files must contain driver and response paths')
    input_bytes = None if input_files is None else [Path(path).read_bytes() for path in input_files]
    config = dict(sampler='dynesty',sampler_version=dynesty.__version__,integration_scheme=SCHEME,
        mode=mode,parameter_names=list(transform.names),priors=transform.priors,
        budget=asdict(budget),seed=int(seed),center_strata=edges,stratum_prior_masses=masses,
        child_seeds=child_seeds,child_seed_derivation='numpy.SeedSequence(parent_seed).spawn; generate_state(1, uint32)',
        likelihood_pickle_sha256=hashlib.sha256(payload).hexdigest(),
        likelihood_pickle_protocol=pickle.HIGHEST_PROTOCOL,
        center_prior='original uniform rectangle conditioned on mu0 < mu1, partitioned completely by sorted mu1',
        parameterization=f'{SCHEME}; center_strata={edges}; conditional sorted-mu1 inverse CDF',
        effort_definition='budget.nlive applies to every stratum independently',
        total_initial_live_points=budget.nlive*len(masses),
        has_input_files=input_bytes is not None,run_input=run_input)
    folder = Path(output_dir).resolve()
    config_path = folder/'sampler_config.json'
    if resume:
        if not config_path.is_file():
            raise ValueError('Stratified resume requires its original parent configuration')
        previous = json.loads(config_path.read_text())
        if _normalized_resume_config(previous) != _normalized_resume_config(config):
            raise ValueError('Stratified resume requires the exact original configuration and likelihood fingerprint')
        if input_bytes is not None:
            for name,content in zip(('input_w2.txt','input_response.txt'),input_bytes):
                if not (folder/name).is_file() or (folder/name).read_bytes() != content:
                    raise ValueError(f'Resume input differs from saved {name}')
        if run_input is not None:
            if not (folder/'run_input.json').is_file() or json.loads((folder/'run_input.json').read_text()) != run_input:
                raise ValueError('Resume run_input differs from saved metadata')
    else:
        if folder.exists() and {path.name for path in folder.iterdir()}-SCHEDULER_FILES:
            raise FileExistsError('Stratified output directory must be new, empty, or contain only recognized scheduler logs')
        folder.mkdir(parents=True,exist_ok=True)
        config_path.write_text(json.dumps(config,indent=2))
        if input_bytes is not None:
            for name,content in zip(('input_w2.txt','input_response.txt'),input_bytes):
                (folder/name).write_bytes(content)
        if run_input is not None:
            (folder/'run_input.json').write_text(json.dumps(run_input,indent=2))

    parent_files = [name for name in PARENT_FINAL if (folder/name).exists()]
    complete_parent = (folder/'completed.json').exists()
    if parent_files and not complete_parent:
        raise ValueError('Partial parent output without completion marker; refusing to overwrite')
    allowed = set(PARENT_FINAL) | {'completed.json','sampler_config.json','strata'} | SCHEDULER_FILES
    if input_bytes is not None:
        allowed.update(('input_w2.txt','input_response.txt'))
    if run_input is not None:
        allowed.add('run_input.json')
    unknown = {path.name for path in folder.iterdir()}-allowed
    if unknown:
        raise ValueError(f'Unrecognized parent state: {sorted(unknown)}')
    strata_root = folder/'strata'
    expected_folders = [f'stratum_{j:02d}' for j in range(len(masses))]
    if strata_root.exists() and {p.name for p in strata_root.iterdir()}-set(expected_folders):
        raise ValueError('Unrecognized child directory under strata')

    # Validate every present child before advancing any child.
    expected_configs=[]; states=[]; preloaded=[]
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        child_transform = PriorTransform(mode,transform.priors,(lo,hi))
        expected = _make_run_config(child_transform,budget,child_seeds[j],payload,mode)
        child_folder = strata_root/expected_folders[j]
        state = _child_state(child_folder)
        if state != 'missing':
            actual = json.loads((child_folder/'sampler_config.json').read_text())
            if _normalized_resume_config(actual) != _normalized_resume_config(expected):
                raise ValueError(f'Child configuration mismatch: {child_folder}')
        if complete_parent and state != 'complete':
            raise ValueError('Completed parent references an incomplete child')
        cached = _load_child(child_folder,expected) if state == 'complete' else None
        expected_configs.append(expected); states.append(state); preloaded.append(cached)
    if complete_parent:
        return _completed_parent(folder,config)

    children=[]; invocation_exact=0; child_records=[]
    for j,state in enumerate(states):
        child_folder = strata_root/expected_folders[j]
        if state != 'complete':
            # Independent copies retain an identical initial callback payload;
            # one child's incidental callback state cannot alter another child.
            run_nested(pickle.loads(payload),child_folder,mode=mode,priors=transform.priors,
                       budget=budget,seed=child_seeds[j],resume=state=='checkpoint',
                       center_interval=edges[j:j+2])
        child = preloaded[j] if state == 'complete' else _load_child(child_folder,expected_configs[j])
        diagnostic=child['diagnostics']
        if state != 'complete':
            invocation_exact += diagnostic['exact_calls_this_invocation']
        children.append(child)
        child_records.append(dict(stratum_id=j,interval=edges[j:j+2],seed=child_seeds[j],
            prior_mass=masses[j],reused_completed=state=='complete',
            raw_sample_count=len(child['logwt']),weight_ess=diagnostic['weight_ess'],
            max_posterior_weight=diagnostic['max_posterior_weight'],
            exact_likelihood_calls_total=diagnostic['exact_likelihood_calls_total'],
            exact_calls_since_instrumentation=diagnostic['exact_calls_since_instrumentation'],
            dynesty_reported_ncall=diagnostic['dynesty_reported_ncall'],
            sampler_effective=diagnostic['sampler_effective']))
    merged=merge_children(children,masses)
    weights=merged['weights']; names=list(transform.names)
    np.savez_compressed(folder/'stratified_results.npz',parameter_names=np.array(names),**merged)
    np.savetxt(folder/'weighted_posterior.csv',np.column_stack((merged['samples'],merged['logl'],
        merged['conditional_logvol'],merged['stratum_id'],merged['original_prior_logwt'],weights)),
        delimiter=',',header=','.join(names+['log_likelihood','conditional_log_prior_mass',
                                          'stratum_id','original_prior_logweight','weight']),comments='')
    summary={name:dict(zip(('q16','q50','q84'),map(float,weighted_quantile(merged['samples'][:,i],weights))))
             for i,name in enumerate(names)}
    for j,record in enumerate(child_records):
        record.update(conditional_log_evidence=float(merged['conditional_logz'][j]),
            conditional_log_evidence_error=float(merged['conditional_logzerr'][j]),
            original_prior_log_evidence_contribution=float(merged['stratum_logz_contribution'][j]),
            posterior_mass=float(merged['stratum_posterior_mass'][j]))
    exact_values=[record['exact_likelihood_calls_total'] for record in child_records]
    reported=sum(record['dynesty_reported_ncall'] for record in child_records)
    diagnostics=dict(sampler='dynesty',integration_scheme=SCHEME,raw_sample_count=len(weights),
        weight_ess=float(1/np.sum(weights**2)),max_posterior_weight=float(weights.max()),
        weight_ess_is_not_autocorrelation_ess=True,raw_log_evidence=merged['logz'],
        raw_log_evidence_error=merged['logzerr'],
        evidence_error_method='delta method from independent conditional logZ errors; does not account for missed modes or conditional sampling bias',
        stratum_posterior_mass_covariance=merged['stratum_posterior_mass_covariance'].tolist(),
        strata=child_records,likelihood_calls=reported,dynesty_reported_ncall=reported,
        likelihood_calls_definition='legacy alias of sum of child dynesty reported counts, not exact callback calls',
        exact_likelihood_calls_total=None if any(value is None for value in exact_values) else sum(exact_values),
        exact_calls_since_instrumentation=sum(record['exact_calls_since_instrumentation'] for record in child_records),
        exact_calls_this_invocation=invocation_exact,
        exact_count_includes_full_history=all(value is not None for value in exact_values),
        exact_count_scope='sum of checkpoint-retained child trajectories, including fresh initialization; excludes discarded post-checkpoint work',
        elapsed_seconds=time.perf_counter()-start,elapsed_seconds_scope='this parent invocation including reused-child reads and merge',
        resumed_from_checkpoint=bool(resume),
        trajectory='concatenated conditional quadratures; no global log-prior-mass trajectory exists',
        stopping_rule='each conditional run uses its recorded dlogz; independent seed/budget validation remains required',
        convergence_status='not_established_pending_stability_gate',evidence_for_model_comparison=False)
    (folder/'posterior_summary.json').write_text(json.dumps(summary,indent=2))
    (folder/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    (folder/'completed.json').write_text(json.dumps(dict(integration_scheme=SCHEME,number_of_strata=len(masses)),indent=2))
    return dict(summary=summary,diagnostics=diagnostics,weighted_posterior=str(folder/'weighted_posterior.csv'))
