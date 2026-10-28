"""Randomized Sobol sampling stratified by every frozen proposal component.

This module does not import or call a scientific GP implementation. Rows are
quadrature contributions, not iid importance draws. Error estimates use the
independent scramble integrals exclusively.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.special import expit, ndtri, logsumexp
from scipy.stats import chi2, qmc

from importance import FrozenProposal, log_prior_z, unit_logs

SCHEME = 'randomized_sobol_full_mixture_component_stratification_v1'
DEFAULT_SETTINGS = {'target_N': 2048, 'scrambles': 8, 'minimum_power': 3,
                    'sobol_bits': 52, 'scramble': True}


def component_budget(proposal, target_N=2048, minimum_power=3):
    if not isinstance(target_N, (int, np.integer)) or target_N < 1:
        raise ValueError('target_N must be a positive integer')
    if minimum_power != 3:
        raise ValueError('The frozen candidate minimum power is three')
    probabilities = np.r_[proposal.component_weights, proposal.epsilon]
    powers = np.maximum(minimum_power, np.ceil(np.log2(target_N * probabilities)).astype(int))
    if np.any(powers > 52):
        raise ValueError('Requested component budget exceeds Sobol bit depth')
    counts = 2 ** powers
    return probabilities, counts, powers


def component_map(uniforms, proposal, component_id):
    """Map open uniforms to exactly one normalized mixture component.

    Endpoint uniforms cause a visible error; there is no clipping, rejection
    loop, scramble replacement or silent endpoint reassignment.
    """
    u = np.asarray(uniforms, dtype=float)
    defensive = component_id == len(proposal.means)
    dimension = proposal.dimension if defensive else proposal.dimension + 1
    if (u.ndim != 2 or u.shape[1] != dimension or not np.isfinite(u).all()
            or np.any(u <= 0) or np.any(u >= 1)):
        raise FloatingPointError('Sobol coordinate outside strict (0,1); no clipping or replacement performed')
    if not 0 <= component_id <= len(proposal.means):
        raise ValueError('Unknown component')
    if defensive:
        z = np.log(u) - np.log1p(-u)
    else:
        normal = ndtri(u[:, :proposal.dimension])
        radial = chi2.ppf(u[:, -1], df=proposal.df)
        z = (proposal.means[component_id]
             + (normal @ proposal._cholesky[component_id].T)
             / np.sqrt(radial / proposal.df)[:, None])
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite mapped Sobol point; no resampling performed')
    return z


def sample_components(proposal, *, seed, stratum_id=0, target_N=2048, scrambles=8):
    if (not isinstance(seed, (int, np.integer)) or seed < 0
            or not isinstance(stratum_id, (int, np.integer)) or stratum_id < 0
            or not isinstance(scrambles, (int, np.integer)) or scrambles < 2):
        raise ValueError('Nonnegative seed/stratum ID and at least two scrambles required')
    probabilities, counts, powers = component_budget(proposal, target_N)
    zs, categories, replicates, factors = [], [], [], []
    seeds = []
    for r in range(scrambles):
        for k, (alpha, count, power) in enumerate(zip(probabilities, counts, powers)):
            # SeedSequence spawn keys identify independent randomization streams;
            # no truncation to a single 32-bit seed or data-dependent reseeding.
            sequence = np.random.SeedSequence(int(seed), spawn_key=(int(stratum_id), r, k))
            rng = np.random.default_rng(sequence)
            d = proposal.dimension + int(k < len(proposal.means))
            engine = qmc.Sobol(d=d, scramble=True, bits=52, rng=rng, optimization=None)
            uniforms = engine.random_base2(int(power))
            z = component_map(uniforms, proposal, k)
            zs.append(z)
            categories.append(np.full(count, k, dtype=np.int64))
            replicates.append(np.full(count, r, dtype=np.int64))
            factors.append(np.full(count, np.log(alpha) - np.log(count) - np.log(scrambles)))
            seeds.append({'root_entropy': int(seed), 'spawn_key': [int(stratum_id), r, k],
                          'count': int(count), 'sobol_dimension': d})
    return {'z': np.concatenate(zs), 'component_id': np.concatenate(categories),
            'scramble_id': np.concatenate(replicates),
            'log_quadrature_factor': np.concatenate(factors),
            'component_probabilities': probabilities, 'component_counts_per_scramble': counts,
            'seed_hierarchy': seeds, 'rows_per_scramble': int(counts.sum()),
            'scrambles': int(scrambles), 'seed': int(seed), 'stratum_id': int(stratum_id)}


def summarize_scrambles(logweight, scramble_id, scrambles, observables=None, saturated=None):
    """logweight includes -log(R); the R integral estimates drive all MCSEs."""
    logweight = np.asarray(logweight, dtype=float)
    ids = np.asarray(scramble_id)
    if (logweight.ndim != 1 or ids.shape != logweight.shape or scrambles < 2
            or np.isnan(logweight).any() or np.isposinf(logweight).any()
            or not np.isfinite(logweight).any()
            or not np.array_equal(np.unique(ids), np.arange(scrambles))):
        raise ValueError('Invalid scramble rows; keep every row including zero contributions')
    observables = observables or {}
    names = list(observables)
    if any(np.asarray(v).shape != logweight.shape or not np.isfinite(v).all()
           for v in observables.values()):
        raise ValueError('Invalid observable rows')
    logz_replicates = np.array([logsumexp(logweight[ids == r]) + np.log(scrambles)
                               for r in range(scrambles)])
    scale = float(logz_replicates.max())
    scaled_rows_per_replicate = np.exp(logweight + np.log(scrambles) - scale)
    # NumPy pairwise sums avoid the long sequential accumulation in bincount;
    # each replicate remains an integral over every component and every row.
    vectors = np.array([[np.sum(scaled_rows_per_replicate[ids == r])] + [
        np.sum(scaled_rows_per_replicate[ids == r] * np.asarray(observables[name])[ids == r])
        for name in names] for r in range(scrambles)])
    means = vectors.mean(axis=0)
    covariance_of_mean = np.atleast_2d(np.cov(vectors, rowvar=False, ddof=1)) / scrambles
    zmean = means[0]
    logz = scale + np.log(zmean)
    weights = np.exp(logweight - logz)
    stderr = float(np.sqrt(covariance_of_mean[0, 0]))
    obs_report = {}
    gradient = np.zeros((len(names), len(names) + 1))
    for j, name in enumerate(names, 1):
        ratio = float(means[j] / zmean)
        gradient[j - 1, 0] = -ratio / zmean
        gradient[j - 1, j] = 1 / zmean
        variance = float(gradient[j - 1] @ covariance_of_mean @ gradient[j - 1])
        obs_report[name] = {'posterior_mean': ratio,
            'ratio_delta_standard_error': float(np.sqrt(max(0., variance))),
            'unnormalized_mean_scaled': float(means[j]),
            'unnormalized_standard_error_scaled': float(np.sqrt(covariance_of_mean[j, j]))}
    result = {'scheme': SCHEME, 'scrambles': int(scrambles), 'N_rows': len(logweight),
        'log_evidence': float(logz),
        'log_evidence_standard_error': float(scale + np.log(stderr)) if stderr else -np.inf,
        'relative_evidence_standard_error': stderr / zmean,
        'scramble_integral_column_order': ['evidence'] + names,
        'scramble_integral_log_scale': scale, 'scramble_integrals_scaled': vectors,
        'scramble_integral_mean_covariance_scaled': covariance_of_mean,
        'observable_ratio_covariance': gradient @ covariance_of_mean @ gradient.T,
        'observables': obs_report, 'zero_weight_rows_retained': int(np.isneginf(logweight).sum()),
        'weight_ess_descriptive_only': float(1 / np.sum(weights**2)),
        'maximum_normalized_row_weight': float(weights.max()),
        'standard_error_method': 'sample covariance of independent scramble integral vectors divided by R; delta method for posterior ratios',
        'iid_row_standard_error_used': False, 'posterior_ratio_unbiased': False,
        'error_caveat': 'Empirical randomization error from finite R; ratio delta method is approximate; rare missed contributions may invalidate apparent precision.'}
    if saturated is not None:
        saturated = np.asarray(saturated, dtype=bool)
        if saturated.shape != logweight.shape:
            raise ValueError('Invalid saturation flags')
        result.update(saturated_rows=int(saturated.sum()),
            saturated_normalized_weight_mass=float(weights[saturated].sum()))
    return result, weights


def evaluate_u(log_likelihood_u, proposal, *, seed, stratum_id=0, target_N=2048,
               scrambles=8, allow_closed_unit=False, observable_functions=None):
    """Evaluate a single-row callback on new component-stratified Sobol draws.

    logweight is logL+logpi_z-logq_z+log(alpha_k)-log(N_k)-log(R).
    No scientific prior stratum mass is included until combine_strata.
    """
    started = time.perf_counter()
    result = sample_components(proposal, seed=seed, stratum_id=stratum_id,
                               target_N=target_N, scrambles=scrambles)
    z = result['z']
    u = expit(z)
    saturated = np.any((u == 0) | (u == 1), axis=1)
    if saturated.any() and not allow_closed_unit:
        raise FloatingPointError('Natural expit saturation: explicitly allow a continuous closed-unit target; no clipping or replacement')
    logp = log_prior_z(z)
    logq = proposal.logpdf_z(z)
    logl = np.array([log_likelihood_u(row) for row in u], dtype=float)
    if np.isnan(logl).any() or np.isposinf(logl).any():
        raise ValueError('Target returned NaN or +inf; no rows dropped')
    logimportance = logl + logp - logq
    logweight = logimportance + result['log_quadrature_factor']
    obs = {name: np.array([f(row) for row in u], dtype=float)
           for name, f in (observable_functions or {}).items()}
    diagnostics, weights = summarize_scrambles(logweight, result['scramble_id'], scrambles, obs, saturated)
    diagnostics.update(seed=int(seed), stratum_id=int(stratum_id),
        target_N=int(target_N), rows_per_scramble=result['rows_per_scramble'],
        component_counts_per_scramble=result['component_counts_per_scramble'],
        component_full_probabilities=result['component_probabilities'],
        likelihood_calls=len(z), elapsed_seconds=time.perf_counter() - started,
        allow_closed_unit=bool(allow_closed_unit), sobol_endpoint_policy='fail-fast; no replacement or clipping',
        training_rows_used_as_evaluation=False)
    logu, log1mu = unit_logs(z)
    result.update(u=u, logu=logu, log1mu=log1mu, logp_z=logp, logq_z=logq, logl=logl,
                  logimportance=logimportance, logweight=logweight, weights=weights,
                  saturated=saturated, observables=obs, diagnostics=diagnostics)
    return result


def combine_strata(evaluations, prior_masses):
    """All centers retained; combine corresponding independent scramble vectors."""
    masses = np.asarray(prior_masses, dtype=float)
    if (masses.shape != (len(evaluations),) or np.any(masses <= 0)
            or not np.isclose(masses.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError('A complete positive prior partition is required')
    R = evaluations[0]['scrambles']
    seed = evaluations[0]['seed']
    ids = [e['stratum_id'] for e in evaluations]
    names = list(evaluations[0]['observables'])
    if (len(set(ids)) != len(ids) or any(e['scrambles'] != R or e['seed'] != seed
            or set(e['observables']) != set(names) for e in evaluations)):
        raise ValueError('Require same R/root seed, unique stratum IDs and common observables')
    logrows = np.concatenate([e['logweight'] + np.log(rho) for e, rho in zip(evaluations, masses)])
    scramble_ids = np.concatenate([e['scramble_id'] for e in evaluations])
    stratum_ids = np.concatenate([np.full(len(e['logweight']), i, dtype=int)
                                  for i, e in enumerate(evaluations)])
    obs = {name: np.concatenate([e['observables'][name] for e in evaluations]) for name in names}
    for i in range(len(evaluations)):
        key = f'__stratum_mass_{i}'
        if key in obs:
            raise ValueError('Reserved stratum observable name')
        obs[key] = (stratum_ids == i).astype(float)
    saturated = np.concatenate([e['saturated'] for e in evaluations])
    summary, weights = summarize_scrambles(logrows, scramble_ids, R, obs, saturated)
    summary.update(prior_masses=masses, all_center_strata_retained=True,
        likelihood_calls=sum(e['diagnostics']['likelihood_calls'] for e in evaluations),
        Ns=[len(e['logweight']) for e in evaluations],
        weight_rule='log rho_s + logL+logpi_z-logq_z+logalpha_k-logNk-logR')
    return {'logweight': logrows, 'weights': weights, 'stratum_id': stratum_ids,
            'scramble_id': scramble_ids, 'observables': obs, 'saturated': saturated,
            'diagnostics': summary}
