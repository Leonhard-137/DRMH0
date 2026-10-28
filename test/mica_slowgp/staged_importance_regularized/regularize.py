"""Deterministic proposal covariance shrinkage; no EM or likelihood calls."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_t

from importance import FrozenProposal, json_safe, log_prior_z


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'staged_importance_refined'
PILOT = Path('/home/leonhard/sci001/output/mica_slowgp/importance_pilot_v1/seed_82103_n16384')
OUT = HERE / 'artifacts/regularized'
METHOD = 'component_ess_block_covariance_shrinkage_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regularize_proposal(source):
    """Return a new normalized proposal and unscaled covariance audit arrays.

    The block total covariance is the covariance of its fitted Gaussian mixture,
    not the inflated Student mixture. d+2 is fixed proposal regularization only.
    """
    d = source.dimension
    inflation = float(source.metadata['inflation_is_standard_deviation_multiplier'])
    if source.df != 7 or inflation != 1.2 or source.epsilon != .05:
        raise ValueError('This fixed candidate requires df=7, inflation=1.2, epsilon=.05')
    block_masses = np.asarray(source.metadata['block_full_probabilities'])
    fits = source.metadata['training_blocks']
    counts = [int(f['n_components']) for f in fits]
    if counts != [12, 12] or not np.array_equal(block_masses, [.475, .475]):
        raise ValueError('Unexpected training block configuration')
    conversion = inflation**2 * (source.df - 2) / source.df
    training_covariances = source.scales / conversion
    new_covariances = np.empty_like(training_covariances)
    ess_all = []
    alpha_all = []
    block_covariances = []
    block_means = []
    block_within_covariances = []
    block_between_covariances = []
    block_reports = []
    offset = 0
    for name, count, mass, fit in zip(('base', 'tail'), counts, block_masses, fits):
        take = slice(offset, offset + count)
        full_weights = source.component_weights[take]
        if not np.isclose(full_weights.sum(), mass, rtol=0, atol=1e-14):
            raise ValueError('Block full probability mismatch')
        weights = full_weights / full_weights.sum()
        if not np.allclose(weights, fit['final_component_probabilities'], rtol=1e-13, atol=1e-15):
            raise ValueError('Component weights do not match training metadata')
        means = source.means[take]
        covariances = training_covariances[take]
        ess = np.asarray(fit['final_component_responsibility_weight_ess'], dtype=float)
        if ess.shape != (count,) or not np.all(np.isfinite(ess)) or np.any(ess <= 0):
            raise ValueError('Invalid component responsibility ESS')
        block_mean = np.einsum('k,kj->j', weights, means)
        centered = means - block_mean
        within = np.einsum('k,kij->ij', weights, covariances)
        between = np.einsum('k,ki,kj->ij', weights, centered, centered)
        block_covariance = within + between
        alpha = (d + 2) / (ess + d + 2)
        new_covariances[take] = ((1 - alpha[:, None, None]) * covariances
                                + alpha[:, None, None] * block_covariance)
        np.linalg.cholesky(block_covariance)
        np.linalg.cholesky(new_covariances[take])
        block_means.append(block_mean)
        block_covariances.append(block_covariance)
        block_within_covariances.append(within)
        block_between_covariances.append(between)
        ess_all.extend(ess)
        alpha_all.extend(alpha)
        low = ess < d + 1
        block_reports.append({
            'name': name, 'component_slice': [offset, offset + count],
            'full_probability': float(full_weights.sum()),
            'responsibility_ess': ess.tolist(), 'alpha': alpha.tolist(),
            'alpha_min': float(alpha.min()), 'alpha_max': float(alpha.max()),
            'low_ess_descriptive_threshold': d + 1,
            'low_ess_components': np.flatnonzero(low).tolist(),
            'low_ess_full_proposal_probability': float(full_weights[low].sum()),
            'block_covariance_min_eigenvalue': float(np.linalg.eigvalsh(block_covariance).min()),
            'training_component_min_eigenvalue': float(np.linalg.eigvalsh(covariances).min()),
            'regularized_component_min_eigenvalue': float(np.linalg.eigvalsh(new_covariances[take]).min()),
            'within_trace': float(np.trace(within)), 'between_trace': float(np.trace(between)),
        })
        offset += count
    new_scales = new_covariances * conversion
    metadata = dict(source.metadata)
    metadata['purpose'] = 'fixed covariance-regularized importance proposal; no fresh likelihood evaluation'
    metadata['proposal_covariance_regularization'] = {
        'method': METHOD, 'dimension': d, 'heuristic_strength': d + 2,
        'alpha_formula': '(d+2)/(component_responsibility_ESS+d+2)',
        'training_covariance_formula': 'Student_shape/[SD_inflation**2*(df-2)/df]',
        'block_covariance_formula': 'sum(pi_k*(Cov_k+(mean_k-block_mean)(mean_k-block_mean)^T))',
        'new_training_covariance_formula': '(1-alpha_k)*Cov_k+alpha_k*Cov_block',
        'scope': 'proposal distribution only; scientific prior and target unchanged',
        'means_full_weights_epsilon_unchanged': True, 'EM_rerun': False,
        'fresh_likelihood_calls': 0, 'blocks': block_reports,
    }
    result = FrozenProposal(source.component_weights, source.means, new_scales,
                            source.df, source.epsilon, metadata)
    audit = {
        'training_covariances': training_covariances,
        'regularized_training_covariances': new_covariances,
        'block_covariances': np.asarray(block_covariances),
        'block_means': np.asarray(block_means),
        'block_within_covariances': np.asarray(block_within_covariances),
        'block_between_covariances': np.asarray(block_between_covariances),
        'component_ess': np.asarray(ess_all), 'alpha': np.asarray(alpha_all),
        'student_conversion_factor': np.array(conversion),
    }
    return result, audit, block_reports


def main():
    started = time.perf_counter()
    if OUT.exists():
        raise FileExistsError('Output exists; no candidate overwrite allowed')
    training_report = json.loads((SOURCE / 'artifacts/trained/training_report.json').read_text())
    source_files = [SOURCE / 'importance.py', SOURCE / 'artifacts/trained/training_report.json']
    source_files += [SOURCE / f'artifacts/trained/stratum_{i:02d}.npz' for i in range(3)]
    source_files += [PILOT / f'stratum_{i:02d}.npz' for i in range(3)]
    before = {str(p): digest(p) for p in source_files}
    if digest(HERE / 'importance.py') != digest(SOURCE / 'importance.py'):
        raise ValueError('Importance evaluator must remain byte-identical to refined source')
    OUT.mkdir(parents=True)
    reports = []
    for i in range(3):
        source_path = SOURCE / f'artifacts/trained/stratum_{i:02d}.npz'
        old = FrozenProposal.load(source_path)
        new, audit, blocks = regularize_proposal(old)
        new.metadata['proposal_covariance_regularization']['source_proposal'] = {
            'path': str(source_path), 'sha256': before[str(source_path)]}
        output = OUT / f'stratum_{i:02d}.npz'
        new.save(output)
        audit_path = OUT / f'stratum_{i:02d}_covariance_audit.npz'
        np.savez_compressed(audit_path, **audit)
        roundtrip = FrozenProposal.load(output)
        probes = training_report['strata'][i]['training_row_density_probes']
        indices = [int(x['pilot_row']) for x in probes]
        with np.load(PILOT / f'stratum_{i:02d}.npz', allow_pickle=False) as raw:
            z = raw['z'][indices]
            first_logq = raw['logq_z'][indices]
        refined_logq = old.logpdf_z(z)
        regularized_logq = new.logpdf_z(z)
        density_probes = []
        for j, probe in enumerate(probes):
            density_probes.append({**probe,
                'first_pilot_logq': float(first_logq[j]),
                'refined_logq': float(refined_logq[j]),
                'regularized_logq': float(regularized_logq[j]),
                'regularized_minus_first_pilot_logq': float(regularized_logq[j] - first_logq[j]),
                'regularized_minus_refined_logq': float(regularized_logq[j] - refined_logq[j]),
                'interpretation': 'training density placement only; no new importance estimate or independent efficiency claim'})
        # Independent normalized Student-t density implementation, without GP calls.
        columns = [np.log(w) + multivariate_t.logpdf(z, loc=mu, shape=scale, df=new.df)
                   for w, mu, scale in zip(new.component_weights, new.means, new.scales)]
        columns.append(np.log(new.epsilon) + log_prior_z(z))
        scipy_logq = logsumexp(np.column_stack(columns), axis=1)
        pdf_error = float(np.max(np.abs(scipy_logq - regularized_logq)))
        if pdf_error > 1e-10:
            raise AssertionError('Independent Student density mismatch')
        smoke_seed = 880100 + i
        draws = new.sample_z(1024, np.random.default_rng(smoke_seed))
        if not np.isfinite(new.logpdf_z(draws)).all():
            raise AssertionError('Nonfinite proposal sampling/pdf smoke')
        checks = {
            'means_bitwise_unchanged': np.array_equal(old.means, new.means),
            'full_weights_bitwise_unchanged': np.array_equal(old.component_weights, new.component_weights),
            'epsilon_unchanged': old.epsilon == new.epsilon,
            'df_unchanged': old.df == new.df,
            'total_probability': float(new.component_weights.sum() + new.epsilon),
            'normalization_absolute_error': float(abs(new.component_weights.sum() + new.epsilon - 1)),
            'all_scales_positive_definite': bool(np.all(np.linalg.eigvalsh(new.scales) > 0)),
            'minimum_scale_eigenvalue': float(np.linalg.eigvalsh(new.scales).min()),
            'save_load_density_bitwise_equal': np.array_equal(roundtrip.logpdf_z(z), regularized_logq),
            'scipy_full_density_max_absolute_log_error': pdf_error,
            'proposal_sample_pdf_finite': True, 'proposal_only_smoke_rows': 1024,
            'proposal_only_smoke_seed': smoke_seed,
        }
        if not all(checks[k] for k in ('means_bitwise_unchanged', 'full_weights_bitwise_unchanged',
               'epsilon_unchanged', 'df_unchanged', 'all_scales_positive_definite',
               'save_load_density_bitwise_equal')):
            raise AssertionError('Proposal preservation validation failed')
        reports.append({'stratum_id': i, 'path': str(output), 'sha256': digest(output),
            'covariance_audit': {'path': str(audit_path), 'sha256': digest(audit_path)},
            'center_interval': old.metadata['center_interval'],
            'stratum_prior_mass': old.metadata['stratum_prior_mass'], 'blocks': blocks,
            'low_ess_full_proposal_probability': sum(b['low_ess_full_proposal_probability'] for b in blocks),
            'checks': checks, 'training_density_probes': density_probes})
    unchanged = {p: digest(p) == value for p, value in before.items()}
    if not all(unchanged.values()):
        raise AssertionError('An input file changed')
    report = {'completed': True, 'candidate': METHOD, 'strata': reports,
        'elapsed_seconds': time.perf_counter() - started,
        'source_files_unchanged': unchanged, 'source_sha256': before,
        'importance_module_byte_identical': True,
        'EM_calls': 0, 'GP_likelihood_calls': 0, 'scientific_fits_started': 0,
        'fresh_scientific_evaluation_draws': 0,
        'training_covariance_not_student_covariance': True,
        'd_plus_2_is_fixed_proposal_regularization_not_scientific_prior': True,
        'independent_fresh_evaluation_required': True,
        'normalization_argument': 'Every positive-definite Student-t component and defensive logistic prior is normalized; unchanged positive full probabilities sum to one.',
        'evidence_boundary': 'Finite numerical covariance/density checks; no claim of lower importance variance, posterior convergence or physical recovery.'}
    path = OUT / 'regularization_report.json'
    path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'completed': True, 'report': str(path), 'elapsed_seconds': report['elapsed_seconds'],
        'GP_likelihood_calls': 0, 'EM_calls': 0,
        'strata': [{k: s[k] for k in ('stratum_id', 'path', 'low_ess_full_proposal_probability', 'checks', 'training_density_probes')} for s in reports]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
