#!/usr/bin/env python3
"""Record only the six completed v3 shared/shared parent posteriors.

No GP evaluation, sampling, weight replacement, threshold change or truth-based
selection occurs here. Conditional raw arrays are checked against their parents.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

BASE = Path('/home/leonhard')
ROOT = BASE / 'sci001/output/mica_slowgp/stability_v3_stratified'
TEST = BASE / 'sci001/test/mica_slowgp'
ARCHIVE = BASE / 'software/MICA2-slowgp/archive/stability-v3-stratified-fix1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(BASE))


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main():
    destinations = {name: ROOT / name for name in (
        'phase_summary.json', 'phase_report.md', 'phase_dynesty_v3_manifest.json',
        'phase_dynesty_v3_task.md', 'phase_dynesty_v3_computation_record.json')}
    for path in destinations.values():
        if path.exists():
            raise FileExistsError(path)
    plan = read(ROOT / 'assessment_plan.json')
    paths = [Path(plan['command'][i + 1]) for i, value in enumerate(plan['command'])
             if value == '--run-dir']
    assert len(paths) == len(set(paths)) == 6
    gate_path = ROOT / 'representative_dynesty_shared.json'
    gate = read(gate_path)
    assert not gate['rejected_runs'] and len(gate['groups']) == 1
    group = gate['groups'][0]
    assert (group['case'], group['mode'], group['sampler']) == ('shared', 'shared', 'dynesty')
    assert not group['design_issues'] and len(group['comparisons']) == 6
    assert {run['path'] for run in group['runs']} == {str(path) for path in paths}
    files, runs, max_weight_error, max_logwt_error = [], [], 0.0, 0.0
    common_priors = common_inputs = None
    expected_mass = np.array([64.0, 465.0, 432.0]) / 961.0
    required = ('sampler_config.json', 'run_input.json', 'input_w2.txt', 'input_response.txt',
                'weighted_posterior.csv', 'stratified_results.npz', 'posterior_summary.json',
                'diagnostics.json', 'completed.json', 'scheduler_state.json')
    for folder in paths:
        assert all((folder / name).is_file() for name in required), folder
        state = read(folder / 'scheduler_state.json')
        assert state['status'] == 'complete' and not (Path('/proc') / str(state['pid'])).exists()
        config, diag = read(folder / 'sampler_config.json'), read(folder / 'diagnostics.json')
        assert config['integration_scheme'] == 'ordered_mu1_stratified_v1'
        assert config['mode'] == 'shared' and config['center_strata'] == [-3.0, 5.0, 20.0, 28.0]
        assert config['budget']['sample'] == 'rslice' and config['budget']['slices'] == 14
        assert config['budget']['update_interval'] == 50.0 and config['budget']['dlogz'] == 0.05
        assert config['run_input']['points'] == [42, 42]
        assert diag['exact_count_includes_full_history']
        input_hashes = [digest(folder / name) for name in ('input_w2.txt', 'input_response.txt')]
        if common_priors is None:
            common_priors, common_inputs = config['priors'], input_hashes
        assert common_priors == config['priors'] and common_inputs == input_hashes
        table = np.genfromtxt(folder / 'weighted_posterior.csv', names=True, delimiter=',')
        with np.load(folder / 'stratified_results.npz') as raw:
            assert np.array_equal(raw['samples'], np.column_stack([table[n] for n in config['parameter_names']]))
            for key, column in [('logl', 'log_likelihood'), ('conditional_logvol', 'conditional_log_prior_mass'),
                                ('stratum_id', 'stratum_id'), ('original_prior_logwt', 'original_prior_logweight'),
                                ('weights', 'weight')]:
                assert np.array_equal(raw[key], table[column]), (folder, key)
            assert np.allclose(raw['stratum_prior_mass'], expected_mass, rtol=0, atol=1e-16)
            weights = np.exp(raw['original_prior_logwt'] - logsumexp(raw['original_prior_logwt']))
            weight_error = float(np.max(np.abs(weights - raw['weights'])))
            assert weight_error < 1e-12
            max_weight_error = max(max_weight_error, weight_error)
            assert abs(float(logsumexp(raw['original_prior_logwt'])) - diag['raw_log_evidence']) < 1e-10
            for index in range(3):
                child = folder / 'strata' / f'stratum_{index:02d}'
                child_config = read(child / 'sampler_config.json')
                assert child_config['seed'] == config['child_seeds'][index]
                mask = raw['stratum_id'] == index
                with np.load(child / 'nested_results.npz') as cr:
                    for pk, ck in [('samples', 'samples'), ('logl', 'logl'),
                                   ('conditional_logvol', 'logvol'), ('conditional_logwt', 'logwt')]:
                        assert np.array_equal(raw[pk][mask], cr[ck]), (folder, index, pk)
                    error = float(np.max(np.abs(raw['original_prior_logwt'][mask] -
                                                (cr['logwt'] + np.log(expected_mass[index])))))
                    max_logwt_error = max(max_logwt_error, error)
                    assert error < 1e-12
                files += [child / name for name in ('sampler_config.json', 'nested_results.npz', 'diagnostics.json')]
            assert np.allclose(np.bincount(raw['stratum_id'], weights=weights),
                               raw['stratum_posterior_mass'], rtol=0, atol=1e-12)
        files += [folder / name for name in required]
        runs.append(dict(path=relative(folder), seed=config['seed'], nlive_per_stratum=config['budget']['nlive'],
                         total_initial_live_points=config['total_initial_live_points'],
                         scheduler_pid=state['pid'], scheduler_pid_exited=True,
                         exact_likelihood_calls=diag['exact_likelihood_calls_total'],
                         dynesty_reported_ncall=diag['dynesty_reported_ncall'],
                         logz=diag['raw_log_evidence'], logzerr=diag['raw_log_evidence_error'],
                         stratum_posterior_masses=[s['posterior_mass'] for s in diag['strata']]))
    source_path = ROOT / 'source_fix1_manifest.json'
    for name, expected in read(source_path)['source_sha256'].items():
        path = ARCHIVE / name
        assert digest(path) == expected, path
        files.append(path)
    comparisons = []
    for comp in group['comparisons']:
        comparisons.append({key: comp[key] for key in ('kind', 'seed_a', 'seed_b', 'effort_a',
                            'effort_b', 'passed', 'failed_parameters', 'failed_branches')})
        comparisons[-1]['short_component_pass'] = {n: comp['parameters'][n]['passed'] for n in ('mu0', 'A0', 'w0')}
        comparisons[-1]['total_tf_pass'] = comp['total_tf']['passed']
    created = datetime.now(timezone.utc).isoformat()
    summary = dict(created_at_utc=created, scope='six shared/shared global parent posteriors only',
                   status=group['status'], design_issues=group['design_issues'],
                   weight_resolution_passed=group['weight_resolution_passed'],
                   passing_comparisons=sum(c['passed'] for c in comparisons), comparison_count=6,
                   comparisons=comparisons, overall_goal_complete=False, active_qmc_included=False,
                   short_component_pass_counts={n: sum(c['short_component_pass'][n] for c in comparisons)
                                                for n in ('mu0', 'A0', 'w0')},
                   total_tf_pass_count=sum(c['total_tf_pass'] for c in comparisons),
                   physical_recovery_assessed=False)
    save(destinations['phase_summary.json'], summary)
    lines = ['# V3 stratified shared/shared phase', '',
             'Scope: six completed two-band parent posteriors, three seeds × 400/800 live points per each of three strata. '
             'The full original prior is retained; conditional child runs are integration pieces, not independent global fits.', '',
             f"Frozen formal gate: **{summary['status']}**; {summary['passing_comparisons']}/6 complete comparisons pass. "
             f"Weight resolution: {group['weight_resolution_passed']}. No gate threshold was changed.", '',
             '| Comparison | Full pass | Failed parameters | mu0/A0/w0 | Total TF |',
             '|---|---|---|---|---|']
    for c in comparisons:
        label = f"{c['seed_a']}:{int(c['effort_a'])} → {c['seed_b']}:{int(c['effort_b'])}"
        short = '/'.join('pass' if c['short_component_pass'][n] else 'fail' for n in ('mu0', 'A0', 'w0'))
        lines.append(f"| {label} | {c['passed']} | {', '.join(c['failed_parameters']) or 'none'} | {short} | {c['total_tf_pass']} |")
    lines += ['', 'Numerical stability and injected-truth recovery are different questions. This record assesses stability only; '
              'it does not establish coverage, disk/BLR identity, distance or H0 recovery. The full 4×3 matrix is unfinished. '
              'The active large QMC run is excluded.', '',
              'Native parent/child sample alignment, original-prior mass weighting and CSV exports were independently checked. '
              'All six completion markers are present and recorded scheduler PIDs have exited. '
              'Evidence errors are conditional delta-method estimates and do not include missed modes or sampling bias.', '']
    destinations['phase_report.md'].write_text('\n'.join(lines), encoding='utf-8')
    manifest = dict(created_at_utc=created, completed_runs=runs, completed_count=6,
                    completed_case_mode_pairs=['shared:shared'], overall_goal_complete=False,
                    active_qmc_included=False, formal_gate=relative(gate_path),
                    exact_likelihood_calls_sum=sum(r['exact_likelihood_calls'] for r in runs),
                    dynesty_reported_ncall_sum=sum(r['dynesty_reported_ncall'] for r in runs),
                    exact_count_scope='checkpoint-retained child histories including initialization, excluding discarded post-checkpoint work',
                    maximum_stored_weight_error=max_weight_error, maximum_logmass_merge_error=max_logwt_error)
    save(destinations['phase_dynesty_v3_manifest.json'], manifest)
    task = destinations['phase_dynesty_v3_task.md']
    task.write_text('Assess and record only the six completed original-prior v3 shared/shared parent posteriors using the '
                    'predeclared exact-directory formal gate. Preserve failures and verify native conditional-to-parent weighting. '
                    'No new fit, active QMC result or full-matrix completion claim.\n', encoding='utf-8')
    record = read(BASE / 'sci001/output/mica_slowgp/stability_v2_rslice/phase_dynesty_v2_computation_record.json')
    record['created_at_utc'] = created
    record['phase_scope'] = dict(name='v3 stratified shared/shared stability phase', completed_fit_count=6,
        completed_case_mode_pairs=['shared:shared'], representative_only=True,
        full_case_mode_matrix_complete=False, overall_goal_complete=False, active_qmc_included=False)
    record['task'] = dict(file=relative(task), sha256=digest(task), object=record['task']['object'],
                         deliverables=['Exact-six formal gate', 'Native weighting checks', 'Phase report and computation record'])
    record['implementation_discovery']['python']['candidate_implementations'] = ['Frozen v3-fix1 stratified dynesty rslice integration']
    record['decision']['selection_reason'] = 'Partition the unchanged ordered-center prior, integrate each conditional prior, merge with exact prior masses'
    record['execution'].update(command_or_input='env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ' + ' '.join(plan['command']),
        code_artifact=relative(TEST / 'assess_stability.py'),
        completion_scope='Formal assessment and independent record of six completed v3 shared/shared parent fits only')
    record['result'] = dict(status='complete', summary=f"Six completed parents assessed: {group['status']}; no recovery or full-matrix claim.",
        result_artifact=relative(destinations['phase_report.md']), stability_gate_passed=group['all_distribution_comparisons_passed'],
        overall_goal_complete=False)
    record['verification'] = dict(methods=['Unchanged exact-six weighted-distribution, branch and total-TF gate',
        'Completion marker and scheduler PID exit checks', 'Parent/child native row alignment and log prior-mass merge',
        'CSV physical rows, logL, conditional logvol, stratum labels and weights matched native arrays',
        'Unchanged common inputs and priors; frozen source digests checked', 'Computation-record schema and artifact digest validation'],
        evidence_level='numerical-evidence', residual_or_error=f'Max native/export weight error {max_weight_error}; max logmass merge error {max_logwt_error}',
        limitations=['Only shared/shared; full 4x3 matrix incomplete', 'Repeated seeds use one data realization; not coverage trials',
        'Kish ESS does not establish independent posterior information or convergence',
        'Evidence delta-method errors exclude missed modes and conditional sampling bias',
        'Callback counts exclude discarded post-checkpoint work; parent elapsed time excludes reused-child computation',
        'No physical recovery, disk identity, distance, H0 or evidence-based model comparison claim', 'Active large QMC results excluded'])
    files += [gate_path, source_path, ROOT / 'assessment_plan.json', ROOT / 'algorithm_settings.json',
              ROOT / 'formal_assessment.log', TEST / 'assess_stability.py', TEST / 'stability_gate_v1.json', Path(__file__)]
    files += [path for name, path in destinations.items() if name != 'phase_dynesty_v3_computation_record.json']
    record['artifacts'] = [dict(role='code' if p.suffix in ('.py', '.pyx', '.so') else 'result',
                               path=relative(p), sha256=digest(p)) for p in sorted(set(files))]
    record['note'] = 'Bounded v3 phase is complete; preserved gate failures remain. Schema validation is not convergence evidence.'
    save(destinations['phase_dynesty_v3_computation_record.json'], record)
    print(json.dumps(dict(status=summary['status'], passing_comparisons=summary['passing_comparisons'],
                         artifacts=len(record['artifacts']), manifest=relative(destinations['phase_dynesty_v3_manifest.json']))))


if __name__ == '__main__':
    main()
