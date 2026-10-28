#!/usr/bin/env python3
"""Record the completed v2 rslice shared/shared phase without rerunning fits.

Requires the final six-run gate and phase report. Writes only new v2 records;
v1 and the ongoing v3 stratified experiment are outside the artifact set.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

BASE = Path('/home/leonhard')
ROOT = BASE / 'sci001/output/mica_slowgp/stability_v2_rslice'
TEST = BASE / 'sci001/test/mica_slowgp'
ARCHIVE = BASE / 'software/MICA2-slowgp/archive/stability-v2-rslice'
SEEDS = (71231, 71247, 71263)
BUDGETS = (400, 800)


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
    manifest_path = ROOT / 'phase_dynesty_v2_manifest.json'
    task_path = ROOT / 'phase_dynesty_v2_task.md'
    record_path = ROOT / 'phase_dynesty_v2_computation_record.json'
    for path in (manifest_path, task_path, record_path):
        if path.exists():
            raise FileExistsError(f'Refusing to replace an existing phase record: {path}')
    gate_path = ROOT / 'representative_dynesty_shared.json'
    report_path = ROOT / 'phase_report.md'
    summary_path = ROOT / 'phase_summary.json'
    assert report_path.is_file() and summary_path.is_file(), 'Final phase artifacts are required'
    gate = read(gate_path)
    assert not gate['rejected_runs'] and len(gate['groups']) == 1
    group = gate['groups'][0]
    assert (group['case'], group['mode'], group['sampler']) == ('shared', 'shared', 'dynesty')
    assert not group['design_issues'] and len(group['runs']) == 6
    assert len(group['comparisons']) == 6
    expected_paths = {str(ROOT / 'dynesty/shared_shared' / f'budget_{budget}' / f'seed_{seed}')
                      for budget in BUDGETS for seed in SEEDS}
    assert {item['path'] for item in group['runs']} == expected_paths

    completed, files, maximum_weight_error = [], [], 0.0
    required = ('sampler_config.json', 'run_input.json', 'input_w2.txt', 'input_response.txt',
                'weighted_posterior.csv', 'nested_results.npz', 'posterior_summary.json',
                'diagnostics.json')
    common_priors = common_inputs = None
    for budget in BUDGETS:
        for seed in SEEDS:
            folder = ROOT / 'dynesty/shared_shared' / f'budget_{budget}' / f'seed_{seed}'
            assert all((folder / name).is_file() for name in required), folder
            config = read(folder / 'sampler_config.json')
            diagnostics = read(folder / 'diagnostics.json')
            run_input = read(folder / 'run_input.json')
            assert config['sampler'] == 'dynesty' and config['mode'] == 'shared'
            assert config['seed'] == seed and config['budget']['nlive'] == budget
            assert config['budget']['sample'] == 'rslice' and config['budget']['slices'] == 14
            assert type(config['budget']['update_interval']) is float
            assert config['budget']['update_interval'] == 50.0
            assert config['budget']['dlogz'] == 0.05
            assert run_input['points'] == [42, 42] and run_input['likelihood_backend'] == 'compiled'
            assert diagnostics['exact_count_includes_full_history']
            assert isinstance(diagnostics['exact_likelihood_calls_total'], int)
            effective = diagnostics['sampler_effective']
            assert effective['internal_sampler_class'] == 'RSliceSampler'
            assert effective['tuning']['slices'] == 14
            assert effective['bound_update_interval_reported_calls'] == 50 * budget
            input_hashes = [digest(folder / name) for name in ('input_w2.txt', 'input_response.txt')]
            if common_priors is None:
                common_priors, common_inputs = config['priors'], input_hashes
            assert config['priors'] == common_priors and input_hashes == common_inputs
            table = np.genfromtxt(folder / 'weighted_posterior.csv', names=True, delimiter=',')
            with np.load(folder / 'nested_results.npz') as raw:
                weights = np.exp(raw['logwt'] - logsumexp(raw['logwt']))
                error = float(np.max(np.abs(weights - table['weight'])))
                assert error < 1e-12
                assert np.array_equal(raw['samples'], np.column_stack([table[n] for n in config['parameter_names']]))
                assert np.array_equal(raw['logl'], table['log_likelihood'])
                assert np.array_equal(raw['logvol'], table['log_prior_mass'])
                assert abs(float(logsumexp(raw['logwt'])) - diagnostics['raw_log_evidence']) < 1e-10
            maximum_weight_error = max(maximum_weight_error, error)
            state_path = folder / 'scheduler_state.json'
            scheduler_state = read(state_path).get('status') if state_path.is_file() else None
            completed.append(dict(
                path=relative(folder), relative_run_path=str(folder.relative_to(ROOT/'dynesty')),
                case='shared', mode='shared', seed=seed, nlive=budget,
                scheduler_status_at_record=scheduler_state,
                completion_evidence='Final native arrays, exported posterior, diagnostics and six-run formal assessment',
                elapsed_seconds=diagnostics['elapsed_seconds'],
                elapsed_seconds_scope=diagnostics['elapsed_seconds_scope'],
                dynesty_reported_ncall=diagnostics['dynesty_reported_ncall'],
                exact_likelihood_calls=diagnostics['exact_likelihood_calls_total'],
                exact_count_scope=diagnostics['exact_count_scope'],
                kish_weight_ess=diagnostics['weight_ess'],
                logz=diagnostics['raw_log_evidence'], logzerr=diagnostics['raw_log_evidence_error']))
            files.extend(folder / name for name in required)
            # Scheduler metadata can outlive its process; retain the observed state,
            # never rewrite it to imply completion. Hash only settled fit products.
    snapshot_path = ROOT / 'source_snapshot.json'
    for item in read(snapshot_path)['files']:
        path = Path(item['path'])
        assert path.is_relative_to(ARCHIVE) and digest(path) == item['sha256']
        files.append(path)
    created = datetime.now(timezone.utc).isoformat()
    manifest = dict(
        created_at_utc=created, phase='v2 rslice shared/shared only', completed_runs=completed,
        completed_count=6, completed_case_mode_pairs=['shared:shared'],
        full_case_mode_matrix_complete=False, overall_goal_complete=False,
        v3_stratified_artifacts_included=False,
        original_prior_and_likelihood_unchanged=True,
        formal_gate_status=group['status'],
        formal_gate_all_distribution_comparisons_passed=group['all_distribution_comparisons_passed'],
        passing_comparisons=sum(bool(item['passed']) for item in group['comparisons']),
        comparison_count=6,
        recorded_run_nested_seconds_sum=sum(run['elapsed_seconds'] for run in completed),
        elapsed_seconds_sum_is_aggregate_not_wall_clock=True,
        dynesty_reported_ncall_sum=sum(run['dynesty_reported_ncall'] for run in completed),
        exact_likelihood_calls_sum=sum(run['exact_likelihood_calls'] for run in completed),
        exact_count_scope='Retained checkpoint trajectories, including fresh initialization; discarded post-checkpoint work is excluded',
        maximum_stored_weight_error=maximum_weight_error,
        phase_report=relative(report_path), formal_gate=relative(gate_path))
    save(manifest_path, manifest)
    with task_path.open('x', encoding='utf-8') as stream:
        stream.write('Record the completed two-band shared/shared dynesty rslice phase: '
                     'three fixed seeds and 400/800 live points, unchanged data, model, priors and '
                     'stability gates. Preserve failures. This phase is one of twelve case/model '
                     'combinations, not completion of the overall goal. Exclude all v3 runs.\n')

    record = read(BASE/'sci001/output/mica_slowgp/stability_v1/phase_dynesty_v1_computation_record.json')
    record['created_at_utc'] = created
    record['phase_scope'] = dict(name='dynesty v2 rslice shared/shared stability phase',
        completed_fit_count=6, completed_case_mode_pairs=['shared:shared'], representative_only=True,
        full_case_mode_matrix_complete=False, overall_goal_complete=False, v3_artifacts_included=False)
    record['task'] = dict(file=relative(task_path), sha256=digest(task_path),
        object='Posterior stability of the unchanged pairwise OU, ordered double-Gaussian and shared slow-GP model',
        deliverables=['Six completed rslice results', 'Unchanged formal gate assessment',
                      'Phase report and run manifest', 'Validated computation record'])
    record['mathematical_context']['assumptions'][3] = 'Shared zero-lag slow process has nonnegative gains qW,qb and common Ts'
    record['implementation_discovery']['python'].update(
        candidate_implementations=['Frozen v2 NumPy/SciPy covariance and dynesty RSliceSampler'],
        existence_evidence='Six completed task-native scientific fits and saved native-array/export comparisons',
        availability_evidence='Six completed dynesty 3.1.0 runs in the isolated mica2-slowgp environment')
    record['decision']['selection_reason'] = 'Same-target random-direction slice sampling in the frozen task-native Python implementation'
    command = record['execution']['command_or_input']
    command = command.replace('archive/stability-v1/src', 'archive/stability-v2-rslice/src')
    command = command.replace('stability_v1/phase_dynesty_v1_manifest', 'stability_v2_rslice/phase_dynesty_v2_manifest')
    command = command.replace('replay_dynesty_v1', 'replay_dynesty_v2_rslice')
    record['execution'].update(command_or_input=command,
        code_artifact=relative(ARCHIVE/'src/mica_slowgp/nested.py'),
        completion_scope='Only six finished shared/shared rslice runs and their assessments; the overall goal remains active')
    record['result'] = dict(status='complete',
        summary=f"Six shared/shared rslice runs assessed; formal group status: {group['status']}. No overall matrix completion or physical recovery is asserted.",
        result_artifact=relative(report_path), stability_gate_passed=group['all_distribution_comparisons_passed'],
        overall_goal_complete=False)
    record['verification'] = dict(methods=[
        'Three-seed and two-budget frozen weighted-distribution, branch and total-TF gate',
        'Exported physical samples, logL, logvol and weights compared directly with native dynesty arrays',
        'Identical proper priors and two input files checked across all six runs',
        'Frozen v2 source digests compared with the pre-run source snapshot',
        'Artifact schema, paths and SHA-256 digest validation'],
        evidence_level='numerical-evidence',
        residual_or_error=f'Maximum native/export weight difference {maximum_weight_error:.3g}; full formal failures retained in the gate JSON',
        limitations=['Only shared/shared: one of twelve case/model combinations',
            'Repeated seeds use one data realization; these are not coverage trials',
            'Kish weight ESS is not an independent posterior sample count',
            'Exact callback counts cover retained checkpoint trajectories, not discarded post-checkpoint work',
            'Elapsed seconds are summed run_nested invocation timings, not experiment wall time',
            'Supplementary mu1-stratum probabilities are exploratory diagnostics, not new gate thresholds',
            'No evidence-based model selection, disk/BLR identity, distance or H0 claim',
            'All v3 stratified runs and the unfinished full matrix are excluded'])
    files += [manifest_path, task_path, gate_path, report_path, summary_path, snapshot_path,
              ROOT/'experiment_plan.json', ROOT/'algorithm_settings.json', Path(__file__),
              TEST/'assess_stability.py', TEST/'stability_gate_v1.json',
              BASE/'sci001/output/mica_slowgp/stability_v1/fast_validation.json']
    record['artifacts'] = [dict(role='code' if p.suffix in ('.py', '.pyx', '.so') else 'result',
                               path=relative(p), sha256=digest(p)) for p in sorted(set(files))]
    record['note'] = 'Completed v2 shared/shared results only; v1 records preserved and all v3 artifacts excluded. Schema validation does not establish convergence.'
    save(record_path, record)
    print(json.dumps(dict(record=str(record_path), manifest=str(manifest_path),
        completed_runs=6, artifacts=len(record['artifacts']),
        maximum_stored_weight_error=maximum_weight_error,
        exact_likelihood_calls_sum=manifest['exact_likelihood_calls_sum'])))


if __name__ == '__main__':
    main()
