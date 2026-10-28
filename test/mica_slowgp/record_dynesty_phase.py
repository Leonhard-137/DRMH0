#!/usr/bin/env python3
"""Archive the completed v1 rwalk phase; unfinished v2 fits are excluded."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

BASE = Path('/home/leonhard')
ROOT = BASE / 'sci001/output/mica_slowgp/stability_v1'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def relative(path):
    return str(path.relative_to(BASE))


def main():
    groups = {}
    for mode in ('independent', 'shared'):
        report = read(ROOT / f'representative_dynesty_{mode}.json')
        assert not report['rejected_runs']
        group = report['groups'][0]
        assert not group['design_issues'] and len(group['runs']) == 6
        assert len(group['comparisons']) == 6
        groups[mode] = group
    figure_files = [ROOT / f'stability_dynesty.{suffix}' for suffix in ('png', 'pdf', 'json')]
    assert all(path.is_file() for path in figure_files), 'Final figure is not complete'

    completed, files, maximum_weight_error = [], [], 0.0
    required = ('sampler_config.json', 'run_input.json', 'input_w2.txt', 'input_response.txt',
                'weighted_posterior.csv', 'nested_results.npz', 'posterior_summary.json',
                'diagnostics.json')
    for mode in ('independent', 'shared'):
        for budget in (400, 800):
            for seed in (71231, 71247, 71263):
                folder = ROOT / 'dynesty' / f'{mode}_{mode}' / f'budget_{budget}' / f'seed_{seed}'
                assert all((folder / name).is_file() for name in required)
                config, diagnostics = read(folder / 'sampler_config.json'), read(folder / 'diagnostics.json')
                if (folder / 'scheduler_state.json').exists():
                    state = read(folder / 'scheduler_state.json')
                    assert state['status'] == 'complete'
                assert config['seed'] == seed and config['budget']['nlive'] == budget
                assert config['budget']['sample'] == 'rwalk' and config['budget']['walks'] == 50
                table = np.genfromtxt(folder / 'weighted_posterior.csv', names=True, delimiter=',')
                with np.load(folder / 'nested_results.npz') as raw:
                    weights = np.exp(raw['logwt'] - logsumexp(raw['logwt']))
                    error = float(np.max(np.abs(weights - table['weight'])))
                    assert error < 1e-12
                    assert np.array_equal(raw['logl'], table['log_likelihood'])
                maximum_weight_error = max(maximum_weight_error, error)
                completed.append(dict(path=relative(folder), relative_run_path=str(folder.relative_to(ROOT/'dynesty')),
                                      mode=mode, seed=seed, nlive=budget,
                                      elapsed_seconds=diagnostics['elapsed_seconds'],
                                      dynesty_reported_ncall=diagnostics['likelihood_calls'],
                                      exact_likelihood_calls=None,
                                      kish_weight_ess=diagnostics['weight_ess']))
                files.extend(folder / name for name in required)
                files.extend(folder / name for name in ('fit.log', 'scheduler_state.json')
                             if (folder / name).is_file())
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), completed_runs=completed,
                    completed_count=len(completed), overall_goal_complete=False,
                    dynesty_reported_ncall_is_not_exact_callback_count=True,
                    recorded_run_nested_seconds_sum=sum(run['elapsed_seconds'] for run in completed),
                    reported_ncall_sum=sum(run['dynesty_reported_ncall'] for run in completed),
                    maximum_stored_weight_error=maximum_weight_error)
    manifest_path = ROOT / 'phase_dynesty_v1_manifest.json'
    save(manifest_path, manifest)

    rows = []
    for mode, group in groups.items():
        comparisons = group['comparisons']
        final = [item for item in comparisons if item['kind'] == 'independent_seed_final_budget']
        budget = [item for item in comparisons if item['kind'] == 'within_seed_budget_extension']
        rows.append(f"| {mode} | {sum(item['passed'] for item in comparisons)}/6 | "
                    f"{sum(item['total_tf']['passed'] for item in final)}/3 | "
                    f"{sum(item['total_tf']['passed'] for item in budget)}/3 | {group['status']} |")
    report_path = ROOT / 'phase_dynesty_v1_report.md'
    with report_path.open('x') as stream:
        stream.write('# 两波段慢信号模型：独立采样器复核\n\n'
                     '本阶段完成 12 次 dynesty rwalk 拟合：独立慢信号和共享慢信号各 3 个种子、'
                     '400/800 两档 live points。每次仅有 W2 和一个响应波段，各 42 个点。'
                     '模型、宽先验、数据与预先冻结的稳定性门槛保持一致。\n\n'
                     '| 模型 | 完整分布比较通过 | 高预算跨种子总 TF 通过 | 预算加倍总 TF 通过 | 判定 |\n'
                     '|---|---:|---:|---:|---|\n' + '\n'.join(rows) + '\n\n'
                     '独立慢信号组高预算下的总转移函数已较一致，但长分量幅度 A1 的主体分布仍有漂移：'
                     '种子 71231 与 71247 的 KS=0.1494、标准化 W1=0.2281。'
                     '条件于 μ1≥5 的主分支后差异仍在，不能全部归因于小概率分支。'
                     '部分延迟差异则主要来自 q05/q95 尾部；近零负延迟面积的标准化指标也较敏感。'
                     '完整参数、分支与曲线指标保存在两个正式 gate JSON 中。\n\n'
                     'Kish 权重 ESS 只衡量权重集中程度，不是独立后验样本数。'
                     '旧版本 diagnostics 的 likelihood_calls 是 dynesty 报告的 ncall，包含部分非实际似然调用，'
                     '本阶段不把它报告为精确计算次数。新版本已加入 checkpoint 保留历史的实际 callback 计数。\n\n'
                     '本记录只覆盖已结束的 rwalk 阶段。正在另一冻结副本测试 rslice；'
                     '沿退化方向联合更新的 DNest 候选已通过先验保持检查，但尚未运行科学拟合。'
                     '后续仍按两波段处理，不做六波段联合；原 MICA 未修改。\n\n'
                     '本阶段完成不代表总体目标完成，也不代表短分量已经获得物理盘响应身份、'
                     '覆盖率验证或无偏距离/H0。四种注入情况乘三种拟合模型的完整稳定性验证仍待完成。\n')
    task_path = ROOT / 'phase_dynesty_v1_task.md'
    with task_path.open('x') as stream:
        stream.write('Complete and assess the two representative pairwise slow-GP rwalk sampling experiments '
                     'at three prespecified seeds and two budgets, retaining failed stability outcomes. '
                     'This is a bounded stage of the ongoing stable-algorithm goal.\n')

    record = read(ROOT / 'phase_dnest_v1_computation_record.json')
    record['created_at_utc'] = manifest['created_at_utc']
    record['mathematical_context']['precision']['target_tolerance'] = 'Fast/reference log-likelihood tolerance 1e-8 from completed shared-core validation; unchanged slowgp-stability-v1 thresholds; stored dynesty weight agreement 1e-12'
    record['phase_scope'] = dict(name='dynesty v1 rwalk representative stability phase',
                                 completed_fit_count=12, representative_only=True, overall_goal_complete=False,
                                 unfinished_rslice_runs_included=False)
    record['task'] = dict(file=relative(task_path), sha256=hashlib.sha256(task_path.read_bytes()).hexdigest(),
                          object='Posterior stability of the same pairwise OU, double-Gaussian and slow-GP model',
                          deliverables=['12 completed rwalk results', 'Fixed-gate assessments', 'Weighted comparison figure', 'Validated computation record'])
    record['implementation_discovery']['python'] = dict(
        candidate_implementations=['Archived NumPy/SciPy covariance and dynesty nested sampler'],
        existence_evidence='Same-target prior-transform checks and twelve completed scientific fits',
        local_availability='available',
        availability_evidence='Twelve completed dynesty 3.1.0 runs in the isolated mica2-slowgp environment')
    for backend in ('mathematica', 'sagemath'):
        record['implementation_discovery'][backend]['availability_evidence'] = 'No CAS was needed to assess the completed task-native Python sampler outputs'
    record['decision'] = dict(selected_backend='python', backend_version='Python 3.11.15; NumPy 2.4.3; SciPy 1.17.1; dynesty 3.1.0',
                               selection_reason='Independent nested sampler using the identical likelihood and normalized physical prior',
                               fallback_reason='not-required: actual task-native Python fits completed')
    command = (record['execution']['command_or_input']
               .replace('run_dnest', 'run_nested').replace('mica_slowgp.sampler', 'mica_slowgp.nested')
               .replace('phase_dnest_v1_manifest', 'phase_dynesty_v1_manifest')
               .replace('replay_dnest_v1', 'replay_dynesty_v1'))
    record['execution'] = dict(status='complete', interface='Isolated WSL processes, one numerical-library thread per fit; at most four simultaneous fits',
                               command_or_input=command, code_artifact='software/MICA2-slowgp/archive/stability-v1/src/mica_slowgp/nested.py',
                               completion_scope='Only the twelve finished rwalk runs and their assessments; overall goal remains active')
    record['result'] = dict(status='complete', summary='Twelve representative runs completed and were evaluated against unchanged gates. See per-group outcomes; no overall stability or physical recovery is asserted.',
                            result_artifact=relative(report_path), stability_gate_passed=all(group['all_distribution_comparisons_passed'] for group in groups.values()), overall_goal_complete=False)
    record['verification'] = dict(methods=['Three-seed and two-budget frozen weighted-distribution, branch and TF gate',
                                          'Direct comparison of exported weights/log likelihoods with saved native dynesty arrays',
                                          'Visual inspection of the completed twelve-run figure', 'Previously completed same-target likelihood and prior-transform checks',
                                          'Artifact schema, path and digest validation'],
                                   evidence_level='numerical-evidence', residual_or_error=f'Maximum native-array/exported weight difference {maximum_weight_error:.3g}; detailed gate outcomes in attached JSON files',
                                   limitations=['This finite phase is complete; the overall algorithm-stability goal is unfinished',
                                                'Only two of twelve case/model combinations', 'Seeds repeat one data realization, not coverage trials',
                                                'Kish ESS is not an independent sample count', 'Reported ncall is not an exact callback count for this old phase',
                                                'No calibrated evidence-based model selection, disk/BLR identity, distance or H0 claim',
                                                'Unfinished rslice fits are excluded'])
    files += [manifest_path, report_path, task_path, *figure_files,
              ROOT/'representative_dynesty_independent.json', ROOT/'representative_dynesty_shared.json',
              ROOT/'representative_dynesty_summary.json',
              ROOT/'representative_dynesty400_parameter_diagnostic.json', ROOT/'fast_validation.json', ROOT/'isolation_check.json']
    files += list((BASE/'software/MICA2-slowgp/archive/stability-v1').rglob('*'))
    artifacts = []
    for path in sorted(set(files)):
        if not path.is_file():
            continue
        role = 'code' if path.suffix in ('.py', '.pyx', '.so') else 'result'
        artifacts.append(dict(role=role, path=relative(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    record['artifacts'] = artifacts
    record['note'] = 'Completed v1 results only. The root pilot record and DNest phase record are preserved. Current candidate runs are excluded.'
    save(ROOT/'phase_dynesty_v1_computation_record.json', record)
    print(json.dumps(dict(report=str(report_path), artifact_count=len(artifacts), completed_runs=12,
                          maximum_stored_weight_error=maximum_weight_error)))


if __name__ == '__main__':
    main()
