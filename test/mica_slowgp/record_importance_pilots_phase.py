#!/usr/bin/env python3
"""Validate and record two finished IS efficiency pilots; never call a GP."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

import numpy as np
from scipy.special import logsumexp, ndtr

BASE = Path('/home/leonhard')
SCI = BASE/'sci001'
ROOT = SCI/'output/mica_slowgp'
OUT = ROOT/'importance_pilots_phase'
RUNS = [('v1', ROOT/'importance_pilot_v1/seed_82103_n16384'),
        ('v2_regularized', ROOT/'importance_pilot_v2_regularized/seed_82119_n16384')]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p):
    return str(p.relative_to(BASE))


def save(p, value):
    with p.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False)+'\n')


def load(p):
    with np.load(p, allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    artifacts, entries, commands = {}, [], []
    target_contract = None
    for label, folder in RUNS:
        assert (folder/'completed.json').is_file(), f'Incomplete pilot: {folder}'
        config, diagnostics = read(folder/'sampler_config.json'), read(folder/'diagnostics.json')
        assert not Path('/proc', str(config['pid'])).exists(), 'Recorded pilot PID still exists'
        assert config['case'] == config['mode'] == 'shared'
        assert config['budget']['draws_per_stratum'] == 16384
        contract = [config['parameter_names'], config['priors'],
                    sha(folder/'input_w2.txt'), sha(folder/'input_response.txt'), config['source_root']]
        if target_contract is None:
            target_contract = contract
        assert contract == target_contract
        for item in config['proposal_files']+[config['proposal_module']]:
            assert sha(Path(item['path'])) == item['sha256']
        mass = np.array([64.,465.,432.])/961
        np.testing.assert_array_equal(config['stratum_prior_masses'], mass)
        rows = [load(folder/f'stratum_{j:02d}.npz') for j in range(3)]
        max_identity_error = 0.
        for j,r in enumerate(rows):
            assert len(r['logweight']) == 16384 and r['samples'].shape == (16384,11)
            assert np.isfinite(r['logweight']).all()
            error = float(np.max(abs(r['logweight']-(r['logl']+r['logp_z']-r['logq_z']))))
            assert error < 1e-12
            max_identity_error = max(max_identity_error,error)
            logp = np.sum(-np.logaddexp(0.,r['z'])-np.logaddexp(0.,-r['z']),axis=1)
            np.testing.assert_array_equal(logp,r['logp_z'])
            assert abs(logsumexp(r['logweight'])-np.log(16384)-diagnostics['layers'][j]['logz']) < 1e-12
        contribution = np.concatenate([np.log(p)+r['logweight']-np.log(len(r['logweight']))
                                       for p,r in zip(mass,rows)])
        logz = float(logsumexp(contribution))
        weights = np.exp(contribution-logz)
        physical = np.concatenate([r['samples'] for r in rows])
        logl = np.concatenate([r['logl'] for r in rows])
        ids = np.repeat(np.arange(3),16384)
        parent = load(folder/'importance_results.npz')
        table = np.genfromtxt(folder/'weighted_posterior.csv',names=True,delimiter=',')
        for key,expected in [('samples',physical),('logl',logl),('weights',weights),('stratum_id',ids),
                             ('original_prior_log_contribution',contribution)]:
            np.testing.assert_array_equal(parent[key],expected)
        np.testing.assert_array_equal(np.column_stack([table[n] for n in config['parameter_names']]),physical)
        for key,expected in [('log_likelihood',logl),('stratum_id',ids),('log_contribution',contribution),('weight',weights)]:
            np.testing.assert_array_equal(table[key],expected)
        assert abs(logz-diagnostics['logz']) < 1e-12
        # Independent longdouble moment arithmetic; no likelihood evaluations.
        shift = max(r['logweight'].max() for r in rows)
        scaled = [np.exp(np.asarray(r['logweight']-shift,dtype=np.longdouble)) for r in rows]
        zscaled = sum(np.longdouble(p)*a.mean() for p,a in zip(mass,scaled))
        vscaled = sum(np.longdouble(p)**2*np.var(a,ddof=1)/len(a) for p,a in zip(mass,scaled))
        rse = float(np.sqrt(vscaled)/zscaled)
        np.testing.assert_allclose(rse,diagnostics['relative_evidence_mcse'],rtol=2e-12,atol=1e-15)
        c = dict(zip(config['parameter_names'],physical.T))
        f = (c['A0']*ndtr(-c['mu0']/c['w0'])+c['A1']*ndtr(-c['mu1']/c['w1']))/(c['A0']+c['A1'])
        mean = float(np.dot(weights,f))
        vf = sum(np.longdouble(p)**2*np.var(a*(f[ids==j]-mean),ddof=1)/len(a)
                 for j,(p,a) in enumerate(zip(mass,scaled)))/zscaled**2
        fse = float(np.sqrt(vf))
        stored = diagnostics['summaries']['total_negative_lag_fraction']
        np.testing.assert_allclose([mean,fse],[stored['mean'],stored['mean_mcse']],rtol=2e-12,atol=1e-15)
        assert diagnostics['scientific_likelihood_evaluations'] == len(weights) == 49152
        entries.append(dict(label=label,path=rel(folder),seed=config['seed'],
            completed_marker_present=True,recorded_pid_exited=True,
            evaluation_draws=49152,draws_per_stratum=16384,
            precision_audit_GP_calls=diagnostics['precision_validation_likelihood_evaluations'],
            seconds=diagnostics['seconds'],logz=logz,relative_evidence_mcse=rse,
            negative_lag_fraction_mean=mean,negative_lag_fraction_mcse=fse,
            weight_kish=diagnostics['weight_kish'],maximum_row_weight=float(weights.max()),
            top_one_percent_row_weight=diagnostics['top_one_percent_row_weight'],
            stratum_posterior_mass=diagnostics['stratum_posterior_mass'],
            raw_logweight_identity_max_absolute_error=max_identity_error,
            parent_and_CSV_array_equality=True,
            zero_or_underflow_weight_rows=int(np.sum(weights==0)),all_evaluation_rows_retained=True))
        for p in folder.iterdir():
            if p.is_file():
                artifacts[p]='result'
        archive = Path(config['proposal_module']['path']).parent
        for p in archive.iterdir():
            if p.is_file():
                artifacts[p]='code' if p.suffix=='.py' else 'proposal-artifact'
        for p in Path(config['source_root']).glob('mica_slowgp/*'):
            if p.is_file():
                artifacts[p]='code' if p.suffix in ('.py','.pyx','.so') else 'source-artifact'
        argv = ['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',
            '/home/leonhard/miniforge3/envs/mica2-slowgp/bin/python','-B',str(archive/'run_importance_pilot.py'),
            '--source-root',config['source_root'],'--proposal-module-root',str(archive),
            '--inputs-root',str(ROOT/'small_cases'),'--case','shared','--mode','shared',
            '--draws-per-stratum','16384','--seed',str(config['seed']),
            '--out',str(ROOT/'replay_importance_pilots'/label)]
        for p in config['proposal_files']:
            argv += ['--proposal',p['path']]
        commands.append(shlex.join(argv))
    # Training/regularization are costs, not additional scientific evaluations.
    cost_paths = [BASE/'software/MICA2-slowgp/archive/importance-pilot-v1/training_report.json',
        SCI/'test/mica_slowgp/staged_importance_refined/artifacts/trained/training_report.json',
        BASE/'software/MICA2-slowgp/archive/importance-pilot-v2-regularized/regularization_report.json']
    costs=[]
    for label,p in zip(('initial_proposal_training','refined_12_component_training','deterministic_covariance_regularization'),cost_paths):
        d=read(p);assert d.get('gp_likelihood_calls',d.get('GP_likelihood_calls'))==0
        costs.append(dict(stage=label,seconds=d['elapsed_seconds'],new_GP_calls=0,report=rel(p)))
        artifacts[p]='training-artifact'
    for p in (SCI/'test/mica_slowgp/staged_importance_refined').glob('*.py'):
        artifacts[p]='code'
    for p in (SCI/'test/mica_slowgp/staged_importance_refined/artifacts/trained').glob('stratum_*.npz'):
        artifacts[p]='training-artifact'
    for p in [SCI/'test/mica_slowgp/staged_importance_refined/artifacts/training_concentration.json',
              SCI/'test/mica_slowgp/staged_importance_regularized/independent_review.json',Path(__file__)]:
        artifacts[p]='code' if p.suffix=='.py' else 'verification-artifact'
    OUT.mkdir()
    manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),pilots=entries,
        scope='Two different proposals, one evaluation seed each; an efficiency pilot, not a stability gate',
        same_physical_target=True,training_costs=costs,
        refined_unregularized_candidate='12 components/block, EM max_iter100; trained only, not scientifically evaluated because of concentration',
        adaptation='The second proposal uses the first pilot as training data; the two pilots are not independent replicate estimates under one fixed proposal',
        raw_evaluation_rows_total=sum(e['evaluation_draws'] for e in entries),
        precision_audit_GP_calls_total=sum(e['precision_audit_GP_calls'] for e in entries),
        active_v3_results_included=False,full_4_by_3_matrix_complete=False,overall_goal_complete=False)
    save(OUT/'manifest.json',manifest)
    task=OUT/'task.md'
    task.write_text('Record and validate both completed frozen-proposal importance pilots, retaining all rows and outcomes. '
                    'Separate proposal-training cost and evaluation cost. This is not a stability gate or the complete case/model matrix.\n')
    rows='\n'.join(f"| {e['label']} | {e['negative_lag_fraction_mean']:.9g} | {e['negative_lag_fraction_mcse']:.9g} | {e['weight_kish']:.3f} | {e['relative_evidence_mcse']:.6f} | {e['seconds']:.3f} |" for e in entries)
    report=OUT/'phase_report.md'
    report.write_text('# 两个重要性采样效率 pilot 的计算记录\n\n'
        '两次均为 shared 输入、shared 慢信号模型，W2 加一个响应波段，各 42 点。每层抽取 16384 个新评价点，'
        '完整保留三层并按原先验质量合并。第一份是初始 proposal；第二份是利用旧 NS 和第一 pilot 训练、再协方差正则化的 proposal。\n\n'
        '| Proposal | 负延迟面积均值 | 条件 MCSE | Kish ESS | 证据相对 MCSE | 评价秒数 |\n'
        '|---|---:|---:|---:|---:|---:|\n'+rows+'\n\n'
        '表中是单次评价的描述性结果，MCSE 条件于各自冻结 proposal。第二 proposal 利用第一 pilot 训练，'
        '不能把两行当作同一 proposal 的独立重复，也不能据此建立稳定性或效率优劣的统计结论。'
        'Kish 衡量权重集中程度，不能证明漏模态已排除。未正则化的 12-component/max_iter100 候选只训练，因集中问题未作科学评价。\n\n'
        '两次逐层 logL+logpi-logq、原层质量/N 合并、父级 native arrays 和 CSV 均通过一致性核验。'
        '两次各 49152 次新评价调用，另各有精度审计调用；这些分开列于 manifest。训练、精细训练和确定性正则化的耗时亦分列；'
        '历史 NS 训练数据的获取成本不包含在这些训练秒数中。\n\n'
        '本阶段不运行稳定性 gate，不包含活跃 v3 结果，不代表完整 4×3 验证或总体目标完成；'
        '不产生盘/BLR 身份、距离或 H0 的新结论。\n')
    record=read(ROOT/'stability_v2_rslice/phase_dynesty_v2_computation_record.json')
    record['created_at_utc']=manifest['created_at_utc']
    record['phase_scope']=dict(name='Two frozen importance-proposal efficiency pilots',completed_pilots=2,
        evaluation_seeds_per_proposal=1,stability_gate_evaluated=False,full_4_by_3_matrix_complete=False,
        overall_goal_complete=False,active_v3_results_included=False)
    record['task']=dict(file=rel(task),sha256=sha(task),object='Fixed-target stratified importance integration and conditional Monte Carlo error',
        deliverables=['Both completed pilot outputs','Native/CSV and stratum-weight validation','Separated training/evaluation cost record','Validated computation record'])
    record['mathematical_context']['assumptions'] += ['Frozen normalized defensive Student-mixture proposals in logit unit-prior coordinates',
        'Fresh iid evaluation rows within each conditional stratum; all rows retained',
        'Second proposal trained partly on the first pilot; posterior ratio estimators are not finite-sample unbiased']
    record['mathematical_context']['precision']['target_tolerance']='Exported native arrays match CSV exactly; logweight and evidence checks 1e-12; independently recomputed RSE and f MCSE rtol2e-12'
    record['implementation_discovery']['python'].update(candidate_implementations=['Frozen Python/SciPy defensive importance pilot and same compiled GP likelihood'],
        existence_evidence='Two completed 49152-row scientific evaluations and saved raw importance identities',
        availability_evidence='Both completed in the isolated mica2-slowgp Python environment')
    record['decision'].update(selection_reason='Audit the actual frozen task-native importance calculations and preserved inputs',
        fallback_reason='not-required: both task-native pilot evaluations completed')
    record['execution']=dict(status='complete',interface='Isolated WSL Python; one BLAS thread per pilot',
        command_or_input='\n'.join(commands),code_artifact=rel(BASE/'software/MICA2-slowgp/archive/importance-pilot-v2-regularized/run_importance_pilot.py'),
        completion_scope='Only two completed single-seed proposal-efficiency pilots; training costs reported separately')
    record['result']=dict(status='complete',summary='Both pilot weight identities and exported arrays validated; no formal stability pass or overall completion asserted.',
        result_artifact=rel(report),overall_goal_complete=False,stability_gate_evaluated=False)
    record['verification']=dict(methods=['Verify completion marker and recorded PID exit before reading final artifacts',
        'Recompute logpi/logweight and original-layer-mass/N merge from all raw rows',
        'Exact comparison of native samples, likelihoods, layer IDs, contributions and weights with CSV',
        'Independent longdouble evidence RSE and negative-lag-fraction ratio MCSE arithmetic',
        'Frozen proposal identities and equal physical-target inputs checked',
        'Artifact schema, paths and SHA-256 validation'],evidence_level='numerical-evidence',
        residual_or_error='Native and CSV arrays equal; every raw importance identity difference <1e-12; MCSE comparisons rtol2e-12',
        limitations=['One evaluation seed for each of two different proposals, not a stability gate',
            'Second proposal depends on the first pilot training data; no independent two-run significance test',
            'Empirical MCSE and Kish cannot exclude unseen rare high-weight regions',
            'Self-normalized posterior ratios have finite-sample bias',
            'Unregularized 12-component candidate was not scientifically evaluated',
            'Historical NS data-generation costs are not included in proposal-training seconds',
            'No active v3 results, full 4x3 completion, physical-component identity, distance or H0 claim'])
    for p in (OUT/'manifest.json',task,report): artifacts[p]='result'
    record['artifacts']=[dict(role=role,path=rel(p),sha256=sha(p)) for p,role in sorted(artifacts.items())]
    record['note']='All finished pilot outcomes retained. Frozen v3 target source is code provenance; no ongoing v3 fit result is included.'
    save(OUT/'computation-record.json',record)
    print(json.dumps(dict(record=str(OUT/'computation-record.json'),artifact_count=len(artifacts),pilots=entries,training_costs=costs)))


if __name__=='__main__': main()
