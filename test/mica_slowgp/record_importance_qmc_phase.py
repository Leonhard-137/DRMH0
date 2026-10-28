#!/usr/bin/env python3
"""Independently verify and record the finished QMC pilot, without GP calls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex

import numpy as np
from scipy.special import ndtr

BASE=Path('/home/leonhard')
ROOT=BASE/'sci001/output/mica_slowgp'
RUN=ROOT/'importance_qmc_pilot_v1/seed_82137_t2048_r8'
OUT=ROOT/'importance_qmc_pilot_v1/phase'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return str(p.relative_to(BASE))
def load(p):
    with np.load(p,allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}
def save(p,x):
    with p.open('x') as f: f.write(json.dumps(x,indent=2,allow_nan=False,ensure_ascii=False)+'\n')


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    assert (RUN/'completed.json').is_file()
    c,d=read(RUN/'sampler_config.json'),read(RUN/'diagnostics.json')
    assert not Path('/proc',str(c['pid'])).exists() and c['status']=='complete'
    assert c['seed']==82137 and c['budget']=={'target_N':2048,'scrambles':8}
    assert c['case']==c['mode']=='shared' and len(c['parameter_names'])==11
    assert not d['iid_row_standard_error_used']
    for item in c['proposal_files']+[c['proposal_module'],c['qmc_module'],c['driver'],c['assessment_module'],c['gate']]:
        assert sha(Path(item['path']))==item['sha256']
    rho=np.array([64.,465.,432.])/961
    np.testing.assert_array_equal(c['stratum_prior_masses'],rho)
    rows=[];layer_checks=[];precision_calls=0
    for s in range(3):
        a=load(RUN/f'stratum_{s:02d}.npz')
        p=load(Path(c['proposal_files'][s]['path']))
        meta=json.loads(str(p['record_json']))
        alpha=np.r_[p['component_weights'],meta['epsilon']]
        nk=2**np.maximum(3,np.ceil(np.log2(2048*alpha)).astype(int))
        N=8*int(nk.sum());assert N==[21568,24704,21248][s]
        design=c['layer_designs'][s]
        assert design['rows']==N and design['scrambles']==8
        np.testing.assert_array_equal(design['component_counts_per_scramble'],nk)
        np.testing.assert_array_equal(design['component_full_probabilities'],alpha)
        np.testing.assert_array_equal(a['component_counts_per_scramble'],nk)
        assert len(a['logweight'])==N
        for r in range(8):
            for k,count in enumerate(nk):
                assert np.count_nonzero((a['scramble_id']==r)&(a['component_id']==k))==count
        expected_keys={(s,r,k) for r in range(8) for k in range(len(nk))}
        assert {tuple(x['spawn_key']) for x in design['seed_hierarchy']}==expected_keys
        assert all(x['root_entropy']==82137 for x in design['seed_hierarchy'])
        factor=np.log(alpha[a['component_id']])-np.log(nk[a['component_id']])-np.log(8)
        np.testing.assert_array_equal(a['log_quadrature_factor'],factor)
        np.testing.assert_array_equal(a['quadrature_factor'],np.exp(factor))
        np.testing.assert_array_equal(a['logimportance'],a['logl']+a['logp_z']-a['logq_z'])
        np.testing.assert_array_equal(a['logweight'],a['logimportance']+factor)
        np.testing.assert_array_equal(a['stratum_id'],np.full(N,s))
        assert np.isfinite(a['logweight']).all()
        precision_calls+=len(a['precision_audit_indices'])
        layer_checks.append(dict(stratum=s,rows=N,rows_per_scramble=int(nk.sum()),
            component_counts=nk.tolist(),factor_identity_exact=True,
            precision_calls=len(a['precision_audit_indices']),
            maximum_precision_logl_delta=float(np.max(abs(a['precision_audit_logl_delta'])))))
        rows.append(a)
    g=load(RUN/'importance_qmc_results.npz')
    cat=lambda name:np.concatenate([a[name] for a in rows])
    logw=np.concatenate([a['logweight']+np.log(p) for a,p in zip(rows,rho)])
    np.testing.assert_array_equal(g['logweight'],logw)
    for name in ('z','samples','logl','logp_z','logq_z','logimportance','component_id','scramble_id','stratum_id','log_quadrature_factor'):
        np.testing.assert_array_equal(g[name],cat(name))
    np.testing.assert_array_equal(g['log_global_quadrature_factor'],cat('log_quadrature_factor')+np.log(rho)[g['stratum_id']])
    table=np.genfromtxt(RUN/'weighted_posterior.csv',names=True,delimiter=',')
    for j,name in enumerate(c['parameter_names']): np.testing.assert_array_equal(table[name],g['samples'][:,j])
    for name in ('component_id','scramble_id','stratum_id','logimportance','log_quadrature_factor','logweight'):
        np.testing.assert_array_equal(table[name],g[name])
    np.testing.assert_array_equal(table['log_likelihood'],g['logl'])
    np.testing.assert_array_equal(table['weight'],g['weights'])
    # Independent longdouble randomization-error calculation from all raw rows.
    x=dict(zip(c['parameter_names'],g['samples'].T))
    f=(x['A0']*ndtr(-x['mu0']/x['w0'])+x['A1']*ndtr(-x['mu1']/x['w1']))/(x['A0']+x['A1'])
    scale=float(g['scramble_integral_log_scale'])
    terms=np.exp(np.asarray(logw,dtype=np.longdouble)+np.log(np.longdouble(8))-scale)
    zr=np.array([terms[g['scramble_id']==r].sum() for r in range(8)])
    ar=np.array([(terms*f)[g['scramble_id']==r].sum() for r in range(8)])
    fm=ar.mean()/zr.mean()
    fse=np.sqrt(np.var(ar-fm*zr,ddof=1)/8)/zr.mean()
    rse=np.sqrt(np.var(zr,ddof=1)/8)/zr.mean()
    logz=scale+np.log(zr.mean())
    pos=list(g['scramble_integral_column_order']).index('total_negative_lag_fraction')
    np.testing.assert_allclose(g['scramble_integrals_scaled'][:,0],zr,rtol=2e-12,atol=1e-14)
    np.testing.assert_allclose(g['scramble_integrals_scaled'][:,pos],ar,rtol=2e-12,atol=1e-14)
    stored=d['summaries']['total_negative_lag_fraction']
    np.testing.assert_allclose([stored['mean'],stored['mean_mcse'],d['relative_evidence_mcse'],d['logz']],
                              [fm,fse,rse,logz],rtol=2e-12,atol=1e-14)
    np.testing.assert_allclose(g['weights'],np.exp(logw-float(logz)),rtol=2e-12,atol=0)
    assert c['scientific_likelihood_evaluations']==d['scientific_likelihood_evaluations']==len(logw)==67520
    assert c['precision_validation_likelihood_evaluations']==d['precision_validation_likelihood_evaluations']==precision_calls
    previous=read(ROOT/'importance_pilots_phase/manifest.json')
    iid_regularized=read(ROOT/'importance_pilot_v2_regularized/seed_82119_n16384/sampler_config.json')
    assert [p['sha256'] for p in iid_regularized['proposal_files']]==[p['sha256'] for p in c['proposal_files']]
    assert iid_regularized['priors']==c['priors']
    for name in ('input_w2.txt','input_response.txt'):
        assert sha(RUN/name)==sha(ROOT/'importance_pilot_v2_regularized/seed_82119_n16384'/name)
    verification=dict(passed=True,completion_marker=True,recorded_pid_exited=True,
        layers=layer_checks,raw_and_CSV_arrays_equal=True,global_layer_mass_merge_exact=True,
        independent_scramble_ratio_mcse=float(fse),reported_scramble_ratio_mcse=stored['mean_mcse'],
        independent_negative_fraction_mean=float(fm),relative_evidence_mcse=float(rse),
        scientific_evaluation_rows=67520,precision_validation_calls=precision_calls,
        R=8,randomization_covariance_degrees_of_freedom=7,
        row_iid_MCSE_used=False,regularized_proposal_identical_to_completed_iid_pilot=True,
        GP_calls_in_this_verification=0,active_v3_results_included=False,stability_claim=False)
    OUT.mkdir()
    save(OUT/'verification.json',verification)
    task=OUT/'task.md';task.write_text('Independently verify and record one completed frozen-proposal randomized-QMC pilot. '
        'Retain all component/scramble/stratum quadrature contributions and use scramble covariance, not iid-row errors. '
        'This is not a formal stability gate or overall project completion.\n')
    report=OUT/'phase_report.md'
    comparison=[dict(label=e['label'],rows=e['evaluation_draws'],precision_calls=e['precision_audit_GP_calls'],
        seconds=e['seconds'],fmean=e['negative_lag_fraction_mean'],fmcse=e['negative_lag_fraction_mcse']) for e in previous['pilots']]
    comparison.append(dict(label='QMC regularized, R=8',rows=67520,precision_calls=precision_calls,
        seconds=d['seconds'],fmean=float(fm),fmcse=float(fse)))
    lines='\n'.join(f"| {a['label']} | {a['rows']} | {a['precision_calls']} | {a['seconds']:.3f} | {a['fmean']:.9g} | {a['fmcse']:.9g} |" for a in comparison)
    report.write_text('# 随机化 QMC pilot：结果复核\n\n'
        '固定正则化 proposal，seed=82137，target_N=2048，8 个独立 scramble；三层分别为 21568、24704、21248 行。'
        '每个分量实际 Nk 向上取为至少 8 的 2 的幂，因此 target_N 不是每层最终行数。'
        '全部行的权重为 rho_s × alpha_k/(Nk R) × L pi_z/q_z。\n\n'
        '| 试验 | 评价行数 | 额外精度调用 | 驱动计时/秒 | 负延迟面积均值 | 条件 MCSE |\n'
        '|---|---:|---:|---:|---:|---:|\n'+lines+'\n\n'
        '逐层 alpha/Nk/R、全局层先验质量、NPZ/CSV 对应关系均通过复核；从原始行独立重建 8 个积分向量，'
        '负延迟面积比值 MCSE 和证据相对 MCSE 与报告一致。记录保存实际精度检查调用次数。\n\n'
        'QMC 使用与第二个 iid pilot 相同的正则化 proposal，本阶段不重训；此前训练成本见 importance_pilots_phase/manifest.json。'
        '三行的评价预算和误差估计方法不同，仅作描述性成本比较。单 seed、R=8 的协方差只有 7 个自由度，'
        'ratio delta-method 为近似，罕见未见贡献可能使经验误差过于乐观。Kish 集中度不能充当 iid 后验样本数。\n\n'
        '未运行正式稳定性 gate，不含活跃 v3 结果，不代表完整 4×3 或总体目标完成；不增加物理盘身份、距离或 H0 结论。\n')
    save(OUT/'manifest.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),run=rel(RUN),
        comparison=comparison,training_reused_without_new_fit=True,verification=rel(OUT/'verification.json'),
        complete_scope='one fixed-proposal QMC pilot',stability_gate_evaluated=False,
        overall_goal_complete=False,active_v3_results_included=False))
    record=read(ROOT/'importance_pilots_phase/computation-record.json')
    record['created_at_utc']=datetime.now(timezone.utc).isoformat()
    record['phase_scope']=dict(name='One randomized-QMC component-stratified importance pilot',completed_pilots=1,
        seed=82137,scrambles=8,evaluation_rows=67520,stability_gate_evaluated=False,
        active_v3_results_included=False,full_4_by_3_matrix_complete=False,overall_goal_complete=False)
    record['task']=dict(file=rel(task),sha256=sha(task),object='Frozen-proposal component-stratified randomized-QMC integration',
        deliverables=['Completed pilot raw arrays','Independent component/stratum-weight and scramble-MCSE audit','Cost comparison and validated computation record'])
    record['mathematical_context']['assumptions']=[
        'Unchanged 42+42 point two-band shared-slow GP likelihood with both constant means marginalized',
        'Same finite proper physical prior, all three sorted-center strata retained',
        'Same frozen regularized defensive mixture as the completed second iid pilot; no retraining',
        'Independent scrambled Sobol stream per stratum, scramble and component; full mixture density denominator',
        'Rows within a scramble are quadrature points, not iid samples; MCSE uses covariance of eight integral vectors',
        'Self-normalized posterior ratios and their delta-method errors are approximate; finite-sample ratios are not unbiased']
    record['mathematical_context']['precision']['target_tolerance']='Exact native/CSV and factor identities; independent longdouble scramble moments/MCSE rtol2e-12'
    record['implementation_discovery']['python'].update(candidate_implementations=['Frozen SciPy Sobol/Student-mixture QMC module and existing compiled GP likelihood'],
        existence_evidence='One completed 67520-row randomized-QMC pilot and retained raw quadrature contributions',
        availability_evidence='Completed in the existing isolated mica2-slowgp Python environment')
    archive=Path(c['driver']['path']).parent
    argv=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','/home/leonhard/miniforge3/envs/mica2-slowgp/bin/python','-B',
        c['driver']['path'],'--source-root',c['source_root'],'--proposal-module-root',str(archive),
        '--inputs-root',str(ROOT/'small_cases'),'--case','shared','--mode','shared','--target-n','2048','--scrambles','8',
        '--seed','82137','--out',str(ROOT/'replay_importance_qmc_pilot_v1')]
    for p in c['proposal_files']: argv+=['--proposal',p['path']]
    record['execution']=dict(status='complete',interface='Isolated WSL Python, one BLAS thread',
        command_or_input=shlex.join(argv),code_artifact=rel(Path(c['driver']['path'])),
        completion_scope='One QMC pilot; no new fitting during verification and no formal stability assessment')
    record['decision']['selection_reason']='Verify the actual frozen task-native randomized-QMC output without another scientific run'
    record['decision']['fallback_reason']='not-required: the task-native randomized-QMC pilot completed'
    record['result']=dict(status='complete',summary='Quadrature factors, global prior-mass merge, CSV alignment and scramble-based MCSE verified; no stability claim.',
        result_artifact=rel(report),overall_goal_complete=False,stability_gate_evaluated=False)
    record['verification']=dict(methods=['Completion marker and recorded PID exit checked',
        'Reconstruct every alpha/Nk/R factor and check component counts in all scrambles',
        'Verify original stratum mass and native/CSV row alignment',
        'Rebuild evidence and f-minus integral vectors in longdouble; independently calculate scramble ratio MCSE',
        'Frozen source/proposal hash checks and computation-record schema/path/hash validator'],
        evidence_level='numerical-evidence',residual_or_error='Factor and native/CSV identities exact; scramble ratio MCSE agrees within rtol2e-12',
        limitations=['One evaluation seed with R=8 (7 covariance degrees of freedom), not a stable error calibration',
            'Within-scramble rows are dependent; no iid-row MCSE or iid-ESS interpretation',
            'Ratio delta method is approximate and unseen rare contributions may be missed',
            'The previous iid pilots used different row budgets; comparison is descriptive, not an equal-cost proof',
            'No active v3 results, complete 4x3 validation, physical identity, distance or H0 claim'])
    artifacts={}
    for folder in (RUN,archive):
        for p in folder.iterdir():
            if p.is_file(): artifacts[p]='code' if p.suffix=='.py' else 'result'
    for p in Path(c['source_root']).glob('mica_slowgp/*'):
        if p.is_file(): artifacts[p]='code' if p.suffix in ('.py','.pyx','.so') else 'source-artifact'
    for p in [task,report,OUT/'verification.json',OUT/'manifest.json',ROOT/'importance_pilots_phase/manifest.json',Path(__file__)]:
        artifacts[p]='code' if p.suffix=='.py' else 'result'
    record['artifacts']=[dict(role=role,path=rel(p),sha256=sha(p)) for p,role in sorted(artifacts.items())]
    record['note']='Frozen v3 source is code provenance only; no active v3 fit result is included. No GP calls were made by this verification script.'
    save(OUT/'computation-record.json',record)
    print(json.dumps(dict(record=str(OUT/'computation-record.json'),artifacts=len(artifacts),verification=verification,comparison=comparison)))


if __name__=='__main__': main()
