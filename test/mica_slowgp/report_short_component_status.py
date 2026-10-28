"""Completed-process-only short component summaries and original scalar gate.

No TF grid, GP calls, posterior rerun, or modification of existing results.
"""
from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone
import json
import sys
import numpy as np
from scipy.special import erfcx,ndtr

ROOT=Path('/home/leonhard/sci001/output/mica_slowgp')
ARCHIVE=Path('/home/leonhard/software/MICA2-slowgp/archive/importance-qmc-pilot-v1')
sys.path.insert(0,str(ARCHIVE))
from assess_stability import compare_scalar,weighted_quantile,normalized_weights,branch_probabilities,finite_json


def main():
    outputs=[ROOT/'short_component_status.json',ROOT/'short_component_status.md']
    if any(p.exists() for p in outputs) and '--refresh-report' not in sys.argv:
        raise FileExistsError('Do not overwrite an existing status snapshot')
    gate=json.loads((ARCHIVE/'stability_gate_v1.json').read_text())
    truth=json.loads((ROOT/'small_cases/shared/truth.json').read_text())
    requested=[]
    for budget in (400,800):
        for seed in (71231,71247,71263):
            requested.append((f'v3_{budget}_{seed}',ROOT/f'stability_v3_stratified/dynesty/shared_shared/budget_{budget}/seed_{seed}','stratified_results.npz',budget,seed))
    requested += [('IS1',ROOT/'importance_pilot_v1/seed_82103_n16384','importance_results.npz',None,82103),
                  ('IS2_regularized',ROOT/'importance_pilot_v2_regularized/seed_82119_n16384','importance_results.npz',None,82119),
                  ('QMC_small',ROOT/'importance_qmc_pilot_v1/seed_82137_t2048_r8','importance_qmc_results.npz',None,82137)]
    runs={};skipped=[]
    for label,path,filename,budget,seed in requested:
        config=json.loads((path/'sampler_config.json').read_text())
        status=json.loads((path/'scheduler_state.json').read_text()) if budget else config
        pid=status.get('pid')
        completed=(path/'completed.json').is_file()
        pid_exited=pid is not None and not Path(f'/proc/{pid}').exists()
        complete_status = status.get('status')=='complete' if budget else status.get('status','complete')=='complete'
        if not completed or not pid_exited or not complete_status:
            skipped.append({'label':label,'path':str(path),'completed_marker':completed,'pid':pid,'pid_exited':pid_exited,'status':status.get('status')})
            continue
        names=config['parameter_names']
        with np.load(path/filename,allow_pickle=False) as raw:
            samples=raw['samples'].copy();weights=normalized_weights(raw['weights'])
        columns=dict(zip(names,samples.T));summaries={}
        for name in ('mu0','A0','w0'):
            values=columns[name];qs=weighted_quantile(values,weights,gate['quantile_probabilities'])
            summaries[name]={'q05':float(qs[0]),'q16':float(qs[1]),'q50':float(qs[2]),'q84':float(qs[3]),'q95':float(qs[4]),
                'mean':float(weights@values),'truth':truth[name],'median_minus_truth':float(qs[2]-truth[name]),
                'truth_in_central_68_percent_interval':bool(qs[1]<=truth[name]<=qs[3]),
                'truth_in_central_90_percent_interval':bool(qs[0]<=truth[name]<=qs[4])}
        rms=columns['A0']*columns['sigma_x']*np.sqrt(erfcx(columns['w0']/columns['tau_x']))
        rms_q=weighted_quantile(rms,weights,[.16,.5,.84])
        rms_truth=truth['A0']*truth['sigma_x']*np.sqrt(erfcx(truth['w0']/truth['tau_x']))
        rms_summary={'formula':'A0*sigma_x*sqrt(erfcx(w0/tau_x))','q16':float(rms_q[0]),
            'q50':float(rms_q[1]),'q84':float(rms_q[2]),'mean':float(weights@rms),'truth':float(rms_truth),
            'scope':'Exploratory stationary short-response RMS; not part of original stability gate or FFA disk flux.'}
        total_area=columns['A0']+columns['A1']
        f0=columns['A0']/total_area*ndtr(-columns['mu0']/columns['w0'])
        f1=columns['A1']/total_area*ndtr(-columns['mu1']/columns['w1'])
        f0mean,f1mean=float(weights@f0),float(weights@f1)
        negative_decomposition={'short_term_posterior_mean':f0mean,'long_term_posterior_mean':f1mean,
            'total_negative_fraction_mean':f0mean+f1mean,
            'short_share_of_total_mean':f0mean/(f0mean+f1mean),'long_share_of_total_mean':f1mean/(f0mean+f1mean),
            'probability_mu0_le_zero':float(weights@(columns['mu0']<=0)),
            'short_term_definition':'A0/(A0+A1)*Phi(-mu0/w0)',
            'long_term_definition':'A1/(A0+A1)*Phi(-mu1/w1)',
            'scope':'Posterior mean of each area-fraction term; no posterior truncation or inverse-lag/H0 moment computed.'}
        runs[label]={'label':label,'path':str(path),'raw_file':filename,'seed':seed,'budget_per_stratum':budget,
            'pid':pid,'pid_exited_before_read':True,'completed_marker_present':True,'summaries':summaries,
            'branches':branch_probabilities(columns,weights),'posterior_rows':len(weights),
            'exploratory_stationary_short_RMS':rms_summary,
            'negative_lag_decomposition':negative_decomposition,
            '_columns':columns,'_weights':weights,'_config':config}

    comparisons=[]
    def compare(a,b,kind):
        ra,rb=runs[a],runs[b]
        if ra['_config']['priors']!=rb['_config']['priors']:
            raise ValueError('Different priors')
        results={}
        for name in ('mu0','A0','w0'):
            log=name in gate['log_parameters']
            xa,xb=ra['_columns'][name],rb['_columns'][name]
            if log:xa,xb=np.log(xa),np.log(xb)
            result=compare_scalar(xa,ra['_weights'],xb,rb['_weights'],gate)
            result['coordinate']='natural log' if log else 'linear'
            results[name]=result
        branches={name:{'a':ra['branches'][name],'b':rb['branches'][name],
            'difference':abs(ra['branches'][name]-rb['branches'][name]),
            'passed':abs(ra['branches'][name]-rb['branches'][name])<=gate['maximum_branch_probability_displacement']}
            for name in gate['branches']}
        comparisons.append({'a':a,'b':b,'kind':kind,'parameters':results,'branches':branches})
    for budget in (400,800):
        labels=[label for label,r in runs.items() if r['budget_per_stratum']==budget]
        for a,b in combinations(labels,2):compare(a,b,f'same_budget_{budget}_cross_seed')
    for seed in (71231,71247,71263):
        a,b=f'v3_400_{seed}',f'v3_800_{seed}'
        if a in runs and b in runs:compare(a,b,'same_seed_400_to_800')
    group_summary={}
    for kind in ('same_budget_400_cross_seed','same_budget_800_cross_seed','same_seed_400_to_800'):
        rows=[c for c in comparisons if c['kind']==kind]
        group_summary[kind]={'comparison_count':len(rows),'all_three_requested_comparisons_present':len(rows)==3,
            'parameters':{name:{'passed_all_scalar_checks_in_all_pairs':bool(len(rows)==3 and all(c['parameters'][name]['passed'] for c in rows)),
                'failures':[{'a':c['a'],'b':c['b'],'failed_checks':[k for k,v in c['parameters'][name]['checks'].items() if not v]}
                            for c in rows if not c['parameters'][name]['passed']]} for name in ('mu0','A0','w0')}}
    public_runs=[{k:v for k,v in r.items() if not k.startswith('_')} for r in runs.values()]
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'One shared-slow-process synthetic realization; completed and exited v3 400/800 runs and descriptive single-run IS/QMC summaries.',
        'source_assessment':str(ARCHIVE/'assess_stability.py'),'original_gate':str(ARCHIVE/'stability_gate_v1.json'),
        'script':str(Path(__file__).resolve()),'truth_file':str(ROOT/'small_cases/shared/truth.json'),
        'quantile_probabilities':gate['quantile_probabilities'],'summary_coordinates':'physical parameter values',
        'comparison_coordinates':{'mu0':'linear','A0':'natural log','w0':'natural log'},
        'scalar_thresholds':{k:gate[k] for k in ('maximum_weighted_ks','maximum_standardized_wasserstein1','maximum_standardized_quantile_displacement','maximum_standardized_mean_displacement','maximum_branch_probability_displacement')},
        'runs':public_runs,'skipped':skipped,'comparisons':comparisons,'group_summary':group_summary,
        'amplitude_definition':'A0 is the integrated area of the short Gaussian transfer-function component, not an FFA disk flux.',
        'limitations':['Only mu0,A0,w0 and existing branch probabilities use original gates; stationary short RMS is a new exploratory summary, not an original gate feature. No full TF grid or whole-model stability gate is evaluated.',
            'IS1, IS2 and small QMC each have one seed and are descriptive comparisons, not formal stability results.',
            'A scalar stability pass does not prove correct physical component identification or truth recovery.',
            'One shared synthetic realization cannot establish repeatable recovery across noise realizations or all generating cases.',
            'No tau0, distance or H0 propagation is performed by this status calculation.'],
        'new_GP_calls':0,'active_partial_results_read':False,'existing_results_modified':False}
    text=['# 短分量状态快照','',f"UTC时间：{report['created_at_utc']}。仅包含完成标记存在、状态complete且PID已退出的结果。",'',
        '这是shared慢过程单个模拟实现；真值mu0=2.5 d、A0=0.8、w0=0.45 d。A0是短Gaussian传递函数面积，不是FFA盘通量。物理坐标分位数使用原assess_stability的加权左逆CDF。scalar比较保持原门槛：mu0在线性坐标，A0/w0在自然对数坐标。','',
        '|run|参数|q05|q16|q50|q84|q95|均值|','|---|---|---:|---:|---:|---:|---:|---:|']
    for run in public_runs:
        for name,v in run['summaries'].items():
            text.append('|'+run['label']+'|'+name+'|'+'|'.join(f'{v[k]:.8g}' for k in ('q05','q16','q50','q84','q95','mean'))+'|')
    text+=['','|run|P(mu0<0)|P(A0<A1)|P(mu1<5)|','|---|---:|---:|---:|']
    for run in public_runs:
        text.append('|'+run['label']+'|'+'|'.join(f"{run['branches'][k]:.8g}" for k in ('mu0<0','A0<A1','mu1<5'))+'|')
    text+=['','探索性平稳短分量RMS：`A0*sigma_x*sqrt(erfcx(w0/tau_x))`，真值约'+f'{rms_truth:.8g}'+'。这是新增摘要，不参加原gate，也不等同FFA盘通量。','',
        '|run|RMS q16|RMS q50|RMS q84|RMS均值|','|---|---:|---:|---:|---:|']
    for run in public_runs:
        text.append('|'+run['label']+'|'+'|'.join(f"{run['exploratory_stationary_short_RMS'][k]:.8g}" for k in ('q16','q50','q84','mean'))+'|')
    text+=['','负延迟面积分解：逐后验样本计算`f0=A0/(A0+A1)*Phi(-mu0/w0)`和`f1=A1/(A0+A1)*Phi(-mu1/w1)`，再分别加权求均值。比例是两项均值占总均值的份额；没有截断后验、计算逆时延均值或伪H0。','',
        '|run|E[f0]|E[f1]|E[f0+f1]|短项份额|长项份额|P(mu0<=0)|','|---|---:|---:|---:|---:|---:|---:|']
    for run in public_runs:
        text.append('|'+run['label']+'|'+'|'.join(f"{run['negative_lag_decomposition'][k]:.8g}" for k in ('short_term_posterior_mean','long_term_posterior_mean','total_negative_fraction_mean','short_share_of_total_mean','long_share_of_total_mean','probability_mu0_le_zero'))+'|')
    text+=['','原scalar门槛结果（全参数/TF gate未执行）：','']
    for kind,g in group_summary.items():
        text.append(f"- {kind}：{g['comparison_count']}对；"+'，'.join(f"{name}={'PASS' if result['passed_all_scalar_checks_in_all_pairs'] else 'FAIL/INCOMPLETE'}" for name,result in g['parameters'].items()))
    text+=['','IS/QMC单seed只列后验摘要，不标正式稳定。scalar门槛通过也不等于物理分量识别正确或模拟真值稳定恢复；此处只有一个shared模拟实现，没有做tau0/H0传播。完整比较指标、各失败项、真值与区间对照、完成/PID信息和路径见JSON。','',f"脚本：`{Path(__file__).resolve()}`；没有TF全grid、新GP、活跃partial读取或既有结果改动。"]
    write_mode='w' if '--refresh-report' in sys.argv else 'x'
    with outputs[0].open(write_mode) as stream:json.dump(finite_json(report),stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    with outputs[1].open(write_mode) as stream:stream.write('\n'.join(text)+'\n')
    print(json.dumps(finite_json({'created_at_utc':report['created_at_utc'],'runs':[{'label':r['label'],'RMS':r['exploratory_stationary_short_RMS'],'negative_lag_decomposition':r['negative_lag_decomposition']} for r in public_runs],
        'group_summary':group_summary,'skipped':skipped}),ensure_ascii=False))


if __name__=='__main__':main()
