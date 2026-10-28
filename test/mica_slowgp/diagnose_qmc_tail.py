"""Save the already-discussed seed82137 tail diagnostic; no GP or new tests.

This entry point refuses existing output files. Input NPZ/JSON files and the
validated QMC phase records are read only. No proposal is fitted or changed.
"""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.special import ndtr, logsumexp
from scipy.linalg import solve_triangular
from scipy.stats import multivariate_t

ROOT = Path('/home/leonhard/sci001/output/mica_slowgp')
PILOT = ROOT/'importance_qmc_pilot_v1/seed_82137_t2048_r8'
JSON_OUT = PILOT.parent/'tail_diagnostic.json'
MD_OUT = PILOT.parent/'tail_diagnostic.md'
sys.path.insert(0, '/home/leonhard/software/MICA2-slowgp/archive/importance-qmc-pilot-v1')
from importance import FrozenProposal, conditional_prior_coordinates, logit_open, log_prior_z, json_safe


def fraction(x):
    return (x[:,2]*ndtr(-x[:,3]/x[:,4])+x[:,5]*ndtr(-x[:,6]/x[:,7]))/(x[:,2]+x[:,5])


def region(x):
    return (x[:,6] > 5) & (x[:,6] < 6) & (x[:,7] > 6)


def main():
    for path in (JSON_OUT, MD_OUT):
        if path.exists():
            raise FileExistsError(f'Existing diagnostic is never overwritten: {path}')
    config = json.loads((PILOT/'sampler_config.json').read_text())
    saved = json.loads((PILOT/'diagnostics.json').read_text())
    names = config['parameter_names']
    with np.load(PILOT/'importance_qmc_results.npz', allow_pickle=False) as raw:
        a = {k: raw[k].copy() for k in raw.files}
    x,w,s,r,k = (a[n] for n in ('samples','weights','stratum_id','scramble_id','component_id'))
    f = fraction(x); R = 8; mean = float(w@f)
    Z = np.array([R*w[r==j].sum() for j in range(R)])
    Y = np.array([R*(w*f)[r==j].sum() for j in range(R)])
    numerator_delta = Y-mean
    denominator_delta = -mean*(Z-1)
    H = Y-mean*Z
    V = float(H@H)
    mcse = float(np.sqrt(V/(R*(R-1))))

    def group(mask):
        h = np.array([R*(w*(f-mean))[mask&(r==j)].sum() for j in range(R)])
        centered = h-h.mean()
        return {'rows': int(mask.sum()), 'posterior_mass': float(w[mask].sum()),
            'tail_numerator_share': float((w*f)[mask].sum()/mean),
            'ratio_variance_covariance_share': float(centered@H/V),
            'isolated_variance_over_total': float(centered@centered/V),
            'H_by_scramble': h,
            'numerator_by_scramble_over_global_Z': [float(R*(w*f)[mask&(r==j)].sum()) for j in range(R)],
            'denominator_by_scramble_over_global_Z': [float(R*w[mask&(r==j)].sum()) for j in range(R)]}

    layers = [{'stratum_id':j, **group(s==j)} for j in range(3)]
    components = [{'stratum_id':j, 'component_id':m,
        'block':'base' if m<12 else ('tail' if m<24 else 'prior'), **group((s==j)&(k==m))}
        for j in range(3) for m in range(25)]
    components.sort(key=lambda row: abs(row['ratio_variance_covariance_share']), reverse=True)
    local_row = np.concatenate([np.arange(np.sum(s==j)) for j in range(3)])
    rowshare = R*w*(f-mean)*H[r]/V

    def row_record(i):
        return {'global_row':int(i),'local_row':int(local_row[i]),'stratum_id':int(s[i]),
            'scramble_id':int(r[i]),'component_id':int(k[i]),'global_posterior_weight':float(w[i]),
            'f_minus':float(f[i]),'tail_numerator_share':float(w[i]*f[i]/mean),
            'signed_ratio_contribution':float(R*w[i]*(f[i]-mean)),
            'ratio_variance_covariance_share':float(rowshare[i]),
            'logimportance':float(a['logimportance'][i]),'logq':float(a['logq_z'][i]),
            'logl':float(a['logl'][i]),'parameters':dict(zip(names,x[i]))}

    top_rows = [row_record(i) for i in np.argsort(rowshare)[-12:][::-1]]
    negative_rows = [row_record(i) for i in np.argsort(rowshare)[:5]]
    coverage_sources = []
    for seed in (71231,71247,71263):
        path = ROOT/f'stability_v3_stratified/dynesty/shared_shared/budget_400/seed_{seed}/strata/stratum_01/nested_results.npz'
        with np.load(path,allow_pickle=False) as raw:
            sx = raw['samples'].copy(); sw = np.exp(raw['logwt']-logsumexp(raw['logwt']))
        sz = logit_open(conditional_prior_coordinates(sx,names,config['priors'],[5.,20.]))
        coverage_sources.append((f'NS{seed}',path,sx,sz,sw,'current proposal training source'))
    for label,path,role in [
        ('IS1',ROOT/'importance_pilot_v1/seed_82103_n16384/stratum_01.npz','current proposal training source'),
        ('IS2',ROOT/'importance_pilot_v2_regularized/seed_82119_n16384/stratum_01.npz','independent validation source; not used to train current q')]:
        with np.load(path,allow_pickle=False) as raw:
            sx=raw['samples'].copy();sz=raw['z'].copy();sw=np.exp(raw['logweight']-logsumexp(raw['logweight']))
        coverage_sources.append((label,path,sx,sz,sw,role))
    history = []
    for label,path,sx,sz,sw,role in coverage_sources:
        take = region(sx);sf = fraction(sx);tail_weights=sw[take]*sf[take]
        history.append({'source':label,'path':str(path),'role':role,
            'rows':int(take.sum()),'positive_weight_rows':int(np.sum(take&(sw>0))),
            'conditional_posterior_mass':float(sw[take].sum()),
            'conditional_tail_numerator_share':float(tail_weights.sum()/(sw@sf)),
            'regional_tail_weight_ESS':float(tail_weights.sum()**2/np.sum(tail_weights**2))})

    audit_path = Path('/home/leonhard/sci001/test/mica_slowgp/staged_importance_regularized/artifacts/regularized/stratum_01_covariance_audit.npz')
    with np.load(audit_path,allow_pickle=False) as audit:
        chol = np.linalg.cholesky(audit['block_covariances'][0])
    middle = np.load(PILOT/'stratum_01.npz',allow_pickle=False)
    proposal = FrozenProposal.load(config['proposal_files'][1]['path'])
    indices = [15240,14610,24019,12780,13261,16338,16223]
    points = middle['z'][indices]
    white_points = solve_triangular(chol,points.T,lower=True).T
    conditional_mean = float((w*f)[s==1].sum()/w[s==1].sum())
    comparisons=[]
    for j,i in enumerate(indices):
        pq=float(np.exp(middle['logimportance'][i]-saved['layers'][1]['log_evidence']))
        rec={'middle_local_row':i,'posterior_density_over_q_estimate':pq,
            'tail_target_density_over_full_q_estimate':pq*float(fraction(middle['samples'][i:i+1])[0])/conditional_mean,
            'normalization':'conditional logZ and conditional E[f] estimated from this QMC pilot',
            'neighbors':[]}
        for label,path,sx,sz,sw,role in coverage_sources:
            whitened=solve_triangular(chol,sz.T,lower=True).T
            distance=np.sqrt(np.sum((whitened-white_points[j])**2,axis=1));distance[sw<=0]=np.inf
            n=int(np.argmin(distance))
            rec['neighbors'].append({'source':label,'row':n,'distance':float(distance[n]),
                'conditional_posterior_weight':float(sw[n]),'f_minus':float(fraction(sx[n:n+1])[0]),
                'parameters':dict(zip(names,sx[n]))})
        terms=np.array([np.log(alpha)+multivariate_t.logpdf(points[j],loc=mu,shape=scale,df=proposal.df)
            for alpha,mu,scale in zip(proposal.component_weights,proposal.means,proposal.scales)]
            +[np.log(proposal.epsilon)+float(log_prior_z(points[j]))])
        responsibilities=np.exp(terms-logsumexp(terms));top=np.argsort(responsibilities)[-3:][::-1]
        rec['full_q_density_responsibilities']={'base':float(responsibilities[:12].sum()),
            'tail':float(responsibilities[12:24].sum()),'prior':float(responsibilities[-1]),
            'top_components':[{'component_id':int(n),'density_fraction':float(responsibilities[n])} for n in top]}
        comparisons.append(rec)

    report={'scope':'Preservation of the already-performed read-only seed82137 tail diagnostic; no new statistical tests or scientific evaluation',
        'pilot_directory':str(PILOT),'reproduction_script':str(Path(__file__).resolve()),
        'index_convention':'All row, component, scramble and stratum IDs are zero-based; local row indexes the matching stratum_XX.npz',
        'method':{'f_minus':'(A0*Phi(-mu0/w0)+A1*Phi(-mu1/w1))/(A0+A1)',
            'Z_r':'R*sum_i_in_r global_normalized_row_weight_i',
            'Y_r':'R*sum_i_in_r global_normalized_row_weight_i*f_i',
            'H_r':'Y_r-f_mean*Z_r; sample mean of H is zero to rounding',
            'MCSE_squared':'sum_r H_r^2/[R*(R-1)]',
            'scramble_variance_share':'H_r^2/sum_r H_r^2',
            'group_covariance_share':'sum_r centered(H_group,r)*H_r/sum_r H_r^2',
            'row_covariance_share':'[R*w_i*(f_i-f_mean)]*H_scramble(i)/sum_r H_r^2',
            'neighbor_metric':'Euclidean distance after whitening original conditional logit-prior coordinates by the fixed base block training covariance; nearest strictly positive-weight row',
            'neighbor_metric_covariance_source':str(audit_path)},
        'f_mean':mean,'f_mean_MCSE':mcse,'R':R,'actual_rows':len(w),
        'scrambles':[{'scramble_id':j,'Z_over_global_mean_Z':float(Z[j]),
            'Y_over_global_mean_Z':float(Y[j]),'individual_ratio':float(Y[j]/Z[j]),
            'numerator_delta':float(numerator_delta[j]),'denominator_delta':float(denominator_delta[j]),
            'H':float(H[j]),'variance_share':float(H[j]**2/V)} for j in range(R)],
        'global_variance_decomposition':{'numerator_variance_share':float(numerator_delta@numerator_delta/V),
            'denominator_variance_share':float(denominator_delta@denominator_delta/V),
            'two_covariance_share':float(2*numerator_delta@denominator_delta/V)},
        'layers':layers,'components_sorted_by_absolute_covariance_share':components,
        'largest_positive_row_covariance_contributions':top_rows,
        'largest_negative_row_covariance_contributions':negative_rows,
        'post_hoc_region':{'definition':'stratum_id=1 and 5<mu1<6 and w1>6',
            'status':'post-hoc descriptive region, not a predeclared hypothesis, prior, branch or stability gate',
            **group((s==1)&region(x)),'historical_sources':history},
        'selected_point_density_and_neighbor_comparisons':comparisons,
        'interpretation':{'decision':'Keep q fixed and inspect the larger independent-budget experiment before deciding whether to change q.',
            'deferred_allocation_candidate':'If the completed larger experiment is still dominated by the same components, consider Neyman-style Nk allocation estimated from an already independent pilot scramble-integral variance, with both evidence denominator and f numerator objectives. Keep q and full alpha fixed, every component and defensive contribution present, power-of-two counts, and alpha/Nk compensation. This is recorded only: no implementation, tuning or reading of partial larger-pilot results.',
            'specific_inefficiency_candidates':['Broad late response just above mu1=5 in the middle stratum: high f with scarce effective training contribution.',
                'Middle-stratum mu1 around 11-12: high posterior/q rows with small f affect the denominator.'],
            'not_new_unseen_mode':'Both types have earlier NS/IS examples; region occupancy does not establish accurate local density.',
            'training_sources':'Three old NS conditional posteriors plus first IS; second regularized IS is independent validation, not current-q training.'},
        'limitations':['Group/row covariance attributions can be negative; they are not disjoint nonnegative variance components.',
            'The region was defined after observing influential rows; no new hypothesis test is performed.',
            'One scientific seed and R=8 give only seven covariance degrees of freedom; delta MCSE and density normalization estimates are uncertain.',
            'Nearest-neighbor distance is a stated descriptive metric, not a calibrated coverage test.',
            'A single pilot cannot establish universal iid-vs-QMC efficiency or scientific stability.',
            'The larger fixed-q experiment was not read; its result is reserved for the next decision.'],
        'GP_likelihood_calls':0,'new_statistical_tests':0,'original_data_modified':False,
        'validated_QMC_phase_records_modified':False}
    report=json_safe(report)
    text='''# QMC pilot 尾部误差诊断（固定记录）

这是 seed82137、target_N=2048、R=8 的事后只读诊断记录。原始结果、提议分布和已验证 QMC phase 记录均未修改；没有新 GP 调用或统计检验。更大的固定 q 试验将用于下一步决定，本记录未读取其结果。

重算 f_mean={mean:.10f}，MCSE={mcse:.8g}。scramble 4、5、7 分别占当前比值方差约58.0%、16.2%、13.8%。scramble4 同时出现分子偏高和分母偏低；scramble5 主要由分母偏高驱动。全局分子方差、分母方差和两倍交叉协方差项分别为58.86%、13.64%、27.50%。

中心层的协方差归因约为14.95%、85.91%、−0.86%。中层 base/tail 各约43.31%/42.60%。这些是与总误差的协方差归因，可以为负；它们不是互相独立、各自非负的方差分块。

|中层本地行号|scramble/component|关键参数|f_minus|主要影响|
|---|---|---|---:|---|
|15240|4/22|mu1=5.076,w1=7.627|0.12950|分子，单行协方差归因约20.65%|
|14610|4/15|mu1=5.550,w1=9.375|0.20434|分子，约8.71%|
|24019|7/18|mu1=5.329,w1=7.627|0.09357|分子，约8.61%|
|16338|5/8|mu1=11.679,w1=0.128,Ts=5.064|4.7e-12|分母，全局权重0.008267|
|16223|5/7|mu1=11.149,w1=2.134|0.000139|分母，全局权重0.005690|

所有索引均从0开始；本地行索引同名stratum_01.npz。事后描述区域 `stratum_id=1 且 5<mu1<6 且 w1>6` 占全局后验质量0.0879%，尾均值分子6.61%，当前比值方差协方差归因51.40%。这不是预先指定的统计分支、先验或稳定性阈值。

旧NS三seed在该区域有32/45/16个正权重点，但区域尾权重ESS仅2.12/4.65/6.98；IS1和IS2约8.94/8.02。NS71263中层row4412已有mu1=5.093,w1=10.576,f=0.1413；IS1中层row14477已有mu1=5.236,w1=9.187,f=0.1093。这些说明该类点此前出现过，不能保证局部密度估准。

对row15240，按本次条件证据估计posterior/q约3.91，但归一化尾积分目标与full q之比约652。其q密度99.47%来自tail，其中component22占97.64%；该分量每scramble仅32点。分母点row16338的posterior/q约154，也有旧NS/IS邻近点。当前q训练使用旧三NS和第一IS；第二regularized IS属于独立验证来源。

建议先以更大的独立预算检验固定q，暂不追着8个scramble中的少数大权重点重新训练。若相同区域持续主导，具体的修复目标是mu1接近5的宽响应尾部密度，以及mu1约11–12的分母密度匹配。单seed、R=8仅有7个协方差自由度；比值delta误差、归一化密度比和事后区域结果都存在有限样本不确定性，不能宣布整体稳定或QMC普遍优劣。

另保留一个尚未实现的分配候选：若已完成的大预算试验仍由相同分量主导，可用此前独立pilot的分量scramble积分方差作Neyman式Nk分配，同时照顾证据分母和f分子。q和完整alpha保持不变，保留防御项与全部分量覆盖、2次幂点数及alpha/Nk补偿。这样可单独检查“提议密度拟合不足”与“已知tail分量仅32点”的差别。本记录不实现、不调参，也不读取大pilot的未完成结果。

可复算脚本：`{script}`。其默认输入为`{pilot}`，输出本JSON/MD；文件存在时拒绝覆盖。JSON保留完整公式、分层和分量归因、行参数、邻近点、距离定义及源文件路径。执行命令：

    env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/leonhard/miniforge3/envs/mica2-slowgp/bin/python -B {script}
'''.format(mean=mean,mcse=mcse,script=Path(__file__).resolve(),pilot=PILOT)
    with JSON_OUT.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    with MD_OUT.open('x') as stream:
        stream.write(text)
    print(json.dumps({'saved':[str(JSON_OUT),str(MD_OUT)],'f_mean':mean,'MCSE':mcse,
                      'GP_calls':0,'new_statistical_tests':0,'original_files_modified':False}))


if __name__=='__main__':
    main()
