#!/usr/bin/env python3
"""Read completed Mrk142 pair fits and create bounded first-pass diagnostics.

This is not a sampler and never retries fits. All new products are below
ROOT/analysis; native parent/child outputs remain read only. Pair posteriors
are combined independently for the explicitly approximate fixed-beta tau0.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import erfcx, ndtr

FROZEN = Path('/home/leonhard/software/MICA2-slowgp/archive/stability-v3-stratified-fix1/src')
sys.path.insert(0, str(FROZEN))
from mica_slowgp.nested import load_saved_posterior
from mica_slowgp.sampler import parameter_names

WAVES = {'W2': 1928., 'M2': 2246., 'W1': 2600., 'U': 3465., 'B': 4392., 'V': 5468.}
BANDS = ('M2', 'W1', 'U', 'B', 'V')
PROBS = np.array([.025, .16, .5, .84, .975])
QLABELS = ('q025', 'q16', 'q50', 'q84', 'q975')
REDSHIFT = .0446
TAU_SEED = 142610
STYLE = Path(__file__).with_name('image-creator.mplstyle')


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def normalized_weights(weights):
    weights = np.asarray(weights, float)
    if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Invalid posterior weights')
    return weights / weights.sum()


def quantile(values, weights, probabilities=PROBS):
    values = np.asarray(values, float)
    weights = normalized_weights(weights)
    if values.shape != weights.shape or not np.isfinite(values).all():
        raise ValueError('Nonfinite or mismatched posterior values')
    keep = weights > 0
    order = np.argsort(values[keep], kind='stable')
    x, w = values[keep][order], weights[keep][order]
    return np.interp(probabilities, np.cumsum(w) - .5*w, x, left=x[0], right=x[-1])


def summarize(values, weights):
    q = quantile(values, weights)
    result = dict(zip(QLABELS, map(float, q)))
    result['mean'] = float(np.dot(values, normalized_weights(weights)))
    return result


def physical_derived(samples, scale_w, scale_b):
    """TF area and stationary short-component RMS in raw input units."""
    if not np.isfinite([scale_w, scale_b]).all() or min(scale_w, scale_b) <= 0:
        raise ValueError('Positive finite mean normalization scales required')
    p = dict(zip(parameter_names('shared'), np.asarray(samples).T))
    gain = p['A0'] * scale_b / scale_w
    rms = p['A0'] * p['sigma_x'] * np.sqrt(erfcx(p['w0']/p['tau_x'])) * scale_b
    f0 = p['A0']/(p['A0']+p['A1']) * ndtr(-p['mu0']/p['w0'])
    f1 = p['A1']/(p['A0']+p['A1']) * ndtr(-p['mu1']/p['w1'])
    return {**p, 'short_gain_raw': gain, 'short_rms_raw': rms,
            'negative_fraction_short': f0, 'negative_fraction_long': f1,
            'negative_fraction_total': f0+f1}


def compare_u(first, second):
    """Exploratory core repeat criterion, distinct from the full sampler gate."""
    checks = {}
    for name, log in [('mu0', False), ('short_gain_raw', True), ('short_rms_raw', True)]:
        x, y = first['derived'][name], second['derived'][name]
        if log:
            if np.any(x <= 0) or np.any(y <= 0):
                raise ValueError('U gain/RMS comparison requires positive values')
            x, y = np.log(x), np.log(y)
        a = quantile(x, first['weights'], [.16, .5, .84])
        b = quantile(y, second['weights'], [.16, .5, .84])
        s = np.sqrt(((a[2]-a[0])**2 + (b[2]-b[0])**2)/8)
        delta = np.abs(a-b)
        passed = bool(delta[1] <= .5*s and delta[0] <= s and delta[2] <= s)
        checks[name] = dict(space='natural_log' if log else 'linear',
            primary_quantiles=a, repeat_quantiles=b, pooled_half68=s,
            absolute_displacements=delta, median_limit=.5*s, endpoint_limit=s, passed=passed)
    return dict(available=True, passed=all(v['passed'] for v in checks.values()), checks=checks,
                definition='s=sqrt((half68_primary**2+half68_repeat**2)/2); median<=0.5s, each 68% endpoint<=s',
                caveat='One repeated band and two seeds: not the original three-seed/two-budget convergence gate.')


def compare_tau0(primary, replaced):
    checks={}
    for frame in ('obs','rest'):
        a=np.array([primary[frame][k] for k in ('q16','q50','q84')])
        b=np.array([replaced[frame][k] for k in ('q16','q50','q84')])
        s=float(np.sqrt(((a[2]-a[0])**2+(b[2]-b[0])**2)/8))
        delta=np.abs(a-b)
        checks[frame]=dict(pooled_half68=s,absolute_displacements=delta,
            median_limit=.5*s,endpoint_limit=s,
            passed=bool(delta[1]<=.5*s and delta[0]<=s and delta[2]<=s))
    return dict(available=True,passed=all(v['passed'] for v in checks.values()),checks=checks,
        bands=primary['bands'],full_five_band=primary['full_five_band'],
        definition='Linear tau0 in each frame; RMS pooled half68, median<=0.5s and each endpoint<=s.')


def tau0_posterior(band_posts, nsamp=10000, seed=TAU_SEED):
    """The existing fixed-beta weighted least-squares draw estimator, no cuts."""
    bands = [b for b in BANDS if b in band_posts]
    if not bands:
        return None
    rng = np.random.default_rng(seed)
    sigmas, draws = [], []
    for band in bands:
        values, weights = band_posts[band]
        q16, q84 = quantile(values, weights, [.16, .84])
        sigma = (q84-q16)/2
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f'{band}: zero or invalid lag 68% width')
        sigmas.append(sigma)
        draws.append(rng.choice(values, nsamp, p=normalized_weights(weights)))
    draws = np.column_stack(draws)
    x = (np.array([WAVES[b] for b in bands])/WAVES['W2'])**(4/3)-1
    invvar = 1/np.square(sigmas)
    obs = draws @ (invvar*x)/np.sum(invvar*x*x)
    weights = np.full(nsamp, 1/nsamp)
    return dict(bands=bands, full_five_band=set(bands)==set(BANDS), nsamp=nsamp, seed=seed,
        beta=4/3, lambda0_obs_angstrom=WAVES['W2'], redshift=REDSHIFT,
        sigma_obs_days=dict(zip(bands, sigmas)), obs=summarize(obs, weights),
        rest=summarize(obs/(1+REDSHIFT), weights), probability_tau0_nonpositive=float(np.mean(obs<=0)),
        samples_obs=obs, samples_rest=obs/(1+REDSHIFT), lag_draws_obs=draws,
        method='Independent weighted band draws; fixed-beta weighted least squares with sigma=half central 68% width.',
        caveat='Pair independence approximation; common-W2 cross-band covariance omitted. No lag or tau0 clipping.')


def load_job(job, manifest):
    folder = Path(job['out_dir'])
    base = {k: job.get(k) for k in ('id', 'band', 'role', 'seed')}
    base['out_dir'] = str(folder)
    if not (folder/'completed.json').is_file():
        return dict(base, state='unfinished', reason='No complete parent marker; partial children are not analyzed.'), None
    from supervise import validate_complete
    validation=validate_complete(job,manifest)
    config = json.loads((folder/'sampler_config.json').read_text())
    if config.get('mode') != 'shared' or config.get('integration_scheme') != 'ordered_mu1_stratified_v1':
        raise ValueError('Expected shared three-stratum frozen implementation')
    if len(config.get('stratum_prior_masses', [])) != 3 or len(config.get('child_seeds', [])) != 3:
        raise ValueError('Parent does not describe exactly three strata')
    names = parameter_names('shared')
    post = load_saved_posterior(folder, names)
    scales = manifest['normalization']
    scale_w, scale_b = float(scales['W2']['scale']), float(scales[job['band']]['scale'])
    derived = physical_derived(post['samples'], scale_w, scale_b)
    weights = post['weights']
    summaries = {name: summarize(values, weights) for name, values in derived.items()}
    probabilities = {'mu0_nonpositive': float(weights[derived['mu0']<=0].sum()),
        'A0_less_than_A1': float(weights[derived['A0']<derived['A1']].sum())}
    f0 = summaries['negative_fraction_short']['mean']
    f1 = summaries['negative_fraction_long']['mean']
    probabilities['negative_fraction_short_share_of_mean'] = f0/(f0+f1) if f0+f1 else None
    flags = []
    if summaries['mu0']['q025'] <= 0:
        flags.append('short lag central 95% interval includes nonpositive values')
    if summaries['mu0']['q84']-summaries['mu0']['q16'] >= 2*abs(summaries['mu0']['q50']):
        flags.append('short lag 68% half-width is at least absolute median')
    boundary = {}
    for name in ('mu0','mu1','w0','w1','Ts'):
        spec = config['priors'][name]
        lo, hi = spec['bounds']; x=derived[name]
        if spec['kind']=='log_uniform':
            x=np.log(x);lo,hi=np.log([lo,hi])
        eps=.01*(hi-lo)
        boundary[name]={'lower_one_percent_prior_coordinate': float(weights[x<=lo+eps].sum()),
                        'upper_one_percent_prior_coordinate': float(weights[x>=hi-eps].sum())}
    if any(v>.05 for edge in boundary.values() for v in edge.values()):
        flags.append('more than 5% posterior mass in an outer 1% prior-coordinate interval')
    diagnostic = json.loads((folder/'diagnostics.json').read_text())
    public = dict(base, state='complete', status='weakly_constrained' if flags else 'firstpass_complete',
        summaries=summaries, probabilities=probabilities, boundary_mass=boundary,validation=validation,
        weak_constraint_flags=flags, weak_flag_definition='Descriptive heuristics, not an adjusted convergence gate.',
        normalization=dict(driver_mean=scale_w, response_mean=scale_b),
        raw_gain_definition='A0 * response_mean / driver_mean; TF area, not disk flux',
        raw_rms_definition='A0 * sigma_x * sqrt(erfcx(w0/tau_x)) * response_mean; stationary model RMS',
        diagnostic={k: diagnostic.get(k) for k in ('weight_ess','max_posterior_weight',
            'exact_likelihood_calls_total','elapsed_seconds','raw_log_evidence','strata')})
    internal=dict(samples=post['samples'], weights=weights, derived=derived, folder=folder,
                  scale_w=scale_w,scale_b=scale_b,config=config,job=job)
    return public, internal


def load_baseline(manifest):
    raw = manifest.get('baseline_root')
    if not raw:
        return {'available':False, 'reason':'No baseline_root configured'}, {}
    base=Path(raw)
    candidates=[base,base/'2comp'/'result',base/'result']
    base=next((p for p in candidates if (p/'mica_tf0_posterior.txt').is_file()),base)
    path=base/'mica_tf0_posterior.txt'
    if not path.is_file():
        return {'available':False, 'reason':f'Baseline lag posterior absent: {path}'}, {}
    table=np.atleast_1d(np.genfromtxt(path,names=True,dtype=None,encoding='utf-8'))
    posts={}; summaries={}
    for band in BANDS:
        group=table[table['band']==band]
        if not len(group): continue
        values=np.asarray(group['tau'],float); weights=np.asarray(group['weight'],float)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite baseline tau')
        weights=normalized_weights(weights)
        posts[band]=(values,weights)
        summaries[band]={'mu0':summarize(values,weights)}
        for source,destination in [('gain','short_gain_raw'),('width','w0')]:
            if source in group.dtype.names:
                summaries[band][destination]=summarize(np.asarray(group[source],float),weights)
    summary_path=base/'mica_component_summary.csv'
    old_summary=[]
    if summary_path.is_file():
        with summary_path.open() as handle:
            old_summary=list(csv.DictReader(handle))
    return dict(available=True,posterior=str(path),component_summary=str(summary_path),
        summaries=summaries,component_summary_rows=old_summary,
        caveat='Existing free_raw two-Gaussian baseline, reused without refitting; no slow GP.'),posts


def reconstruction(internal, destination):
    """Reconstruct from minimal copies so sampler parent remains resumable."""
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('sampler_config.json','weighted_posterior.csv','input_w2.txt','input_response.txt'):
        source=internal['folder']/name; target=destination/name
        if target.exists() and target.read_bytes()!=source.read_bytes():
            raise ValueError(f'Reconstruction cache input differs: {target}')
        if not target.exists(): shutil.copyfile(source,target)
    output=destination/'joint_reconstruction.npz'
    metadata=destination/'reconstruction_metadata.json'
    if not output.is_file() or not metadata.is_file():
        from mica_slowgp.cli import reconstruct
        reconstruct(destination,draws=64,points=100,seed=142320+int(internal['job']['seed']))
    meta=json.loads(metadata.read_text())
    if meta.get('draws')!=64 or meta.get('points')!=100:
        raise ValueError('Reconstruction cache uses a different requested budget')
    with np.load(output) as f: arrays={key:f[key] for key in f.files}
    physical={key:arrays[key].copy() for key in ('time','parameter_indices')}
    for key in ('x','slow_w','w'): physical[key]=arrays[key]*internal['scale_w']
    for key in ('disk','blr','slow_b','b'): physical[key]=arrays[key]*internal['scale_b']
    physical['offsets']=arrays['offsets']*np.array([internal['scale_w'],internal['scale_b']])
    np.savez_compressed(destination/'joint_reconstruction_raw_units.npz',**physical)
    return physical,meta


def setup_plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use(STYLE)
    return plt


def save_figure(fig,stem):
    for extension in ('png','pdf'):
        fig.savefig(str(stem)+'.'+extension,dpi=300,bbox_inches='tight')


def plot_reconstruction(internal,arrays,path):
    plt=setup_plotting()
    fig,axes=plt.subplots(4,1,figsize=(7.1,8.6),sharex=True,layout='constrained')
    t=arrays['time']
    def curve(ax,key,label,color,ls='-'):
        lo,med,hi=np.quantile(arrays[key],[.16,.5,.84],axis=0)
        ax.fill_between(t,lo,hi,color='#B8B8B8',alpha=.32,linewidth=0)
        ax.plot(t,med,color=color,lw=1.7,ls=ls,label=label)
    for ax,key,filename,scale,label in [(axes[0],'w','input_w2.txt',internal['scale_w'],'W2'),
        (axes[1],'b','input_response.txt',internal['scale_b'],internal['job']['band'])]:
        data=np.loadtxt(internal['folder']/filename)
        curve(ax,key,label+' total','#1F4E9E')
        ax.errorbar(data[:,0],data[:,1]*scale,yerr=data[:,2]*scale,fmt='o',ms=2.6,
            mfc='#222222',mec='#222222',mew=.5,ecolor='#B8B8B8',elinewidth=.7,capsize=0,label='Data')
    curve(axes[2],'disk','Short Gaussian','#1F4E9E')
    curve(axes[2],'blr','Long Gaussian','#C43C39')
    curve(axes[3],'slow_w','W2 slow','#1F4E9E')
    curve(axes[3],'slow_b',internal['job']['band']+' slow','#009E73')
    for ax in axes:
        ax.set_ylabel('Flux (input units)');ax.legend(loc='best',ncol=2)
    axes[-1].set_xlabel('MJD')
    save_figure(fig,path);plt.close(fig)


def plot_lags(primary,baseline,tau,path):
    if not primary:return
    plt=setup_plotting();fig,ax=plt.subplots(figsize=(6.8,4.3),layout='constrained')
    for posts,color,marker,label in [(primary,'#1F4E9E','o','Shared slow GP'),
        (baseline,'#C43C39','s','Existing two-Gaussian')]:
        bands=[b for b in BANDS if b in posts]
        if not bands:continue
        q=np.array([quantile(*posts[b],[.16,.5,.84]) for b in bands])
        ax.errorbar([WAVES[b] for b in bands],q[:,1],yerr=np.array([q[:,1]-q[:,0],q[:,2]-q[:,1]]),
            fmt=marker,color=color,mfc=color,mec='#222222',mew=.6,ms=4.2,
            ecolor='#B8B8B8',elinewidth=.7,capsize=0,label=label)
    if tau is not None:
        wave=np.linspace(WAVES['W2'],WAVES['V'],200);x=(wave/WAVES['W2'])**(4/3)-1
        ax.fill_between(wave,x*tau['obs']['q16'],x*tau['obs']['q84'],color='#B8B8B8',alpha=.32,linewidth=0)
        ax.plot(wave,x*tau['obs']['q50'],color='#1F4E9E',lw=1.7,label=r'$\beta=4/3$')
    ax.axhline(0,color='#D9D9D9',lw=.9,zorder=0)
    ax.set_xlabel(r'$\lambda_{\rm obs}\ (\AA)$');ax.set_ylabel('Lag relative to W2 (observer days)');ax.legend()
    save_figure(fig,path);plt.close(fig)


def public_tau(tau):
    return {k:v for k,v in tau.items() if not k.startswith('samples_') and k!='lag_draws_obs'}


def report_markdown(report):
    lines=['# Mrk142 shared slow-GP 首轮结果','',f"生成时间（UTC）：{report['generated_utc']}",'',
        f"总体状态：**{report['status_zh']}**。主种子完成波段：{', '.join(report['complete_primary_bands']) or '无'}；缺失：{', '.join(report['missing_primary_bands']) or '无'}。",'',
        '模型是每对波段独立的两个 Gaussian 响应加共享慢 GP。短/长只表示排序分量，不证明盘/BLR 身份；本报告没有运行 FFA 或 H₀。', '',
        '| 波段/种子 | 状态 | μ₀ 观测系天，median [16%,84%] | 原单位短 TF 面积 | 原单位平稳短 RMS |',
        '|---|---|---|---|---|']
    def fmt(s):return f"{s['q50']:.5g} [{s['q16']:.5g}, {s['q84']:.5g}]"
    for job in report['jobs']:
        if job['state']=='complete':
            s=job['summaries'];lines.append(f"| {job['band']}/{job['seed']} | {job['status']} | {fmt(s['mu0'])} | {fmt(s['short_gain_raw'])} | {fmt(s['short_rms_raw'])} |")
        else:lines.append(f"| {job['band']}/{job['seed']} | {job['state']} | — | — | — |")
    lines+=['','TF 面积不是 FFA 盘通量。平稳 RMS 是模型解析统计量，不是观测时间窗内样本 RMS。完整的 2.5/16/50/84/97.5% 分位数与均值保存在 summary.json。','']
    repeat=report['u_repeat']
    lines.append('U 重复：'+('通过本轮核心比较门槛。' if repeat.get('passed') else '未通过本轮核心比较门槛。') if repeat.get('available') else 'U 重复：尚无两个完整结果，不能比较。')
    lines+=['门槛：μ₀ 在线性空间、面积与 RMS 在自然对数空间；s=sqrt((s₁²+s₂²)/2)，其中 s₁/s₂ 各为 half68 宽度；中位数差≤0.5s、两端点差各≤s。这只是两个种子的单带检查，不是原三种子×两预算完整稳定性认证。','',
        '| τ₀ 方案 | 实际波段 | 观测系天 median [16%,84%] | 静止系天 median [16%,84%] |',
        '|---|---|---|---|']
    for name,tau in report['tau0'].items():
        lines.append(f"| {name} | {','.join(tau['bands'])} | {fmt(tau['obs'])} | {fmt(tau['rest'])} |")
    tr=report['tau0_u_repeat']
    lines+=['',('τ₀ 的 U 替换重复比较：'+('通过' if tr['passed'] else '未通过')+'与上述相同的 pooled half68 门槛（τ₀ 在线性空间）。') if tr.get('available') else 'τ₀ 的 U 替换重复比较：所需完整结果尚不可用。','',
        '| 波段 | 新短 lag median [16%,84%] | 旧短 lag median [16%,84%] | 新原单位 TF 面积 | 旧原单位 TF 面积 |',
        '|---|---|---|---|---|']
    old=report['baseline'].get('summaries',{})
    for job in report['jobs']:
        if job['state']!='complete' or job.get('role','primary')!='primary':continue
        s=job['summaries'];b=old.get(job['band'],{})
        lines.append(f"| {job['band']} | {fmt(s['mu0'])} | {fmt(b['mu0']) if 'mu0' in b else '缺失'} | {fmt(s['short_gain_raw'])} | {fmt(b['short_gain_raw']) if 'short_gain_raw' in b else '缺失'} |")
    lines+=['','τ₀ 固定 β=4/3，参考观测波长 1928 Å；各带按原权重独立抽取 10000 次，按既有 fit_tau 的 half68 误差加权拟合。静止系除以 1+0.0446。没有剪除负 lag 或负 τ₀；共同 W2 导致的跨带相关性未传播，结果是独立 pair 组合近似。缺失主波段时结果均为 partial，不称五带结论。','',
        '旧 free_raw 双 Gaussian 后验只读复用，采用相同 τ₀ 估计器，不重新拟合。其 TF 面积与新 slow-GP 分量的物理归属仍须另行验证。','',
        '光变图为 64 个超参数/潜变量联合 draw，100 点共同时间网格，灰带为逐点中央 68% 后验区间；点误差为原观测误差。各 pair 内短/长、慢过程与两个常数偏移联合抽样，pair 之间不联合。图例短/长不含物理身份声明。重建位于 analysis/reconstruction 的最小输入副本，不修改原采样 parent。','']
    for job in report['jobs']:
        if job.get('weak_constraint_flags'):lines.append(f"- {job['id']} 弱约束提示："+'；'.join(job['weak_constraint_flags']))
        if job.get('reason'):lines.append(f"- {job['id']}：{job['reason']}")
        if job.get('reconstruction_error'):lines.append(f"- {job['id']} 重建未完成：{job['reconstruction_error']}")
    lines+=['','弱约束提示的预设描述条件：μ₀ 的 95% 区间触及非正值，或 68% 半宽不小于中位数绝对值，或所检查先验坐标两侧各 1% 区间内后验质量超过 5%。这些提示不修改先验、权重或样本，不替代收敛检验。','',
        '本阶段到此停止，不自动重跑、调先验或追加预算。', '']
    return '\n'.join(lines)


def analyse(manifest_path, do_reconstruct=True, do_plots=True):
    manifest_path=Path(manifest_path);manifest=json.loads(manifest_path.read_text())
    root=manifest_path.parent;out=root/'analysis';out.mkdir(parents=True,exist_ok=True)
    public=[];internals={};primary={};repeat=None
    for job in manifest['jobs']:
        try:
            result,internal=load_job(job,manifest)
        except Exception as exc:
            result={**{k:job.get(k) for k in ('id','band','role','seed')},'state':'invalid','reason':f'{type(exc).__name__}: {exc}'};internal=None
        public.append(result)
        if internal is None:continue
        internals[job['id']]=internal
        if job.get('role','primary')=='primary':
            primary[job['band']]=(internal['derived']['mu0'],internal['weights'])
        elif job['band']=='U':repeat=internal
        if do_reconstruct:
            try:
                arrays,meta=reconstruction(internal,out/'reconstruction'/job['id'])
                result['reconstruction']=meta
                if do_plots:plot_reconstruction(internal,arrays,out/f"lightcurves_{job['id']}")
            except Exception as exc:
                result['reconstruction_error']=f'{type(exc).__name__}: {exc}'
    first_u=next((x for x in internals.values() if x['job']['band']=='U' and x['job'].get('role','primary')=='primary'),None)
    comparison=compare_u(first_u,repeat) if first_u is not None and repeat is not None else {'available':False}
    if comparison.get('available') and not comparison['passed']:
        for job in public:
            if job['band']=='U' and job['state']=='complete':job['status']='unstable_core_repeat'
    try:baseline,baseline_posts=load_baseline(manifest)
    except Exception as exc:baseline={'available':False,'reason':f'{type(exc).__name__}: {exc}'};baseline_posts={}
    scenarios={'primary':primary,'exclude_U':{k:v for k,v in primary.items() if k!='U'}}
    if repeat is not None and first_u is not None:
        scenarios['replace_U_with_repeat']={**primary,'U':(repeat['derived']['mu0'],repeat['weights'])}
    if baseline_posts:
        scenarios['baseline_primary']=baseline_posts
        scenarios['baseline_exclude_U']={k:v for k,v in baseline_posts.items() if k!='U'}
    tau_results={};tau_errors={};fulltau={}
    for name,posts in scenarios.items():
        try:
            tau=tau0_posterior(posts)
            if tau is None:continue
            fulltau[name]=tau;tau_results[name]=public_tau(tau)
            np.savez_compressed(out/f'tau0_{name}.npz',obs=tau['samples_obs'],rest=tau['samples_rest'],
                lag_draws_obs=tau['lag_draws_obs'],bands=np.array(tau['bands']))
        except Exception as exc:tau_errors[name]=f'{type(exc).__name__}: {exc}'
    if do_plots:plot_lags(primary,baseline_posts,fulltau.get('primary'),out/'lag_wavelength')
    tau_comparison=compare_tau0(fulltau['primary'],fulltau['replace_U_with_repeat']) if all(k in fulltau for k in ('primary','replace_U_with_repeat')) else {'available':False}
    missing=[b for b in BANDS if b not in primary]
    full=not missing
    all_jobs_complete=all(j['state']=='complete' for j in public)
    postprocess_errors=bool(tau_errors or any(j.get('reconstruction_error') for j in public))
    unstable=(comparison.get('available') and not comparison['passed']) or (tau_comparison.get('available') and not tau_comparison['passed'])
    if not primary:status='未完成：尚无完整主波段后验'
    elif missing:status='未完成：部分波段结果（partial）'
    elif not all_jobs_complete:status='未完成：五主波段完成，但 U 重复缺失或无效'
    elif postprocess_errors:status='未完成：部分后处理产物生成失败'
    elif unstable:status='不稳定：U 核心或 τ₀ 重复比较未通过'
    elif any(j.get('weak_constraint_flags') for j in public):status='首轮完成，存在弱约束提示'
    else:status='首轮完成；尚未获得完整稳定性认证'
    overall_status=('pending' if not primary else 'partial' if not all_jobs_complete or postprocess_errors else
        'unstable' if unstable else
        'weakly_constrained' if any(j.get('weak_constraint_flags') for j in public) else 'firstpass_complete')
    report=dict(schema_version=1,generated_utc=datetime.now(timezone.utc).isoformat(),manifest=str(manifest_path),
        overall_status=overall_status,all_requested_jobs_complete=all_jobs_complete,postprocess_errors=postprocess_errors,
        status_zh=status,full_five_primary=full,complete_primary_bands=[b for b in BANDS if b in primary],
        missing_primary_bands=missing,jobs=public,u_repeat=comparison,tau0_u_repeat=tau_comparison,baseline=baseline,tau0=tau_results,
        tau0_errors=tau_errors,scientific_fits_started=0,ffa_or_h0_performed=False,
        pair_independence_approximation=True,source_frozen=str(FROZEN),
        reconstruction_requested=do_reconstruct,plots_requested=do_plots)
    save_json(out/'summary.json',report)
    (root/'RESULTS.md').write_text(report_markdown(report))
    print(json.dumps(dict(status=status,complete_primary_bands=report['complete_primary_bands'],
        summary=str(out/'summary.json'),report=str(root/'RESULTS.md')),ensure_ascii=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--root',type=Path);group.add_argument('--manifest',type=Path)
    parser.add_argument('--skip-reconstruction',action='store_true')
    parser.add_argument('--skip-plots',action='store_true')
    args=parser.parse_args()
    manifest=args.manifest or args.root/'manifest.json'
    report=analyse(manifest,not args.skip_reconstruction,not args.skip_plots)
    return 0 if report['complete_primary_bands'] else 3


if __name__=='__main__':
    raise SystemExit(main())
