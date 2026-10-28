#!/usr/bin/env python3
"""Read completed F9 runs and produce a reproducible, uncut posterior display.

Chart contract: static scientific PNG/PDF figures, in observer-frame days.
Dot/interval plots compare posterior medians and central 68.27% intervals;
open diamonds denote input moment benchmarks, never a random-signal total truth.
Line charts show observed flux and native total reconstruction uncertainty.
CDF charts show whole-kernel cumulative area, with pointwise weighted intervals.
Blue=smaller-centroid tuple, orange=larger-centroid tuple, purple=total;
markers and labels supplement color. All 10,000 raw samples retain their weight.
Files are written only to a separate presentation directory. No MICA is run.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
import numpy as np
from scipy.special import gammainc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter, MaxNLocator
from collect_mica_results import (
    component_parameter_arrays, mica_flux_scales, normalized_weights,
    param_names, reorder_component_columns, weighted_quantile,
)
from summarize_experiment import native_parameter_metadata, edge_screen

FONT = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
plt.rcParams.update({'font.family': ['DejaVu Sans', FONT.get_name()], 'font.size': 10,
                     'mathtext.fontset': 'dejavusans',
                     'axes.unicode_minus': False, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.facecolor': 'white'})
CASE_NAMES = {'broad': 'F9 实数据', 'disk_only': '纯盘',
              'response_within': '真响应 · 60天核质心', 'random_within': '独立慢信号 · 窗口内',
              'response_beyond': '真响应 · 800天核质心', 'random_beyond': '独立慢信号 · 超长'}
CASES = ['disk_only', 'response_within', 'random_within', 'response_beyond', 'random_beyond']
COLORS = {'short': '#2457A6', 'long': '#C67C18', 'total': '#7656A3', 'truth': '#25816F'}
Q = [.15865, .5, .84135]
ARRAYS = {}


def quant(x, w):
    lo, mid, hi = weighted_quantile(x, w, Q)
    return {'low': float(lo), 'median': float(mid), 'high': float(hi)}


def read_block(path, index):
    lines = path.read_text().splitlines()
    sizes = [int(x) for x in lines[1].lstrip('# ').split(':')]
    data = np.loadtxt(path, ndmin=2)
    if sum(sizes) != len(data):
        raise ValueError(f'Reconstruction block shape mismatch: {path}')
    start = sum(sizes[:index])
    return data[start:start+sizes[index]]


def read_result(case, band, truth):
    source = ROOT/'F9_a' if case == 'broad' else HERE/'cases'/case
    out = source/('runs/mica/gamma/broad_np4' if case == 'broad' else 'runs/mica')
    run = out/'2comp'/f'run_W2_to_{band}_2comp_gamma'
    d = run/'data'
    record = {'case': case, 'band': band, 'run': str(run), 'truth': truth}
    required = ['sample1d.txt_2', 'sample_info1d.txt_2', 'posterior_sample1d.txt_2',
                'weights.txt', 'para_names_line.txt_2', 'tranfunc_0_1.txt_2',
                'pline.txt_2_comp0', 'pline.txt_2_comp1', 'pall.txt_2',
                'fig_2.pdf', 'fig_line_decomp_2.pdf', 'cdnest_2.pdf']
    missing = [n for n in required if not (d/n).is_file() or (d/n).stat().st_size == 0]
    if missing:
        return {**record, 'status': 'incomplete', 'missing': missing}
    raw = np.loadtxt(d/'sample1d.txt_2', ndmin=2)
    table = np.loadtxt(d/'weights.txt', ndmin=2)
    info = np.loadtxt(d/'sample_info1d.txt_2', ndmin=2)
    posterior = np.loadtxt(d/'posterior_sample1d.txt_2', ndmin=2)
    names = param_names(d/'para_names_line.txt_2')
    assert len(raw) == len(table) == len(info) == 10000, str(run)
    assert raw.shape[1] == posterior.shape[1] == len(names)
    assert np.isfinite(raw).all() and np.isfinite(posterior).all()
    assert len(posterior) > 0
    assert np.allclose(table[:,1], info[:,1], rtol=1e-6, atol=1e-5)
    w = normalized_weights(table[:,-1])
    native = native_parameter_metadata(d/'para_names_line.txt_2')
    metadata = {x['name']: x for x in native}
    for suffix in ['amplitude', 'centroid', 'sigma']:
        a,b = [metadata[f'{i}-th_component_{suffix}'] for i in (0,1)]
        assert a['native_low'] == b['native_low'] and a['native_high'] == b['native_high']
    ordered, swaps = reorder_component_columns(raw,names,2,'gamma',{'component_order':'moment1'})
    onset,width,amp,tau = component_parameter_arrays(ordered,names,2,'gamma')
    fraction = amp/amp.sum(axis=0)
    total_tau = (fraction*tau).sum(axis=0)
    total_var = (fraction*(2*width**2+(tau-total_tau)**2)).sum(axis=0)
    ds,ts,mode = mica_flux_scales(run)
    # Integrals use full-support Gamma CDFs, not the truncated export grid.
    negative_fraction = (fraction*gammainc(2,np.maximum(0,-onset/width))).sum(axis=0)
    export_fraction = (fraction*(gammainc(2,np.maximum(0,(500-onset)/width))
                                -gammainc(2,np.maximum(0,(-100-onset)/width)))).sum(axis=0)
    metrics = {'tau_short':tau[0], 'tau_long':tau[1], 'tau_total':total_tau,
               'w_short':width[0], 'w_long':width[1], 'sd_total':np.sqrt(total_var),
               'fraction_long':fraction[1], 'gain_total':amp.sum(axis=0)*ts/ds,
               'negative_area_fraction':negative_fraction, 'export_area_fraction':export_fraction}
    priors = {}
    for metric,suffix in [('tau','centroid'),('w','sigma'),('amplitude','amplitude')]:
        m=metadata[f'0-th_component_{suffix}']
        bounds=[m['native_low'],m['native_high']]
        priors[metric]=bounds if metric=='tau' else list(map(float,np.exp(bounds)))
    edges=[]
    for m in native:
        if '_component_' not in m['name'] or m['fixed']: continue
        edges.append({'name':m['name'], 'coordinate':'centroid' if m['name'].endswith('centroid') else 'natural_log',
                      **edge_screen(raw[:,m['index']],w,m['native_low'],m['native_high'])})
    log=out/'_logs'/f'run_W2_to_{band}_2comp_gamma_run.log'
    logtext=log.read_text(errors='replace') if log.is_file() else ''
    log_done='Effective sample size =' in logtext and 'end RM modeling' in logtext
    assert log_done, f'Missing normal completion markers: {log}'
    assert np.loadtxt(d/'tranfunc_0_1.txt_2').shape == (12001,4)
    for n in ['pall.txt_2','pline.txt_2_comp0','pline.txt_2_comp1']:
        assert np.isfinite(np.loadtxt(d/n)).all()
    ARRAYS[(case,band)]={'w':w,'onset':onset,'width':width,'amp':amp,
                        'fraction':fraction,'tau':tau,'run':run,'metrics':metrics}
    return {**record,'status':'complete','metrics':{k:quant(v,w) for k,v in metrics.items()},
            'prior_bounds':priors,'native_prior_edges':edges,
            'diagnostics':{'raw_rows':len(raw),'posterior_rows':len(posterior),
                           'importance_weight_ess':float(1/(w*w).sum()),
                           'entropy_weight_ess':float(np.exp(-np.sum(w[w>0]*np.log(w[w>0])))),
                           'label_swap_mass':float(w[swaps].sum()),'retained_posterior_mass':1.,
                           'p_long_fraction_lt_01':float(w[fraction[1]<.01].sum()),
                           'p_long_fraction_lt_05':float(w[fraction[1]<.05].sum()),
                           'normal_end_markers':log_done},
            'files':{str(d/n):{'bytes':(d/n).stat().st_size,'mtime':(d/n).stat().st_mtime} for n in required}}


def save(fig,out,name):
    for ax in fig.axes:
        if ax.get_xscale() in ('log','symlog'):
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p: f'{v:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
    fig.savefig(out/f'{name}.png',dpi=170,bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def dot(ax,q,y,color,marker='o'):
    ax.errorbar(q['median'],y,xerr=[[q['median']-q['low']],[q['high']-q['median']]],
                fmt=marker,color=color,capsize=3,ms=6,lw=1.5)


def plot_real(records,out):
    rr=[x for x in records if x['case']=='broad' and x['status']=='complete']
    fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained')
    panels=[('tau_short','小质心分量：lag（天）','short'),('tau_long','大质心分量：lag（天）','long'),
            ('tau_total','总响应：面积加权 lag（天）','total'),('w_short','小质心分量：宽度 w（天）','short'),
            ('w_long','大质心分量：宽度 w（天）','long'),('fraction_long','大质心分量：响应面积占比','long')]
    for ax,(key,title,color) in zip(axes.flat,panels):
        for i,r in enumerate(rr):
            dot(ax,r['metrics'][key],i,COLORS[color])
            if key=='tau_short':
                ax.plot(r['truth']['project_scalar_disk_lag_days'],i,'D',mfc='none',mec=COLORS['truth'],ms=6)
        ax.set_yticks(range(len(rr)),[r['band'] for r in rr]);ax.invert_yaxis()
        ax.set_title(title,loc='left',fontsize=12);ax.grid(axis='x',alpha=.2)
        if key.startswith('w_'):
            ax.set_xscale('log');ax.set_xlabel('对数坐标；w 不是核的标准差')
            if key=='w_short':ax.set_xticks([.3,1,3])
        elif key=='fraction_long':ax.set_xlim(-.03,1.03)
        else:ax.axvline(0,color='.7',lw=.8)
    p=rr[0]['prior_bounds']
    fig.suptitle('F9 宽先验实数据 · W2 → 五波段\n点：后验中位数；线：中央 68.27% 区间；空心菱形：项目 X=2.49 盘 lag 参考',fontsize=14)
    fig.supxlabel(f"原生分量先验：质心 [{p['tau'][0]:g}, {p['tau'][1]:g}] 天；自动宽度下界 0.204–0.222 天、上界约 236 天。分量仅按质心排序。",fontsize=10)
    save(fig,out,'01_real_posteriors')


def plot_sim_intervals(records,out):
    lookup={(r['case'],r['band']):r for r in records}
    fig,axes=plt.subplots(2,3,figsize=(15,8.2),layout='constrained')
    for row,band in enumerate(['M2','B']):
        for col,(key,title,color) in enumerate([('tau_short','小质心分量 lag','short'),('tau_long','大质心分量 lag','long'),('tau_total','总响应 lag','total')]):
            ax=axes[row,col]
            for i,case in enumerate(CASES):
                r=lookup[(case,band)]
                if r['status']!='complete':
                    ax.text(.02,i,'未完成：不展示后验',transform=ax.get_yaxis_transform(),color='.55',fontsize=9);continue
                dot(ax,r['metrics'][key],i,COLORS[color])
                truth=None
                if key=='tau_short':truth=r['truth'].get('short_relative_centroid_days')
                elif key=='tau_long' and case.startswith('response'):truth=r['truth'].get('injected_slow_centroid_days')
                elif key=='tau_total':truth=r['truth'].get('injected_total_relative_centroid_days',r['truth'].get('total_relative_centroid_benchmark_days'))
                if truth is not None:ax.plot(truth,i,'D',mfc='none',mec=COLORS['truth'],ms=7)
            ax.set_yticks(range(5),[CASE_NAMES[c] for c in CASES]);ax.set_ylim(4.5,-.5)
            ax.set_title(f'{band} · {title}',loc='left',fontsize=12)
            if col==0:
                ax.xaxis.set_major_locator(MaxNLocator(5));ax.set_xlabel('天（线性局部放大）')
            elif col==1:
                ax.set_xscale('log');ax.set_xlim(30,2100);ax.set_xticks([30,60,100,300,1000,2000]);ax.set_xlabel('天（对数坐标）')
            else:
                ax.set_xscale('symlog',linthresh=1);ax.set_xlim(-.1,2100);ax.set_xticks([0,1,10,100,1000]);ax.set_xlabel('天（近零线性，远离零对数）')
            ax.grid(axis='x',alpha=.2)
    fig.suptitle('纯盘 → 真慢响应 / 独立慢信号：配对模拟结果\n点及横线：后验中位数与 68.27% 区间；空心菱形：注入矩参考',fontsize=14)
    fig.supxlabel('各分量质心先验 [-10, 2000] 天，宽度先验 [0.01, 1200] 天。独立慢信号没有真总 lag；小分量不自动等于盘。',fontsize=10)
    save(fig,out,'02_simulation_lags')


def plot_lightcurves(records,out,case_family,band=None):
    if case_family=='broad':
        selected=[r for r in records if r['case']=='broad'];title='F9 宽先验：观测与总光变重构';name='03_real_lightcurves'
    else:
        selected=[next(r for r in records if r['case']==c and r['band']==band) for c in CASES]
        title=f'{band} 模拟：观测、注入真值与总光变重构';name=f'04_simulation_lightcurves_{band}'
    fig,axes=plt.subplots(len(selected)+1,1,figsize=(13,13),sharex=True,layout='constrained')
    first=Path(selected[0]['run']);cont=np.loadtxt(first/'input/cont.txt')
    epoch=float(cont[:,0].min()) if case_family=='broad' else 0.
    cm=cont[:,1].mean()
    axes[0].errorbar(cont[:,0]-epoch,cont[:,1]/cm,yerr=cont[:,2]/cm,fmt='.',color='.25',ms=2.5,elinewidth=.4,alpha=.7)
    axes[0].set_ylabel('W2 / 均值');axes[0].set_title(title,loc='left',fontsize=14)
    for ax,r in zip(axes[1:],selected):
        if r['status']!='complete':
            ax.text(.5,.5,CASE_NAMES[r['case']]+'：拟合未完成',ha='center',transform=ax.transAxes,color='.5')
            continue
        run=Path(r['run']);obs=np.loadtxt(run/'input/line.txt');pred=read_block(run/'data/pall.txt_2',1)
        norm=obs[:,1].mean()
        ax.errorbar(obs[:,0]-epoch,obs[:,1]/norm,yerr=obs[:,2]/norm,fmt='.',color='.3',ms=2.3,elinewidth=.4,alpha=.7)
        within=(pred[:,0]>=obs[:,0].min())&(pred[:,0]<=obs[:,0].max());pred=pred[within]
        ax.plot(pred[:,0]-epoch,pred[:,1]/norm,color=COLORS['short'],lw=1.2)
        ax.fill_between(pred[:,0]-epoch,(pred[:,1]-pred[:,2])/norm,(pred[:,1]+pred[:,2])/norm,color=COLORS['short'],alpha=.15)
        if case_family!='broad':
            t=np.genfromtxt(HERE/'cases'/r['case']/f'truth_at_{band}_observations.csv',delimiter=',',names=True)
            ax.plot(t['observer_day'],(t['observed_flux']-t['noise'])/norm,'--',color=COLORS['long'],lw=.9,alpha=.8)
        ax.set_ylabel(r['band']+' / 均值' if case_family=='broad' else CASE_NAMES[r['case']].replace(' · ','\n'))
        ax.grid(alpha=.15)
    axes[-1].set_xlabel('距首个 W2 观测的天数' if case_family=='broad' else '模拟观测时间（天）')
    axes[-1].set_xlim(0,710)
    fig.supxlabel('灰点：观测及误差；蓝线/阴影：原生总重构及其不确定度（200 点网格）'+('；橙虚线：无噪声注入总光变' if case_family!='broad' else ''),fontsize=10)
    save(fig,out,name)


def plot_sim_width_area(records,out):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,band in enumerate(['M2','B']):
        for j,(key,title) in enumerate([('w_long','大质心分量的宽度 w（天）'),('fraction_long','大质心分量的响应面积占比')]):
            ax=axes[i,j]
            for y,case in enumerate(CASES):
                r=next(x for x in records if x['case']==case and x['band']==band)
                if r['status']!='complete':ax.text(.02,y,'未完成',transform=ax.get_yaxis_transform(),color='.5');continue
                dot(ax,r['metrics'][key],y,COLORS['long'])
                if case.startswith('response'):
                    t=r['truth'];v=t['injected_slow_width_days'] if key=='w_long' else t['slow_scale']/(1+t['slow_scale'])
                    ax.plot(v,y,'D',mec=COLORS['truth'],mfc='none',ms=7)
            ax.set_xscale('log');ax.grid(axis='x',alpha=.2)
            ax.set_yticks(range(5),[CASE_NAMES[c] for c in CASES]);ax.set_ylim(4.5,-.5)
            ax.set_title(f'{band} · {title}',loc='left',fontsize=12)
            ax.set_xlabel('对数坐标')
    fig.suptitle('判断“长分量”时，必须同时看宽度与响应面积\n点/线：后验中位数与 68.27% 区间；空心菱形：真慢响应注入值',fontsize=14)
    fig.supxlabel('纯盘和独立随机组没有一个已识别为慢响应的 Gamma 成分真值。面积占比不同于光变 RMS 占比。',fontsize=10)
    save(fig,out,'06_simulation_width_area')


def cdf_quantiles(a,grid):
    values=(a['fraction'][:,:,None]*gammainc(2,np.maximum(0,(grid[None,None,:]-a['onset'][:,:,None])/a['width'][:,:,None]))).sum(axis=0)
    return np.array([weighted_quantile(values[:,i],a['w'],Q) for i in range(len(grid))]).T


def plot_cdfs(records,out,band):
    fig,axes=plt.subplots(5,2,figsize=(12,13),layout='constrained')
    grids=[np.linspace(-5,15,350),np.unique(np.r_[-np.geomspace(.01,3000,180),np.linspace(-5,15,300),np.geomspace(.01,15000,420)])]
    for i,case in enumerate(CASES):
        r=next(x for x in records if x['case']==case and x['band']==band)
        for j,grid in enumerate(grids):
            ax=axes[i,j]
            if r['status']=='complete':
                lo,mid,hi=cdf_quantiles(ARRAYS[(case,band)],grid)
                ax.plot(grid,mid,color=COLORS['total']);ax.fill_between(grid,lo,hi,color=COLORS['total'],alpha=.2)
                if j==0:ax.axvline(r['truth']['short_relative_centroid_days'],color=COLORS['truth'],ls='--',lw=1)
            else:ax.text(.5,.5,'拟合未完成',transform=ax.transAxes,ha='center',color='.55')
            if j==1:ax.set_xscale('symlog',linthresh=10)
            ax.set_ylim(-.02,1.02);ax.set_xlim(grid[0],grid[-1]);ax.grid(alpha=.15)
            if j==0:ax.set_ylabel(CASE_NAMES[case].replace(' · ','\n')+'\n累计响应面积比例')
            ax.set_xlabel('lag（天）');ax.axvline(0,color='.6',lw=.7)
    axes[0,0].set_title('短时延局部');axes[0,1].set_title('含远端尾部的宽范围')
    fig.suptitle(f'{band}：拟合总转移函数的累计面积\n紫色：逐个联合样本先归一化，再取点位中位数及 68.27% 区间',fontsize=14)
    fig.supxlabel('纵轴表示该 lag 之前的响应面积比例；绿虚线仅为盘质心参考，不是真累计核。负 lag 相对 W2，不自动代表违反因果性。',fontsize=10)
    save(fig,out,f'05_simulation_cdf_{band}')


def write_tables(records,out):
    rows=[]
    for r in records:
        if r['status']!='complete':continue
        for key,value in r['metrics'].items():rows.append({'case':r['case'],'band':r['band'],'metric':key,**value})
    with (out/'posterior_intervals.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# F9 本批 MICA 结果展示','',f"生成：{datetime.now(timezone.utc).isoformat()}",'',
           '[中文分析说明](分析说明.md)','',
           f"完整结果 {sum(r['status']=='complete' for r in records)}/15；未完成项不使用中间采样推断后验。",'',
           '所有时间均为观测者系天数。区间为加权中央 68.27%；使用每次拟合全部联合原始样本及原始权重，未裁剪后验。',
           '小/大分量仅按每个联合样本的质心排序；两者未被识别为独立物理区域。', '',
           '| 条件 | 波段 | 小质心 | 大质心 | 总质心 | 大分量面积占比 | 权重 ESS |',
           '|---|---|---|---|---|---|---|']
    def fmt(q):return f"{q['median']:.3g} [{q['low']:.3g}, {q['high']:.3g}]"
    for r in records:
        if r['status']!='complete':lines.append(f"| {CASE_NAMES[r['case']]} | {r['band']} | 未完成 | — | — | — | — |");continue
        m=r['metrics'];v=[fmt(m[k]) for k in ['tau_short','tau_long','tau_total','fraction_long']]
        lines.append(f"| {CASE_NAMES[r['case']]} | {r['band']} | "+' | '.join(v)+f" | {r['diagnostics']['importance_weight_ess']:.0f} |")
    lines+=['','实数据先验：质心 [-10,100] 天，宽度使用每次运行原生自动范围；模拟先验：质心 [-10,2000] 天、宽度 [0.01,1200] 天。',
            '原生参数表 centroid 已经是 Gamma 质心，不能再加 2w；w 为 Gamma 尺度参数，单核标准差为 sqrt(2)w。',
            '总质心和方差逐样本计算；CDF 以每个样本总面积归一化。CDF 不是核点位中位数的积分。',
            '实数据绿菱形来自项目 X=2.49 标量盘表；模拟盘基准来自差分盘核，两者不混用。',
            '独立慢信号没有注入的总核或真总 lag。模拟真响应总矩为相对核的低频矩基准，非精确双 Gamma 注入。',
            '光变图使用原生 pall 总重构，不把按原生标签平均的 pline 曲线误当成重新排序后的物理成分。',
            '光变图阴影为 MICA 原生重构不确定度；它取重构散布与条件 GP 不确定度的保守组合，不是中央68%可信区间。',
            '先验边缘比例存于 JSON，按原生标签与原生采样坐标计算；接近先验不是截断的充分证据。','',
            '## 图表','']
    for p in sorted(out.glob('*.png')):lines.extend([f'![{p.stem}]({p.name})',''])
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'outputs/presentation_20260905');args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    truth=json.loads((HERE/'outputs/truth.json').read_text())
    disk=json.loads((HERE/'cases/disk_only/truth.json').read_text())
    theory={r['band']:float(r['theory_relative_lag_days']) for r in csv.DictReader((ROOT/'F9_a/raw/theoretical_disk_lags.csv').open())}
    records=[read_result('broad',b,{'project_scalar_disk_lag_days':theory[b]}) for b in ['M2','W1','U','B','V']]
    cases={c['name']:c for c in truth['cases']}
    for case in CASES:
        for band in ['M2','B']:
            t=disk['bands'][band] if case=='disk_only' else cases[case]['bands'][band]
            records.append(read_result(case,band,t))
    summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'complete':sum(r['status']=='complete' for r in records),
             'scope':15,'records':records,'notes':['No posterior cuts or changed priors.','Incomplete runs omitted from posterior plots.']}
    (out/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    plot_real(records,out);plot_sim_intervals(records,out);plot_sim_width_area(records,out);plot_lightcurves(records,out,'broad')
    for band in ['M2','B']:
        plot_lightcurves(records,out,'simulation',band);plot_cdfs(records,out,band)
    write_tables(records,out)
    print(json.dumps({'complete':summary['complete'],'output':str(out),'figures':len(list(out.glob('*.png')))},ensure_ascii=False))


if __name__=='__main__':main()
