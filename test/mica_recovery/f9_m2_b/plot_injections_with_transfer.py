#!/usr/bin/env python3
"""F9 simulated drivers, injected components, and recovered transfer functions.

Chart contract: four mixed conditions per band; actual observed W2 and target,
injected disk and slow terms in distinct panels, reconstructed total TF at
short and wide lags. Flux axes share scales within each band. Transfer gain
is in input-flux units; full raw joint posterior weights are retained. Black
is the weighted posterior mean total kernel, grey is a pointwise central
68.27% band, and faint grey curves are 12 weighted joint posterior draws.
Blue/orange are the means of centroid-sorted tuples, not physical labels.
No positive relative disk truth is manufactured. Only response cases have
an overlaid known additional slow Gamma kernel. No MICA fit is run.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse,json,csv
import numpy as np
from scipy.special import gammainc
import present_results as p
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,MaxNLocator,LogLocator,NullFormatter
from matplotlib.backends.backend_pdf import PdfPages
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/injections_and_transfer_20260905'
OUT.mkdir(parents=True,exist_ok=True)
CASES=['response_within','random_within','response_beyond','random_beyond']
LABEL={'response_within':'真实慢响应 · 60 天质心','random_within':'独立慢信号 · 较短档',
       'response_beyond':'真实慢响应 · 800 天质心','random_beyond':'独立慢信号 · 超长档'}
BLUE='#2457A6';ORANGE='#C67C18';BLACK='#242424';GRAY='#9b9b9b'
GRID=np.unique(np.r_[np.linspace(-3000,-3,450),np.linspace(-3,8,1101),np.arange(8,3000.01,2),np.arange(3025,15000.01,25)])
FLOOR=1e-8
plt.rcParams.update({'font.size':10,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8})
parser=argparse.ArgumentParser();parser.add_argument('--reuse',action='store_true');args=parser.parse_args()
truth=json.loads((HERE/'outputs/truth.json').read_text());case_refs={c['name']:c for c in truth['cases']}
base=float(truth['baseline_days'])


def kernel(grid,onset,width,gain):
    z=np.maximum(0,(grid[None,:]-onset[:,None])/width[:,None])
    return gain[:,None]*z*np.exp(-z)/width[:,None]


def source_stats(run):
    return {n:{'bytes':(run/'data'/n).stat().st_size,'mtime_ns':(run/'data'/n).stat().st_mtime_ns}
            for n in ['sample1d.txt_2','weights.txt','para_names_line.txt_2']}


def make_tf(case,band,record):
    a=p.ARRAYS[(case,band)];run=a['run'];stats=source_stats(run)
    npz=OUT/f'{band}_{case}_transfer.npz';meta=OUT/f'{band}_{case}_transfer.json'
    if args.reuse and npz.exists() and meta.exists():
        info=json.loads(meta.read_text());assert info['source_files']==stats,'Changed posterior; rerun without --reuse'
        with np.load(npz) as saved:return {k:saved[k] for k in saved.files},info
    weights=a['w'];ds,ts,mode=p.mica_flux_scales(run);gain=a['amp']*(ts/ds)
    n=len(weights);means=np.zeros((2,len(GRID)));qs=np.zeros((3,len(GRID)))
    for start in range(0,len(GRID),96):
        stop=min(start+96,len(GRID));grid=GRID[start:stop]
        v0=kernel(grid,a['onset'][0],a['width'][0],gain[0]);v1=kernel(grid,a['onset'][1],a['width'][1],gain[1])
        total=v0+v1
        means[0,start:stop]=weights@v0;means[1,start:stop]=weights@v1
        for j in range(len(grid)):qs[:,start+j]=p.weighted_quantile(total[:,j],weights,p.Q)
    rng=np.random.default_rng(2026090507)
    indices=np.searchsorted(np.cumsum(weights),(np.arange(12)+rng.random(12))/12)
    draws=kernel(GRID,a['onset'][0,indices],a['width'][0,indices],gain[0,indices])+kernel(GRID,a['onset'][1,indices],a['width'][1,indices],gain[1,indices])
    total_mean=means.sum(axis=0)
    captured=(gain*(gammainc(2,np.maximum(0,(GRID[-1]-a['onset'])/a['width']))-gammainc(2,np.maximum(0,(GRID[0]-a['onset'])/a['width'])))).sum(axis=0)
    mean_area=float(weights@gain.sum(axis=0));captured_mean=float(weights@captured)
    numerical=float(np.trapezoid(total_mean,GRID));relerr=abs(numerical-captured_mean)/captured_mean
    assert relerr<.007,(case,band,relerr)
    assert np.isfinite(means).all() and np.isfinite(qs).all() and np.isfinite(draws).all()
    truth_kernel=np.zeros_like(GRID)
    if case.startswith('response'):
        t=record['truth'];width=float(t['injected_slow_width_days'])
        truth_kernel=kernel(GRID,np.array([0.]),np.array([width]),np.array([t['slow_scale']]))[0]
    result={'lag':GRID,'mean_small':means[0],'mean_large':means[1],'mean_total':total_mean,
            'q16':qs[0],'median':qs[1],'q84':qs[2],'draws':draws,'draw_indices':indices,'injected_slow':truth_kernel}
    info={'case':case,'band':band,'run':str(run),'source_files':stats,'normalization':mode,
          'native_to_input_gain':ts/ds,'raw_samples':n,'weight_ess':record['diagnostics']['importance_weight_ess'],
          'posterior_mass_retained':1.,'curve_summary':'weighted posterior mean; pointwise central 68.27% interval; 12 illustrative weighted joint draws',
          'display_lag_range_days':[float(GRID[0]),float(GRID[-1])],'mean_full_support_area':mean_area,
          'mean_area_inside_display':captured_mean,'fraction_mean_area_inside_display':captured_mean/mean_area,
          'numerical_integral_relative_error':relerr,'has_known_additional_slow_kernel':case.startswith('response'),
          'known_total_relative_kernel':False,'total_equals_sum_of_component_means':bool(np.allclose(total_mean,means.sum(axis=0))),
          'short_reference_centroid_days':record['truth']['short_relative_centroid_days']}
    np.savez_compressed(npz,**result);meta.write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(f'Computed {case}/{band}: mean area captured {captured_mean/mean_area:.8f}, quadrature error {relerr:.3g}',flush=True)
    return result,info


DATA={};METADATA=[]
for band in ['M2','B']:
    for case in CASES:
        record=p.read_result(case,band,case_refs[case]['bands'][band]);assert record['status']=='complete'
        root=HERE/'cases'/case;run=Path(record['run'])
        dense=np.genfromtxt(root/f'noiseless_{band}.csv',delimiter=',',names=True)
        obs_truth=np.genfromtxt(root/f'truth_at_{band}_observations.csv',delimiter=',',names=True)
        driver=np.loadtxt(run/'input/cont.txt');target=np.loadtxt(run/'input/line.txt')
        assert np.allclose(target[:,0],obs_truth['observer_day'])
        assert np.allclose(target[:,1],1+obs_truth['disk_variation']+obs_truth['slow_variation']+obs_truth['noise'],atol=1e-9)
        assert np.allclose(dense['total_noiseless_flux'],1+dense['disk_variation']+dense['slow_variation'],atol=1e-8)
        tf,info=make_tf(case,band,record);METADATA.append(info)
        pred=p.read_block(run/'data/pall.txt_2',1);pred=pred[(pred[:,0]>=0)&(pred[:,0]<=base)]
        DATA[(band,case)]={'record':record,'dense':dense,'obs_truth':obs_truth,'driver':driver,'target':target,'pred':pred,'tf':tf}
for case in CASES:
    assert np.array_equal(DATA[('M2',case)]['driver'],DATA[('B',case)]['driver'])
for case in CASES[1:]:assert np.array_equal(DATA[('M2',case)]['driver'],DATA[('M2',CASES[0])]['driver'])


def flux_limits(band):
    vals=[]
    for case in CASES:
        d=DATA[(band,case)]
        vals.extend([d['driver'][:,1]-1-d['driver'][:,2],d['driver'][:,1]-1+d['driver'][:,2],
                     d['target'][:,1]-1-d['target'][:,2],d['target'][:,1]-1+d['target'][:,2],d['dense']['disk_variation'],d['dense']['slow_variation']])
    v=np.concatenate(vals);extent=max(abs(v.min()),abs(v.max()))*1.04
    return (-extent,extent)


def setup_flux(ax,limits,bottom=True):
    ax.set_xlim(0,base);ax.set_ylim(*limits);ax.axhline(0,color='#d7d7d7',lw=.6)
    ax.grid(axis='x',color='#ebebeb',lw=.55);ax.set_axisbelow(True)
    ax.set_xticks([0,200,400,600]);ax.yaxis.set_major_locator(MaxNLocator(3))
    if bottom:ax.set_xlabel('观测时间（天）')
    else:ax.tick_params(labelbottom=False)
    ax.set_ylabel('通量变化')


def draw_row(fig,spec,band,case,limits,detail=False):
    d=DATA[(band,case)];rec=d['record'];tf=d['tf'];dense=d['dense'];t=dense['observer_day']
    layout=spec.subgridspec(2,3,width_ratios=[1.15,1.15,1.08],hspace=.62,wspace=.32)
    driver_ax=fig.add_subplot(layout[0,0]);target_ax=fig.add_subplot(layout[1,0])
    disk_ax=fig.add_subplot(layout[0,1]);slow_ax=fig.add_subplot(layout[1,1])
    near_ax=fig.add_subplot(layout[0,2]);wide_ax=fig.add_subplot(layout[1,2])
    obs=d['driver']
    driver_ax.errorbar(obs[:,0],obs[:,1]-1,yerr=obs[:,2],fmt='.',ms=2,color='#555555',alpha=.75,elinewidth=.45)
    driver_ax.plot(obs[:,0],obs[:,1]-1,color='#999999',lw=.55,alpha=.7)
    driver_ax.set_title('W2 驱动：实际 MICA 输入',loc='left')
    obs=d['target'];pred=d['pred']
    target_ax.errorbar(obs[:,0],obs[:,1]-1,yerr=obs[:,2],fmt='.',ms=2,color='#888888',alpha=.7,elinewidth=.45)
    target_ax.plot(t,dense['total_noiseless_flux']-1,color='#b2b2b2',ls='--',lw=.8)
    target_ax.plot(pred[:,0],pred[:,1]-1,color=BLACK,lw=.9)
    target_ax.fill_between(pred[:,0],pred[:,1]-pred[:,2]-1,pred[:,1]+pred[:,2]-1,color='#777777',alpha=.16,linewidth=0)
    target_ax.set_title(f'{band} 总光变：灰点观测 / 黑线 MICA',loc='left')
    disk_ax.plot(t,dense['disk_variation'],color=BLUE,lw=.85)
    disk_ax.set_title(f'一般响应：注入的盘光变（RMS={rec["truth"]["disk_rms"]:.3f}）',loc='left',color=BLUE)
    slow_ax.plot(t,dense['slow_variation'],color=ORANGE,lw=1.35)
    slow_kind='卷积慢响应' if case.startswith('response') else '独立慢信号'
    slow_ax.set_title(f'长时标项：{slow_kind}（RMS={rec["truth"]["slow_rms"]:.3f}）',loc='left',color=ORANGE)
    for ax,bottom in [(driver_ax,False),(target_ax,True),(disk_ax,False),(slow_ax,True)]:setup_flux(ax,limits,bottom)
    for ax,iswide in [(near_ax,False),(wide_ax,True)]:
        x=tf['lag']
        if iswide:
            for draw in tf['draws']:ax.plot(x,np.where(draw>FLOOR,draw,np.nan),color='#bbbbbb',lw=.4,alpha=.42,zorder=1)
            ax.fill_between(x,np.maximum(tf['q16'],FLOOR),np.maximum(tf['q84'],FLOOR),where=tf['q84']>FLOOR,color='#b7b7b7',alpha=.32,linewidth=0,zorder=2)
            for key,color,ls in [('mean_total',BLACK,'-'),('mean_small',BLUE,':'),('mean_large',ORANGE,'--')]:
                ax.plot(x,np.where(tf[key]>FLOOR,tf[key],np.nan),color=color,ls=ls,lw=1.15,zorder=3)
            ax.set_xscale('symlog',linthresh=2);ax.set_yscale('log');ax.set_xlim(-3000,15000)
            ax.set_ylim(FLOOR,max(tf['mean_total'].max(),tf['q84'].max())*2)
            ax.set_xticks([-1000,-10,0,10,100,1000,10000]);ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'));ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_title('恢复转移函数：长尾（横轴近零线性，其余对数）',loc='left',fontsize=9)
        else:
            ax.fill_between(x,tf['q16'],tf['q84'],color='#b7b7b7',alpha=.35,linewidth=0)
            for key,color,ls in [('mean_total',BLACK,'-'),('mean_small',BLUE,':'),('mean_large',ORANGE,'--')]:ax.plot(x,tf[key],color=color,ls=ls,lw=1.15)
            limits_near=(-.6,1.1) if band=='M2' else (-1,5)
            ax.set_xlim(*limits_near);sel=(x>=limits_near[0])&(x<=limits_near[1]);ax.set_ylim(0,max(tf['mean_total'][sel].max(),tf['q84'][sel].max())*1.12)
            ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(3));ax.set_title('恢复转移函数：短时延主体',loc='left')
        if case.startswith('response'):
            known=tf['injected_slow'];ax.plot(x,np.where(known>FLOOR,known,np.nan) if iswide else known,color=BLACK,ls='-.',lw=1.1,zorder=4)
        ax.axvline(0,color='#aaaaaa',lw=.6);ax.set_xlabel('相对 W2 的 lag（天）');ax.set_ylabel('核增益 / 天');ax.grid(alpha=.15,lw=.5)
    return driver_ax


legend=[Line2D([0],[0],color=BLACK,lw=1.3,label='右图：后验平均总核'),Line2D([0],[0],color=BLUE,ls=':',lw=1.3,label='小质心分量平均核'),Line2D([0],[0],color=ORANGE,ls='--',lw=1.3,label='大质心分量平均核'),Line2D([0],[0],color=BLACK,ls='-.',lw=1.3,label='注入慢核（仅真响应组）'),Line2D([0],[0],color='#b7b7b7',lw=6,alpha=.6,label='总核逐点 68.27% 区间')]
with PdfPages(OUT/'F9_recovery_lightcurves_and_transfer_detail.pdf') as detail_pdf:
    for band in ['M2','B']:
        limits=flux_limits(band)
        for case in CASES:
            fig=plt.figure(figsize=(19,5.6));spec=fig.add_gridspec(1,1,left=.055,right=.987,top=.80,bottom=.155)
            draw_row(fig,spec[0],band,case,limits,True)
            fig.suptitle(f'F9 {band} · {LABEL[case]}：驱动、注入成分与恢复转移函数',fontsize=16,y=.984)
            fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.93),ncol=5,frameon=False,fontsize=9)
            fig.text(.5,.065,'左、中显示相对基准通量 1 的变化量；同一波段各条件共用纵轴。中列为真实注入成分，不是 MICA 分解。W2 是观测 driver，不是隐藏中心驱动。',ha='center',fontsize=9)
            fig.text(.5,.03,'右图按全部联合后验计算；细灰线为 12 次后验抽样。小/大分量只按质心排序；平均核不等于某个单独解。独立信号没有真转移函数。',ha='center',fontsize=9)
            fig.savefig(OUT/f'F9_{band}_{case}.png',dpi=180);fig.savefig(OUT/f'F9_{band}_{case}.pdf');detail_pdf.savefig(fig);plt.close(fig)
        fig=plt.figure(figsize=(19,16));grid=fig.add_gridspec(4,1,left=.055,right=.987,top=.905,bottom=.08,hspace=.37)
        anchors=[]
        for i,case in enumerate(CASES):anchors.append((draw_row(fig,grid[i],band,case,limits),case))
        fig.suptitle(f'F9 {band} · 四种混合实验：光变成分与 MICA 恢复核',fontsize=18,y=.985)
        fig.text(.5,.96,'每组左列：W2 驱动与目标总光变；中列：分别绘制注入的盘响应和长时标项；右列：恢复核的主体与长尾',ha='center',fontsize=11)
        fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.948),ncol=5,frameon=False,fontsize=10)
        for anchor,case in anchors:
            pos=anchor.get_position();fig.text(.055,pos.y1+.022,LABEL[case],fontsize=12,weight='bold')
        fig.text(.5,.032,'中列为注入真值；所有通量面板保持同一波段的相同刻度，慢/盘 RMS=0.3。目标光变阴影为 MICA 原生重构不确定度。',ha='center',fontsize=10)
        fig.text(.5,.014,'右图：输入通量单位；全部加权联合后验、无筛选。细灰线为12次后验抽样；平均核不是单独解。独立组没有真核；不把绝对盘核当相对真核。',ha='center',fontsize=9)
        fig.savefig(OUT/f'F9_{band}_overview.png',dpi=180);fig.savefig(OUT/f'F9_{band}_overview.pdf');plt.close(fig)
(OUT/'validation.json').write_text(json.dumps({'generated_utc':datetime.now(timezone.utc).isoformat(),'fits':8,'records':METADATA,'all_components_match_saved_targets':True,'shared_driver_verified':True},ensure_ascii=False,indent=2)+'\n')
lines=['# F9 混合实验：驱动、注入成分与恢复转移函数','',
'四个混合条件，各自 M2/B，共 8 项已完成拟合。所有时间均为观测者系天数，沿用实际 F9 观测时间。','',
'每组左列为 W2 输入及目标总光变（灰点观测、黑线 MICA 总重构，灰虚线无噪真值）；中列将盘响应与长时标项分开；右列为恢复转移函数的主体和长尾。',
'左、中显示相对基准通量 1 的变化量；注入成分已去均值，保留实际相对幅度。每个波段的四条件共用纵轴。中列注入成分不是将 MICA 原生标签强行作物理分解。',
'右图逐个完整联合样本计算 Gamma 核，再使用最终权重汇总。核幅度恢复到输入通量单位，没有逐核单位面积归一化。粗黑线为后验平均总核；蓝点线/橙虚线为质心排序后的分量平均核。灰带为总核逐点中央 68.27% 区间，细灰线为 12 次加权后验抽样。均值可能在非对称的中央区间以外；均值核不代表某一条独立后验解。',
'宽范围为 -3000 到 15000 天、纵轴对数；低于 1e-8 的密度不显示，原始计算保留。平均核解析面积与显示区间覆盖及数值积分校验见 validation.json。',
'真响应组黑点划线仅是已知的额外慢 Gamma 项。独立信号没有真转移函数；盘只有相对隐藏驱动的绝对核，因此不画一个未经建立的正相对盘真核或真总核。',
'无需重新拟合；未修改输入或科学配置。','',
'[八页逐条件大图 PDF](F9_recovery_lightcurves_and_transfer_detail.pdf)','']
for band in ['M2','B']:
    lines.extend([f'## {band} 总览','',f'![{band} 总览](F9_{band}_overview.png)',''])
    for case in CASES:lines.extend([f'[{band} · {LABEL[case]} 大图](F9_{band}_{case}.png)',''])
(OUT/'README.md').write_text('\n'.join(lines))
print(json.dumps({'output':str(OUT),'fits':8,'overview_figures':2,'detail_pages':8},ensure_ascii=False),flush=True)
