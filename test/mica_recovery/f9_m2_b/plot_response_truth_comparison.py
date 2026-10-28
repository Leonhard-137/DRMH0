#!/usr/bin/env python3
"""Compare four response simulations against their actual discrete W2-relative kernels.

Chart contract: static scientific PNG/PDF; three columns (disk correspondence,
slow correspondence, total); density and cumulative gain rows. Truth is the
signed equivalent kernel of the actual 0.02-day discrete generator, obtained
without regularization and verified by independent linear reconvolution.
MICA curves use every raw joint draw with its original weight; individual
Gamma bin masses and cumulative gains are computed before posterior summaries.
No area renormalization, posterior filtering, or MICA rerun.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, csv
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammainc
import present_results as p
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/response_truth_comparison_20260905'
DT=.02
CASES=['response_within','response_beyond']
BANDS=['M2','B']
PARTS=['disk','slow','total']
COLORS={'disk':'#2457A6','slow':'#C67C18','total':'#242424'}
TITLES={'disk':'盘部分 ↔ 小质心分量','slow':'慢响应部分 ↔ 大质心分量','total':'总转移函数'}
parser=argparse.ArgumentParser();parser.add_argument('--reuse',action='store_true');args=parser.parse_args()
OUT.mkdir(parents=True,exist_ok=True)
truth=json.loads((HERE/'outputs/truth.json').read_text())
refs={c['name']:c for c in truth['cases']}
KDIR=HERE/'outputs/kernels'
kernels={b:np.genfromtxt(KDIR/f'disk_kernel_{b}.csv',delimiter=',',names=True) for b in ['W2',*BANDS]}
ABS={b:kernels[b]['mass'] for b in kernels}
lag_abs=kernels['W2']['lag_observer_day']
assert np.allclose(np.diff(lag_abs),DT)


def inv_disk(band):
    found=[]
    for n in [32768,131072]:
        W=np.fft.rfft(ABS['W2'],n)
        R=np.fft.fftshift(np.fft.irfft(np.fft.rfft(ABS[band],n)/W,n))
        t=(np.arange(n)-n//2)*DT
        keep=(t>=-50)&(t<=100)
        found.append((t[keep],R[keep]))
    t,m=found[-1]
    stability=float(np.max(abs(m-found[0][1])))
    recon=fftconvolve(m,ABS['W2'])
    expected=np.zeros_like(recon);j0=round(-t[0]/DT)
    expected[j0:j0+len(ABS[band])]=ABS[band]
    err=float(np.linalg.norm(recon-expected)/np.linalg.norm(expected))
    mu=float(t@m/m.sum())
    target_mu=float(lag_abs@ABS[band]/ABS[band].sum()-lag_abs@ABS['W2']/ABS['W2'].sum())
    assert stability<1e-12 and err<1e-10 and abs(mu-target_mu)<1e-9
    info={'band':band,'dt_days':DT,'fft_lengths':[32768,131072],
          'minimum_w2_fourier_modulus':float(abs(W).min()),'max_mass_change_fft_length':stability,
          'linear_reconvolution_relative_l2_error':err,
          'retained_lag_range_days':[float(t[0]),float(t[-1])],
          'signed_area':float(m.sum()),'zero_lag_cell_mass':float(m[np.argmin(abs(t))]),
          'negative_lobe_absolute_area':float(-m[m<0].sum()),'relative_centroid_days':mu}
    np.savetxt(OUT/f'{band}_true_relative_disk.csv',np.c_[t,m,m/DT,np.cumsum(m)],
               delimiter=',',header='lag_observer_day,cell_mass,cell_mean_gain_per_day,cumulative_gain',comments='')
    return t,m,info


DISK={};VALID={'generated_utc':datetime.now(timezone.utc).isoformat(),'truth_definition':'Exact equivalent signed kernel of saved discrete generator, dt=0.02 observer days; no regularization. Not a continuum-limit positive disk kernel.','disk_checks':{},'runs':{}}
for band in BANDS:
    t,m,info=inv_disk(band);DISK[band]=(t,m);VALID['disk_checks'][band]=info


def true_parts(case,band):
    d_t,d_m=DISK[band]
    name=case.removeprefix('response_')
    slow=np.genfromtxt(KDIR/f'slow_{name}_kernel.csv',delimiter=',',names=True)
    st=slow['relative_lag_observer_day'];sm=slow['density_per_day']*DT
    gain=float(refs[case]['bands'][band]['slow_scale']);sm*=gain
    assert abs(sm.sum()-gain)<1e-8
    n=round((max(d_t[-1],st[-1])-d_t[0])/DT)+1
    tt=d_t[0]+np.arange(n)*DT;tm=np.zeros(n);tm[:len(d_m)]+=d_m
    offset=round(-d_t[0]/DT);tm[offset:offset+len(sm)]+=sm
    assert abs(tm.sum()-(1+gain))<1e-8
    mu=float(tt@tm/tm.sum())
    ref=refs[case]['bands'][band]['injected_total_relative_centroid_days']
    # Saved slow masses truncate the Gamma tail at 25 widths; use a
    # tolerance far below one simulation cell for the analytic benchmark.
    assert abs(mu-ref)<DT/1000
    return {'disk':(d_t,d_m),'slow':(st,sm),'total':(tt,tm)}


def grid_for(part,case):
    near=np.arange(-3,10.001,.02)
    if part=='disk':return np.unique(np.r_[np.arange(-50,-3,.2),near,np.arange(10,100.001,.2)])
    if case=='response_within':
        mid=np.arange(-150,450.001,.5)
    else:mid=np.arange(-2200,6000.001,10.)
    return np.unique(np.r_[np.arange(-2800,-150,50.),near,mid,np.arange(450,15000.1,50.)])


def summarize(case,band,record):
    a=p.ARRAYS[(case,band)];run=a['run']
    source={n:{'bytes':(run/'data'/n).stat().st_size,'mtime_ns':(run/'data'/n).stat().st_mtime_ns}
            for n in ['sample1d.txt_2','weights.txt','para_names_line.txt_2']}
    cache=OUT/f'{band}_{case}_posterior.npz';meta=OUT/f'{band}_{case}_posterior.json'
    if args.reuse and cache.exists() and meta.exists():
        info=json.loads(meta.read_text());assert info['source_files']==source
        with np.load(cache) as f:return {k:f[k] for k in f.files},info
    w=a['w'];ds,ts,mode=p.mica_flux_scales(run);gain=a['amp']*ts/ds
    data={};rng=np.random.default_rng(2026090517)
    ids=np.searchsorted(np.cumsum(w),(np.arange(10)+rng.random(10))/10)
    for part in PARTS:
        x=grid_for(part,case);data[part+'_lag']=x
        which=[0] if part=='disk' else [1] if part=='slow' else [0,1]
        means=np.zeros((2,len(x)));quant=np.zeros((2,3,len(x)));draws=np.zeros((len(ids),len(x)))
        for start in range(0,len(x),80):
            stop=min(len(x),start+80);xx=x[start:stop][None,:]
            den=np.zeros((len(w),stop-start));cdf=np.zeros_like(den)
            for k in which:
                onset=a['onset'][k,:,None];width=a['width'][k,:,None];g=gain[k,:,None]
                lo=np.maximum(0,(xx-DT/2-onset)/width);hi=np.maximum(0,(xx+DT/2-onset)/width)
                den+=g*(gammainc(2,hi)-gammainc(2,lo))/DT
                cdf+=g*gammainc(2,np.maximum(0,(xx-onset)/width))
            for kind,arr in enumerate([den,cdf]):
                means[kind,start:stop]=w@arr
                for j in range(stop-start):quant[kind,:,start+j]=p.weighted_quantile(arr[:,j],w,p.Q)
            draws[:,start:stop]=den[ids]
        for k,kind in enumerate(['density','cumulative']):
            data[f'{part}_{kind}_mean']=means[k]
            data[f'{part}_{kind}_q16']=quant[k,0]
            data[f'{part}_{kind}_median']=quant[k,1]
            data[f'{part}_{kind}_q84']=quant[k,2]
            assert np.isfinite(means[k]).all() and np.all(quant[k,0]<=quant[k,1]) and np.all(quant[k,1]<=quant[k,2])
        data[f'{part}_density_draws']=draws
    info={'case':case,'band':band,'source_files':source,'run':str(run),
          'raw_rows':len(w),'retained_posterior_mass':1.,'weight_ess':float(1/(w@w)),
          'density_definition':'Gamma mass integrated over centered 0.02d cell, divided by 0.02d',
          'amplitude_definition':'input flux gain; no area renormalization',
          'summary':'weighted mean; pointwise central 68.27 percent interval; all joint draws',
          'component_order':'whole tuple sorted by centroid, not by physical identity'}
    np.savez_compressed(cache,**data);meta.write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n')
    return data,info


def qdict(x,w):return p.quant(x,w)


def metrics(case,band,record,tr):
    a=p.ARRAYS[(case,band)];w=a['w'];ds,ts,_=p.mica_flux_scales(a['run']);gain=a['amp']*ts/ds
    rows=[]
    for part in PARTS:
        tt,mm=tr[part];true_gain=float(mm.sum());mu=float(tt@mm/true_gain)
        if part=='disk':areas=gain[0];taus=a['tau'][0];variances=2*a['width'][0]**2
        elif part=='slow':areas=gain[1];taus=a['tau'][1];variances=2*a['width'][1]**2
        else:
            areas=gain.sum(axis=0);fractions=gain/areas;taus=(fractions*a['tau']).sum(axis=0)
            variances=(fractions*(2*a['width']**2+(a['tau']-taus)**2)).sum(axis=0)
        for name,true_val,values in [('area_gain',true_gain,areas),('centroid_days',mu,taus),
                                     ('second_cumulant_days2',float(((tt-mu)**2)@mm/true_gain),variances)]:
            q=qdict(values,w);rows.append({'case':case,'band':band,'part':part,'metric':name,'truth':true_val,**q})
    return rows


DATA={};TRUTH={};ROWS=[]
for band in BANDS:
    for case in CASES:
        rec=p.read_result(case,band,refs[case]['bands'][band]);assert rec['status']=='complete'
        TRUTH[case,band]=true_parts(case,band)
        DATA[case,band],info=summarize(case,band,rec)
        VALID['runs'][f'{band}_{case}']=info
        ROWS.extend(metrics(case,band,rec,TRUTH[case,band]))
        print(f'Compared {band}/{case}: full weighted posterior and signed disk truth',flush=True)


plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})
LEGEND=[Line2D([0],[0],color='#242424',lw=1.8,ls='--',label='真实等效相对核'),
        Line2D([0],[0],color='#2457A6',lw=1.7,label='MICA 后验均值（颜色按列）'),
        Line2D([0],[0],color='#777777',lw=6,alpha=.25,label='逐点中央 68.27% 后验区间')]

def metric_annotation(case,band,part):
    chosen={r['metric']:r for r in ROWS if r['case']==case and r['band']==band and r['part']==part}
    g=chosen['area_gain'];t=chosen['centroid_days']
    return f"质心：真值 {t['truth']:.3g}；恢复 {t['median']:.3g} [{t['low']:.3g}, {t['high']:.3g}] 天\n面积：真值 {g['truth']:.3g}；恢复 {g['median']:.3g} [{g['low']:.3g}, {g['high']:.3g}]"


def panel(ax,case,band,part,kind='density',annotate=True):
    d=DATA[case,band];x=d[part+'_lag'];color=COLORS[part]
    m=d[f'{part}_{kind}_mean'];lo=d[f'{part}_{kind}_q16'];hi=d[f'{part}_{kind}_q84']
    tt,mm=TRUTH[case,band][part];y=mm/DT if kind=='density' else np.cumsum(mm)
    # Every sign of the truth is retained; symmetric log is used only when needed.
    ax.fill_between(x,lo,hi,color=color,alpha=.18,lw=0)
    ax.plot(x,m,color=color,lw=1.7,label='MICA 后验均值',zorder=3)
    ax.plot(tt,y,color='#202020',ls='--',lw=1.5,label='真实核',zorder=4)
    ax.axhline(0,color='#aaaaaa',lw=.7);ax.axvline(0,color='#bbbbbb',lw=.6)
    if part=='disk':
        lim=(-.5,3.) if band=='M2' else (-.8,8.)
        if kind=='density':
            ax.set_yscale('symlog',linthresh=.04)
            ax.set_yticks([-.1,0,.1,1,10])
            ax.set_ylabel('核增益 / 天（有符号对数纵轴）')
            ax.text(.97,.95,f"零延迟格面积 {VALID['disk_checks'][band]['zero_lag_cell_mass']:.3f}",ha='right',va='top',transform=ax.transAxes,fontsize=9)
        else:ax.set_ylabel('累积响应增益（未归一化）')
    elif part=='slow':
        lim=(-100,350) if case=='response_within' else (-1800,4500)
        ax.set_ylabel('核增益 / 天' if kind=='density' else '累积响应增益（未归一化）')
        if kind=='density':ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    else:
        lim=(-100,750) if case=='response_within' else (-2800,15000)
        ax.set_xscale('symlog',linthresh=2)
        ax.set_xticks([-100,-10,0,1,10,100,500] if case=='response_within' else [-1000,-10,0,1,10,100,1000,10000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'));ax.xaxis.set_minor_formatter(NullFormatter())
        if kind=='density':
            ax.set_yscale('symlog',linthresh=.00003);ax.set_ylabel('核增益 / 天（有符号对数纵轴）')
            ax.set_yticks([-.1,-.01,-.001,0,.001,.01,.1,1,10])
        else:ax.set_ylabel('累积响应增益（未归一化）')
    ax.set_xlim(*lim)
    ax.set_xlabel('相对 W2 的延迟（观测者系天）'+('；横轴近零线性、远端对数' if part=='total' else ''))
    ax.set_title(TITLES[part]+(' · 累积量' if kind=='cumulative' else ''),loc='left')
    ax.grid(alpha=.15,lw=.6)
    if annotate:
        ax.text(.025,.025,metric_annotation(case,band,part),transform=ax.transAxes,va='bottom',fontsize=8.5,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.86,'pad':2})
    if kind=='cumulative':
        sel=(x>=lim[0])&(x<=lim[1]);ts=(tt>=lim[0])&(tt<=lim[1]);ymin=min(float(lo[sel].min()),float(y[ts].min()),0)
        ymax=max(float(hi[sel].max()),float(y[ts].max()));ax.set_ylim(ymin-.06*(ymax-ymin),ymax+.08*(ymax-ymin))


FOOT='真值来自实际 0.02 天离散生成核；保留零延迟集中质量及负瓣。实线为后验均值；括号为中央 68.27% 参数区间。\n分量按质心排序，仅作盘/慢部分对应比较；完整保留增益，未做单位面积归一化。'
with PdfPages(OUT/'F9_response_truth_comparison_detail.pdf') as pdf:
    for band in BANDS:
        for case in CASES:
            days=60 if case=='response_within' else 800
            fig,axes=plt.subplots(2,3,figsize=(17,9.5))
            fig.subplots_adjust(left=.06,right=.985,bottom=.125,top=.83,wspace=.27,hspace=.43)
            for col,part in enumerate(PARTS):
                panel(axes[0,col],case,band,part,'density',False)
                panel(axes[1,col],case,band,part,'cumulative',True)
            fig.suptitle(f'F9 {band} · 真实慢响应质心 {days} 天：恢复转移函数 vs 真实转移函数\n上排：核形状；下排：累积响应增益',fontsize=16,y=.97)
            fig.legend(handles=LEGEND,loc='upper center',bbox_to_anchor=(.5,.905),ncol=3,frameon=False)
            fig.text(.5,.035,FOOT,ha='center',fontsize=10)
            name=f'F9_{band}_{case}_truth_vs_recovered';fig.savefig(OUT/f'{name}.png',dpi=160);fig.savefig(OUT/f'{name}.pdf');pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(17,9.3))
        fig.subplots_adjust(left=.065,right=.985,bottom=.125,top=.85,wspace=.28,hspace=.48)
        for row,case in enumerate(CASES):
            for col,part in enumerate(PARTS):panel(axes[row,col],case,band,part,'density',True)
            axes[row,0].set_title(('60 天慢核｜' if row==0 else '800 天慢核｜')+TITLES['disk'],loc='left',fontsize=12)
        fig.suptitle(f'F9 {band} · 仅真实慢响应实验：盘、慢项与总核逐项比较',fontsize=17,y=.97)
        fig.legend(handles=LEGEND,loc='upper center',bbox_to_anchor=(.5,.925),ncol=3,frameon=False)
        fig.text(.5,.035,FOOT,ha='center',fontsize=10)
        name=f'F9_{band}_response_truth_overview';fig.savefig(OUT/f'{name}.png',dpi=170);fig.savefig(OUT/f'{name}.pdf');plt.close(fig)

with (OUT/'parameter_comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(ROWS[0]));writer.writeheader();writer.writerows(ROWS)
VALID['parameter_comparisons']=ROWS
(OUT/'validation.json').write_text(json.dumps(VALID,indent=2,ensure_ascii=False)+'\n')
(OUT/'README.md').write_text('''# F9 真实慢响应：恢复核与真核逐项比较

仅使用 response_within 和 response_beyond，M2/B 共四个已完成拟合。未重跑 MICA。

## 定义

所有核都以 W2 为参考，时间为观测者系天数。
真盘核由保存的目标盘核与 W2 盘核的频域比值求得，没有正则化、强制非负或形状拟合。
这是实际 0.02 天离散生成器的等效相对核，不宣称是连续极限下光滑、正、因果的核。
用不同 FFT 长度检查稳定性，并将截取的相对核与 W2 mass 做完整线性卷积，验证可重现原目标盘核。
真慢核为实际保存的 Gamma 慢核乘实际注入增益；真总核为真盘核加真慢核。

每个完整联合后验样本先按质心排序整个参数组，再分别计算核和累积增益，最后用所有原始权重求均值和逐点中央 68.27% 区间。
核密度统一表达为 0.02 天格内质量除格宽；图上的零格峰高依赖格宽，应结合标注的格面积与累积量阅读。
零点集中质量和负瓣完整保留：它们是当前离散相对核的性质，不直接等于物理先行或负响应区域，也不能被正双 Gamma 精确逐点表达。
灰带/色带是逐点区间，不是同时置信带；后验均值也不是某一个解。
累积增益未归一化，真核因包含负瓣可能局部下降，因此不是概率分布的 CDF。
盘对小质心、慢项对大质心仅是检查分量归属的对应方案，不预先保证 MICA 正确区分物理部分。

## 阅读

总览每波段一张，两行分别为慢核质心 60、800 天，三列为盘、慢项、总核。
黑虚线是真值，实线及色带为恢复结果。盘和总核采用有符号对数纵轴，保留负值；慢核图线性坐标。
详细 PDF 每页一个拟合，上排形状，下排累积增益。标注参数为后验中位数和中央 68.27% 区间。

[四页详细 PDF](F9_response_truth_comparison_detail.pdf)

![M2](F9_M2_response_truth_overview.png)

![B](F9_B_response_truth_overview.png)

核数值、参数表和验证记录在相邻 CSV、NPZ 和 validation.json 中。

另有不用反卷积的补充图：将恢复核与 W2 盘核再卷积，统一到隐藏驱动参考系。此图仅显示后验均值。

- [M2 平滑后对照](../response_truth_forward_20260905/F9_M2_forward_truth_comparison.pdf)
- [B 平滑后对照](../response_truth_forward_20260905/F9_B_forward_truth_comparison.pdf)
''')
print(f'Wrote {OUT}',flush=True)
