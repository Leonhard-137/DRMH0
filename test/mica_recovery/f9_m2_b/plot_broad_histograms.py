#!/usr/bin/env python3
"""Six transfer-parameter histograms for the five F9 broad-prior real-data fits.

Static scientific figure contract: one 2x3 page per band, centroid-ordered
whole tuples; weighted raw samples, no cuts or KDE. Full native prior ranges
remain visible. A centroid inset enlarges -2 to 6 days with its posterior
mass printed. Width/amplitude densities are per dex, not per linear unit.
Blue/orange rows identify smaller/larger centroid and are also directly named.
"""
from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, csv
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'src'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from collect_mica_results import param_names, normalized_weights, weighted_quantile, reorder_component_columns, component_parameter_arrays

FONT_PATH='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
fontManager.addfont(FONT_PATH)
FONT=FontProperties(fname=FONT_PATH).get_name()
plt.rcParams.update({'font.family':['DejaVu Sans',FONT], 'font.size':11,
 'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
 'savefig.facecolor':'white','mathtext.fontset':'dejavusans'})
BANDS=['M2','W1','U','B','V']
COLORS=['#2457A6','#C67C18']
OUT=HERE/'outputs/broad_histograms_20260905'
OUT.mkdir(parents=True,exist_ok=True)
rows=[]; records=[]


def metadata(path):
    result={}
    for line in path.read_text().splitlines():
        p=line.split()
        if p and p[0].isdigit():
            result[p[1]]={'index':int(p[0]),'native_type':p[2],
                         'low':float(p[3]),'high':float(p[4]),'fixed':bool(int(p[5]))}
    return result


def hist(ax, x, weights, edges, color):
    mass,_=np.histogram(x,bins=edges,weights=weights)
    assert np.isclose(mass.sum(),1.0,atol=1e-8),mass.sum()
    density=mass/np.diff(edges)
    ax.stairs(density,edges,fill=True,color=color,alpha=.22,linewidth=0)
    ax.stairs(density,edges,color=color,linewidth=1.05)
    ax.set_ylim(bottom=0)
    return density


with PdfPages(OUT/'F9_broad_all_bands_histograms.pdf') as pdf:
    for band in BANDS:
        run=ROOT/'F9_a/runs/mica/gamma/broad_np4/2comp'/f'run_W2_to_{band}_2comp_gamma'
        data=run/'data'
        raw=np.loadtxt(data/'sample1d.txt_2',ndmin=2)
        table=np.loadtxt(data/'weights.txt',ndmin=2)
        info=np.loadtxt(data/'sample_info1d.txt_2',ndmin=2)
        names=param_names(data/'para_names_line.txt_2'); meta=metadata(data/'para_names_line.txt_2')
        assert raw.shape==(10000,len(names)) and len(table)==len(info)==len(raw)
        assert np.isfinite(raw).all() and np.allclose(table[:,1],info[:,1],rtol=1e-6,atol=1e-5)
        w=normalized_weights(table[:,-1]); ess=float(1/(w*w).sum())
        for kind in ['centroid','sigma','amplitude']:
            a,b=[meta[f'{k}-th_component_{kind}'] for k in [0,1]]
            assert a['low']==b['low'] and a['high']==b['high']
        ordered,swapped=reorder_component_columns(raw,names,2,'gamma',{'component_order':'moment1'})
        onset,width,amp,tau=component_parameter_arrays(ordered,names,2,'gamma')
        fig,axes=plt.subplots(2,3,figsize=(15.8,8.1))
        fig.subplots_adjust(left=.073,right=.982,bottom=.125,top=.835,wspace=.29,hspace=.48)
        fig.suptitle(f'F9 {band} · 宽先验拟合：六个转移函数参数的后验直方图',fontsize=17,y=.985)
        fig.text(.5,.935,f'W2 驱动  |  broad_np4 / 2comp  |  全部 10,000 个加权样本  |  权重 ESS = {ess:.0f}',ha='center',fontsize=11)
        fig.legend(handles=[Line2D([0],[0],color='#777777',ls='--',lw=1.3,label='原生先验边界'),
                            Line2D([0],[0],color='#242424',ls=':',lw=1.3,label='后验中位数')],loc='upper center',bbox_to_anchor=(.5,.923),ncol=2,frameon=False,fontsize=10)
        rec={'band':band,'run':str(run),'raw_rows':len(raw),'weight_ess':ess,'label_swap_mass':float(w[swapped].sum()),'retained_posterior_mass':1.,'parameters':[]}
        for k in [0,1]:
            color=COLORS[k]; label='小质心分量' if k==0 else '大质心分量'
            for j,(kind,suffix,values,title) in enumerate([
                ('centroid','centroid',tau[k],'质心 τ（天）'),
                ('width','sigma',width[k],'宽度 w（天，对数横轴）'),
                ('amplitude','amplitude',amp[k],'幅度 A（对数横轴）')]):
                ax=axes[k,j]; m=meta[f'{k}-th_component_{suffix}']
                lo,hi=m['low'],m['high']; islog=kind!='centroid'
                if islog:
                    limits=np.array([lo,hi])/np.log(10); x=np.log10(values)
                    iqr=np.diff(weighted_quantile(x,w,[.25,.75]))[0]
                    fd=2*iqr/ess**(1/3)
                    nbins=int(np.clip(np.ceil(np.ptp(limits)/max(fd,1e-8)),50,800))
                    edges=np.linspace(*limits,nbins+1)
                    physical_prior=np.exp([lo,hi])
                else:
                    limits=np.array([lo,hi]); x=values
                    edges=np.linspace(lo,hi,1101 if k==0 else 56)
                    physical_prior=limits
                density=hist(ax,x,w,edges,color)
                q=weighted_quantile(values,w,[.15865,.5,.84135])
                ax.axvline(np.log10(q[1]) if islog else q[1],color='#242424',ls=':',lw=1.25)
                for bound in limits: ax.axvline(bound,color='#777777',ls='--',lw=1.2)
                ax.set_xlim(limits[0]-.025*np.ptp(limits),limits[1]+.025*np.ptp(limits))
                ax.set_ylabel('后验密度 / dex' if islog else '后验密度 / 天',fontsize=10)
                ax.set_xlabel(title,fontsize=11)
                ax.set_title(f'{label}  ·  {q[1]:.3g} [{q[0]:.3g}, {q[2]:.3g}]',fontsize=11,loc='left',pad=24,color=color)
                ax.text(0,1.035,f'原生先验：[{physical_prior[0]:.4g}, {physical_prior[1]:.4g}]'+('，对数均匀' if islog else '，均匀'),transform=ax.transAxes,fontsize=9,color='#555555')
                ax.grid(axis='y',color='#e8e8e8',lw=.6);ax.set_axisbelow(True)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                if islog:
                    ticks=np.arange(np.ceil(limits[0]),np.floor(limits[1])+1)
                    ax.set_xticks(ticks)
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{10**v:g}'))
                else:
                    ax.set_xticks([-10,0,20,40,60,80,100])
                display_mass=None
                if kind=='centroid' and k==0:
                    inset=ax.inset_axes([.34,.33,.62,.57])
                    hist(inset,x,w,edges,color)
                    inset.set_xlim(-2,6)
                    local=(edges[:-1]>=-2)&(edges[:-1]<=6)
                    inset.set_ylim(0,1.12*max(density[local].max(),1e-10))
                    inset.axvline(q[1],color='#242424',ls=':',lw=1.)
                    inset.set_xticks([-2,0,2,4,6]);inset.set_yticks([])
                    inset.tick_params(labelsize=8);inset.patch.set_facecolor('#ffffff')
                    display_mass=float(w[(values>=-2)&(values<=6)].sum())
                    inset.set_title(f'局部放大 −2～6 天（含 {100*display_mass:.1f}% 后验）',fontsize=8,pad=4)
                item={'component':label,'parameter':kind,'prior_low':float(physical_prior[0]),'prior_high':float(physical_prior[1]),'q15865':float(q[0]),'median':float(q[1]),'q84135':float(q[2]),'density_coordinate':'log10' if islog else 'linear','histogram_bins':len(edges)-1,'centroid_inset_mass':display_mass}
                rec['parameters'].append(item);rows.append({'band':band,**item})
        fig.text(.5,.047,'每个样本的（质心、宽度、幅度）整组按质心排序；标题为中位数 [中央 68.27% 区间]。没有筛除、平滑或重新拟合。',ha='center',fontsize=10)
        fig.text(.5,.022,'A 为 MICA 归一化光变下的核面积参数；w 为 Gamma 尺度参数。先验边界指排序前原生分量，排序后的质心先验并非均匀分布。',ha='center',fontsize=9,color='#555555')
        fig.savefig(OUT/f'F9_{band}_six_parameter_histograms.png',dpi=180)
        fig.savefig(OUT/f'F9_{band}_six_parameter_histograms.pdf')
        pdf.savefig(fig);plt.close(fig);records.append(rec)

(OUT/'results.json').write_text(json.dumps({'generated_utc':datetime.now(timezone.utc).isoformat(),'family':'F9_a/runs/mica/gamma/broad_np4/2comp','posterior_source':'all raw sample1d.txt_2 with aligned final weights.txt column','bands':records},indent=2,ensure_ascii=False)+'\n')
with (OUT/'parameter_intervals.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# F9 宽先验实数据：参数直方图','','W2 驱动 M2/W1/U/B/V；结果来自 `F9_a/runs/mica/gamma/broad_np4/2comp`。','',
'每个波段展示六个转移函数参数。上排为每个完整样本中质心较小的分量，下排为质心较大的分量；这只是排序标签，不预先指定盘或 BLR。',
'全部原始联合样本保留其后验权重；没有后验筛选、KDE 平滑或重新拟合。幅度和宽度用对数横轴及每 dex 概率密度，不能把其纵轴理解为每线性单位的密度。',
'灰虚线是排序前原生分量先验边界，不是排序后先验密度曲线。黑点线为中位数。小质心插图放大 −2 到 6 天并标出其中的后验质量；主图仍覆盖完整先验。',
'图中 A 是 MICA 归一化光变下的核面积参数；w 是 Gamma 尺度，核标准差为 sqrt(2) * w。质心字段已是 tau，不能再加 2w。','',
'[五页合并 PDF](F9_broad_all_bands_histograms.pdf)','']
for band in BANDS: lines.extend([f'## {band}','',f'![F9 {band} 参数直方图](F9_{band}_six_parameter_histograms.png)',''])
(OUT/'README.md').write_text('\n'.join(lines))
print(json.dumps({'output':str(OUT),'bands':BANDS,'plots':5,'parameter_intervals':len(rows)},ensure_ascii=False))
