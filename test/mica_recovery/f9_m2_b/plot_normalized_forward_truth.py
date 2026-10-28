#!/usr/bin/env python3
"""Unit-area versions of the existing four response-only forward comparisons.

Chart contract: two band PNGs and PDFs plus a two-page PDF. Rows: 60/800-day
slow kernels; columns: disk correspondence, slow correspondence, total. Black
dashed preset truth versus colored solid normalized posterior mean. Normalize
each previously displayed mean/truth by its full dense-grid integral, after
forward convolution to the common hidden-driver frame. This is not the mean
of per-draw unit-area kernels. Total is constructed before normalization, so
its original component gain proportions are retained. No new fit or inverse.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import plot_forward_truth_check as f
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, NullFormatter, MaxNLocator

OUT=Path(__file__).resolve().parent/'outputs/response_truth_unit_area_20260905'
COLORS=f.COLORS
PARTS=['disk','slow','total']


def normalize_result(result):
    x,recovered,truth,info=result
    areas_r=[float(np.trapezoid(y,x)) for y in recovered]
    areas_t=[float(np.trapezoid(y,x)) for y in truth]
    assert min(areas_r+areas_t)>0
    nr=[y/a for y,a in zip(recovered,areas_r)]
    nt=[y/a for y,a in zip(truth,areas_t)]
    normalized_areas_r=[float(np.trapezoid(y,x)) for y in nr]
    normalized_areas_t=[float(np.trapezoid(y,x)) for y in nt]
    assert np.allclose(normalized_areas_r+normalized_areas_t,1,rtol=0,atol=1e-12)
    # Total normalized independently only after summing the original components.
    assert np.allclose(nr[2],(areas_r[0]*nr[0]+areas_r[1]*nr[1])/areas_r[2],rtol=1e-10,atol=1e-12)
    assert np.allclose(nt[2],(areas_t[0]*nt[0]+areas_t[1]*nt[1])/areas_t[2],rtol=1e-10,atol=1e-12)
    meta={**info,'normalization':'each existing posterior-mean or preset curve divided by its own integral over the full dense forward-convolution grid',
          'posterior_summary':'unit-area normalized posterior mean, not posterior mean of individually normalized draws',
          'uncertainty':'mean curves only; no intervals',
          'before_normalization_recovered_areas':areas_r,'before_normalization_truth_areas':areas_t,
          'after_normalization_recovered_areas':normalized_areas_r,'after_normalization_truth_areas':normalized_areas_t,
          'recovered_total_component_fractions':[areas_r[i]/areas_r[2] for i in [0,1]],
          'truth_total_component_fractions':[areas_t[i]/areas_t[2] for i in [0,1]]}
    return x,nr,nt,meta


def draw(band,records):
    fig,axes=plt.subplots(2,3,figsize=(17,8.7))
    fig.subplots_adjust(left=.065,right=.985,top=.81,bottom=.145,hspace=.48,wspace=.28)
    for row,(x,rec,tru,info) in enumerate(records):
        mask=f.sparse_indices(x);xx=x[mask]
        export={'lag_observer_day':xx}
        for col,ax in enumerate(axes[row]):
            ax.plot(xx,rec[col][mask],color=COLORS[col],lw=1.9,zorder=2)
            ax.plot(xx,tru[col][mask],color='#202020',ls='--',lw=1.6,zorder=3)
            ax.axvline(0,color='#bfbfbf',lw=.7);ax.axhline(0,color='#dddddd',lw=.6)
            ax.set_ylabel('归一化核密度（1/天）')
            ax.set_xlabel('相对隐藏驱动的延迟（观测者系天）')
            ax.grid(alpha=.16,lw=.6);ax.tick_params(labelsize=10)
            if col==0:
                lim=(-1,6 if band=='M2' else 12);ax.set_xlim(*lim)
                sel=(x>=lim[0])&(x<=lim[1])
                ax.set_ylim(0,max(rec[col][sel].max(),tru[col][sel].max())*1.12)
                ax.set_title('盘部分：预设盘核 vs 小质心分量',loc='left',fontsize=11)
            elif col==1:
                lim=(-100,350) if row==0 else (-1800,4500);ax.set_xlim(*lim)
                sel=(x>=lim[0])&(x<=lim[1]);ax.set_ylim(0,max(rec[col][sel].max(),tru[col][sel].max())*1.12)
                ax.set_title('慢部分：预设慢核 vs 大质心分量',loc='left',fontsize=11)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
            else:
                ax.set_xscale('symlog',linthresh=2);ax.set_yscale('symlog',linthresh=1e-7)
                ax.set_xlim(f.START,f.END+48)
                ax.set_ylim(0,max(rec[col].max(),tru[col].max())*1.12)
                ax.set_xticks([-1000,-10,0,10,100,1000,10000])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_title('总核：保留原分量比例后归一化',loc='left',fontsize=11)
            export[f'recovered_normalized_mean_{PARTS[col]}']=rec[col][mask]
            export[f'preset_normalized_{PARTS[col]}']=tru[col][mask]
        pos=axes[row,0].get_position()
        fig.text(.065,pos.y1+.038,'60 天慢核质心' if row==0 else '800 天慢核质心',fontsize=13,weight='bold')
        np.savez_compressed(OUT/f'{band}_{info["case"]}_unit_area_curves.npz',**export)
    fig.suptitle(f'F9 {band} · 归一化转移函数：预设核 vs MICA 恢复结果',fontsize=18,y=.98)
    fig.text(.5,.935,'各条预设核和后验平均核分别归一化到单位面积；统一到隐藏驱动参考系',ha='center',fontsize=12)
    handles=[Line2D([0],[0],color='#202020',ls='--',lw=1.6,label='单位面积预设核'),
             Line2D([0],[0],color=COLORS[0],lw=1.9,label='归一化后的小质心平均核'),
             Line2D([0],[0],color=COLORS[1],lw=1.9,label='归一化后的大质心平均核'),
             Line2D([0],[0],color=COLORS[2],lw=1.9,label='归一化后的总平均核')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.909),ncol=4,frameon=False,fontsize=10)
    fig.text(.5,.081,'比较形状：每条曲线用完整计算区间的积分归一化；只展示后验均值，不显示后验区间。',ha='center',fontsize=11)
    fig.text(.5,.042,'分量按质心排序。左、中列为局部线性坐标；右列保留完整尾部，两轴均近零线性、远端对数。',ha='center',fontsize=10)
    return fig


def main():
    OUT.mkdir(parents=True,exist_ok=True);w2=f.disk_mass('W2');validation=[]
    with PdfPages(OUT/'F9_response_unit_area_comparison.pdf') as pdf:
        for band in ['M2','B']:
            records=[]
            for case in f.CASES:
                result=normalize_result(f.compute(band,case,w2));records.append(result);validation.append(result[-1])
                print(band,case,'normalized areas',result[-1]['after_normalization_recovered_areas'],flush=True)
            fig=draw(band,records)
            for ext in ['png','pdf']:fig.savefig(OUT/f'F9_{band}_unit_area_comparison.{ext}',dpi=180)
            pdf.savefig(fig);plt.close(fig)
    (OUT/'validation.json').write_text(json.dumps({'generated_utc':datetime.now(timezone.utc).isoformat(),'records':validation},indent=2,ensure_ascii=False)+'\n')
    (OUT/'README.md').write_text('''# F9 单位面积转移函数比较

仅展示 M2/B 的 response_within、response_beyond 四个已完成结果。
沿用预设核与恢复核的正向卷积对照，统一到隐藏驱动参考系。

对上一版的每条后验平均核和预设核，分别除以自身完整计算网格上的积分，使面积为 1。
这是“后验平均核再归一化”，不是“每个样本归一化后再求平均”。不改变样本或其权重，仅改变显示幅度。没有反卷积或新拟合。

盘分量、慢分量各自归一化；总核先保留原增益相加，然后按总面积归一化，因此总核仍保留其原有分量比例。
只画均值，不展示后验区间。归一化积分包含负延迟和完整尾部；局部坐标窗口不参与归一化。
正向卷积和原始缓存误差、归一化前后面积及来源均记在 validation.json。
输出 NPZ 为显示网格，不用于重新积分；面积校验采用完整 0.02 天网格。

- [两页总 PDF](F9_response_unit_area_comparison.pdf)
- [M2 PDF](F9_M2_unit_area_comparison.pdf)
- [B PDF](F9_B_unit_area_comparison.pdf)

![M2](F9_M2_unit_area_comparison.png)

![B](F9_B_unit_area_comparison.png)
''')
    print(OUT,flush=True)

if __name__=='__main__':main()
