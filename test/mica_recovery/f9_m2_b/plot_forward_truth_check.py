#!/usr/bin/env python3
"""Forward compare existing posterior mean kernels in the hidden-driver frame.

This supplementary figure performs no fitting, deconvolution, posterior cuts,
renormalization, or uncertainty propagation. Linear convolution commutes with
posterior averaging, so each cached full weighted mean may be convolved once.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import numpy as np
from scipy.signal import fftconvolve
os.environ.setdefault('MPLBACKEND', 'Agg')
HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

OUT = HERE / 'outputs/response_truth_forward_20260905'
CACHE = HERE / 'outputs/injections_and_transfer_20260905'
KERNELS = HERE / 'outputs/kernels'
DT = .02
START = -3000.
END = 15000.
FONT = FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
plt.rcParams.update({'font.family': ['DejaVu Sans', FONT.get_name()], 'font.size': 11,
                     'axes.unicode_minus': False, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.facecolor': 'white'})
COLORS = ['#2457A6', '#C67C18', '#7656A3']
CASES = ['response_within', 'response_beyond']
TRUTH = json.loads((HERE / 'outputs/truth.json').read_text())
REFS = {v['name']: v for v in TRUTH['cases']}


def disk_mass(band):
    data = np.genfromtxt(KERNELS / f'disk_kernel_{band}.csv', delimiter=',', names=True)
    # Use the explicit cell masses used by the actual simulator.
    names = data.dtype.names
    values = data[names[3]]
    assert np.allclose(data[names[0]], np.arange(len(values)) * DT)
    assert abs(values.sum() - 1) < 1e-10
    return values


def compute(band, case, w2):
    source = CACHE / f'{band}_{case}_transfer.npz'
    meta = json.loads(source.with_suffix('.json').read_text())
    run = Path(meta['run'])
    for name, expected in meta['source_files'].items():
        st = (run / 'data' / name).stat()
        assert st.st_size == expected['bytes'] and st.st_mtime_ns == expected['mtime_ns'], 'stale posterior cache'
    with np.load(source) as cached:
        lag = cached['lag']
        fine = START + np.arange(round((END-START)/DT)+1) * DT
        component_means = [np.interp(fine, lag, cached[key], left=0., right=0.)
                           for key in ['mean_small', 'mean_large', 'mean_total']]
        cached_integral = float(np.trapezoid(cached['mean_total'], lag))
    assert np.allclose(component_means[0]+component_means[1], component_means[2], rtol=2e-13, atol=1e-13)
    recovered = [fftconvolve(v, w2, mode='full') for v in component_means]
    x = START + np.arange(len(recovered[0])) * DT
    t = REFS[case]['bands'][band]
    scale = t['slow_scale']
    timescale = case.split('_')[1]
    slow_csv = np.loadtxt(KERNELS / f'slow_{timescale}_kernel.csv', delimiter=',', skiprows=1)
    assert np.allclose(slow_csv[:,0], np.arange(len(slow_csv))*DT)
    slow_mass = slow_csv[:,1] * DT
    assert abs(slow_mass.sum()-1) < 1e-8
    absolute_disk = disk_mass(band) / DT
    absolute_slow = scale * fftconvolve(slow_mass, w2) / DT
    truth = [np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]
    zero = round(-START/DT)
    truth[0][zero:zero+len(absolute_disk)] = absolute_disk
    truth[1][zero:zero+len(absolute_slow)] = absolute_slow
    truth[2] = truth[0] + truth[1]
    mean_areas = [float(v.sum()*DT) for v in recovered]
    truth_areas = [float(v.sum()*DT) for v in truth]
    preconv_areas = [float(v.sum()*DT) for v in component_means]
    convolution_error = max(abs(a-b)/max(abs(a),1e-20) for a,b in zip(mean_areas,preconv_areas))
    area_error = abs(mean_areas[2]-meta['mean_full_support_area'])/meta['mean_full_support_area']
    assert convolution_error < 1e-11
    assert area_error < .002
    assert np.allclose(truth_areas, [1,scale,1+scale], rtol=1e-9)
    assert np.allclose(recovered[0]+recovered[1], recovered[2], rtol=1e-11, atol=1e-12)
    assert all(np.isfinite(v).all() for v in recovered+truth)
    # Convolution can create tiny negative roundoff outside kernel support.
    for v in recovered+truth:
        assert v.min() >= -1e-12*max(1.,v.max())
        v[v<0] = 0.
    info = {'band':band,'case':case,'source':str(source),'source_files':meta['source_files'],
            'reference':'hidden driver; MICA relative posterior mean convolved with W2 absolute disk mass',
            'uncertainty':'mean only; no intervals or uncertainty propagation',
            'dt_days':DT,'lag_range_days':[float(x[0]),float(x[-1])],
            'truth_areas_disk_slow_total':truth_areas,
            'recovered_mean_areas_small_large_total':mean_areas,
            'cached_grid_integral':cached_integral,'analytic_full_mean_area':meta['mean_full_support_area'],
            'total_area_relative_error_including_cache_interpolation':area_error,
            'convolution_area_relative_error':convolution_error,
            'cached_analytic_area_fraction':meta['fraction_mean_area_inside_display'],
            'posterior_mass_retained':meta['posterior_mass_retained'],
            'weighted_joint_draws_in_original_mean':meta['raw_samples']}
    return x, recovered, truth, info


def sparse_indices(x):
    # Export and render at a common multiscale grid; all calculations use DT.
    index = np.arange(len(x))
    keep = ((np.abs(x)<=15) | ((np.abs(x)<=300)&(index%10==0))
            | ((np.abs(x)<=1500)&(index%50==0)) | (index%250==0))
    return keep


def make_figure(band, records):
    fig, axes = plt.subplots(2,3,figsize=(17,8.5))
    fig.subplots_adjust(left=.065,right=.985,top=.82,bottom=.14,hspace=.46,wspace=.28)
    for row, (x, recovered, truth, info) in enumerate(records):
        case = info['case']; mask=sparse_indices(x); xx=x[mask]
        for col, ax in enumerate(axes[row]):
            ax.plot(xx,recovered[col][mask],color=COLORS[col],lw=1.9,zorder=2)
            ax.plot(xx,truth[col][mask],color='#202020',ls='--',lw=1.5,zorder=3)
            ax.axvline(0,color='#c0c0c0',lw=.7); ax.axhline(0,color='#dedede',lw=.6)
            ax.set_ylabel('核增益 / 天');ax.set_xlabel('相对隐藏驱动的延迟（观测者系天）')
            ax.grid(alpha=.16,lw=.6);ax.tick_params(labelsize=10)
            if col==0:
                ax.set_xlim(-1,6 if band=='M2' else 12)
                ax.set_ylim(bottom=0)
                ax.set_title('盘部分：小质心恢复分量 vs 真盘核',loc='left',fontsize=11)
            elif col==1:
                ax.set_xlim((-100,350) if row==0 else (-1800,4500))
                selected=(x>=ax.get_xlim()[0])&(x<=ax.get_xlim()[1])
                ax.set_ylim(0,max(recovered[col][selected].max(),truth[col][selected].max())*1.15)
                ax.set_title('慢部分：大质心恢复分量 vs 真慢核',loc='left',fontsize=11)
                ax.yaxis.set_major_locator(MaxNLocator(4))
            else:
                ax.set_xscale('symlog',linthresh=2)
                ax.set_yscale('symlog',linthresh=1e-7)
                ax.set_xlim(START,END+48); ax.set_ylim(0,max(recovered[col].max(),truth[col].max())*1.2)
                ax.set_xticks([-1000,-10,0,10,100,1000,10000])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_title('总核：完整范围（两轴近零线性）',loc='left',fontsize=11)
            ta=info['truth_areas_disk_slow_total'][col];ra=info['recovered_mean_areas_small_large_total'][col]
            ax.text(.98,.95,f'全核面积：真值 {ta:.3f} / 恢复 {ra:.3f}',transform=ax.transAxes,
                    ha='right',va='top',fontsize=9,color='#444444',bbox={'facecolor':'white','alpha':.8,'edgecolor':'none','pad':1.5})
        pos=axes[row,0].get_position()
        fig.text(.065,pos.y1+.035,'60 天慢核质心' if row==0 else '800 天慢核质心',fontsize=13,weight='bold')
        export = {'lag':xx}
        for col,label in enumerate(['disk','slow','total']):
            export[f'recovered_mean_{label}']=recovered[col][mask]
            export[f'truth_{label}']=truth[col][mask]
        np.savez_compressed(OUT/f'{band}_{case}_forward_curves.npz',**export)
    fig.suptitle(f'F9 {band} · 恢复核与真核：统一到隐藏驱动参考系',fontsize=18,y=.98)
    fig.text(.5,.936,'恢复核与已知 W2 盘核再卷积后比较；没有反卷积，没有重新拟合',ha='center',fontsize=11)
    handles=[Line2D([0],[0],color='#202020',ls='--',lw=1.5,label='真实核'),
             Line2D([0],[0],color=COLORS[0],lw=1.9,label='小质心分量的后验均值'),
             Line2D([0],[0],color=COLORS[1],lw=1.9,label='大质心分量的后验均值'),
             Line2D([0],[0],color=COLORS[2],lw=1.9,label='总核的后验均值')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.915),ncol=4,frameon=False,fontsize=11)
    fig.text(.5,.075,'小/大分量按质心排序；与盘/慢真值并列检验，不预设物理对应。只画完整加权后验均值，不显示后验区间。',ha='center',fontsize=11)
    fig.text(.5,.036,'输入通量单位，不重新归一化；总核保留负延迟。左、中列放大局部，右列展示完整尾部；面积使用完整计算网格。',ha='center',fontsize=10)
    for ext in ['png','pdf']:fig.savefig(OUT/f'F9_{band}_forward_truth_comparison.{ext}',dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);w2=disk_mass('W2');validation=[]
    for band in ['M2','B']:
        records=[]
        for case in CASES:
            result=compute(band,case,w2);records.append(result);validation.append(result[-1])
            print(band,case,'area error',result[-1]['total_area_relative_error_including_cache_interpolation'],flush=True)
        make_figure(band,records)
    (OUT/'validation.json').write_text(json.dumps({'generated_utc':datetime.now(timezone.utc).isoformat(),'records':validation},ensure_ascii=False,indent=2)+'\n')
    (OUT/'README.md').write_text('''# 隐藏驱动参考系的补充核对照

只展示 M2/B 的 response_within 和 response_beyond，不运行新的拟合。

三列分别比较盘部分、慢部分、总核。真实绝对盘核直接读取实际生成器保存的 cell mass；真实绝对慢核为实际慢核乘注入 gain 后与 W2 绝对盘核卷积。真实总核为两者之和。

恢复曲线来自已有完整加权后验 mean_small/mean_large/mean_total；先在 0.02 天网格上线性插值，再与 W2 绝对盘 cell mass 进行完整线性卷积。平均与线性卷积可交换，因此无需重新抽样。没有对点位后验区间作卷积；本图只展示均值，不展示不确定度。

小/大质心标签不自动等于盘/慢物理分量。左右对照检验其对应程度，不能预设分量识别成功。核增益沿用输入通量单位，无额外面积归一化。数值误差、原缓存来源和面积核验见 validation.json。输出 NPZ 是稀疏显示网格，积分验证采用完整 0.02 天网格。

- [M2 PNG](F9_M2_forward_truth_comparison.png)
- [M2 PDF](F9_M2_forward_truth_comparison.pdf)
- [B PNG](F9_B_forward_truth_comparison.png)
- [B PDF](F9_B_forward_truth_comparison.pdf)
''')
    print(str(OUT),flush=True)

if __name__=='__main__':main()
