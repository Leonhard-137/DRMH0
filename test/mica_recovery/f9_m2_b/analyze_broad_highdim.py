#!/usr/bin/env python3
"""Explore the completed F9 broad-prior six-dimensional transfer posterior.

Chart contract: scientific PNG/PDF scatter, parallel-coordinate, stability and
CDF figures. PCA is a display only; HDBSCAN operates in all six coordinates.
All raw weights are retained for moments/PCA. HDBSCAN uses stratified posterior
resamples with size near importance ESS; cluster fractions are Monte Carlo
estimates, not exact raw-weight masses or independent-run uncertainty.
No fit, prior, likelihood, or science result is changed by this script.
"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig')
import sys,json,csv,itertools
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.special import gammainc
from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import BoundaryNorm
from collect_mica_results import (param_names,normalized_weights,weighted_quantile,
    reorder_component_columns,component_parameter_arrays)

OUT=HERE/'outputs/highdim_broad_20260905'
BANDS=['M2','W1','U','B','V']
FEATURES=['tau_short','ln_width_short','ln_area_short','tau_long','ln_width_long','ln_area_long']
XLABELS=['小质心\nτs（天）','小分量\nln ws','小分量\nln As','大质心\nτl（天）','大分量\nln wl','大分量\nln Al']
SEEDS=[42,2026,917]
SETTINGS=[(f,k) for f in [.05,.10] for k in [10,25]]
COLORS={-1:'#B4B8C0',0:'#356BB3',1:'#CE8131',2:'#7A579A',3:'#388875'}
FONT=FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
plt.rcParams.update({'font.family':['DejaVu Sans',FONT.get_name()],'axes.unicode_minus':False,
 'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)

def resample(w,n,seed):
    rng=np.random.default_rng(seed)
    perm=rng.permutation(len(w));cdf=np.cumsum(w[perm]);cdf[-1]=1
    return perm[np.searchsorted(cdf,(np.arange(n)+rng.random(n))/n)]

def read(band):
    run=ROOT/f'F9_a/runs/mica/gamma/broad_np4/2comp/run_W2_to_{band}_2comp_gamma'
    d=run/'data';raw=np.loadtxt(d/'sample1d.txt_2');wt=np.loadtxt(d/'weights.txt')
    info=np.loadtxt(d/'sample_info1d.txt_2');names=param_names(d/'para_names_line.txt_2')
    assert raw.shape==(10000,len(names)) and len(wt)==len(info)==len(raw)
    assert np.allclose(wt[:,1],info[:,1],rtol=1e-6,atol=1e-5)
    log=(run.parents[1]/'_logs'/f'run_W2_to_{band}_2comp_gamma_run.log').read_text()
    assert 'end RM modeling' in log and 'Effective sample size =' in log
    w=normalized_weights(wt[:,-1]);ordered,swaps=reorder_component_columns(raw,names,2,'gamma',{'component_order':'moment1'})
    onset,width,amp,tau=component_parameter_arrays(ordered,names,2,'gamma')
    X=np.column_stack([tau[0],np.log(width[0]),np.log(amp[0]),tau[1],np.log(width[1]),np.log(amp[1])])
    mu=w@X;sd=np.sqrt(w@((X-mu)**2));q=np.array([weighted_quantile(X[:,i],w,[.25,.75]) for i in range(6)])
    scales={'sd':sd,'iqr':(q[:,1]-q[:,0])/1.349}
    assert all(np.all(s>0) for s in scales.values())
    Z=(X-mu)/sd;ev,evec=np.linalg.eigh((Z*w[:,None]).T@Z);ix=np.argsort(ev)[::-1];ev=ev[ix];evec=evec[:,ix]
    for i in range(6):
        if evec[np.argmax(np.abs(evec[:,i])),i]<0:evec[:,i]*=-1
    explained=ev/ev.sum();ess=1/np.sum(w*w);n=min(1800,int(ess));idx=resample(w,n,42)
    return dict(band=band,source=str(d),X=X,w=w,tau=tau,width=width,onset=onset,amp=amp,
      frac=amp/amp.sum(axis=0),mu=mu,scales=scales,pca=Z@evec,loadings=evec,
      explained=explained,ess=ess,n=n,idx=idx,swap_mass=float(w[swaps].sum()))

def cluster(a,scale,seed,frac,mins):
    idx=resample(a['w'],a['n'],seed);Z=(a['X']-a['mu'])/a['scales'][scale]
    model=HDBSCAN(min_cluster_size=max(20,round(frac*len(idx))),min_samples=mins,
        allow_single_cluster=True,n_jobs=1,copy=True).fit(Z[idx])
    labels=model.labels_.copy()
    order=sorted(set(labels)-{-1},key=lambda k:np.median(a['tau'][0,idx[labels==k]]))
    for new,old in enumerate(order):labels[model.labels_==old]=new
    return idx,labels

def groups(a,idx,labels):
    result=[]
    for k in sorted(set(labels)):
        ii=idx[labels==k];total=(a['frac'][:,ii]*a['tau'][:,ii]).sum(axis=0)
        metrics={'tau_short':a['tau'][0,ii],'tau_long':a['tau'][1,ii],
          'width_short':a['width'][0,ii],'width_long':a['width'][1,ii],
          'fraction_short':a['frac'][0,ii],'total_lag':total}
        result.append({'label':int(k),'estimated_mass':len(ii)/len(idx),'n_resampled':len(ii),
          'metrics':{name:np.quantile(v,[.15865,.5,.84135]).tolist() for name,v in metrics.items()}})
    return result

def run_sweep(a):
    rows=[];fits={}
    for scale,seed,(frac,mins) in itertools.product(['sd','iqr'],SEEDS,SETTINGS):
        idx,labels=cluster(a,scale,seed,frac,mins)
        key=f'{scale}_seed{seed}_f{frac:g}_k{mins}'
        fits[key]=(idx,labels)
        rows.append({'band':a['band'],'scale':scale,'seed':seed,'min_cluster_fraction':frac,
          'min_samples':mins,'cluster_count':len(set(labels)-{-1}),
          'unassigned_fraction_estimate':float(np.mean(labels==-1)),'key':key})
    baseidx,baselabels=fits['sd_seed42_f0.05_k25']
    a['base_idx']=baseidx;a['labels']=baselabels;a['sweep']=rows
    base_map=dict(zip(map(int,baseidx),map(int,baselabels)))
    for row in rows:
        idx,labels=fits[row['key']];m=dict(zip(map(int,idx),map(int,labels)))
        common=sorted(set(m)&set(base_map));row['common_raw_ids']=len(common)
        row['ari_on_common_raw_ids_including_unassigned']=float(adjusted_rand_score([base_map[i] for i in common],[m[i] for i in common]))
    return rows

def exchange_check(a):
    # Both native components have identical priors. Same scales for both tuples
    # make min(direct, swapped) a label-invariant metric on the quotient space.
    bounds=[]
    for line in (Path(a['source'])/'para_names_line.txt_2').read_text().splitlines():
        p=line.split()
        if len(p)>4 and p[1] in ['0-th_component_centroid','0-th_component_sigma','0-th_component_amplitude']:
            bounds.append((p[1],float(p[4])-float(p[3])))
    bd=dict(bounds);s=np.array([bd['0-th_component_centroid'],bd['0-th_component_sigma'],bd['0-th_component_amplitude']]*2)
    Z=a['X'][a['idx']]/s
    D=np.minimum(cdist(Z,Z),cdist(Z,Z[:,[3,4,5,0,1,2]]));D=(D+D.T)/2;np.fill_diagonal(D,0)
    result=[]
    for f,k in SETTINGS:
        labels=HDBSCAN(min_cluster_size=max(20,round(f*a['n'])),min_samples=k,
          allow_single_cluster=True,metric='precomputed',copy=True,n_jobs=1).fit_predict(D)
        result.append({'min_cluster_fraction':f,'min_samples':k,'cluster_count':len(set(labels)-{-1}),
          'unassigned_fraction_estimate':float(np.mean(labels==-1)),
          'groups_estimated_from_resample':groups(a,a['idx'],labels),
          'overlap_with_baseline_counts':{str(int(g)):{str(int(k)):int(np.sum((labels==g)&(a['labels']==k)))
            for k in sorted(set(a['labels']))} for g in sorted(set(labels))}})
    return result

def plot_overview(all_data):
    fig,axes=plt.subplots(2,3,figsize=(14,9));fig.subplots_adjust(wspace=.28,hspace=.4,bottom=.12,top=.90,right=.92)
    for ax,a in zip(axes.flat,all_data):
        i=a['idx'];P=a['pca'][i];h=ax.scatter(P[:,0],P[:,1],c=a['frac'][0,i],cmap='viridis',vmin=0,vmax=1,s=8,alpha=.65,rasterized=True,linewidths=0)
        ax.set_title(f"{a['band']} · 前两轴保留 {100*a['explained'][:2].sum():.1f}% 线性方差",loc='left')
        ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.grid(alpha=.15)
    axes.flat[-1].axis('off');axes.flat[-1].text(0,.9,'每个点是一组完整的六参数样本。\n颜色表示小质心分量的面积占比。\n\nPCA 只负责投影；\n聚类在完整六维空间进行。\n\n各波段单独标准化、单独投影，\n不同面板的坐标不能直接比较。',va='top',linespacing=1.8)
    cax=fig.add_axes([.945,.23,.012,.55]);fig.colorbar(h,cax=cax,label='小质心分量的面积占比')
    fig.suptitle('F9 宽先验：六维联合后验的二维概览',fontsize=16)
    fig.text(.08,.035,'幅度与宽度用自然对数；质心用天。PCA 按原始后验权重计算；散点为按权重重采样的展示。\n“看起来分开”需要六维聚类与尺度敏感性检查支持，不能单凭此图决定分支。',fontsize=10)
    save(fig,'01_five_band_pca')

def plot_stability(all_data):
    mat=np.array([[r['cluster_count'] for r in a['sweep']] for a in all_data]);cmap=plt.get_cmap('Blues',int(mat.max())+1)
    fig,ax=plt.subplots(figsize=(15,4.8));fig.subplots_adjust(bottom=.31,top=.84,right=.94)
    im=ax.imshow(mat,aspect='auto',cmap=cmap,norm=BoundaryNorm(np.arange(-.5,mat.max()+1.5),cmap.N))
    for y in range(len(mat)):
        for x in range(mat.shape[1]):ax.text(x,y,str(mat[y,x]),ha='center',va='center',color='white' if mat[y,x]>=2 else '#222',fontsize=10)
    ax.set_yticks(range(5),BANDS)
    labels=[f"{r['seed']}\n{int(100*r['min_cluster_fraction'])}% / {r['min_samples']}" for r in all_data[0]['sweep']]
    ax.set_xticks(range(len(labels)),labels,fontsize=7.5)
    ax.axvline(11.5,color='#222',lw=2)
    ax.set_title('检出的密集簇数量：左半按标准差缩放；右半按四分位距缩放',loc='left',fontsize=12)
    fig.suptitle('六维 HDBSCAN：24 种设置的敏感性筛查',fontsize=15)
    fig.text(.08,.095,'每列：重采样种子 / 最小簇比例 / 邻居数。允许一个簇，不预设必须分成两组。\n“1”仅表示检出一个密集核心，不代表唯一解；未归簇样本仍属于后验，也不能称为非物理。',fontsize=10)
    save(fig,'02_cluster_stability')

def weighted_ranks(X,w):
    result=np.zeros_like(X)
    for j in range(X.shape[1]):
        o=np.argsort(X[:,j],kind='stable');cdf=np.cumsum(w[o])-.5*w[o];result[o,j]=cdf
    return result

def plot_v(a):
    i=a['base_idx'];labels=a['labels'];P=a['pca'][i]
    fig,axes=plt.subplots(2,2,figsize=(13.4,9));fig.subplots_adjust(hspace=.4,wspace=.29,top=.9,bottom=.12)
    for ax,(j,k) in zip(axes[0],[(0,1),(0,2)]):
        for lab in sorted(set(labels)):
            mask=labels==lab
            ax.scatter(P[mask,j],P[mask,k],s=8,color=COLORS[lab],alpha=.55,linewidths=0,rasterized=True,
             label=('未归入核心' if lab<0 else f'密集核心 {lab+1}')+f' · {np.mean(mask):.0%}')
        ax.set(xlabel=f'PC{j+1}',ylabel=f'PC{k+1}');ax.legend(frameon=False,fontsize=8)
        ax.set_title(f'投影 PC{j+1}–PC{k+1}；颜色来自六维聚类',loc='left',fontsize=11)
    ax=axes[1,0];R=weighted_ranks(a['X'],a['w']);order=np.random.default_rng(42).permutation(len(i))
    for z in order:ax.plot(range(6),R[i[z]],color=plt.cm.viridis(a['frac'][0,i[z]]),alpha=.045,lw=.6,rasterized=True)
    ax.set_xticks(range(6),XLABELS,fontsize=8);ax.set_ylim(0,1);ax.set_ylabel('各参数在边际后验中的百分位')
    ax.set_title('六轴平行坐标：一条线是一个联合样本\n颜色表示小分量面积占比（紫色低，黄色高）',loc='left',fontsize=10)
    ax=axes[1,1]
    sc=ax.scatter(a['tau'][0,i],a['frac'][0,i],c=np.log10(a['width'][1,i]),cmap='cividis',s=9,alpha=.65,rasterized=True,linewidths=0)
    ax.set(xlabel='小质心 τs（天）',ylabel='小分量的面积占比',ylim=(0,1))
    ax.set_title('直接参数关系：颜色表示大分量宽度',loc='left',fontsize=11)
    fig.colorbar(sc,ax=ax,label='log10 wl（天）',fraction=.045,pad=.02)
    fig.suptitle('V：分解结构明显，但尚未稳定分成多个簇',fontsize=16)
    cum=np.cumsum(a['explained'])
    fig.text(.07,.027,f"前两轴保留 {cum[1]:.1%}、前三轴保留 {cum[2]:.1%} 的线性方差。平行坐标用百分位统一显示范围，聚类使用原六维变换后的坐标。\n上排仅展示基准设置（标准差缩放、5%最小簇、25邻居、种子42）；灰色不是坏样本，未删除。",fontsize=9.5)
    save(fig,'03_V_six_dimensional')

def plot_b(a):
    i=a['base_idx'];labels=a['labels'];P=a['pca'][i];fig,axes=plt.subplots(2,2,figsize=(13,9));fig.subplots_adjust(hspace=.38,wspace=.30,top=.90,bottom=.13)
    for lab in sorted(set(labels)):
        m=labels==lab;ii=i[m];color=COLORS[lab];name='未归簇' if lab<0 else f'候选组 {lab+1}'
        label=f'{name} · 约 {np.mean(m):.1%}'
        axes[0,0].scatter(P[m,0],P[m,1],s=9,color=color,alpha=.65,rasterized=True,linewidths=0,label=label)
        axes[0,1].scatter(a['tau'][0,ii],a['tau'][1,ii],s=9,color=color,alpha=.65,rasterized=True,linewidths=0)
        if lab<0:continue
        for ax,grid in zip(axes[1],[np.linspace(-20,20,600),np.linspace(-150,300,800)]):
            cdf=(a['frac'][:,ii,None]*gammainc(2,np.maximum(0,(grid[None,None,:]-a['onset'][:,ii,None])/a['width'][:,ii,None]))).sum(axis=0)
            q=np.quantile(cdf,[.15865,.5,.84135],axis=0)
            ax.plot(grid,q[1],color=color,label=name,lw=1.7);ax.fill_between(grid,q[0],q[2],color=color,alpha=.15)
    axes[0,0].set(xlabel='PC1',ylabel='PC2');axes[0,0].legend(frameon=False,fontsize=9)
    axes[0,0].set_title('完整六维中检出的候选组，投影到二维',loc='left',fontsize=11)
    axes[0,1].set(xlabel='小质心 τs（天）',ylabel='大质心 τl（天）');axes[0,1].set_title('两组的小质心接近，其他参数仍不同',loc='left',fontsize=11)
    for ax in axes[1]:ax.set(xlabel='lag（天）',ylabel='总转移函数累计面积',ylim=(0,1));ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    axes[1,0].set_title('总核的主体：中位数及 68% 区间',loc='left',fontsize=11);axes[1,1].set_title('总核的宽范围：候选组是否对应不同形状',loc='left',fontsize=11)
    fig.suptitle('B：主要分组较清楚，细分依赖坐标与分辨率',fontsize=16)
    fig.text(.07,.03,'基准设置：标准差缩放、5%最小簇、25邻居、种子42。比例及组内区间来自后验重采样，是近似估计。\n每个样本先归一化总核，再计算累计面积。候选组不是已确认的物理成分；未改变先验、未筛除后验。',fontsize=9.5)
    save(fig,'04_B_candidate_groups')

def main():
    OUT.mkdir(parents=True,exist_ok=True);all_data=[];summaries=[];all_rows=[]
    for band in BANDS:
        a=read(band);all_rows.extend(run_sweep(a));all_data.append(a)
        counts=[r['cluster_count'] for r in a['sweep']]
        summary={'band':band,'source':a['source'],'features':FEATURES,'raw_rows':len(a['w']),'importance_ess':a['ess'],'resample_size':a['n'],
          'unique_raw_ids_base':len(np.unique(a['base_idx'])),'label_swap_mass':a['swap_mass'],
          'weighted_pca_explained_fractions':a['explained'].tolist(),'pca_loadings_columns_are_pcs':a['loadings'].tolist(),
          'pca_dimension_for_90pct_linear_variance':int(np.searchsorted(np.cumsum(a['explained']),.9)+1),
          'cluster_count_range':[min(counts),max(counts)],'multi_cluster_settings':sum(k>=2 for k in counts),'total_settings':len(counts),
          'baseline_key':'sd_seed42_f0.05_k25','baseline_groups':groups(a,a['base_idx'],a['labels']),
          'sweep':a['sweep'],'fraction_with_centroid_gap_lt_5days':float(a['w'][(a['tau'][1]-a['tau'][0])<5].sum())}
        if band in ['B','V']:
            summary['exchange_invariant_prior_scaled_check']=exchange_check(a)
            summary['width_gt_200d_raw_weight_mass']=float(a['w'][np.max(a['width'],axis=0)>200].sum())
            summary['width_gt_200d_note']='Descriptive raw-weight region probability, not an exact propagated cluster mass or a posterior cut.'
        summaries.append(summary)
        np.savez_compressed(OUT/f'{band}_joint_samples.npz',features=a['X'],weights=a['w'],pca=a['pca'],
          baseline_raw_indices=a['base_idx'],baseline_labels=a['labels'])
        print(band,'counts',summary['cluster_count_range'],'multi',summary['multi_cluster_settings'],'PCA',np.round(a['explained'][:3],3),flush=True)
    doc={'method':'HDBSCAN in six transformed dimensions, weighted PCA for display; bounded sensitivity screen, not a proof of modes or uniqueness',
      'cluster_mass_note':'Fractions are equal-weight stratified posterior resample estimates; labels are not propagated to all original rows. No raw sample is removed.',
      'settings':{'seeds':SEEDS,'scales':['posterior_sd','posterior_iqr/1.349'],'min_cluster_fraction':[.05,.1],'min_samples':[10,25],'allow_single_cluster':True},'bands':summaries}
    (OUT/'results.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'cluster_sensitivity.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    candidate_rows=[]
    for summary in summaries:
        for group in summary['baseline_groups']:
            row={'band':summary['band'],'candidate_id':group['label'],
              'resampled_posterior_fraction_estimate':group['estimated_mass'],'n_resampled':group['n_resampled']}
            for key,values in group['metrics'].items():
                for suffix,value in zip(['q16','median','q84'],values):row[f'{key}_{suffix}']=value
            candidate_rows.append(row)
    with (OUT/'candidate_groups.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(candidate_rows[0]));writer.writeheader();writer.writerows(candidate_rows)
    plot_overview(all_data);plot_stability(all_data);plot_v(all_data[-1]);plot_b(all_data[-2])
    print('saved',OUT,flush=True)
if __name__=='__main__':main()
