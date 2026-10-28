from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'comparison_all_s20'
OUT.mkdir(exist_ok=True)
plt.style.use('/home/leonhard/sci001/src/image-creator.mplstyle')
COLORS={1:'#1F4E9E',2:'#C43C39'}

def block(p):
    with p.open() as f:
        assert int(f.readline().strip('# \n'))==1
        sizes=list(map(int,f.readline().strip('# \n').split(':')))
    a=np.loadtxt(p)
    assert sum(sizes)==len(a) and np.isfinite(a).all()
    return np.split(a,np.cumsum(sizes)[:-1])

def load(model,band,n):
    case='cases_highpass_single' if n==1 else 'cases_highpass'
    d=ROOT/case/'all_s20'/'runs/mica'/model/'free'/f'{n}comp'/f'run_UVW2_to_{band}_{n}comp_{model}'
    state=json.loads((d/'run_mica_state.json').read_text())
    obs=block(d/'data/data_input.txt')[1]
    rec=block(d/'data'/f'pall.txt_{n}')[1]
    return obs,rec

metrics=[]
for bands,name in [(['U','B','V'],'UBV'),(['UVM2','UVW1'],'UV')]:
    fig=plt.figure(figsize=(13,3.35*len(bands)))
    gs=fig.add_gridspec(2*len(bands),2,height_ratios=[2.3,1]*len(bands),hspace=.13,wspace=.19)
    for col,model in enumerate(['gamma','gaussian']):
        for row,band in enumerate(bands):
            ax=fig.add_subplot(gs[2*row,col]); ar=fig.add_subplot(gs[2*row+1,col],sharex=ax)
            obs0=None
            for n in [1,2]:
                obs,rec=load(model,band,n)
                if obs0 is None:obs0=obs
                else: assert np.array_equal(obs0,obs),'Inputs differ!'
                valid=(rec[:,0]>=obs[0,0])&(rec[:,0]<=obs[-1,0])
                t,y,e=rec[valid].T
                label=f'{n} component'+('s' if n>1 else '')
                ax.fill_between(t-59000,y-e,y+e,color=COLORS[n],alpha=.13,lw=0)
                ax.plot(t-59000,y,color=COLORS[n],lw=1.2,ls='-' if n==1 else '--',label=label)
                pred=np.interp(obs[:,0],rec[:,0],rec[:,1])
                residual=obs[:,1]-pred
                ar.plot(obs[:,0]-59000,residual,color=COLORS[n],marker='.',ms=2,lw=.55,alpha=.85,ls='-' if n==1 else '--')
                metrics.append(dict(model=model,band=band,ncomp=n,rms_interpolated=float(np.sqrt(np.mean(residual**2))),grid_days=float(np.median(np.diff(rec[:,0])))))
            ax.errorbar(obs[:,0]-59000,obs[:,1],yerr=obs[:,2],fmt='o',ms=2,color='#222222',ecolor='#B8B8B8',elinewidth=.55,capsize=0,zorder=5,label='Data')
            ar.axhline(0,color='#222222',lw=.65)
            ax.set_ylabel(f'{band} flux');ar.set_ylabel('Residual')
            ax.tick_params(labelbottom=False)
            if row==0:
                ax.set_title('Gamma' if col==0 else 'Gaussian',fontsize=14)
                ax.legend(loc='upper right',ncol=3,fontsize=9,frameon=False)
            if row==len(bands)-1: ar.set_xlabel('MJD − 59000 [day]')
            else:ar.tick_params(labelbottom=False)
            for a in [ax,ar]:
                a.xaxis.set_minor_locator(AutoMinorLocator(4));a.yaxis.set_minor_locator(AutoMinorLocator(4));a.set_xlim(obs[0,0]-59000-3,obs[-1,0]-59000+3)
    fig.subplots_adjust(left=.065,right=.99,top=.96,bottom=.065)
    for ext in ['png','pdf']:fig.savefig(OUT/f'single_double_{name}.{ext}',dpi=300)
    plt.close(fig)
(OUT/'diagnostics.json').write_text(json.dumps(metrics,indent=2))
(OUT/'README.md').write_text('all_s20 single versus double component MICA reconstructions. Both driver and responses use original minus Gaussian sigma20-day trend plus original mean. Each panel uses identical native data_input.txt observations (checked exactly). Left: Gamma; right: Gaussian. Blue solid: single; red dashed: double. Black: measurements with supplied errors. Shading: native pall reconstruction mean +/- stored error. Residual = observed minus linearly interpolated native total reconstruction, in the same native flux units as the upper panel. Native grids are approximately 2 days; interpolation residuals are visual diagnostics, not exact likelihood residuals or a formal model comparison. Filter-induced error correlations are not modeled here. No fitting was performed. UBV main figure, ultraviolet supplement.\n')
print(json.dumps(metrics,indent=2))
