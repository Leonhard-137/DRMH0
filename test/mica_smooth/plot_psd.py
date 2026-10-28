"""Exploratory unweighted LS PSD, original vs existing all_s20; no fits launched."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
HERE=Path(__file__).resolve().parent
OUT=HERE/'psd_all_s20'
OUT.mkdir(exist_ok=True)
BANDS=['UVW2','UVM2','UVW1','U','B','V']
rng=np.random.default_rng(817)
plt.style.use(HERE.parents[1]/'src/image-creator.mplstyle')
fig,axes=plt.subplots(3,2,figsize=(9.2,9.4),sharex=True,sharey=True)
summary=[]
for ax,band in zip(axes.flat,BANDS):
 a=np.loadtxt(HERE/'data/original'/f'Mrk817_a_{band}.txt')
 b=np.loadtxt(HERE/'cases_highpass/all_s20/prepared'/f'Mrk817_a_{band}.txt')
 assert a.shape==b.shape and np.allclose(a[:,0],b[:,0],rtol=0,atol=1e-7)
 t=a[:,0]-a[0,0];T=np.ptp(t);N=len(t);mean=a[:,1].mean()
 W=np.exp(-0.5*((t[:,None]-t[None,:])/20)**2);W/=W.sum(axis=1,keepdims=True)
 assert np.allclose(b[:,1],a[:,1]-W@a[:,1]+mean,rtol=1e-9,atol=1e-8)
 # Conservative display ceiling, not a claimed irregular-sampling Nyquist limit.
 freq=np.arange(1/T,0.30,1/(5*T))
 def power(y):
  return LombScargle(t,y,fit_mean=True,normalization='psd').power(freq,method='cython')*(2*T/N)
 po=power(a[:,1]/mean-1);ph=power((b[:,1]-mean)/mean)
 noises=rng.normal(size=(64,N))*a[:,2][None,:]/mean
 no=np.mean([power(n) for n in noises],axis=0)
 nh=np.mean([power(n-W@n) for n in noises],axis=0)
 edges=np.geomspace(freq[0],freq[-1]*(1+1e-10),24)
 rows=[]
 for lo,hi in zip(edges[:-1],edges[1:]):
  mask=(freq>=lo)&(freq<hi)
  if mask.any(): rows.append([np.exp(np.mean(np.log(freq[mask]))),*[np.mean(p[mask]) for p in (po,ph,no,nh)],mask.sum()])
 bins=np.array(rows)
 ax.loglog(freq,po,color='#222222',lw=.45,alpha=.18,rasterized=True)
 ax.loglog(freq,ph,color='#1F4E9E',lw=.45,alpha=.20,rasterized=True)
 ax.loglog(bins[:,0],bins[:,1],color='#222222',lw=1.6,label='Original')
 ax.loglog(bins[:,0],bins[:,2],color='#1F4E9E',lw=1.6,label=r'High-pass ($\sigma=20$ d)')
 ax.loglog(bins[:,0],bins[:,3],color='#B8B8B8',lw=1,ls='--',label='Measurement noise (original)')
 ax.loglog(bins[:,0],bins[:,4],color='#009E73',lw=1,ls='-.',label='Measurement noise (filtered)')
 ax.set_title(band,fontsize=12)
 ax.set_xlim(.0023,.3)
 if ax in axes[0]:
  def reciprocal(x): return 1/np.maximum(np.asarray(x),1e-12)
  sec=ax.secondary_xaxis('top',functions=(reciprocal,reciprocal))
  sec.set_xticks([300,100,30,10,4],labels=['300','100','30','10','4'])
  sec.set_xlabel('Timescale 1/f (days)')
 np.savetxt(OUT/f'{band}_periodogram.csv',np.column_stack([freq,po,ph,no,nh]),delimiter=',',header='frequency_day^-1,original,highpass_s20,noise_original,noise_filtered',comments='')
 np.savetxt(OUT/f'{band}_binned.csv',bins,delimiter=',',header='frequency_day^-1,original,highpass_s20,noise_original,noise_filtered,grid_points',comments='')
 summary.append(dict(band=band,n=N,span_days=T,median_cadence_days=float(np.median(np.diff(t))),max_gap_days=float(np.max(np.diff(t))),low_frequency_power_ratio=float(np.sum(ph[freq<.01])/np.sum(po[freq<.01]))))
for ax in axes[-1]: ax.set_xlabel(r'Frequency (day$^{-1}$)')
for ax in axes[:,0]: ax.set_ylabel(r'Approx. fractional PSD (day)')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=10)
fig.tight_layout(rect=(0,.075,1,1))
fig.savefig(OUT/'psd_original_vs_all_s20.png',dpi=300)
fig.savefig(OUT/'psd_original_vs_all_s20.pdf')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'README.md').write_text('''# Mrk 817 exploratory periodograms\n\nOriginal input: ../data/original; filtered input: ../cases_highpass/all_s20/prepared. All times are observer-frame days.\n\nUnweighted, floating-mean Lomb–Scargle on fractional flux using the ORIGINAL mean for both series. No interpolation. Astropy psd normalization multiplied by 2T/N: a uniform-sampling-equivalent one-sided fractional PSD normalization, approximate for these irregular times. This is a sampled/dirty periodogram, not a window-deconvolved intrinsic PSD.\n\nGrid: 1/T to 0.30 per day, spacing 1/(5T). Display ceiling is not a strict Nyquist limit. Thin curves are raw ordinates; thick curves are arithmetic means in 23 logarithmic frequency bins. Oversampled ordinates and bins need not be independent; no confidence intervals or significance are claimed.\n\nNoise curves average 64 independent Gaussian measurement-noise realizations using original quoted errors and observation times. Filtered noise is transformed by the same normalized Gaussian matrix (I-W); the restored constant mean is absorbed by LS floating mean. These curves estimate measurement noise only, not red-noise/window uncertainty. Powers are not noise-subtracted.\n\nNo PSD slope, break, periodicity or physical component is fitted. Frequency below 1/T is not displayed; longest timescales are poorly sampled. LS power is not additive between trend and residual because of their cross term. The summary low-frequency ratio is a descriptive ratio of sampled periodogram powers below 0.01/day, not an intrinsic variance fraction.\n\nReference: https://docs.astropy.org/en/stable/timeseries/lombscargle.html\n''')
print(json.dumps(summary,indent=2))
