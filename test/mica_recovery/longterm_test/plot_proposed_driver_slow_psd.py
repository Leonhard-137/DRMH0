"""Analytic PSDs for proposed parameters; does not generate data or launch fits."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import configure_style, style_axis, save_figure, COLORS
HERE=Path(__file__).resolve().parent
root=HERE/'cases_fast_driver/disk_blr_uonly_gp_tauI0p5'
g=json.loads((root/'config/generation.json').read_text())
tau_i,alpha,tau_s=2.,1.45,g['parameters']['independent_gp_tau_days']
# Rescale the preserved slow realization to sample RMS 1.3. Its stationary
# process amplitude differs from its sample RMS over the finite observation.
slow_amp=g['parameters']['independent_gp_amplitude']*1.3/g['statistics']['U']['realized_rms']['independent_S']
f=np.geomspace(1e-5,10,6000)
pi=4*alpha**2*tau_i/(1+(2*np.pi*f*tau_i)**2)
ps=8*slow_amp**2*tau_s/(1+(2*np.pi*f*tau_s)**2)**2
configure_style()
fig,axs=plt.subplots(1,2,figsize=(11,4.2))
for ax,weight in zip(axs,[np.ones_like(f),f]):
 ax.loglog(f,pi*weight,color=COLORS['PRIMARY'],lw=1.8,label=r'Central: $\tau_I=2$ d')
 ax.loglog(f,ps*weight,color=COLORS['ALTERNATIVE'],lw=1.8,label=r'Slow: $\tau_S=55.37$ d')
 for period,ls in [(100,':'),(10,'--'),(5,'-.')]:
  ax.axvline(1/period,color='#999999',ls=ls,lw=.8,label=f'{period} d')
 ax.set_xlim(1e-5,10)
 ax.set_xlabel(r'$f$ (day$^{-1}$)')
 style_axis(ax,y_on_right=False)
axs[0].set_ylabel(r'$P(f)$ (flux$^2$ day)')
axs[1].set_ylabel(r'$fP(f)$ (flux$^2$)')
axs[0].legend(frameon=False,fontsize=10)
fig.tight_layout()
out=root/'results/proposed_tau2_driver_slow_psd'
save_figure(fig,out)
print(out.with_suffix('.jpg'))
print('stationary RMS: central=',alpha,'slow=',slow_amp)
print('slow sample RMS target=1.3; no new draw generated')
