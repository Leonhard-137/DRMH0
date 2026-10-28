"""Paper-style NK22 figure analogues, with reproducible labels and statistics."""
from pathlib import Path
import csv, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator,MaxNLocator
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs";FIG=OUT/"figures"
sys.path.insert(0,"/mnt/c/Users/26330/.codex/skills/image-creator/scripts")
from plot_style import apply_style,configure_axes

# NK22 visual language overrides general project role colors:
# black observations, red/cyan alternative fits, discretized diverging maps.
apply_style({"font.size":10,"axes.labelsize":11,"axes.titlesize":11,
             "xtick.labelsize":8,"ytick.labelsize":8,"axes.linewidth":.7,
             "xtick.major.size":4,"ytick.major.size":4})
CMAP=plt.get_cmap("coolwarm",20)
CAPTIONS=[]


def save(fig,name,pdf,caption):
    fig.savefig(FIG/(name+".pdf"),bbox_inches="tight")
    fig.savefig(FIG/(name+".png"),dpi=300,bbox_inches="tight")
    fig.savefig(FIG/(name+"_preview.jpg"),dpi=110,bbox_inches="tight")
    pdf.savefig(fig,bbox_inches="tight")
    CAPTIONS.append((name,caption));plt.close(fig)


def map_axis(ax,z,key,tp,stats=None,scale=None):
    q=z[key]
    s=float(np.quantile(abs(q),.99)) if scale is None else float(scale)
    im=ax.pcolormesh(tp,np.log10(z["u"]),q/max(s,1e-12),shading="nearest",
                     vmin=-1,vmax=1,cmap=CMAP,rasterized=True)
    ax.set(xlim=(tp[0],tp[-1]),ylim=(0,3),yticks=[0,1,2,3])
    ax.set_xticks([6650,6750,6850])
    ax.set_xlim(tp[0],tp[-1])
    configure_axes(ax)
    if stats is not None:
        ax.text(.04,.06,r"$\chi^2/N_d=$"+f"{stats['chi2_per_point']:.2f}"+"\n"+
                f"scale = {s:.2f}",transform=ax.transAxes,fontsize=8,
                bbox=dict(facecolor="white",alpha=.65,edgecolor="none",pad=1))
    return im


def grid_figure(z,m,pdf):
    fig,axes=plt.subplots(4,4,figsize=(12,11),sharex=True,sharey=True)
    for row,nt in enumerate([50,100,250,500]):
        for col,xi in enumerate([1,10,100,1000]):
            key=f"local_nt{nt}_xi{xi}"
            im=map_axis(axes[row,col],z,"q_"+key,z[f"local_tp_{nt}"],m[key])
            if row==0:axes[row,col].set_title(r"$\xi=$"+str(xi))
        axes[row,-1].text(1.03,.5,r"$N_t=$"+str(nt),rotation=-90,va="center",ha="left",transform=axes[row,-1].transAxes)
    fig.subplots_adjust(left=.07,right=.87,bottom=.075,top=.94,wspace=.035,hspace=.055)
    fig.supxlabel("HJD - 2450000 [observer days]",y=.025)
    fig.supylabel(r"$\log_{10}(R/R_{\rm in})$",x=.015)
    cax=fig.add_axes([.925,.16,.018,.65])
    fig.colorbar(im,cax=cax,label=r"$(\delta T/T)/{\rm scale}$",ticks=np.linspace(-1,1,9))
    fig.suptitle("NGC 5548 — local STORM 1 data",y=.985)
    save(fig,"01_grid_regularization",pdf,
         "NK22 Fig.5 analogue. Primary local 4203 points; Nr=50; each map independently divided by its 99th percentile of |deltaT/T|. Full local time span retained. Low-xi amplitudes can exceed the small-perturbation regime; colors alone are not significance.")


def moderate(z,m,pdf):
    fig,axs=plt.subplots(1,4,figsize=(13,3.9),sharey=True)
    for ax,xi in zip(axs,[10,30,50,100]):
        key=f"local_nt250_xi{xi}"
        im=map_axis(ax,z,"q_"+key,z["local_tp_250"],m[key])
        ax.set_title(r"$\xi=$"+str(xi))
    fig.subplots_adjust(left=.055,right=.92,bottom=.19,top=.88,wspace=.035)
    axs[0].set_ylabel(r"$\log_{10}(R/R_{\rm in})$")
    fig.supxlabel("HJD - 2450000 [observer days]",y=.025)
    cax=fig.add_axes([.935,.19,.015,.69]);fig.colorbar(im,cax=cax,label=r"$(\delta T/T)/{\rm scale}$")
    save(fig,"02_moderate_regularization",pdf,
         "NK22 Fig.7 analogue, Nt=250. Independent scale per panel. No trajectories are fitted or overlaid: NK22's highlighted velocities were chosen by eye, and are not measurements reproduced here.")


def lightcurves(z,pdf):
    fig,axs=plt.subplots(5,4,figsize=(13,9),sharex=True)
    t,f,e,bi=(z["local_"+k] for k in ["t","flux","error","bi"])
    p1=z["pred_local_nt250_xi1"];p2=z["pred_local_nt250_xi1000"]
    for b,ax in enumerate(axs.flat):
        if b==19:ax.axis("off");continue
        ii=bi==b
        shift=(f[ii].max()+f[ii].min())/2
        scale=np.ptp(f[ii])/2
        ax.errorbar(t[ii],(f[ii]-shift)/scale,e[ii]/scale,fmt=".",color="black",ms=1.8,elinewidth=.4)
        ax.plot(t[ii],(p1[ii]-shift)/scale,color="red",lw=.75,label=r"$\xi=1$")
        ax.plot(t[ii],(p2[ii]-shift)/scale,color="#00A9C9",lw=.85,label=r"$\xi=1000$")
        ax.text(.97,.08,str(z["local_bands"][b]),ha="right",transform=ax.transAxes,fontsize=9)
        ax.set(xlim=(t.min(),t.max()),yticks=[-1,0,1],ylim=(-1.65,1.65))
        ax.set_xticks([6650,6750,6850]);configure_axes(ax)
    axs.flat[0].legend(loc="upper left",fontsize=7,ncol=2)
    fig.subplots_adjust(left=.07,right=.99,bottom=.09,top=.94,wspace=.17,hspace=.08)
    fig.supxlabel("HJD - 2450000 [observer days]",y=.025)
    fig.supylabel("Normalized flux",x=.015)
    fig.suptitle(r"NGC 5548 — observed and reconstructed light curves ($N_t=250$)")
    save(fig,"03_lightcurve_fits",pdf,
         "NK22 Fig.6 analogue. Black data, red xi=1, cyan xi=1000. Each band uses the same affine transform for data and both fits: (flux-midrange)/(half-range). No post-fit multiplicative refitting.")


def residuals(z,pdf):
    fig,axs=plt.subplots(5,4,figsize=(13,9),sharex=True)
    t,f,e,bi=(z["local_"+k] for k in ["t","flux","error","bi"])
    pred=z["pred_local_nt250_xi10"];rows=[]
    for b,ax in enumerate(axs.flat):
        if b==19:ax.axis("off");continue
        ii=bi==b;r=f[ii]-pred[ii]
        ax.errorbar(t[ii],r,e[ii],fmt=".",color="black",ms=1.9,elinewidth=.4)
        ax.axhline(0,color="red",lw=.7,ls="--")
        chi=float(np.mean((r/e[ii])**2))
        ax.text(.97,.08,str(z["local_bands"][b])+f"\n"+r"$\chi^2/N_b=$"+f"{chi:.2f}",ha="right",transform=ax.transAxes,fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(3));ax.set_xticks([6650,6750,6850])
        ax.set_xlim(t.min(),t.max());configure_axes(ax)
        rows.append(dict(band=str(z["local_bands"][b]),n=int(ii.sum()),chi2_per_point=chi,
                         rms_residual=float(np.sqrt(np.mean(r*r))),
                         residual_flux_pearson=float(np.corrcoef(r,f[ii])[0,1])))
    fig.subplots_adjust(left=.075,right=.99,bottom=.09,top=.94,wspace=.29,hspace=.1)
    fig.supxlabel("HJD - 2450000 [observer days]",y=.025)
    fig.supylabel(r"Residual [$10^{-15}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]",x=.01)
    fig.suptitle(r"NGC 5548 — residuals ($N_t=250,\ \xi=10$)")
    save(fig,"04_residuals_all_bands",pdf,"Residual=data-prediction, in observed flux units. Error bars show triplet-adjusted input errors, not full covariance of a fitted residual. Different vertical scales per band. U/u residuals are not automatically BLR light curves.")
    (OUT/"band_residuals.json").write_text(json.dumps(rows,indent=2))


def coverage(z,pdf):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5),gridspec_kw=dict(width_ratios=[1.5,1]))
    for b,band in enumerate(z["local_bands"]):
        t=z["local_t"][z["local_bi"]==b]
        color="black" if str(band).startswith("HST") else ("#C43C39" if str(band) in ["W2","M2","W1","Swift U","Swift B","Swift V"] else "#1F4E9E")
        ax.plot(t,np.full(len(t),b),"|",ms=4,color=color,alpha=.8)
        k=z["local_kernel"][b];bx.plot(np.log10(z["u"]),k/k.max(),color=plt.cm.turbo(b/18),lw=1)
    ax.set(yticks=np.arange(19),yticklabels=z["local_bands"],xlabel="HJD - 2450000 [observer days]",title="Time coverage: 4,203 measurements")
    ax.invert_yaxis();ax.yaxis.set_minor_locator(plt.NullLocator())
    bx.set(xlabel=r"$\log_{10}(R/R_{\rm in})$",ylabel="Normalized response per log-radius cell",title="Radial kernels",xlim=(0,3))
    configure_axes(bx);fig.tight_layout()
    save(fig,"05_coverage_and_kernels",pdf,"Black HST; red Swift; blue ground. The kernel curves are each normalized to one and color-ordered from short to long wavelength. Broadband coverage does not imply 50 independently resolved radii; neighboring radial kernels overlap strongly.")


def compare(z,m,pdf):
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    colors=["#1F4E9E","#C43C39","#009E73","#7A5195"]
    for nt,c in zip([50,100,250,500],colors):
        rows=[m[f"local_nt{nt}_xi{xi}"] for xi in [1,10,100,1000]]
        for ax,field,pfield in zip(axs,["chi2_per_point","scale99"],["paper_chi2_per_point","paper_scale99"]):
            ax.plot([1,10,100,1000],[r[field] for r in rows],"o-",color=c,ms=4,label=rf"$N_t={nt}$")
            ax.plot([1,10,100,1000],[r[pfield] for r in rows],"s--",color=c,ms=4,mfc="white")
            ax.set_xscale("log");ax.set_xlabel(r"$\xi$");configure_axes(ax)
    axs[0].set_ylabel(r"$\chi^2/N_d$");axs[1].set_ylabel(r"99th percentile of $|\delta T/T|$")
    axs[1].set_yscale("log");axs[0].legend(ncol=2,fontsize=9)
    fig.suptitle("Solid/circles: local reconstruction     Dashed/squares: NK22")
    fig.tight_layout()
    save(fig,"06_paper_numeric_comparison",pdf,"Fixed physical parameters and regularization, not tuned toward publication. NK22 numbers read from Fig.5 annotations. Input Swift release, wavelengths, time selection and weighting are not identical; deviations do not by themselves diagnose a solver error.")
    rows=[v for v in m.values() if v["release"]=="local"]
    with (OUT/"paper_comparison.csv").open("w",newline="") as h:
        fields=["nt","xi","n_data","chi2_per_point","paper_chi2_per_point","scale99","paper_scale99"]
        w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)


def sensitivity(z,m,pdf):
    fig,axs=plt.subplots(1,3,figsize=(12,4),sharey=True)
    scale=m["local_nt250_xi10"]["scale99"]
    panels=[("q_local_nt250_xi10","All 19 bands"),("q_local_without_Uu","Exclude U and u")]
    for ax,(key,title) in zip(axs,panels):
        im=map_axis(ax,z,key,z["local_tp_250"],scale=scale);ax.set_title(title)
    difference=dict(z);difference["difference"]=z[panels[1][0]]-z[panels[0][0]]
    im=map_axis(axs[2],difference,"difference",z["local_tp_250"],scale=scale);axs[2].set_title("Difference (excluded - all)")
    axs[0].set_ylabel(r"$\log_{10}(R/R_{\rm in})$")
    fig.subplots_adjust(left=.06,right=.91,bottom=.19,top=.87,wspace=.04)
    cax=fig.add_axes([.93,.19,.016,.68]);fig.colorbar(im,cax=cax,label=r"$(\delta T/T)/$ baseline scale")
    fig.supxlabel("HJD - 2450000 [observer days]",y=.02)
    save(fig,"07_Uu_exclusion",pdf,"Identical 50x250 grid and xi=10; U/u measurements removed and remaining bands refit. All three panels use one baseline color scale, including the difference. Agreement tests sensitivity to two bands, not BLR absence in the remaining bands.")


def lag_figure(pdf):
    path=OUT/"lag_baseline_results.npz"
    if not path.exists():return
    d=np.load(OUT/"lag_plot_data.npz",allow_pickle=False);r=np.load(path,allow_pickle=False)
    s=json.loads((OUT/"lag_baseline_summary.json").read_text())
    fig=plt.figure(figsize=(12,5));gs=fig.add_gridspec(2,4,width_ratios=[1.6,1.6,1,1],hspace=.4,wspace=.65)
    grid=r["baseline_prediction_time"]
    for row,name in enumerate(["driver","response"]):
        ax=fig.add_subplot(gs[row,:2]);obs=d["driver"] if row==0 else d["residual"]
        ax.errorbar(d["t"],obs,d["error"],fmt=".",color="black",ms=2,elinewidth=.4)
        mean=r["baseline_"+name+"_mean"];std=r["baseline_"+name+"_std"]
        ax.plot(grid,mean,color="red",lw=1);ax.fill_between(grid,mean-std,mean+std,color="red",alpha=.15)
        ax.set_ylabel("U flux" if row==0 else "U residual")
        if row==1:ax.set_xlabel("HJD - 2450000 [observer days]")
        configure_axes(ax)
    samples=r["baseline_samples"]
    for i,(col,label) in enumerate([(0,r"$\log_{10}\sigma$"),(1,r"$\log_{10}\tau_{\rm DRW}$"),(2,r"$t_{\rm lag}$ [days]"),(4,r"$s_{\rm lag}$")]):
        ax=fig.add_subplot(gs[i//2,2+i%2]);ax.hist(samples[:,col],bins=45,density=True,histtype="step",color="black")
        med=np.median(samples[:,col]);ax.axvline(med,color="red",lw=.8)
        ax.set_xlabel(label);ax.set_yticks([]);configure_axes(ax)
        if col in [2,4]:ax.axvline(.11 if col==2 else .22,color="#1F4E9E",ls="--",lw=1)
    status="50-tau criterion passed" if s["converged_50tau"] else "50-tau criterion NOT met"
    fig.suptitle("NGC 5548 — conditional U-residual fit ("+status+")",fontsize=12)
    save(fig,"08_U_residual_lag",pdf,"2024 Fig.16 analogue. JAVELIN covariance, flat priors with explicit bounds, emcee sampling. Dashed blue lines mark published medians. This conditional likelihood ignores temperature-map uncertainty and shared U/residual noise; it cannot establish that BLR contamination is absent. DRW sigma uses the saved observed flux unit.")


def main():
    z=dict(np.load(OUT/"fits.npz",allow_pickle=False))
    m={r["name"]:r for r in json.loads((OUT/"metrics.json").read_text())}
    with PdfPages(OUT/"NGC5548_reproduction.pdf") as pdf:
        grid_figure(z,m,pdf);moderate(z,m,pdf);lightcurves(z,pdf);residuals(z,pdf)
        coverage(z,pdf);compare(z,m,pdf);sensitivity(z,m,pdf);lag_figure(pdf)
    (OUT/"figure_captions.md").write_text("# Figure definitions and caveats\n\n"+"\n\n".join("## "+n+"\n\n"+c for n,c in CAPTIONS)+"\n")
    print("Saved",len(CAPTIONS),"figure sets and PDF atlas",flush=True)


if __name__=="__main__":main()
