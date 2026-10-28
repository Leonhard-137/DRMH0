"""Paper-style figure atlas for the Mrk 817 reproduction."""
from pathlib import Path
import json, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from matplotlib.patches import Rectangle
from temperature_mapping import disk_temperature, DEFAULT, C, DAY
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"
FIG=OUT/"figures"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelsize":12,
    "xtick.direction":"in","ytick.direction":"in","xtick.top":True,"ytick.right":True,
    "axes.linewidth":.7,"xtick.labelsize":9,"ytick.labelsize":9,
    "savefig.dpi":180,"pdf.fonttype":42,"figure.facecolor":"white"})
CMAP=plt.get_cmap("coolwarm",20)
ORDER=["HST_1180","HST_1398","HST_1502","HST_1739","Swift_UVW2",
       "Swift_UVM2","Swift_UVW1","Swift_U","Ground_u","Swift_B","Ground_B",
       "Ground_g","Swift_V","Ground_V","Ground_r","Ground_i","Ground_z"]
PAPER={
"xi_1":(1.59,.54),"xi_10":(1.75,.20),"xi_100":(1.93,.10),"xi_1000":(2.23,.08),
"baseline":(1.75,.20),"smooth_7":(.83,.19),"smooth_28":(.68,.16),"smooth_56":(.57,.14),
"subtract_7":(.92,.07),"subtract_28":(1.09,.09),"subtract_56":(1.22,.12),
"blr_0p1":(1.89,.22),"blr_0p2":(2.08,.23),"blr_0p5":(2.93,.32)}
def pretty(b):
    a,v=b.split("_")
    if a=="HST": return "HST "+v+r" $\AA$"
    return a+" "+v
def style(ax):
    if ax.get_xscale()=="linear": ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ax.get_yscale()=="linear": ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="major",length=5)
    ax.tick_params(which="minor",length=2.5)
def save(fig,name,pdf,caption=""):
    if caption:
        fig.text(.5,.015,caption,ha="center",va="bottom",fontsize=8,color=".3")
    fig.savefig(FIG/(name+".png"),bbox_inches="tight",facecolor="white")
    pdf.savefig(fig,bbox_inches="tight",facecolor="white")
    plt.close(fig)
def map_ax(ax,q,z,title="",annotate=None):
    scale=np.quantile(np.abs(q),.99)
    im=ax.pcolormesh(z["tp"],np.log10(z["u"]),q/max(scale,1e-20),cmap=CMAP,
                     shading="nearest",vmin=-1,vmax=1,rasterized=True)
    ax.set_xlim(z["tp"][0],z["tp"][-1]);ax.set_ylim(0,3)
    ax.set_title(title);ax.set_xticks([9200,9300,9400,9500,9600])
    style(ax)
    if annotate:
        ax.text(.06,.08,r"$\chi^2/N_d = $"+f'{annotate["chi2_per_point"]:.2f}'+
                "\n"+f"scale = {scale:.2f}",transform=ax.transAxes,fontsize=10)
    return im
def maps(z,metric,keys,titles,name,pdf,caption):
    fig,axs=plt.subplots(1,len(keys),figsize=(3.1*len(keys),3.8),sharey=True)
    axs=np.atleast_1d(axs)
    for ax,key,title in zip(axs,keys,titles):
        im=map_ax(ax,z["q_"+key],z,title,metric[key])
    axs[0].set_ylabel(r"$\log R/R_{\rm in}$")
    fig.supxlabel("THJD",y=.09)
    fig.subplots_adjust(left=.06,right=.91,bottom=.23,top=.87,wspace=.04)
    cax=fig.add_axes([.925,.23,.016,.64])
    fig.colorbar(im,cax=cax,label=r"$(\delta T/T)\,/\,{\rm scale}$",ticks=[-1,-.5,0,.5,1])
    save(fig,name,pdf,caption)
def lc_atlas(z,key,name,pdf,overlay=False):
    fig=plt.figure(figsize=(13,11))
    gs=fig.add_gridspec(5,4,hspace=.14,wspace=.14,left=.075,right=.985,top=.94,bottom=.08)
    positions=[(0,0),(0,1),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(2,3),
               (3,0),(3,1),(3,2),(3,3),(4,0),(4,1),(4,2),(4,3)]
    for b,(r,c) in zip(ORDER,positions):
        sub=gs[r,c].subgridspec(2,1,height_ratios=[2,1],hspace=.03)
        ax=fig.add_subplot(sub[0]);rx=fig.add_subplot(sub[1],sharex=ax)
        ii=z["band"]==b;t=z["t"][ii]
        y=z["flux_"+key][ii]*10;p=z["pred_"+key][ii]*10;e=z["error"][ii]*10
        norm=1.;shift=0.
        if overlay:
            shift=y.min();norm=np.ptp(y)
            y=(y-shift)/norm;p=(p-shift)/norm;e=e/norm
        ax.errorbar(t,y,e,fmt=".",color="black",ms=2,elinewidth=.45)
        ax.plot(t,p,"r-",lw=.8)
        if overlay:
            sy=(z["flux_smooth_56"][ii]*10-shift)/norm
            sp=(z["pred_smooth_56"][ii]*10-shift)/norm
            ax.plot(t,sy,color="royalblue",lw=1)
            ax.plot(t,sp,color=".6",lw=1)
        ax.text(.97,.94,pretty(b),transform=ax.transAxes,ha="right",va="top",fontsize=10,
                bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=.8))
        ax.tick_params(labelbottom=False)
        rx.errorbar(t,y-p,e,fmt="none",color="black",elinewidth=.4)
        if overlay:rx.plot(t,sy-sp,color="royalblue",lw=.8)
        rx.axhline(0,color="red",lw=.6,ls="--")
        lim=max(np.quantile(np.abs(y-p)+e,.98)*1.2,.03 if overlay else 1)
        rx.set_ylim(-lim,lim)
        ax.yaxis.set_major_locator(MaxNLocator(3));rx.yaxis.set_major_locator(MaxNLocator(3))
        for a in [ax,rx]:
            a.set_xlim(z["tp"][0],z["tp"][-1]);a.set_xticks([9200,9300,9400,9500,9600])
            style(a);a.tick_params(labelsize=7.5)
        if r<4:rx.tick_params(labelbottom=False)
    fig.text(.58,.89,"Data",color="black",fontsize=13)
    fig.text(.58,.867,"Temperature-map model",color="red",fontsize=13)
    fig.text(.58,.837,"Lower panels: residuals",fontsize=11)
    if overlay:fig.text(.58,.802,"56-day low-pass: blue data / gray model",fontsize=10)
    fig.suptitle(name.replace("_"," ")+r"  ($\xi=10$)",y=.985,fontsize=14)
    fig.supxlabel("THJD",y=.038)
    ylabel="Flux / residual, normalized to raw-data range" if overlay else r"Flux [$10^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]"
    fig.supylabel(ylabel,x=.016)
    caption="Common affine normalization per band: (flux - raw minimum) / raw range; residuals / raw range." if overlay else "2026 public release; errors are the triplet-adjusted observational errors."
    save(fig,name,pdf,caption)
def summary_page(z,metric,pdf):
    fig=plt.figure(figsize=(11.7,8.3));ax=fig.add_axes([.08,.08,.84,.84]);ax.axis("off")
    lines=[
      "MRK 817  |  TEMPERATURE-FLUCTUATION MAPS",
      "Reproduction of Neustadt et al. (2024), ApJ 961, 219",
      "",
      f"Input: {len(z['t'])} measurements, 17 bands; 2026 final public ground release.",
      "HST/Swift observation times matched to the published HJD tables.",
      "Model: 50 radial cells x 250 time nodes; axisymmetric linear delta-T.",
      "M = 3.85e7 Msun; DL = 136 Mpc; z = 0.031455; L/LEdd = 0.2;",
      "Rin = 6 Rg; eta = 0.1; inclination = 30 degrees.",
      "",
      "CORE FITS                         This run                         Paper",
      "xi                            chi2/N       scale99              chi2/N      scale99"]
    for x in [1,10,100,1000]:
        m=metric[f"xi_{x}"];p=PAPER[f"xi_{x}"]
        lines.append(f"{x:<8}                       {m['chi2_per_point']:6.3f}       {m['scale99']:6.3f}                 {p[0]:4.2f}        {p[1]:4.2f}")
    lines+=["","INTERPRETATION OF THE ATLAS",
      "Maps are regularized inversions, not direct images of measured temperatures.",
      "Color scales are normalized separately; the printed scale gives |delta T/T|_99.",
      "Same campaign does not imply the identical 2024 input table (7380 points).",
      "The final ground wavelengths are those published in 2026 (z band: 8700 A).",
      "Unspecified boundary / smoothing conventions are documented in config.json.",
      "",
      "COVERAGE",
      "Core maps, fitted light curves, smoothing/subtraction, U/u BLR injections,",
      "residual-lag checks, X-ray comparison, and information-criterion scan.",
      "Fig. 13 full Netzer template was not obtained; only the specified U/u test is run.",
      "Fig. 16 is a separate NGC 5548 experiment and is outside this Mrk 817 dataset.",
      "Fig. 14/15 use reconstructed priors; convergence status is printed on each figure."]
    for j,line in enumerate(lines):
        ax.text(0,1-j*.033,line,va="top",fontsize=14 if j==0 else (11 if j==1 else 10),
                family="DejaVu Sans Mono" if j>=9 and j<=15 else "DejaVu Sans")
    save(fig,"00_summary",pdf)
def run():
    FIG.mkdir(parents=True,exist_ok=True)
    z=np.load(OUT/"fits.npz")
    metric={x["name"]:x for x in json.loads((OUT/"metrics.json").read_text())}
    with PdfPages(OUT/"Mrk817_reproduction.pdf") as pdf:
        summary_page(z,metric,pdf)
        fig,ax=plt.subplots(figsize=(11.5,3.7))
        colors=plt.cm.turbo(np.linspace(0,.95,len(z["bands"])))
        for b,k,color in zip(z["bands"],z["kernel"]/z["temp"][None,:],colors):
            ax.plot(np.log10(z["u"]),k/k.max(),label=pretty(b),lw=1.2,color=color)
        ax.set(xlim=(0,3),ylim=(0,1.05),xlabel=r"$\log R/R_{\rm in}$",ylabel="Relative contribution")
        ax.legend(ncol=6,fontsize=8,loc="lower center",bbox_to_anchor=(.5,1.01),frameon=False)
        style(ax);fig.subplots_adjust(top=.72,bottom=.18)
        save(fig,"fig01_radial_kernels",pdf)
        # Fig. 2: illustrative scenarios, with explicit non-inferred parameters.
        tt=z["tp"][None,:];lu=np.log10(z["u"])[:,None]
        rin=disk_temperature(z["u"],DEFAULT)[1]
        slow=.1*np.sin(2*np.pi*((tt-9200)/180+lu/.7))
        fast=.1*np.sin(2*np.pi*((tt-9200-z["u"][:,None]*rin/C/DAY*(1+DEFAULT["redshift"]))/180))
        ii=z["band"]=="Ground_g";gy=z["flux"][ii];gt=z["t"][ii]
        driver=(gy-gy.mean())/max(abs(gy-gy.mean()))
        realistic=.1*np.array([np.interp(z["tp"]-r*rin/C/DAY*(1+DEFAULT["redshift"]),gt,driver) for r in z["u"]])
        fig,axs=plt.subplots(1,4,figsize=(12.5,3.5),sharey=True)
        for ax,q,title in zip(axs,[slow,fast,(slow+.5*fast)/1.5,realistic],
             ["Slow ingoing","Outgoing at c","Ingoing + outgoing","Ground g, outgoing at c"]):
            im=ax.pcolormesh(z["tp"],lu.ravel(),q/.1,cmap=CMAP,vmin=-1,vmax=1,shading="nearest",rasterized=True)
            ax.set_title(title);style(ax);ax.set_ylim(0,3);ax.set_xticks([9200,9400,9600])
        axs[0].set_ylabel(r"$\log R/R_{\rm in}$");fig.supxlabel("THJD",y=.10)
        fig.subplots_adjust(left=.06,right=.91,bottom=.24,top=.84,wspace=.04)
        fig.colorbar(im,cax=fig.add_axes([.93,.24,.015,.60]),label=r"$(\delta T/T)/0.1$")
        save(fig,"fig02_illustrative_scenarios",pdf,"Illustrations, not fits: period 180 d; ingoing radial wavelength 0.7 dex.")
        maps(z,metric,["xi_1","xi_10","xi_100","xi_1000"],
             [r"$\xi=1$",r"$\xi=10$",r"$\xi=100$",r"$\xi=1000$"],
             "fig03_regularization",pdf,"Independent NK22-equation implementation; 2026 public dataset.")
        lc_atlas(z,"baseline","fig04_lightcurves",pdf)
        fig,ax=plt.subplots(figsize=(9,5))
        im=map_ax(ax,z["q_baseline"],z,r"Mrk 817  |  $\xi=10$",metric["baseline"])
        ax.set(xlabel="THJD",ylabel=r"$\log R/R_{\rm in}$")
        ax.add_patch(Rectangle((9225,0),50,1.5,fill=False,ls=":",lw=1.5,color="black"))
        end=max(z["t"][np.char.startswith(z["band"],"HST")|np.char.startswith(z["band"],"Swift")])
        ax.add_patch(Rectangle((end,0),max(z["tp"])-end,1.5,facecolor="white",alpha=.32,hatch="//",edgecolor=".5"))
        fig.colorbar(im,ax=ax,label=r"$(\delta T/T)/{\rm scale}$")
        fig.subplots_adjust(bottom=.17)
        save(fig,"fig05_fiducial_map",pdf,"Dotted box: original-paper anomaly interval. Hatched corner: no space-based data.")
        maps(z,metric,["baseline","smooth_7","smooth_28","smooth_56"],
             ["Unsmoothed","Smoothed by 7 d","Smoothed by 28 d","Smoothed by 56 d"],
             "fig06_lowpass_maps",pdf,"Gaussian FWHM; original error bars retained.")
        lc_atlas(z,"smooth_7","fig07_lowpass_lightcurves",pdf)
        maps(z,metric,["baseline","subtract_7","subtract_28","subtract_56"],
             ["Unsmoothed","Subtract 7 d smooth","Subtract 28 d smooth","Subtract 56 d smooth"],
             "fig08_highpass_maps",pdf,"Map amplitude declines when slow variations are subtracted.")
        lc_atlas(z,"subtract_56","fig09_highpass_lightcurves",pdf)
        lc_atlas(z,"baseline","fig10_slow_fast_comparison",pdf,overlay=True)
        maps(z,metric,["baseline","blr_0p1","blr_0p2","blr_0p5"],
             ["No added BLR",r"$f_{\rm BLR}=0.1$",r"$f_{\rm BLR}=0.2$",r"$f_{\rm BLR}=0.5$"],
             "fig11_blr_injection_maps",pdf,"Only Swift U and Ground u receive the 21-day delayed, 10.5-day smoothed signal.")
        fig=plt.figure(figsize=(13,9))
        gs=fig.add_gridspec(4,4,left=.075,right=.985,top=.92,bottom=.09,hspace=.15,wspace=.12)
        for r,b in enumerate(["Swift_UVW1","Swift_U","Ground_u","Swift_B"]):
            ii=z["band"]==b;t=z["t"][ii];err=z["error"][ii]*10
            for c,key in enumerate(["baseline","blr_0p1","blr_0p2","blr_0p5"]):
                sub=gs[r,c].subgridspec(2,1,height_ratios=[2,1],hspace=.02)
                ax=fig.add_subplot(sub[0]);rx=fig.add_subplot(sub[1],sharex=ax)
                ax.plot(t,z["flux"][ii]*10,"k.",ms=2)
                if c:ax.plot(t,z["flux_"+key][ii]*10,color="turquoise",lw=.9)
                ax.plot(t,z["pred_"+key][ii]*10,"r-",lw=.9)
                res=(z["flux_"+key][ii]-z["pred_"+key][ii])*10
                rx.errorbar(t,res,err,fmt="none",ecolor="black",elinewidth=.45)
                rx.axhline(0,ls="--",color="red",lw=.6)
                if r==0:ax.set_title(["No BLR","f = 0.1","f = 0.2","f = 0.5"][c])
                ax.text(.95,.9,pretty(b),ha="right",va="top",transform=ax.transAxes,fontsize=9,
                        bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=.8))
                ax.tick_params(labelbottom=False)
                if r<3:rx.tick_params(labelbottom=False)
                for a in [ax,rx]:
                    a.set_xlim(z["tp"][0],z["tp"][-1]);a.set_xticks([9200,9400,9600])
                    a.yaxis.set_major_locator(MaxNLocator(3));style(a);a.tick_params(labelsize=7)
        fig.supylabel(r"Flux [$10^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]",x=.014)
        fig.supxlabel("THJD",y=.045)
        save(fig,"fig12_blr_lightcurves",pdf,"Black: original. Turquoise: injected. Red: fit. Lower panels: injected data minus fit.")
        maps(z,metric,["baseline","blr_0p5","xi_1"],
             ["No BLR, xi = 10","U/u BLR, f = 0.5","No BLR, xi = 1"],
             "fig13_available_blr_comparison",pdf,"Netzer full-spectrum injection is omitted: its exact per-band template was not obtained.")
        # Lag-module figures can be incorporated after that independent run finishes.
        lagpath=OUT/"lag_results.npz"
        if lagpath.exists():
            from residual_lags import make_figures
            make_figures(OUT,pdf,FIG)
        xrtpath=ROOT/"derived/xrt.csv"
        if xrtpath.exists():
            x=np.genfromtxt(xrtpath,delimiter=",",names=True)
            fig,axs=plt.subplots(3,1,figsize=(10,8),sharex=True,
                                 gridspec_kw={"height_ratios":[1,2,1],"hspace":.08})
            axs[0].errorbar(x["thjd"],x["rate"],x["error"],fmt="k.",ms=2,elinewidth=.5)
            axs[0].set_yscale("log");axs[0].set_ylabel("XRT [count/s]")
            im=map_ax(axs[1],z["q_baseline"],z)
            axs[1].invert_yaxis();axs[1].set_ylabel(r"$\log R/R_{\rm in}$")
            ii=z["band"]=="Swift_UVW2"
            axs[2].errorbar(z["t"][ii],z["flux"][ii]/10,z["error"][ii]/10,fmt="k.",ms=2,elinewidth=.5)
            axs[2].plot(z["t"][ii],z["pred_baseline"][ii]/10,"r-",lw=1)
            axs[2].set_ylabel("UVW2 flux\n"+r"[$10^{-14}$ cgs / $\AA$]");axs[2].set_xlabel("THJD")
            for ax in axs:style(ax)
            fig.subplots_adjust(right=.87,bottom=.09)
            fig.colorbar(im,cax=fig.add_axes([.9,.34,.018,.36]),label=r"$(\delta T/T)/0.210$")
            save(fig,"fig17_xray_map_uv",pdf,"XRT = matched soft + hard bands; map axis reversed as in the original.")
        scan=[m for k,m in metric.items() if k.startswith("xi_")]
        xs=np.array([m["xi"] for m in scan]);aic=np.array([m["aic"] for m in scan]);bic=np.array([m["bic"] for m in scan])
        fig,ax=plt.subplots(figsize=(6,4))
        ax.plot(np.log10(xs),bic/1e4,"k.-",label="BIC")
        ax.plot(np.log10(xs),aic/1e4,"k.--",label="AIC")
        se=np.array([m["effective_parameters_mcse"] for m in scan])
        for yy,ee in [(aic,2*se),(bic,np.log(len(z["t"]))*se)]:
            ax.fill_between(np.log10(xs),(yy-ee)/1e4,(yy+ee)/1e4,color=".7",alpha=.4)
        for yy,ls in [(aic,"--"),(bic,":")]:
            i=yy.argmin();ax.axvline(np.log10(xs[i]),color=".5",ls=ls,lw=.8)
            ax.annotate(r"$\xi=$"+f"{xs[i]:g}",(np.log10(xs[i]),yy[i]/1e4),xytext=(8,10),textcoords="offset points")
        ax.set(xlabel=r"$\log \xi$",ylabel=r"Criterion / $10^4$");ax.legend(frameon=False);style(ax)
        fig.subplots_adjust(bottom=.2)
        save(fig,"fig18_information_criteria",pdf,"Effective parameters: Hutchinson trace estimate (64 probes); MC errors saved in metrics.json.")
        spath=OUT/"sensitivity.npz"
        if spath.exists():
            sz=np.load(spath)
            ss=json.loads((OUT/"sensitivity.json").read_text())
            keys=["baseline","original_errors","ground_z_8897","nt_500"]
            titles=["Baseline","Original errors",r"Ground z: 8897 $\AA$",r"$N_t=500$, fixed $\xi$"]
            fig,axs=plt.subplots(1,4,figsize=(12.4,3.8),sharey=True)
            common_scale=metric["baseline"]["scale99"]
            for ax,key,title in zip(axs,keys,titles):
                q=sz["q_baseline"] if key=="baseline" else sz["q_common_"+key]
                im=ax.pcolormesh(z["tp"],np.log10(z["u"]),q/common_scale,
                    cmap=CMAP,shading="nearest",vmin=-1,vmax=1,rasterized=True)
                ax.set(title=title,ylim=(0,3),xticks=[9200,9300,9400,9500,9600])
                style(ax)
                scale=ss["baseline"]["scale99"] if key=="baseline" else ss["variants"][key]["scale99"]
                ax.text(.05,.08,f"native-grid scale99 = {scale:.3f}",transform=ax.transAxes,fontsize=8)
            axs[0].set_ylabel(r"$\log R/R_{\rm in}$");fig.supxlabel("THJD",y=.09)
            fig.subplots_adjust(left=.06,right=.91,bottom=.23,top=.87,wspace=.04)
            fig.colorbar(im,cax=fig.add_axes([.925,.23,.016,.64]),label=r"$(\delta T/T)/0.210$")
            save(fig,"appendix_sensitivity",pdf,"Common color scale; Nt change is a fixed-penalty sensitivity test, not a grid-convergence proof.")
    print("Wrote",OUT/"Mrk817_reproduction.pdf",flush=True)
if __name__=="__main__":run()
