"""Export numerical per-band diagnostics and measured runtime metadata."""
from pathlib import Path
import json, platform
import numpy as np
import scipy, matplotlib
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"

def main():
    z=np.load(OUT/"fits.npz")
    metrics=json.loads((OUT/"metrics.json").read_text())
    summary={"n_data":len(z["t"]),"n_bands":len(z["bands"]),
             "units":"flux and residual: 1e-15 erg s^-1 cm^-2 Angstrom^-1; q is dimensionless",
             "chi2_note":"chi2/N is per datum, not chi2 divided by residual degrees of freedom",
             "bands":[],"versions":{"python":platform.python_version(),"numpy":np.__version__,
                         "scipy":scipy.__version__,"matplotlib":matplotlib.__version__},
             "baseline_q_rms":float(np.sqrt(np.mean(z["q_baseline"]**2)))}
    for band in z["bands"]:
        ii=z["band"]==band
        y=z["flux"][ii];p=z["pred_baseline"][ii];e=z["error"][ii]
        residual=y-p
        row={"band":str(band),"n":int(ii.sum()),"tmin":float(z["t"][ii].min()),
             "tmax":float(z["t"][ii].max()),"chi2_per_point":float(np.mean((residual/e)**2)),
             "flux_std":float(np.std(y,ddof=1)),"residual_std":float(np.std(residual,ddof=1)),
             "residual_mean":float(np.mean(residual)),"error_median":float(np.median(e))}
        row["residual_to_flux_std"]=row["residual_std"]/row["flux_std"]
        if band in ["Swift_U","Ground_u"]:
            row["injection_residual_std"]={name:float(np.std((z["flux_"+name]-z["pred_"+name])[ii],ddof=1))
                for name in ["blr_0p1","blr_0p2","blr_0p5"]}
        summary["bands"].append(row)
    scan=[m for m in metrics if m["name"].startswith("xi_")]
    summary["sampled_xi_minimum_aic"]=min(scan,key=lambda m:m["aic"])["xi"]
    summary["sampled_xi_minimum_bic"]=min(scan,key=lambda m:m["bic"])["xi"]
    ii=z["band"]=="Swift_U"
    t=z["t"][ii];y=z["flux"][ii]
    r0=y-z["pred_baseline"][ii]
    r5=z["flux_blr_0p5"][ii]-z["pred_blr_0p5"][ii]
    injected=.5*z["blr_template"][ii]
    delta=r5-r0
    gain,intercept=np.linalg.lstsq(np.column_stack([injected,np.ones(len(t))]),delta,rcond=None)[0]
    scanlags=np.round(np.arange(-40,80.01,.1),8)
    book={"injection_definition_max_error":float(np.max(np.abs(z["flux_blr_0p5"][ii]-y-injected))),
          "delta_residual_injection_correlation":float(np.corrcoef(delta,injected)[0,1]),
          "delta_residual_projection_gain":float(gain),"projection_intercept":float(intercept),
          "original_residual_injection_correlation":float(np.corrcoef(r0,injected)[0,1]),
          "descriptive_lags":{},
          "lag_caution":"One-way interpolation correlation peak; finite overlap; no uncertainty; not JAVELIN or physical identification"}
    for name,response in [("baseline_residual",r0),("injected_residual",r5),("residual_increment",delta),("injection",injected)]:
        corr=[]
        for lag in scanlags:
            inside=(t-lag>=t.min())&(t-lag<=t.max())
            corr.append(np.corrcoef(np.interp(t[inside]-lag,t,y),response[inside])[0,1])
        idx=np.argmax(corr)
        book["descriptive_lags"][name]={"lag_days":float(scanlags[idx]),"correlation":float(corr[idx]),
                                       "std_flux":float(np.std(response,ddof=1))}
    summary["injection_bookkeeping"]=book
    summary["runtime_temperature_suite_seconds"]=json.loads((OUT/"config.json").read_text())["elapsed_seconds"]
    (OUT/"band_diagnostics.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
