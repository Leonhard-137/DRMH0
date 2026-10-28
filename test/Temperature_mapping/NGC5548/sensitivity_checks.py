"""Small prereported convention checks. No optimization toward paper numbers."""
import json, sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from run_reproduction import Inversion,PARAMS


if __name__=="__main__":
    threadpool_limits(1)
    d=np.load(ROOT/"derived/local_input.npz",allow_pickle=False)
    z=np.load(ROOT/"outputs/fits.npz",allow_pickle=False)
    base=z["q_local_nt250_xi10"]
    tests=[];arrays={}
    wave=d["wavelengths"].copy()
    for band,value in [("Swift U",3494),("SDSS u",3590),("SDSS z",8897)]:
        wave[d["bands"]==band]=value
    cases=[("NK22_three_quoted_wavelengths",wave,d["error"],1.),
           ("extra_redshift_factor",d["wavelengths"],d["error"]*1.017,1.017),
           ("unadjusted_errors",d["wavelengths"],d["error_original"],1.)]
    for name,w,e,fs in cases:
        m=Inversion(d["t"],d["bi"],w,e,PARAMS)
        m.factorize(10.)
        fit=m.solve(d["flux"]*fs)
        mask=(np.log10(z["u"])[:,None]>.8)&(np.log10(z["u"])[:,None]<2.8)&(m.tp[None,:]>6706)&(m.tp[None,:]<6830)
        tests.append(dict(name=name,chi2_per_point=fit["chi2_per_point"],scale99=fit["scale99"],
                          pearson_full_map=float(np.corrcoef(base.ravel(),fit["q"].ravel())[0,1]),
                          pearson_central_window=float(np.corrcoef(base[mask],fit["q"][mask])[0,1]),
                          normal_equation_relative_residual=fit["normal_equation_relative_residual"]))
        arrays[name]=fit["q"]
    np.savez_compressed(ROOT/"outputs/sensitivity.npz",**arrays)
    (ROOT/"outputs/sensitivity.json").write_text(json.dumps(tests,indent=2))
    # Replace only our optional convention-check array; all primary results stay unchanged.
    saved=dict(z)
    saved["q_flux_scaled_1pz"]=arrays["extra_redshift_factor"]
    np.savez_compressed(ROOT/"outputs/fits.npz",**saved)
    print(json.dumps(tests,indent=2),flush=True)
