"""NGC5548 Fig16 analogue: conditional DRW/top-hat fit, not BLR identification."""
from pathlib import Path
from types import SimpleNamespace
import json, sys
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import residual_lags as lag


if __name__=="__main__":
    threadpool_limits(1)
    out=ROOT/"outputs"
    z=np.load(out/"fits.npz",allow_pickle=False)
    mask=z["local_bands"][z["local_bi"]]=="Swift U"
    t=z["local_t"][mask]
    f=z["local_flux"][mask]
    e=z["local_error"][mask]
    residual=f-z["pred_local_nt250_xi10"][mask]
    assert len(t)==267
    official=lag.load_official()
    lag.selftest(out)
    args=SimpleNamespace(seed=5548,walkers=24,burn=400,steps=2500,max_steps=6500,workers=2)
    (out/"lag_run_contract.json").write_text(json.dumps(dict(
        input="local Nt250 xi10 temperature-map residual; 267 Swift U points",
        official_javelin_covariance=official,parameters=vars(args),
        bounds_internal=lag.BOUNDS.tolist(),
        paper={"lag_days":[.11,.17,.36],"scale":[.22,.02,.02]},
        caveat="CONDITIONAL phenomenological fit: ignores the map-estimation covariance and shared measurement noise between U and its fitted residual. Not proof for/against BLR; priors and sampling differ from unpublished author configuration."),indent=2))
    data,summary=lag.run_case("baseline",t,f,residual,e,args,out)
    np.savez_compressed(out/"lag_plot_data.npz",t=t,driver=f,residual=residual,error=e)
