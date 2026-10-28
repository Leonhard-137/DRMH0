"""Final consistency checks of the actual ten-source numerical deliverable."""
from pathlib import Path
import csv
import json

import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"


def main():
    manifest=json.loads((ROOT/"derived/input_manifest.json").read_text())
    status=json.loads((OUT/"batch_status.json").read_text())
    summaries=json.loads((OUT/"summary.json").read_text())
    with (ROOT.parents[2]/"DRMH0/tables/sample_observations.csv").open() as f:
        expected=[r["source"] for r in csv.DictReader(f)]
    assert len(expected)==10 and status["completed"]==expected and not status["failures"]
    assert [r["source"] for r in summaries]==expected
    assert [r["source"] for r in manifest["sources"]]==expected
    physical=json.loads((ROOT/"physical_parameters.json").read_text())
    records=[]
    for meta,s in zip(manifest["sources"],summaries):
        source=s["source"]
        z=np.load(OUT/source/"fits.npz",allow_pickle=False)
        d=np.load(meta["input_path"],allow_pickle=False)
        assert np.array_equal(z["t"],d["t"]-d["t"].min())
        for key in ["flux","error","bi","bands","wavelengths"]:
            assert np.array_equal(z[key],d[key]),(source,key)
        assert len(z["t"])==s["n_data"]==meta["n_data"]
        for key,value in physical["sources"][source]["params"].items():
            assert s["params"][key]==value,(source,key)
        assert np.isclose(z["edges"][0],s["params"]["umin"])
        assert np.isclose(z["edges"][-1],s["params"]["umax"])
        metrics=json.loads((OUT/source/"metrics.json").read_text())
        maximum=max(m["normal_equation_relative_residual"] for m in metrics)
        assert maximum<1e-8
        names={m["name"] for m in metrics}
        assert {"xi1","xi10","xi100","xi1000","nt100","nt500","without_U"}<=names
        linerr=[]
        for width in (7,28,56):
            assert {f"low{width}",f"high{width}"}<=names
            assert np.allclose(z[f"flux_low{width}"]+z[f"flux_high{width}"],z["flux"],rtol=0,atol=1e-10)
            err=float(np.max(abs(z[f"q_low{width}"]+z[f"q_high{width}"]-z["q_xi10"])))
            linerr.append(err)
            assert err<1e-7
        chi=float(np.mean(((z["flux"]-z["pred_xi10"])/z["error"])**2))
        q99=float(np.quantile(abs(z["q_xi10"]),.99))
        assert np.isclose(chi,s["chi2_per_point"],rtol=1e-12)
        assert np.isclose(q99,s["q99"],rtol=1e-12)
        if source=="NGC5548":
            old=np.load(ROOT.parent/"NGC5548/outputs/fits.npz",allow_pickle=False)
            regression=float(np.max(abs(old["q_local_nt250_xi10"]-z["q_xi10"])))
            assert regression<1e-7
        else:
            regression=None
        records.append(dict(source=source,passed=True,n_fits=len(metrics),
            normal_equation_relative_residual_max=maximum,
            low_high_q_error_max=max(linerr),previous_5548_q_regression_error=regression))
    result=dict(passed=True,sources=records,total_sources=len(records),
        total_fits=sum(r["n_fits"] for r in records),
        total_input_points=sum(s["n_data"] for s in summaries),
        scope="Numerical/input consistency only; not a scientific significance or physical-identification test")
    (OUT/"deliverable_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
