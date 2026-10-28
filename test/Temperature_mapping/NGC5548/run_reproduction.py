"""NGC 5548 / NK22 reproduction. Writes only inside this object directory.

Reuse the verified Mrk817 linear inverse operator, not its object-specific
preprocessing. Public TempMap and local STORM releases are kept separate.
"""
from pathlib import Path
import argparse, csv, gc, json, sys, time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from temperature_mapping import Inversion, DEFAULT, temporal_weights

OUT = ROOT / "outputs"
PARAMS = dict(DEFAULT, mass_msun=10**7.505, distance_mpc=75., redshift=.017,
              eddington_ratio=.1, efficiency=.1, inclination_deg=30.,
              nr=50, nt=250, rin_rg=6., umax=1000.)
LOCAL_FILTERS = ["W2", "M2", "W1", "U", "B", "V", "HST1157.5", "HST1367",
                 "HST1478.5", "HST1746", "u", "BG", "g", "VG", "r", "R", "i", "I", "z"]
PAPER = {
    50: [(2.40,.88),(2.54,.28),(2.67,.09),(2.78,.06)],
    100: [(1.56,.64),(1.72,.22),(1.87,.08),(2.01,.06)],
    250: [(1.08,.45),(1.29,.14),(1.47,.07),(1.67,.06)],
    500: [(.82,.33),(1.08,.12),(1.30,.07),(1.63,.06)]}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")


def extra_error(t, f, e):
    """NK22 mean of nonnegative per-triplet extra standard deviations."""
    floors = []
    for j in range(len(t)-2):
        x,y,s = t[j:j+3],f[j:j+3],e[j:j+3]
        span = x[2]-x[0]
        if span <= 0:
            continue
        a = np.array([x[1]-x[2],x[2]-x[0],x[0]-x[1]])/span
        floors.append(np.sqrt(max(0., ((a@y)**2-(a*a)@(s*s))/(a@a))))
    return float(np.mean(floors))


def prepare():
    z = np.load(ROOT/"reference/NGC5548_TempMap.npz", allow_pickle=False)
    labels = z["filters"].astype(str)
    wavelengths = z["wavelength"]
    order = np.argsort(wavelengths)
    bands = labels[order]
    waves = wavelengths[order]
    # Source arrays use CGS f_lambda per centimetre, so x1e7 gives
    # units of 1e-15 erg/s/cm^2/Angstrom. No second triplet correction.
    arrays = []
    inventory = []
    for k,j in enumerate(order):
        valid = np.isfinite(z["dates"][j])
        t = z["dates"][j,valid]-50000.
        f = z["flux"][j,valid]*1e7
        e = z["error"][j,valid]*1e7
        assert np.all(np.isfinite(f)) and np.all(np.isfinite(e)) and np.all(e>0)
        ix = np.argsort(t)
        arrays.append(np.column_stack([t[ix],f[ix],e[ix],np.full(len(t),k)]))
        inventory.append(dict(band=str(bands[k]), wavelength_obs_A=float(waves[k]),
                              n=len(t), thjd_min=float(t.min()), thjd_max=float(t.max()),
                              median_cadence_days=float(np.median(np.diff(t[ix])))))
    a = np.vstack(arrays)
    public = dict(t=a[:,0], flux=a[:,1], error=a[:,2], bi=a[:,3].astype(int),
                  bands=bands, wavelengths=waves)
    np.savez_compressed(ROOT/"derived/public_input.npz", **public)

    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    from astropy.utils import iers
    import astropy.units as units
    iers.conf.auto_download = False
    raw = ROOT.parents[2]/"NGC5548/raw/NGC5548.csv"
    with raw.open() as h:
        records = [r for r in csv.DictReader(h) if r["Filter"] in LOCAL_FILTERS]
    # Local metadata say geocentric MJD_UTC. The inverse heliocentric
    # conversion restores published HST/ground time precision to <0.001s.
    mjd = np.array([float(r["MJD"]) for r in records])
    dates = Time(mjd, format="mjd", scale="utc",
                 location=EarthLocation.from_geocentric(0,0,0,unit=units.m))
    coord = SkyCoord("14h17m59.51s", "+25d08m12.5s")
    helio = dates.light_travel_time(coord, kind="heliocentric", ephemeris="builtin").to_value(units.day)
    times = mjd-49999.5+helio
    arrays = []
    local_inventory = []
    for k,j in enumerate(order):
        ii = np.array([r["Filter"]==LOCAL_FILTERS[j] for r in records])
        rr = [r for r,keep in zip(records,ii) if keep]
        t = times[ii]
        f = np.array([float(r["Flux"]) for r in rr])
        e = np.array([float(r["Error"]) for r in rr])
        ix = np.argsort(t);t,f,e = t[ix],f[ix],e[ix]
        floor = extra_error(t,f,e)
        arrays.append(np.column_stack([t,f,np.hypot(e,floor),np.full(len(t),k),e]))
        local_inventory.append(dict(band=str(bands[k]), local_filter=LOCAL_FILTERS[j],
            wavelength_obs_A=float(waves[k]),n=len(t),thjd_min=float(t.min()),thjd_max=float(t.max()),
            triplet_floor=floor, median_error=float(np.median(e)),
            median_error_corrected=float(np.median(np.hypot(e,floor)))))
    a = np.vstack(arrays)
    local = dict(t=a[:,0],flux=a[:,1],error=a[:,2],bi=a[:,3].astype(int),
                 error_original=a[:,4],bands=bands,wavelengths=waves)
    np.savez_compressed(ROOT/"derived/local_input.npz",**local)
    dump(ROOT/"derived/data_audit.json", dict(public=inventory,local=local_inventory,
        public_source="https://github.com/Zstone19/TempMap/blob/main/Examples/NGC5548_Data.npz",
        public_n=int(len(public["t"])),local_n=int(len(local["t"])),
        public_units="CGS per cm converted by x1e7 to 1e-15 CGS per Angstrom",
        public_times="SUPPLIED dates minus 50000: mixed HJD (HST/ground) and MJD (matched Swift2019); NOT a coherent HJD timeline",
        public_status="DIAGNOSTIC ONLY: mixed Swift releases and verified half-day relative time-system inconsistency; local input is primary",
        public_error="already triplet-corrected according to source notebook; not inflated again",
        local_times="geocentric MJD_UTC converted to heliocentric UTC with built-in Astropy ephemeris",
        local_flux="raw observed f_lambda; no extinction/host subtraction",
        local_error="NK22 mean of nonnegative adjacent-triplet additional standard deviations",
        wavelength_caveat="Public vector used consistently in both runs; NK22 text quotes U=3494, u=3590 and longest=8897 A, different from this later public bundle.",
        data_identity_caveat="Public notebook calls this NK22 data; identity with unpublished author selections/preprocessing has not been independently verified."))
    for name,data in [("public",public),("local",local)]:
        with (ROOT/f"derived/{name}_lightcurves.csv").open("w",newline="") as h:
            writer=csv.writer(h)
            writer.writerow(["band","wavelength_obs_A","thjd","flux","error"])
            for t,f,e,b in zip(data["t"],data["flux"],data["error"],data["bi"]):
                writer.writerow([data["bands"][b],data["wavelengths"][b],t,f,e])
    print("Prepared",len(public["t"]),"public and",len(local["t"]),"local points",flush=True)


def run():
    threadpool_limits(1)
    start = time.perf_counter()
    metrics = []
    archive = {}
    for release in ["public","local"]:
        data = dict(np.load(ROOT/f"derived/{release}_input.npz",allow_pickle=False))
        for k,v in data.items(): archive[release+"_"+k]=v
        for nt in [50,100,250,500]:
            p = dict(PARAMS,nt=nt)
            model = Inversion(data["t"],data["bi"],data["wavelengths"],data["error"],p)
            xis = [1.,10.,30.,50.,100.,1000.] if nt==250 else [1.,10.,100.,1000.]
            archive[f"{release}_tp_{nt}"]=model.tp
            archive["u"],archive["edges"],archive["temp"] = model.u,model.edges,model.temp
            archive[release+"_kernel"]=model.kernel
            for xi in xis:
                key=f"{release}_nt{nt}_xi{xi:g}"
                factor_seconds=model.factorize(xi)
                fit=model.solve(data["flux"])
                assert fit["normal_equation_relative_residual"]<1e-8
                assert np.isfinite(fit["q"]).all()
                edf,edfse=model.edf(32,seed=5548)
                row=dict(name=key,release=release,nt=nt,xi=xi,n_data=len(data["t"]),
                    **{k:fit[k] for k in ["chi2","chi2_per_point","scale99","penalty","normal_equation_relative_residual"]},
                    effective_parameters=edf,effective_parameters_mcse=edfse,factor_seconds=factor_seconds)
                if xi in [1,10,100,1000]:
                    row["paper_chi2_per_point"],row["paper_scale99"]=PAPER[nt][[1,10,100,1000].index(xi)]
                elif nt==250:
                    row["paper_chi2_per_point"],row["paper_scale99"]={30:(1.38,.09),50:(1.42,.08)}[int(xi)]
                archive["q_"+key],archive["pred_"+key]=fit["q"],fit["pred"]
                metrics.append(row)
                if xi==10 and nt==250:
                    archive[release+"_baseline"]=fit["baseline"]
                    # Offset invariance: arbitrary per-band constants must not
                    # masquerade as disk-temperature variability.
                    shifted=model.solve(data["flux"]+np.arange(len(data["bands"]))[data["bi"]])
                    row["offset_invariance_max_abs_q"]=float(np.max(abs(shifted["q"]-fit["q"])))
                    assert row["offset_invariance_max_abs_q"]<1e-8
                    if release=="local":
                        # Alternative convention used in the public tutorial:
                        # rest flux x(1+z)^2 instead of the physical x(1+z).
                        model.factorize(10.*(1+p["redshift"])**2)
                        red=model.solve(data["flux"]*(1+p["redshift"]))
                        archive["q_flux_scaled_1pz"]=red["q"]
                        # Exclusion is a sensitivity test, not a BLR subtraction.
                        keep=~np.isin(data["bands"][data["bi"]],["Swift U","SDSS u"])
                        used=np.unique(data["bi"][keep]);lookup={int(b):i for i,b in enumerate(used)}
                        sub=Inversion(data["t"][keep],np.array([lookup[int(b)] for b in data["bi"][keep]]),
                            data["wavelengths"][used],data["error"][keep],p)
                        sub.factorize(10.)
                        sf=sub.solve(data["flux"][keep])
                        archive["q_local_without_Uu"]=sf["q"]
                        archive["pred_local_without_Uu"]=sf["pred"]
                        row["without_Uu_map_pearson_all"]=float(np.corrcoef(fit["q"].ravel(),sf["q"].ravel())[0,1])
                        row["without_Uu_chi2_per_point"]=sf["chi2_per_point"]
                        del sub
                print(key,"chi2/N",round(fit["chi2_per_point"],4),"scale99",round(fit["scale99"],4),"factor_s",round(factor_seconds,2),flush=True)
                # Durable intermediate result, including before the 500-time grid.
                dump(OUT/"metrics.json",metrics)
            del model
            gc.collect()
    np.savez_compressed(OUT/"fits.npz",**archive)
    # Verify inclined-ring temporal probabilities sum to unity.
    tt=np.linspace(0,100,80)
    test=max(np.max(abs(temporal_weights(tt,np.linspace(0,100,100),w).sum(1)-1)) for w in [0.,.1,3.,30.])
    assert test<1e-12
    dump(OUT/"config.json",dict(PARAMS,elapsed_seconds=time.perf_counter()-start,
        edf="32 Rademacher probes, seed5548; Monte Carlo SE recorded; NOT integer degrees of freedom",
        time_weight_sum_max_error=float(test),
        redshift_flux_convention="f_lambda,obs=f_lambda,rest/(1+z), with luminosity distance",
        grid="50 logarithmic radial cells with cellwise deltaT; uniform time hats; same operator as Mrk817",
        fit_statistic="chi-square divided by N_data, not reduced chi-square",
        paper_reference="Neustadt & Kochanek 2022 Figures5-7; independent method-level reproduction"))
    print("ALL FITS COMPLETE",round(time.perf_counter()-start,2),"seconds",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    prepare()
    if not args.prepare_only: run()
