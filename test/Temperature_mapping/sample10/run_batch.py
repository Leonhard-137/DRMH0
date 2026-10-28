"""Frozen NK22-family mapping and slow/fast diagnostics for the main ten AGN.

No MICA inputs or old single-object outputs are changed. All computation is
sequential with one BLAS thread. Input and physical provenance are separate.
"""
from pathlib import Path
import argparse
import csv
import gc
import json
import sys
import time

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from temperature_mapping import Inversion, DEFAULT, smooth_values, temporal_weights

OUT = ROOT / "outputs"
WIDTHS = (7, 28, 56)
XIS = (1, 10, 100, 1000)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def validate_input(d):
    n = len(d["t"])
    nb = len(d["bands"])
    assert nb >= 3 and n > nb
    for key in ("t", "flux", "error", "bi"):
        assert len(d[key]) == n and np.all(np.isfinite(d[key])), key
    assert np.all(d["error"] > 0)
    assert np.all(d["wavelengths"] > 0) and len(d["wavelengths"]) == nb
    assert np.array_equal(np.unique(d["bi"]), np.arange(nb))
    assert len(set(d["bands"].tolist())) == nb
    for b in range(nb):
        assert np.all(np.diff(d["t"][d["bi"] == b]) >= 0)


def smooth_band(t, f, e, width):
    # Tied times have no order-dependent interpolation: inverse-variance mean
    # only for the smoothing input; original likelihood retains every point.
    unique, inv = np.unique(t, return_inverse=True)
    w = np.bincount(inv, weights=1/e**2)
    y = np.bincount(inv, weights=f/e**2) / w
    sm = smooth_values(unique, y, width)
    return np.interp(t, unique, sm)


def coverage(d, p):
    records = []
    for b, name in enumerate(d["bands"]):
        t = np.unique(d["t"][d["bi"] == b])
        gaps = np.diff(t)
        records.append(dict(band=str(name), wavelength_obs_A=float(d["wavelengths"][b]),
            n=int(np.sum(d["bi"] == b)), t_min=float(t.min()), t_max=float(t.max()),
            span_days=float(np.ptp(t)), median_cadence_days=float(np.median(gaps)),
            max_gap_days=float(gaps.max()), tied_times=int(np.sum(d["bi"] == b)-len(t)),
            span_over_filter_width={str(w): float(np.ptp(t)/w) for w in WIDTHS},
            gaps_over_filter_width={str(w): int(np.sum(gaps >= w)) for w in WIDTHS}))
    common_start = max(r["t_min"] for r in records)
    common_end = min(r["t_max"] for r in records)
    return dict(bands=records, common_range=[common_start, common_end],
        common_span_days=max(0., common_end-common_start),
        time_grid_step_days=(p["time_grid_max"]-p["time_grid_min"])/(p["nt"]-1))


def numeric_self_test():
    """Independent dense augmented least squares and preservation of defaults."""
    t = np.tile(np.linspace(0., 20., 21), 3)
    bi = np.repeat(np.arange(3), 21)
    e = np.full(t.shape, .15)
    wave = np.array([1928., 3465., 5468.])
    p = dict(DEFAULT, nr=5, nt=8)
    m = Inversion(t, bi, wave, e, p)
    m.factorize(10.)
    f = 5 + bi + .3*np.sin(t*.4 + bi*.1)
    fit = m.solve(f)
    # Dense block normal equations include all free per-band offsets.
    matrix = np.block([[(m.normal+10*m.P).toarray(), m.cross],
                       [m.cross.T, np.diag(m.bd)]])
    rhs = np.r_[m.Aw.T@(f/e), m.Bw.T@(f/e)]
    dense = np.linalg.solve(matrix, rhs)
    qerr = float(np.max(np.abs(dense[:40].reshape(8,5).T-fit["q"])))
    pred = m.A@dense[:40] + m.B@dense[40:]
    pred_error = float(np.max(np.abs(pred-fit["pred"])))
    m2 = Inversion(t, bi, wave, e, dict(p, umin=1., time_grid_min=0., time_grid_max=20.))
    assert (m.A-m2.A).nnz == 0 and np.array_equal(m.u, m2.u)
    ring = temporal_weights(t, m.tp, 2.)
    ringerr = float(np.max(abs(ring.sum(axis=1)-1.)))
    constant_error = float(np.max(abs(smooth_band(t[:21], np.ones(21), e[:21], 28)-1)))
    assert max(qerr,pred_error,ringerr,constant_error) < 1e-8
    return dict(passed=True, dense_q_max_abs_error=qerr,
                dense_prediction_max_abs_error=pred_error,
                ring_partition_error=ringerr, constant_smoothing_error=constant_error,
                optional_parameters_preserve_default_matrix_exactly=True)


def choose_swift(bands):
    wanted = ({"w2", "uvw2"}, {"m2", "uvm2"}, {"w1", "uvw1"}, {"u"}, {"b"}, {"v"})
    labels = [str(b).lower().replace("_", " ").replace("swift", "").strip() for b in bands]
    selected = []
    for candidates in wanted:
        matches = [j for j, label in enumerate(labels) if label in candidates]
        # Do not mistake explicitly labelled ground bands for Swift counterparts.
        if len(matches) != 1:
            return None
        selected.extend(matches)
    return np.array(selected)


def fit_source(source, entry, physical):
    start = time.perf_counter()
    dest = OUT/source
    dest.mkdir(parents=True, exist_ok=True)
    inpath = Path(entry["input_path"])
    if not inpath.is_absolute():
        inpath = ROOT/inpath
    d = dict(np.load(inpath, allow_pickle=False))
    validate_input(d)
    original_zero = float(d["t"].min())
    d["t"] = d["t"] - original_zero
    p = dict(DEFAULT, **physical.get("defaults", {}))
    obj = physical["sources"][source]
    p.update(obj["params"])
    p.update(nr=50, nt=250, time_grid_min=0., time_grid_max=float(np.ptp(d["t"])),
             time_unit="observer days since first selected measurement")
    cov = coverage(d, p)
    m = Inversion(d["t"], d["bi"], d["wavelengths"], d["error"], p)
    saved = dict(d, tp=m.tp, u=m.u, edges=m.edges, temp=m.temp, kernel=m.kernel,
                 input_time_zero=np.array(original_zero))
    metrics = []
    checks = {}
    band_residuals = []
    variants = {}
    for width in WIDTHS:
        low = np.empty_like(d["flux"])
        for b in range(len(d["bands"])):
            ii = np.flatnonzero(d["bi"] == b)
            low[ii] = smooth_band(d["t"][ii], d["flux"][ii], d["error"][ii], width)
        variants[f"low{width}"] = low
        variants[f"high{width}"] = d["flux"]-low

    def record(name, fit, nt=250, xi=10, **extra):
        assert np.isfinite(fit["q"]).all() and np.isfinite(fit["pred"]).all()
        assert fit["normal_equation_relative_residual"] < 1e-8, (source, name)
        metrics.append(dict(name=name, nt=nt, xi=xi, **{k:fit[k] for k in
            ("chi2", "chi2_per_point", "scale99", "normal_equation_relative_residual", "penalty")},
            max_abs_q=float(np.max(abs(fit["q"]))),
            fraction_abs_q_over_0p2=float(np.mean(abs(fit["q"]) > .2)), **extra))
        saved["q_"+name], saved["pred_"+name] = fit["q"], fit["pred"]

    for xi in XIS:
        sec = m.factorize(xi)
        fit = m.solve(d["flux"])
        record(f"xi{xi}", fit, xi=xi, factor_seconds=sec)
        if xi != 10:
            continue
        edf, edfse = m.edf(32, seed=1729)
        metrics[-1].update(effective_parameters=edf, effective_parameters_mcse=edfse)
        base = fit
        for name, flux in variants.items():
            saved["flux_"+name] = flux
            record(name, m.solve(flux))
        for width in WIDTHS:
            checks[f"low_high_{width}_q_max_abs_error"] = float(np.max(abs(
                saved[f"q_low{width}"]+saved[f"q_high{width}"]-base["q"])))
            checks[f"low_high_{width}_prediction_max_abs_error"] = float(np.max(abs(
                saved[f"pred_low{width}"]+saved[f"pred_high{width}"]-base["pred"])))
        shifted = m.solve(d["flux"] + (1+np.arange(len(d["bands"])))[d["bi"]]*3.)
        checks["constant_band_offset_q_max_abs_error"] = float(np.max(abs(shifted["q"]-base["q"])))
        for b, band in enumerate(d["bands"]):
            ii=d["bi"]==b
            r=base["residual"][ii]
            band_residuals.append(dict(band=str(band), n=int(ii.sum()),
                chi2_per_point=float(np.mean((r/d["error"][ii])**2)),
                rms_flux=float(np.sqrt(np.mean(r*r)))))
    assert max(checks.values()) < 1e-7, (source, checks)
    # Shape comparisons use only the common band window and the central 90%
    # of combined normalized radial kernels. This is NOT a resolution estimate.
    k = m.kernel / np.sum(m.kernel, axis=1, keepdims=True)
    radial_cdf = np.cumsum(k.mean(axis=0))
    radial = (radial_cdf >= .05) & (radial_cdf <= .95)
    window = (m.tp >= cov["common_range"][0]) & (m.tp <= cov["common_range"][1])
    mask = radial[:,None] & window[None,:]
    if mask.sum() < 10:
        mask = np.ones(base["q"].shape, dtype=bool)
        comparison_region = "full grid: no adequate all-band common window"
    else:
        comparison_region = "all-band common window, central 90% cumulative mean normalized kernel"
    saved["comparison_mask"] = mask
    diagnostics = {}
    for width in WIDTHS:
        diagnostics[f"low{width}_q_rms_ratio"] = float(np.sqrt(np.mean(saved[f"q_low{width}"][mask]**2)/np.mean(base["q"][mask]**2)))
        diagnostics[f"high{width}_q_rms_ratio"] = float(np.sqrt(np.mean(saved[f"q_high{width}"][mask]**2)/np.mean(base["q"][mask]**2)))
    for xi in XIS:
        diagnostics[f"xi{xi}_map_correlation"] = correlation(base["q"][mask], saved[f"q_xi{xi}"][mask])
    del m
    gc.collect()
    # Nt changes are discretization/penalty sensitivity, not proof of convergence.
    for nt in (100, 500):
        mm = Inversion(d["t"], d["bi"], d["wavelengths"], d["error"], dict(p, nt=nt))
        mm.factorize(10)
        fit = mm.solve(d["flux"])
        record(f"nt{nt}", fit, nt=nt)
        saved[f"tp_nt{nt}"] = mm.tp
        interpolated = np.vstack([np.interp(saved["tp"], mm.tp, q) for q in fit["q"]])
        diagnostics[f"nt{nt}_map_correlation"] = correlation(base["q"][mask], interpolated[mask])
        del mm
        gc.collect()
    subset_info = {}
    swift = choose_swift(d["bands"])
    sets = {"without_U":np.flatnonzero((d["wavelengths"]<3200)|(d["wavelengths"]>3800))}
    if swift is not None:
        sets["swift6"] = np.sort(swift)
    for name, bands in sets.items():
        subset_info[name] = dict(bands=d["bands"][bands].tolist(),
            excluded=[str(x) for j,x in enumerate(d["bands"]) if j not in bands])
        if len(bands) == len(d["bands"]):
            saved["q_"+name] = base["q"].copy()
            diagnostics[name+"_map_correlation"] = 1.
            subset_info[name]["status"] = "identical to primary selection; no refit needed"
            continue
        if len(bands)<3:
            subset_info[name]["status"] = "not run: fewer than three remaining bands"
            continue
        ii = np.flatnonzero(np.isin(d["bi"], bands))
        remap = {int(b):j for j,b in enumerate(bands)}
        bi = np.array([remap[int(b)] for b in d["bi"][ii]])
        mm = Inversion(d["t"][ii], bi, d["wavelengths"][bands], d["error"][ii], p)
        assert np.array_equal(mm.tp, saved["tp"])
        mm.factorize(10)
        fit = mm.solve(d["flux"][ii])
        record(name, fit, n_subset=int(len(ii)))
        diagnostics[name+"_map_correlation"] = correlation(base["q"][mask], fit["q"][mask])
        subset_info[name]["status"] = "refit on identical radial/time grid and xi"
        saved["index_"+name] = ii
        del mm
        gc.collect()
    if swift is None:
        subset_info["swift6"] = {"status":"not run: six Swift bands not uniquely identified"}
    warnings = list(entry.get("warnings", [])) + list(obj.get("warnings", []))
    span = float(np.ptp(d["t"]))
    if len(d["bands"]) <= 6:
        warnings.append("Only six (or fewer) bands: radial inverse problem remains strongly underdetermined.")
    if span < 112:
        warnings.append("Monitoring spans fewer than two 56-day FWHM widths; 56-day result is strongly window/edge dependent.")
    if cov["common_span_days"] < 112:
        warnings.append("All-band common time envelope spans fewer than two 56-day widths; a longer total span from extra ground-only coverage does not repair the multiband limitation.")
    if any(r["max_gap_days"]>=28 for r in cov["bands"]):
        warnings.append("At least one band has a gap >=28 days; Gaussian filtering interpolates across gaps.")
    if base["scale99"]>.2:
        warnings.append("99th percentile |deltaT/T| >0.2: inspect the small-perturbation approximation; 0.2 is a descriptive flag, not a validated rejection threshold.")
    warnings.append("Maps are regularized estimates, not directly resolved disk images; no propagation-significance or physical-component identification is established.")
    runtime = time.perf_counter()-start
    summary = dict(source=source, n_bands=len(d["bands"]), n_data=len(d["t"]),
        time_span_days=span, dt_days=span/249, q99=base["scale99"],
        chi2_per_point=base["chi2_per_point"], fit_seconds=runtime, params=p,
        original_time_zero=original_zero, input_path=str(inpath),
        quality_warnings=warnings, diagnostics=diagnostics, subset_tests=subset_info,
        comparison_region=comparison_region, coverage=cov,
        low_high_statistical_scope="Same original errors used as inversion weights; smoothing-induced covariance NOT propagated. Filtered chi2 values are diagnostics, not likelihood goodness-of-fit.",
        regularization_scope="Same xi and grid counts do not impose the same physical smoothing in sources with different spans, masses, errors and band kernels.",
        uncertainty_scope="Fixed physical parameters, no pixel credible intervals or model-discrepancy posterior.")
    np.savez_compressed(dest/"fits.npz", **saved)
    dump(dest/"metrics.json", metrics)
    dump(dest/"summary.json", summary)
    dump(dest/"validation.json", dict(passed=True, **checks))
    dump(dest/"band_residuals.json", band_residuals)
    print(f"DONE {source}: {len(d['bands'])} bands, {len(d['t'])} points, chi2/N={base['chi2_per_point']:.3f}, q99={base['scale99']:.3f}, {runtime:.1f}s", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="*")
    parser.add_argument("--reuse", action="store_true", help="Explicitly reuse existing source summaries; caller verifies unchanged input/config")
    args = parser.parse_args()
    threadpool_limits(1)
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT/"numerical_validation.json", numeric_self_test())
    manifest = json.loads((ROOT/"derived/input_manifest.json").read_text())
    physical = json.loads((ROOT/"physical_parameters.json").read_text())
    results, failures = [], []
    for entry in manifest["sources"]:
        source=entry["source"]
        if args.sources and source not in args.sources:
            continue
        try:
            prior=OUT/source/"summary.json"
            result=json.loads(prior.read_text()) if args.reuse and prior.exists() else fit_source(source,entry,physical)
            results.append(result)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            failures.append(dict(source=source, error=repr(exc)))
        gc.collect()
        dump(OUT/"batch_status.json", dict(completed=[r["source"] for r in results], failures=failures))
    dump(OUT/"summary.json", results)
    fields=["source","n_bands","n_data","time_span_days","dt_days","chi2_per_point","q99","fit_seconds"]
    with (OUT/"summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(results)
    if failures:
        raise SystemExit(f"{len(failures)} source(s) failed; see batch_status.json")


if __name__ == "__main__":
    main()
