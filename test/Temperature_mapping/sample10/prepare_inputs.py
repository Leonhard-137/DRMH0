#!/usr/bin/env python3
"""Prepare the fixed main-ten sample without modifying any source data.

Run using the astro Python. All outputs are confined to this directory's
derived/ and DATA_PROVENANCE.md. No additional dereddening is applied by this
pipeline; upstream correction histories are not universally established.
No MICA/FFA host-subtracted inputs are used. The physical origin of a structure is not
established by preparing these inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TM = HERE.parent
OUT = HERE / "derived"
ORDER = ["Mrk817_a", "Mrk142", "Mrk335", "F9_a", "Mrk279_b", "Mrk509",
         "NGC5548", "NGC4151_a", "NGC6814", "NGC2617_a"]
FLUX_UNITS = "1e-15 erg s^-1 cm^-2 Angstrom^-1"
UNIT_EVIDENCE_TABLE = """## Physical-flux unit evidence

The adopted numerical unit is `1e-15 erg s^-1 cm^-2 Angstrom^-1` for flux and error. A documented unit is not an independent reconstruction of photometric calibration.

| Input | Evidence strength and boundary |
|---|---|
| Mrk817_a; NGC5548 | Reuse previously audited physical-flux inputs. Mrk817_a HST/Swift flux/error rows match the full reference; all NGC5548 arrays equal the audited input. |
| Mrk142 | Published machine-readable table explicitly specifies these physical units; the arbitrary mean-normalized Lijiang lV curve is excluded. |
| Mrk335 | Ground-table units explicitly verified. Swift adopts the existing project physical-flux convention; its calibration was not independently reconstructed in this run. |
| F9_a; Mrk279_b; NGC6814 | Public processed-data archive explicitly specifies these units; local-release identity is not certified row-for-row. NGC6814 six-band counts and first records match corrected record20430106, not an entire-array comparison. |
| Mrk509; NGC4151_a | Campaign/source identified; physical-flux units adopt the existing project convention, not an independently reconstructed calibration in this run. |
| NGC2617_a | Assumed project f_lambda units; original Vega zero-point/band conversion and whether prior dereddening occurred have NOT been reconstructed in this run. |

No additional dereddening is applied by this pipeline. This does not establish that every upstream source is uncorrected; absolute temperature amplitudes remain conditional on the stated unit/calibration assumptions."""
ERROR_METHOD = "NK22 arithmetic MEAN of nonnegative extra standard deviations from adjacent three-point straight-line chi2=1 fits; added in quadrature"
SWIFT_ALIAS = {"W2": "W2", "UVW2": "W2", "M2": "M2", "UVM2": "M2",
               "W1": "W1", "UVW1": "W1", "U": "U", "B": "B", "V": "V"}
GROUND_WAVES = {"Mrk142": {"u": 3540., "g": 4770., "r": 6215., "i": 7545., "z": 8700.},
                "Mrk335": {"g": 4770., "r": 6215., "i": 7545., "z": 8700.}}
PAPERS = {
    "Mrk142": "https://arxiv.org/html/2005.03685",
    "Mrk335": "https://arxiv.org/html/2302.07342",
    "F9_a": "https://doi.org/10.5281/zenodo.14930797",
    "Mrk279_b": "https://doi.org/10.5281/zenodo.14930797",
    "Mrk509": "https://arxiv.org/abs/1811.07956",
    "NGC4151_a": "https://arxiv.org/abs/1811.07956",
    "NGC6814": "https://doi.org/10.1093/mnras/stae1353",
    "NGC2617_a": "https://arxiv.org/abs/1310.2241",
}
WARNINGS = {
    "Mrk142": ["Lijiang lV (62 rows) excluded: arbitrary flux normalized to mean 1, NOT physical f_lambda (Cackett2020 Table2).",
               "Ground r/i/z counts differ from paper Table1; this is the supplied local screened release, not an exact published selection.",
               "Swift wavelength convention follows project config (M2=2246,U=3465 A); paper Table3 gives 2236,3467 A.",
               "Ground monitoring extends beyond Swift; those epochs have weaker radial information."],
    "Mrk335": ["Swift physical-flux units adopt the existing project convention; Swift calibration was not independently reconstructed in this run. Ground-table physical-flux units were explicitly verified.",
               "Ground griz are the supplied local intercalibrated release; counts and some flux/error values differ from Kara2023 published Table2.",
               "Time origin is retained as local MJD. Cross-instrument UTC/TDB and heliocentric-conversion provenance is not fully documented. Published ground-table comparisons show telescope-dependent offsets; sub-day propagation interpretation needs a separate timing audit.",
               "z combines z-prime/z_s; 8700 A is the paper Table3 effective-wavelength approximation, not a measured transmission kernel.",
               "Ground monitoring extends beyond Swift; those epochs have weaker radial information."],
    "F9_a": ["Late Fairall9 campaign only; local screened/cropped release differs from Wang2025 segment2 and is NOT the early Fairall9 dataset mapped in NK22."],
    "Mrk279_b": ["Segment2 only; no data from Mrk279_a or segment3 are merged."],
    "Mrk509": ["2017 Swift campaign; six UVOT bands only in this local source table.",
               "Physical-flux units adopt the existing project convention; photometric calibration was not independently reconstructed in this run."],
    "NGC4151_a": ["2016 Swift campaign only; no later NGC4151_c data are merged.",
                  "Physical-flux units adopt the existing project convention; photometric calibration was not independently reconstructed in this run."],
    "NGC6814": ["No additional dereddening is applied by this pipeline despite substantial Galactic reddening; upstream correction history is not independently established, and temperature amplitudes are conditional on the supplied-flux convention.",
                "Gonzalez2024 correction (stae1353) changes UV calibration and withdraws the extreme blue-spectrum/truncation conclusion of the original paper.",
                "Zenodo14930797 has erroneous NGC6814 bands. Local counts and first time/flux/error rows of ALL six bands match corrected record20430106, Lightcurves_new/NGC6814(All); entire-array equality was not checked. No re-calibration or detrending applied."],
    "NGC2617_a": ["2013 outburst Swift subset only (about50 d); no later NGC2617 campaign is merged.",
                  "Assumed project f_lambda units; original Vega zero-point/band conversion and whether prior dereddening occurred have NOT been reconstructed in this run."]}


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def floor_stats(t, f, e):
    """Exact analytic additional SD for a weighted three-point straight line."""
    floors, chi, skipped = [], [], 0
    for j in range(len(t) - 2):
        x, y, s = t[j:j+3], f[j:j+3], e[j:j+3]
        span = x[-1] - x[0]
        if span <= 0:
            skipped += 1
            continue
        a = np.array([x[1]-x[2], x[2]-x[0], x[0]-x[1]]) / span
        contrast2 = float(a @ y)**2
        variance = float((a*a) @ (s*s))
        floors.append(np.sqrt(max(0., (contrast2-variance)/(a@a))))
        chi.append(contrast2/variance)
    if not floors:
        raise ValueError("No valid adjacent triplets for mean error-floor estimation")
    return dict(systematic_floor=float(np.mean(floors)), triplets_used=len(floors),
                triplets_skipped=skipped, median_triplet_chi2_original=float(np.median(chi)))


def pack(series, existing_errors=False):
    """Series dict maps labels to wavelength,t,f,e_original[,e_corrected]."""
    bands = sorted(series, key=lambda b: (series[b][0], b))
    chunks, inventory = [], []
    for k, band in enumerate(bands):
        wave, t, f, e0, *corrected = series[band]
        ix = np.argsort(t, kind="stable")
        t, f, e0 = (np.asarray(x, dtype=float)[ix] for x in [t, f, e0])
        assert np.isfinite(np.column_stack([t, f, e0])).all() and np.all(e0 > 0)
        assert len(np.unique(t)) >= 2
        if existing_errors:
            e = np.asarray(corrected[0], dtype=float)[ix]
            delta = e*e-e0*e0
            assert np.min(delta) > -1e-10
            # Preserve values exactly; this check does not reapply a floor.
            stat = dict(systematic_floor=float(np.sqrt(max(0., np.median(delta)))),
                        triplets_used=None, triplets_skipped=None,
                        floor_reused_without_reinflation=True)
            assert np.max(abs(delta-np.median(delta))) < 1e-8
        else:
            stat = floor_stats(t, f, e0)
            e = np.hypot(e0, stat["systematic_floor"])
        unique = np.unique(t)
        gaps = np.diff(unique)
        exact_duplicates = len(t)-len(np.unique(np.column_stack([t, f, e0]), axis=0))
        if exact_duplicates:
            raise ValueError(f"{band}: exact duplicated measurements require provenance resolution")
        chunks.append(np.column_stack([t, f, e, np.full(len(t), k), e0]))
        inventory.append(dict(band=band, wavelength_obs_A=float(wave), n=len(t),
            t_min=float(t.min()), t_max=float(t.max()), span_days=float(np.ptp(t)),
            median_cadence_days=float(np.median(gaps)), max_gap_days=float(np.max(gaps)),
            duplicate_times=len(t)-len(unique), exact_duplicate_records=exact_duplicates,
            median_flux=float(np.median(f)), median_error_original=float(np.median(e0)),
            median_error=float(np.median(e)), median_error_ratio=float(np.median(e/e0)), **stat))
    a = np.vstack(chunks)
    data = dict(t=a[:, 0], flux=a[:, 1], error=a[:, 2], bi=a[:, 3].astype(int),
                error_original=a[:, 4], bands=np.asarray(bands),
                wavelengths=np.asarray([series[b][0] for b in bands]))
    return data, inventory


def generic(source):
    raw = ROOT/source/"raw"/(source+".csv")
    config = ROOT/source/"config/source_config.yaml"
    cfg = yaml.safe_load(config.read_text())
    records = read_csv(raw)
    series, omissions, waveprov = {}, [], {}
    for band in dict.fromkeys(r["Filter"] for r in records):
        rr = [r for r in records if r["Filter"] == band]
        if band in SWIFT_ALIAS:
            label = "Swift_"+SWIFT_ALIAS[band]
            wave = float(cfg["bands"][band])
            waveprov[label] = str(config)+":bands"
        elif band in GROUND_WAVES.get(source, {}):
            label = "Ground_"+band
            wave = GROUND_WAVES[source][band]
            waveprov[label] = PAPERS[source]+" Table3, observed-frame effective wavelength"
        else:
            reason = "arbitrary normalized flux, not calibrated f_lambda" if band == "lV" else "X-ray count rate/energy flux or unsupported non-continuum observable"
            omissions.append(dict(band=band, n=len(rr), reason=reason))
            continue
        a = np.array([[float(r["MJD"]), float(r["Flux"]), float(r["Error"])] for r in rr])
        # No rescaling, host subtraction, extinction correction or time shift.
        series[label] = (wave, a[:, 0], a[:, 1], a[:, 2])
    data, inventory = pack(series)
    warnings = list(WARNINGS.get(source, []))
    if len(series) == 6:
        warnings.append("Only six Swift UVOT bands:1928--5468 A; radial leverage is substantially smaller than HST+optical sets.")
    warnings.append("UTC/TDB and observatory light-travel conventions are not fully documented in local CSV; preserve the common within-source MJD column, do not assert exact sub-day timing calibration.")
    for row in inventory:
        if row["duplicate_times"]:
            warnings.append(f"{row['band']} retains {row['duplicate_times']} same-time nonidentical measurements (rounded times/multiple sites); interpolation must average tied times instead of selecting an arbitrary point.")
    meta = dict(source=source, input_files=[str(raw), str(config), str(ROOT/source/"raw/info.txt")],
                time_origin="Original local MJD column retained, observer-frame days; no origin shift", time_system="local labelled MJD; detailed UTC/TDB/heliocentric provenance not fully recorded",
                flux_units=FLUX_UNITS, flux_conversion_factor=1.,
                flux_provenance=PAPERS[source]+" and existing project raw-flux convention; no fitted/host-subtracted products",
                wavelength_provenance=waveprov, error_method=ERROR_METHOD,
                campaign_selection="entire source-specific raw CSV, without merging similarly named campaigns",
                omitted_bands=omissions, warnings=warnings, bands=inventory)
    return data, meta


def mrk817():
    full_path = TM/"derived/lightcurves.csv"
    raw_path = ROOT/"Mrk817_a/raw/Mrk817_a.csv"
    full, raw = read_csv(full_path), read_csv(raw_path)
    aliases = {**{f"{x}A": f"HST_{x}" for x in [1180,1398,1502,1739]},
               **{x:"Swift_"+x for x in ["UVW2","UVM2","UVW1","U","B","V"]}}
    matched, checks = [], []
    # Match the selected campaign's *individual measurements* to the audited
    # published-time sequence; +/-0.5 alone is not a heliocentric conversion.
    for old, new in aliases.items():
        rr = [r for r in raw if r["Filter"] == old]
        ff = [r for r in full if r["band"] == new]
        a = np.array([[float(r["MJD"]),float(r["Flux"]),float(r["Error"])] for r in rr])
        b = np.array([[float(r["thjd"]),float(r["flux"]),float(r["error_original"])] for r in ff])
        j = np.argmin(abs(a[:,0,None]-49999.5-b[:,0]), axis=1)
        assert len(set(j)) == len(j)
        assert np.max(abs(a[:,1:]-b[j,1:])) < 1e-7
        assert np.max(abs(a[:,0]-49999.5-b[j,0])) < .01
        matched.extend(b[j,0])
        checks.append(dict(local_band=old, full_band=new, matched=len(j),
            maximum_flux_error_difference=float(np.max(abs(a[:,1:]-b[j,1:]))),
            max_abs_naive_time_offset_days=float(np.max(abs(a[:,0]-49999.5-b[j,0])))))
    lower, upper = float(min(matched)), float(max(matched))
    selected = [r for r in full if lower-1e-9 <= float(r["thjd"]) <= upper+1e-9]
    series = {}
    for band in dict.fromkeys(r["band"] for r in selected):
        rr = [r for r in selected if r["band"] == band]
        label = band
        for old,new in [("UVW2","W2"),("UVM2","M2"),("UVW1","W1")]:
            if band == "Swift_"+old: label = "Swift_"+new
        a = np.array([[float(r["thjd"]),float(r["flux"]),float(r["error_original"])] for r in rr])
        series[label] = (float(rr[0]["wavelength_obs_A"]), a[:,0], a[:,1], a[:,2])
    data, inventory = pack(series)
    assert sum(x["n"] for x in inventory if x["band"].startswith(("HST_","Swift_"))) == len(raw)
    return data, dict(source="Mrk817_a", input_files=[str(full_path),str(raw_path),str(TM/"derived/data_metadata.json")],
        time_origin="THJD=published HJD-2450000, observer-frame days", time_system="published heliocentric JD; inherited audited full817 timeline",
        flux_units=FLUX_UNITS, flux_conversion_factor=1., wavelength_provenance=str(TM/"derived/data_metadata.json"),
        error_method=ERROR_METHOD+"; recomputed from error_original AFTER campaign selection, not from existing median-expanded errors",
        campaign_selection="_a endpoints identified by exact flux/error matching of all local HST/Swift rows, then applied as a common published-THJD window to all17 bands",
        campaign_thjd_bounds=[lower,upper], campaign_mjd_bounds=[min(float(r["MJD"]) for r in raw),max(float(r["MJD"]) for r in raw)],
        endpoint_matching_checks=checks, n_full_reference=len(full), n_outside_selected_epoch=len(full)-len(selected), omitted_bands=[], bands=inventory,
        warnings=["This is Mrk817_a (about420 d), not the full524-day reference. The original full817 reproduction is unchanged.",
                  "New errors use NK22 mean floors, whereas the full817 reference used NK24 median floors. Compare with awareness of both changes.",
                  "Time bounds use matched published HJD, NOT an assumed MJD+0.5 conversion."])


def ngc5548():
    path = TM/"NGC5548/derived/local_input.npz"
    with np.load(path, allow_pickle=False) as z:
        original = {k:z[k].copy() for k in z.files}
    series = {str(b):(original["wavelengths"][k], original["t"][original["bi"]==k],
                original["flux"][original["bi"]==k],original["error_original"][original["bi"]==k],original["error"][original["bi"]==k])
              for k,b in enumerate(original["bands"])}
    data, inventory = pack(series, existing_errors=True)
    assert set(data) == set(original)
    assert all(np.array_equal(data[k],original[k]) for k in data), "NGC5548 reuse must be array-identical"
    return data, dict(source="NGC5548", input_files=[str(path),str(TM/"NGC5548/derived/data_audit.json"),str(TM/"NGC5548/outputs/source_quality.json")],
        time_origin="THJD=HJD-2450000, geocentric local MJD_UTC plus builtin heliocentric correction", time_system="heliocentric UTC approximation; detailed original observatory timescale is not fully recorded",
        flux_units=FLUX_UNITS, flux_conversion_factor=1., wavelength_provenance=str(TM/"NGC5548/derived/data_audit.json"),
        error_method=ERROR_METHOD+"; already applied in audited local_input.npz; ARRAY-IDENTICAL reuse, no second inflation",
        campaign_selection="Full audited 2014 STORM campaign including HST/ground coverage outside the shorter Swift window", bands=inventory,
        omitted_bands=[dict(band="HX+SX",n=536,reason="X-ray count rates excluded in original preparation")],
        warnings=["Later public5900-point TempMap bundle is NOT used: it mixes Swift MJD and HST/ground HJD and includes another Swift release.",
                  "This input is4203 original observed points; no extinction or host correction.",
                  "Public wavelength vector is inherited; NK22 text lists slightly different U/u/z wavelengths. This is already tested in the individual5548 reproduction."])


def validate(data):
    n, nb = len(data["t"]), len(data["bands"])
    assert all(data[k].shape == (n,) for k in ["flux","error","error_original","bi"])
    assert len(set(data["bands"])) == nb and data["wavelengths"].shape == (nb,)
    assert np.all(np.diff(data["wavelengths"]) >= 0)
    assert np.array_equal(np.unique(data["bi"]),np.arange(nb))
    assert np.isfinite(np.column_stack([data[k] for k in ["t","flux","error","error_original"]])).all()
    assert np.all(data["error"] > 0) and np.all(data["error"] >= data["error_original"])
    assert all(np.all(np.diff(data["t"][data["bi"]==k]) >= 0) for k in range(nb))
    return dict(finite_arrays=True, positive_errors=True, no_double_inflation=True,
                wavelength_ordered_unique_labels=True, bandwise_time_ordered=True,
                minimum_two_unique_epochs_per_band=True)


def write_source(source):
    data, meta = mrk817() if source == "Mrk817_a" else ngc5548() if source == "NGC5548" else generic(source)
    checks = validate(data)
    path = OUT/(source+"_input.npz")
    np.savez_compressed(path, **data)
    with np.load(path, allow_pickle=False) as z:
        assert all(np.array_equal(z[k],data[k]) for k in data)
    meta.update(input_path=str(path), n_data=len(data["t"]), n_bands=len(data["bands"]),
        t_min=float(data["t"].min()), t_max=float(data["t"].max()), span_days=float(np.ptp(data["t"])),
        extinction_correction="no additional dereddening applied by this pipeline; upstream correction history is source-dependent and not fully reconstructed", host_subtraction="none", validation=checks,
        swift_bands=[str(b) for b in data["bands"] if str(b).startswith("Swift_") or str(b) in ["W2","M2","W1","Swift U","Swift B","Swift V"]])
    csvpath = OUT/(source+"_lightcurves.csv")
    with csvpath.open("w",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["band","wavelength_obs_A","t","flux","error","error_original"])
        for j in range(len(data["t"])):
            k=data["bi"][j]
            writer.writerow([data["bands"][k],data["wavelengths"][k],data["t"][j],data["flux"][j],data["error"][j],data["error_original"][j]])
    dump(OUT/(source+"_metadata.json"),meta)
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources",nargs="+",choices=ORDER,default=ORDER)
    args=parser.parse_args()
    actual=[r["source"] for r in read_csv(ROOT/"DRMH0/tables/sample_observations.csv")]
    assert actual == ORDER, "Main-ten sample changed: explicitly review before running"
    OUT.mkdir(parents=True,exist_ok=True)
    index=OUT/"input_manifest.json"
    sources={r["source"]:r for r in json.loads(index.read_text())["sources"]} if index.exists() else {}
    for s in args.sources:
        sources[s]=write_source(s)
        # Publish each successful source immediately; fitter need not await all.
        dump(index,dict(schema_version=1,source_order=ORDER,sources=[sources[s] for s in ORDER if s in sources],
            sample_definition=str(ROOT/"DRMH0/tables/sample_observations.csv"),
            flux_units=FLUX_UNITS,error_method=ERROR_METHOD,absolute_time_handling="Each source retains its coherent local origin; fitter may subtract its own minimum time. No cross-source absolute phase is meaningful."))
        m=sources[s]
        print(f"{s}: {m['n_data']} points, {m['n_bands']} bands, {m['span_days']:.3f} days -> {m['input_path']}",flush=True)
    rows=["# Main-ten temperature-mapping input provenance", "", "These inputs preserve supplied flux values and measurement cadence. No additional dereddening is applied by this pipeline; upstream correction histories are not universally established. No MICA/FFA host-subtracted inputs are used. Flux and error adopt the same units with the evidence strengths below. Numerical validation does not establish the physical origin of map patterns.", "", UNIT_EVIDENCE_TABLE, "", "## Data inventory", "", "| Source | Bands | Points | Span (d) |", "|---|---:|---:|---:|"]
    for s in ORDER:
        if s in sources:
            m=sources[s];rows.append(f"| {s} | {m['n_bands']} | {m['n_data']} | {m['span_days']:.3f} |")
    rows += ["", "## Frozen preprocessing", "", "- "+ERROR_METHOD+".",
        "- Mrk817_a: exact matching determines published-time campaign endpoints; original errors are re-expanded using the mean floor. Full817 reference remains unchanged.",
        "- NGC5548: every NPZ array is verified identical to the existing audited local_input.npz; no reinflation.",
        "- Other sources: local MJD and physical-flux convention are retained. No unknown UTC/TDB/heliocentric correction is invented.",
        "- X-ray bands and Mrk142's uncalibrated Lijiang lV are omitted. Nonidentical same-time rows are retained and flagged.",
        "- Per-band cadence/gaps, exact paths, wavelength provenance, error floors and exclusions are in derived/input_manifest.json and per-source metadata."]
    for s in ORDER:
        if s not in sources: continue
        m=sources[s];rows += ["", "## "+s, "", "Input: `"+m["input_path"]+"`", "", "Time: "+m["time_origin"], "", "Selection: "+m["campaign_selection"], ""]
        rows.extend("- "+w for w in m["warnings"])
        if s in PAPERS: rows += ["", "Primary source: "+PAPERS[s]]
    (HERE/"DATA_PROVENANCE.md").write_text("\n".join(rows)+"\n")


if __name__ == "__main__":
    main()
