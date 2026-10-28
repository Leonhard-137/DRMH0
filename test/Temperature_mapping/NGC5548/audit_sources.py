#!/usr/bin/env python3
"""Bounded NGC5548 source audit; writes only outputs/source_quality.json.

Run with the existing astro Python and -B. No downloads, fits, plots, or
reproduction imports. --stdout performs the same calculations without writing.
An exit code of zero means the audit reproduced, not that the public data pass
scientific acceptance. Inspect severity and public_primary_eligible separately.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import astropy
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import data as astro_data, iers
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parents[2] / "NGC5548/raw/NGC5548.csv"
PUBLIC = ROOT / "reference/NGC5548_TempMap.npz"
LOCAL = ROOT / "derived/local_input.npz"
OUTPUT = ROOT / "outputs/source_quality.json"
# Explicit identities: ground B/V are BG/VG, distinct from Swift B/V.
BANDS = [
    ("W2", "W2", "Swift", 566, 260),
    ("M2", "M2", "Swift", 519, 249),
    ("W1", "W1", "Swift", 544, 261),
    ("Swift U", "U", "Swift", 552, 267),
    ("Swift B", "B", "Swift", 557, 271),
    ("Swift V", "V", "Swift", 545, 263),
    ("HST 1158", "HST1157.5", "HST", 171, 171),
    ("HST 1367", "HST1367", "HST", 171, 171),
    ("HST 1479", "HST1478.5", "HST", 171, 171),
    ("HST 1746", "HST1746", "HST", 171, 171),
    ("SDSS u", "u", "Ground", 164, 164),
    ("Johnson B", "BG", "Ground", 176, 180),
    ("SDSS g", "g", "Ground", 204, 204),
    ("Johnson V", "VG", "Ground", 494, 497),
    ("SDSS r", "r", "Ground", 203, 203),
    ("Johnson R", "R", "Ground", 157, 161),
    ("SDSS i", "i", "Ground", 208, 208),
    ("Johnson I", "I", "Ground", 115, 119),
    ("SDSS z", "z", "Ground", 212, 212),
]
TIME_TOLERANCE_SECONDS = 0.002
FLUX_TOLERANCE = 1e-9  # In local 1e-15 per-Angstrom numerical units.


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "median": float(np.median(a)),
            "max": float(a.max())} if a.size else None


def coverage(t):
    t = np.asarray(t, dtype=float)
    dt = np.diff(np.sort(t))
    return {"n": int(t.size), "time_min": float(t.min()),
            "time_max": float(t.max()), "span_days": float(np.ptp(t)),
            "median_cadence_days": float(np.median(dt)),
            "mean_cadence_days": float(np.mean(dt)),
            "max_gap_days": float(dt.max()),
            "duplicate_times": int(t.size - np.unique(t).size),
            "strictly_increasing": bool(np.all(np.diff(t) > 0))}


def heliocentric_days(mjd):
    """Same geocentric UTC convention tested against the published HJD grids."""
    iers.conf.auto_download = False
    astro_data.conf.allow_internet = False
    location = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    target = SkyCoord("14h17m59.51s", "+25d08m12.5s", frame="icrs")
    # Explicit packaged IERS table avoids IERS_Auto and its cache/network paths.
    with iers.earth_orientation_table.set(iers.IERS_B.open(iers.IERS_B_FILE)):
        t = Time(mjd, format="mjd", scale="utc", location=location)
        return t.light_travel_time(target, kind="heliocentric",
                                  ephemeris="builtin").to_value(u.day)


def audit():
    with RAW.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    public, local = load_npz(PUBLIC), load_npz(LOCAL)
    labels = public["filters"].astype(str).tolist()
    local_labels = local["bands"].astype(str).tolist()
    required = {row[0] for row in BANDS}
    if len(labels) != 19 or set(labels) != required or set(local_labels) != required:
        raise ValueError("Unexpected band identities; audit mapping needs review")
    if public["dates"].shape != (19, 566):
        raise ValueError("Unexpected public array shape")
    if any(public[k].shape != public["dates"].shape for k in ("flux", "error")):
        raise ValueError("Public flux/error shapes differ from dates")
    if any(local[k].shape != local["t"].shape
           for k in ("flux", "error", "error_original", "bi")):
        raise ValueError("Derived local arrays have inconsistent shapes")
    checks = []

    def check(name, passed, measured=None):
        checks.append({"name": name, "passed": bool(passed), "measured": measured})

    mask = np.isfinite(public["dates"])
    check("public_padding_aligned", all(np.array_equal(mask, np.isfinite(public[k]))
                                        for k in ("flux", "error")))
    check("public_valid_values", np.all(public["error"][mask] > 0)
          and np.all(public["flux"][mask] > 0))
    check("raw_row_count", len(records) == 4739, len(records))
    raw_filters = [row[1] for row in BANDS]
    optical = [r for r in records if r["Filter"] in raw_filters]
    excluded = [r for r in records if r["Filter"] not in raw_filters]
    check("raw_optical_count", len(optical) == 4203, len(optical))
    check("excluded_only_xray", len(excluded) == 536
          and {r["Filter"] for r in excluded} == {"HX", "SX"})
    check("raw_optical_units", {r["Unit"] for r in optical}
          == {"1e-15 erg cm^-2 s^-1 Angstrom^-1"})
    check("raw_time_labels", {r["TimeSystem"] for r in records} == {"MJD_UTC"})
    check("raw_exact_duplicates_absent", len({tuple(r.values()) for r in records})
          == len(records))
    check("public_point_count", int(mask.sum()) == 5900, int(mask.sum()))
    check("derived_local_point_count", len(local["t"]) == 4203, len(local["t"]))
    check("derived_local_valid_values", all(np.isfinite(local[k]).all()
          for k in ("t", "flux", "error", "error_original", "bi"))
          and np.all(local["error"] > 0) and np.all(local["error_original"] > 0))
    check("derived_band_indices_valid", np.issubdtype(local["bi"].dtype, np.integer)
          and np.all((local["bi"] >= 0) & (local["bi"] < 19)))

    raw_arrays = {}
    for name in raw_filters:
        a = np.array([[float(r[k]) for k in ("MJD", "Flux", "Error")]
                      for r in optical if r["Filter"] == name])
        raw_arrays[name] = a[np.argsort(a[:, 0], kind="stable")]
    mjd = np.concatenate([raw_arrays[b][:, 0] for b in raw_filters])
    corrections = heliocentric_days(mjd)
    reports, matches, offset = [], [], 0
    for label, raw_band, facility, expected_public, expected_raw in BANDS:
        a = raw_arrays[raw_band]
        corr = corrections[offset:offset + len(a)]
        offset += len(a)
        h = a[:, 0] + 0.5 + corr  # HJD - 2400000, tested UTC convention.
        j = labels.index(label)
        valid = mask[j]
        public_columns = np.flatnonzero(valid)
        pt = public["dates"][j, valid]
        pf = public["flux"][j, valid] * 1e7
        pe = public["error"][j, valid] * 1e7
        check(f"{raw_band}:counts", (len(pt), len(a)) == (expected_public, expected_raw),
              {"public": len(pt), "raw": len(a)})
        check(f"{raw_band}:raw_valid", np.isfinite(a).all() and np.all(a[:, 1:] > 0))
        check(f"{raw_band}:time_keys", np.all(np.diff(pt) > 0)
              and np.all(np.diff(a[:, 0]) > 0))
        # Time-only matches, independent of the flux normalization being tested.
        # Swift: locate local MJD inside public. HST/ground: locate public HJD in raw.
        if facility == "Swift":
            ri = np.arange(len(a))
            pi = np.abs(pt[:, None] - a[None, :, 0]).argmin(axis=0)
            residual = pt[pi] - a[:, 0]
        else:
            pi = np.arange(len(pt))
            ri = np.abs(pt[:, None] - h[None, :]).argmin(axis=1)
            residual = pt - h[ri]
        accepted = np.abs(residual) * 86400 <= TIME_TOLERANCE_SECONDS
        check(f"{raw_band}:time_match_complete", accepted.all())
        pi, ri = pi[accepted], ri[accepted]
        check(f"{raw_band}:match_one_to_one", len(np.unique(pi)) == len(pi)
              and len(np.unique(ri)) == len(ri))
        public_extra = np.setdiff1d(np.arange(len(pt)), pi)
        local_missing = np.setdiff1d(np.arange(len(a)), ri)
        if not len(pi):
            raise ValueError(f"No supported time matches in {raw_band}")

        k = local_labels.index(label)
        di = np.flatnonzero(local["bi"] == k)
        di = di[np.argsort(local["t"][di], kind="stable")]
        if len(di) != len(a):
            raise ValueError(f"Derived local count differs in {raw_band}")
        dt = local["t"][di] - (a[:, 0] - 49999.5 + corr)
        df = local["flux"][di] - a[:, 1]
        de = local["error_original"][di] - a[:, 2]
        check(f"{raw_band}:derived_time", np.max(np.abs(dt)) * 86400 < 0.002)
        check(f"{raw_band}:derived_flux_and_original_error", np.max(np.abs(df))
              < FLUX_TOLERANCE and np.max(np.abs(de)) < FLUX_TOLERANCE)
        check(f"{raw_band}:derived_error_not_reduced", np.all(local["error"][di]
              >= local["error_original"][di]))
        variance_added = local["error"][di] ** 2 - a[:, 2] ** 2
        check(f"{raw_band}:derived_constant_added_variance", np.ptp(variance_added) < 1e-10)
        check(f"{raw_band}:wavelength_mapping", np.isclose(local["wavelengths"][k],
              public["wavelength"][j], rtol=0, atol=1e-8))
        info = {
            "band": label, "local_filter": raw_band, "facility": facility,
            "public_band_index": j, "wavelength_observed_A": float(public["wavelength"][j]),
            "raw_MJD": coverage(a[:, 0]), "public_supplied_dates": coverage(pt),
            "derived_local_THJD": coverage(local["t"][di]),
            "matched_n": len(pi), "public_unmatched_n": len(public_extra),
            "local_not_in_public_n": len(local_missing),
            "public_time_minus_raw_MJD_days": stats(pt[pi] - a[ri, 0]),
            "public_time_minus_reconstructed_HJD_days": stats(pt[pi] - h[ri]),
            "time_match_max_abs_residual_seconds": float(np.max(np.abs(residual[accepted])) * 86400),
            "matched_public_column_indices": public_columns[pi].tolist(),
            "matched_raw_sorted_indices": ri.tolist(),
            "public_unmatched_column_indices": public_columns[public_extra].tolist(),
            "public_unmatched_coverage": coverage(pt[public_extra]) if len(public_extra) > 1 else None,
            "matched_example": {
                "public_column_index": int(public_columns[pi[0]]),
                "raw_sorted_index": int(ri[0]),
                "raw_MJD": float(a[ri[0], 0]),
                "raw_flux_local_units": float(a[ri[0], 1]),
                "public_supplied_date": float(pt[pi[0]]),
                "public_flux_per_cm": float(pf[pi[0]] / 1e7),
                "public_flux_local_units": float(pf[pi[0]]),
                "reconstructed_HJD_minus_2400000": float(h[ri[0]]),
                "public_minus_reconstructed_HJD_days": float(pt[pi[0]] - h[ri[0]]),
            },
            "local_not_in_public": [dict(mjd=float(a[q, 0]), reconstructed_hjd_minus_2400000=float(h[q]),
                                         flux=float(a[q, 1])) for q in local_missing],
            "derived_check": {"time_max_abs_difference_seconds": float(np.max(np.abs(dt)) * 86400),
                              "flux_max_abs_difference": float(np.max(np.abs(df))),
                              "original_error_max_abs_difference": float(np.max(np.abs(de))),
                              "extra_error_floor_from_variance": float(np.sqrt(max(0., np.median(variance_added))))},
        }
        reports.append(info)
        matches.append((info, pf[pi], a[ri, 1], pe[pi], a[ri, 2]))

    ratios = np.concatenate([pf / rf for _, pf, rf, _, _ in matches])
    factor = float(np.median(ratios))  # Direct ratios only; no model or regression.
    check("matched_total", len(ratios) == 4188, len(ratios))
    check("common_factor", abs(factor - 1.00130680972) < 1e-12, factor)
    max_flux_residual = 0.
    for info, pf, rf, pe, re in matches:
        delta = float(np.max(np.abs(pf / factor - rf)))
        max_flux_residual = max(max_flux_residual, delta)
        info["public_to_local_flux_ratio_after_unit_conversion"] = stats(pf / rf)
        info["flux_max_abs_residual_after_factor_removal"] = delta
        info["error_ratio_after_factor_removal"] = stats(pe / factor / re)
        info["public_errors_smaller_than_raw_n"] = int(np.sum(pe / factor < re - 1e-10))
        check(f"{info['local_filter']}:uniform_flux_factor", delta < FLUX_TOLERANCE)
    swift = [r for r in reports if r["facility"] == "Swift"]
    other = [r for r in reports if r["facility"] != "Swift"]
    check("Swift_unchanged_MJD", all(r["public_time_minus_raw_MJD_days"]["min"] == 0
          and r["public_time_minus_raw_MJD_days"]["max"] == 0 for r in swift))
    check("HST_ground_heliocentric_match", all(r["time_match_max_abs_residual_seconds"]
          < TIME_TOLERANCE_SECONDS for r in other))
    unmatched = sum(r["public_unmatched_n"] for r in reports)
    omitted = sum(r["local_not_in_public_n"] for r in reports)
    check("public_additional_count", unmatched == 1712, unmatched)
    check("local_omitted_count", omitted == 15, omitted)
    swift_mixed_offset = {
        "min": min(r["public_time_minus_reconstructed_HJD_days"]["min"] for r in swift),
        "max": max(r["public_time_minus_reconstructed_HJD_days"]["max"] for r in swift),
    }
    reproduced = all(c["passed"] for c in checks)
    return {
        "schema_version": 1,
        "status": "audit_reproduced" if reproduced else "audit_checks_failed",
        "severity": "High", "public_primary_eligible": False,
        "primary_recommendation": "Use the local input as primary, conditional on its source checks passing; public bundle is diagnostic only.",
        "sources": {"raw": str(RAW), "public": str(PUBLIC), "derived_local": str(LOCAL)},
        "public_source_provenance": {
            "download_url": "https://raw.githubusercontent.com/Zstone19/TempMap/main/Examples/NGC5548_Data.npz",
            "downloaded_filename": "NGC5548_Data.npz",
            "saved_filename": "NGC5548_TempMap.npz",
            "relationship": "Same downloaded file, renamed locally; not a different bundle.",
            "provenance_basis": "Download and rename history supplied by the coordinating user; this audit uses the existing local file and performs no download.",
        },
        "scope": "Source-quality checks only; no fits, plots, downloads, source mutation, or independent validation of inversion physics.",
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "astropy": astropy.__version__,
                    "ephemeris": "builtin", "iers": "packaged IERS_B", "internet_enabled": False,
                    "np_load_allow_pickle": False},
        "method": {"grain": "one band measurement at one supplied epoch; padding excluded",
                   "matching": "Time-only nearest matches, one-to-one checked; raw MJD for Swift, reconstructed HJD for HST/ground. Flux ratios are independent confirmation.",
                   "time_tolerance_seconds": TIME_TOLERANCE_SECONDS,
                   "flux_absolute_tolerance_local_units": FLUX_TOLERANCE,
                   "reference_flux_factor": 1.00130680972,
                   "reference_flux_factor_absolute_tolerance": 1e-12,
                   "coordinates_icrs": "14h17m59.51s +25d08m12.5s",
                   "observer": "Earth geocenter", "input_time_scale": "UTC",
                   "time_conversion": "HJD_minus_2400000 = raw_MJD + 0.5 + heliocentric_light_travel_days; THJD = HJD_minus_2400000 - 50000",
                   "factor_estimator": "median(public_flux * 1e7 / matched_raw_flux), no fitting",
                   "derived_error_check": "Original errors preserved; corrected errors not reduced and added variance constant within each band. Triplet algorithm not rerun."},
        "units": {"public_flux_error": "CGS f_lambda per cm, consistent with tutorial and numerical source matching",
                  "local_flux_error": "1e-15 erg s^-1 cm^-2 Angstrom^-1",
                  "public_to_local_unit_multiplier": 1e7,
                  "public_to_physical_per_Angstrom_multiplier": 1e-8,
                  "common_flux_ratio": factor, "common_flux_ratio_range": stats(ratios),
                  "factor_origin": "Not established; retain for as-supplied public diagnostics.",
                  "max_flux_residual_after_factor_removal_local_units": max_flux_residual},
        "totals": {"raw_all": len(records), "raw_optical": len(optical), "raw_xray_excluded": len(excluded),
                   "public": int(mask.sum()), "derived_local": len(local["t"]),
                   "matched": len(ratios), "public_additional": unmatched, "local_omitted": omitted,
                   "facilities": {facility: {"public": sum(r["public_supplied_dates"]["n"] for r in reports if r["facility"] == facility),
                                             "local": sum(r["raw_MJD"]["n"] for r in reports if r["facility"] == facility)}
                                  for facility in ("Swift", "HST", "Ground")}},
        "findings": [
            {"id": "mixed_public_time_conventions", "severity": "High", "confidence": "High",
             "evidence": {"Swift_exact_MJD_matches": sum(r["matched_n"] for r in swift),
                          "HST_ground_HJD_matches": sum(r["matched_n"] for r in other),
                          "Swift_public_minus_corrected_HJD_days": swift_mixed_offset,
                          "HST_ground_max_time_residual_seconds": max(r["time_match_max_abs_residual_seconds"] for r in other),
                          "W2_example": swift[0]["matched_example"]},
             "impact": "A blanket HJD interpretation displaces the matched Swift observations relative to HST/ground by about half a day plus the heliocentric correction.",
             "cause": "Matched Swift timestamps were retained as raw MJD inside arrays otherwise containing HJD-minus-2400000 epochs.",
             "recommendation": "Public bundle must not be primary. Retain as-supplied chronology only for a clearly labeled diagnostic; use consistently converted local times for primary reconstruction."},
            {"id": "additional_Swift_exposures_unresolved", "severity": "High", "confidence": "High for counts; exposure identity unresolved",
             "evidence": {"public_Swift": sum(r["public_supplied_dates"]["n"] for r in swift),
                          "all_local_Swift_present": sum(r["matched_n"] for r in swift), "additional_Swift": unmatched},
             "impact": "The 1712 additional Swift records cannot be classified as duplicate or independent exposures without exposure-level matching.",
             "cause": "The bundle contains all 1571 local Swift measurements plus 1712 additional records; older-style numerical precision is suggestive, not proof of their exposure identity.",
             "recommendation": "Report the observed counts without assigning duplicate or independent exposure counts. Do not silently pool public and local records."},
            {"id": "public_normalization_and_errors", "severity": "Medium", "confidence": "High",
             "evidence": {"uniform_flux_ratio": factor, "matched_rows": len(ratios),
                          "public_errors_smaller_than_raw_n": sum(r["public_errors_smaller_than_raw_n"] for r in reports)},
             "impact": "Unit conversion alone does not equalize flux normalization or measurement weights; supplied errors cannot be modeled as a universal rescaling of local errors.",
             "recommendation": "Keep public normalization and supplied errors for public diagnostics; do not apply a second triplet correction."}],
        "limitations": [
            "No additional source release was downloaded or matched; independence and time convention of the 1712 extra Swift records remain unresolved.",
            "Geocentric UTC conversion matches the HST/ground bundle epochs; the original ingestion code and original UTC/TDB/topocentric choices are unavailable.",
            "This audit does not establish historical identity with every NK22 selection, wavelength choice, or error-processing step.",
            "Both derived and public wavelength arrays are compared to each other, not certified against NK22's differing textual wavelengths."],
        "bands": reports,
        "verification": {"all_checks_passed": reproduced, "check_count": len(checks), "checks": checks},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true", help="Print JSON; do not write any file")
    args = parser.parse_args()
    result = audit()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.stdout:
        print(payload, end="")
    else:
        OUTPUT.write_text(payload, encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("status", "severity", "public_primary_eligible", "totals")}, indent=2))
        print(f"Checks: {result['verification']['check_count']}; all passed: {result['verification']['all_checks_passed']}")
        print(f"Report: {OUTPUT}")
    return 0 if result["verification"]["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
