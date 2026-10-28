#!/usr/bin/env python3
"""Prepare source-verified STORM 2 light curves for temperature-map inversion.

All output times are observer-frame THJD = HJD - 2450000.
All output fluxes/errors are f_lambda in 1e-15 erg s^-1 cm^-2 Angstrom^-1.
Run with the astro environment. Source inputs are never overwritten.
"""
from __future__ import annotations
import csv
import json
import re
import urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference"
DERIVED = ROOT / "derived"
INPUT = ROOT / "Mrk817_data"
SWIFT_URL = "https://content.cld.iop.org/journals/0004-637X/958/2/195/revision1/apjacfdact1_mrt.txt"
HST_URL = "https://content.cld.iop.org/journals/0004-637X/948/2/85/revision1/apjacc45at4_mrt.txt"
SOURCES = {
    "Swift": {"url": SWIFT_URL, "doi": "10.3847/1538-4357/acfdac", "table": 1},
    "HST": {"url": HST_URL, "doi": "10.3847/1538-4357/acc45a", "table": 4,
            "mast_doi": "10.17909/n734-k698"},
    "Ground": {"doi": "10.3847/1538-4357/ae6103", "table": 3,
               "wavelength_table": 4, "paper": "https://arxiv.org/html/2605.02875"},
    "mapping_method": {"paper": "https://arxiv.org/html/2310.01497", "section": "2.1"}
}
WAVELENGTHS = {
    "HST_1180": 1180.0, "HST_1398": 1398.0, "HST_1502": 1502.0,
    "HST_1739": 1739.0,
    "Swift_UVW2": 1928.0, "Swift_UVM2": 2246.0, "Swift_UVW1": 2600.0,
    "Swift_U": 3465.0, "Swift_B": 4392.0, "Swift_V": 5468.0,
    "Ground_u": 3540.0, "Ground_B": 4361.0, "Ground_g": 4770.0,
    "Ground_V": 5448.0, "Ground_r": 6215.0, "Ground_i": 7545.0,
    "Ground_z": 8700.0,
}
FIELDS = ["band", "instrument", "wavelength_obs_A", "thjd", "flux",
          "error", "error_original", "telescope"]


def source_table(url: str) -> str:
    path = REFERENCE / url.rsplit("/", 1)[-1]
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "STORM2-temperature-reproduction/1.0"})
        payload = urllib.request.urlopen(request, timeout=60).read()
        if not payload.startswith(b"Title:"):
            raise ValueError(f"Unexpected source table response from {url}")
        path.write_bytes(payload)
    text = path.read_text(encoding="utf-8")
    if "Byte-by-byte Description" not in text:
        raise ValueError(f"Missing source metadata in {path}")
    return text


def numeric_rows(text: str, regex: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines() if re.match(regex, line)]


def triplet_floor(time: np.ndarray, flux: np.ndarray, error: np.ndarray):
    """Median nonnegative extra 1-sigma error from consecutive triplets.

    A weighted straight-line fit has 3 - 2 = 1 residual degree of freedom.
    For a vector a orthogonal to [1, t], its minimized chi-square is
      (a dot f)^2 / sum[a_i^2 * (error_i^2 + floor^2)].
    Thus solving chi-square = 1 has the analytic expression below.
    This refits the line as the trial variance changes; no unweighted
    residual approximation or multiplicative scale fit is used.
    """
    floors = []
    original_chi2 = []
    for index in range(len(time) - 2):
        t, y, e = time[index:index+3], flux[index:index+3], error[index:index+3]
        span = t[2] - t[0]
        if span <= 0:
            continue
        a = np.array([t[1]-t[2], t[2]-t[0], t[0]-t[1]]) / span
        norm2 = float(a @ a)
        contrast2 = float(a @ y) ** 2
        variance = float((a*a) @ (e*e))
        floors.append(np.sqrt(max(0.0, (contrast2 - variance) / norm2)))
        original_chi2.append(contrast2 / variance)
    if not floors:
        raise ValueError("No valid adjacent triplets")
    return float(np.median(floors)), len(floors), float(np.median(original_chi2))


def verified_local_band(local, band: str, source: np.ndarray, flux_tol: float):
    selected = sorted([r for r in local if r["Filter"] == band], key=lambda r: float(r["MJD"]))
    source = source[np.argsort(source[:, 0], kind="stable")]
    if len(selected) != len(source):
        raise ValueError(f"{band}: local/source counts differ")
    local_array = np.array([[float(r["MJD"]), float(r["Flux"]), float(r["Error"])] for r in selected])
    delta_flux = local_array[:, 1] - source[:, 1]
    delta_error = local_array[:, 2] - source[:, 2]
    # Source col 0 is normalized here to HJD - 2400000.
    delta_time = local_array[:, 0] + 0.5 - source[:, 0]
    if np.max(np.abs(delta_flux)) > flux_tol or np.max(np.abs(delta_error)) > 1e-8:
        raise ValueError(f"{band}: local flux/error do not match the published sequence")
    if np.max(np.abs(delta_time)) > 0.02:
        raise ValueError(f"{band}: candidate match has unexpectedly different times")
    audit = {
        "n": len(selected),
        "max_abs_flux_difference": float(np.max(np.abs(delta_flux))),
        "max_abs_error_difference": float(np.max(np.abs(delta_error))),
        "max_abs_time_difference_days": float(np.max(np.abs(delta_time))),
        "median_time_difference_days": float(np.median(delta_time)),
        "time_difference_definition": "local labelled MJD + 0.5 - published (HJD - 2400000)",
        "adopted_time": "published HJD, matched by sorted epoch and validated flux/error",
        "preserved_flux_error": "local CSV values",
    }
    return local_array, source[:, 0] - 50000.0, audit


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    REFERENCE.mkdir(exist_ok=True)
    DERIVED.mkdir(exist_ok=True)
    with (INPUT / "HST_Swift" / "Mrk817_HST_Swift.csv").open(newline="") as handle:
        local = list(csv.DictReader(handle))
    swift = numeric_rows(source_table(SWIFT_URL), r"^\d{4}\.\d{3}\s")
    hst = np.asarray(numeric_rows(source_table(HST_URL), r"^\d{5}\.\d{4}\s"), dtype=float)
    if len(local) != 2577 or hst.shape != (165, 9):
        raise ValueError("Unexpected HST/Swift dataset shape")
    soft = {r[0]: (float(r[2]), float(r[3])) for r in swift if r[1] == "S"}
    hard = {r[0]: (float(r[2]), float(r[3])) for r in swift if r[1] == "H"}
    if len(soft) != 414 or len(hard) != 414 or set(soft) != set(hard):
        raise ValueError("XRT soft/hard snapshots do not match exactly")
    xrt = [dict(thjd=float(t), rate=soft[t][0]+hard[t][0],
                error=float(np.hypot(soft[t][1], hard[t][1])))
           for t in sorted(soft, key=float)]
    write_csv(DERIVED / "xrt.csv", xrt, ["thjd", "rate", "error"])
    records = []
    audit = {}

    def append(instrument, band, times, flux, error, telescopes):
        full_band = f"{instrument}_{band}"
        for t, f, e, telescope in zip(times, flux, error, telescopes):
            records.append(dict(band=full_band, instrument=instrument,
                wavelength_obs_A=WAVELENGTHS[full_band], thjd=float(t),
                flux=float(f), error=float(e), error_original=float(e), telescope=telescope))

    for index, band in enumerate(["1180", "1398", "1502", "1739"]):
        source = hst[:, [0, 1 + 2*index, 2 + 2*index]]
        data, times, audit["HST_" + band] = verified_local_band(local, band + "A", source, 5e-6)
        append("HST", band, times, data[:, 1], data[:, 2], ["HST/COS"] * len(times))

    for band in ["UVW2", "UVM2", "UVW1", "U", "B", "V"]:
        source = np.array([[float(r[0]) + 50000, float(r[2]), float(r[3])]
                           for r in swift if r[1] == band])
        data, times, audit["Swift_" + band] = verified_local_band(local, band, source, 1e-8)
        append("Swift", band, times, data[:, 1], data[:, 2], ["Swift/UVOT"] * len(times))

    ground_file = INPUT / "ground_2026" / "apjae6103t3_mrt.txt"
    ground_text = ground_file.read_text(encoding="utf-8")
    if "mid-point; -2400000" not in ground_text or "10^-14" not in ground_text:
        raise ValueError("Ground data header units differ from audited source")
    ground = numeric_rows(ground_text, r"^[uBgVriz]\s+\d{5}\.\d{3}\s")
    if len(ground) != 4874:
        raise ValueError(f"Expected 4874 ground points, got {len(ground)}")
    for band in ["u", "B", "g", "V", "r", "i", "z"]:
        selected = [r for r in ground if r[0] == band]
        append("Ground", band, [float(r[1])-50000 for r in selected],
               [10*float(r[2]) for r in selected], [10*float(r[3]) for r in selected],
               [r[4] for r in selected])

    records.sort(key=lambda r: (r["wavelength_obs_A"], r["band"], r["thjd"], r["telescope"]))
    summaries = []
    for band in sorted(WAVELENGTHS, key=WAVELENGTHS.get):
        selected = [r for r in records if r["band"] == band]
        time, flux, error = np.array([[r["thjd"], r["flux"], r["error_original"]]
                                     for r in selected]).T
        if not np.isfinite(np.column_stack([time, flux, error])).all() or np.any(error <= 0):
            raise ValueError(f"{band}: invalid data values")
        floor, triplets, median_chi2 = triplet_floor(time, flux, error)
        expanded = np.sqrt(error**2 + floor**2)
        for r, e in zip(selected, expanded):
            r["error"] = float(e)
        unique_times = np.unique(time)
        summary = dict(band=band, instrument=selected[0]["instrument"],
            wavelength_obs_A=WAVELENGTHS[band], n=len(selected),
            t_min=float(time.min()), t_max=float(time.max()), mean_flux=float(flux.mean()),
            rms_flux=float(flux.std(ddof=1)), median_error_original=float(np.median(error)),
            systematic_floor=floor, median_error=float(np.median(expanded)),
            median_error_ratio=float(np.median(expanded/error)),
            median_cadence=float(np.median(np.diff(unique_times))), triplets=triplets,
            median_triplet_chi2_original=median_chi2,
            duplicate_times=len(time)-len(unique_times),
            telescope_count=len(set(r["telescope"] for r in selected)))
        summaries.append(summary)

    keys = [(r["band"], r["thjd"], r["telescope"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate band/time/telescope records")
    if len(records) != 7451 or len(summaries) != 17:
        raise ValueError("Incorrect final dataset size")
    write_csv(DERIVED / "lightcurves.csv", records, FIELDS)
    original = [dict(r, error=r["error_original"]) for r in records]
    write_csv(DERIVED / "lightcurves_original_errors.csv", original, FIELDS)
    write_csv(DERIVED / "band_summary.csv", summaries, list(summaries[0]))
    metadata = {
        "sources": SOURCES, "row_count": len(records), "band_count": len(summaries),
        "time_system": "observer-frame THJD = HJD - 2450000",
        "flux_unit": "1e-15 erg s^-1 cm^-2 Angstrom^-1",
        "xrt": {
            "source": "Cackett et al. 2023 Table 1, same download as Swift UVOT",
            "n": len(xrt), "energy_keV": [0.3, 10.0], "unit": "count s^-1",
            "time_system": "observer-frame THJD = HJD - 2450000",
            "matching": "Exact equality of published THJD strings, tolerance 0 days; all 414 S/H snapshots match uniquely",
            "rate": "S(0.3-1.5 keV) + H(1.5-10 keV)",
            "error": "sqrt(error_S^2 + error_H^2); assumes errors in disjoint energy bands are independent, no covariance published",
            "t_min": min(r["thjd"] for r in xrt), "t_max": max(r["thjd"] for r in xrt),
        },
        "source_matching": audit,
        "ground_time_transform": "published HJD_minus_2400000 - 50000",
        "ground_flux_error_transform": "published table value multiplied by 10",
        "wavelengths_observed_angstrom": WAVELENGTHS,
        "wavelength_provenance": "HST/Swift: Cackett et al. 2023 Table 2; Ground: Montano et al. 2026 Table 4",
        "wavelength_difference_from_2024": "2024 mapping states reddest band 8897 A but does not tabulate all ground central wavelengths; the public 2026 Table 4 gives z=8700 A, adopted here",
        "triplet_method": "Consecutive sorted triplets, weighted straight-line fit with 1 residual degree of freedom; each triplet's nonnegative additive floor solves chi2=1; median floor per band added in quadrature",
        "triplet_gap_policy": "All consecutive triplets retained, including gaps; no gap rejection was specified in mapping paper Sec 2.1",
        "triplet_error_policy": "Recomputed from the supplied final-publication light curves; no hard-coded 3.3-3.5 multiplier",
        "flux_processing": "No extinction correction, host subtraction, detrending, or cross-instrument flux rescaling applied; those are separate modelling operations",
        "limitations": [
            "Final public ground release has 4874 points; total 7451 differs from the 2024 mapping paper's 7380.",
            "Local time conversion history is not established; published HJD times replace local labelled MJD after validated one-to-one matching.",
            "The published HST/Swift times have 1e-4/1e-3 day precision.",
            "The mapping paper's HST 1709 A reference is interpreted as the 1739 A band, consistent with all figures and the source data.",
            "Overlapping ground timestamps from different telescopes remain distinct independent measurements."
        ],
    }
    (DERIVED / "data_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(f"Prepared {len(records)} rows in {len(summaries)} bands")
    for s in summaries:
        print(f"{s['band']:12s} n={s['n']:4d} floor={s['systematic_floor']:.6g} median_error_ratio={s['median_error_ratio']:.6g}")


if __name__ == "__main__":
    main()
