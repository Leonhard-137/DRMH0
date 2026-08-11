#!/usr/bin/env python3
"""Summarize all MICA posterior parameters to a single CSV.

Reads posterior_sample1d.txt_N + para_names_line.txt_N from every run
under RUN_ROOT, computes 16th/50th/84th percentiles, exponentiates
log-space parameters back to linear, and writes the result.

Usage:
  python src/mica_post_summary.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Set these two variables before running
# ---------------------------------------------------------------------------
SOURCE_DIR = "Mrk817"
RUN_ROOT = "runs/gaussian0_100_yes"  # relative to SOURCE_DIR

# ---------------------------------------------------------------------------
# parameter descriptions: key → (physical_meaning, unit, is_log10)
# ---------------------------------------------------------------------------
PARAM_INFO = {
    "sys_err_con":     ("Continuum systematic error",     "mJy",          False),
    "sys_err_line":    ("Line systematic error",           "mJy",          False),
    "sigmad":          ("DRW amplitude σ_d",               "days⁻¹ᐟ²",     True),
    "taud":            ("DRW damping timescale τ_d",       "days",         True),
    "component_amplitude": ("TF component amplitude",      "",             True),
    "component_center":    ("TF component center / lag",   "days",         False),
    "component_sigma":     ("TF component width σ",        "days",         True),
}

OUT_COLS = [
    "source", "run_name", "driver_band", "resp_band", "ncomp", "type_tf",
    "param_index", "param_name", "physical_meaning", "unit",
    "prior", "prior_min", "prior_max",
    "median", "p16", "p84",
]

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
def _q68(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    p16, p50, p84 = np.percentile(x, [15.865, 50.0, 84.135])
    return float(p50), float(p16), float(p84)


def _classify(name):
    if name in PARAM_INFO:
        return PARAM_INFO[name]
    for key, info in PARAM_INFO.items():
        if key in name:
            return info
    return (name, "", False)


def _process_run(run_dir, source_name):
    m = re.match(r"^run_(.+?)_to_(.+?)_(\d+)comp(?:_(.+))?$", run_dir.name)
    if not m:
        print(f"  SKIP: cannot parse '{run_dir.name}'")
        return []
    driver, resp, ncomp, tf = m.groups()
    ncomp, tf = int(ncomp), (tf or "gaussian")

    data_dir = run_dir / "data"
    posts = sorted(data_dir.glob("posterior_sample1d.txt_*"))
    if not posts:
        print(f"  WARNING: no posterior in {data_dir}")
        return []
    post_ncomp = int(posts[-1].name.rsplit("_", 1)[-1])
    names_path = data_dir / f"para_names_line.txt_{post_ncomp}"
    if not names_path.exists():
        print(f"  WARNING: {names_path} not found")
        return []

    sample = np.loadtxt(posts[-1], comments="#")
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)

    # parse para_names_line
    params = []
    for line in names_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        params.append((int(parts[0]), parts[1], parts[2], float(parts[3]), float(parts[4])))

    if sample.shape[1] != len(params):
        print(f"  WARNING: column mismatch in {data_dir}: {sample.shape[1]} vs {len(params)}")
        return []

    rows = []
    for idx, name, prior, pmin, pmax in params:
        if idx >= sample.shape[1]:
            continue
        median, p16, p84 = _q68(sample[:, idx])
        meaning, unit, is_log = _classify(name)
        if is_log:
            median, p16, p84 = 10.0**median, 10.0**p16, 10.0**p84
        rows.append({
            "source": source_name, "run_name": run_dir.name,
            "driver_band": driver, "resp_band": resp,
            "ncomp": ncomp, "type_tf": tf,
            "param_index": idx, "param_name": name,
            "physical_meaning": meaning, "unit": unit,
            "prior": prior, "prior_min": pmin, "prior_max": pmax,
            "median": median, "p16": p16, "p84": p84,
        })
    return rows


def main():
    src = ROOT / SOURCE_DIR
    run_root = src / RUN_ROOT
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    cfg = yaml.safe_load((src / "config" / "source_config.yaml").read_text(encoding="utf-8"))
    source_name = cfg.get("source", src.name)

    run_dirs = sorted(d for d in run_root.glob("*/run_*") if d.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run directories under {run_root}")
    print(f"source:   {source_name}")
    print(f"run root: {run_root}")
    print(f"run dirs: {len(run_dirs)}\n")

    all_rows = []
    for rd in run_dirs:
        print(f"  {rd.relative_to(src)}")
        all_rows.extend(_process_run(rd, source_name))

    if not all_rows:
        raise RuntimeError("No posterior data found.")

    # output: results/<name>/mica_post_param/mica_post_summary.csv
    parts = [p for p in RUN_ROOT.split("/") if p and p != "runs"]
    out_dir = src / "results" / Path(*parts) / "mica_post_param"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mica_post_summary.csv"

    pd.DataFrame(all_rows)[OUT_COLS].to_csv(out_path, index=False, float_format="%.6g")
    print(f"\nSaved {len(all_rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
