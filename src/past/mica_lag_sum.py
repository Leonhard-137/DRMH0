#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


OUT_COLS = [
    "source",
    "drive_band",
    "resp_band",
    "component",
    "tf_index",
    "lambda_obs",
    "tau",
    "tau_err_low",
    "tau_err_high",
]


def read_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def rel(base, p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else base / p


def arr(p):
    x = np.loadtxt(p, comments="#")
    return x.reshape(1, -1) if x.ndim == 1 else x


def q68(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan

    p16, p50, p84 = np.percentile(x, [15.865, 50.0, 84.135])
    return float(p50), float(p50 - p16), float(p84 - p50)


def run_info(name):
    m = re.match(r"^run_(.+?)_to_(.+?)_(\d+)comp(?:_(.+))?$", name)
    if not m:
        return None

    drive, resp, ncomp, tf = m.groups()
    return drive, resp, int(ncomp), tf


def latest_sample(data_dir):
    files = list(data_dir.glob("posterior_sample1d.txt_*"))
    if not files:
        raise FileNotFoundError(f"missing posterior_sample1d.txt_* in {data_dir}")

    def ncomp(p):
        return int(p.name.rsplit("_", 1)[-1])

    p = sorted(files, key=ncomp)[-1]
    return p, ncomp(p)


def names(data_dir, ncomp):
    p = data_dir / f"para_names_line.txt_{ncomp}"
    if not p.exists():
        raise FileNotFoundError(p)

    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        parts = s.split()
        if len(parts) >= 2 and re.fullmatch(r"[-+]?\d+", parts[0]):
            out.append(parts[1])
        else:
            out.append(parts[-1])

    if not out:
        raise ValueError(f"no parameter names in {p}")

    return out


def pick_cols(ns, ncomp, key):
    cols = [i for i, x in enumerate(ns) if f"component_{key}" in x]
    if len(cols) < ncomp:
        cols = [i for i, x in enumerate(ns) if "component" in x and key in x]
    if len(cols) < ncomp:
        raise ValueError(f"cannot find {ncomp} component {key} columns")
    return cols[:ncomp]


def comp_cols(ns, ncomp):
    amp = pick_cols(ns, ncomp, "amplitude")
    cen = pick_cols(ns, ncomp, "center")
    sig = pick_cols(ns, ncomp, "sigma")
    return list(zip(amp, cen, sig))


def tf_kind(cfg, run_tf=None):
    opt = cfg.get("mica_lag_summary", {})
    x = opt.get("type_tf") or run_tf or cfg.get("mica_fit", {}).get("type_tf", "gaussian")
    x = str(x).strip().lower()

    aliases = {
        "gauss": "gaussian",
        "g": "gaussian",
        "top-hat": "tophat",
        "top_hat": "tophat",
        "hat": "tophat",
        "exponential": "exp",
    }
    return aliases.get(x, x)


def tau_comp(sample, ccen, csig, kind):
    cen = sample[:, ccen]
    wid = np.exp(sample[:, csig])

    if kind in {"gaussian", "tophat"}:
        return cen
    if kind == "gamma":
        return cen + 2.0 * wid
    if kind == "exp":
        return cen + wid

    raise ValueError(f"unknown type_tf={kind!r}; use gaussian, tophat, gamma, or exp")


def tau_total(sample, cols, kind):
    taus = []
    wts = []

    for camp, ccen, csig in cols:
        taus.append(tau_comp(sample, ccen, csig, kind))
        wts.append(np.exp(sample[:, camp]))

    taus = np.vstack(taus)
    wts = np.vstack(wts)
    den = np.sum(wts, axis=0)

    return np.sum(wts * taus, axis=0) / den


def time_scale(cfg):
    if not cfg.get("use_rest_frame", False):
        return 1.0

    z = cfg.get("redshift")
    if z is None:
        raise ValueError("use_rest_frame is true but redshift is null")

    return 1.0 / (1.0 + float(z))


def add_row(rows, source, drive, resp, lam, comp, idx, tau_samp, scale):
    tau, elo, ehi = q68(tau_samp)
    rows.append(
        {
            "source": source,
            "drive_band": drive,
            "resp_band": resp,
            "component": comp,
            "tf_index": idx,
            "lambda_obs": lam,
            "tau": tau * scale,
            "tau_err_low": elo * scale,
            "tau_err_high": ehi * scale,
        }
    )


def rows_one(source_dir, cfg, run_dir):
    info = run_info(run_dir.name)
    if info is None:
        return []

    drive, resp, ncomp_from_name, run_tf = info

    if resp not in cfg["bands"]:
        raise KeyError(f"{resp} not found in top-level bands")

    data_dir = run_dir / "data"
    sample_path, ncomp = latest_sample(data_dir)

    if ncomp != ncomp_from_name:
        raise ValueError(
            f"ncomp mismatch in {run_dir.name}: name={ncomp_from_name}, file={ncomp}"
        )

    sample = arr(sample_path)
    ns = names(data_dir, ncomp)
    cols = comp_cols(ns, ncomp)

    kind = tf_kind(cfg, run_tf)
    scale = time_scale(cfg)
    source = cfg.get("source", source_dir.name)
    lam = float(cfg["bands"][resp])

    rows = []
    for k, (_, ccen, csig) in enumerate(cols):
        add_row(
            rows,
            source,
            drive,
            resp,
            lam,
            f"tf{k}",
            k,
            tau_comp(sample, ccen, csig, kind),
            scale,
        )

    opt = cfg.get("mica_lag_summary", {})
    if opt.get("include_total", True):
        add_row(
            rows,
            source,
            drive,
            resp,
            lam,
            "total",
            ncomp,
            tau_total(sample, cols, kind),
            scale,
        )

    return rows


def build(source_dir, cfg):
    opt = cfg.get("mica_lag_summary", {})
    in_dir = rel(source_dir, opt.get("input_dir", "runs/gaussian/2comp"))

    rows = []
    for run_dir in sorted(p for p in in_dir.iterdir() if p.is_dir()):
        rows.extend(rows_one(source_dir, cfg, run_dir))

    if not rows:
        raise RuntimeError(f"no lag rows found in {in_dir}")

    return (
        pd.DataFrame(rows)[OUT_COLS]
        .sort_values(["lambda_obs", "tf_index"])
        .reset_index(drop=True)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_dir", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    cfg_path = args.config or source_dir / "config/source_config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = source_dir / cfg_path

    cfg = read_yaml(cfg_path)
    df = build(source_dir, cfg)

    out = args.output or Path(
        cfg.get("mica_lag_summary", {}).get(
            "output",
            "results/mica_tf_lag_summary.csv",
        )
    )
    out = rel(source_dir, out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out, index=False, float_format="%.6f", na_rep="")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
