#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ffa import sed


def src(p):
    return Path(p).expanduser().resolve()


def val(df, col, default=""):
    if col not in df.columns or len(df) == 0:
        return default
    x = df[col].iloc[0]
    return default if pd.isna(x) else x


def max_x0(df):
    if "X0" not in df.columns or "Filter" not in df.columns:
        return "", np.nan

    x = pd.to_numeric(df["X0"], errors="coerce")
    if not np.isfinite(x).any():
        return "", np.nan

    i = x.idxmax()
    return str(df.loc[i, "Filter"]), float(x.loc[i])


def write_sum(out, df):
    band, x0 = max_x0(df)

    bands = ", ".join(df["Filter"].astype(str)) if "Filter" in df.columns else ""
    xgal = val(df, "X_gal", np.nan)
    xfaint = val(df, "X_faint", np.nan)
    xbright = val(df, "X_bright", np.nan)
    gal_ref = val(df, "gal_ref", "")
    units = val(df, "units", "")

    path = out.with_name(out.stem + "_summary.txt")

    lines = [
        "FVG component extraction",
        f"output: {out}",
        f"n_band: {len(df)}",
        f"bands: {bands}",
        f"gal_ref: {gal_ref}",
        f"max_x0_band: {band}",
        f"max_x0: {x0:.10e}" if np.isfinite(x0) else "max_x0: nan",
        f"X_gal: {float(xgal):.10e}" if np.isfinite(float(xgal)) else "X_gal: nan",
        f"X_faint: {float(xfaint):.10e}" if np.isfinite(float(xfaint)) else "X_faint: nan",
        f"X_bright: {float(xbright):.10e}" if np.isfinite(float(xbright)) else "X_bright: nan",
        f"units: {units}",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(source_dir):
    source_dir = src(source_dir)

    out, df = sed.comp(str(source_dir))
    out = Path(out).expanduser().resolve()

    sum_path = write_sum(out, df)

    print(f"wrote: {out}")
    print(f"summary: {sum_path}")

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.to_string(index=False))

    return out, df


def main():
    ap = argparse.ArgumentParser(
        description="Extract FVG components from FFA output."
    )
    ap.add_argument("source")
    args = ap.parse_args()

    run(args.source)


if __name__ == "__main__":
    main()