#!/usr/bin/env python3
"""Convert raw/<source>.csv to per-band prepared/<source>_<band>.txt for MICA."""
import argparse
from pathlib import Path

import pandas as pd


def prepare(source_dir: Path) -> None:
    source = source_dir.name
    raw_csv = source_dir / "raw" / f"{source}.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(f"missing {raw_csv}")

    df = pd.read_csv(raw_csv)
    prepared = source_dir / "prepared"
    prepared.mkdir(exist_ok=True)

    rows = []
    for band, grp in df.groupby("Filter"):
        fname = f"{source}_{band}.txt"
        path = prepared / fname
        grp[["MJD", "Flux", "Error"]].to_csv(
            path, sep=" ", index=False, header=False, float_format="%.6f"
        )
        rows.append({
            "Object": source,
            "Filter": band,
            "N": len(grp),
            "MJD_min": grp["MJD"].min(),
            "MJD_max": grp["MJD"].max(),
            "File": str(path),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(prepared / "summary.csv", index=False)
    print(f"prepared {len(rows)} bands → {prepared}")
    for _, r in summary.iterrows():
        print(f"  {r['Filter']:4s}  {r['N']:4d} pts  {r['MJD_min']:.3f} – {r['MJD_max']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw CSV to MICA prepared txt files.")
    parser.add_argument("source", type=Path, help="source directory, e.g. Mrk509")
    args = parser.parse_args()
    path = args.source.expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[1] / path).resolve()
    prepare(path)


if __name__ == "__main__":
    main()
