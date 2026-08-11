#!/usr/bin/env python3

from pathlib import Path
import argparse
import yaml
from mpi4py import MPI

from ffa.disk import prep, fit, summ
from ffa.ext import dered


def read_cfg(source_dir):
    path = source_dir / "config" / "source_config.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["fflux_analysis"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="source directory, e.g. Mrk817")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    source_dir = Path(args.source).expanduser().resolve()
    ff = read_cfg(source_dir)

    # 1. prepare
    # 如果不想跑这一步，注释掉这个 if 块
    if rank == 0:
        rows = prep(source_dir, ff)
        print("Prepared ffa inputs")
        for band, info in rows:
            print(
                f"{band:6s} "
                f"n_in={info['n_in']:5d} "
                f"n_out={info['n_out']:5d} "
                f"out={info['out']}"
            )
    comm.Barrier()

    # 2. Galactic dereddening
    # 如果不想跑这一步，注释掉这个 if 块
    if rank == 0:
        rows = dered(source_dir, ff)
        print("Dereddened ffa inputs")
        for band, info in rows:
            print(
                f"{band:6s} "
                f"A={info['A']:.6e} "
                f"out={info['out']}"
            )
    comm.Barrier()

    # 3. ffa fit
    # 如果不想跑这一步，注释掉下面这几行
    info = fit(source_dir, ff)
    if info["rank"] == 0:
        print("Finished ffa fit")
        print("bands:", ", ".join(info["bands"]))
        print("sample_dir:", info["sample_dir"])
    comm.Barrier()

    # 4. summary
    # 如果不想跑这一步，注释掉这个 if 块
    if rank == 0:
        out_csv, df = summ(source_dir, ff)
        print(f"Wrote {out_csv}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()