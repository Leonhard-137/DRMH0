#!/usr/bin/env python3
"""
Run MICA for one response band.

Usage (from test/s_vartest)::

    mpiexec -n 2 python run_mica.py NGC4593 B

The script reads::

    SOURCE/config/source_config.yaml
    SOURCE/fflux/host_sub/SOURCE_W2.txt
    SOURCE/fflux/host_sub/SOURCE_<BAND>.txt

out:

    SOURCE/runs/mica/gamma2/2comp/

Run each response band separately.  ``num_particles`` and
``max_num_saves`` come from the source config.  Files in ``raw`` are not
used by this script.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pymica
import yaml
from mpi4py import MPI


# ============================================================
# 1. Command line
# ============================================================

if len(sys.argv) != 3:
    print(__doc__)
    raise SystemExit(1)

source_dir = Path(sys.argv[1]).resolve()
band = sys.argv[2]


# ============================================================
# 2. Read source_config.yaml
# ============================================================

config_file = source_dir / "config" / "source_config.yaml"

with config_file.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

source = config["source"]
mica = config

driver = mica["driver"]
responses = mica["responses"]

if band not in responses:
    raise ValueError(
        f"Unknown response band {band!r}. "
        f"Available bands: {responses}"
    )


# ============================================================
# 3. Locate the two light curves
# ============================================================

input_dir = source_dir / mica["input_dir"]

driver_file = input_dir / f"{source}_{driver}.txt"
response_file = input_dir / f"{source}_{band}.txt"

if not driver_file.exists():
    raise FileNotFoundError(driver_file)

if not response_file.exists():
    raise FileNotFoundError(response_file)


# ============================================================
# 4. Load data
# ============================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    continuum = np.loadtxt(driver_file)
    response = np.loadtxt(response_file)

    data = {
        "set1": [continuum, response]
    }
else:
    data = None

# Give the same data to all MPI processes
data = comm.bcast(data, root=0)


# ============================================================
# 5. Create run directory
# ============================================================

ncomp = int(mica["ncomp"])
type_tf = mica["type_tf"]

run_name = f"run_{driver}_to_{band}_{ncomp}comp_{type_tf}"

run_dir = (
    source_dir
    / mica["output_root"]
    / f"{ncomp}comp"
    / run_name
)

if rank == 0:
    run_dir.mkdir(parents=True, exist_ok=True)

comm.Barrier()

# MICA writes its data/, param/, plots, etc. in the current directory
os.chdir(run_dir)


# ============================================================
# 6. Set up MICA
# ============================================================

model = pymica.gmodel()

model.setup(
    data=data,
    type_tf=type_tf,
    # Match the original runner's per-light-curve component specification.
    number_component=[ncomp, ncomp],
    lag_limit=mica["lag_limit"],
    type_lag_prior=mica["type_lag_prior"],
    flag_lag_posivity=mica["flag_lag_posivity"],
    flag_trend=mica["flag_trend"],
    max_num_saves=mica["max_num_saves"],
    **{
        key: mica[key]
        for key in (
            "lag_prior",
            "width_limit",
            "width_prior",
            "flag_negative_resp",
            "flag_con_sys_err",
            "flag_line_sys_err",
            "flag_gap",
            "gap_prior",
            "ptol",
            "num_particles",
            "thread_steps_factor",
            "new_level_interval_factor",
            "save_interval_factor",
            "lam",
            "beta",
            "max_num_levels",
            "nd_rec",
            "trec_ext",
        )
        if key in mica and mica[key] is not None
    },
)


# ============================================================
# 7. Run MICA
# ============================================================

if rank == 0:
    print()
    print("=" * 60)
    print(f"Source : {source}")
    print(f"Fit    : {driver} -> {band}")
    print(f"TF     : {ncomp}-component {type_tf}")
    print(f"Input  : {input_dir}")
    print(f"Output : {run_dir}")
    print("=" * 60)
    print()

model.run()

comm.Barrier()


# ============================================================
# 8. Decompose the response
# ============================================================

model.decompose()

comm.Barrier()


# ============================================================
# 9. Plot and post-process
# ============================================================

if rank == 0:
    model.plot_results(
        doshow=False,
        tf_lag_range=mica["lag_limit"],
        hist_lag_range=mica["lag_limit"],
    )

    model.plot_decomp(doshow=False)

    model.post_process()

comm.Barrier()


if rank == 0:
    print()
    print("Done.")
    print(f"Results: {run_dir}")
