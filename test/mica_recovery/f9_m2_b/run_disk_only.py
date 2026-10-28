#!/usr/bin/env python3
"""Run the prepared M2/B disk-only control with the normal project scheduler."""
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CASE = HERE / "cases" / "disk_only"

if __name__ == "__main__":
    if not (CASE / "config/source_config.yaml").is_file():
        raise SystemExit("Disk-only inputs have not been prepared.")
    if (CASE / "runs").exists():
        raise SystemExit("Disk-only run outputs already exist; inspect them before restarting.")
    # Exec preserves ordinary terminal signal handling; this is not an MPI wrapper.
    os.execv(sys.executable, [sys.executable, "-u", str(ROOT / "src/run_mica.py"),
                            str(CASE), "--stage", "mica_recovery"])
