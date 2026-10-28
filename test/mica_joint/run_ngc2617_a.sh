#!/usr/bin/env bash
set -euo pipefail
cd /home/leonhard/sci001
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python_bin=/home/leonhard/miniforge3/envs/astro/bin/python
date -Is
"$python_bin" -B src/run_mica_joint.py NGC2617_a --stage mica_gamma2_broad
"$python_bin" -B src/run_mica.py NGC2617_a --stage mica_gamma2_broad --fit-only
date -Is
