#!/usr/bin/env bash
set -euo pipefail
cd /home/leonhard/sci001
export PATH=/home/leonhard/miniforge3/envs/mica2/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/tmp
unset TMP TEMP
for model in gamma gaussian; do
  python -B -u test/mica_recovery/longterm_test/fit.py --config test/mica_recovery/longterm_test/experiment_uonly_gp_highpass.yaml --suite paired_ablation --model "$model" --batch-size 2 --run --plot-results
done
printf 'pipeline_complete\n'
