#!/usr/bin/env bash
set -euo pipefail
cd /home/leonhard/sci001
export PATH=/home/leonhard/miniforge3/envs/mica2/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/tmp
unset TMP TEMP
output_dir=/home/leonhard/sci001/test/mica_recovery/longterm_test/cases_fast_driver
exec >> "$output_dir/pipeline.log" 2>&1
trap 'pipeline_code=$?; printf "%s\n" "$pipeline_code" > "$output_dir/pipeline_exit_code"; printf "pipeline_exit_code=%s\n" "$pipeline_code"; date --iso-8601=seconds' EXIT
printf '%s\n' "$$" > "$output_dir/pipeline.pid"
date --iso-8601=seconds
python -B -u test/mica_recovery/longterm_test/fit.py \
  --config test/mica_recovery/longterm_test/experiment_uonly_gp_fast_driver.yaml \
  --suite paired_ablation --batch-size 10 --run --plot-results
printf 'pipeline_complete\n'
