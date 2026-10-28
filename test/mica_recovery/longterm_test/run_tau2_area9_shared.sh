#!/usr/bin/env bash
set -euo pipefail
cd /home/leonhard/sci001
export PATH=/home/leonhard/miniforge3/envs/mica2/bin:$PATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/tmp
unset TMP TEMP
output_dir=/home/leonhard/sci001/test/mica_recovery/longterm_test/cases_tau2_area9_shared
[[ -d "$output_dir" ]]
exec >> "$output_dir/pipeline.log" 2>&1
trap 'pipeline_code=$?; printf "%s\n" "$pipeline_code" > "$output_dir/pipeline_exit_code"; printf "pipeline_exit_code=%s\n" "$pipeline_code"; date --iso-8601=seconds' EXIT
printf '%s\n' "$$" > "$output_dir/pipeline.pid"
date --iso-8601=seconds
/home/leonhard/miniforge3/envs/astro/bin/python -B -u test/mica_recovery/longterm_test/check_tau2_area9_shared_psd.py
python -B - <<'PY'
import json
from pathlib import Path
p=Path('test/mica_recovery/longterm_test/cases_tau2_area9_shared/diagnostics/psd_preflight/report.json')
r=json.loads(p.read_text())
assert r['status']=='pass' and not r['blocking_anomalies'], 'PSD preflight blocked fitting; inspect '+str(p)
print('PSD preflight passed: starting 12 two-component fits.')
PY
python -B -u test/mica_recovery/longterm_test/fit.py \
  --config test/mica_recovery/longterm_test/experiment_tau2_area9_shared.yaml \
  --suite paired_ablation --batch-size 12 --run --plot-results
printf 'pipeline_complete\n'
