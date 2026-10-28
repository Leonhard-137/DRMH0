#!/usr/bin/env python3
"""Launch one prepared job only when the older scheduler has no queued work.

This permits the next frozen experiment to use an otherwise idle slot while
the previous experiment's final jobs finish. It never resumes or overwrites a
directory. Use the ordinary bounded scheduler after the older one completes.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan', type=Path, required=True)
parser.add_argument('--job-index', type=int, required=True)
args = parser.parse_args()
source = Path(__file__).with_name('run_stability_suite.py')
spec = importlib.util.spec_from_file_location('bounded_scheduler', source)
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)
plan = json.loads(args.plan.read_text())
registry = scheduler.read_json(scheduler.COORDINATION / 'registry.json')
older = scheduler.read_json(registry['manifest'])
if any(j['status'] in ('planned', 'starting', 'waiting_external') for j in older['jobs']):
    raise RuntimeError('The older scheduler still has queued work; no launch')
active = scheduler.active_fits()
if len(active) >= 4:
    raise RuntimeError('All four fit slots are occupied; no launch')
job = dict(plan['jobs'][args.job_index])
folder = Path(job['output'])
if folder.exists():
    raise FileExistsError(folder)
snapshot, priors = scheduler.snapshot_inputs(Path(plan['inputs_root']), [(job['case'], job['mode'])])
scheduler.verify_snapshot(snapshot)
budget_path = Path(job['budget_path'])
budget_path.parent.mkdir(parents=True, exist_ok=True)
if budget_path.exists():
    if json.dumps(scheduler.read_json(budget_path), sort_keys=True) != json.dumps(job['budget'], sort_keys=True):
        raise ValueError('Existing budget differs from the prepared plan')
else:
    scheduler.write_json(budget_path, job['budget'], exclusive=True)
folder.mkdir(parents=True, exist_ok=False)
job.update(status='running', started_at=datetime.now(timezone.utc).isoformat(),
           launch_scope='one free slot; older scheduler has no queued jobs',
           active_fits_before_launch=active, prepared_plan=str(args.plan.resolve()))
with (folder / 'fit.log').open('x') as log:
    process = subprocess.Popen(job['command'], cwd=scheduler.PROJECT,
                               env={**os.environ, **plan['environment_overrides']},
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    job['pid'] = process.pid
    scheduler.write_json(folder / 'scheduler_state.json', job, exclusive=True)
    print(json.dumps({'started_pid': process.pid, 'output': job['output'],
                      'active_before': len(active), 'package_root': plan['package_root']}), flush=True)
    code = process.wait()
error = scheduler.completion_error(job, snapshot, priors) if code == 0 else f'exit {code}'
job.update(status='complete' if error is None else 'failed', returncode=code,
           detail=error, ended_at=datetime.now(timezone.utc).isoformat())
scheduler.write_json(folder / 'scheduler_state.json', job)
print(json.dumps({'status': job['status'], 'detail': error}), flush=True)
sys.exit(0 if error is None else 1)
