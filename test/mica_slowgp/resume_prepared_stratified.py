#!/usr/bin/env python3
"""Explicitly resume one stopped stratified job from a reviewed prepared plan."""
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
spec = importlib.util.spec_from_file_location('bounded_scheduler', Path(__file__).with_name('run_stability_suite.py'))
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)
plan = json.loads(args.plan.read_text())
job = dict(plan['jobs'][args.job_index])
folder = Path(job['output'])
active = scheduler.active_fits()
if any(item['output'] == job['output'] for item in active):
    raise RuntimeError('The requested fit is still running; refusing duplicate resume')
if len(active) >= 4:
    raise RuntimeError('All four scientific fit slots are occupied')
registry = scheduler.read_json(scheduler.COORDINATION / 'registry.json')
previous_plan = scheduler.read_json(registry['manifest'])
if any(j['status'] in ('planned', 'starting', 'waiting_external') for j in previous_plan['jobs']):
    raise RuntimeError('The current scheduler has queued work; refusing manual resume')
if not (folder / 'sampler_config.json').is_file() or (folder / 'completed.json').exists():
    raise RuntimeError('Explicit resume requires an existing unfinished stratified parent')
snapshot, priors = scheduler.snapshot_inputs(Path(plan['inputs_root']), [(job['case'], job['mode'])])
scheduler.verify_snapshot(snapshot)
previous_state = scheduler.read_json(folder / 'scheduler_state.json')
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
logs = Path(plan['out_root']) / '_scheduler/resumes'
logs.mkdir(parents=True, exist_ok=True)
log_path = logs / f"{job['effort']}_{job['seed']}_{stamp}.log"
job['command'] += ['--resume']
job.update(status='resuming', started_at=datetime.now(timezone.utc).isoformat(),
           prepared_plan=str(args.plan.resolve()), package_root=plan['package_root'],
           resume_log=str(log_path), previous_attempt=previous_state)
with log_path.open('x') as log:
    process = subprocess.Popen(job['command'], cwd=scheduler.PROJECT,
        env={**os.environ, **plan['environment_overrides']},
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    job.update(pid=process.pid, status='running')
    scheduler.write_json(folder / 'scheduler_state.json', job)
    print(json.dumps({'resumed_pid': process.pid, 'output': job['output'],
                      'log': str(log_path), 'active_before': len(active)}), flush=True)
    code = process.wait()
error = scheduler.completion_error(job, snapshot, priors) if code == 0 else f'exit {code}'
job.update(status='complete' if error is None else 'failed', returncode=code,
           detail=error, ended_at=datetime.now(timezone.utc).isoformat())
scheduler.write_json(folder / 'scheduler_state.json', job)
print(json.dumps({'status': job['status'], 'detail': error}), flush=True)
sys.exit(0 if error is None else 1)
