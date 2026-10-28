#!/usr/bin/env python3
"""Fresh-process stability suite. Dry-run is the default; --run starts fits.

Run this script in WSL with the isolated mica2-slowgp Python. One fixed flock
serializes scheduler invocations across output roots. All visible native CLI
fit processes count against --jobs, including manually launched references.
Existing incomplete directories are never resumed, erased, or overwritten.
Completion means files are consistent, not that a posterior passed stability.

Examples (add --run only when ready):
  python -B run_stability_suite.py --sampler dnest \
      --pair independent:independent --pair shared:shared --jobs 4
  python -B run_stability_suite.py --sampler dynesty --case clean --mode none
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROJECT = Path('/home/leonhard/sci001')
PYTHON = Path('/home/leonhard/miniforge3/envs/mica2-slowgp/bin/python')
PACKAGE = Path('/home/leonhard/software/MICA2-slowgp/src')
INPUTS = PROJECT / 'output/mica_slowgp/small_cases'
OUTPUT = PROJECT / 'output/mica_slowgp/stability_v1'
# Deliberately independent of --out-root: alternate roots cannot bypass it.
COORDINATION = PROJECT / 'output/mica_slowgp/_stability_scheduler'
CASES = ('clean', 'response_only', 'independent', 'shared')
MODES = ('none', 'independent', 'shared')
NAMES = ('sigma_x', 'tau_x', 'A0', 'mu0', 'w0', 'A1', 'mu1', 'w1')
SLOW_NAMES = ('qW', 'qb', 'Ts')
ENVIRONMENT = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
               'UCX_TLS': 'self', 'UCX_NET_DEVICES': 'lo',
               'PYTHONPATH': str(PACKAGE)}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, exclusive=False):
    """Atomic state updates; exclusive initial artifacts never replace files."""
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(text)
    else:
        temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
        with temporary.open('x') as stream:
            stream.write(text)
        temporary.replace(path)


def budget_for(sampler, effort):
    if sampler == 'dnest':
        return dict(max_num_saves=effort, num_particles=16,
                    thread_steps_factor=1.0, new_level_interval_factor=2.0,
                    save_interval_factor=1.0, max_num_levels=32,
                    lam=10.0, beta=100.0, ptol=0.1)
    return dict(nlive=effort, dlogz=0.05, bound='multi', sample='rwalk',
                walks=50, bootstrap=0, checkpoint_every=30.0)


def active_fits():
    """Read actual Linux argv, not stale PIDs in old launch/state documents."""
    processes = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().decode().split('\0')
            module = argv.index('-m')
            if argv[module + 1:module + 3] != ['mica_slowgp.cli', 'fit']:
                continue
            out_index = argv.index('--out')
            output = argv[out_index + 1]
            if not Path(output).is_absolute():
                output = str(((entry / 'cwd').resolve() / output).resolve())
            else:
                output = str(Path(output).resolve())
            processes.append({'pid': int(entry.name), 'output': output})
        except (OSError, UnicodeError, ValueError, IndexError):
            continue  # Includes finished processes and unrelated executables.
    return sorted(processes, key=lambda item: item['pid'])


def snapshot_inputs(inputs_root, pairs):
    """Validate the fixed 42+42 point inputs, and retain bytes against drift."""
    snapshot = {}
    for case in sorted({case for case, _ in pairs}):
        for name in ('w2.txt', 'response.txt'):
            path = inputs_root / case / name
            raw = path.read_bytes()
            rows = [list(map(float, line.split())) for line in raw.decode().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')]
            if len(rows) != 42 or any(len(row) != 3 for row in rows):
                raise ValueError(f'Expected fixed 42 by 3 small-case input: {path}')
            if any(not all(map(math.isfinite, row)) or row[2] <= 0 for row in rows):
                raise ValueError(f'Invalid input values/errors: {path}')
            snapshot[path] = raw
    prior_path = inputs_root / 'priors.json'
    snapshot[prior_path] = prior_path.read_bytes()
    return snapshot, json.loads(snapshot[prior_path])


def verify_snapshot(snapshot):
    for path, expected in snapshot.items():
        if path.read_bytes() != expected:
            raise ValueError(f'Fixed input or prior changed during scheduling: {path}')


def make_job(args, case, mode, effort, seed):
    output = args.out_root / args.sampler / f'{case}_{mode}' / f'budget_{effort}' / f'seed_{seed}'
    if args.sampler == 'dnest' and len(str(output).encode()) > 160:
        raise ValueError(f'DNest native path too long (limit 160 bytes): {output}')
    budget_path = args.out_root / '_scheduler/budgets' / f'{args.sampler}_{effort}.json'
    command = [str(PYTHON), '-B', '-m', 'mica_slowgp.cli', 'fit',
               '--driver', str(args.inputs_root / case / 'w2.txt'),
               '--response', str(args.inputs_root / case / 'response.txt'),
               '--mode', mode, '--priors', str(args.inputs_root / 'priors.json'),
               '--out', str(output), '--seed', str(seed), '--sampler', args.sampler,
               '--likelihood', 'fast', '--budget', str(budget_path)]
    return dict(case=case, mode=mode, sampler=args.sampler, effort=effort,
                seed=seed, output=str(output), budget_path=str(budget_path),
                budget=budget_for(args.sampler, effort), command=command)


def completion_error(job, snapshot, priors):
    """Return None only for a complete posterior with the requested setup."""
    folder = Path(job['output'])
    required = ('weighted_posterior.csv', 'diagnostics.json', 'posterior_summary.json',
                'sampler_config.json', 'run_input.json', 'input_w2.txt', 'input_response.txt')
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        return 'missing artifacts: ' + ', '.join(missing)
    try:
        config = read_json(folder / 'sampler_config.json')
        run_input = read_json(folder / 'run_input.json')
        names = list(NAMES + (() if job['mode'] == 'none' else SLOW_NAMES))
        expected = dict(sampler=job['sampler'], mode=job['mode'], likelihood='fast')
        for key, value in expected.items():
            if run_input.get(key) != value:
                return f'run_input {key} mismatch'
        if run_input.get('points') != [42, 42]:
            return 'input point counts mismatch'
        if (config.get('sampler', 'dnest') != job['sampler'] or
                config.get('mode') != job['mode'] or config.get('seed') != job['seed'] or
                config.get('budget') != job['budget'] or config.get('parameter_names') != names):
            return 'sampler, mode, seed, budget, or parameter names mismatch'
        if config.get('priors') != {name: priors[name] for name in names}:
            return 'active parameter priors mismatch'
        for key, copied in [('driver', 'input_w2.txt'), ('response', 'input_response.txt')]:
            source = Path(job['command'][job['command'].index('--' + key) + 1])
            if run_input.get(key) != str(source) or (folder / copied).read_bytes() != snapshot[source]:
                return f'{key} input differs from fixed source'
        summary = read_json(folder / 'posterior_summary.json')
        diagnostics = read_json(folder / 'diagnostics.json')
        if not isinstance(diagnostics, dict) or not diagnostics:
            return 'empty or invalid diagnostics'
        for name in names:
            quantiles = [summary[name][key] for key in ('q16', 'q50', 'q84')]
            if not all(map(math.isfinite, quantiles)) or quantiles != sorted(quantiles):
                return f'invalid posterior summary for {name}'
        rows = 0
        total_weight = 0.0
        with (folder / 'weighted_posterior.csv').open() as stream:
            reader = csv.DictReader(stream)
            if not set(names + ['weight', 'log_likelihood']).issubset(reader.fieldnames or []):
                return 'weighted posterior columns missing'
            for row in reader:
                values = [float(row[name]) for name in names]
                weight = float(row['weight'])
                if not all(map(math.isfinite, values)) or not math.isfinite(weight) or weight < 0:
                    return 'invalid posterior parameters or weights'
                if weight > 0 and not math.isfinite(float(row['log_likelihood'])):
                    return 'nonfinite likelihood with positive posterior weight'
                if values[3] > values[6]:
                    return 'unordered Gaussian centers in posterior'
                total_weight += weight
                rows += 1
        if rows < 2 or not math.isclose(total_weight, 1.0, rel_tol=1e-6, abs_tol=1e-8):
            return 'posterior is empty or weights are not normalized'
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        return f'unreadable or malformed artifacts: {error}'
    return None


def classify_job(job, active, snapshot, priors):
    """An active writer takes precedence over apparently complete artifacts."""
    running = [item['pid'] for item in active if item['output'] == job['output']]
    if running:
        return 'waiting_external', f'active fit PID(s): {running}'
    if not Path(job['output']).exists():
        return 'planned', 'new directory'
    error = completion_error(job, snapshot, priors)
    if error is None:
        return 'skipped_complete', 'validated existing result; stability not inferred'
    return 'blocked_unknown_incomplete', error


def build_plan(args):
    pairs = ([tuple(pair.split(':')) for pair in args.pair] if args.pair else
             [(case, mode) for case in (args.case or CASES) for mode in (args.mode or MODES)])
    pairs = list(dict.fromkeys(pairs))
    if any(case not in CASES or mode not in MODES for case, mode in pairs):
        raise ValueError('--pair must be CASE:MODE using the documented case/mode choices')
    snapshot, priors = snapshot_inputs(args.inputs_root, pairs)
    efforts = args.budgets
    if args.budget_level != 'both':
        efforts = [efforts[0 if args.budget_level == 'lower' else 1]]
    jobs = [make_job(args, case, mode, effort, seed) for effort in efforts
            for case, mode in pairs for seed in args.seeds]
    active = active_fits()
    for job in jobs:
        job['status'], job['detail'] = classify_job(job, active, snapshot, priors)
    plan = dict(schema='mica_slowgp_fresh_process_suite_v1', created_at=utcnow(),
                dry_run=not args.run, scheduler_pid=os.getpid(), jobs_limit=args.jobs,
                global_lock=str(COORDINATION / 'scheduler.lock'),
                global_active_fits=active, available_slots=max(0, args.jobs - len(active)),
                sampler=args.sampler, inputs_root=str(args.inputs_root),
                out_root=str(args.out_root), fixed_priors=priors, points_per_case=[42, 42],
                environment_overrides=ENVIRONMENT, budgets=args.budgets,
                budget_level=args.budget_level, seeds=args.seeds, jobs=jobs,
                status='planned', completion_does_not_establish_convergence=True)
    return plan, snapshot, priors


def print_summary(plan):
    summary = {key: plan[key] for key in
               ('dry_run', 'sampler', 'jobs_limit', 'global_active_fits', 'available_slots')}
    summary['counts'] = dict(Counter(job['status'] for job in plan['jobs']))
    summary['blocked'] = [dict(output=job['output'], detail=job['detail'])
                          for job in plan['jobs'] if job['status'].startswith('blocked')]
    summary['first_planned_command'] = next((shlex.join(job['command']) for job in plan['jobs']
                                             if job['status'] == 'planned'), None)
    print(json.dumps(summary, indent=2), flush=True)


def execute(plan, snapshot, priors, args):
    """Called only after --run. Launch at most one fresh process per directory."""
    import fcntl

    COORDINATION.mkdir(parents=True, exist_ok=True)
    with (COORDINATION / 'scheduler.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another global stability scheduler is running; no fits launched') from error
        # Re-evaluate after acquiring the global lock, closing the planning race.
        active = active_fits()
        for job in plan['jobs']:
            job['status'], job['detail'] = classify_job(job, active, snapshot, priors)
        blocked = [job for job in plan['jobs'] if job['status'].startswith('blocked')]
        if blocked:
            print_summary(plan)
            raise RuntimeError('Existing incomplete/mismatched directories block dispatch; no fits launched')
        verify_snapshot(snapshot)
        scheduler_dir = args.out_root / '_scheduler'
        (scheduler_dir / 'budgets').mkdir(parents=True, exist_ok=True)
        for job in plan['jobs']:
            budget_path = Path(job['budget_path'])
            if budget_path.exists():
                if read_json(budget_path) != job['budget']:
                    raise ValueError(f'Existing budget file differs: {budget_path}')
            else:
                write_json(budget_path, job['budget'], exclusive=True)
        invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + f'_{os.getpid()}'
        manifest = scheduler_dir / f'manifest_{invocation}.json'
        plan.update(status='running', started_at=utcnow(), manifest=str(manifest))
        write_json(manifest, plan, exclusive=True)
        owned = {}
        failed = False

        def save():
            plan['updated_at'] = utcnow()
            plan['global_active_fits'] = active_fits()
            plan['available_slots'] = max(0, args.jobs - len(plan['global_active_fits']))
            write_json(manifest, plan)
            write_json(COORDINATION / 'registry.json', {
                'updated_at': plan['updated_at'], 'scheduler_pid': os.getpid(),
                'manifest': str(manifest), 'out_root': str(args.out_root),
                'jobs_limit': args.jobs, 'status': plan['status'],
                'active_fits': plan['global_active_fits']})

        try:
            save()
            while True:
                changed = False
                for output, (process, log, job) in list(owned.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    error = completion_error(job, snapshot, priors) if code == 0 else f'fit exited {code}'
                    job.update(status='complete' if error is None else 'failed',
                               detail=error or 'validated completed result; stability not inferred',
                               returncode=code, ended_at=utcnow())
                    write_json(Path(output) / 'scheduler_state.json', job)
                    del owned[output]
                    failed = failed or error is not None
                    changed = True
                active = active_fits()
                for job in plan['jobs']:
                    if job['status'] != 'waiting_external':
                        continue
                    status, detail = classify_job(job, active, snapshot, priors)
                    if status != 'waiting_external':
                        # A disappearing external directory is not permission to start it again.
                        if status == 'planned':
                            status, detail = 'blocked_unknown_incomplete', 'external run disappeared'
                        job.update(status=status, detail=detail)
                        failed = failed or status.startswith('blocked')
                        changed = True
                if not failed:
                    for job in plan['jobs']:
                        if job['status'] != 'planned':
                            continue
                        active = active_fits()
                        # Count just-launched children even before /proc exposes their final argv.
                        active_pids = {item['pid'] for item in active}
                        active_pids.update(process.pid for process, _, _ in owned.values()
                                           if process.poll() is None)
                        if len(active_pids) >= args.jobs:
                            break
                        verify_snapshot(snapshot)
                        status, detail = classify_job(job, active, snapshot, priors)
                        if status != 'planned':
                            job.update(status=status, detail=detail)
                            failed = status.startswith('blocked')
                            changed = True
                            if failed:
                                break
                            continue
                        output = Path(job['output'])
                        output.mkdir(parents=True, exist_ok=False)
                        log = (output / 'fit.log').open('x')
                        job.update(status='starting', started_at=utcnow(),
                                   scheduler_manifest=str(manifest))
                        write_json(output / 'scheduler_state.json', job, exclusive=True)
                        try:
                            process = subprocess.Popen(job['command'], cwd=PROJECT,
                                                       env={**os.environ, **ENVIRONMENT},
                                                       stdout=log, stderr=subprocess.STDOUT,
                                                       start_new_session=True)
                        except OSError as error:
                            log.close()
                            job.update(status='failed', detail=str(error), ended_at=utcnow())
                            write_json(output / 'scheduler_state.json', job)
                            failed = changed = True
                            break
                        job.update(status='running', pid=process.pid, detail='fresh CLI process')
                        owned[str(output)] = (process, log, job)
                        write_json(output / 'scheduler_state.json', job)
                        changed = True
                if changed:
                    save()
                    print(json.dumps({'at': utcnow(), 'counts': dict(Counter(
                        job['status'] for job in plan['jobs']))}), flush=True)
                awaiting = any(job['status'] in ('planned', 'waiting_external') for job in plan['jobs'])
                if not owned and (failed or not awaiting):
                    break
                time.sleep(args.poll_seconds)
            plan.update(status='failed' if failed else 'complete', ended_at=utcnow())
            save()
            return 1 if failed else 0
        except BaseException as error:
            # Independent sessions continue. A later scheduler observes their live PIDs.
            plan.update(status='interrupted', detail=f'{type(error).__name__}: {error}', ended_at=utcnow())
            save()
            raise
        finally:
            for _, log, _ in owned.values():
                log.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sampler', choices=('dnest', 'dynesty'), required=True)
    parser.add_argument('--case', action='append', choices=CASES)
    parser.add_argument('--mode', action='append', choices=MODES)
    parser.add_argument('--pair', action='append', metavar='CASE:MODE',
                        help='Select exact pairs instead of the case by mode Cartesian product')
    parser.add_argument('--seeds', type=int, nargs='+', default=[71231, 71247, 71263])
    parser.add_argument('--budgets', type=int, nargs=2, metavar=('LOW', 'HIGH'),
                        help='Defaults: DNest 1000 2000; dynesty 400 800; HIGH must be >=2*LOW')
    parser.add_argument('--budget-level', choices=('both', 'lower', 'upper'), default='both',
                        help='Stage one budget while keeping the fixed two-budget design explicit')
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=2)
    parser.add_argument('--inputs-root', type=Path, default=INPUTS)
    parser.add_argument('--out-root', type=Path, default=OUTPUT)
    parser.add_argument('--plan-json', type=Path, help='Write a new reviewable plan; existing file is never overwritten')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true', help='Launch fresh fits; default is dry-run')
    action.add_argument('--dry-run', action='store_true', help='Explicitly request the default planning-only behavior')
    parser.add_argument('--poll-seconds', type=float, default=2.0)
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or not Path('/proc').is_dir():
        parser.error('Run inside WSL/Linux, not the Windows Python interpreter')
    if args.pair and (args.case or args.mode):
        parser.error('--pair cannot be combined with --case or --mode')
    if args.pair and any(pair.count(':') != 1 for pair in args.pair):
        parser.error('--pair must contain exactly one colon')
    args.budgets = args.budgets or ([1000, 2000] if args.sampler == 'dnest' else [400, 800])
    if args.budgets[0] <= 0 or args.budgets[1] < 2 * args.budgets[0]:
        parser.error('Budgets require 0 < LOW and HIGH >= 2*LOW')
    if args.sampler == 'dynesty' and args.budgets[0] < 50:
        parser.error('dynesty requires nlive >= 50')
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 1 or seed >= 2**32 for seed in args.seeds):
        parser.error('Seeds must be distinct integers in [1, 2**32)')
    if not math.isfinite(args.poll_seconds) or not 0.1 <= args.poll_seconds <= 60:
        parser.error('--poll-seconds must be between 0.1 and 60')
    if not PYTHON.is_file() or not (PACKAGE / 'mica_slowgp/cli.py').is_file():
        parser.error('The isolated Python environment or package is unavailable')
    args.inputs_root = args.inputs_root.resolve()
    args.out_root = args.out_root.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    plan, snapshot, priors = build_plan(args)
    print_summary(plan)
    if args.plan_json:
        write_json(args.plan_json.resolve(), plan, exclusive=True)
    if args.run:
        return execute(plan, snapshot, priors, args)
    return 2 if any(job['status'].startswith('blocked') for job in plan['jobs']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
