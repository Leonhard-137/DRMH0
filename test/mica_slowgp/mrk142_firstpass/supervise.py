#!/usr/bin/env python3
"""Own only the six processes in one prepared manifest; never retry fits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TERMINAL = {'finished', 'deadline', 'interrupted', 'supervisor_failed'}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value, temporary_dir=None):
    path = Path(path)
    # The frozen scientific runner rejects unknown names in its run directory.
    # Scheduler writes during startup stage their temporary file outside it.
    directory = Path(temporary_dir) if temporary_dir is not None else path.parent
    temporary = directory / (path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_complete(job, manifest):
    """Require all three conditional runs and reuse frozen native readers."""
    source = manifest['run_settings']['source']
    if source not in sys.path:
        sys.path.insert(0, source)
    from mica_slowgp.stratified import _child_state, _load_child, _completed_parent
    folder = Path(job['out_dir'])
    if not (folder / 'completed.json').is_file():
        raise ValueError('No parent completion marker')
    config = json.loads((folder / 'sampler_config.json').read_text())
    settings = manifest['run_settings']
    if (config['mode'] != 'shared' or config['seed'] != job['seed'] or
        config['center_strata'] != settings['center_strata'] or len(config['child_seeds']) != 3 or
        config['priors'] != json.loads(Path(job['priors']).read_text()) or config['budget'] != settings['budget']):
        raise ValueError('Completed parent differs from the fixed manifest')
    count = 0
    for index in range(3):
        child = folder / 'strata' / f'stratum_{index:02d}'
        if _child_state(child) != 'complete':
            raise ValueError(f'Stratum {index} is incomplete')
        cc = json.loads((child / 'sampler_config.json').read_text())
        if (cc['seed'] != config['child_seeds'][index] or cc['priors'] != config['priors'] or cc['budget'] != config['budget'] or
            cc['center_interval'] != config['center_strata'][index:index + 2]):
            raise ValueError(f'Stratum {index} config differs from parent')
        count += len(_load_child(child, cc)['samples'])
    result = _completed_parent(folder, config)
    if result['diagnostics']['raw_sample_count'] != count:
        raise ValueError('Parent does not contain all conditional rows')
    return dict(raw_sample_count=count, strata_complete=3)


def progress_for(folder):
    """Read small metadata only; a checkpoint timestamp is not convergence."""
    result = []
    for index in range(3):
        child = folder / 'strata' / f'stratum_{index:02d}'
        check = child / 'dynesty.checkpoint'
        diagnostic = child / 'diagnostics.json'
        item = dict(stratum=index, checkpoint_exists=check.is_file())
        if check.is_file():
            stat = check.stat()
            item.update(checkpoint_bytes=stat.st_size, checkpoint_mtime=stat.st_mtime)
        if diagnostic.is_file():
            try:
                d = json.loads(diagnostic.read_text())
                item.update(final_diagnostics_present=True, raw_sample_count=d.get('raw_sample_count'),
                            exact_likelihood_calls=d.get('exact_likelihood_calls_total'))
            except (ValueError, OSError):
                item['final_diagnostics_present'] = False
        result.append(item)
    return result


def monitor(root, interval):
    root = Path(root)
    monitor_dir = root / 'monitor'
    monitor_dir.mkdir(exist_ok=True)
    last_write = -float('inf')
    while True:
        try:
            state = json.loads((root / 'status.json').read_text())
        except (ValueError, OSError):
            time.sleep(min(10, interval))
            continue
        terminal = state.get('state') in TERMINAL
        now = time.monotonic()
        if now - last_write >= interval or terminal:
            snapshot = dict(observed_at_utc=utcnow(), status=state)
            atomic_json(monitor_dir / 'latest.json', snapshot)
            with (monitor_dir / 'progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(snapshot, allow_nan=False) + '\n')
            last_write = now
        if terminal:
            return 0
        supervisor_pid = state.get('pid')
        if supervisor_pid and not Path(f'/proc/{supervisor_pid}').exists():
            atomic_json(monitor_dir / 'supervisor_missing.json', dict(observed_at_utc=utcnow(), pid=supervisor_pid,
                note='Supervisor disappeared; no fit has been restarted or signalled by the monitor'))
            return 2
        time.sleep(min(10, interval))


def signal_owned(process, signum):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


def supervise(manifest_path, validator=validate_complete, start_monitor=True):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    root, settings = Path(manifest['root']), manifest['run_settings']
    lock = root / 'supervisor.started.json'
    # Exclusive marker prevents duplicate launches even after a crashed supervisor.
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), started_at_utc=utcnow(), manifest=str(manifest_path)), stream)
    environment = os.environ.copy()
    environment.update(settings['environment'])
    os.environ.update(settings['environment'])
    start = time.monotonic()
    deadline = start + settings['global_fit_deadline_seconds']
    status = dict(schema='mrk142-supervisor-status-v1', state='running', pid=os.getpid(),
        started_at_utc=utcnow(), deadline_seconds=settings['global_fit_deadline_seconds'],
        deadline_unix=time.time() + settings['global_fit_deadline_seconds'], manifest=str(manifest_path),
        jobs=[], postprocess=None)
    processes, logs = {}, {}
    stop = {'reason': None}
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, lambda signum, frame: stop.update(reason='interrupted'))
    old_handlers[signal.SIGHUP] = signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def publish():
        status['updated_at_utc'] = utcnow()
        status['fit_elapsed_seconds'] = time.monotonic() - start
        for item in status['jobs']:
            item['progress'] = progress_for(Path(item['out_dir']))
        atomic_json(root / 'status.json', status)

    def reap():
        for item in status['jobs']:
            process = processes.get(item['id'])
            if process is None or item['state'] != 'running' or process.poll() is None:
                continue
            item.update(returncode=process.returncode, ended_at_utc=utcnow())
            if process.returncode == 0:
                try:
                    item['completion'] = validator(next(j for j in manifest['jobs'] if j['id'] == item['id']), manifest)
                    item['state'] = 'complete'
                except Exception as error:
                    item.update(state='failed', error=f'Completion validation: {type(error).__name__}: {error}')
            else:
                item['state'] = 'stopped' if stop['reason'] else 'failed'
            logs[item['id']].close()

    try:
        publish()
        if start_monitor:
            monitor_log = (root / 'monitor_process.log').open('x')
            process = subprocess.Popen([settings['python'], '-B', str(Path(__file__).resolve()), '--monitor',
                '--manifest', str(manifest_path)], cwd=settings['cwd'], env=environment, stdin=subprocess.DEVNULL,
                stdout=monitor_log, stderr=subprocess.STDOUT, start_new_session=True)
            status['monitor_pid'] = process.pid
            monitor_log.close()
        if len(manifest['jobs']) > settings['max_parallel']:
            raise ValueError('The fixed first pass must fit within its simultaneous-job limit')
        for job in manifest['jobs']:
            if stop['reason'] or time.monotonic() >= deadline:
                stop['reason'] = stop['reason'] or 'deadline'
                break
            folder = Path(job['out_dir'])
            item = {key: job[key] for key in ('id', 'band', 'role', 'seed', 'out_dir')}
            status['jobs'].append(item)
            try:
                folder.mkdir(parents=True, exist_ok=False)
                log = (folder / 'fit.log').open('x')
                logs[job['id']] = log
                process = subprocess.Popen(job['command'], cwd=settings['cwd'], env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes[job['id']] = process
                item.update(state='running', pid=process.pid, started_at_utc=utcnow())
                atomic_json(folder / 'scheduler_state.json', dict(status='running', pid=process.pid,
                    seed=job['seed'], supervisor_pid=os.getpid()), temporary_dir=root)
            except Exception as error:
                item.update(state='failed', error=f'{type(error).__name__}: {error}')
                if job['id'] in logs:
                    logs[job['id']].close()
            publish()
        while any(item['state'] == 'running' for item in status['jobs']):
            reap()
            if time.monotonic() >= deadline and stop['reason'] is None:
                stop['reason'] = 'deadline'
            if stop['reason']:
                break
            publish()
            if any(item['state'] == 'running' for item in status['jobs']):
                time.sleep(min(settings['poll_seconds'], max(0, deadline - time.monotonic())))
    except BaseException as error:
        stop['reason'] = 'supervisor_failed'
        status['error'] = f'{type(error).__name__}: {error}'
    finally:
        if any(p.poll() is None for p in processes.values()):
            stop['reason'] = stop['reason'] or 'interrupted'
            status['state'] = 'stopping'
            status['stop_reason'] = stop['reason']
            publish()
            for sig, wait in ((signal.SIGINT, settings['interrupt_grace_seconds']),
                              (signal.SIGTERM, settings['terminate_grace_seconds']), (signal.SIGKILL, 5)):
                living = [p for p in processes.values() if p.poll() is None]
                if not living:
                    break
                status['last_stop_signal'] = signal.Signals(sig).name
                for process in living:
                    signal_owned(process, sig)
                until = time.monotonic() + wait
                while time.monotonic() < until and any(p.poll() is None for p in living):
                    time.sleep(min(0.2, max(0, until - time.monotonic())))
                reap()
                publish()
        reap()
        for item in status['jobs']:
            folder = Path(item['out_dir'])
            if folder.is_dir() and item.get('pid'):
                atomic_json(folder / 'scheduler_state.json', dict(status=item['state'], pid=item['pid'],
                    seed=item['seed'], supervisor_pid=os.getpid(), returncode=item.get('returncode'),
                    ended_at_utc=item.get('ended_at_utc'), error=item.get('error')))
        status['fit_finish_reason'] = stop['reason'] or 'all_jobs_exited'
        status['state'] = 'postprocessing'
        publish()
        if manifest.get('postprocess_command'):
            with (root / 'postprocess.log').open('x') as log:
                try:
                    post = subprocess.Popen(manifest['postprocess_command'], cwd=settings['cwd'], env=environment,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    status['postprocess'] = dict(state='running', pid=post.pid, started_at_utc=utcnow())
                    publish()
                    while post.poll() is None:
                        time.sleep(min(settings['poll_seconds'], 10))
                        publish()
                    status['postprocess'].update(state='complete' if post.returncode == 0 else 'partial_or_failed',
                        returncode=post.returncode, ended_at_utc=utcnow())
                except Exception as error:
                    status['postprocess'] = dict(state='failed', error=f'{type(error).__name__}: {error}')
        status.update(state=stop['reason'] or 'finished', ended_at_utc=utcnow(),
                      complete_jobs=sum(i['state'] == 'complete' for i in status['jobs']))
        publish()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--monitor', action='store_true')
    args = parser.parse_args()
    if args.monitor:
        manifest = json.loads(args.manifest.read_text())
        raise SystemExit(monitor(manifest['root'], manifest['run_settings']['monitor_interval_seconds']))
    state = supervise(args.manifest)
    print(json.dumps(dict(state=state['state'], complete_jobs=state['complete_jobs'])))
    raise SystemExit(0 if state['complete_jobs'] == len(state['jobs']) else 3)


if __name__ == '__main__':
    main()
