#!/usr/bin/env python3
"""Rolling queue for saved PSD-candidate inputs, with <=8 active workers.

Each fit uses the existing full native MICA workflow and two MPI ranks.
Project transfer/lightcurve/corner plots run one job at a time and occupy a
worker slot. Existing native checkpoints are resumed by run_mica, never reset.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE/'cases_psd_candidates'
CONFIG = BASE/'experiment.yaml'
PYTHON = Path('/home/leonhard/miniforge3/envs/mica2/bin/python')
MONITOR = BASE/'monitoring'
ORDER = ('C045','C001','C078','C075','C089','C069')
VARIANTS = ('raw','all_s5','all_s10','all_s20','all_s40')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def native_complete(run):
    try:
        state=json.loads((run/'run_mica_state.json').read_text())
        return state.get('status')=='complete' and state.get('target_saves')==10000
    except (OSError,ValueError):
        return False


def plots_complete(case, model):
    output=case/'results'/model
    try:
        summary=json.loads((output/'summary.json').read_text())
    except (OSError,ValueError):
        return False
    if not {'transfer','lightcurves','corner'} <= summary.keys():
        return False
    return all((output/f'{name}.{ext}').is_file() and (output/f'{name}.{ext}').stat().st_size>100
               for name in ('transfer_function','light_curves','corner') for ext in ('jpg','pdf'))


def prepare_jobs():
    # build_plan is read-only: it validates input/configuration and native priors.
    from fit import build_plan, check_config_target
    validation=json.loads((BASE/'validation.json').read_text())
    if validation.get('status')!='pass':
        raise RuntimeError('Input validation is not pass; fits are not started')
    plan=build_plan(CONFIG,suite='paired_ablation',model='both')
    if len(plan)!=60:
        raise RuntimeError(f'Expected 60 fits, found {len(plan)}')
    for job in plan:
        check_config_target(job['case'],job['config'])
        f=job['fit']
        if (f['ncomp'],f['mpi_per_band'],f['max_num_saves'],f['ptol'])!=(2,2,10000,.01):
            raise RuntimeError(f'Unexpected fit configuration for {job["case"]}')
    lookup={(j['case'].name,j['fit']['type_tf']):j for j in plan}
    jobs=[]
    for model in ('gamma','gaussian'):
        for variant in VARIANTS:
            for candidate in ORDER:
                name=f'{candidate}_{variant}'
                item=lookup[name,model]
                complete=native_complete(item['run_dir'])
                phase=('done' if plots_complete(item['case'],model) else 'plot_ready') if complete else 'fit_ready'
                jobs.append(dict(id=f'{name}/{model}',candidate_id=candidate,case=name,
                    case_dir=str(item['case']),model=model,stage=item['stage'],
                    run_dir=str(item['run_dir']),phase=phase))
    return jobs


def available_memory_gib():
    values={row.split(':')[0]:int(row.split()[1]) for row in Path('/proc/meminfo').read_text().splitlines()}
    return values['MemAvailable']/1024**2, (values['SwapTotal']-values['SwapFree'])/1024**2


def active_prior_workers():
    result=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid():
            continue
        try:
            command=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:
            continue
        if str(BASE) in command and any(name in command for name in ('/fit.py','/run_mica.py','/plot_results.py')):
            result.append(dict(pid=int(process.name),command=command))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Start the validated queue; default only reports plan')
    parser.add_argument('--max-running',type=int,default=8,choices=range(1,9))
    args=parser.parse_args()
    jobs=prepare_jobs()
    if not args.run:
        print(json.dumps(dict(jobs=len(jobs),max_running=args.max_running,
            counts=dict(Counter(j['phase'] for j in jobs)),first_jobs=[j['id'] for j in jobs[:8]],
            config=str(CONFIG)),indent=2))
        return
    MONITOR.mkdir(exist_ok=True)
    logs=MONITOR/'jobs'
    logs.mkdir(exist_ok=True)
    lock=(MONITOR/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    existing=active_prior_workers()
    if existing:
        raise RuntimeError(f'Previous workers are still active; not duplicating jobs: {existing}')
    env=os.environ.copy()
    env.update(PATH=str(PYTHON.parent)+':'+env.get('PATH',''),TMPDIR='/tmp',
               PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    env.pop('TMP',None)
    env.pop('TEMP',None)
    active={}
    errors=[]
    started=now()
    shutting_down=False

    def stopping(signum, frame):
        nonlocal shutting_down
        shutting_down=True
        print(f'{now()} controller stop requested by signal {signum}; draining active jobs',flush=True)
    signal.signal(signal.SIGTERM,stopping)
    signal.signal(signal.SIGHUP,stopping)
    signal.signal(signal.SIGINT,stopping)

    def snapshot(status='running'):
        memory,swap=available_memory_gib()
        payload=dict(status=status,started_utc=started,updated_utc=now(),controller_pid=os.getpid(),
            max_active_workers=args.max_running,max_parallel_plots=1,mpi_per_fit=2,
            input_definition='W2 and U: original minus Gaussian trend plus original mean; original errors',
            config=str(CONFIG),total_jobs=len(jobs),counts=dict(Counter(j['phase'] for j in jobs)),
            active_fits=sum(v['kind']=='fit' for v in active.values()),
            active_plots=sum(v['kind']=='plot' for v in active.values()),
            memory_available_gib=memory,swap_used_gib=swap,errors=errors,jobs=jobs)
        write_json(MONITOR/'status.json',payload)

    def start(job,kind):
        script=HERE/('fit.py' if kind=='fit' else 'plot_results.py')
        command=[str(PYTHON),'-B','-u',str(script),'--config',str(CONFIG),
                 '--case',job['case'],'--model',job['model']]
        if kind=='fit':
            command.append('--run')
        else:
            command.extend(['--component-sum-tolerance-sigma','0.1'])
        stem=f"{job['case']}_{job['model']}_{kind}"
        attempt=1
        while True:
            path=logs/f'{stem}_{attempt:03d}.log'
            try:
                handle=path.open('x')
                break
            except FileExistsError:
                attempt+=1
        try:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        except BaseException:
            handle.close()
            raise
        job.update(phase=kind+'_running',pid=process.pid,log=str(path),stage_started_utc=now())
        active[process.pid]=dict(process=process,handle=handle,job=job,kind=kind)
        print(f'{now()} START {kind} {job["id"]} pid={process.pid}',flush=True)

    print(f'{now()} queue start: {len(jobs)} jobs, max {args.max_running} active workers, gamma first',flush=True)
    while True:
        for pid,item in list(active.items()):
            code=item['process'].poll()
            if code is None:
                continue
            item['handle'].close()
            job,kind=item['job'],item['kind']
            if code==0:
                valid=(native_complete(Path(job['run_dir'])) if kind=='fit'
                       else plots_complete(Path(job['case_dir']),job['model']))
                if not valid:
                    code=99
            job.update(last_returncode=code,stage_finished_utc=now())
            job.pop('pid',None)
            if code:
                job['phase']='failed'
                errors.append(dict(job=job['id'],kind=kind,exit_code=code,log=job['log']))
                print(f'{now()} FAILED {kind} {job["id"]} exit={code}; no new jobs will start',flush=True)
            else:
                job['phase']='plot_ready' if kind=='fit' else 'done'
                print(f'{now()} DONE {kind} {job["id"]}',flush=True)
            del active[pid]
        if not errors and not shutting_down:
            # A plot worker counts toward the same cap; never overlap corners.
            plot_jobs=[j for j in jobs if j['phase']=='plot_ready']
            fit_jobs=[j for j in jobs if j['phase']=='fit_ready']
            if plot_jobs and len(active)<args.max_running and not any(v['kind']=='plot' for v in active.values()):
                memory,_=available_memory_gib()
                if memory>=2.0:
                    start(plot_jobs[0],'plot')
            # Reserve the next freed slot for a waiting corner if memory is tight.
            waiting_plot=any(j['phase']=='plot_ready' for j in jobs) and not any(v['kind']=='plot' for v in active.values())
            while fit_jobs and len(active)<args.max_running and not waiting_plot:
                memory,_=available_memory_gib()
                if memory<.75:
                    break
                start(fit_jobs.pop(0),'fit')
        done=all(j['phase']=='done' for j in jobs)
        final='complete' if done else ('failed' if errors and not active else ('stopped' if shutting_down and not active else 'running'))
        snapshot(final)
        if final!='running':
            print(f'{now()} queue {final}',flush=True)
            return 0 if done else 1
        time.sleep(2)


if __name__=='__main__':
    sys.exit(main())
