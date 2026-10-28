#!/usr/bin/env python3
"""Run the two matched no-BLR fits concurrently, then plot serially."""
from pathlib import Path
import datetime, json, os, subprocess, sys

ROOT=Path(__file__).resolve().parent
CASE=ROOT/'cases/abl_disk_uonly_gp_raw'
CONFIG=ROOT/'experiment_disk_uonly_gp_raw.yaml'
LOG=CASE/'logs/pipeline.log'
STATUS=CASE/'logs/pipeline_status.json'
ENV=os.environ.copy()
ENV.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
state={'case':str(CASE),'config':str(CONFIG),'pid':os.getpid(),
       'max_parallel_fits':2,'mpi_per_fit':2}
def status(phase, **extra):
    state.update(phase=phase,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    tmp=STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2));tmp.replace(STATUS)

def main():
    selection=['--config',str(CONFIG),'--suite','paired_ablation','--case',CASE.name,'--model','both']
    commands=[
        ('fitting',[sys.executable,'-u',str(ROOT/'fit.py'),*selection,'--batch-size','2','--run']),
        ('plotting',[sys.executable,'-u',str(ROOT/'plot_results.py'),*selection,'--plots','transfer','lightcurves','corner'])]
    with LOG.open('a',buffering=1) as log:
        for phase,command in commands:
            status(phase,command=command)
            log.write('\n'+state['updated_at']+' '+phase+'\n')
            log.write('COMMAND '+json.dumps(command)+'\n')
            code=subprocess.run(command,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:
                status('failed',failed_phase=phase,returncode=code)
                log.write(f'Failed {phase}: exit {code}\n')
                return code
        for model in ['gamma','gaussian']:
            run=CASE/f'runs/mica/2comp/run_UVW2_to_U_2comp_{model}'
            assert json.loads((run/'run_mica_state.json').read_text())['status']=='complete'
            results=CASE/'results'/model
            json.loads((results/'summary.json').read_text())
            for name in ['transfer_function','light_curves','corner']:
                for ext in ['jpg','pdf']:
                    assert (results/f'{name}.{ext}').stat().st_size>100
        status('complete',returncode=0)
        log.write('Verified both native fits and all requested result artifacts.\n')
    return 0

if __name__=='__main__':
    try:
        code=main()
    except Exception as error:
        status('failed',error=repr(error))
        raise
    raise SystemExit(code)
