from pathlib import Path
import os,signal,subprocess,time,sys
root=Path('/home/leonhard/sci001')
base=root/'test/mica_recovery/longterm_test'
out=base/'cases_uonly_gp_highpass'
env=os.environ.copy()
env.update(PATH='/home/leonhard/miniforge3/envs/mica2/bin:'+env['PATH'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR='/tmp',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
env.pop('TMP',None);env.pop('TEMP',None)
jobs=[]
for s in (20,40):
 log=(out/f'gaussian_s{s}.log').open('a')
 cmd=[sys.executable,'-B','-u',str(base/'fit.py'),'--config',str(base/'experiment_uonly_gp_highpass.yaml'),'--case',f'uonly_gp_raw_u_s{s}','--model','gaussian','--run','--plot-results']
 p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 jobs.append((p,log));print('Started Gaussian',s,p.pid,flush=True)
codes=[]
for p,log in jobs:
 codes.append(p.wait());log.close()
# The original Gamma controller retains its two active fits and plotting.
while Path('/proc/5704/stat').exists():
 state=Path('/proc/5704/stat').read_text().split(') ')[1].split()[0]
 if state=='Z':break
 time.sleep(10)
# Retire only the stopped old shell so it cannot launch Gaussian a second time.
try:os.kill(5703,signal.SIGKILL)
except ProcessLookupError:pass
print('Gaussian exit codes',codes,flush=True)
print('All controllers exited; inspect model states and summaries for completion.',flush=True)
sys.exit(any(codes))
