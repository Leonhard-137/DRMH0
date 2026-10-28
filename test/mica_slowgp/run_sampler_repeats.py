"""Two fixed-budget sampler-seed checks, without changing simulated data."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/leonhard/sci001/output/mica_slowgp')
BUDGET = Path(__file__).with_name('small_budget.json')


def one(case, seed):
    out = ROOT / 'repeats' / case
    out.parent.mkdir(exist_ok=True)
    if (out / 'sample.txt').exists():
        raise FileExistsError(out)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', UCX_TLS='self', UCX_NET_DEVICES='lo')
    cmd = [sys.executable, '-B', '-m', 'mica_slowgp.cli', 'fit',
           '--driver', str(ROOT / 'small_cases' / case / 'w2.txt'),
           '--response', str(ROOT / 'small_cases' / case / 'response.txt'),
           '--mode', case, '--priors', str(ROOT / 'small_cases/priors.json'),
           '--out', str(out), '--budget', str(BUDGET), '--seed', str(seed)]
    with (out.parent / f'{case}.log').open('w') as f:
        result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Repeat failed: {case}, see log')
    print(json.dumps(dict(case=case, seed=seed, status='complete')), flush=True)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(one, 'independent', 91243), pool.submit(one, 'shared', 91255)]
        for job in jobs:
            job.result()
