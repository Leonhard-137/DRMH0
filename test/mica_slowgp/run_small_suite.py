"""Run only the small, matched, pairwise test suite (two jobs at most)."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--budget', type=Path, default=Path(__file__).with_name('small_budget.json'))
    p.add_argument('--workers', type=int, default=2, choices=[1, 2])
    args = p.parse_args()
    root = args.root.resolve()
    (root / 'runs').mkdir(exist_ok=True)
    cases = ['clean', 'response_only', 'independent', 'shared']
    modes = ['none', 'independent', 'shared']
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', UCX_TLS='self', UCX_NET_DEVICES='lo')

    def run_one(case, mode, seed):
        output = root / 'runs' / f'{case}_{mode}'
        if (output / 'diagnostics.json').exists():
            return dict(case=case, mode=mode, status='existing_complete')
        if (output / 'sample.txt').exists():
            raise RuntimeError(f'Incomplete run retained; inspect before restarting: {output}')
        input_dir = root / 'small_cases' / case
        cmd = [sys.executable, '-B', '-m', 'mica_slowgp.cli', 'fit',
               '--driver', str(input_dir / 'w2.txt'), '--response', str(input_dir / 'response.txt'),
               '--mode', mode, '--priors', str(root / 'small_cases/priors.json'),
               '--out', str(output), '--budget', str(args.budget), '--seed', str(seed)]
        started = time.time()
        with (root / 'runs' / f'{case}_{mode}.log').open('w') as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        record = dict(case=case, mode=mode, exit_code=result.returncode, elapsed_seconds=time.time()-started)
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise RuntimeError(f'Fit failed: {case}/{mode}; inspect its log')
        return record

    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, case, mode, 61022 + i*10+j)
                   for i, case in enumerate(cases) for j, mode in enumerate(modes)]
        for f in as_completed(futures):
            results.append(f.result())
            (root / 'suite_progress.json').write_text(json.dumps(results, indent=2))
    (root / 'suite_timing.json').write_text(json.dumps(dict(elapsed_seconds=time.time()-started,
                                                          workers=args.workers, runs=results), indent=2))


if __name__ == '__main__':
    main()
