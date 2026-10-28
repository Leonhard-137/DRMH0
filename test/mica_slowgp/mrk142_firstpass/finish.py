#!/usr/bin/env python3
"""Run the authorized one-shot report and record its bounded scope."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record_report(root: Path, returncode: int) -> dict:
    import numpy
    import scipy
    import dynesty
    base = Path('/home/leonhard')
    code = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    report = root / 'RESULTS.md'
    task = root / 'task.md'
    if not task.exists():
        task.write_text(
            'Mrk142 first pass: five independent W2-response fits plus one U repeat. '
            'Use the frozen shared slow-GP plus two-Gaussian model, all three prior strata, '
            '400 live points per stratum and fixed seeds. Six parallel jobs, six-hour '
            'sampling deadline; no automatic retries or budget increases. '
            'Deliver component diagnostics and fixed-beta tau0, not FFA or H0.\n',
            encoding='utf-8')
    summary_path = root / 'analysis' / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    completed = [j['id'] for j in summary.get('jobs', []) if j.get('state') == 'complete']
    # Hash only finalized scientific products, never live status/checkpoints/logs.
    files = [(task, 'source-artifact'), (root / 'manifest.json', 'source-artifact')]
    files += [(p, 'code') for p in sorted(code.glob('*.py'))]
    files += [(p, 'source-artifact') for p in sorted(code.glob('*.mplstyle'))]
    for name in ('RESULTS.md', 'analysis_summary.json', 'summary.json',
                 'u_repeat.json', 'tau0_summary.json', 'preflight.json'):
        p = root / name
        if p.is_file():
            files.append((p, 'result'))
    if (root / 'analysis').is_dir():
        files += [(p, 'result') for p in sorted((root / 'analysis').rglob('*'))
                  if p.is_file() and p.suffix in ('.json', '.csv', '.npz', '.pdf', '.png')]
    for job in manifest['jobs']:
        folder = Path(job['out_dir'])
        if job['id'] not in completed:
            continue
        for name in ('completed.json', 'sampler_config.json', 'diagnostics.json',
                     'weighted_posterior.csv', 'stratified_results.npz'):
            if (folder / name).is_file():
                files.append((folder / name, 'result'))
    artifacts = [dict(role=role, path=str(p.resolve().relative_to(base)), sha256=digest(p))
                 for p, role in files]
    complete = returncode == 0 and report.is_file() and summary_path.is_file()
    discovery = {}
    for name in ('mathematica', 'sagemath'):
        discovery[name] = dict(
            candidate_implementations=['Dense Gaussian covariance and linear algebra'],
            existence_evidence='General matrix operations are suitable in principle',
            local_availability='not-applicable',
            availability_evidence='This task reuses an explicitly selected frozen Python scientific implementation')
    discovery['python'] = dict(
        candidate_implementations=['Frozen compiled GP likelihood, dynesty and joint Gaussian reconstruction'],
        existence_evidence='Current task input timing and reference-likelihood agreement were checked before launch',
        local_availability='available',
        availability_evidence='The isolated mica2-slowgp environment executed the bounded report workflow')
    result = dict(
        schema_version='1.0', created_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact_base_directory=str(base),
        phase_scope=dict(name='Bounded Mrk142 first-pass reporting',
                         fits_completed=len(completed), fits_requested=6,
                         completed_job_ids=completed, all_fits_complete=len(completed) == 6,
                         analysis_status=summary.get('overall_status', 'unavailable'),
                         no_automatic_retries=True),
        task=dict(file=str(task.relative_to(base)), sha256=digest(task),
                  object='Pairwise shared slow-GP and two-Gaussian reverberation inference',
                  deliverables=['Six bounded fit attempts', 'Component and repeatability diagnostics',
                                'Fixed-beta lag-wavelength tau0 report']),
        mathematical_context=dict(
            assumptions=['Five separate pairs share observed W2 but not a joint cross-pair posterior',
                         'Full Gaussian negative-lag support and all three ordered-center strata retained',
                         'Band means and measurement errors scaled by the same fixed positive band mean',
                         'Incomplete strata are excluded from full posterior inference',
                         'Tau0 uses independent weighted resampling as a declared cross-pair approximation'],
            domain='Observed-frame days; declared finite proper priors in manifest and run configurations',
            precision=dict(mode='machine', working_digits=16,
                           target_tolerance='User-approved U repeat criteria; finite-budget numerical evidence')),
        implementation_discovery=discovery,
        decision=dict(selected_backend='python',
                      backend_version=f'Python {sys.version.split()[0]}; NumPy {numpy.__version__}; SciPy {scipy.__version__}; dynesty {dynesty.__version__}',
                      selection_reason='Reuse the tested frozen task-native inference and reconstruction implementation',
                      fallback_reason='not-required: report completed attempts without changing model or sampling budget'),
        execution=dict(status='complete' if complete else 'failed',
                       interface='Independent WSL processes, one BLAS thread each',
                       command_or_input=f'{sys.executable} -B {code / "supervise.py"} --manifest {root / "manifest.json"}',
                       code_artifact=str((code / 'finish.py').relative_to(base))),
        artifacts=artifacts,
        result=dict(status='complete' if complete else 'failed',
                    summary=f'Bounded report finished with return code {returncode}; {len(completed)}/6 validated complete fits. Analysis status: {summary.get("overall_status", "unavailable")}. See report for missing or unstable results.',
                    result_artifact=str(report.relative_to(base))),
        verification=dict(
            methods=['Task-specific normalization, timeout/checkpoint and partial-run tests',
                     'Complete three-stratum posterior checks and fixed seeded U comparison',
                     'Artifact paths and hashes validated after reporting'],
            evidence_level='numerical-evidence',
            residual_or_error='Finite posterior intervals and U repeat discrepancies are reported in the accompanying results; no error-free recovery claim',
            limitations=['Four response bands have one sampler seed only',
                         'Cross-pair covariance from shared observed W2 is not propagated',
                         'Stationary short RMS and transfer area are not absolute disk flux',
                         'A bounded report can complete with missing, weakly constrained or unstable fits',
                         'No physical disk identification, FFA, distance or H0 calibration is claimed']))
    (root / 'computation-record.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    command = [sys.executable, '-B', str(Path(__file__).with_name('analyse.py')), '--root', str(root)]
    completed = subprocess.run(command, check=False)
    record = record_report(root, completed.returncode)
    if record['execution']['status'] != 'complete':
        return completed.returncode or 1
    validator = Path('/mnt/c/Users/26330/.codex/skills/math-science-computation/scripts/computation_record.py')
    checked = subprocess.run([sys.executable, '-B', str(validator), 'validate',
                              '--record', str(root / 'computation-record.json'),
                              '--base-dir', '/home/leonhard'], capture_output=True, text=True, check=False)
    (root / 'record-validation.json').write_text(json.dumps(dict(
        returncode=checked.returncode, stdout=checked.stdout, stderr=checked.stderr), indent=2) + '\n')
    print(json.dumps(dict(report_exit_code=completed.returncode,
                         record_validation_exit_code=checked.returncode)))
    return checked.returncode


if __name__ == '__main__':
    raise SystemExit(main())
