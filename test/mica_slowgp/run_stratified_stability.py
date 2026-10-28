#!/usr/bin/env python3
"""Run the unchanged stability design with exact prior-stratified integration.

Uses the original scheduler's global process limit, input validation and
no-overwrite behavior. Every parent fit integrates all three center intervals.
The budget is per interval; no original-prior region is removed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys


def main(argv=None):
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--budget-settings', type=Path, required=True)
    extra.add_argument('--package-root', type=Path, required=True)
    extra.add_argument('--center-strata', type=float, nargs=4, required=True)
    options, remaining = extra.parse_known_args(argv)
    source = Path(__file__).with_name('run_stability_suite.py')
    spec = importlib.util.spec_from_file_location('slowgp_stratified_scheduler', source)
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    scheduler.PACKAGE = options.package_root.resolve()
    scheduler.ENVIRONMENT['PYTHONPATH'] = str(scheduler.PACKAGE)
    args = scheduler.parse_args(remaining)
    if args.sampler != 'dynesty':
        raise ValueError('Stratified integration requires --sampler dynesty')
    sys.path.insert(0, str(scheduler.PACKAGE))
    from mica_slowgp.nested import NestedBudget
    from mica_slowgp.stratified import SCHEME, strata_definition

    settings = json.loads(options.budget_settings.read_text())
    if not isinstance(settings, dict) or 'nlive' in settings:
        raise ValueError('Budget settings must be an object without nlive; use --budgets')
    original_budget = scheduler.budget_for

    def budget_for(sampler, effort):
        return asdict(NestedBudget(**{**original_budget(sampler, effort), **settings}))

    scheduler.budget_for = budget_for
    original_make_job = scheduler.make_job

    def make_job(args, case, mode, effort, seed):
        _, edges, masses = strata_definition(mode, args.inputs_root / 'priors.json',
                                             options.center_strata)
        job = original_make_job(args, case, mode, effort, seed)
        job.update(integration_scheme=SCHEME, center_strata=edges,
                   stratum_prior_masses=masses,
                   effort_definition='nlive per stratum; all strata integrated')
        job['command'] += ['--center-strata', *map(str, edges)]
        return job

    scheduler.make_job = make_job
    original_completion_error = scheduler.completion_error

    def completion_error(job, snapshot, priors):
        error = original_completion_error(job, snapshot, priors)
        if error is not None:
            return error
        folder = Path(job['output'])
        try:
            config = scheduler.read_json(folder / 'sampler_config.json')
            run_input = scheduler.read_json(folder / 'run_input.json')
            marker = scheduler.read_json(folder / 'completed.json')
            if config.get('integration_scheme') != SCHEME:
                return 'integration scheme differs'
            for field in ('center_strata', 'stratum_prior_masses'):
                if config.get(field) != job[field]:
                    return f'{field} differs'
            if run_input.get('center_strata') != job['center_strata']:
                return 'run_input center strata differ'
            if marker != dict(integration_scheme=SCHEME, number_of_strata=3):
                return 'stratified completion marker differs'
            if not (folder / 'stratified_results.npz').is_file():
                return 'missing raw stratified integration result'
            if json.dumps(config['budget'], sort_keys=True) != json.dumps(job['budget'], sort_keys=True):
                return 'budget values or numeric types differ'
        except (OSError, ValueError, KeyError, TypeError) as error:
            return f'invalid stratified completion: {error}'
        return None

    scheduler.completion_error = completion_error
    plan, snapshot, priors = scheduler.build_plan(args)
    plan.update(algorithm_settings_file=str(options.budget_settings.resolve()),
                algorithm_settings=settings, package_root=str(scheduler.PACKAGE),
                integration_scheme=SCHEME, center_strata=options.center_strata,
                effort_definition='nlive per stratum; all three strata integrated',
                preserves_original_prior=True)
    scheduler.print_summary(plan)
    if args.plan_json:
        scheduler.write_json(args.plan_json.resolve(), plan, exclusive=True)
    if args.run:
        return scheduler.execute(plan, snapshot, priors, args)
    return 2 if any(job['status'].startswith('blocked') for job in plan['jobs']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
