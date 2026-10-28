#!/usr/bin/env python3
"""Use the existing bounded scheduler with explicit algorithm settings.

The original v1 scheduler remains unchanged. This wrapper adds only
--budget-settings JSON and --package-root PATH. All ordinary scheduler flags,
dry-run defaults, exact input checks and the global four-fit limit are reused.
The package root must contain mica_slowgp/ (usually a frozen snapshot's src/).
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
    extra.add_argument('--budget-settings', type=Path)
    extra.add_argument('--package-root', type=Path)
    options, remaining = extra.parse_known_args(argv)

    source = Path(__file__).with_name('run_stability_suite.py')
    spec = importlib.util.spec_from_file_location('slowgp_stability_scheduler', source)
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    if options.package_root:
        scheduler.PACKAGE = options.package_root.resolve()
        scheduler.ENVIRONMENT['PYTHONPATH'] = str(scheduler.PACKAGE)
    args = scheduler.parse_args(remaining)
    # Validate and canonicalize the complete budget with the selected code.
    sys.path.insert(0, str(scheduler.PACKAGE))
    from mica_slowgp.sampler import SamplerBudget
    from mica_slowgp.nested import NestedBudget
    settings = (json.loads(options.budget_settings.read_text())
                if options.budget_settings else {})
    if not isinstance(settings, dict):
        raise ValueError('--budget-settings must contain a JSON object')
    effort_field = 'max_num_saves' if args.sampler == 'dnest' else 'nlive'
    if effort_field in settings:
        raise ValueError(f'Set {effort_field} through --budgets, not --budget-settings')
    original_budget = scheduler.budget_for

    def budget_for(sampler, effort):
        budget = {**original_budget(sampler, effort), **settings}
        cls = SamplerBudget if sampler == 'dnest' else NestedBudget
        return asdict(cls(**budget))

    scheduler.budget_for = budget_for
    original_completion_error = scheduler.completion_error

    def completion_error(job, snapshot, priors):
        error = original_completion_error(job, snapshot, priors)
        if error is not None:
            return error
        saved = scheduler.read_json(Path(job['output']) / 'sampler_config.json')
        # dynesty interprets an integer interval as an absolute count but a
        # floating interval as a multiple of nlive. Dict equality loses this.
        if json.dumps(saved['budget'], sort_keys=True) != json.dumps(job['budget'], sort_keys=True):
            return 'budget values or numeric types differ'
        return None

    scheduler.completion_error = completion_error
    plan, snapshot, priors = scheduler.build_plan(args)
    plan['algorithm_settings_file'] = (str(options.budget_settings.resolve())
                                       if options.budget_settings else None)
    plan['algorithm_settings'] = settings
    plan['package_root'] = str(scheduler.PACKAGE)
    scheduler.print_summary(plan)
    if args.plan_json:
        scheduler.write_json(args.plan_json.resolve(), plan, exclusive=True)
    if args.run:
        return scheduler.execute(plan, snapshot, priors, args)
    return 2 if any(job['status'].startswith('blocked') for job in plan['jobs']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
