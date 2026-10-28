"""Independent dynesty backend for the same pairwise likelihood and priors.

Exchangeable Gaussian components are represented without a hard ordering
boundary in the unit cube, then sorted as complete triples. This is exactly
the normalized ordered prior, not an extra physical restriction.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.special import logsumexp

from .sampler import load_priors, parameter_names, weighted_quantile


class PriorTransform:
    def __init__(self, mode='independent', priors=None):
        self.names = parameter_names(mode)
        self.priors = load_priors(priors, mode)
        self.bounds = np.array([self.priors[k]['bounds'] for k in self.names])
        self.logscale = np.array([self.priors[k]['kind'] == 'log_uniform' for k in self.names])
        self.uniform_bounds = self.bounds.copy()
        self.uniform_bounds[self.logscale] = np.log(self.uniform_bounds[self.logscale])
        self.exchangeable = all(self.priors[a] == self.priors[b]
                                for a, b in [('A0', 'A1'), ('mu0', 'mu1'), ('w0', 'w1')])

    def __call__(self, u):
        u = np.asarray(u)
        x = self.uniform_bounds[:, 0] + u * np.diff(self.uniform_bounds, axis=1)[:, 0]
        x[self.logscale] = np.exp(x[self.logscale])
        if self.exchangeable:
            if x[3] > x[6]:
                x[2:5], x[5:8] = x[5:8].copy(), x[2:5].copy()
            return x
        # Inverse marginal CDF of mu0 over the ordered bounded rectangle.
        a, b = self.bounds[3]; c, d = self.bounds[6]
        flat_end = min(b, c)
        flat_area = max(0., flat_end-a) * (d-c)
        start, end = max(a, c), min(b, d)
        slope_area = max(0., end-start) * (d-(start+end)/2)
        area = u[3] * (flat_area+slope_area)
        if flat_area > 0 and area <= flat_area:
            x[3] = a + area/(d-c)
        else:
            v = area-flat_area; h = d-start
            x[3] = start + 2*v/(h+np.sqrt(max(0., h*h-2*v)))
        low = max(c, x[3])
        x[6] = low + u[6]*(d-low)
        return x


class VectorLikelihood:
    """Count actual callback entries in the checkpoint-retained trajectory.

    This counter is exact for this backend's serial, queue_size=1 execution.
    A checkpoint cannot recover work discarded after its last saved state.
    """
    def __init__(self, callback, names):
        self.callback, self.names = callback, names
        self.exact_calls = 0
        self.counted_from_start = True

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'exact_calls' not in state:
            # Legacy checkpoints did not instrument the callback. Count only
            # their future suffix; never invent an exact historical total.
            self.exact_calls = 0
            self.counted_from_start = False
        elif 'counted_from_start' not in state:
            self.counted_from_start = False

    def __call__(self, x):
        self.exact_calls += 1
        return self.callback(dict(zip(self.names, x)))


@dataclass(frozen=True)
class NestedBudget:
    nlive: int = 400
    dlogz: float = 0.05
    bound: str = 'multi'
    sample: str = 'rwalk'
    walks: int = 50
    bootstrap: int = 0
    checkpoint_every: float = 30.0
    slices: int | None = None
    update_interval: int | float | None = None
    facc: float = 0.5

    def __post_init__(self):
        for name, minimum in (('nlive', 50), ('walks', 1), ('bootstrap', 0)):
            value = getattr(self, name)
            if not np.isfinite(value) or int(value) != value or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        for name in ('dlogz', 'checkpoint_every'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.slices is not None:
            if (isinstance(self.slices, bool) or not isinstance(self.slices, int)
                    or self.slices < 1):
                raise ValueError('slices must be a positive integer or None')
        if self.update_interval is not None:
            value = self.update_interval
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError('update_interval must be a positive finite int, float, or None')
        if (isinstance(self.facc, bool) or not isinstance(self.facc, (int, float))
                or not np.isfinite(self.facc) or not 0 < self.facc <= 1):
            raise ValueError('facc must lie in (0, 1]')


def _normalized_resume_config(config):
    """Add only new legacy defaults; retain unknown fields and interval units."""
    normalized = dict(config)
    budget = dict(config['budget'])
    for key, default in dict(slices=None, update_interval=None, facc=0.5).items():
        budget.setdefault(key, default)
    # Validate without filling any other omitted legacy field.
    NestedBudget(**budget)
    interval = budget['update_interval']
    if interval is not None:
        # 50 == 50.0 in Python, but these mean 50 versus 50*nlive in dynesty.
        budget['update_interval'] = dict(
            type='int' if isinstance(interval, int) else 'float', value=interval)
    normalized['budget'] = budget
    return normalized


def _sampler_kwargs(budget, ndim, seed):
    """Pass only applicable tuning options; retain previous auto behavior.

    In dynesty 3.1.0, auto constructs a sampler instance before processing
    walks/slices. Consequently the old adapter's walks=50 was ignored for
    auto: its 11-dimensional RWalkSampler uses ndim+20=31 steps. Leaving
    inactive tuning keywords out retains that behavior.
    """
    kwargs = dict(nlive=budget.nlive, bound=budget.bound, sample=budget.sample,
                  bootstrap=budget.bootstrap, rstate=np.random.default_rng(seed))
    if budget.sample == 'rwalk':
        kwargs.update(walks=budget.walks, facc=budget.facc)
    elif budget.sample in ('slice', 'rslice'):
        default = 3 if budget.sample == 'slice' else ndim + 3
        kwargs['slices'] = default if budget.slices is None else budget.slices
    if budget.update_interval is not None:
        kwargs['update_interval'] = budget.update_interval
    return kwargs


def _counter_for_sampler(sampler):
    """Retrieve the restored callback, not a newly constructed unused object."""
    if sampler.pool is not None or sampler.queue_size != 1:
        raise RuntimeError('Exact callback counting currently requires pool=None and queue_size=1')
    try:
        counter = sampler.loglikelihood.loglikelihood.func
    except AttributeError as error:
        raise RuntimeError('Unsupported dynesty likelihood wrapper structure') from error
    if not isinstance(counter, VectorLikelihood):
        raise RuntimeError('Checkpoint does not contain this backend VectorLikelihood')
    return counter


def _effective_sampler_settings(sampler):
    # During initial unit-cube draws internal_sampler is UnitCubeSampler.
    # internal_sampler_next stores the configured method used after that phase.
    internal = sampler.internal_sampler_next
    options = internal.sampler_kwargs
    tuning = {key: int(options[key]) for key in ('walks', 'slices') if key in options}
    if hasattr(internal, 'facc'):
        tuning['facc'] = float(internal.facc)
    return dict(internal_sampler_class=type(internal).__name__, tuning=tuning,
                bound_update_interval_reported_calls=int(sampler.bound_update_interval),
                first_update_min_ncall=int(sampler.first_bound_update_ncall),
                first_update_min_eff=float(sampler.first_bound_update_eff),
                queue_size=int(sampler.queue_size), pool=None)


def run_nested(log_likelihood, output_dir, mode='independent', priors=None,
               budget=None, seed=12345, resume=False):
    """Run or resume a checkpoint with the same importable, pickleable callback.

    Dynesty restores the callback stored in its checkpoint, so matching only
    priors and sampler options is insufficient. The initial callback payload
    fingerprint also binds the fixed likelihood data/settings on resume.
    Keep its implementation and environment unchanged while resuming a run.
    """
    import dynesty
    budget = NestedBudget() if budget is None else (NestedBudget(**budget) if isinstance(budget, dict) else budget)
    if not isinstance(seed, (int, np.integer)) or seed < 0 or seed > 2**32-1:
        raise ValueError('seed must be a uint32 integer')
    try:
        callback_payload = pickle.dumps(log_likelihood, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PickleError, AttributeError, TypeError) as error:
        raise TypeError('Dynesty checkpointing requires a pickleable callback; use an importable module-level callable or functools.partial, not a lambda/local function') from error
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'weighted_posterior.csv').exists():
        raise FileExistsError(folder)
    transform = PriorTransform(mode, priors)
    checkpoint = folder/'dynesty.checkpoint'
    config = dict(sampler='dynesty', sampler_version=dynesty.__version__, mode=mode,
                  parameter_names=list(transform.names), priors=transform.priors,
                  budget=asdict(budget), seed=int(seed),
                  likelihood_pickle_sha256=hashlib.sha256(callback_payload).hexdigest(),
                  likelihood_pickle_protocol=pickle.HIGHEST_PROTOCOL,
                  center_prior='uniform rectangle conditioned on mu0 < mu1',
                  parameterization='sort whole exchangeable triples' if transform.exchangeable else 'ordered marginal/conditional inverse CDF')
    if resume:
        previous = json.loads((folder/'sampler_config.json').read_text())
        if _normalized_resume_config(previous) != _normalized_resume_config(config):
            raise ValueError('Resume requires the exact saved sampler configuration')
        sampler = dynesty.NestedSampler.restore(str(checkpoint))
    else:
        if checkpoint.exists():
            raise FileExistsError('A checkpoint exists; use resume with the same configuration')
        (folder/'sampler_config.json').write_text(json.dumps(config, indent=2))
        sampler = dynesty.NestedSampler(VectorLikelihood(log_likelihood, transform.names), transform,
                    len(transform.names), **_sampler_kwargs(budget, len(transform.names), seed))
    counter = _counter_for_sampler(sampler)
    invocation_start_count = counter.exact_calls if resume else 0
    effective_settings = _effective_sampler_settings(sampler)
    start = time.time()
    sampler.run_nested(dlogz=budget.dlogz, print_progress=False,
                       checkpoint_file=str(checkpoint), checkpoint_every=budget.checkpoint_every,
                       resume=resume)
    result = sampler.results
    weights = np.exp(result.logwt-logsumexp(result.logwt))
    np.savetxt(folder/'weighted_posterior.csv',
               np.column_stack([result.samples, result.logl, result.logvol, weights]),
               delimiter=',', header=','.join((*transform.names, 'log_likelihood', 'log_prior_mass', 'weight')), comments='')
    np.savez_compressed(folder/'nested_results.npz', samples=result.samples, samples_u=result.samples_u,
                        logl=result.logl, logwt=result.logwt, logvol=result.logvol,
                        logz=result.logz, logzerr=result.logzerr, ncall=result.ncall)
    summary = {name:dict(zip(['q16','q50','q84'], map(float,weighted_quantile(result.samples[:,i],weights))))
               for i,name in enumerate(transform.names)}
    diagnostics = dict(sampler='dynesty', raw_sample_count=len(weights),
                       weight_ess=float(1/np.sum(weights**2)), max_posterior_weight=float(weights.max()),
                       weight_ess_is_not_autocorrelation_ess=True, likelihood_calls=int(np.sum(result.ncall)),
                       likelihood_calls_definition='legacy alias of dynesty_reported_ncall, not exact callback calls',
                       dynesty_reported_ncall=int(np.sum(result.ncall)),
                       dynesty_state_ncall=int(sampler.ncall),
                       exact_likelihood_calls_total=int(counter.exact_calls) if counter.counted_from_start else None,
                       exact_calls_since_instrumentation=int(counter.exact_calls),
                       exact_calls_this_invocation=int(counter.exact_calls-invocation_start_count),
                       exact_count_includes_full_history=bool(counter.counted_from_start),
                       exact_count_scope='checkpoint-retained sampler trajectory; includes initialization on a fresh run; excludes discarded post-checkpoint work',
                       sampler_effective=effective_settings,
                       elapsed_seconds=time.time()-start, raw_log_evidence=float(result.logz[-1]),
                       elapsed_seconds_scope='run_nested call in this invocation, excluding initialization or restore; reported counts include checkpoint history',
                       resumed_from_checkpoint=bool(resume),
                       raw_log_evidence_error=float(result.logzerr[-1]),
                       stopping_rule='remaining evidence estimate dlogz; independent seed/budget validation still required',
                       convergence_status='not_established_pending_stability_gate', evidence_for_model_comparison=False)
    (folder/'posterior_summary.json').write_text(json.dumps(summary, indent=2))
    (folder/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    return dict(summary=summary, diagnostics=diagnostics, weighted_posterior=str(folder/'weighted_posterior.csv'))


def load_saved_posterior(folder, names):
    folder = Path(folder)
    config = json.loads((folder/'sampler_config.json').read_text())
    if config.get('sampler') != 'dynesty':
        from .sampler import read_weighted_posterior
        return read_weighted_posterior(folder, names)
    table = np.genfromtxt(folder/'weighted_posterior.csv', names=True, delimiter=',')
    samples = np.column_stack([table[name] for name in names])
    weights = np.atleast_1d(table['weight'])
    logl, logvol = np.atleast_1d(table['log_likelihood']), np.atleast_1d(table['log_prior_mass'])
    if (not np.isfinite(samples).all() or not np.isfinite(weights).all()
            or not np.isfinite(logl).all() or not np.isfinite(logvol).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('Invalid saved weighted posterior')
    return dict(samples=samples, weights=weights/weights.sum(),
                log_likelihood=logl, log_prior_mass=logvol)
