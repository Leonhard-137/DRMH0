"""Isolated toy checks for two proposed ridge moves; no scientific sampler run.

Nothing imports or modifies mica_slowgp.sampler. Bounds and prior types match
the currently declared experimental prior. All boundary exits are rejections,
not draws from a state-dependent truncated proposal distribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kstest


SPECS = {
    'ou': dict(names=['sigma_x', 'tau_x'], bounds=[[.1, 3.], [.3, 40.]],
               logscale=[True, True], powers=[1., 2.], prior_powers=[-1., -1.],
               logh_coefficient=0., tilt=[1.5, -1.1]),
    'slow': dict(names=['qW', 'qb', 'Ts'], bounds=[[0., 2.5], [0., 2.5], [5., 120.]],
                 logscale=[False, False, True], powers=[1., 1., 1.],
                 prior_powers=[0., 0., -1.], logh_coefficient=2., tilt=[.9, -1.3, .7]),
}


def coordinates(spec):
    bounds = np.asarray(spec['bounds'], dtype=float)
    logged = np.asarray(spec['logscale'])
    transformed = bounds.copy()
    transformed[logged] = np.log(transformed[logged])
    return bounds, logged, transformed[:, 0], np.diff(transformed, axis=1)[:, 0]


def from_unit(u, spec):
    _, logged, low, width = coordinates(spec)
    x = low+u*width
    x[:, logged] = np.exp(x[:, logged])
    return x


def to_unit(x, spec):
    _, logged, low, width = coordinates(spec)
    value = x.copy()
    value[:, logged] = np.log(value[:, logged])
    return (value-low)/width


def mapped(x, s, spec):
    return x*np.exp(s[:, None]*np.asarray(spec['powers']))


def invariance_check(spec, target, rng, n, wrong_hastings=False):
    u = rng.random((n, len(spec['names'])))
    k = np.asarray(spec['tilt'])
    if target == 'tilted':
        u = np.log1p(u*np.expm1(k))/k
    elif target == 'hard_level':
        # Exact conditional prior on U0+U1>1, directly matching a nested level.
        u[:, 0] = np.sqrt(u[:, 0])
        u[:, 1] = 1-u[:, 0]+u[:, 0]*u[:, 1]
    x = from_unit(u, spec)
    s = rng.normal(0., .55, n)  # State independent and symmetric, no redraws.
    candidate = mapped(x, s, spec)
    bounds, _, _, _ = coordinates(spec)
    allowed = np.all((candidate >= bounds[:, 0]) & (candidate <= bounds[:, 1]), axis=1)
    logh = np.zeros(n) if wrong_hastings else spec['logh_coefficient']*s
    uc = to_unit(candidate, spec)
    if target == 'tilted':
        logh = logh+(uc-u)@k
    elif target == 'hard_level':
        allowed &= uc[:, 0]+uc[:, 1] > 1.
    accepted = allowed & (np.log(rng.random(n)) < np.minimum(0., logh))
    y = x.copy()
    y[accepted] = candidate[accepted]
    v = to_unit(y, spec)
    # Exact independent probability-integral transforms under each test target.
    def target_cdf(z):
        z = z.copy()
        if target == 'tilted':
            z = np.expm1(k*z)/np.expm1(k)
        elif target == 'hard_level':
            first = z[:, 0].copy()
            z[:, 0] = first**2
            z[:, 1] = (z[:, 1]-1+first)/first
        return z
    pit = target_cdf(v)
    prior_pit = target_cdf(u)
    mean_z = abs(pit.mean(axis=0)-.5)*np.sqrt(12*n)
    ks = [kstest(pit[:, j], 'uniform').statistic for j in range(pit.shape[1])]
    f, g = prior_pit-.5, pit-.5
    asymmetry_z = []
    for i in range(f.shape[1]):
        for j in range(i+1, f.shape[1]):
            term = f[:, i]*g[:, j]-f[:, j]*g[:, i]
            se = term.std(ddof=1)/np.sqrt(n)
            asymmetry_z.append(float(abs(term.mean())/se) if se else 0.)
    passed = bool(max(mean_z) < 6 and max(ks) < .015 and max(asymmetry_z) < 6)
    return dict(target=target, n=n, acceptance_fraction=float(accepted.mean()),
                boundary_or_level_rejection_fraction=float(1-allowed.mean()),
                uniform_mean_z=mean_z.tolist(), maximum_ks_D=float(max(ks)),
                maximum_detailed_balance_moment_z=max(asymmetry_z), passed=passed)


def mapping_check(spec, rng):
    x = from_unit(rng.random((300, len(spec['names']))), spec)
    s = rng.normal(0., .2, len(x))
    y = mapped(x, s, spec)
    back = mapped(y, -s, spec)
    inverse_error = float(np.max(abs(back-x)/np.maximum(abs(x), 1e-100)))
    powers = np.asarray(spec['powers'])
    prior_powers = np.asarray(spec['prior_powers'])
    # The augmented involution (x,s)->(T_s(x),-s) has |det|=exp(sum(p_i)*s).
    log_jacobian = powers.sum()*s
    log_prior_ratio = (np.log(y)-np.log(x))@prior_powers
    logh_error = float(np.max(abs(log_jacobian+log_prior_ratio-spec['logh_coefficient']*s)))
    jacobian_error = 0.
    for values, aux in zip(x[:30], s[:30]):
        z = np.r_[values, aux]
        def involution(v):
            return np.r_[v[:-1]*np.exp(powers*v[-1]), -v[-1]]
        jac = np.empty((len(z), len(z)))
        for j in range(len(z)):
            h = 1e-6*max(1., abs(z[j])); plus=z.copy(); minus=z.copy()
            plus[j]+=h; minus[j]-=h
            jac[:, j]=(involution(plus)-involution(minus))/(2*h)
        expected=np.exp(powers.sum()*aux)
        jacobian_error=max(jacobian_error, abs(abs(np.linalg.det(jac))/expected-1))
    assert inverse_error < 2e-14 and logh_error < 2e-13 and jacobian_error < 1e-7
    return dict(max_relative_inverse_error=inverse_error,
                max_log_hastings_identity_error=logh_error,
                max_relative_finite_difference_jacobian_error=float(jacobian_error), passed=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n', type=int, default=120000)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise FileExistsError(args.output)
    rng=np.random.default_rng(2026102208)
    result={'seed':2026102208, 'proposal_s':'N(0, 0.55^2), symmetric and independent of state',
            'scope':'Candidate toy tests only; no scientific likelihood, sampler integration, or fit',
            'specifications':SPECS, 'checks':{}}
    for name,spec in SPECS.items():
        entries={'mapping':mapping_check(spec,rng),
                 'targets':[invariance_check(spec,t,rng,args.n) for t in ('prior','hard_level','tilted')]}
        assert all(v['passed'] for v in entries['targets'])
        result['checks'][name]=entries
    negative=invariance_check(SPECS['slow'],'prior',rng,args.n,wrong_hastings=True)
    assert not negative['passed'], 'Negative control failed to detect the omitted Jacobian correction'
    result['negative_control_slow_logH_zero']=negative
    result['passed']=True
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
