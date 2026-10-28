"""Small matched injections, generated independently of the fitted covariance code.

No existing source data or outputs are read or overwritten. All four cases share
the same OU realization, Gaussian responses, cadence, and measurement noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.ndimage import gaussian_filter1d


def stationary_ou(rng, n, dt, sigma, tau):
    x = np.empty(n)
    x[0] = rng.normal(scale=sigma)
    a = np.exp(-dt / tau)
    noise = rng.normal(scale=sigma * np.sqrt(-np.expm1(-2 * dt / tau)), size=n - 1)
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise[i - 1]
    return x


def stationary_matern32(rng, n, dt, timescale):
    # Exact state-space transition for k(d)=(1+|d|/T)exp(-|d|/T).
    p = np.diag([1.0, 1.0 / timescale**2])
    f = expm(np.array([[0.0, 1.0], [-1 / timescale**2, -2 / timescale]]) * dt)
    q = p - f @ p @ f.T
    values, vectors = np.linalg.eigh((q + q.T) / 2)
    assert values.min() > -1e-12
    factor = vectors * np.sqrt(np.maximum(values, 0))
    state = rng.normal(size=2) * np.sqrt(np.diag(p))
    out = np.empty(n)
    for i in range(n):
        out[i] = state[0]
        state = f @ state + factor @ rng.normal(size=2)
    return out


def generate(out: Path, seed=20261022, points=42):
    out.mkdir(parents=True, exist_ok=False)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    rng_x, rng_sw, rng_sb, rng_t, rng_n = streams
    truth_params = dict(sigma_x=1.0, tau_x=3.0, A0=0.8, mu0=2.5, w0=0.45,
                        A1=0.3, mu1=12.0, w1=2.0, qW=0.8, qb=1.0, Ts=22.0)
    dt = 0.025
    grid = np.arange(-90.0, 180.0 + dt / 2, dt)
    x = stationary_ou(rng_x, len(grid), dt, truth_params['sigma_x'], truth_params['tau_x'])
    disk = truth_params['A0'] * np.interp(grid - truth_params['mu0'], grid,
                                        gaussian_filter1d(x, truth_params['w0'] / dt, truncate=8))
    broad = truth_params['A1'] * np.interp(grid - truth_params['mu1'], grid,
                                         gaussian_filter1d(x, truth_params['w1'] / dt, truncate=8))
    sw = stationary_matern32(rng_sw, len(grid), dt, truth_params['Ts'])
    sb = stationary_matern32(rng_sb, len(grid), dt, truth_params['Ts'])
    # Irregular paired times, neither a common regular grid nor identical bands.
    base = np.linspace(1.0, 89.0, points)
    tw = np.sort(base + rng_t.uniform(-0.65, 0.65, points))
    tb = np.sort(base + rng_t.uniform(-0.65, 0.65, points))
    ew = rng_n.uniform(0.07, 0.11, points)
    eb = rng_n.uniform(0.07, 0.11, points)
    nw, nb = rng_n.normal(size=points) * ew, rng_n.normal(size=points) * eb
    # Keep truth on the integration grid: a coarse truth export would smooth
    # the OU realization again when assessing arbitrary reconstruction times.
    pred_t = np.linspace(0, 90, 3601)
    cases = {
        'clean': (np.zeros_like(sw), np.zeros_like(sb)),
        'response_only': (np.zeros_like(sw), truth_params['qb'] * sb),
        'independent': (truth_params['qW'] * sw, truth_params['qb'] * sb),
        'shared': (truth_params['qW'] * sw, truth_params['qb'] * sw),
    }
    for name, (s_w, s_b) in cases.items():
        case = out / name
        case.mkdir()
        yw = 7 + np.interp(tw, grid, x + s_w) + nw
        yb = 9 + np.interp(tb, grid, disk + broad + s_b) + nb
        np.savetxt(case / 'w2.txt', np.column_stack([tw, yw, ew]), header='time flux error')
        np.savetxt(case / 'response.txt', np.column_stack([tb, yb, eb]), header='time flux error')
        components = {key: np.interp(pred_t, grid, val) for key, val in
                      dict(x=x, disk=disk, blr=broad, slow_w=s_w, slow_b=s_b).items()}
        np.savez(case / 'truth.npz', time=pred_t, **components)
        p = truth_params.copy()
        if name == 'clean': p.update(qW=0, qb=0)
        if name == 'response_only': p.update(qW=0)
        (case / 'truth.json').write_text(json.dumps(p, indent=2))
    # Identical proper priors for every case. Both centers span the same range;
    # only ordering distinguishes components, not truth-centered restrictions.
    priors = {
        'sigma_x': {'kind': 'log_uniform', 'bounds': [0.1, 3.0]},
        'tau_x': {'kind': 'log_uniform', 'bounds': [0.3, 40.0]},
        'A0': {'kind': 'log_uniform', 'bounds': [0.02, 2.5]},
        'mu0': {'kind': 'uniform', 'bounds': [-3.0, 28.0]},
        'w0': {'kind': 'log_uniform', 'bounds': [0.1, 12.0]},
        'A1': {'kind': 'log_uniform', 'bounds': [0.02, 2.5]},
        'mu1': {'kind': 'uniform', 'bounds': [-3.0, 28.0]},
        'w1': {'kind': 'log_uniform', 'bounds': [0.1, 12.0]},
        'qW': {'kind': 'uniform', 'bounds': [0.0, 2.5]},
        'qb': {'kind': 'uniform', 'bounds': [0.0, 2.5]},
        'Ts': {'kind': 'log_uniform', 'bounds': [5.0, 120.0]},
    }
    (out / 'priors.json').write_text(json.dumps(priors, indent=2))
    (out / 'manifest.json').write_text(json.dumps({
        'seed': seed, 'points_per_band': points, 'duration_days': 90,
        'generation': 'exact OU and Matern32 discrete transitions; fine-grid Gaussian convolution',
        'integration_step_days': dt, 'gaussian_truncation_sigma': 8,
        'cases': list(cases), 'matched': ['driver', 'responses', 'cadence', 'measurement_noise'],
        'scope': 'one matched realization; not a repeated-simulation coverage study',
    }, indent=2))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--points', type=int, default=42)
    args = parser.parse_args()
    print(generate(args.out, args.seed, args.points))
