"""Unadopted candidate: exchangeable-triple conditional ordered prior.

This standalone module imports no GP, sampler, or production experiment code.
The L-shaped latent center region is uniform, then entire triples are sorted.
"""
from __future__ import annotations

import numpy as np

BASE_NAMES = ('sigma_x', 'tau_x', 'A0', 'mu0', 'w0', 'A1', 'mu1', 'w1')
SLOW_NAMES = ('qW', 'qb', 'Ts')


def lshape_centers(u0, u1, a, lo, hi):
    """Uniform unordered centers in [a,hi]^2 with max(center)>=lo.

    Accept closed unit-cube endpoints. Equal centers at endpoint/corner sets
    have prior measure zero and are not shifted to introduce a new prior.
    """
    if not np.isfinite([a, lo, hi]).all() or not a <= lo < hi:
        raise ValueError('Require finite a <= lo < hi')
    u0, u1 = np.broadcast_arrays(np.asarray(u0, dtype=float), np.asarray(u1, dtype=float))
    if (not np.isfinite(u0).all() or not np.isfinite(u1).all()
            or np.any(u0 < 0) or np.any(u0 > 1) or np.any(u1 < 0) or np.any(u1 > 1)):
        raise ValueError('Unit coordinates must lie in [0,1]')
    width = hi-a
    notch = lo-a
    height = hi-lo
    # Factoring Delta avoids subtracting nearly equal squared widths.
    area = height*(width+notch)
    probability_lower = notch/(width+notch)
    lower = u0 < probability_lower
    c0 = np.where(lower, a+u0*(width+notch),
                  lo+(u0-probability_lower)*(area/width))
    c1 = np.where(lower, lo+u1*height, a+u1*width)
    return c0, c1


class LShapePriorTransform:
    """Standalone candidate for exactly the exchangeable conditional prior.

    Other coordinate transforms keep their declared uniform/log-uniform
    priors. A0/A1 and w0/w1 must be exchangeable for whole-triple sorting.
    Accepts either one unit vector or an array of unit vectors.
    """
    def __init__(self, priors, center_interval, mode='shared'):
        if mode not in ('none', 'independent', 'shared'):
            raise ValueError('Unknown mode')
        self.names = BASE_NAMES+(() if mode == 'none' else SLOW_NAMES)
        self.priors = {name:dict(priors[name]) for name in self.names}
        for left,right in (('A0','A1'),('mu0','mu1'),('w0','w1')):
            if self.priors[left] != self.priors[right]:
                raise ValueError('L-shaped transform requires exchangeable component priors')
        if self.priors['mu0']['kind'] != 'uniform':
            raise ValueError('Centers require identical uniform priors')
        self.bounds = np.asarray([self.priors[name]['bounds'] for name in self.names],dtype=float)
        if (self.bounds.shape != (len(self.names),2) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:,1] <= self.bounds[:,0])):
            raise ValueError('Finite increasing bounds required')
        kinds = [self.priors[name]['kind'] for name in self.names]
        if any(kind not in ('uniform','log_uniform') for kind in kinds):
            raise ValueError('Unsupported prior kind')
        self.logscale = np.array([kind == 'log_uniform' for kind in kinds])
        if np.any(self.bounds[self.logscale] <= 0):
            raise ValueError('Log-uniform bounds must be positive')
        self.uniform_bounds = self.bounds.copy()
        self.uniform_bounds[self.logscale] = np.log(self.uniform_bounds[self.logscale])
        interval = np.asarray(center_interval,dtype=float)
        a,b = self.bounds[3]
        if (interval.shape != (2,) or not np.isfinite(interval).all()
                or not a <= interval[0] < interval[1] <= b):
            raise ValueError('Invalid conditional center interval')
        self.center_interval = interval
        lo,hi = interval
        self.center_prior_mass = (hi-lo)*(hi+lo-2*a)/(b-a)**2

    def __call__(self, u):
        u = np.asarray(u,dtype=float)
        if u.ndim not in (1,2) or u.shape[-1] != len(self.names):
            raise ValueError('Expected one unit vector or a matrix of unit vectors')
        if not np.isfinite(u).all() or np.any(u < 0) or np.any(u > 1):
            raise ValueError('Unit coordinates must lie in [0,1]')
        one = u.ndim == 1
        u = np.atleast_2d(u)
        x = self.uniform_bounds[:,0]+u*np.diff(self.uniform_bounds,axis=1)[:,0]
        x[:,self.logscale] = np.exp(x[:,self.logscale])
        a = self.bounds[3,0]
        x[:,3],x[:,6] = lshape_centers(u[:,3],u[:,6],a,*self.center_interval)
        crossed = x[:,3] > x[:,6]
        first = x[crossed,2:5].copy()
        x[crossed,2:5] = x[crossed,5:8]
        x[crossed,5:8] = first
        return x[0] if one else x
