"""Cached two-band likelihood, with core.py retained as the reference.

No target, parameterization, prior, numerical jitter or mean marginalization
is changed. Only data-dependent arrays, symmetric kernel reuse and dense
linear-algebra work are optimized. Instantiate once per pair and mode, then
pass the instance itself to the sampler as its physical-dict likelihood.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .core import PairData, gaussian_ou_covariance, matern32_covariance, validate_parameters, MODES

try:
    from . import _fast_kernel
except ImportError:
    _fast_kernel = None


class FastPairLikelihood:
    """Callable likelihood for fixed PairData and none/independent/shared mode.

    ``covariance(params)`` exposes the assembled (unfactorized) covariance for
    reference checks. Every call uses fresh work arrays; instances hold no
    mutable parameter-dependent scratch space and do not load native MICA.
    """

    def __init__(self, data: PairData, mode: str = "independent", backend: str = "auto"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if backend not in ("auto", "python", "compiled"):
            raise ValueError("backend must be auto, python or compiled")
        if backend == "compiled" and _fast_kernel is None:
            raise ImportError("Optional kernel is absent; run build_fast.py build_ext --inplace in the isolated project")
        self.backend = "compiled" if _fast_kernel is not None and backend != "python" else "python"
        self.data = data
        self.mode = mode
        self.nw = len(data.tw)
        self.n = self.nw + len(data.tb)
        self._dww = data.tw[:, None] - data.tw[None, :]
        self._dwb = data.tw[:, None] - data.tb[None, :]
        self._dbb = data.tb[:, None] - data.tb[None, :]
        self._aww = np.abs(self._dww)
        self._design = data.design
        self._y = data.y
        self._rhs = np.asfortranarray(np.column_stack((self._design, self._y)))
        self._noise = data.errors ** 2
        self._diagonal = np.diag_indices(self.n)
        self._normalization = (self.n - 2) * np.log(2.0 * np.pi)
        self._mode_code = MODES.index(mode)

    def covariance(self, params: Mapping[str, float]) -> np.ndarray:
        """Return the same training covariance as core.build_covariance."""
        p = validate_parameters(params, self.mode)
        if self.backend == "compiled":
            values = np.array([p[name] for name in
                               ("sigma_x", "tau_x", "A0", "mu0", "w0", "A1", "mu1", "w1", "qW", "qb", "Ts")])
            return _fast_kernel.covariance(self._dww, self._dwb, self._dbb,
                                           self._noise, values, self._mode_code)
        sigma2, tau = p["sigma_x"] ** 2, p["tau_x"]
        a0, a1, mu0, mu1, w0, w1 = (p[name] for name in ("A0", "A1", "mu0", "mu1", "w0", "w1"))
        c = np.empty((self.n, self.n), order="F")
        with np.errstate(over="ignore", under="ignore"):
            c[:self.nw, :self.nw] = sigma2 * np.exp(-self._aww / tau)
        wb = np.zeros(self._dwb.shape)
        bb = np.zeros(self._dbb.shape)
        if a0 != 0:
            wb += sigma2 * a0 * gaussian_ou_covariance(self._dwb + mu0, w0, tau)
            bb += sigma2 * a0 * a0 * gaussian_ou_covariance(self._dbb, np.hypot(w0, w0), tau)
        if a0 != 0 and a1 != 0:
            cross = gaussian_ou_covariance(self._dbb - mu0 + mu1, np.hypot(w0, w1), tau)
            bb += sigma2 * a0 * a1 * cross
            bb += sigma2 * a1 * a0 * cross.T
        if a1 != 0:
            wb += sigma2 * a1 * gaussian_ou_covariance(self._dwb + mu1, w1, tau)
            bb += sigma2 * a1 * a1 * gaussian_ou_covariance(self._dbb, np.hypot(w1, w1), tau)
        if self.mode != "none":
            qw, qb, ts = p["qW"], p["qb"], p["Ts"]
            if qw != 0:
                c[:self.nw, :self.nw] += qw * qw * matern32_covariance(self._dww, ts)
            if qb != 0:
                bb += qb * qb * matern32_covariance(self._dbb, ts)
            if self.mode == "shared" and qw != 0 and qb != 0:
                wb += qw * qb * matern32_covariance(self._dwb, ts)
        c[:self.nw, self.nw:] = wb
        c[self.nw:, :self.nw] = wb.T
        c[self.nw:, self.nw:] = bb
        c[self._diagonal] += self._noise
        return c

    def __call__(self, params: Mapping[str, float]) -> float:
        try:
            covariance = self.covariance(params)
            if not np.all(np.isfinite(covariance)):
                return -np.inf
            chol = cho_factor(covariance, lower=True, overwrite_a=True, check_finite=False)
            inverse_rhs = cho_solve(chol, self._rhs, check_finite=False)
            inverse_design, inverse_y = inverse_rhs[:, :2], inverse_rhs[:, 2]
            normal = self._design.T @ inverse_design
            normal_chol = cho_factor(normal, lower=True, check_finite=False)
            offsets = cho_solve(normal_chol, self._design.T @ inverse_y, check_finite=False)
            residual = self._y - self._design @ offsets
            alpha = cho_solve(chol, residual, check_finite=False)
            value = -0.5 * (residual @ alpha
                            + 2.0 * np.log(np.diag(chol[0])).sum()
                            + 2.0 * np.log(np.diag(normal_chol[0])).sum()
                            + self._normalization)
            return float(value) if np.isfinite(value) else -np.inf
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError):
            return -np.inf
