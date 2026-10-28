#!/usr/bin/env python3
"""Forward PSD simulations at the original Mrk817 W2/U epochs.

No MICA fitting, parameter search, input mutation, or variance renormalization.
ForwardModel(seed, nreal).evaluate([tauI,tauS,AW,AU,BW,BU]) -> (nreal,2,23).
The normalization is fractional flux relative to each observed sample mean.
I and S have unit stationary variance and independent fixed random streams.
W=AW*DW*I+BW*S+nW; U=AU*(0.9*DU+0.1*B)*I+BU*S+nU.

I responses are drawn jointly from the FFT-interpolated covariance, retaining
the shared central driver. S is drawn by an exact Matérn-3/2 state transition
at the merged actual epochs. This avoids adding a nugget to its nearly
singular covariance when adjacent W/U epochs are only 0.001 day apart.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.linalg import cholesky, eigh
from scipy.special import gammainc
from threadpoolctl import threadpool_limits

from prepare import _fft_covariance, _real_disk_transform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "test/mica_smooth/data/original"
BANDS = ("UVW2", "U")
PARAMETERS = ("tauI", "tauS", "AW", "AU", "BW", "BU")


def _lru_put(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def _lru_get(cache, key):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _parameters(params):
    values = np.array([params[k] for k in PARAMETERS] if isinstance(params, dict) else params,
                      dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all() or np.any(values[:2] <= 0) or np.any(values[2:] < 0):
        raise ValueError("Expected finite tauI,tauS>0 and AW,AU,BW,BU>=0")
    return values


def _factor_covariance(covariance):
    """Factor without a nugget, clipping only roundoff negative eigenvalues."""
    covariance = (covariance + covariance.T) / 2
    try:
        return cholesky(covariance, lower=True, check_finite=False), dict(method="cholesky", jitter=0.0)
    except np.linalg.LinAlgError:
        values, vectors = eigh(covariance, check_finite=False)
        tolerance = 256 * np.finfo(float).eps * len(values) * max(values[-1], np.finfo(float).tiny)
        if values[0] < -tolerance:
            raise ValueError(f"Non-PSD illumination covariance: {values[0]}, tolerance {tolerance}")
        factor = vectors * np.sqrt(np.maximum(values, 0))[None, :]
        return factor, dict(method="eigen_roundoff_clip", jitter=0.0,
                            min_eigenvalue=float(values[0]), tolerance=float(tolerance),
                            clipped_variance=float(-values[values < 0].sum()))


class ForwardModel:
    """Reproducible sampled spectra with cached tau-dependent components.

    Public fields: obs/observed_psd (2,23), freq_bins/bin_centers (2,23),
    frequencies {band:(nf,)}, bin_counts (2,23), data/times/errors dictionaries.

    ``components(tauI,tauS,sigma=None)`` returns a dict keyed by band. Each
    value contains centered ``illumination``, ``slow``, ``noise`` curves
    (nreal,N), their ``projection`` (nreal,3,2,nf), and ``quadratic``
    (nreal,23,3,3). Projection includes sqrt(T/N), so PSD is its square sum.

    Sigma applies the same row-normalized Gaussian high-pass at each band's
    original epochs; it does not approximate filtering by a Fourier multiplier.
    """
    def __init__(self, seed=2026100501, nreal=64, data_dir=DATA, nbins=23,
                 tau_cache_size=8, bundle_cache_size=3):
        if int(nreal) != nreal or nreal < 1 or nbins != 23:
            raise ValueError("nreal must be positive and this matching grid uses 23 bins")
        self.seed, self.nreal = int(seed), int(nreal)
        self.bands = BANDS
        self._thread_limit = threadpool_limits(limits=1, user_api="blas")
        self.data = {b: np.loadtxt(Path(data_dir) / f"Mrk817_a_{b}.txt") for b in BANDS}
        self.means = {b: float(d[:, 1].mean()) for b, d in self.data.items()}
        t0 = min(d[0, 0] for d in self.data.values())
        self.time_zero = float(t0)
        self.times = {b: d[:, 0] - t0 for b, d in self.data.items()}
        self.errors = {b: d[:, 2] / self.means[b] for b, d in self.data.items()}
        self.observed_curves = {b: d[:, 1] / self.means[b] - 1 for b, d in self.data.items()}
        self.sizes = {b: len(d) for b, d in self.data.items()}
        self.spans = {b: float(np.ptp(t)) for b, t in self.times.items()}
        self.frequencies, self._projection, self._binning, self._counts = {}, {}, {}, {}
        centers = []
        for b in BANDS:
            d, t = self.data[b], self.times[b]
            if d.ndim != 2 or d.shape[1] != 3 or not np.isfinite(d).all() or np.any(d[:, 2] <= 0) or np.any(np.diff(t) <= 0):
                raise ValueError(f"Invalid original data for {b}")
            f = np.arange(1 / self.spans[b], .30, 1 / (5 * self.spans[b]))
            # Orthogonalize the constant-removed sinusoidal design. The two
            # squared coefficients equal the floating-mean chi-square reduction.
            angle = 2 * np.pi * (t - t.mean())[:, None] * f[None, :]
            cosine, sine = np.cos(angle), np.sin(angle)
            cosine -= cosine.mean(axis=0)
            sine -= sine.mean(axis=0)
            q0 = cosine / np.linalg.norm(cosine, axis=0)[None, :]
            sine -= q0 * np.sum(q0 * sine, axis=0)[None, :]
            q1 = sine / np.linalg.norm(sine, axis=0)[None, :]
            projection = np.stack([q0, q1], axis=1) * np.sqrt(self.spans[b] / len(t))
            edges = np.geomspace(f[0], f[-1] * (1 + 1e-10), nbins + 1)
            bins = np.searchsorted(edges, f, side="right") - 1
            counts = np.bincount(bins, minlength=nbins)
            if np.any(counts == 0) or len(counts) != nbins:
                raise ValueError("Frequency binning produced empty/outside bins")
            binning = np.zeros((len(f), nbins))
            binning[np.arange(len(f)), bins] = 1 / counts[bins]
            self.frequencies[b], self._projection[b] = f, projection
            self._binning[b], self._counts[b] = binning, counts
            centers.append(np.exp(np.log(f) @ binning))
        self.freq_bins = self.bin_centers = np.asarray(centers)
        self.bin_counts = np.asarray([self._counts[b] for b in BANDS])
        self.observed_psd = self.obs = self.periodograms(self.observed_curves)[0]

        params_path = HERE / "cases_tau2_area9/raw/config/generation.json"
        saved_parameters = json.loads(params_path.read_text())["parameters"]
        kernel_keys = ("covariance_dt_days", "covariance_n", "disk_support_days", "t1_kelvin",
                       "driver_wavelength_observed_angstrom", "response_wavelength_observed_angstrom",
                       "blr_theta_days", "blr_onset_days", "redshift", "inclination_deg",
                       "disk_fine_factor", "disk_origin_cells", "disk_origin_order")
        self.kernel_parameters = {key: deepcopy(saved_parameters[key]) for key in kernel_keys}
        self.kernel_parameters.update(response_disk_gain=.9, response_blr_gain=.1)
        p = self.kernel_parameters
        self.dt, self.nfft = p["covariance_dt_days"], p["covariance_n"]
        self.omega = 2 * np.pi * np.fft.rfftfreq(self.nfft, self.dt)
        dw, aw = _real_disk_transform(p, p["driver_wavelength_observed_angstrom"])
        du, au = _real_disk_transform(p, p["response_wavelength_observed_angstrom"])
        # Enforce exactly unit kernel areas; original quadrature errors are <1e-6.
        dw, du = dw / aw, du / au
        blr = np.exp(-1j * self.omega * p["blr_onset_days"]) / (1 + 1j * self.omega * p["blr_theta_days"])**2
        self.filters = {"UVW2": dw, "U": .9 * du + .1 * blr}
        self._products = {(b1, b2): self.filters[b2] * self.filters[b1].conj()
                          for i, b1 in enumerate(BANDS) for b2 in BANDS[i:]}
        self._lags = {(b1, b2): self.times[b2][None, :] - self.times[b1][:, None]
                      for i, b1 in enumerate(BANDS) for b2 in BANDS[i:]}
        self._joint_times = np.concatenate([self.times[b] for b in BANDS])
        self._sort_order = np.argsort(self._joint_times)
        self._inverse_sort = np.argsort(self._sort_order)
        self._sorted_times = self._joint_times[self._sort_order]
        self._nw = self.sizes["UVW2"]
        count = len(self._joint_times)
        seed_i, seed_s, seed_n = np.random.SeedSequence(self.seed).spawn(3)
        # Realization is the first axis, making the 64/256 prefix exactly shared.
        self._z_i = np.random.default_rng(seed_i).standard_normal((self.nreal, count))
        self._z_s = np.random.default_rng(seed_s).standard_normal((self.nreal, count, 2))
        z_noise = np.random.default_rng(seed_n).standard_normal((self.nreal, count))
        self._noise = self._split_center(z_noise * np.concatenate([self.errors[b] for b in BANDS])[None, :])
        self._tau_cache_size, self._bundle_cache_size = int(tau_cache_size), int(bundle_cache_size)
        self._i_cache, self._s_cache, self._bundle_cache, self._hp_cache = (OrderedDict() for _ in range(4))
        self.factor_diagnostics = {}
        self.metadata = dict(seed=self.seed, nreal=self.nreal, samples=self.sizes,
            normalization="(flux / fixed observed sample mean) - 1; error / same mean; no RMS renormalization",
            estimator="unweighted floating-mean LombScargle psd times 2*T/N; arithmetic mean in 23 log bins",
            driver="unit stationary-variance OU", slow="unit stationary-variance Matern3/2, K(h)=(1+|h|/tauS)exp(-|h|/tauS)",
            formula="W=AW*DW*I+BW*S+noiseW; U=AU*(0.9*DU+0.1*B)*I+BU*S+noiseU",
            disk_quadrature_areas_before_normalization={"UVW2": aw, "U": au},
            disk_area_normalization="each disk Fourier kernel divided by its quadrature area",
            kernel_parameters=p, fft_period_days=self.dt * self.nfft,
            filtering="original - row-normalized Gaussian trend + original sample mean at actual epochs")

    def _split_center(self, joint):
        return {b: part - part.mean(axis=1, keepdims=True) for b, part in
                zip(BANDS, np.split(joint, [self._nw], axis=1))}

    def illumination_covariance(self, tauI):
        tauI = float(tauI)
        if not np.isfinite(tauI) or tauI <= 0:
            raise ValueError("tauI must be positive")
        spectrum = 2 * tauI / (1 + (self.omega * tauI)**2)
        blocks = {}
        for (b1, b2), prod in self._products.items():
            blocks[b1, b2] = _fft_covariance(spectrum * prod, self._lags[b1, b2], self.dt, self.nfft)
        return np.block([[blocks["UVW2", "UVW2"], blocks["UVW2", "U"]],
                         [blocks["UVW2", "U"].T, blocks["U", "U"]]])

    def slow_covariance(self, tauS):
        distance = np.abs(self._joint_times[:, None] - self._joint_times[None, :]) / float(tauS)
        return (1 + distance) * np.exp(-distance)

    def _illumination(self, tauI):
        cached = _lru_get(self._i_cache, tauI)
        if cached is not None:
            return cached
        covariance = self.illumination_covariance(tauI)
        factor, diag = _factor_covariance(covariance)
        self.factor_diagnostics[str(tauI)] = diag
        curves = self._split_center(self._z_i @ factor.T)
        return _lru_put(self._i_cache, tauI, curves, self._tau_cache_size)

    def _slow(self, tauS):
        cached = _lru_get(self._s_cache, tauS)
        if cached is not None:
            return cached
        x = np.diff(self._sorted_times) / tauS
        e = np.exp(-x)
        # State=(S,tau*dS/dt); stationary covariance is identity. Exact F=e^-x
        # [[1+x,x],[-x,1-x]], Q=identity-F F.T. gammainc stabilizes Q00~4x^3/3.
        q00 = gammainc(3, 2*x)
        q01 = 2*x*x*np.exp(-2*x)
        q11 = -np.expm1(-2*x) + np.exp(-2*x)*(2*x-2*x*x)
        l00 = np.sqrt(q00)
        l10 = np.divide(q01, l00, out=np.zeros_like(q01), where=l00 > 0)
        remainder = q11-l10*l10
        if np.any(remainder < -64*np.finfo(float).eps):
            raise ValueError("Invalid state-transition innovation covariance")
        l11 = np.sqrt(np.maximum(remainder, 0))
        states = self._z_s[:, 0, :].copy()
        values = np.empty((self.nreal, len(self._joint_times)))
        values[:, 0] = states[:, 0]
        for j in range(len(x)):
            s, v = states[:, 0].copy(), states[:, 1].copy()
            z0, z1 = self._z_s[:, j+1, 0], self._z_s[:, j+1, 1]
            states[:, 0] = e[j]*((1+x[j])*s+x[j]*v)+l00[j]*z0
            states[:, 1] = e[j]*(-x[j]*s+(1-x[j])*v)+l10[j]*z0+l11[j]*z1
            values[:, j+1] = states[:, 0]
        curves = self._split_center(values[:, self._inverse_sort])
        return _lru_put(self._s_cache, tauS, curves, self._tau_cache_size)

    def _highpass(self, band, sigma):
        if sigma is None:
            return None
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive or None")
        key = (band, sigma)
        cached = _lru_get(self._hp_cache, key)
        if cached is not None:
            return cached
        t = self.times[band]
        weights = np.exp(-.5*((t[:, None]-t[None, :])/sigma)**2)
        weights /= weights.sum(axis=1, keepdims=True)
        return _lru_put(self._hp_cache, key, np.eye(len(t))-weights+1/len(t), 8)

    def periodograms(self, curves, sigma=None):
        """Compute actual sampled PSDs, accepting {band:(R,N)} or 1-D curves."""
        result = []
        for b in BANDS:
            y = np.atleast_2d(np.asarray(curves[b], dtype=float))
            hp = self._highpass(b, sigma) if hasattr(self, "_hp_cache") else None
            if hp is not None:
                y = y @ hp.T
            projection = y @ self._projection[b].reshape(len(self.times[b]), -1)
            projection = projection.reshape(len(y), 2, -1)
            result.append(np.sum(projection**2, axis=1) @ self._binning[b])
        return np.stack(result, axis=1)

    def observed(self, sigma=None):
        return self.periodograms(self.observed_curves, sigma=sigma)[0]

    def components(self, tauI, tauS, sigma=None):
        tauI, tauS = float(tauI), float(tauS)
        if min(tauI, tauS) <= 0 or not np.isfinite([tauI, tauS]).all():
            raise ValueError("Time scales must be finite and positive")
        key = (tauI, tauS, None if sigma is None else float(sigma))
        cached = _lru_get(self._bundle_cache, key)
        if cached is not None:
            return cached
        illumination, slow = self._illumination(tauI), self._slow(tauS)
        bundle = {}
        for b in BANDS:
            curves = np.stack([illumination[b], slow[b], self._noise[b]], axis=1)
            hp = self._highpass(b, sigma)
            if hp is not None:
                curves = curves @ hp.T
            projection = curves @ self._projection[b].reshape(len(self.times[b]), -1)
            projection = projection.reshape(self.nreal, 3, 2, -1)
            quadratic = np.empty((self.nreal, 23, 3, 3))
            for j in range(3):
                for k in range(j, 3):
                    cross = np.sum(projection[:, j]*projection[:, k], axis=1) @ self._binning[b]
                    quadratic[:, :, j, k] = cross
                    quadratic[:, :, k, j] = cross
            bundle[b] = dict(illumination=curves[:, 0], slow=curves[:, 1], noise=curves[:, 2],
                             projection=projection, quadratic=quadratic)
        return _lru_put(self._bundle_cache, key, bundle, self._bundle_cache_size)

    def evaluate(self, params, sigma=None):
        tauI, tauS, aw, au, bw, bu = _parameters(params)
        bundle = self.components(tauI, tauS, sigma=sigma)
        result = []
        for b, a, gain_s in zip(BANDS, (aw, au), (bw, bu)):
            coeff = np.array([a, gain_s, 1.0])
            result.append(np.einsum("rkij,i,j->rk", bundle[b]["quadratic"], coeff, coeff, optimize=False))
        return np.stack(result, axis=1)

    def curves(self, params, sigma=None):
        tauI, tauS, aw, au, bw, bu = _parameters(params)
        bundle = self.components(tauI, tauS, sigma=sigma)
        return {b: a*bundle[b]["illumination"]+gain_s*bundle[b]["slow"]+bundle[b]["noise"]
                for b, a, gain_s in zip(BANDS, (aw, au), (bw, bu))}


def core_check(output=None, nreal=64):
    """Small deterministic validation/benchmark; no search and no MICA."""
    from astropy.timeseries import LombScargle
    started = time.perf_counter()
    model = ForwardModel(seed=2026100501, nreal=nreal)
    init_seconds = time.perf_counter()-started
    params = np.array([2., 55.3735295468, .15, .12, .15, .12])
    started = time.perf_counter()
    result = model.evaluate(params)
    new_tau_seconds = time.perf_counter()-started
    errors = {}
    curves = model.curves(params)
    for b in BANDS:
        t, f = model.times[b], model.frequencies[b]
        for label, y in (("observed", model.observed_curves[b]), ("simulation", curves[b][0])):
            expected = LombScargle(t, y, fit_mean=True, normalization="psd").power(f, method="cython")*2*model.spans[b]/len(t)
            expected = expected @ model._binning[b]
            actual = model.obs[BANDS.index(b)] if label == "observed" else result[0, BANDS.index(b)]
            error = float(np.max(np.abs(actual-expected)/np.maximum(expected, 1e-30)))
            errors[f"{b}_{label}_astropy_max_relative"] = error
            if error > 1e-10:
                raise AssertionError(errors)
    direct = model.periodograms(curves)
    np.testing.assert_allclose(result, direct, rtol=2e-12, atol=1e-15)
    for sigma in (20, 40):
        filtered = model.evaluate(params, sigma=sigma)
        explicit = model.periodograms(curves, sigma=sigma)
        np.testing.assert_allclose(filtered, explicit, rtol=2e-11, atol=1e-15)
        errors[f"sigma{sigma}_max_relative"] = float(np.max(np.abs(filtered-explicit)/np.maximum(explicit, 1e-30)))
    # Verify exact state-space covariance identities at short/long steps,
    # including the small step that makes dense Matérn Cholesky delicate.
    transition_errors = []
    for tau in (10., 55.3735295468, 500.):
        for step in (.001, .2, 10., 100.):
            x = step/tau
            f = np.exp(-x)*np.array([[1+x, x], [-x, 1-x]])
            q00 = gammainc(3, 2*x)
            q01 = 2*x*x*np.exp(-2*x)
            q11 = -np.expm1(-2*x)+np.exp(-2*x)*(2*x-2*x*x)
            q = np.array([[q00, q01], [q01, q11]])
            transition_errors.append(float(np.max(np.abs(f@f.T+q-np.eye(2)))))
            assert np.linalg.eigvalsh(q)[0] >= 0
            np.testing.assert_allclose(f[0, 0], (1+x)*np.exp(-x), rtol=1e-14)
    assert max(transition_errors) < 2e-15
    # Changing amplitudes uses cached quadratic forms only.
    started = time.perf_counter()
    for j in range(200):
        trial = params.copy(); trial[2:] *= 1+j/10000
        model.evaluate(trial)
    amplitude_seconds = (time.perf_counter()-started)/200
    started = time.perf_counter()
    model.evaluate([.1, 500., .15, .12, .15, .12])
    extreme_tau_seconds = time.perf_counter()-started
    report = dict(passed=True, metadata=model.metadata, shape=list(result.shape),
                  errors=errors, transition_covariance_max_abs_error=max(transition_errors),
                  illumination_factors=model.factor_diagnostics,
                  timing_seconds=dict(initialization=init_seconds, new_tau_pair=new_tau_seconds,
                                      new_extreme_tau_pair=extreme_tau_seconds,
                                      cached_amplitude_evaluation=amplitude_seconds),
                  max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  output_is_fractional_psd=True)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--nreal", type=int, default=64)
    parser.add_argument("--output", type=Path, default=HERE/"psd_match_817/core_check.json")
    args = parser.parse_args()
    print(json.dumps(core_check(args.output, args.nreal), indent=2))
