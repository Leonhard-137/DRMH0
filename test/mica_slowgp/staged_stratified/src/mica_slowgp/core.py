"""Two-band Gaussian-transfer GP with optional slow variability.

The model is ``yW = cW + x + sW`` and
``yb = cb + A0 * (G0 * x) + A1 * (G1 * x) + sb``.  ``x`` is a
stationary OU process. Gaussian transfer kernels have unit integral over the
whole real line; positive ``mu`` means a delayed response. No negative-lag
truncation, empirical driver prior, hidden normalization, or numerical jitter
is applied here.

All parameters are physical (not logarithms). Required parameters are
``sigma_x, tau_x, A0, mu0, w0, A1, mu1, w1``. ``sigma_x`` is the stationary
driver standard deviation, ``A0/A1`` are integrated response amplitudes,
``mu0/mu1`` are ordered Gaussian centers, and ``w0/w1`` are Gaussian standard
deviations. Slow modes additionally require ``qW, qb, Ts``: two standard
deviation amplitudes and a common time scale. The slow kernel is exactly
``(1 + abs(dt)/Ts) * exp(-abs(dt)/Ts)``; thus Ts is the usual Matern-3/2
length scale divided by sqrt(3). Slow modes are:

* ``none``: no slow process;
* ``independent``: independent sW and sb, with a common Ts;
* ``shared``: sW=qW*z and sb=qb*z for one unit-variance, zero-lag process z.

Constant means have an improper flat prior and are analytically marginalized.
Likelihoods include log|L.T C^-1 L| and (n-2)*log(2*pi), up to the common
arbitrary normalization of that flat prior. The core contains no other prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erfcx, ndtr


PARAMETER_NAMES = ("sigma_x", "tau_x", "A0", "mu0", "w0", "A1", "mu1", "w1")
SLOW_PARAMETER_NAMES = ("qW", "qb", "Ts")
MODES = ("none", "independent", "shared")
COMPONENTS = ("x", "disk", "blr", "slow_w", "slow_b", "w", "b")


def _vector(value: ArrayLike, name: str, *, nonempty: bool = True) -> NDArray:
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != 1 or (nonempty and result.size == 0):
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite numbers")
    return result


@dataclass(frozen=True)
class PairData:
    """Unscaled two-band observations, with 1-sigma measurement errors.

    Times may be irregular or unsorted. All times/widths/time scales must use
    the same unit. Input arrays are copied; later caller mutation is harmless.
    The two fluxes need not share units: response amplitudes convert x units
    into response-band units, and each q is in its own band's units.
    """

    tw: ArrayLike
    yw: ArrayLike
    ew: ArrayLike
    tb: ArrayLike
    yb: ArrayLike
    eb: ArrayLike

    def __post_init__(self) -> None:
        for name in ("tw", "yw", "ew", "tb", "yb", "eb"):
            array = _vector(getattr(self, name), name)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for band in ("w", "b"):
            t, y, e = (getattr(self, prefix + band) for prefix in ("t", "y", "e"))
            if not (len(t) == len(y) == len(e)):
                raise ValueError(f"t{band}, y{band}, e{band} must have equal lengths")
            if np.any(e < 0):
                raise ValueError("measurement standard deviations must be nonnegative")

    @property
    def y(self) -> NDArray:
        return np.concatenate((self.yw, self.yb))

    @property
    def errors(self) -> NDArray:
        return np.concatenate((self.ew, self.eb))

    @property
    def design(self) -> NDArray:
        """Two independent constant-mean columns in W-then-b row order."""
        matrix = np.zeros((len(self.tw) + len(self.tb), 2))
        matrix[: len(self.tw), 0] = 1.0
        matrix[len(self.tw) :, 1] = 1.0
        return matrix


def _mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def validate_parameters(params: Mapping[str, float], mode: str = "independent") -> dict:
    """Return a validated physical parameter dict; no prior is imposed.

    Zero amplitudes/widths and equal centers are allowed as mathematical
    boundaries. A sampler may use strictly positive scales and strict order.
    """
    _mode(mode)
    names = PARAMETER_NAMES + (() if mode == "none" else SLOW_PARAMETER_NAMES)
    missing = [name for name in names if name not in params]
    if missing:
        raise ValueError(f"missing required parameters: {', '.join(missing)}")
    result = {name: float(params[name]) for name in names}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("all physical parameters must be finite")
    positive = ("sigma_x", "tau_x") + (() if mode == "none" else ("Ts",))
    nonnegative = ("A0", "A1", "w0", "w1") + (() if mode == "none" else ("qW", "qb"))
    if any(result[name] <= 0 for name in positive):
        raise ValueError(f"parameters {positive} must be strictly positive")
    if any(result[name] < 0 for name in nonnegative):
        raise ValueError(f"parameters {nonnegative} must be nonnegative")
    if result["mu0"] > result["mu1"]:
        raise ValueError("Gaussian centers must satisfy mu0 <= mu1")
    if mode == "none":
        result.update(qW=0.0, qb=0.0, Ts=1.0)
    return result


def gaussian_ou_covariance(delta: ArrayLike, width: float, tau: float) -> NDArray:
    """E[exp(-abs(delta-Z)/tau)], Z~Normal(0,width**2).

    This is the unit-variance OU kernel convolved with a unit-area Gaussian.
    For two filtered processes, pass the quadrature sum of their widths.
    An erfcx branch avoids cancellation between r**2/2 and log Gaussian tails
    when width/tau is large; a separate CDF branch avoids erfcx overflow on
    negative arguments. Width zero is the exact delta-transfer limit.
    """
    delta = np.asarray(delta, dtype=float)
    width, tau = float(width), float(tau)
    if not np.all(np.isfinite(delta)) or not np.isfinite(width) or not np.isfinite(tau):
        raise ValueError("delta, width and tau must be finite")
    if width < 0 or tau <= 0:
        raise ValueError("width must be nonnegative and tau strictly positive")
    d = np.abs(delta).reshape(-1)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="raise"):
        if width == 0:
            return np.exp(-d / tau).reshape(delta.shape)
        r = width / tau
        if np.isinf(r):
            # For r beyond floating range, the relative asymptotic error is
            # O(r^-2), below float precision. Keep representable subnormals.
            z = d / width
            return (np.sqrt(2.0 / np.pi) * (tau / width)
                    * np.exp(-0.5 * z * z)).reshape(delta.shape)
        if r == 0:
            return np.exp(-d / tau).reshape(delta.shape)
        z = d / width
        envelope = 0.5 * np.exp(-0.5 * z * z)
        root_two = np.sqrt(2.0)
        result = envelope * erfcx((r + z) / root_two)
        central = z <= r
        result[central] += envelope[central] * erfcx((r - z[central]) / root_two)
        tail = ~central
        # This exponent is nonpositive because z > r in this branch.
        # Small r uses d/tau directly: d/width can overflow even though
        # d/tau is moderate for an almost-delta Gaussian.
        exponent = (0.5 * r * r - d[tail] / tau if r <= 1.0
                    else r * (0.5 * r - z[tail]))
        result[tail] += np.exp(exponent) * ndtr(z[tail] - r)
    return result.reshape(delta.shape)


def matern32_covariance(delta: ArrayLike, Ts: float) -> NDArray:
    """Unit-variance slow kernel, (1+abs(delta)/Ts)*exp(-abs(delta)/Ts)."""
    delta = np.asarray(delta, dtype=float)
    Ts = float(Ts)
    if not np.all(np.isfinite(delta)) or not np.isfinite(Ts) or Ts <= 0:
        raise ValueError("delta must be finite and Ts finite and positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ratio = np.abs(delta) / Ts
        # log1p avoids inf*0 at extreme time separations.
        result = np.exp(np.log1p(ratio) - ratio)
    return np.where(np.isinf(ratio), 0.0, result)


def _terms(component: str, p: dict) -> list[tuple[float, float, float]]:
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}, got {component!r}")
    if component in ("x", "w"):
        return [(1.0, 0.0, 0.0)]
    disk = (p["A0"], p["mu0"], p["w0"])
    blr = (p["A1"], p["mu1"], p["w1"])
    return {"disk": [disk], "blr": [blr], "b": [disk, blr]}.get(component, [])


def _slow_band(component: str) -> int | None:
    if component in ("slow_w", "w"):
        return 0
    if component in ("slow_b", "b"):
        return 1
    return None


def _component_covariance(ta: NDArray, ca: str, tb: NDArray, cb: str, p: dict, mode: str) -> NDArray:
    terms_a, terms_b = _terms(ca, p), _terms(cb, p)
    delta = ta[:, None] - tb[None, :]
    covariance = np.zeros(delta.shape)
    for amplitude_a, center_a, width_a in terms_a:
        for amplitude_b, center_b, width_b in terms_b:
            if amplitude_a == 0 or amplitude_b == 0:
                continue
            covariance += (p["sigma_x"] ** 2 * amplitude_a * amplitude_b
                           * gaussian_ou_covariance(delta - center_a + center_b,
                                                    np.hypot(width_a, width_b), p["tau_x"]))
    slow_a, slow_b = _slow_band(ca), _slow_band(cb)
    if mode != "none" and slow_a is not None and slow_b is not None:
        if mode == "shared" or slow_a == slow_b:
            amplitudes = (p["qW"], p["qb"])
            covariance += amplitudes[slow_a] * amplitudes[slow_b] * matern32_covariance(delta, p["Ts"])
    return covariance


def component_covariance(times_a: ArrayLike, component_a: str, times_b: ArrayLike,
                         component_b: str, params: Mapping[str, float],
                         mode: str = "independent") -> NDArray:
    """Covariance of any two latent components (without means or noise).

    Names are x, disk, blr, slow_w, slow_b, w=x+slow_w, and
    b=disk+blr+slow_b. Disk and BLR remain correlated through x; all terms,
    including their cross terms, are retained in a total-response covariance.
    """
    p = validate_parameters(params, mode)
    ta, tb = _vector(times_a, "times_a"), _vector(times_b, "times_b")
    return _component_covariance(ta, component_a, tb, component_b, p, mode)


def build_covariance(data: PairData, params: Mapping[str, float], mode: str = "independent") -> NDArray:
    """Training covariance in W-then-b order, including measurement variance."""
    p = validate_parameters(params, mode)
    nw = len(data.tw)
    n = nw + len(data.tb)
    covariance = np.empty((n, n))
    covariance[:nw, :nw] = _component_covariance(data.tw, "w", data.tw, "w", p, mode)
    covariance[:nw, nw:] = _component_covariance(data.tw, "w", data.tb, "b", p, mode)
    covariance[nw:, :nw] = covariance[:nw, nw:].T
    covariance[nw:, nw:] = _component_covariance(data.tb, "b", data.tb, "b", p, mode)
    covariance[np.diag_indices(n)] += data.errors ** 2
    return covariance


@dataclass
class LikelihoodState:
    """Dense-factorization result used by the likelihood and reconstruction."""

    covariance: NDArray
    cholesky: tuple[NDArray, bool]
    inverse_design: NDArray
    offset_mean: NDArray
    offset_covariance: NDArray
    alpha: NDArray
    log_likelihood: float


def factorize(data: PairData, params: Mapping[str, float], mode: str = "independent") -> LikelihoodState:
    """Factor C and marginalize both constant means; raise on invalid matrices.

    No jitter is added: numerical loss of positive definiteness is visible to
    the caller, while log_likelihood maps these failures to -inf for sampling.
    """
    covariance = build_covariance(data, params, mode)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance contains non-finite values")
    cholesky = cho_factor(covariance, lower=True, check_finite=False)
    design, y = data.design, data.y
    inverse_design = cho_solve(cholesky, design, check_finite=False)
    normal = design.T @ inverse_design
    normal_cholesky = cho_factor(normal, lower=True, check_finite=False)
    inverse_y = cho_solve(cholesky, y, check_finite=False)
    offset_mean = cho_solve(normal_cholesky, design.T @ inverse_y, check_finite=False)
    offset_covariance = cho_solve(normal_cholesky, np.eye(2), check_finite=False)
    residual = y - design @ offset_mean
    alpha = cho_solve(cholesky, residual, check_finite=False)
    logdet_covariance = 2.0 * np.sum(np.log(np.diag(cholesky[0])))
    logdet_normal = 2.0 * np.sum(np.log(np.diag(normal_cholesky[0])))
    loglike = -0.5 * (residual @ alpha + logdet_covariance + logdet_normal
                     + (len(y) - 2) * np.log(2.0 * np.pi))
    if not np.isfinite(loglike):
        raise ValueError("marginal likelihood is non-finite")
    return LikelihoodState(covariance, cholesky, inverse_design, offset_mean,
                           offset_covariance, alpha, float(loglike))


def log_likelihood(data: PairData, params: Mapping[str, float], mode: str = "independent") -> float:
    """Return the flat-mean-marginalized Gaussian log likelihood.

    Invalid physical parameters or failed dense factorizations return -inf.
    Unknown mode names still raise, since those indicate configuration errors.
    """
    _mode(mode)
    try:
        return factorize(data, params, mode).log_likelihood
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError):
        return -np.inf


@dataclass
class Prediction:
    """Joint conditional prediction at one fixed physical parameter point.

    Rows follow targets' insertion order; slices map each target label to its
    rows. covariance includes uncertainty in the two fitted constants, plus
    every cross-component covariance. offset_cross_covariance is Cov(predicted
    quantities, [cW, cb]), with shape (number of prediction rows, 2).
    Hyperparameter/posterior-sample uncertainty is not included by one call.
    """

    mean: NDArray
    covariance: NDArray
    slices: dict[str, slice]
    offset_mean: NDArray
    offset_covariance: NDArray
    offset_cross_covariance: NDArray
    log_likelihood: float


def condition(data: PairData, params: Mapping[str, float],
              targets: Mapping[str, tuple[str, ArrayLike]], mode: str = "independent",
              include_means: bool = True) -> Prediction:
    """Jointly reconstruct selected components at arbitrary times.

    Example: targets={"x": ("x", times), "disk": ("disk", times),
                      "response": ("b", times)}.
    With include_means=True, only w and b include their own unknown constant;
    all other components remain zero-mean latent processes. With False, w/b
    also represent fluctuations. Future measurement noise is never added.
    """
    if not targets:
        raise ValueError("targets must contain at least one named component")
    p = validate_parameters(params, mode)
    state = factorize(data, p, mode)
    prepared, slices, start = [], {}, 0
    for label, (component, times) in targets.items():
        _terms(component, p)
        t = _vector(times, f"target {label!r} times")
        section = slice(start, start + len(t))
        slices[label] = section
        prepared.append((component, t, section))
        start += len(t)
    n_prediction = start
    cross = np.empty((n_prediction, len(data.y)))
    prior = np.empty((n_prediction, n_prediction))
    target_design = np.zeros((n_prediction, 2))
    for i, (component, times, section) in enumerate(prepared):
        cross[section, : len(data.tw)] = _component_covariance(times, component, data.tw, "w", p, mode)
        cross[section, len(data.tw) :] = _component_covariance(times, component, data.tb, "b", p, mode)
        if include_means and component in ("w", "b"):
            target_design[section, 0 if component == "w" else 1] = 1.0
        for other_component, other_times, other_section in prepared[i:]:
            block = _component_covariance(times, component, other_times, other_component, p, mode)
            prior[section, other_section] = block
            prior[other_section, section] = block.T
    inverse_cross = cho_solve(state.cholesky, cross.T, check_finite=False)
    mean_residual = target_design - cross @ state.inverse_design
    mean = target_design @ state.offset_mean + cross @ state.alpha
    offset_cross = mean_residual @ state.offset_covariance
    covariance = prior - cross @ inverse_cross + offset_cross @ mean_residual.T
    # Symmetrize round-off only; do not clip eigenvalues or add hidden jitter.
    covariance = 0.5 * (covariance + covariance.T)
    return Prediction(mean, covariance, slices, state.offset_mean,
                      state.offset_covariance, offset_cross, state.log_likelihood)
