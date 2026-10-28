"""Descriptive physical attributes for PSD-matching candidates.

RMS means sqrt(E[ddof=0 sample variance]) at the observed epochs, rather
than stationary RMS, a realized RMS, a kernel area, or E[sample RMS].
This module draws no light curves and applies no physical-selection penalty.
"""
from __future__ import annotations

import numpy as np

from prepare import _fft_covariance


PARAMETERS = ("tauI", "tauS", "AW", "AU", "BW", "BU")


def _variance_weight(times, sigma):
    """Q such that sum(Q*K) = E[var(H*y)], with sample means removed."""
    times = np.asarray(times, dtype=float)
    count = len(times)
    if sigma is None:
        return np.eye(count) / count - np.ones((count, count)) / count**2
    weights = np.exp(-0.5 * ((times[:, None] - times[None, :]) / sigma)**2)
    weights /= weights.sum(axis=1, keepdims=True)
    highpass = np.eye(count) - weights + 1.0 / count
    centered = highpass - highpass.mean(axis=0, keepdims=True)
    return centered.T @ centered / count


def _expected_variance(covariance, weight):
    terms = np.asarray(covariance) * weight
    variance = float(terms.sum())
    tolerance = 256 * np.finfo(float).eps * max(float(np.abs(terms).sum()), 1e-300)
    if variance < -tolerance:
        raise ValueError(f"Negative expected sample variance: {variance}")
    return max(variance, 0.0)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def physics_metrics(model, params):
    """Return JSON-serializable RMS and intrinsic-PSD attributes, without cuts.

    ``model`` is a psd_match_forward.ForwardModel. ``params`` may be a
    dictionary or sequence ordered tauI,tauS,AW,AU,BW,BU. For sigma20/40,
    both bands use their own actual-epoch H = identity - Gaussian weights
    + sample-mean operator. Disk and BLR share I, so their covariance is
    retained in ``illumination`` and ``total``; RMS values are not summed.
    """
    values = np.asarray([params[k] for k in PARAMETERS] if isinstance(params, dict)
                        else params, dtype=float)
    if (values.shape != (6,) or not np.isfinite(values).all()
            or np.any(values[:2] <= 0) or np.any(values[2:] < 0)):
        raise ValueError("Expected finite tauI,tauS>0 and AW,AU,BW,BU>=0")
    tau_i, tau_s, a_w, a_u, b_w, b_u = values
    p = model.kernel_parameters
    omega = np.asarray(model.omega)
    spectrum_i = 2 * tau_i / (1 + (omega * tau_i)**2)
    blr = (np.exp(-1j * omega * p["blr_onset_days"])
           / (1 + 1j * omega * p["blr_theta_days"])**2)
    # U's stored filter is exactly 0.9*DU + 0.1*B. Extract the separately
    # area-normalized disk term without another disk quadrature calculation.
    transfer = {
        "UVW2": {"disk": a_w * model.filters["UVW2"],
                 "blr": np.zeros_like(blr),
                 "illumination": a_w * model.filters["UVW2"]},
        "U": {"disk": a_u * (model.filters["U"] - 0.1 * blr),
              "blr": 0.1 * a_u * blr,
              "illumination": a_u * model.filters["U"]},
    }
    by_sigma = {label: {} for label in ("raw", "sigma20", "sigma40")}
    sampling = {}
    for band, slow_amplitude in (("UVW2", b_w), ("U", b_u)):
        times = np.asarray(model.times[band], dtype=float)
        errors = np.asarray(model.errors[band], dtype=float)
        lags = times[None, :] - times[:, None]
        covariance = {
            name: _fft_covariance(spectrum_i * np.abs(response)**2, lags,
                                  model.dt, model.nfft)
            for name, response in transfer[band].items()
        }
        distance = np.abs(lags) / tau_s
        covariance["slow"] = slow_amplitude**2 * (1 + distance) * np.exp(-distance)
        covariance["noise"] = np.diag(errors**2)
        sampling[band] = {"n": len(times), "span_days": float(np.ptp(times))}
        for label, sigma in (("raw", None), ("sigma20", 20.0), ("sigma40", 40.0)):
            weight = _variance_weight(times, sigma)
            variances = {name: _expected_variance(cov, weight)
                         for name, cov in covariance.items()}
            variances["total"] = (variances["illumination"] + variances["slow"]
                                  + variances["noise"])
            rms = {name: float(np.sqrt(value)) for name, value in variances.items()}
            rms["slow_to_illumination"] = _ratio(rms["slow"], rms["illumination"])
            rms["blr_to_noise"] = _ratio(rms["blr"], rms["noise"])
            # Cross term after mean removal and filtering: Var(disk+BLR)
            # = Var(disk) + Var(BLR) + 2*Cov(disk,BLR).
            rms["disk_blr_cross_variance"] = (variances["illumination"]
                                              - variances["disk"] - variances["blr"])
            by_sigma[label][band] = rms
    x_s = 2 * np.pi * tau_s / 100.0
    fractions = {
        "I_period_lt_5_days": float(2 / np.pi * np.arctan(5 / (2 * np.pi * tau_i))),
        "I_period_lt_10_days": float(2 / np.pi * np.arctan(10 / (2 * np.pi * tau_i))),
        "S_period_gt_100_days": float(2 / np.pi * (np.arctan(x_s) + x_s / (1 + x_s**2))),
    }
    return {
        "parameters": dict(zip(PARAMETERS, map(float, values))),
        "sampling": sampling,
        "rms_by_sigma": by_sigma,
        "psd_power_fractions": fractions,
        "definitions": {
            "rms": "sqrt(E[ddof=0 sample variance]); fractional-flux units at actual epochs",
            "window": "trace(H K H.T)/N - mean(H K H.T), including removal of sample mean",
            "filter": "raw: identity; sigma20/40: actual-epoch I-W+J, applied to each band and noise",
            "illumination": "disk+BLR sharing I; includes their cross covariance",
            "disk_blr_cross_variance": "2*E[sample covariance(disk, BLR)], not an RMS",
            "psd_power_fractions": "infinite stationary-process spectral integrals, not finite-window fractions",
            "physical_selection": "none; descriptive attributes only, no penalty or acceptance cut",
        },
    }
