"""Starkey-parameterized, linear standard-disk response for injection tests.

Model
-----
T(R) = T1 * (R / 1 light-day)**(-3/4).
The response weight is R*T*dB_lambda/dT, NOT R*B_lambda.  The delay is
tau_obs = (1+z)*(R/c)*(1 - sin(i)*cos(phi)).  This is the far-zone,
instantaneously reprocessing, flat blackbody disk with R in (0, infinity).
The temperature slope is fixed at 3/4: changing it while retaining a
radius-independent fractional irradiation response would change the model.
T1 describes the TOTAL background heating; normalization does not remove
the dependence of the disk size on this temperature.

This is a first-order, background-fixed response to the central driver.
It is neither a W2-to-other-band kernel nor a finite-amplitude correction.
Inner-edge terms, finite lamp height, GR, and bandpass averaging are omitted.

Parameterization: Starkey et al. (2016), equations 3--6,
https://doi.org/10.1093/mnras/stv2744
and Starkey et al. (2017), T1 at 1 light-day, alpha=3/4,
https://arxiv.org/abs/1611.06051 (equations 4--5 and Appendix).
The radius integral is evaluated analytically using the delay delta
function; only a smooth azimuth integral is evaluated numerically.

Example (the temperature is illustrative, not a measured Mrk 817 value)::

    import numpy as np
    from starkey_disk_transfer import disk_transfer_function, disk_moments

    z = 0.031
    tau = np.linspace(0.0, 50.0, 10001)  # observer-frame days
    psi = disk_transfer_function(
        tau, wavelength_rest_angstrom=1928.0/(1+z),
        t1_kelvin=20000.0, inclination_deg=45.0, redshift=z,
    )
    # psi is a DENSITY in day^-1; quadrature/convolution needs time weights.
    # Its full-domain area is 1. A short input grid is NOT renormalized.
    truth = disk_moments(
        wavelength_rest_angstrom=1928.0/(1+z),
        t1_kelvin=20000.0, inclination_deg=45.0, redshift=z,
    )

Dependencies: numpy, scipy. Run this file directly for numerical self-checks.
Importing the module performs no simulation and writes no files.
"""

from functools import lru_cache

import numpy as np
from scipy.constants import c, h, k
from scipy.special import gamma, zeta


_ALPHA = 0.75
_HC_OVER_K_ANGSTROM_K = h * c / k * 1e10
# For y=R/R_lambda, p(y) = y*q(y**alpha)/N, q(x)=x*exp(x)/(exp(x)-1)**2.
_RADIAL_NORMALIZATION = gamma(1 + 2/_ALPHA) * zeta(2/_ALPHA) / _ALPHA
_MEAN_RADIUS_FACTOR = (
    gamma(1 + 3/_ALPHA) * zeta(3/_ALPHA)
    / (gamma(1 + 2/_ALPHA) * zeta(2/_ALPHA))
)
_SECOND_RADIUS_FACTOR = (
    gamma(1 + 4/_ALPHA) * zeta(4/_ALPHA)
    / (gamma(1 + 2/_ALPHA) * zeta(2/_ALPHA))
)

__all__ = ["temperature_profile", "disk_transfer_function", "disk_moments"]


def _positive_scalar(name, value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _parameters(wavelength_rest_angstrom, t1_kelvin, inclination_deg, redshift):
    wavelength = _positive_scalar("wavelength_rest_angstrom", wavelength_rest_angstrom)
    temperature = _positive_scalar("t1_kelvin", t1_kelvin)
    inclination = float(inclination_deg)
    redshift = float(redshift)
    if not np.isfinite(inclination) or not 0 <= inclination < 90:
        raise ValueError("inclination_deg must satisfy 0 <= i < 90")
    if not np.isfinite(redshift) or redshift < 0:
        raise ValueError("redshift must be finite and nonnegative")
    r_lambda = (wavelength * temperature / _HC_OVER_K_ANGSTROM_K)**(1/_ALPHA)
    return r_lambda, (1 + redshift) * r_lambda, inclination


def temperature_profile(radius_light_days, t1_kelvin):
    """Return background T in K at REST-FRAME radii measured in light-days."""
    temperature = _positive_scalar("t1_kelvin", t1_kelvin)
    radius = np.asarray(radius_light_days, dtype=float)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0):
        raise ValueError("radius_light_days must be finite and positive")
    result = temperature * radius**(-_ALPHA)
    return float(result) if result.ndim == 0 else result


@lru_cache(maxsize=16)
def _azimuth_quadrature(inclination_deg, n_phi):
    # Symmetry reduces the uniform 0..2pi azimuth average to 0..pi.
    nodes, weights = np.polynomial.legendre.leggauss(n_phi)
    phi = 0.5 * np.pi * (nodes + 1)
    g = 1 - np.sin(np.deg2rad(inclination_deg)) * np.cos(phi)
    return g, 0.5 * weights


def disk_transfer_function(
    tau_days,
    *,
    wavelength_rest_angstrom,
    t1_kelvin,
    inclination_deg=45.0,
    redshift=0.0,
    n_phi=128,
):
    """Evaluate the unit-area central-driver-to-disk response density.

    Parameters
    ----------
    tau_days : float or array-like
        Observer-frame delay(s) in days. Negative and zero delays return 0.
        The array need not be uniform or sorted; it is only an evaluation grid.
    wavelength_rest_angstrom : float
        REST-FRAME monochromatic wavelength in Angstrom. Convert an observed
        filter wavelength with lambda_rest=lambda_observed/(1+redshift).
    t1_kelvin : float
        Background temperature at a radius of 1 rest-frame light-day, in K.
        Required explicitly; no object-specific temperature is assumed.
    inclination_deg : float, default 45
        Disk inclination: 0 is face-on; must be less than 90 degrees.
    redshift : float, default 0
        Applies time dilation AND the inverse factor to the density. It does
        not convert the supplied wavelength a second time.
    n_phi : int, default 128
        Azimuth quadrature order. Check convergence near edge-on inclinations.

    Returns
    -------
    float or ndarray
        Density psi(tau) in day^-1, with the same shape as tau_days.
        Integral from 0 to infinity equals 1. No sampled-grid normalization
        is performed: truncation or coarse sampling can lose response mass.

    Notes
    -----
    Let s=(1+z)*R_lambda/c in days, g=1-sin(i)*cos(phi), y=tau/(s*g).
    Then psi(tau)=<p(y)/(s*g)>_phi. The 1/(s*g) Jacobian is essential.
    This implements Delta f = amplitude * integral psi(tau)*Delta L(t-tau)dtau.
    Convolution with sampled densities requires the corresponding dt factor.
    """
    _, scale_days, inclination = _parameters(
        wavelength_rest_angstrom, t1_kelvin, inclination_deg, redshift
    )
    if isinstance(n_phi, (bool, np.bool_)) or not isinstance(n_phi, (int, np.integer)):
        raise ValueError("n_phi must be an integer >= 16")
    if n_phi < 16:
        raise ValueError("n_phi must be an integer >= 16")
    tau = np.asarray(tau_days, dtype=float)
    if np.any(~np.isfinite(tau)):
        raise ValueError("tau_days must contain only finite values")
    flat = tau.reshape(-1)
    response = np.zeros_like(flat)
    positive = np.flatnonzero(flat > 0)
    if inclination == 0:
        g, angular_weights = np.ones(1), np.ones(1)
    else:
        g, angular_weights = _azimuth_quadrature(inclination, int(n_phi))
    delay_scale = scale_days * g
    for start in range(0, positive.size, 2048):
        indices = positive[start:start + 2048]
        y = flat[indices, None] / delay_scale[None, :]
        x = y**_ALPHA
        # exp(-x) and expm1 avoid overflow and small-x cancellation.
        q = x * np.exp(-x) / (-np.expm1(-x))**2
        radial_pdf = y * q / _RADIAL_NORMALIZATION
        response[indices] = (radial_pdf / delay_scale) @ angular_weights
    response = response.reshape(tau.shape)
    return float(response) if response.ndim == 0 else response


def disk_moments(
    *, wavelength_rest_angstrom, t1_kelvin, inclination_deg=45.0, redshift=0.0
):
    """Return analytic infinite-disk moments, independent of a sampled grid.

    r_lambda_light_days is a rest-frame radius defined by k*T(R_lambda)=hc/lambda.
    mean_delay_days and variance_days2 are observer-frame quantities.
    Mean = 5.04146...*(1+z)*R_lambda/c, independent of inclination.
    """
    r_lambda, scale_days, inclination = _parameters(
        wavelength_rest_angstrom, t1_kelvin, inclination_deg, redshift
    )
    mean = scale_days * _MEAN_RADIUS_FACTOR
    second = scale_days**2 * _SECOND_RADIUS_FACTOR * (
        1 + 0.5 * np.sin(np.deg2rad(inclination))**2
    )
    return {
        "area": 1.0,
        "r_lambda_light_days": float(r_lambda),
        "mean_delay_days": float(mean),
        "variance_days2": float(second - mean**2),
    }


def _self_check():
    """Check independent radial integrals and the delay-density mapping."""
    from scipy.integrate import quad

    # Direct integration in x tests the analytic normalization and moments.
    def radial_integrand(x, order):
        if x == 0:
            return 0.0
        return (x**((order + 2)/_ALPHA) * np.exp(-x)
                / (-np.expm1(-x))**2 / _ALPHA)

    radial = [quad(radial_integrand, 0, np.inf, args=(j,), epsabs=1e-9)[0]
              for j in range(3)]
    np.testing.assert_allclose(radial[0], _RADIAL_NORMALIZATION, rtol=1e-9)
    np.testing.assert_allclose(radial[1]/radial[0], _MEAN_RADIUS_FACTOR, rtol=1e-9)
    np.testing.assert_allclose(radial[2]/radial[0], _SECOND_RADIUS_FACTOR, rtol=1e-9)
    print(f"Analytic response centroid coefficient: {_MEAN_RADIUS_FACTOR:.9f}")

    base = dict(wavelength_rest_angstrom=1928.0, t1_kelvin=20000.0)
    for inclination in (0.0, 45.0, 75.0):
        params = dict(base, inclination_deg=inclination, redshift=0.031)
        truth = disk_moments(**params)
        measured = [quad(
            lambda t: t**j * disk_transfer_function(t, **params),
            0, np.inf, epsabs=1e-9, epsrel=1e-9,
        )[0] for j in range(3)]
        expected = [1.0, truth["mean_delay_days"],
                    truth["variance_days2"] + truth["mean_delay_days"]**2]
        np.testing.assert_allclose(measured, expected, rtol=2e-7, atol=1e-9)
        probe = np.geomspace(1e-5, 100.0, 1001)
        values = disk_transfer_function(probe, **params)
        np.testing.assert_allclose(values, disk_transfer_function(probe, n_phi=256, **params),
                                   rtol=1e-6, atol=1e-11)
        assert np.all(values >= 0)
        print(f"i={inclination:4.0f} deg: area={measured[0]:.10f}, "
              f"centroid={measured[1]:.9f} days; moments and quadrature PASS")

    t = np.geomspace(1e-4, 30, 1000)
    p = disk_transfer_function(t, **base)
    stretched = disk_transfer_function(t * 1.5, redshift=0.5, **base)
    np.testing.assert_allclose(stretched * 1.5, p, rtol=1e-12, atol=1e-14)
    bigger = dict(base, wavelength_rest_angstrom=2*base["wavelength_rest_angstrom"])
    stretch = 2**(4/3)
    np.testing.assert_allclose(
        disk_transfer_function(t * stretch, **bigger) * stretch, p,
        rtol=1e-12, atol=1e-14,
    )
    assert disk_transfer_function(-1.0, **base) == 0.0
    assert disk_transfer_function(0.0, **base) == 0.0
    assert disk_transfer_function(np.zeros((2, 3)), **base).shape == (2, 3)
    assert temperature_profile(1.0, 20000.0) == 20000.0
    # A short evaluation window must retain its missing tail, not hide it.
    short_area = quad(lambda t: disk_transfer_function(t, **base), 0, 0.1)[0]
    assert 0 < short_area < 0.5
    print("Causality, positivity, wavelength/redshift scaling, and finite-window checks PASS")


if __name__ == "__main__":
    _self_check()
