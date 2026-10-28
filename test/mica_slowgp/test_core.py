"""Independent numerical checks for the isolated two-band slow-GP core.

Run with PYTHONPATH=/home/leonhard/software/MICA2-slowgp/src and Python -B.
Only tiny synthetic arrays are used. No MICA fit, simulator, or data file is run.
The covariance oracle uses numerical Fourier integration, not the production
Gaussian/OU closed form; the conditional oracle uses dense Gaussian algebra.
"""
from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

SOURCE = Path(os.environ.get(
    "MICA_SLOWGP_SRC", "/home/leonhard/software/MICA2-slowgp/src"))
sys.path.insert(0, str(SOURCE))
from mica_slowgp import core  # noqa: E402


def parameters(**updates):
    result = dict(sigma_x=0.8, tau_x=3.7, A0=0.9, mu0=0.65,
                  w0=0.28, A1=0.43, mu1=2.3, w1=1.05,
                  qW=0.31, qb=0.47, Ts=8.4)
    result.update(updates)
    return result


def data_fixture(yw=None, yb=None):
    return core.PairData(
        np.array([0.0, 0.8, 2.7, 5.0]),
        np.array([1.2, 0.75, 1.1, 1.35]) if yw is None else np.asarray(yw),
        np.array([0.09, 0.12, 0.07, 0.11]),
        np.array([0.1, 1.4, 3.2, 6.0]),
        np.array([2.1, 2.25, 1.8, 2.4]) if yb is None else np.asarray(yb),
        np.array([0.13, 0.08, 0.14, 0.1]),
    )


@lru_cache(maxsize=4096)
def spectral_ou(delta, width, tau):
    """OU spectrum integrated after a Gaussian response, independent oracle.

    k(d)=integral_0^inf [2 tau/pi/(1+(tau*w)^2)] cos(w*d) dw.
    Gaussian smoothing multiplies the spectrum by exp(-width^2*w^2/2).
    The zero-width identity avoids slowly converging oscillatory quadrature.
    """
    if width == 0:
        return math.exp(-abs(delta) / tau)
    value, error = quad(
        lambda omega: (2 * tau / math.pi
                       / (1 + (tau * omega) ** 2)
                       * math.exp(-0.5 * (width * omega) ** 2)
                       * math.cos(omega * delta)),
        0.0, 12.0 / width, epsabs=2e-11, epsrel=2e-10, limit=300,
    )
    if error > 2e-8:
        raise AssertionError(f"Independent quadrature did not converge: {error}")
    return value


def direct_normal_ou(delta, width, tau):
    """Direct normal-expectation integral with its OU cusp split explicitly."""
    if width == 0:
        return math.exp(-abs(delta) / tau)
    integrand = lambda z: (math.exp(-z*z/2 - abs(delta + width*z)/tau)
                          / math.sqrt(2*math.pi))
    cusp = -delta / width
    pieces = [-11.0, 11.0]
    if -11 < cusp < 11:
        pieces.insert(1, cusp)
    return sum(quad(integrand, lo, hi, epsabs=2e-12, epsrel=2e-11)[0]
               for lo, hi in zip(pieces[:-1], pieces[1:]))


def basis(component, p):
    if component in ("x", "w"):
        return [(1.0, 0.0, 0.0)]
    if component == "disk":
        return [(p["A0"], p["mu0"], p["w0"])]
    if component == "blr":
        return [(p["A1"], p["mu1"], p["w1"])]
    if component == "b":
        return basis("disk", p) + basis("blr", p)
    if component in ("slow_w", "slow_b"):
        return []
    raise ValueError(component)


def slow_band(component):
    return {"w": 0, "slow_w": 0, "b": 1, "slow_b": 1}.get(component)


def oracle_component(ta, ca, tb, cb, p, mode):
    ta, tb = np.asarray(ta), np.asarray(tb)
    result = np.zeros((len(ta), len(tb)))
    for aa, ma, wa in basis(ca, p):
        for ab, mb, wb in basis(cb, p):
            for i, t1 in enumerate(ta):
                for j, t2 in enumerate(tb):
                    result[i, j] += (p["sigma_x"] ** 2 * aa * ab
                        * spectral_ou(float(t1-t2-ma+mb), math.hypot(wa, wb),
                                      p["tau_x"]))
    ia, ib = slow_band(ca), slow_band(cb)
    if mode != "none" and ia is not None and ib is not None:
        if mode == "shared" or ia == ib:
            amplitudes = [p["qW"], p["qb"]]
            r = np.abs(ta[:, None] - tb[None, :]) / p["Ts"]
            result += amplitudes[ia]*amplitudes[ib]*(1+r)*np.exp(-r)
    return result


def oracle_observation_covariance(data, p, mode):
    ww = oracle_component(data.tw, "w", data.tw, "w", p, mode)
    wb = oracle_component(data.tw, "w", data.tb, "b", p, mode)
    bb = oracle_component(data.tb, "b", data.tb, "b", p, mode)
    return (np.block([[ww, wb], [wb.T, bb]])
            + np.diag(np.r_[data.ew**2, data.eb**2]))


def mean_design(data):
    nw, nb = len(data.tw), len(data.tb)
    result = np.zeros((nw + nb, 2))
    result[:nw, 0] = 1
    result[nw:, 1] = 1
    return result


def oracle_gls(data, covariance):
    """Flat-mean integral, with its exact relative normalization."""
    y = np.r_[data.yw, data.yb]
    design = mean_design(data)
    ki = np.linalg.inv(covariance)
    q = design.T @ ki @ design
    qinv = np.linalg.inv(q)
    mu = qinv @ design.T @ ki @ y
    residual = y - design @ mu
    loglike = -0.5*(residual @ ki @ residual
                    + np.linalg.slogdet(covariance)[1]
                    + np.linalg.slogdet(q)[1]
                    + (len(y)-2)*math.log(2*math.pi))
    return loglike, ki, qinv, mu


def oracle_prediction(data, p, mode, targets, include_means):
    k = oracle_observation_covariance(data, p, mode)
    _, ki, qinv, mu = oracle_gls(data, k)
    design = mean_design(data)
    y = np.r_[data.yw, data.yb]
    components = list(targets.values())
    cross = np.vstack([
        np.hstack([oracle_component(t, c, data.tw, "w", p, mode),
                   oracle_component(t, c, data.tb, "b", p, mode)])
        for c, t in components])
    prior = np.block([[oracle_component(t, c, u, d, p, mode)
                       for d, u in components] for c, t in components])
    offset_design = np.zeros((len(prior), 2))
    cursor = 0
    for c, t in components:
        if include_means and c in ("w", "b"):
            offset_design[cursor:cursor+len(t), 0 if c == "w" else 1] = 1
        cursor += len(t)
    mean = offset_design @ mu + cross @ ki @ (y - design @ mu)
    offset_coupling = offset_design - cross @ ki @ design
    conditional = (prior - cross @ ki @ cross.T
                   + offset_coupling @ qinv @ offset_coupling.T)
    return mean, conditional, mu, qinv, offset_coupling @ qinv


class CovarianceTests(unittest.TestCase):
    def test_gaussian_ou_matches_direct_normal_quadrature(self):
        cases = [(0., .3, 4.), (.7, .4, 1.8), (-2.1, 1.3, .7),
                 (0., 5., .2), (7., .03, 2.)]
        for delta, width, tau in cases:
            with self.subTest(delta=delta, width=width, tau=tau):
                expected = direct_normal_ou(delta, width, tau)
                actual = core.gaussian_ou_covariance(delta, width, tau)
                assert_allclose(actual, expected, rtol=3e-9, atol=2e-11)

    def test_gaussian_ou_limits_evenness_and_stability(self):
        delta = np.array([-1000., -4., -0.1, 0., .1, 4., 1000.])
        actual = core.gaussian_ou_covariance(delta, .7, 2.)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.all(actual >= 0))
        assert_allclose(actual, actual[::-1], rtol=2e-14, atol=1e-15)
        assert_allclose(core.gaussian_ou_covariance(delta, 0., 2.),
                        np.exp(-np.abs(delta)/2.), rtol=2e-14, atol=1e-15)
        assert_allclose(core.gaussian_ou_covariance(np.array([-.8, .8]), 1e-6, 2.),
                        np.exp(-.4), rtol=1e-10, atol=1e-12)

    def test_matern_timescale_is_not_standard_lengthscale(self):
        # At |d|=Ts the present convention is 2/e, not (1+sqrt(3))*exp(-sqrt(3)).
        actual = core.matern32_covariance(np.array([-8., 0., 8.]), 8.)
        assert_allclose(actual, [2/math.e, 1., 2/math.e], rtol=2e-14)

    def test_positive_delay_sign_and_nontrivial_cross_response(self):
        p = parameters()
        at_peak = core.component_covariance([0.], "x", [p["mu0"]], "disk", p, "none")
        wrong_sign = core.component_covariance([0.], "x", [-p["mu0"]], "disk", p, "none")
        self.assertGreater(float(at_peak[0, 0]), float(wrong_sign[0, 0]))
        t, u = np.array([-.4, 1.7]), np.array([.8, 2.9])
        actual = core.component_covariance(t, "disk", u, "blr", p, "none")
        expected = oracle_component(t, "disk", u, "blr", p, "none")
        assert_allclose(actual, expected, rtol=3e-9, atol=3e-10)
        self.assertGreater(np.min(actual), 0.01)

    def test_full_observation_matrix_includes_both_tf_cross_terms(self):
        data, p = data_fixture(), parameters()
        for mode in ("none", "independent", "shared"):
            with self.subTest(mode=mode):
                expected = oracle_observation_covariance(data, p, mode)
                actual = core.build_covariance(data, p, mode)
                assert_allclose(actual, expected, rtol=4e-9, atol=4e-10)
        bb = core.component_covariance([1.], "b", [1.], "b", p, "none")[0, 0]
        only_variances = sum(core.component_covariance([1.], c, [1.], c, p, "none")[0, 0]
                             for c in ("disk", "blr"))
        self.assertGreater(bb - only_variances, 0.1)

    def test_shared_slow_adds_cross_band_covariance_only(self):
        data, p = data_fixture(), parameters()
        independent = core.build_covariance(data, p, "independent")
        shared = core.build_covariance(data, p, "shared")
        nw = len(data.tw)
        difference = shared - independent
        assert_allclose(difference[:nw, :nw], 0, atol=1e-14)
        assert_allclose(difference[nw:, nw:], 0, atol=1e-14)
        r = np.abs(data.tw[:, None]-data.tb[None, :])/p["Ts"]
        assert_allclose(difference[:nw, nw:], p["qW"]*p["qb"]*(1+r)*np.exp(-r),
                        rtol=2e-13, atol=1e-14)

    def test_joint_component_prior_is_symmetric_positive_semidefinite(self):
        p = parameters()
        specs = [("x", [-.4, .9]), ("disk", [0., 2.1]),
                 ("blr", [.1, 3.]), ("slow_w", [0., .7]),
                 ("slow_b", [0., .7])]
        for mode in ("none", "independent", "shared"):
            covariance = np.block([
                [core.component_covariance(t, c, u, d, p, mode) for d, u in specs]
                for c, t in specs])
            assert_allclose(covariance, covariance.T, rtol=1e-13, atol=1e-14)
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -2e-12)

    def test_zero_slow_nests_none_exactly(self):
        data, p = data_fixture(), parameters(qW=0., qb=0.)
        baseline = core.build_covariance(data, p, "none")
        for mode in ("independent", "shared"):
            assert_allclose(core.build_covariance(data, p, mode), baseline, atol=0, rtol=0)
            self.assertAlmostEqual(core.log_likelihood(data, p, mode),
                                   core.log_likelihood(data, p, "none"), places=12)


class MarginalLikelihoodTests(unittest.TestCase):
    def test_exact_flat_mean_integral_including_mean_logdet(self):
        data = data_fixture()
        for mode in ("none", "independent", "shared"):
            for p in (parameters(), parameters(sigma_x=1.6, tau_x=.7, Ts=18.)):
                expected = oracle_gls(data, oracle_observation_covariance(data, p, mode))[0]
                assert_allclose(core.log_likelihood(data, p, mode), expected, rtol=1e-8, atol=2e-8)

    def test_independent_constant_flux_shifts_do_not_change_likelihood(self):
        data = data_fixture()
        shifted = data_fixture(data.yw+17., data.yb-23.)
        for mode in ("none", "independent", "shared"):
            assert_allclose(core.log_likelihood(data, parameters(), mode),
                            core.log_likelihood(shifted, parameters(), mode), rtol=0, atol=2e-10)

    def test_common_time_origin_does_not_change_likelihood(self):
        data = data_fixture()
        moved = core.PairData(data.tw+59000., data.yw, data.ew,
                             data.tb+59000., data.yb, data.eb)
        for mode in ("none", "independent", "shared"):
            assert_allclose(core.log_likelihood(data, parameters(), mode),
                            core.log_likelihood(moved, parameters(), mode), rtol=0, atol=3e-9)

    def test_covariance_scale_has_n_minus_two_normalization(self):
        data = data_fixture(np.repeat(1.7, 4), np.repeat(-.8, 4))
        scale = 3.2
        scaled_data = core.PairData(data.tw, data.yw, data.ew*scale,
                                    data.tb, data.yb, data.eb*scale)
        p = parameters()
        scaled = parameters(sigma_x=p["sigma_x"]*scale,
                            qW=p["qW"]*scale, qb=p["qb"]*scale)
        difference = (core.log_likelihood(scaled_data, scaled, "shared")
                      - core.log_likelihood(data, p, "shared"))
        assert_allclose(difference, -(len(data.tw)+len(data.tb)-2)*math.log(scale), atol=2e-11)


class ConditionalTests(unittest.TestCase):
    @staticmethod
    def targets():
        return {"x": ("x", np.array([.3, 2.])),
                "disk": ("disk", np.array([.9, 2.6])),
                "blr": ("blr", np.array([.9, 2.6])),
                "slow_w": ("slow_w", np.array([.3, 2.])),
                "slow_b": ("slow_b", np.array([.9, 2.6])),
                "w_total": ("w", np.array([.3, 2.])),
                "b_total": ("b", np.array([.9, 2.6]))}

    def test_joint_means_covariances_and_offset_cross_terms(self):
        data, p, targets = data_fixture(), parameters(), self.targets()
        for mode in ("none", "independent", "shared"):
            for include in (False, True):
                with self.subTest(mode=mode, include_means=include):
                    actual = core.condition(data, p, targets, mode, include_means=include)
                    expected = oracle_prediction(data, p, mode, targets, include)
                    for observed, reference in zip(
                            (actual.mean, actual.covariance, actual.offset_mean,
                             actual.offset_covariance, actual.offset_cross_covariance), expected):
                        assert_allclose(observed, reference, rtol=3e-8, atol=2e-9)
                    self.assertGreaterEqual(np.linalg.eigvalsh(actual.covariance).min(), -2e-11)

    def test_component_sum_preserves_covariance_and_offset_uncertainty(self):
        data, p, targets = data_fixture(), parameters(), self.targets()
        for mode in ("none", "independent", "shared"):
            actual = core.condition(data, p, targets, mode, include_means=True)
            n = len(actual.mean)
            joint_mean = np.r_[actual.mean, actual.offset_mean]
            joint_cov = np.block([[actual.covariance, actual.offset_cross_covariance],
                                  [actual.offset_cross_covariance.T, actual.offset_covariance]])
            for total, parts, offset in (("w_total", ("x", "slow_w"), 0),
                                         ("b_total", ("disk", "blr", "slow_b"), 1)):
                relation = np.zeros((2, n+2))
                relation[:, actual.slices[total]] = np.eye(2)
                for part in parts:
                    relation[:, actual.slices[part]] -= np.eye(2)
                relation[:, n+offset] = -1
                assert_allclose(relation @ joint_mean, 0, atol=2e-10)
                assert_allclose(relation @ joint_cov @ relation.T, 0, atol=2e-10)
            self.assertGreater(np.max(np.abs(actual.offset_cross_covariance)), 1e-4)

    def test_shared_slow_is_one_process_in_posterior_not_two_draws(self):
        data, p = data_fixture(), parameters()
        times = np.array([.5, 1.8, 4.2])
        targets = {"sw": ("slow_w", times), "sb": ("slow_b", times)}
        actual = core.condition(data, p, targets, "shared", include_means=False)
        relation = np.hstack([np.eye(3)/p["qW"], -np.eye(3)/p["qb"]])
        assert_allclose(relation @ actual.mean, 0, atol=2e-10)
        assert_allclose(relation @ actual.covariance @ relation.T, 0, atol=2e-10)

    def test_flat_means_agree_with_broad_proper_gaussian_limit(self):
        data, p = data_fixture(), parameters()
        targets = {"disk": ("disk", np.array([.5, 2.0])),
                   "total": ("b", np.array([.5, 2.0]))}
        actual = core.condition(data, p, targets, "independent", include_means=True)
        k = oracle_observation_covariance(data, p, "independent")
        design = mean_design(data)
        specs = list(targets.values())
        cross = np.vstack([np.hstack([
            oracle_component(t, c, data.tw, "w", p, "independent"),
            oracle_component(t, c, data.tb, "b", p, "independent")]) for c, t in specs])
        prior = np.block([[oracle_component(t, c, u, d, p, "independent")
                           for d, u in specs] for c, t in specs])
        offset_design = np.array([[0., 0.], [0., 0.], [0., 1.], [0., 1.]])
        variance = 1e5
        measured = k + variance * design @ design.T
        augmented_cross = cross + variance * offset_design @ design.T
        augmented_prior = prior + variance * offset_design @ offset_design.T
        finite_mean = augmented_cross @ np.linalg.solve(measured, np.r_[data.yw, data.yb])
        finite_cov = augmented_prior - augmented_cross @ np.linalg.solve(measured, augmented_cross.T)
        assert_allclose(actual.mean, finite_mean, rtol=0, atol=2e-5)
        assert_allclose(actual.covariance, finite_cov, rtol=0, atol=2e-5)

    def test_constant_shifts_change_only_requested_total_means(self):
        data, p, targets = data_fixture(), parameters(), self.targets()
        shifted = data_fixture(data.yw+4., data.yb-7.)
        a = core.condition(data, p, targets, "shared", include_means=True)
        b = core.condition(shifted, p, targets, "shared", include_means=True)
        expected = np.zeros_like(a.mean)
        expected[a.slices["w_total"]] = 4.
        expected[a.slices["b_total"]] = -7.
        assert_allclose(b.mean-a.mean, expected, atol=2e-10)
        assert_allclose(b.offset_mean-a.offset_mean, [4., -7.], atol=2e-10)
        assert_allclose(b.covariance, a.covariance, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
