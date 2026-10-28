from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal


SCRIPT = Path(__file__).resolve().parent / "non-response" / "drw_test.py"
SPEC = importlib.util.spec_from_file_location("drw_likelihood_under_test", SCRIPT)
drw = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = drw
SPEC.loader.exec_module(drw)


class DRWLikelihoodTests(unittest.TestCase):
    @staticmethod
    def make_data(scale=1.0):
        flux = scale * np.array([-2, -1.5, -1, -0.5, 0, 0, 0.5, 1, 1.5, 2])
        times = np.array([0, 0.4, 1.2, 2.1, 4, 6.3, 9, 12, 16, 21])
        errors = np.full(flux.size, 0.1 * scale)
        return drw.prepare_data(times, flux, errors, verbose=False)

    def test_both_fit_entry_points_return_log_density(self):
        # These scales give positive and negative log densities, respectively.
        for scale in (0.03, 1.0):
            data = self.make_data(scale)
            variance = float(np.mean(data.y**2))
            jitter = np.sqrt(variance - data.yerr[0] ** 2)
            expected = multivariate_normal.logpdf(
                data.y, mean=np.zeros(data.n), cov=variance * np.eye(data.n)
            )
            for warmstart in (False, True):
                with self.subTest(scale=scale, warmstart=warmstart):
                    likelihood = drw.WhiteNoiseLikelihood(data)
                    if warmstart:
                        starts = [np.log([jitter * factor]) for factor in (0.5, 2)]
                        fit = drw.fit_model_warmstart(likelihood, starts, maxiter=200)
                    else:
                        fit = drw.fit_model(likelihood, data, maxiter=200)
                    self.assertAlmostEqual(fit.jitter, jitter, delta=1e-6 * scale)
                    self.assertAlmostEqual(fit.ln_like, expected, places=8)

    def test_information_criteria_reward_necessary_extra_variance(self):
        data = self.make_data()
        fixed = drw.fit_model(drw.WhiteNoiseLikelihood(data, fit_jitter=False), data)
        fitted = drw.fit_model(drw.WhiteNoiseLikelihood(data), data, maxiter=200)
        drw.information_criteria([fixed, fitted], data.n)

        # The mean counts as one parameter in both models; jitter adds one.
        for fit, variance, k in (
            (fixed, data.yerr[0] ** 2, 1),
            (fitted, float(np.mean(data.y**2)), 2),
        ):
            expected = multivariate_normal.logpdf(
                data.y, mean=np.zeros(data.n), cov=variance * np.eye(data.n)
            )
            self.assertAlmostEqual(fit.aic, 2 * k - 2 * expected, places=8)
            self.assertAlmostEqual(fit.bic, k * np.log(data.n) - 2 * expected, places=8)
        self.assertGreater(fitted.ln_like, fixed.ln_like)
        self.assertLess(fitted.aic, fixed.aic)
        self.assertLess(fitted.bic, fixed.bic)
        self.assertGreater(drw.aic_weights([fixed, fitted])[1], 0.99)

    def test_ou_negative_log_likelihood_matches_gaussian_density(self):
        data = self.make_data()
        separations = np.abs(data.t[:, None] - data.t[None, :])
        backends = [False, True] if drw._HAVE_CELERITE else [False]
        for sigmas, taus in (([0.8], [3.0]), ([0.8, 0.4], [3.0, 10.0])):
            jitter = 0.2
            covariance = np.diag(data.yerr**2 + jitter**2)
            for sigma, tau in zip(sigmas, taus):
                covariance += sigma**2 * np.exp(-separations / tau)
            expected = multivariate_normal.logpdf(
                data.y, mean=np.zeros(data.n), cov=covariance
            )
            for use_celerite in backends:
                with self.subTest(n_terms=len(sigmas), use_celerite=use_celerite):
                    likelihood = drw.OULikelihood(
                        data, n_terms=len(sigmas), fit_jitter=True,
                        use_celerite=use_celerite,
                    )
                    theta = likelihood.pack(sigmas, taus, jitter)
                    self.assertAlmostEqual(likelihood.neg_log_like(theta), -expected, places=8)
                    if use_celerite:
                        self.assertFalse(likelihood._celerite_failed)


class DRWStructureFunctionTests(unittest.TestCase):
    def test_short_lag_sf_slope_tends_to_one_half(self):
        tau = 30.0
        lags = tau * np.geomspace(1e-7, 1e-4, 20)
        sf = drw.sf_model(lags, sigma=2.0, tau=tau)
        table = {"dt": lags, "sf": sf, "ok": np.ones(lags.size, dtype=bool)}
        self.assertAlmostEqual(drw.short_lag_slope(table, tau), 0.5, delta=1e-5)
        # Squaring SF, in contrast, gives a limiting slope of one.
        squared_table = {**table, "sf": sf**2}
        self.assertAlmostEqual(drw.short_lag_slope(squared_table, tau), 1.0, delta=1e-5)

    def test_sf_verdict_uses_one_half_in_both_branches(self):
        fit = drw.FitResult(
            label="fixture", n_params=0, k_total=1, ln_like=0.0,
            theta=np.array([]), params={}, bic=0.0,
        )
        result = {
            "fits": {key: fit for key in ("wn", "drw", "drw_jitter")},
            "p_gof": np.nan, "p_sf": 0.5, "sf_slope_short": 0.43,
            "p_jitter": np.nan, "jitter": 0.0, "baseline": 10.0, "tau": 1.0,
            "lb_p": np.nan, "z_std": np.nan,
        }
        for p_sf in (0.01, 0.5):
            with self.subTest(p_sf=p_sf):
                result["p_sf"] = p_sf
                verdict = drw._make_verdict(result)
                sf_lines = [line for line in verdict if "结构函数" in line]
                self.assertEqual(len(sf_lines), 1)
                self.assertRegex(sf_lines[0], r"DRW.*0\.5")


if __name__ == "__main__":
    unittest.main()
