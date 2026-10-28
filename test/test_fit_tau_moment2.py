from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fit_tau  # noqa: E402


class GaussianMomentTests(unittest.TestCase):
    def test_single_component_variance_is_width_squared(self):
        centers = np.array([[3.0, 5.0]])
        widths = np.array([[2.0, 4.0]])
        amplitudes = np.ones_like(centers)
        component, centroid, total, fractions = fit_tau.gaussian_mixture_moments(
            centers,
            widths,
            amplitudes,
        )
        np.testing.assert_allclose(component, [[4.0, 16.0]])
        np.testing.assert_allclose(centroid, [3.0, 5.0])
        np.testing.assert_allclose(total, [4.0, 16.0])
        np.testing.assert_allclose(fractions, [[1.0, 1.0]])

    def test_equal_centers_reduce_to_weighted_component_variance(self):
        centers = np.array([[2.0], [2.0]])
        widths = np.array([[1.0], [3.0]])
        amplitudes = np.array([[1.0], [3.0]])
        _, _, total, _ = fit_tau.gaussian_mixture_moments(
            centers,
            widths,
            amplitudes,
        )
        np.testing.assert_allclose(total, [7.0])

    def test_separated_centers_include_between_component_variance(self):
        centers = np.array([[0.0], [4.0]])
        widths = np.array([[1.0], [1.0]])
        amplitudes = np.array([[1.0], [1.0]])
        _, centroid, total, _ = fit_tau.gaussian_mixture_moments(
            centers,
            widths,
            amplitudes,
        )
        np.testing.assert_allclose(centroid, [2.0])
        np.testing.assert_allclose(total, [5.0])

    def test_vanishing_component_recovers_dominant_component(self):
        centers = np.array([[1.0], [100.0]])
        widths = np.array([[2.0], [20.0]])
        amplitudes = np.array([[1.0], [1.0e-14]])
        _, centroid, total, _ = fit_tau.gaussian_mixture_moments(
            centers,
            widths,
            amplitudes,
        )
        self.assertAlmostEqual(float(centroid[0]), 1.0, places=10)
        self.assertAlmostEqual(float(total[0]), 4.0, places=8)


class MixedMomentTests(unittest.TestCase):
    def test_mix220_uses_each_kernel_variance_and_preserves_comp0(self):
        cfg = {
            "_mica_stage": "mica_mix220",
            "response_bands": ["U"],
            "mica_mix220": {
                "driver": "W2",
                "responses": ["U"],
                "type_tf": "220",
                "number_component": [3, 3],
            },
        }
        rows = []
        values = [
            ("tf0", "gamma", 0.2, 0.1, 1.0),
            ("tf1", "gamma", 6.0, 0.5, 2.0),
            ("tf2", "gaussian", 60.0, 10.0, 1.0),
        ]
        for component, kind, tau, width, amp in values:
            rows.append({
                "run_name": "run_W2_to_U_3comp_mix220",
                "driver_band": "W2",
                "band": "U",
                "model": "mix220",
                "ncomp": 3,
                "selection": "all",
                "retained_weight": 1.0,
                "lambda_obs": 3465.0,
                "sample": 0,
                "component": component,
                "tf_kind": kind,
                "center": tau,
                "tau": tau,
                "width": width,
                "amp": amp,
                "amp_frac": amp / 4.0,
                "weight": 1.0,
            })
        result = fit_tau.transfer_moment2_posterior(pd.DataFrame(rows), cfg)
        by_component = result.set_index("component")
        self.assertAlmostEqual(float(by_component.loc["tf0", "moment2"]), 0.02)
        self.assertAlmostEqual(float(by_component.loc["tf1", "moment2"]), 0.5)
        self.assertAlmostEqual(float(by_component.loc["tf2", "moment2"]), 100.0)
        expected_centroid = 0.25 * 0.2 + 0.5 * 6.0 + 0.25 * 60.0
        self.assertAlmostEqual(
            float(by_component.loc["total", "moment1"]), expected_centroid
        )


class MomentWavelengthFitTests(unittest.TestCase):
    def test_fixed_gamma_recovers_scale(self):
        lambda0 = 1928.0
        waves = np.array([2236.0, 2600.0, 3467.0, 4392.0, 5468.0])
        for gamma in (8.0 / 3.0, 4.0):
            expected = fit_tau.moment2_model(waves, 0.35, gamma, lambda0)
            draws = np.tile(expected, (64, 1))
            recovered = fit_tau.fixed_beta_posterior(
                draws,
                waves,
                np.ones_like(waves),
                lambda0,
                gamma,
            )
            np.testing.assert_allclose(recovered, 0.35, rtol=1.0e-12)

    def test_joint_free_gamma_recovers_synthetic_relation(self):
        lambda0 = 1928.0
        waves = np.array([2236.0, 2600.0, 3467.0, 4392.0, 5468.0])
        truth = np.array([0.4, 3.2])
        median = fit_tau.moment2_model(waves, truth[0], truth[1], lambda0)
        error = np.full_like(waves, 0.04)
        error_high = error.copy()
        error_high[0] = 0.0
        samples, log_probability, diagnostics = fit_tau.joint_free_gamma_posterior(
            waves,
            median,
            error,
            error_high,
            lambda0,
            {
                "moment20_bounds": [1.0e-3, 10.0],
                "free_gamma_bounds": [0.2, 7.0],
                "moment20_prior": "log_uniform",
                "free_likelihood": "split_normal",
                "free_mcmc_nwalkers": 20,
                "free_mcmc_steps": 1200,
                "free_mcmc_burn_frac": 0.3,
                "free_mcmc_thin": 4,
                "free_mcmc_progress": False,
            },
            seed=24680,
        )
        recovered = np.median(samples, axis=0)
        self.assertAlmostEqual(float(recovered[0]), truth[0], delta=0.12)
        self.assertAlmostEqual(float(recovered[1]), truth[1], delta=0.7)
        self.assertTrue(np.isfinite(log_probability).all())
        self.assertEqual(diagnostics["inference"], "joint_mcmc")
        self.assertEqual(diagnostics["regularized_error_sides"], 1)
        self.assertGreater(diagnostics["acceptance_fraction"], 0.1)


if __name__ == "__main__":
    unittest.main()
