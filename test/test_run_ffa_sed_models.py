from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_ffa  # noqa: E402


class WeightedPowerLawTests(unittest.TestCase):
    def setUp(self):
        self.waves = np.array([1928.0, 2236.0, 2600.0, 3467.0, 4392.0, 5468.0])
        self.pivot = float(np.exp(np.mean(np.log(self.waves))))
        self.x = np.log(self.waves / self.pivot)
        self.sigma = np.full(len(self.waves), 0.05)

    def test_fixed_standard_slope_recovers_normalization(self):
        alpha = -4.0 / 3.0
        normalizations = np.array([2.0, 3.5, 8.0])
        log_y = np.log(
            normalizations[:, None] * (self.waves[None, :] / self.pivot) ** alpha
        )
        norm, slope, chi2 = run_ffa._weighted_log_powerlaw(
            log_y, self.x, self.sigma, alpha
        )
        np.testing.assert_allclose(norm, normalizations, rtol=1.0e-12)
        np.testing.assert_allclose(slope, alpha, rtol=1.0e-12)
        np.testing.assert_allclose(chi2, 0.0, atol=1.0e-24)

    def test_free_slope_recovers_each_joint_draw(self):
        normalizations = np.array([1.5, 4.0, 9.0])
        alphas = np.array([-2.0, -4.0 / 3.0, 0.25])
        log_y = np.log(
            normalizations[:, None]
            * (self.waves[None, :] / self.pivot) ** alphas[:, None]
        )
        norm, slope, chi2 = run_ffa._weighted_log_powerlaw(
            log_y, self.x, self.sigma, None
        )
        np.testing.assert_allclose(norm, normalizations, rtol=1.0e-12)
        np.testing.assert_allclose(slope, alphas, rtol=1.0e-12)
        np.testing.assert_allclose(chi2, 0.0, atol=1.0e-24)

    def test_nonpositive_uncertainty_is_rejected(self):
        with self.assertRaises(ValueError):
            run_ffa._weighted_log_powerlaw(
                np.zeros((2, len(self.x))), self.x, np.zeros_like(self.x), None
            )


class SedWorkflowTests(unittest.TestCase):
    def test_only_joint_posterior_sed_step_is_exposed(self):
        self.assertIn("sedfit", run_ffa.STEPS)
        self.assertNotIn("plfit", run_ffa.STEPS)
        with self.assertRaises(ValueError):
            run_ffa.parse_steps("plfit", {})

    def test_total_metadata_has_no_legacy_branch(self):
        metadata = run_ffa._sed_product_metadata({
            "component": {"name": "total"},
        })
        self.assertEqual(metadata["component"], "total")
        self.assertEqual(metadata["product_id"], "total")
        self.assertEqual(metadata["upstream_status"], "not_applicable")

    def test_posterior_branch_metadata_is_preserved(self):
        metadata = run_ffa._sed_product_metadata({
            "component": {"name": "comp0"},
            "posterior_branch": {
                "id": "disk_like",
                "upstream_status": "warning",
                "warning_bands": "U",
            },
        })
        self.assertEqual(metadata["component"], "comp0")
        self.assertEqual(metadata["product_id"], "disk_like")
        self.assertEqual(metadata["warning_bands"], "U")

    def test_plot_text_uses_source_only_title(self):
        self.assertEqual(
            run_ffa._sed_plot_text("Mrk817", "total"),
            ("Mrk817", "FFA total"),
        )
        self.assertEqual(
            run_ffa._sed_plot_text("Mrk817", "comp0"),
            ("Mrk817", "FFA component 0"),
        )


class BranchConfigTests(unittest.TestCase):
    def test_existing_branch_quality_is_propagated(self):
        source = ROOT / "Mrk142"
        summary = (
            source
            / "runs/mica_gaussian/2comp/result/branches/q0_ge_3"
            / "decomposition_summary.csv"
        )
        if not summary.is_file():
            self.skipTest("Mrk142 branch decomposition summary is not available")
        config = run_ffa.read_yaml(source / "config/source_config.yaml")
        ff = run_ffa.posterior_branch_fflux_config(
            source, config, "fflux_comp0", "q0_ge_3"
        )
        self.assertEqual(ff["posterior_branch"]["upstream_status"], "warning")
        self.assertIn("M2", ff["posterior_branch"]["warning_bands"])
        self.assertIn("q0_ge_3", ff["out"])
        self.assertNotIn("nuc", ff["posterior_branch_steps"])
        self.assertNotIn("plfit", ff["posterior_branch_steps"])
        self.assertIn("sedfit", ff["posterior_branch_steps"])
        self.assertIn("q0_ge_3", ff["mica"]["root"])


if __name__ == "__main__":
    unittest.main()
