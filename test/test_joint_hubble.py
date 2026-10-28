"""Fast numerical/regression checks; no access to production fit outputs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joint_hubble as jh


class JointHubbleTests(unittest.TestCase):
    def test_distance_convention(self):
        for z in [0.0032, 0.03, 0.046]:
            self.assertAlmostEqual(jh.distance_scale(z, 0, "comoving"), jh.C_KMS * z)
            self.assertAlmostEqual(jh.distance_scale(z, 0.3, "luminosity") /
                                   jh.distance_scale(z, 0.3, "comoving"), 1 + z)
        self.assertEqual(jh.velocity_ln_sigma(0.03, 0.3, "comoving", 0), 0)
        with self.assertRaises(ValueError):
            jh.distance_scale(0.03, 0.3, "unknown")

    def test_duplicate_epochs_rejected(self):
        jh.validate_sources(["Mrk817_a", "F9_a"])
        for sources in [["Mrk817", "Mrk817_a"], ["NGC2617_a", "NGC2617_b"], []]:
            with self.assertRaises(ValueError):
                jh.validate_sources(sources)

    def test_common_covariance_does_not_shrink_by_band_count(self):
        n = 6
        c = 0.04 * np.ones((n, n)) + 0.0001 * np.eye(n)
        mixture = jh.BandMixture.from_summary(np.repeat(4.0, n), c, (0, 0))
        self.assertAlmostEqual(mixture.mean[0], 4.0)
        self.assertAlmostEqual(mixture.variance[0], 0.04 + 0.0001 / n)
        self.assertGreater(mixture.variance[0], 5.9 * c[0, 0] / n)

    def test_common_scatter_matches_full_covariance(self):
        m = np.array([3.8, 4.1, 4.2])
        c = np.diag([0.03, 0.05, 0.07]) + 0.02 * np.ones((3, 3))
        f, extra = 0.12, 0.2**2
        mixture = jh.BandMixture.from_summary(m, c, (f, f))
        mus = np.array([3.4, 4.0, 4.7])
        direct = np.array([multivariate_normal.logpdf(m, mean=np.full(3, mu),
                           cov=c + f*f*np.eye(3) + extra*np.ones((3, 3))) for mu in mus])
        reduced = mixture.log_likelihood(mus, extra)
        np.testing.assert_allclose(reduced - reduced[0], direct - direct[0], atol=2e-12)

    def test_band_scatter_quadrature_matches_direct_integration(self):
        m = np.array([3.8, 4.1, 4.2])
        c = np.diag([0.03, 0.05, 0.07]) + 0.02 * np.ones((3, 3))
        mixture = jh.BandMixture.from_summary(m, c, (0, 1), 128)
        mus = np.array([3.4, 4.0, 4.7])
        direct = np.array([np.log(quad(lambda f: multivariate_normal.pdf(
            m, mean=np.full(3, mu), cov=c + f*f*np.eye(3) + 0.04*np.ones((3, 3))),
            0, 1, epsabs=1e-11, epsrel=1e-11)[0]) for mu in mus])
        reduced = mixture.log_likelihood(mus, 0.04)
        np.testing.assert_allclose(reduced - reduced[0], direct - direct[0], atol=2e-10)

    def test_flat_h0_prior_applied_once(self):
        h0, scatter = np.linspace(10, 250, 10001), np.array([0.0])
        sigma = 0.1
        data = {"mixture": jh.BandMixture.from_summary([np.log(100)], [[sigma**2]], (0, 0)),
                "scale": 7000.0, "velocity": 0.0}
        log_grid = jh.source_log_grid(data, h0, scatter)
        fit = jh.fit_joint(log_grid, h0, scatter)
        self.assertAlmostEqual(float(np.sum(fit["h0_pdf"] * jh.integration_weights(h0))), 1.0)
        self.assertAlmostEqual(fit["h0_q"][2], 70 * np.exp(sigma**2), places=4)
        np.testing.assert_allclose(fit["fixed_pdf"], fit["h0_pdf"], atol=1e-14)
        # Two independent sources halve the log variance; not the old H0 prior twice.
        joint = jh.fit_joint(log_grid * 2, h0, scatter)
        self.assertAlmostEqual(joint["h0_q"][2], 70 * np.exp(sigma**2 / 2), places=4)

    def test_log_grid_still_means_flat_h0_prior(self):
        h0, scatter = np.geomspace(1, 800, 10001), np.array([0.0])
        data = {"mixture": jh.BandMixture.from_summary([np.log(100)], [[0.01]], (0, 0)),
                "scale": 7000.0, "velocity": 0.0}
        fit = jh.fit_joint(jh.source_log_grid(data, h0, scatter), h0, scatter)
        self.assertAlmostEqual(fit["h0_q"][2], 70 * np.exp(0.01), places=4)

    def test_scatter_density_and_joint_density_normalized(self):
        h0, scatter = np.linspace(20, 150, 501), np.linspace(0, 0.8, 71)
        m = jh.BandMixture.from_summary([4, 4.1], [[0.05, 0.04], [0.04, 0.08]])
        log_grid = jh.source_log_grid({"mixture": m, "scale": 5000, "velocity": 0.03}, h0, scatter)
        fit = jh.fit_joint(log_grid, h0, scatter)
        self.assertAlmostEqual(np.sum(fit["scatter_pdf"] * jh.integration_weights(scatter)), 1)
        self.assertAlmostEqual(np.sum(fit["h0_pdf"] * jh.integration_weights(h0)), 1)
        self.assertTrue(np.all(np.diff(fit["h0_q"]) > 0))

    def test_bad_covariance_rejected(self):
        for c in [[[1, 1], [1, 1]], [[1, 0.2], [0.1, 1]], [[1, 0], [0, -1]]]:
            with self.assertRaises(ValueError):
                jh.BandMixture.from_summary([1, 2], c)

    def test_loader_aligns_labels_and_is_read_only(self):
        with tempfile.TemporaryDirectory(prefix="joint_hubble_test_") as temp:
            root = Path(temp)
            source = root / "AGN_a"
            folder = source / "runs/result/comp0/distance/standard"
            folder.mkdir(parents=True)
            (source / "config").mkdir()
            (source / "config/source_config.yaml").write_text("redshift: 0.03\n")
            pd.DataFrame({"band": ["U", "V"], "mean_log_D": [4.0, 4.1]}).to_csv(folder / "log_distance_mean.csv", index=False)
            covariance = pd.DataFrame([[0.06, 0.03], [0.03, 0.05]], index=["V", "U"], columns=["V", "U"])
            covariance.to_csv(folder / "log_distance_covariance.csv")
            pd.DataFrame([dict(source="AGN_a", distance_model="standard", n_sample=20,
                               inclination_deg=45)]).to_csv(folder / "distance_run_summary.csv", index=False)
            cfg = {"bands": "all", "inference": {"band_scatter_range": [0, 0], "band_scatter_nodes": 128},
                   "cosmology": {"distance_kind": "comoving", "om0": 0.3, "peculiar_velocity_kms": 250},
                   "quality": {"warn_distance_samples_below": 100}}
            before = sorted(root.rglob("*"))
            data = jh.load_source(root, "AGN_a", "test", "runs/result/comp0", "standard", cfg)
            reference = jh.BandMixture.from_summary([4, 4.1], [[0.05, 0.03], [0.03, 0.06]], (0, 0))
            np.testing.assert_allclose(data["mixture"].mean, reference.mean)
            np.testing.assert_allclose(data["mixture"].variance, reference.variance)
            self.assertEqual(before, sorted(root.rglob("*")))
            self.assertTrue(any("20 saved" in w for w in data["meta"]["warnings"]))
            json.dumps(data["meta"], allow_nan=False)
            row = jh.source_row(data)
            self.assertAlmostEqual(row["DL_q50"] / row["D_saved_q50"], 1.03)
            with self.assertRaises(FileNotFoundError):
                jh.load_source(root, "AGN_a", "test", "runs/missing", "standard", cfg)
            with self.assertRaises(ValueError):
                jh.under(root, "../outside")


if __name__ == "__main__":
    unittest.main()
