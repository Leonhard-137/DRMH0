from __future__ import annotations

import sys
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


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

    def test_plot_text_omits_target_title(self):
        self.assertEqual(
            run_ffa._sed_plot_text("Mrk817", "total"),
            ("", "FFA total RMS amplitude"),
        )
        self.assertEqual(
            run_ffa._sed_plot_text("Mrk817", "comp0"),
            ("", "FFA component 0 RMS amplitude"),
        )

    def test_joint_sed_uses_r_not_bright_faint_amplitude(self):
        posterior = pd.DataFrame({
            "R_W2": [1.0, 2.0, 3.0],
            "R_U": [4.0, 5.0, 6.0],
            "Delta_F_W2": [100.0, 200.0, 300.0],
            "Delta_F_U": [400.0, 500.0, 600.0],
        })
        samples = run_ffa._joint_rms_sed_samples(
            posterior,
            ["W2", "U"],
            np.array([2.0, 3.0]),
        )
        np.testing.assert_allclose(
            samples,
            np.array([[2.0, 12.0], [4.0, 15.0], [6.0, 18.0]]),
        )

    def test_powerlaw_envelope_retains_normalization_at_pivot(self):
        pivot = 3000.0
        wavelengths = np.array([1500.0, pivot, 6000.0])
        normalizations = np.array([1.0, 10.0, 100.0])
        slopes = np.array([-2.0, -1.0, 0.0])
        lo, med, hi = run_ffa._powerlaw_posterior_envelope(
            normalizations, slopes, wavelengths, pivot
        )
        expected = np.percentile(normalizations, [16.0, 50.0, 84.0])
        np.testing.assert_allclose([lo[1], med[1], hi[1]], expected)
        self.assertLess(lo[0], hi[0])
        self.assertLess(lo[2], hi[2])


class MicaResponseSupportTests(unittest.TestCase):
    def test_padded_reconstruction_is_clipped_and_interpolated(self):
        data = np.array([
            [-2.0, 0.0, 0.2],
            [-1.0, 1.0, 0.3],
            [0.0, 2.0, 0.4],
            [1.0, 3.0, 0.5],
            [2.0, 4.0, 0.6],
        ])
        clipped = run_ffa.clip_light_curve_to_support(
            data, -0.5, 1.5, Path("pline.txt_2_comp0")
        )
        np.testing.assert_allclose(clipped[:, 0], [-0.5, 0.0, 1.0, 1.5])
        np.testing.assert_allclose(clipped[:, 1], [1.5, 2.0, 3.0, 3.5])
        np.testing.assert_allclose(clipped[:, 2], [0.35, 0.4, 0.5, 0.55])

    def test_reconstruction_must_span_driver_support(self):
        data = np.array([
            [0.0, 1.0, 0.1],
            [1.0, 2.0, 0.1],
        ])
        with self.assertRaisesRegex(ValueError, "does not span the driver support"):
            run_ffa.clip_light_curve_to_support(
                data, -0.1, 1.0, Path("pline.txt_2_comp0")
            )


class BranchConfigTests(unittest.TestCase):
    def test_relabelled_free_components_are_rejected_for_direct_ffa(self):
        config = {
            "source": "Synthetic",
            "reference_band": "W2",
            "bands": {"W2": 1928.0, "U": 3465.0},
            "fflux_comp0": {"component": {"name": "comp0"}},
            "mica_gamma2_free": {
                "output_root": "runs/mica/gamma/free",
                "driver": "W2",
                "responses": ["U"],
                "ncomp": 2,
                "type_tf": "gamma",
                "component_order": "moment1",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "cannot be relabelled"):
                run_ffa.mica_component_fflux_config(
                    Path(directory), config, "fflux_comp0", "mica_gamma2_free"
                )

    def test_native_mix_comp0_is_routed_inside_stage_result(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            fit = {
                "output_root": "runs/mica/mix220/phys12_slow3",
                "prepared_dir": "fflux/host_sub",
                "driver": "W2",
                "responses": ["U", "B"],
                "number_component": [3, 3],
                "type_tf": "220",
                "run_template": "run_{driver}_to_{band}_{ncomp}comp_{type_tf}",
            }
            config = {
                "source": "Synthetic",
                "reference_band": "W2",
                "bands": {"W2": 1928.0, "U": 3465.0, "B": 4392.0},
                "fflux_comp0": {"component": {"name": "comp0"}},
                "mica_mix220": fit,
            }
            run_root = source / "runs/mica/mix220/phys12_slow3"
            for band in ("U", "B"):
                (run_root / f"run_W2_to_{band}_3comp_mix220").mkdir(parents=True)
            ff = run_ffa.mica_component_fflux_config(
                source, config, "fflux_comp0", "mica_mix220", "comp0"
            )
            expected = run_root / "result/comp0/fflux"
            self.assertEqual(Path(ff["out"]), expected)
            self.assertEqual(ff["mica"]["type_tf"], "mix220")
            self.assertEqual(ff["mica"]["ncomp"], 3)
            self.assertEqual(ff["mica"]["component_index"], 0)
            self.assertEqual(ff["mica_component"]["selection"], "full_native_posterior")
            self.assertFalse(expected.exists())

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


class FitRestartValidationTests(unittest.TestCase):
    @staticmethod
    def _write_rows(path: Path, count: int) -> None:
        path.write_text("# header\n" + "0\n" * count, encoding="utf-8")

    def test_matching_checkpoint_and_samples_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit = Path(tmp)
            checkpoint = fit / "restart_dnest.txt_2"
            checkpoint.write_bytes(struct.pack("@i", 2) + b"checkpoint")
            self._write_rows(fit / "sample.txt", 2)
            self._write_rows(fit / "sample_info.txt", 2)
            path, count = run_ffa.validate_fit_restart(fit, checkpoint.name, 10)
            self.assertEqual(path, checkpoint)
            self.assertEqual(count, 2)

    def test_mismatched_append_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit = Path(tmp)
            checkpoint = fit / "restart_dnest.txt_2"
            checkpoint.write_bytes(struct.pack("@i", 2) + b"checkpoint")
            self._write_rows(fit / "sample.txt", 3)
            self._write_rows(fit / "sample_info.txt", 2)
            with self.assertRaisesRegex(ValueError, "has 3 data rows"):
                run_ffa.validate_fit_restart(fit, checkpoint.name, 10)

    def test_checkpoint_at_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit = Path(tmp)
            checkpoint = fit / "restart_dnest.txt_10"
            checkpoint.write_bytes(struct.pack("@i", 10) + b"checkpoint")
            self._write_rows(fit / "sample.txt", 10)
            self._write_rows(fit / "sample_info.txt", 10)
            with self.assertRaisesRegex(ValueError, "not below configured"):
                run_ffa.validate_fit_restart(fit, checkpoint.name, 10)


if __name__ == "__main__":
    unittest.main()
