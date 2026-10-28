"""Numerical and input-contract checks; no MICA or FFA sampling."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mica_smooth_prepare", HERE / "prepare.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SmoothingTests(unittest.TestCase):
    def test_irregular_times_and_boundaries_preserve_a_constant(self):
        times = np.array([0., 0.1, 1., 20., 85.])
        weights = module.gaussian_weights(times, 20.)
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-15)
        np.testing.assert_allclose(weights @ np.full(5, 7.5), 7.5, atol=1e-14)
        self.assertLess(weights[0, 4], weights[0, 3])
        expected_ratio = np.exp(-0.5 * (20 / 20)**2)
        self.assertAlmostEqual(weights[0, 3] / weights[0, 0], expected_ratio)

    def test_fast_variability_is_retained_and_slow_variability_removed(self):
        t = np.arange(-300., 301.)
        slow = 4 * np.sin(2 * np.pi * t / 200.)
        fast = .5 * np.sin(2 * np.pi * t / 10.)
        data = np.column_stack([t, 20 + slow + fast, np.full(len(t), .2)])
        filtered, _ = module.detrend_curve(data, 20.)
        slow_gain = 1 - np.exp(-2 * np.pi**2 * 20**2 / 200**2)
        fast_gain = 1 - np.exp(-2 * np.pi**2 * 20**2 / 10**2)
        expected = 20 + slow_gain * slow + fast_gain * fast
        interior = np.abs(t) < 100
        np.testing.assert_allclose(filtered[interior, 1], expected[interior], atol=1e-11)
        self.assertLess(slow_gain, .2)
        self.assertGreater(fast_gain, .99)

    def test_noise_covariance_includes_subtraction_and_restored_sample_mean(self):
        data = np.array([[0., 5., .2], [2., 7., .4], [9., 4., .3]])
        filtered, covariance = module.detrend_curve(data, 4.)
        expected = np.zeros((3, 3))
        for j in range(len(data)):
            perturbed = data.copy()
            perturbed[j, 1] += 1
            response = module.detrend_curve(perturbed, 4.)[0][:, 1] - filtered[:, 1]
            expected += data[j, 2]**2 * np.outer(response, response)
        np.testing.assert_allclose(covariance, expected, atol=1e-15)
        np.testing.assert_array_equal(filtered[:, 2], data[:, 2])
        propagated, _ = module.detrend_curve(data, 4., "propagated")
        np.testing.assert_allclose(propagated[:, 2]**2, np.diag(covariance))

    def test_constant_baseline_does_not_change_fast_flux_differences(self):
        data = np.array([[0., 5., .2], [2., 7., .4], [9., 4., .3]])
        filtered, _ = module.detrend_curve(data, 4.)
        shifted = data.copy()
        shifted[:, 1] += 100
        result, _ = module.detrend_curve(shifted, 4.)
        np.testing.assert_allclose(np.diff(result[:, 1]), np.diff(filtered[:, 1]), atol=1e-13)
        np.testing.assert_allclose(result[:, 1] - filtered[:, 1], 100, atol=1e-13)
        data[:, 1] = 7.5
        constant, _ = module.detrend_curve(data, 4.)
        np.testing.assert_allclose(constant[:, 1], 7.5, atol=1e-14)

    def test_invalid_smoothing_widths_are_rejected(self):
        for sigma in (0, -1, np.nan, np.inf):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                module.gaussian_weights(np.array([1., 2.]), sigma)


class CaseTests(unittest.TestCase):
    def fixture(self, root):
        source = dict(source="Mrk817_a", reference_band="UVW2", driver_band="UVW2",
                      band_list=list(module.BANDS), response_bands=list(module.BANDS[1:]),
                      bands={band: 1900 + 600 * i for i, band in enumerate(module.BANDS)},
                      fflux={"extinction": {"ebv": .0058}}, fflux_comp0={}, lambda_tau={})
        for name, kind in zip(module.STAGES, ("gaussian", "gamma")):
            source[name] = dict(ncomp=2, type_tf=kind, lag_prior=[[0., 100.], [0., 100.]],
                                lag_limit=[0., 100.], width_prior=None,
                                component_order="moment1" if kind == "gamma" else "native",
                                output_root=f"runs/mica/{kind}/free", max_num_saves=10000,
                                ptol=.1, driver="UVW2", responses=list(module.BANDS[1:]))
        (root / "source.yaml").write_text(yaml.safe_dump(source))
        inp = root / "original"
        inp.mkdir()
        t = np.array([0., .2, 1.5, 30., 100.])
        for i, band in enumerate(module.BANDS):
            y = np.array([5., 7., 4., 9., 3.]) + i
            np.savetxt(inp / f"Mrk817_a_{band}.txt", np.column_stack([t, y, np.full(5, .2)]))
        experiment = dict(source_config="source.yaml", input_dir="original", cases_dir="cases",
                          error_mode="original", cases={
                              "responses_s20": dict(sigma_days=20., smooth_driver=False),
                              "all_s40": dict(sigma_days=40., smooth_driver=True)})
        config = root / "experiment.yaml"
        config.write_text(yaml.safe_dump(experiment))
        return config, inp

    def test_case_generation_preserves_bands_epochs_priors_and_driver_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, inp = self.fixture(root)
            with redirect_stdout(io.StringIO()):
                paths = module.prepare(config, make_plots=False)
            for case in paths:
                cfg = yaml.safe_load((case / "config/source_config.yaml").read_text())
                self.assertEqual(cfg["mica_stages"], list(module.STAGES))
                self.assertEqual(cfg["smoothing"]["output"], "highpass_residual_plus_original_mean")
                self.assertEqual(cfg["fflux"]["extinction"]["ebv"], 0)
                self.assertEqual(cfg["mica_gamma2_free"]["component_order"], "moment1")
                for stage in module.STAGES:
                    self.assertEqual(cfg[stage]["lag_prior"], [[0., 100.], [0., 100.]])
                    self.assertEqual(cfg[stage]["max_parallel"], 1)
                for band in module.BANDS:
                    original = np.loadtxt(inp / f"Mrk817_a_{band}.txt")
                    output = np.loadtxt(case / "prepared" / f"Mrk817_a_{band}.txt")
                    np.testing.assert_array_equal(output[:, 0], original[:, 0])
                    np.testing.assert_array_equal(output[:, 2], original[:, 2])
                    if case.name == "responses_s20" and band == "UVW2":
                        self.assertEqual((case / "prepared/Mrk817_a_UVW2.txt").read_bytes(),
                                         (inp / "Mrk817_a_UVW2.txt").read_bytes())
                    else:
                        sigma = cfg["smoothing"]["sigma_days"]
                        trend = np.array([np.average(original[:, 1], weights=
                            np.exp(-.5 * ((ti - original[:, 0]) / sigma)**2))
                            for ti in original[:, 0]])
                        expected = original[:, 1] - trend + np.mean(original[:, 1])
                        np.testing.assert_allclose(output[:, 1], expected, atol=1e-11, rtol=0)
                        self.assertFalse(np.allclose(output[:, 1], trend))
                    with np.load(case / "noise" / f"{band}.npz") as noise:
                        self.assertEqual(noise["covariance"].shape, (5, 5))
            with redirect_stdout(io.StringIO()):
                module.prepare(config, make_plots=False)  # exact repeat is allowed
            changed = np.loadtxt(inp / "Mrk817_a_U.txt")
            changed[0, 1] += 1
            np.savetxt(inp / "Mrk817_a_U.txt", changed)
            with self.assertRaises(FileExistsError):
                module.prepare(config, make_plots=False)

    def test_unknown_case_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.fixture(root)
            with self.assertRaises(ValueError):
                module.prepare(config, ["typo"], make_plots=False)
            self.assertFalse((root / "cases").exists())


if __name__ == "__main__":
    unittest.main()
