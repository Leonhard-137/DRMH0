"""Causal real-disk/mixed-driver regressions; no MICA sampling is started."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent / "mica_recovery/longterm_test"
sys.path.insert(0, str(HERE))
import prepare


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fit = load_module("real_disk_fit_tests", "fit.py")
plots = load_module("real_disk_plot_tests", "plot_inputs.py")


class RealDiskMismatchTests(unittest.TestCase):
    def setUp(self):
        self.source_path = HERE / "experiment_real_disk_mismatch.yaml"
        self.config = prepare.load_config(self.source_path)
        self.name, self.parameters = next(iter(
            prepare.real_disk_mismatch_cases(self.config).items()
        ))

    def test_full_mrk817_expected_covariance_regression(self):
        driver, response, _, times, _, target = prepare.observation_setup(
            self.source_path, self.config
        )
        self.assertEqual((len(times[driver]), len(times[response])), (314, 324))
        span = max(times[driver].max(), times[response].max())
        self.assertAlmostEqual(span, 458.51889700000174, places=9)
        covariances, diagnostics = prepare.real_disk_mismatch_component_covariances(
            times[driver], times[response], self.parameters, target
        )
        self.assertAlmostEqual(diagnostics["alpha"], 0.16963697514341483, places=14)
        self.assertAlmostEqual(
            diagnostics["expected_response_window_rms"],
            0.16760379279706783,
            places=14,
        )
        for covariance in covariances.values():
            self.assertEqual(covariance.shape, (638, 638))
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-14)

        lag, kernels = prepare.real_disk_mismatch_input_kernels(self.parameters)
        dt = lag[1] - lag[0]
        self.assertTrue(np.all(lag > 0))
        self.assertTrue(np.all(kernels["disk"] >= 0))
        self.assertTrue(np.all(kernels["blr"] >= 0))
        self.assertTrue(np.all(kernels["blr"][lag < 6] == 0))
        blr = kernels["blr"] / self.parameters["response_blr_gain"]
        self.assertAlmostEqual(blr.sum() * dt, 1.0, places=4)
        # The display grid stops at 100 d, leaving a tiny analytic Gamma tail.
        self.assertAlmostEqual((blr * lag).sum() * dt, 20.0, places=2)

    def test_generate_plot_and_fit_contract(self):
        config = deepcopy(self.config)
        config["experiment"]["output_dir"] = "cases"
        suite = config["real_disk_mismatch"]
        suite.update(
            covariance_dt_days=0.05,
            covariance_n=16384,
            disk_support_days=20.0,
            disk_fine_factor=4,
            disk_origin_cells=128,
            disk_origin_order=16,
        )
        times = {
            "UVW2": np.array([0.0, 1.2, 9.0, 30.0, 79.0]),
            "U": np.array([0.4, 4.0, 18.0, 44.0, 80.0, 100.0]),
        }
        errors = {band: np.full(len(value), 0.01) for band, value in times.items()}
        setup = (
            "UVW2", "U", 59176.004915, times, errors,
            config["observations"]["driver_window_rms"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            path.write_text(yaml.safe_dump(config))
            with patch.object(prepare, "observation_setup", return_value=setup), \
                    redirect_stdout(io.StringIO()):
                prepare.prepare_real_disk_mismatch(path)
                with self.assertRaises(FileExistsError):
                    prepare.prepare_real_disk_mismatch(path)
            case = Path(directory) / "cases" / self.name
            saved = {}
            parsed = prepare.real_disk_mismatch_cases(config)[self.name]
            for band in times:
                curve = np.loadtxt(case / "input" / f"LTSIM_{band}.txt")
                components = plots.load_components(
                    case, band, curve, real_case=parsed
                )
                saved[band] = curve.copy()
                self.assertEqual(components.shape, (len(times[band]), 5))
                np.testing.assert_allclose(curve[:, 0], times[band])
                np.testing.assert_allclose(curve[:, 2], errors[band])
                np.testing.assert_allclose(components[:, 1:4].mean(axis=0), 0, atol=1e-12)
                np.testing.assert_allclose(
                    1 + components[:, 1:].sum(axis=1), curve[:, 1], atol=1e-10
                )
            realized = np.std(
                np.loadtxt(case / "components/UVW2_components.txt")[:, 1:4].sum(axis=1),
                ddof=1,
            )
            self.assertGreater(abs(realized - setup[-1]), 1e-3)

            changed = {**parsed, "short_lag_days": parsed["short_lag_days"] + 1}
            with self.assertRaisesRegex(ValueError, "Configuration does not match"):
                plots.load_components(
                    case, "U", saved["U"], real_case=changed
                )
            with patch.object(prepare, "observation_setup", return_value=setup), \
                    redirect_stdout(io.StringIO()):
                prepare.prepare_real_disk_mismatch(path, overwrite=True)
            for band in times:
                np.testing.assert_array_equal(
                    np.loadtxt(case / "input" / f"LTSIM_{band}.txt"), saved[band]
                )

            plan = fit.build_plan(path, suite="real_disk_mismatch")
            self.assertEqual(len(plan), 2)
            by_family = {job["fit"]["type_tf"]: job["fit"] for job in plan}
            expected_native = np.array([0.14195358, 152.84])
            np.testing.assert_allclose(by_family["gaussian"]["width_prior"][0], expected_native)
            np.testing.assert_allclose(by_family["gamma"]["width_prior"][0], expected_native)
            self.assertFalse(by_family["gaussian"]["flag_negative_resp"])
            self.assertFalse(by_family["gamma"]["flag_negative_resp"])

            lag, curves = plots.build_transfer_functions(config)
            physical = curves[self.name]
            self.assertIn("not a total W2-to-U kernel", physical["title"])
            self.assertIn("no single deterministic", physical["note"])
            self.assertTrue(np.all(physical["disk"] >= 0))
            self.assertTrue(np.all(physical["blr"] >= 0))
            with patch.object(plots.plt.Figure, "savefig") as save, \
                    redirect_stdout(io.StringIO()):
                plots.plot_case(
                    case, "LTSIM", ("UVW2", "U"), real_case=parsed
                )
                plots.plot_transfer_function(case, self.name, lag, physical)
            self.assertEqual(save.call_count, 4)


if __name__ == "__main__":
    unittest.main()
