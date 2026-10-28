"""Broad-response preparation tests; no formal cases or MICA runs are written."""
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
from scipy.signal import fftconvolve
from scipy.special import ndtr
import yaml

HERE = Path(__file__).resolve().parent / "mica_recovery/longterm_test"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("longterm_broad_prepare", HERE / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class BroadPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = prepare.load_config(HERE / "experiment_broad.yaml")
        # Coarser convolution bins, but preserve the production disk quadrature.
        cls.config["grid"].update(dt_days=0.1, disk_bin_oversample=50)
        cls.times = {"UVW2": np.arange(0.0, 421.0, 10.0),
                     "U": np.arange(0.5, 421.0, 10.0)}
        cls.errors = {"UVW2": np.full(43, 0.01), "U": np.full(43, 0.02)}
        setup = ("UVW2", "U", 59000.0, cls.times, cls.errors, 0.17)
        with patch.object(prepare, "observation_setup", return_value=setup):
            cls.background = prepare.build_disk_background(HERE / "experiment_broad.yaml", cls.config)

    def test_gaussian_grid_preserves_negative_mass_and_moments(self):
        cfg = prepare.load_config(HERE / "experiment_broad.yaml")
        kernel = next(iter(prepare.known_response_kernels(cfg).values()))
        lag, mass = kernel["lag"], kernel["mass"]
        self.assertAlmostEqual(lag[0] - 0.01, -610.0)
        self.assertAlmostEqual(lag[-1] + 0.01, 670.0)
        self.assertGreater(kernel["sampled_area"], 1 - 1e-12)
        self.assertAlmostEqual(mass.sum(), 1.0, places=14)
        self.assertAlmostEqual(np.sum(lag * mass), 30.0, places=9)
        self.assertAlmostEqual(np.sqrt(np.sum((lag - 30.0)**2 * mass)), 80.0, places=5)
        self.assertAlmostEqual(mass[lag < 0].sum(), ndtr(-30.0 / 80.0), places=12)
        self.assertAlmostEqual(kernel["negative_fraction"], ndtr(-30.0 / 80.0), places=14)

    def test_hidden_grid_covers_both_sides_of_double_convolution(self):
        b = self.background
        k = next(iter(b["known_response_kernels"].values()))
        earliest = self.times["U"].min() - (b["lag"][-1] + k["lag"][-1])
        latest = self.times["U"].max() - (b["lag"][0] + k["lag"][0])
        self.assertLess(b["start"], earliest - 199.0)
        self.assertGreater(b["end"], latest + 199.0)
        self.assertEqual(len(b["hidden"]), len(b["hidden_time"]))

    def test_preparation_is_paired_and_components_sum_to_flux(self):
        b = self.background
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            path.write_text(yaml.safe_dump(self.config))
            with patch.object(prepare, "build_disk_background", return_value=b), redirect_stdout(io.StringIO()):
                prepare.prepare_known_response(path)
                prepare.prepare_shared_slow(path)
                with self.assertRaises(FileExistsError):
                    prepare.prepare_known_response(path)
            known_name = next(iter(self.config["known_response"]["cases"]))
            shared_name = next(iter(self.config["shared_slow"]["cases"]))
            outputs = {}
            for case in (known_name, shared_name):
                for band in ("UVW2", "U"):
                    root = Path(directory) / "cases" / case
                    lc = np.loadtxt(root / "input" / f"LTSIM_{band}.txt")
                    components = np.loadtxt(root / "components" / f"{band}_components.txt")
                    np.testing.assert_allclose(lc[:, 0], self.times[band])
                    np.testing.assert_allclose(lc[:, 1], 1 + components[:, 1:].sum(axis=1), atol=1e-10)
                    np.testing.assert_allclose(lc[:, 2], self.errors[band])
                    np.testing.assert_allclose(components[:, 3], b["noise"][band], atol=1e-10)
                    outputs[case, band] = components
            np.testing.assert_allclose(outputs[known_name, "UVW2"][:, 1], outputs[shared_name, "UVW2"][:, 1])
            np.testing.assert_allclose(outputs[known_name, "U"][:, 1], 0.7 * outputs[shared_name, "U"][:, 1])
            np.testing.assert_array_equal(outputs[known_name, "UVW2"][:, 2], 0)
            shared_w = outputs[shared_name, "UVW2"]
            self.assertAlmostEqual(np.std(shared_w[:, 2], ddof=1) / np.std(shared_w[:, 1], ddof=1), 0.5)

            # Independent sequential convolution checks that Gaussian acts on
            # W0=D*disk_W2, rather than acting directly on raw D.
            kernel = next(iter(b["known_response_kernels"].values()))
            w0_continuous = fftconvolve(b["hidden"], b["kernels"]["UVW2"])
            raw = prepare.convolve_and_sample(w0_continuous, kernel["mass"],
                                              b["start"] + b["lag"][0], kernel["lag"][0],
                                              b["dt"], self.times["U"])
            expected = 0.3 * b["amplitude"] * (raw - raw.mean())
            np.testing.assert_allclose(outputs[known_name, "U"][:, 2], expected, atol=1e-10)

    def test_missing_known_suite_preserves_legacy_grid_formula(self):
        cfg = deepcopy(self.config)
        cfg.pop("known_response")
        setup = ("UVW2", "U", 59000.0, self.times, self.errors, 0.17)
        with patch.object(prepare, "observation_setup", return_value=setup):
            b = prepare.build_disk_background(HERE / "experiment.yaml", cfg)
        self.assertAlmostEqual(b["start"], -400.0)
        self.assertAlmostEqual(b["end"], 640.5)
        self.assertEqual(b["known_response_kernels"], {})

    def test_invalid_fractions_are_rejected(self):
        cfg = deepcopy(self.config)
        next(iter(cfg["known_response"]["cases"].values()))["disk_fraction"] = 0.9
        with self.assertRaisesRegex(ValueError, "sum to one"):
            prepare.known_response_kernels(cfg)


if __name__ == "__main__":
    unittest.main()
