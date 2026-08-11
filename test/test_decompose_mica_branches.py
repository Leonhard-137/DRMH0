from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decompose_mica_branches import (  # noqa: E402
    DNEST_STR_MAX_LENGTH,
    DNestOptions,
    decomposition_settings,
    file_digest,
    posterior_column_count,
    systematic_resample,
    validate_decomposition_file,
    validate_run_outputs,
)
import ctypes


class SystematicResamplingTests(unittest.TestCase):
    def test_resampling_is_deterministic_for_a_seed(self):
        weights = np.array([0.1, 0.2, 0.7])
        first = systematic_resample(weights, 1000, np.random.default_rng(1234))
        second = systematic_resample(weights, 1000, np.random.default_rng(1234))
        np.testing.assert_array_equal(first, second)

    def test_resampling_counts_follow_weights(self):
        weights = np.array([0.1, 0.2, 0.7])
        draws = systematic_resample(weights, 1000, np.random.default_rng(9876))
        counts = np.bincount(draws, minlength=3)
        np.testing.assert_allclose(counts, [100, 200, 700], atol=1)

    def test_invalid_sample_size_is_rejected(self):
        with self.assertRaises(ValueError):
            systematic_resample([1.0], 0, np.random.default_rng(1))


class NativeLayoutTests(unittest.TestCase):
    def test_dnest_options_layout_matches_64_bit_c_struct(self):
        self.assertEqual(ctypes.sizeof(DNestOptions), 1176)
        self.assertEqual(DNestOptions.posterior_sample_file.offset, 704)
        self.assertEqual(
            DNestOptions.limits_file.offset + DNEST_STR_MAX_LENGTH,
            1172,
        )

    def test_posterior_column_count_ignores_header(self):
        path = (
            ROOT
            / "Mrk142/runs/mica_gaussian/2comp"
            / "run_W2_to_M2_2comp_gaussian/data/posterior_sample1d.txt_2"
        )
        if not path.is_file():
            self.skipTest("Mrk142 MICA posterior is not available")
        self.assertGreater(posterior_column_count(path), 3)

    def test_settings_fingerprint_is_order_independent(self):
        left = decomposition_settings({"ncomp": 2, "type_tf": "gaussian"})
        right = decomposition_settings({"type_tf": "gaussian", "ncomp": 2})
        self.assertEqual(left, right)

    def test_missing_file_has_no_digest(self):
        self.assertIsNone(file_digest(ROOT / "this_file_does_not_exist"))


class ExistingDecompositionValidationTests(unittest.TestCase):
    def test_existing_mrk142_decomposition_is_valid(self):
        path = (
            ROOT
            / "Mrk142/runs/mica_gaussian/2comp"
            / "run_W2_to_M2_2comp_gaussian/data/pline.txt_2_comp0"
        )
        if not path.is_file():
            self.skipTest("Mrk142 MICA decomposition is not available")
        n_driver, n_response, n_total = validate_decomposition_file(path)
        self.assertEqual(n_driver, 200)
        self.assertEqual(n_response, 200)
        self.assertEqual(n_total, 400)

    def test_existing_conditional_run_is_complete_when_available(self):
        target = (
            ROOT
            / "Mrk142/runs/mica_gaussian/2comp/result/branches/q0_lt_3"
            / "decomposition/run_W2_to_M2_2comp_gaussian"
        )
        if not (target / "data/fig_line_decomp_2.pdf").is_file():
            self.skipTest("conditional Mrk142 decomposition is not available")
        result = validate_run_outputs(target, 2, require_plot=True)
        self.assertEqual(result["total_rows"], 400)
        self.assertEqual(result["comp0_rows"], 400)
        self.assertEqual(result["comp1_rows"], 400)


if __name__ == "__main__":
    unittest.main()
