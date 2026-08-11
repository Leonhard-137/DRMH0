from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from posterior_branches import (  # noqa: E402
    branch_definitions,
    branch_mask,
    effective_sample_size,
    posterior_features,
    stable_seed_offset,
)


class PosteriorFeatureTests(unittest.TestCase):
    def setUp(self):
        self.centers = np.array([[1.0, 3.0, 9.0], [4.0, 5.0, 7.0]])
        self.widths = np.array([[1.0, 1.0, 2.0], [2.0, 2.0, 2.0]])
        self.amplitudes = np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]])
        self.features = posterior_features(
            self.centers,
            self.widths,
            self.amplitudes,
        )

    def test_expected_derived_features(self):
        np.testing.assert_allclose(self.features["q0"], [1.0, 3.0, 4.5])
        np.testing.assert_allclose(self.features["amp_frac0"], [0.75, 0.25, 0.5])
        np.testing.assert_allclose(self.features["center_separation"], [3.0, 2.0, -2.0])

    def test_adjacent_ranges_form_an_exact_partition(self):
        low = branch_mask(
            "low",
            {"ranges": {"q0": [None, 3.0]}},
            self.features,
        )
        high = branch_mask(
            "high",
            {"ranges": {"q0": [3.0, None]}},
            self.features,
        )
        np.testing.assert_array_equal(low, [True, False, False])
        np.testing.assert_array_equal(high, [False, True, True])
        np.testing.assert_array_equal(low | high, np.ones(3, dtype=bool))
        np.testing.assert_array_equal(low & high, np.zeros(3, dtype=bool))

    def test_multiple_ranges_are_combined(self):
        selected = branch_mask(
            "combined",
            {
                "ranges": {
                    "q0": [3.0, None],
                    "center_separation": [0.0, None],
                }
            },
            self.features,
        )
        np.testing.assert_array_equal(selected, [False, True, False])


class PosteriorBranchConfigTests(unittest.TestCase):
    def test_disabled_branches_are_omitted(self):
        fit = {
            "posterior_branches": {
                "enabled": True,
                "branches": {
                    "keep": {"ranges": {"q0": [None, 3.0]}},
                    "skip": {"enabled": False, "ranges": {"q0": [3.0, None]}},
                },
            }
        }
        definitions = branch_definitions(fit)
        self.assertEqual(list(definitions), ["keep"])
        self.assertEqual(definitions["keep"]["label"], "keep")

    def test_effective_sample_size(self):
        self.assertAlmostEqual(effective_sample_size([1.0, 1.0, 1.0, 1.0]), 4.0)
        self.assertAlmostEqual(effective_sample_size([1.0, 0.0, 0.0]), 1.0)

    def test_seed_offset_is_stable_and_branch_specific(self):
        self.assertEqual(stable_seed_offset("branch_a"), stable_seed_offset("branch_a"))
        self.assertNotEqual(stable_seed_offset("branch_a"), stable_seed_offset("branch_b"))


if __name__ == "__main__":
    unittest.main()
