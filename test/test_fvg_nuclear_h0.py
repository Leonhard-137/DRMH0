from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fvg_nuclear_h0 as nuclear  # noqa: E402


class FvgNuclearExtinctionTests(unittest.TestCase):
    def setUp(self):
        self.waves = {"W2": 1928.0, "V": 5468.0}

    def test_known_colour_excess_is_recovered(self):
        expected = 0.025
        k, _ = nuclear.extinction_k(np.array([1928.0, 5468.0]), law="SMC")
        intrinsic = (5468.0 / 1928.0) ** (7.0 / 3.0)
        observed = intrinsic * 10.0 ** (-0.4 * (k[0] - k[1]) * expected)
        posterior = pd.DataFrame({"R_W2": [observed], "R_V": [1.0]})
        raw, used, summary = nuclear.infer_fvg_ebv(posterior, self.waves)
        np.testing.assert_allclose(raw, expected, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(used, expected, rtol=0.0, atol=1.0e-12)
        self.assertAlmostEqual(summary["ebv_p50"], expected, places=12)

    def test_negative_colour_excess_is_clipped_only_when_requested(self):
        intrinsic = (5468.0 / 1928.0) ** (7.0 / 3.0)
        posterior = pd.DataFrame({"R_W2": [1.1 * intrinsic], "R_V": [1.0]})
        raw, clipped, _ = nuclear.infer_fvg_ebv(
            posterior, self.waves, truncate=True
        )
        _, retained, _ = nuclear.infer_fvg_ebv(
            posterior, self.waves, truncate=False
        )
        self.assertLess(raw[0], 0.0)
        self.assertEqual(clipped[0], 0.0)
        self.assertEqual(retained[0], raw[0])

    def test_fluxes_scale_and_epsilon_is_invariant(self):
        target = pd.DataFrame({
            "sample": [0, 1],
            "F_Faint_comp0_W2": [2.0, 3.0],
            "F_Bright_comp0_W2": [4.0, 6.0],
            "Delta_F_comp0_W2": [2.0, 3.0],
            "Epsilon_comp0_W2": [0.5, 0.5],
            "F_Faint_comp0_V": [1.0, 1.5],
            "F_Bright_comp0_V": [2.0, 3.0],
            "Delta_F_comp0_V": [1.0, 1.5],
            "Epsilon_comp0_V": [0.5, 0.5],
        })
        cfg = {
            "hubble_distance": {
                "flux_columns": {
                    "bright": "F_Bright_comp0_{band}",
                    "epsilon": "Epsilon_comp0_{band}",
                }
            }
        }
        ebv = np.array([0.02, 0.02])
        corrected, summary = nuclear.apply_nuclear_extinction(
            target,
            ebv,
            ebv,
            cfg,
            ["W2", "V"],
            self.waves,
            seed=1,
        )
        np.testing.assert_allclose(
            corrected["Epsilon_comp0_W2"], target["Epsilon_comp0_W2"]
        )
        np.testing.assert_allclose(
            corrected["Epsilon_comp0_V"], target["Epsilon_comp0_V"]
        )
        for band in ("W2", "V"):
            factor = corrected[f"corr_nuc_{band}"].to_numpy(float)
            np.testing.assert_allclose(
                corrected[f"F_Bright_comp0_{band}"],
                target[f"F_Bright_comp0_{band}"] * factor,
            )
        self.assertTrue((summary["epsilon_max_abs_difference"] < 1.0e-12).all())

    def test_model_specific_raw_flux_path_is_overridden(self):
        root = Path("/synthetic")
        corrected = root / "corrected.txt"
        cfg = {
            "hubble_distance": {
                "models": {
                    "standard": {
                        "lag_fit_model": "4/3",
                        "flux_posterior": "raw-standard.txt",
                    },
                    "slim": {
                        "lag_fit_model": "2",
                        "flux_posterior": "raw-slim.txt",
                    },
                    "free": {
                        "lag_fit_model": "free",
                        "flux_posterior": "raw-free.txt",
                    },
                }
            }
        }
        with mock.patch.object(Path, "is_file", return_value=True):
            routed = nuclear.h0_config(
                cfg,
                input_branch_root=root / "input",
                output_root=root / "output",
                corrected_flux=corrected,
            )
        for model in routed["hubble_distance"]["models"].values():
            self.assertEqual(model["flux_posterior"], str(corrected))

    def test_output_cannot_overlap_baseline_results(self):
        input_root = Path("/source/branch")
        with self.assertRaisesRegex(ValueError, "overlaps protected"):
            nuclear.validate_output_root(
                input_root / "h0" / "nuclear",
                input_root=input_root,
                total_path=Path("/source/total/analysis/total.txt"),
                target_path=input_root / "fflux/analysis/raw.txt",
            )

    def test_new_sibling_output_is_allowed(self):
        input_root = Path("/source/branch")
        nuclear.validate_output_root(
            input_root / "nuclear_extinction_new",
            input_root=input_root,
            total_path=Path("/source/total/analysis/total.txt"),
            target_path=input_root / "fflux/analysis/raw.txt",
        )

    def test_new_stage_uses_model_local_native_comp0_root(self):
        cfg = {
            "mica_mix220": {
                "output_root": "runs/mica/mix220/phys12_slow3",
                "type_tf": "220",
                "number_component": [3, 3],
            }
        }
        self.assertEqual(
            nuclear.branch_root(
                Path("/source"), cfg, "ignored_legacy_branch", "mica_mix220"
            ),
            Path("/source/runs/mica/mix220/phys12_slow3/result/comp0"),
        )


if __name__ == "__main__":
    unittest.main()
