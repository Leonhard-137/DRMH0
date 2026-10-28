"""Small analytic fixtures for the comparison reader; no MICA runs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml
from scipy.special import ndtr

SPEC = importlib.util.spec_from_file_location("mica_smooth_compare", Path(__file__).with_name("compare.py"))
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def gaussian_draws():
    rows = []
    # The first draw has zero posterior weight and invalid physical values.
    for sample, weight, centers, amps in (
        (0, 0.0, (np.nan, np.nan), (np.nan, np.nan)),
        (10, 0.25, (0.0, -1.0), (9.0, 1.0)),
        (11, 0.75, (2.0, -4.0), (1.0, 9.0)),
    ):
        for i in range(2):
            rows.append(dict(band="U", sample=sample, component=f"tf{i}", tf_kind="gaussian",
                             center=centers[i], width=float(i + 1), amp=amps[i], weight=weight))
    return pd.DataFrame(rows)


def write_fixture(source_dir: Path):
    config_dir = source_dir / "config"
    config_dir.mkdir(parents=True)
    config = {
        stage: dict(ncomp=2, type_tf=model, output_root=f"custom_runs/{model}")
        for model, stage in compare.STAGES.items()
    }
    (config_dir / "source_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    result = source_dir / "custom_runs/gaussian/2comp/result"
    result.mkdir(parents=True)
    rows = []
    for component in compare.COMPONENTS:
        rows.append(dict(band="U", component=component, lambda_obs=3465.0,
                         tau=17.0, tau_err_low=2.0, tau_err_high=4.0,
                         width=3.0 if component != "total" else np.nan,
                         width_err_low=1.0, width_err_high=2.0))
    pd.DataFrame(rows).to_csv(result / "mica_component_summary.csv", index=False)
    gaussian_draws().fillna(0.0).to_csv(result / "mica_component_posterior.txt", sep=" ", index=False)


class ComparisonTests(unittest.TestCase):
    def test_gaussian_total_is_computed_per_paired_draw_and_zero_weight_is_ignored(self):
        data = gaussian_draws().sample(frac=1, random_state=3)
        result = compare.negative_area_statistics(data, "gaussian")
        # Gaussian CDF values are known for these two draws. Their cumulative
        # weights are 0.25 and 1: the weighted median interpolates one third.
        total_lo = (9.0 * 0.5 + ndtr(0.5)) / 10.0
        total_hi = (ndtr(-2.0) + 9.0 * ndtr(2.0)) / 10.0
        expected_median = total_lo + (total_hi - total_lo) / 3.0
        self.assertAlmostEqual(result["total"]["negative_area_fraction"], expected_median)
        self.assertAlmostEqual(result["total"]["negative_area_fraction_err_low"], expected_median - total_lo)
        self.assertAlmostEqual(result["tf0"]["negative_area_fraction"], ndtr(-2.0))

    def test_gamma_uses_onset_and_shape_two(self):
        data = pd.DataFrame([
            dict(sample=7, component="tf0", tf_kind="gamma", center=-2.0, width=2.0, amp=1.0, weight=1.0),
            dict(sample=7, component="tf1", tf_kind="gamma", center=1.0, width=3.0, amp=3.0, weight=1.0),
        ])
        result = compare.negative_area_statistics(data, "gamma")
        exact = 1.0 - 2.0 / np.e
        self.assertAlmostEqual(result["tf0"]["negative_area_fraction"], exact)
        self.assertEqual(result["tf1"]["negative_area_fraction"], 0.0)
        self.assertAlmostEqual(result["total"]["negative_area_fraction"], exact / 4.0)
        self.assertEqual(result["total"]["negative_area_fraction_err_high"], 0.0)

    def test_total_rejects_unpaired_samples(self):
        data = gaussian_draws()
        data.loc[data["component"].eq("tf1") & data["sample"].eq(11), "sample"] = 12
        with self.assertRaisesRegex(ValueError, "sample IDs"):
            compare.negative_area_statistics(data, "gaussian")

    def test_total_rejects_inconsistent_component_weights(self):
        data = gaussian_draws()
        data.loc[data["component"].eq("tf1") & data["sample"].eq(11), "weight"] = 0.7
        with self.assertRaisesRegex(ValueError, "weights"):
            compare.negative_area_statistics(data, "gaussian")

    def test_missing_cases_are_reported_without_invented_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "experiment"
            baseline = Path(temp) / "baseline"
            write_fixture(baseline)
            frame, status = compare.collect_comparison(root, baseline)
            self.assertEqual(len(frame), 3)
            self.assertEqual(set(frame["case"]), {"baseline"})
            self.assertEqual(set(frame["band"]), {"U"})
            self.assertEqual(set(frame["tau"]), {17.0})
            self.assertEqual(set(frame["tau_err_high"]), {4.0})
            self.assertEqual(len(status), 50)
            self.assertEqual(status["status"].eq("available").sum(), 1)
            self.assertEqual(status["status"].eq("missing_config").sum(), 40)
            self.assertEqual(status["status"].eq("missing_results").sum(), 5)
            self.assertEqual(status["status"].eq("missing_band").sum(), 4)

    def test_cli_writes_existing_data_status_and_both_figure_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "experiment"
            baseline = Path(temp) / "baseline"
            output = Path(temp) / "comparison"
            write_fixture(baseline)
            self.assertEqual(compare.main(["--root", str(root), "--baseline-dir", str(baseline),
                                           "--output-dir", str(output)]), 0)
            self.assertEqual(len(pd.read_csv(output / "comparison.csv")), 3)
            self.assertTrue((output / "comparison_status.csv").is_file())
            self.assertIn("Gamma k=2 scale", (output / "comparison_notes.txt").read_text())
            self.assertTrue((output / "comparison_gaussian.png").read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((output / "comparison_gaussian.pdf").read_bytes().startswith(b"%PDF"))
            self.assertFalse((output / "comparison_gamma.png").exists())


if __name__ == "__main__":
    unittest.main()
