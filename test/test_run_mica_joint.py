"""Joint MICA prior/checkpoint regressions; no MPI or native fits are run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
with mock.patch.dict(os.environ):
    os.environ.pop("PMI_RANK", None)
    os.environ.pop("PMIX_RANK", None)
    import run_mica
    import run_mica_joint as joint
import mica_resume


def make_fit(**overrides):
    stage = {
        "type_tf": "gamma", "ncomp": 2, "type_lag_prior": 4,
        "lag_limit": [-10, 100], "lag_prior": [[-10, 100], [-10, 100]],
        "prepared_dir": "prepared", "max_num_saves": 2,
    }
    stage.update(overrides)
    return run_mica.fit_config({
        "reference_band": "W2", "response_bands": ["U", "B"],
        "joint_test": stage,
    }, "joint_test")


def curve(times):
    times = np.asarray(times, dtype=float)
    return np.column_stack((times, np.ones(len(times)), np.full(len(times), 0.1)))


def prior_rows(text):
    """Read the native seven-column format without relying on writer helpers."""
    rows = {}
    for line in text.splitlines():
        words = line.split()
        if len(words) != 7 or words[2] != "UNI" or words[5:] != ["0", "0"]:
            raise AssertionError(f"Invalid native prior row: {line!r}")
        index = int(words[0])
        if index in rows:
            raise AssertionError(f"Duplicate native prior slot: {index}")
        rows[index] = (float(words[3]), float(words[4]))
    return rows


class JointMicaPriorTests(unittest.TestCase):
    def setUp(self):
        # Deliberately different sampling and spans. Native %.6e serialization
        # changes these timestamps, so raw-array and global-joint bounds fail.
        self.cont = curve([58000.014, 58050.026, 58100.084])
        self.u = curve([58010.049, 58011.249, 58012.453, 58013.657, 58014.851])
        self.b = curve([57900.002, 58200.006])
        self.data = {"set1": [self.cont, self.u, self.b]}

    def test_input_order_preserves_response_order_and_one_driver(self):
        src = Path("/not-created/Synthetic")
        fit = make_fit(responses=["B", "U"])
        files = joint.input_files(src, "Synthetic", fit)
        self.assertEqual(list(files), ["cont.txt", "line_B.txt", "line_U.txt"])
        self.assertEqual(list(files.values()), [
            src / "prepared/Synthetic_W2.txt",
            src / "prepared/Synthetic_B.txt",
            src / "prepared/Synthetic_U.txt",
        ])

    def test_each_band_has_its_own_gamma_centroid_and_log_width_slots(self):
        lag = {"U": [[-4, 10], [20, 80]], "B": [[-8, 15], [25, 90]]}
        width = {"U": [[0.2, 2], [4, 14]], "B": [[0.3, 3], [5, 18]]}
        fit = make_fit(lag_prior_by_band=lag, width_prior_by_band=width)
        kwargs, text = joint.setup(fit, self.data)
        self.assertIs(kwargs["data"], self.data)
        self.assertEqual(kwargs["type_tf"], "gamma")
        self.assertEqual(kwargs["type_lag_prior"], 4)
        self.assertEqual(kwargs["number_component"], [2, 2])
        self.assertTrue(text.endswith("\n"))
        rows = prior_rows(text)
        # Three DRW slots, then one systematic plus two triples for each band.
        # No DRW, amplitude, or systematic parameter may be overridden here.
        self.assertEqual(set(rows), {5, 6, 8, 9, 12, 13, 15, 16})
        for band, centers, widths in [("U", (5, 8), (6, 9)), ("B", (12, 15), (13, 16))]:
            for component in range(2):
                self.assertEqual(rows[centers[component]], tuple(lag[band][component]))
                np.testing.assert_allclose(
                    rows[widths[component]], np.log(width[band][component]),
                    rtol=0, atol=1e-12,
                )

    def test_default_width_bounds_match_each_serialized_pair(self):
        _, text = joint.setup(make_fit(), self.data)
        rows = prior_rows(text)
        # Hand-calculated from the native serialized endpoints:
        # W2 span=100.07, n=3; U span=4.80, n=5; B span=300.01, n=2.
        expected = {
            6: (0.12, 100.07 / 3), 9: (0.12, 100.07 / 3),
            13: (5.0035, 300.01 / 3), 16: (5.0035, 300.01 / 3),
        }
        for index, bounds in expected.items():
            with self.subTest(index=index):
                np.testing.assert_allclose(np.exp(rows[index]), bounds, rtol=0, atol=1e-9)
        for index in (5, 8, 12, 15):
            self.assertEqual(rows[index], (-10.0, 100.0))

    def test_shared_explicit_width_prior_applies_to_each_band(self):
        _, text = joint.setup(make_fit(width_prior=[[0.1, 1], [2, 20]]), self.data)
        rows = prior_rows(text)
        for index, bounds in {6: (0.1, 1), 13: (0.1, 1), 9: (2, 20), 16: (2, 20)}.items():
            np.testing.assert_allclose(rows[index], np.log(bounds), rtol=0, atol=1e-12)

    def test_joint_entry_rejects_unsupported_model_settings(self):
        for overrides in ({"type_tf": "gaussian"}, {"ncomp": 1}, {"type_lag_prior": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                joint.setup(make_fit(**overrides), self.data)

    def test_explicit_priors_match_pairwise_native_decimal_precision(self):
        fit = make_fit(lag_prior=[[-1.12345678, 3.12345678]] * 2,
                       width_prior=[[0.12345678, 2.12345678]] * 2)
        _, text = joint.setup(fit, self.data)
        rows = prior_rows(text)
        for index in (5, 8, 12, 15):
            self.assertEqual(rows[index], (-1.123457, 3.123457))
        for index in (6, 9, 13, 16):
            np.testing.assert_allclose(rows[index], np.log([0.123457, 2.123457]),
                                       rtol=0, atol=1e-12)

    def test_repeated_driver_or_response_is_rejected(self):
        for responses in (["W2", "U"], ["U", "U"]):
            with self.subTest(responses=responses), self.assertRaises(ValueError):
                joint.input_files(Path("/not-created/Synthetic"), "Synthetic",
                                  make_fit(responses=responses))

    def test_missing_band_specific_prior_is_rejected(self):
        for overrides in (
            {"lag_prior_by_band": {"U": [[-10, 100], [-10, 100]]}},
            {"width_prior_by_band": {"U": [[0.1, 1], [2, 20]]}},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                joint.setup(make_fit(**overrides), self.data)


class JointMicaCompletionTests(unittest.TestCase):
    def make_run(self, rd, *, responses=5, fit_only=True):
        data = rd / "data"
        data.mkdir()
        dimension = 3 + responses * 7
        header = "# 1\n# " + ":".join(["2"] * (responses + 1)) + "\n"
        (data / "data_input.txt").write_text(header + "0 1 0.1\n1 1 0.1\n" * (responses + 1))
        sample = " ".join(["0"] * dimension) + "\n"
        (data / "sample1d.txt_2").write_text(sample * 2)
        (data / "sample_info1d.txt_2").write_text("0 0 0 0\n" * 2)
        (data / "posterior_sample1d.txt_2").write_text("# 1\n" + sample)
        (data / "posterior_sample.txt").write_text("# 1\n0 0 0\n")
        (data / "evidence.txt").write_text("2 0.0\n")
        return {"state": {"target_saves": 2, "ncomp": 2,
                          "settings": {"_fit_only": fit_only}}}

    def test_response_dimension_reads_six_curve_header_and_preserves_pairs(self):
        for responses, dimension in ((5, 38), (1, 10)):
            with self.subTest(responses=responses), tempfile.TemporaryDirectory() as name:
                rd = Path(name)
                self.make_run(rd, responses=responses)
                self.assertEqual(mica_resume.response_dimension(rd, 2), dimension)

    def test_fit_only_marks_complete_without_reconstruction_or_decomposition(self):
        with tempfile.TemporaryDirectory() as name:
            rd = Path(name)
            plan = self.make_run(rd)
            mica_resume.record_complete(rd, plan)
            saved = json.loads((rd / mica_resume.STATE_FILE).read_text())
            self.assertEqual(saved["status"], "complete")
            self.assertIn("data/posterior_sample.txt", saved["frozen_files"])
            self.assertFalse((rd / "data/pall.txt_2").exists())
            self.assertFalse((rd / "data/pline.txt_2_comp0").exists())

    def test_fit_only_still_requires_samples_posterior_evidence_and_continuum(self):
        required = ["sample1d.txt_2", "sample_info1d.txt_2",
                    "posterior_sample1d.txt_2", "evidence.txt", "posterior_sample.txt"]
        for missing in required:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as name:
                rd = Path(name)
                plan = self.make_run(rd)
                (rd / "data" / missing).unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    mica_resume.record_complete(rd, plan)
                self.assertFalse((rd / mica_resume.STATE_FILE).exists())

    def test_joint_completion_rejects_pair_dimension_samples_and_short_runs(self):
        for dimension, count in ((10, 2), (38, 1)):
            with self.subTest(dimension=dimension, count=count), tempfile.TemporaryDirectory() as name:
                rd = Path(name)
                plan = self.make_run(rd)
                (rd / "data/sample1d.txt_2").write_text((" ".join(["0"] * dimension) + "\n") * count)
                with self.assertRaises(ValueError):
                    mica_resume.record_complete(rd, plan)

    def test_normal_completion_still_requires_reconstruction_and_decomposition(self):
        for present in ([], ["pall.txt_2"]):
            with self.subTest(present=present), tempfile.TemporaryDirectory() as name:
                rd = Path(name)
                plan = self.make_run(rd, responses=1, fit_only=False)
                for filename in present:
                    (rd / "data" / filename).write_text("0 1 0.1\n")
                with self.assertRaisesRegex(ValueError, "pall|pline"):
                    mica_resume.record_complete(rd, plan)


if __name__ == "__main__":
    unittest.main()
