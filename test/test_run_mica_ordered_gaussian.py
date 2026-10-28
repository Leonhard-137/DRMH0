"""Ordered Gaussian prior regressions; no MPI, fits, or run files are created."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Import the scheduler branch even when the test runner inherits MPI variables.
with mock.patch.dict(os.environ):
    os.environ.pop("PMI_RANK", None)
    os.environ.pop("PMIX_RANK", None)
    import run_mica


PRIOR_PATH = Path("param/ordered_gaussian_lag_priors.txt")
PRIOR_TEXT = (
    "5 0-th_component_centroid UNI 0 100 0 0\n"
    "8 1-th_component_centroid UNI 0 100 0 0\n"
)


def make_fit(**overrides):
    stage = {
        "type_tf": "gaussian", "ncomp": 2, "type_lag_prior": 0,
        "lag_limit": [0, 100], "lag_prior": [[0, 100], [0, 100]],
    }
    stage.update(overrides)
    return run_mica.fit_config({
        "reference_band": "W2", "response_bands": ["U"], "ordered": stage,
    }, "ordered")


class OrderedGaussianPriorTests(unittest.TestCase):
    def setUp(self):
        self.fit = make_fit()
        curve = np.array([[0.0, 1.0, 0.1], [10.0, 1.2, 0.1]])
        self.data = {"set1": [curve, curve.copy()]}

    def test_setup_keeps_type_zero_without_explicit_lag_prior(self):
        fit = make_fit(width_prior=[[0.1, 1], [0.2, 2]])
        kwargs = run_mica.setup_kwargs(fit, self.data, "U")
        self.assertEqual(kwargs["type_lag_prior"], 0)
        self.assertNotIn("lag_prior", kwargs)
        self.assertEqual(kwargs["lag_limit"], [0, 100])
        self.assertEqual(kwargs["number_component"], [2, 2])
        self.assertEqual(kwargs["width_prior"], [[0.1, 1], [0.2, 2]])
        self.assertIs(kwargs["data"], self.data)

    def test_native_prior_text_is_exactly_two_seven_column_rows(self):
        text = run_mica.ordered_gaussian_prior_text(self.fit, self.data, "U")
        self.assertEqual(text, PRIOR_TEXT)
        self.assertEqual([len(line.split()) for line in text.splitlines()], [7, 7])

    def test_prior_four_gamma_and_mixed_setup_are_unchanged(self):
        cases = [
            {"type_lag_prior": 4, "lag_prior": [[0, 10], [10, 100]]},
            {"type_tf": "gamma"},
            {"type_tf": "00"},
            {"type_tf": "220", "ncomp": 3, "lag_prior": [[0, 100]] * 3},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                fit = make_fit(**overrides)
                self.assertIsNone(run_mica.ordered_gaussian_prior_text(fit, self.data, "U"))
                kwargs = run_mica.setup_kwargs(fit, self.data, "U")
                self.assertEqual(kwargs["lag_prior"], fit["lag_prior"])
                self.assertEqual(kwargs["type_lag_prior"], fit["type_lag_prior"])
                self.assertEqual(kwargs["type_tf"], fit["_native_type_tf"])

    def test_absent_or_null_explicit_prior_does_not_trigger(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                fit = make_fit(lag_prior=None)
                if missing:
                    fit.pop("lag_prior")
                self.assertIsNone(run_mica.ordered_gaussian_prior_text(fit, self.data))
                self.assertNotIn("lag_prior", run_mica.setup_kwargs(fit, self.data))

    def test_response_specific_prior_uses_the_selected_band(self):
        fit = make_fit(lag_prior=None, lag_prior_by_band={"U": [[0, 100], [0, 100]]})
        self.assertEqual(run_mica.ordered_gaussian_prior_text(fit, self.data, "U"), PRIOR_TEXT)
        self.assertNotIn("lag_prior", run_mica.setup_kwargs(fit, self.data, "U"))
        for response in (None, "B"):
            with self.subTest(response=response), self.assertRaises(ValueError):
                run_mica.ordered_gaussian_prior_text(fit, self.data, response)

    def test_invalid_component_counts_ranges_and_prior_shapes_are_rejected(self):
        cases = [
            {"ncomp": 1, "lag_prior": [[0, 100]]},
            {"ncomp": 3, "lag_prior": [[0, 100]] * 3},
            {"lag_prior": [[0, 100], [1, 100]]},
            {"lag_prior": [[0, 100]]},
            {"lag_prior": [[0, 100], [0]]},
            {"lag_prior": "0:100"},
            {"lag_prior": [[100, 0], [100, 0]]},
            {"lag_prior": [[0, 0], [0, 0]]},
            {"lag_prior": [[0, float("inf")], [0, float("inf")]]},
            {"lag_prior": [[float("nan"), 100], [float("nan"), 100]]},
        ]
        for overrides in cases:
            fit = make_fit(**overrides)
            for helper in (run_mica.ordered_gaussian_prior_text, run_mica.setup_kwargs):
                with self.subTest(overrides=overrides, helper=helper.__name__):
                    with self.assertRaises(ValueError):
                        helper(fit, self.data, "U")

    def test_only_one_driver_response_dataset_is_supported(self):
        curves = self.data["set1"]
        for data in ({}, {"a": curves, "b": curves}, {"a": []},
                     {"a": curves[:1]}, {"a": curves + curves[:1]}):
            with self.subTest(curve_counts=[len(value) for value in data.values()]):
                with self.assertRaises(ValueError):
                    run_mica.ordered_gaussian_prior_text(self.fit, data)

    def test_rank_zero_writes_then_every_rank_registers_after_barrier(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                events = []
                comm, model = mock.Mock(), mock.Mock()
                comm.Barrier.side_effect = lambda: events.append("barrier")
                model.set_priors.side_effect = lambda path: events.append("set_priors")
                with (
                    mock.patch.object(run_mica, "COMM", comm),
                    mock.patch.object(run_mica, "RANK", rank),
                    mock.patch.object(run_mica, "log"),
                    mock.patch.object(Path, "mkdir", autospec=True) as mkdir,
                    mock.patch.object(Path, "write_text", autospec=True,
                                      side_effect=lambda *a, **k: events.append("write")) as write,
                ):
                    run_mica.set_ordered_gaussian_priors(model, self.fit, self.data, "U")
                if rank == 0:
                    mkdir.assert_called_once_with(PRIOR_PATH.parent, parents=True, exist_ok=True)
                    write.assert_called_once_with(PRIOR_PATH, PRIOR_TEXT, encoding="utf-8")
                else:
                    mkdir.assert_not_called()
                    write.assert_not_called()
                self.assertEqual(events, (["write"] if rank == 0 else []) + ["barrier", "set_priors"])
                comm.Barrier.assert_called_once_with()
                model.set_priors.assert_called_once_with(str(PRIOR_PATH))
                model.print_para_names.assert_not_called()

    def test_unmatched_stage_does_not_write_synchronize_or_register(self):
        comm, model = mock.Mock(), mock.Mock()
        with (
            mock.patch.object(run_mica, "COMM", comm),
            mock.patch.object(run_mica, "RANK", 0),
            mock.patch.object(Path, "mkdir", autospec=True) as mkdir,
            mock.patch.object(Path, "write_text", autospec=True) as write,
        ):
            run_mica.set_ordered_gaussian_priors(model, make_fit(lag_prior=None), self.data)
        mkdir.assert_not_called()
        write.assert_not_called()
        comm.Barrier.assert_not_called()
        model.set_priors.assert_not_called()

    def test_all_worker_actions_setup_then_register_before_execution(self):
        import mica_resume_native

        execution = ["setup", "set_priors", "native_run", "decompose", "plot_results", "plot_decomp", "post_process"]
        expected = {
            "run": execution,
            "fresh": execution,
            "resume": execution,
            "post": ["setup", "set_priors", "post_run", "decompose", "plot_results", "plot_decomp", "post_process"],
            "plot": ["setup", "set_priors", "decompose", "plot_decomp"],
        }
        for action, methods in expected.items():
            with self.subTest(action=action):
                native, model, comm = mock.Mock(), mock.Mock(), mock.Mock()
                native.gmodel.return_value = model
                comm.Get_size.return_value = 2
                native_run = mock.Mock(return_value={"continuum_prior": {}})
                model.attach_mock(native_run, "native_run")
                mode = "fresh" if action in {"run", "fresh"} else action
                plan = {"mode": mode, "phase": "response", "state": {}}

                def root_result(function, *args, **kwargs):
                    # Filesystem preflight and state bookkeeping have their own
                    # tests. Here only native setup/prior/execution order matters.
                    if function.__name__ == "<lambda>":
                        return self.data
                    if function.__name__ in {"plan_run", "prepare_fresh", "prepare_existing"}:
                        return plan
                    if function is run_mica.tempfile.mkdtemp:
                        return "/not-created/preview"
                    return None

                with (
                    mock.patch.object(run_mica, "pymica", native),
                    mock.patch.object(run_mica, "COMM", comm),
                    mock.patch.object(run_mica, "RANK", 0),
                    mock.patch.object(run_mica, "root_call", side_effect=root_result),
                    mock.patch.object(mica_resume_native, "run_native_resume", native_run),
                    mock.patch.object(run_mica.importlib, "import_module",
                                      return_value=mock.Mock(__file__="/not-created/pymica.so")),
                    mock.patch.object(run_mica, "copy_inputs"),
                    mock.patch.object(run_mica, "have_decomp", return_value=False),
                    mock.patch.object(run_mica, "log"),
                    mock.patch.object(run_mica.os, "chdir"),
                    mock.patch.object(run_mica.shutil, "rmtree"),
                    mock.patch.object(Path, "mkdir", autospec=True),
                    mock.patch.object(Path, "write_text", autospec=True),
                ):
                    run_mica.run_band(Path("/not-created/Synthetic"), {"source": "Synthetic"},
                                      self.fit, "U", action)
                self.assertEqual([call[0] for call in model.mock_calls], methods)
                self.assertEqual(model.setup.call_args.kwargs["type_lag_prior"], 0)
                self.assertNotIn("lag_prior", model.setup.call_args.kwargs)
                model.set_priors.assert_called_once_with(str(PRIOR_PATH))
                model.print_para_names.assert_not_called()
                native.mmap.assert_not_called()
                if mode in {"fresh", "resume"}:
                    native_run.assert_called_once_with(
                        mixed=False, phase="fresh" if mode == "fresh" else "response"
                    )
                else:
                    native_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
