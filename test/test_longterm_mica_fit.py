"""Longterm MICA entrypoint tests; no MPI or sampling is started."""
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
spec = importlib.util.spec_from_file_location("longterm_mica_fit", HERE / "fit.py")
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


def config():
    return {
        "experiment": {"output_dir": "cases", "simulation_source": "SIM"},
        "observations": {"driver_band": "UVW2", "response_band": "U"},
        "drw": {"tau_days": [40, 4000]},
        "response": {"gaussian_cases": {"blended": {}, "separated": {}}},
        "independent_response": {"cases": {"disk_only": {}, "ind_short": {}, "ind_long": {}}},
        "shared_slow": {"cases": {"shared": {}}},
        "mica_fit": {"centroid_lag_days": [-100, 300], "sigma_days": [0.05, 100],
                     "flag_negative_resp": False, "max_num_saves": 10000,
                     "mpi_per_fit": 2, "num_particles": 1, "ptol": 0.1},
    }


class LongtermMicaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "experiment.yaml"
        self.path.write_text(yaml.safe_dump(config()))
        names = ["shared", "disk_only", "ind_short", "ind_long"] + [
            fit.case_name(tau, kind) for tau in (40, 4000) for kind in ("blended", "separated")]
        for name in names:
            directory = self.root / "cases" / name / "input"
            directory.mkdir(parents=True)
            for band in ("UVW2", "U"):
                (directory / f"SIM_{band}.txt").write_text("0 1 0.01\n1 1.1 0.02\n2 0.9 0.01\n")

    def run_cli(self, *args):
        with patch.object(sys, "argv", ["fit.py", "--config", str(self.path), *args]), redirect_stdout(io.StringIO()):
            fit.main()

    def test_priors_match_physical_widths_and_preserve_full_native_workflow(self):
        cfg = fit.source_config(config())
        gauss, gamma = [cfg[key] for key in cfg["mica_stages"]]
        np.testing.assert_allclose(gauss["width_prior"], np.sqrt(2)*np.asarray(gamma["width_prior"]))
        for stage in cfg["mica_stages"]:
            native = fit.run_mica.setup_kwargs(fit.run_mica.fit_config(cfg, stage), {"set1": [None, None]}, "U")
            self.assertEqual(native["lag_prior"], [[-100, 300], [-100, 300]])
            self.assertEqual(native["number_component"], [2, 2])
            self.assertEqual(native["type_lag_prior"], 4)
            self.assertFalse(native["flag_negative_resp"])
            self.assertFalse(native["flag_lag_posivity"])
            self.assertFalse(cfg[stage]["fit_only"])
            self.assertTrue(cfg[stage]["plot_decomp"])
            self.assertNotIn("fit_only", native)

    def test_case_and_model_selection(self):
        for suite, expected in ((None, 2), ("all", 16), ("gaussian", 8), ("independent", 6), ("shared_slow", 2)):
            with self.subTest(suite=suite):
                jobs = fit.build_plan(self.path, suite=suite)
                self.assertEqual(len(jobs), expected)
                self.assertEqual(len({job["run_dir"] for job in jobs}), expected)
        self.assertEqual(len(fit.build_plan(self.path, cases=["disk_only"], model="gamma")), 1)
        for arguments in ({"cases": ["unknown"]}, {"cases": ["disk_only"], "suite": "shared_slow"}):
            with self.assertRaises(ValueError):
                fit.build_plan(self.path, **arguments)

    def test_known_response_only_config_and_native_sampling_settings(self):
        cfg = config()
        for section in ("drw", "response", "independent_response", "shared_slow"):
            cfg.pop(section)
        names = ["disk_shared_tau150d_lag015d", "disk_plus_broad_gaussian_mu030d_sigma080d"]
        cfg["known_response"] = {"cases": dict.fromkeys(names, {})}
        cfg["mica_fit"].update(sigma_days=[0.05, 300], num_particles=4, ptol=0.01)
        self.path = self.root / "experiment_broad.yaml"
        self.path.write_text(yaml.safe_dump(cfg))
        for name in names:
            directory = self.root / "cases" / name / "input"
            directory.mkdir(parents=True)
            for band in ("UVW2", "U"):
                (directory / f"SIM_{band}.txt").write_text("0 1 0.01\n1 1.1 0.02\n2 0.9 0.01\n")

        jobs = fit.build_plan(self.path, suite="known_response")
        self.assertEqual(len(jobs), 4)
        self.assertEqual({job["case"].name for job in jobs}, set(names))
        self.assertEqual(len(fit.build_plan(self.path, suite="all")), 4)
        self.assertEqual(len(fit.build_plan(self.path, cases=[names[0]], model="gamma")), 1)
        with self.assertRaisesRegex(ValueError, "does not belong"):
            fit.build_plan(self.path, cases=[names[0]], suite="shared_slow")
        for job in jobs:
            stage_fit = job["fit"]
            native = fit.run_mica.setup_kwargs(stage_fit, {"set1": [None, None]}, "U")
            self.assertEqual(native["num_particles"], 4)
            self.assertEqual(native["ptol"], 0.01)
            self.assertEqual(native["max_num_saves"], 10000)
            self.assertEqual(native["lag_prior"], [[-100, 300], [-100, 300]])
            self.assertFalse(native["flag_negative_resp"])
            self.assertEqual(stage_fit["mpi_per_band"], 2)
            scale = np.sqrt(2) if stage_fit["type_tf"] == "gamma" else 1
            np.testing.assert_allclose(np.asarray(native["width_prior"]) * scale,
                                       [[0.05, 300], [0.05, 300]])
            self.assertFalse(stage_fit["fit_only"])
            self.assertTrue(stage_fit["plot_decomp"])

        before = set(self.root.rglob("*"))
        with patch.object(fit.run_mica, "launch") as launch:
            self.run_cli("--suite", "known_response")
            launch.assert_not_called()
            self.assertEqual(set(self.root.rglob("*")), before)
            self.run_cli("--suite", "known_response", "--run")
        self.assertEqual(launch.call_count, 4)
        for call in launch.call_args_list:
            self.assertEqual(call.args[1]["num_particles"], 4)
            self.assertEqual(call.args[1]["ptol"], 0.01)
        for name in names:
            self.assertFalse((self.root / "cases" / name / "runs").exists())

    def test_default_validation_is_read_only(self):
        before = set(self.root.rglob("*"))
        with patch.object(fit.run_mica, "launch") as launch:
            self.run_cli()
        launch.assert_not_called()
        self.assertEqual(set(self.root.rglob("*")), before)

    def test_prepare_only_creates_config_but_no_runs(self):
        with patch.object(fit.run_mica, "launch") as launch:
            self.run_cli("--prepare-only")
        launch.assert_not_called()
        case = self.root / "cases/shared"
        self.assertEqual(fit.run_mica.read_yaml(case / "config/source_config.yaml"), fit.source_config(config()))
        self.assertFalse((case / "runs").exists())

    def test_run_delegates_both_stages_without_fit_only(self):
        with patch.object(fit.run_mica, "launch") as launch:
            self.run_cli("--run")
        self.assertEqual(launch.call_count, 2)
        for call, family in zip(launch.call_args_list, ("gaussian", "gamma")):
            case, stage_fit, responses, action, stage = call.args
            self.assertEqual(case, self.root / "cases/shared")
            self.assertEqual((responses, action, stage), (["U"], "run", f"mica_{family}2"))
            self.assertFalse(stage_fit["fit_only"])
            self.assertTrue(stage_fit["plot_decomp"])
        self.assertFalse((self.root / "cases/shared/runs").exists())

    def test_different_config_with_existing_run_is_not_overwritten(self):
        self.run_cli("--prepare-only")
        case = self.root / "cases/shared"
        runs = case / "runs/mica"
        runs.mkdir(parents=True)
        old = (case / "config/source_config.yaml").read_bytes()
        changed = fit.source_config(config())
        changed["mica_gaussian2"]["lag_limit"] = [-10, 100]
        with self.assertRaises(ValueError):
            fit.write_config(case, changed)
        self.assertEqual((case / "config/source_config.yaml").read_bytes(), old)

    def test_reject_invalid_parameters_and_input_before_launch(self):
        for key, value in (("sigma_days", [0, 100]), ("centroid_lag_days", [3, 2]),
                           ("max_num_saves", 0), ("mpi_per_fit", 2.5), ("ptol", 1),
                           ("flag_negative_resp", "false")):
            cfg = deepcopy(config())
            cfg["mica_fit"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                fit.source_config(cfg)
        (self.root / "cases/shared/input/SIM_U.txt").write_text("0 1 0\n1 1 0.01\n")
        with patch.object(fit.run_mica, "launch") as launch, self.assertRaises(ValueError):
            self.run_cli("--run")
        launch.assert_not_called()

    def test_batches_are_bounded_and_wait_for_whole_group(self):
        jobs = fit.build_plan(self.path, suite="all")
        processes, live, peaks, boundaries = [], [], [], []
        class Process:
            def __init__(self, command, **kwargs):
                self.command, self.returncode, self.pid = command, None, len(processes) + 1
                self.polls = 0
                if not live:
                    boundaries.append(len(processes))
                processes.append(self)
                live.append(self)
                peaks.append(len(live))
            def poll(self):
                self.polls += 1
                if self.returncode is None and self.polls >= 2:
                    self.returncode = 0
                    live.remove(self)
                return self.returncode
            def wait(self, timeout=None):
                return self.returncode
            def terminate(self):
                raise AssertionError("Completed fits should not be terminated")
        with patch.object(fit.subprocess, "Popen", Process), patch.object(fit.time, "sleep"), redirect_stdout(io.StringIO()):
            fit.run_batches(jobs, self.path, 5)
        self.assertEqual(len(processes), 16)
        self.assertEqual(max(peaks), 5)
        self.assertEqual(boundaries, [0, 5, 10, 15])
        for process in processes:
            self.assertNotIn("--batch-size", process.command)
            self.assertIn("--run", process.command)

    def test_batch_failure_stops_remaining_batches_and_cleans_owned_children(self):
        jobs = fit.build_plan(self.path, suite="all")
        processes = []
        class Process:
            def __init__(self, command, **kwargs):
                self.pid, self.returncode = len(processes) + 1, None
                self.terminated, self.waited = False, False
                processes.append(self)
            def poll(self):
                if self.pid == 1:
                    self.returncode = 1
                return self.returncode
            def terminate(self):
                self.terminated, self.returncode = True, -15
            def wait(self, timeout=None):
                self.waited = True
                return self.returncode
        with patch.object(fit.subprocess, "Popen", Process), redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "later batches will not start"):
            fit.run_batches(jobs, self.path, 5)
        self.assertEqual(len(processes), 5)
        self.assertTrue(all(process.waited for process in processes))
        self.assertTrue(all(process.terminated for process in processes[1:]))


if __name__ == "__main__":
    unittest.main()
