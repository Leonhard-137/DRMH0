"""Bounded concurrent scheduling checks; every science subprocess is mocked."""
from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import yaml

HERE = Path(__file__).resolve().parent
RUN_SPEC = importlib.util.spec_from_file_location("mica_smooth_controller_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(RUN_SPEC)
sys.modules[RUN_SPEC.name] = runner
RUN_SPEC.loader.exec_module(runner)
SPEC = importlib.util.spec_from_file_location("mica_smooth_controller", HERE / "orchestrate.py")
controller = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {"run": runner}):
    SPEC.loader.exec_module(controller)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.monitor = self.root / "monitoring"
        for target, value in (("HERE", self.root), ("MONITOR", self.monitor)):
            patcher = patch.object(controller, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.cfg = {
            "smoothing": {"output": "highpass_residual_plus_original_mean"},
        }
        for model, stage in controller.STAGES.items():
            self.cfg[stage] = {
                "type_tf": model, "ncomp": 2, "max_parallel": 5,
                "responses": ["UVM2", "UVW1", "U", "B", "V"],
            }
        for name in controller.CASES:
            directory = self.root / "cases" / name / "config"
            directory.mkdir(parents=True)
            (directory / "source_config.yaml").write_text(yaml.safe_dump(self.cfg))

    def status(self):
        return json.loads((self.monitor / "status.json").read_text())

    def test_three_cases_overlap_and_completed_case_is_immediately_replaced(self):
        started = {name: threading.Event() for name in controller.CASES}
        release = {name: threading.Event() for name in controller.CASES}
        active = set()
        observed = {"max_active": 0, "fourth_predecessor_phase": None}
        record_lock = threading.Lock()

        def finish_pipeline(argv, **kwargs):
            name = argv[argv.index("--case") + 1]
            self.assertEqual(argv[2:], [str(self.root / "run.py"), "all",
                                       "--case", name, "--model", "gamma", "gaussian",
                                       "--cases-dir", str(self.root / "cases")])
            state = self.status()
            self.assertEqual(state["max_parallel_cases"], 3)
            self.assertEqual(state["max_parallel_bands_per_case"], 5)
            self.assertEqual(state["model_order"], ["gamma", "gaussian"])
            self.assertEqual(state["cases"][name]["phase"], "pipeline")
            self.assertTrue(kwargs["check"])
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
            self.assertEqual(kwargs["env"]["MICA_SMOOTH_POST_LOCK"],
                             str(self.monitor / "postprocess.lock"))
            with record_lock:
                active.add(name)
                observed["max_active"] = max(observed["max_active"], len(active))
                if name == controller.CASES[3]:
                    observed["fourth_predecessor_phase"] = state["cases"][controller.CASES[0]]["phase"]
            started[name].set()
            try:
                if not release[name].wait(5):
                    raise TimeoutError("Test did not release mocked case")
            finally:
                with record_lock:
                    active.remove(name)

        with patch.object(controller.subprocess, "run", side_effect=finish_pipeline) as launch:
            with contextlib.redirect_stdout(io.StringIO()), ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(controller.run)
                try:
                    for name in controller.CASES[:3]:
                        self.assertTrue(started[name].wait(5), name)
                    self.assertFalse(started[controller.CASES[3]].is_set())
                    with record_lock:
                        self.assertEqual(active, set(controller.CASES[:3]))
                    release[controller.CASES[0]].set()
                    self.assertTrue(started[controller.CASES[3]].wait(5))
                    with record_lock:
                        self.assertEqual(active, set(controller.CASES[1:]))
                    self.assertEqual(observed["fourth_predecessor_phase"], "complete")
                finally:
                    for event in release.values():
                        event.set()
                result = future.result(timeout=5)
        self.assertEqual(result, 0)
        self.assertEqual(observed["max_active"], 3)
        self.assertEqual(launch.call_count, 4)
        self.assertEqual(self.status()["status"], "complete")
        self.assertTrue(all(case["phase"] == "complete" for case in self.status()["cases"].values()))
        self.assertEqual(len(list(self.monitor.glob("*.log"))), 4)

    def test_failed_case_stops_new_submissions_and_preserves_active_cases(self):
        started = {name: threading.Event() for name in controller.CASES}
        release = {name: threading.Event() for name in controller.CASES}
        failure_recorded = threading.Event()
        original_update = controller.Controller.update

        def track_update(instance, name=None, **changes):
            original_update(instance, name, **changes)
            if name is None and changes.get("status") == "failed":
                failure_recorded.set()

        def finish_pipeline(argv, **kwargs):
            name = argv[argv.index("--case") + 1]
            started[name].set()
            if not release[name].wait(5):
                raise TimeoutError("Test did not release mocked case")
            if name == controller.CASES[0]:
                raise subprocess.CalledProcessError(2, "mocked case pipeline")

        with patch.object(controller.subprocess, "run", side_effect=finish_pipeline) as launch:
            with patch.object(controller.Controller, "update", autospec=True, side_effect=track_update):
                with contextlib.redirect_stdout(io.StringIO()), ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(controller.run)
                    try:
                        for name in controller.CASES[:3]:
                            self.assertTrue(started[name].wait(5), name)
                        release[controller.CASES[0]].set()
                        self.assertTrue(failure_recorded.wait(5))
                        self.assertFalse(started[controller.CASES[3]].is_set())
                        self.assertFalse(future.done())
                        state = self.status()
                        for name in controller.CASES[1:3]:
                            self.assertEqual(state["cases"][name]["phase"], "pipeline")
                    finally:
                        for event in release.values():
                            event.set()
                    result = future.result(timeout=5)
        self.assertEqual(result, 1)
        self.assertEqual(launch.call_count, 3)
        state = self.status()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["failed_case"], controller.CASES[0])
        self.assertEqual(state["failed_cases"], [controller.CASES[0]])
        phases = [state["cases"][name]["phase"] for name in controller.CASES]
        self.assertEqual(phases, ["failed", "complete", "complete", "queued"])
        self.assertIn("CalledProcessError", state["cases"][controller.CASES[0]]["error"])
        self.assertFalse((self.monitor / f"{controller.CASES[3]}.log").exists())

    def test_validation_keeps_five_band_parallelism_inside_active_case(self):
        self.cfg[controller.STAGES["gamma"]]["max_parallel"] = 3
        target = self.root / "cases" / controller.CASES[-1] / "config/source_config.yaml"
        target.write_text(yaml.safe_dump(self.cfg))
        with patch.object(controller.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "exactly five bands concurrently"):
                controller.run()
        launch.assert_not_called()
        self.assertFalse(self.monitor.exists())

    def test_prepared_config_rejection_prevents_any_controller_or_fit_start(self):
        with patch.object(controller, "read_config", side_effect=ValueError("invalid smoothing output")):
            with patch.object(controller.subprocess, "run") as launch:
                with self.assertRaisesRegex(ValueError, "invalid smoothing output"):
                    controller.run()
        launch.assert_not_called()
        self.assertFalse(self.monitor.exists())

    def test_alternative_cases_directory_is_validated_forwarded_and_recorded(self):
        alternative = self.root / "corrected_cases"
        (self.root / "cases").rename(alternative)
        # Include a redundant segment to verify resolution before validation and launch.
        selected = alternative / ".." / alternative.name
        with patch.object(controller, "read_config", wraps=controller.read_config) as read_config:
            with patch.object(controller.subprocess, "run") as launch:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = controller.run(selected)
        self.assertEqual(result, 0)
        self.assertEqual([call.args[0] for call in read_config.call_args_list],
                         [alternative / name for name in controller.CASES])
        self.assertEqual(launch.call_count, 4)
        for call in launch.call_args_list:
            argv = call.args[0]
            self.assertEqual(argv[argv.index("--cases-dir") + 1], str(alternative))
        self.assertEqual(self.status()["cases_dir"], str(alternative))
        with patch.object(sys, "argv", ["orchestrate.py", "run", "--cases-dir", str(selected)]):
            with patch.object(controller, "run", return_value=0) as run:
                with self.assertRaises(SystemExit) as exit_result:
                    controller.main()
        self.assertEqual(exit_result.exception.code, 0)
        run.assert_called_once_with(selected)

    def test_default_action_only_reads_status(self):
        with patch.object(sys, "argv", ["orchestrate.py"]):
            with patch.object(controller, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                controller.main()
        run.assert_not_called()
        self.assertFalse(self.monitor.exists())


if __name__ == "__main__":
    unittest.main()
