"""Tests for the one-off serial pipeline; no reconstruction or FFA is run."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mrk817_free100_sed_pipeline",
                                            ROOT / "scripts/run_mrk817_free100_sed.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class Free100SedPipelineTests(unittest.TestCase):
    def test_commands_use_existing_full_posterior_entrypoints(self):
        decompose, ffa = pipeline.commands("raw")
        self.assertIn("--skip-complete", decompose)
        self.assertEqual(decompose[decompose.index("--branch") + 1], "full")
        self.assertEqual(Path(decompose[decompose.index("--config") + 1]).name,
                         "free100_raw_sed.yaml")
        self.assertEqual(ffa[:3], [str(Path(sys.executable).with_name("mpiexec")), "-n", "2"])
        self.assertEqual(ffa[ffa.index("--steps") + 1], "prep,dered,fit,summ,fvg,sedfit")
        self.assertEqual(ffa[ffa.index("--posterior-branch") + 1], "full")

    def test_pid_identity_rejects_worker_other_group_and_other_script(self):
        good = ["python", "/src/decompose_mica_branches.py", "Mrk817", "--config",
                "/config/free100_host_sed.yaml", "--branch", "full"]
        pipeline.validate_host_controller(good, 123)
        for wrong in (good + ["--worker"],
                      [value.replace("free100_host", "free100_raw") for value in good],
                      [value.replace("decompose_mica_branches.py", "run_ffa.py") for value in good]):
            with self.subTest(command=wrong), self.assertRaises(RuntimeError):
                pipeline.validate_host_controller(wrong, 123)

    def test_wait_polls_existing_controller_without_launching(self):
        good = ["python", "decompose_mica_branches.py", "--config", "free100_host_sed.yaml"]
        with patch.object(pipeline, "controller_cmdline", side_effect=[good, None]), \
                patch.object(pipeline.time, "sleep") as sleep, \
                patch.object(pipeline.subprocess, "run") as run:
            pipeline.wait_host_controller(123)
        sleep.assert_called_once_with(30)
        run.assert_not_called()

    def test_incomplete_ffa_samples_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "SOURCE", Path(directory)):
            sample = pipeline.group_root("host") / "comp0/fflux/fit/sample.txt"
            sample.parent.mkdir(parents=True)
            sample.write_text("# existing samples\n1 2 3\n")
            with patch.object(pipeline.subprocess, "run") as run, self.assertRaisesRegex(
                RuntimeError, "no overwrite"
            ):
                pipeline.run_group("host")
            run.assert_not_called()
            self.assertEqual(sample.read_text(), "# existing samples\n1 2 3\n")

    def test_complete_sed_is_skipped_only_when_all_core_files_are_nonempty(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "SOURCE", Path(directory)):
            for path in pipeline.sed_outputs("host"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("completed artifact\n")
            self.assertTrue(pipeline.sed_complete("host"))
            with patch.object(pipeline.subprocess, "run") as run:
                pipeline.run_group("host")
            run.assert_not_called()
            pipeline.sed_outputs("host")[-1].write_text("")
            self.assertFalse(pipeline.sed_complete("host"))

    def test_waited_host_validates_before_ffa_without_second_reconstruction(self):
        order = []
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, "SOURCE", Path(directory)), \
                patch.object(pipeline, "sed_complete", side_effect=[False, True]), \
                patch.object(pipeline, "commands", return_value=(["decompose"], [sys.executable, "ffa"])), \
                patch.object(pipeline, "wait_host_controller", side_effect=lambda pid: order.append("wait")), \
                patch.object(pipeline, "validate_decomposition", side_effect=lambda group: order.append("validate")), \
                patch.object(pipeline.subprocess, "run", side_effect=lambda *a, **k: order.append("ffa")) as run:
            pipeline.run_group("host", wait_pid=123)
        self.assertEqual(order, ["wait", "validate", "ffa"])
        run.assert_called_once()

    def test_main_runs_host_then_raw(self):
        with patch.object(sys, "argv", ["pipeline", "--wait-host-decomposition-pid", "123"]), \
                patch.object(pipeline, "run_group") as run:
            pipeline.main()
        self.assertEqual(run.call_args_list[0].args, ("host",))
        self.assertEqual(run.call_args_list[0].kwargs, {"wait_pid": 123})
        self.assertEqual(run.call_args_list[1].args, ("raw",))


if __name__ == "__main__":
    unittest.main()
