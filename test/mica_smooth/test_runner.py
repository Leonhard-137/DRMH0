"""Bounded runner checks: no sampler or science-data writes."""
from __future__ import annotations

import contextlib
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("mica_smooth_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def fixture_config():
    cfg = {
        "smoothing": {"output": "highpass_residual_plus_original_mean"},
        "source": "Mrk817_a", "bands": {"UVW2": 1928, "U": 3465},
        "pipeline": {"ffa_mpi": 2},
        "fflux_comp0": {"fit": {"max_saves": 10000, "tau_min": 30},
                        "extinction": {"law": "F99", "ebv": 0, "rv": 3.1}},
    }
    for model, stage in runner.STAGES.items():
        cfg[stage] = {
            "type_tf": model, "ncomp": 2, "component_order": "moment1" if model == "gamma" else "native",
            "prepared_dir": "prepared", "driver": "UVW2", "responses": ["U"],
            "output_root": f"runs/mica/{model}/free", "lag_prior": [[0, 100], [0, 100]],
            "posterior_branches": {"enabled": False},
        }
    return cfg


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name)
        (self.case / "config").mkdir()
        self.cfg = fixture_config()
        (self.case / "config/source_config.yaml").write_text(yaml.safe_dump(self.cfg))

    def test_all_and_post_put_comp0_ffa_after_mica_and_before_lagfit(self):
        plan = runner.pipeline_commands(self.case, "gamma", "all", sys.executable)
        self.assertEqual([c.label for c in plan],
                         ["mica", "collect", "corner", "transfer", "decompose_full", "sed", "lagfit"])
        for command in plan:
            self.assertNotIn("all", command.argv)
            self.assertNotIn("fresh", command.argv)
            self.assertNotIn("distance", command.argv)
        self.assertEqual(Path(plan[1].argv[2]).name, "collect_mica_results.py")
        self.assertEqual([c.label for c in runner.pipeline_commands(self.case, "gamma", "post", sys.executable)],
                         ["collect", "corner", "transfer", "decompose_full", "sed", "lagfit"])
        ffa = plan[5].argv
        self.assertEqual(ffa[ffa.index("--config-section") + 1], "fflux_comp0")
        self.assertNotIn("host", ffa[-1].split(","))
        self.assertNotIn("run_ffa.py", " ".join(plan[0].argv))

    def test_lowpass_inputs_are_rejected_before_any_subprocess(self):
        self.cfg["smoothing"]["output"] = "smoothed_flux"
        (self.case / "config/source_config.yaml").write_text(yaml.safe_dump(self.cfg))
        with patch.object(runner.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "Old low-pass inputs"):
                runner.run_case(self.case, "gaussian", "all", False)
        launch.assert_not_called()
        self.assertFalse((self.case / "logs").exists())

    def test_missing_corner_stops_before_transfer_and_ffa(self):
        with patch.object(runner, "execute") as execute:
            with self.assertRaisesRegex(RuntimeError, "Missing outputs after corner"):
                runner.run_case(self.case, "gaussian", "all", False)
        self.assertEqual([c.args[0].label for c in execute.call_args_list],
                         ["mica", "collect", "corner"])

    def test_required_figures_include_every_band_and_both_transfer_forms(self):
        self.cfg[runner.STAGES["gaussian"]]["responses"] = ["UVM2", "UVW1", "U", "B", "V"]
        corner = runner.step_outputs(self.case, self.cfg, "gaussian", "corner")
        transfer = runner.step_outputs(self.case, self.cfg, "gaussian", "transfer")
        lag = runner.step_outputs(self.case, self.cfg, "gaussian", "lagfit")
        self.assertEqual(len(corner), 10)
        self.assertEqual(len(transfer), 44)
        self.assertEqual([p.name for p in lag], ["tau_tf0.png", "tau_tf0.pdf"])
        for p in corner + transfer + lag:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"mock figure")
        for step in ("corner", "transfer", "lagfit"):
            runner.check_step_outputs(self.case, self.cfg, "gaussian", step)
        (next(p for p in transfer if p.name == "Mrk817_a_V_total_transfer_median_parameters.pdf")).unlink()
        with self.assertRaisesRegex(RuntimeError, "V_total_transfer_median_parameters.pdf"):
            runner.check_step_outputs(self.case, self.cfg, "gaussian", "transfer")

    def test_main_stage_commands_are_accepted_by_actual_cli(self):
        spec = importlib.util.spec_from_file_location("project_main_for_smooth_test", runner.ROOT / "main.py")
        main = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main)
        for command in runner.pipeline_commands(self.case, "gamma", "all", sys.executable):
            if Path(command.argv[2]).name == "main.py":
                with patch.object(main, "execute") as execute:
                    main.main(list(command.argv[3:]))
                execute.assert_called_once_with(self.case.resolve(), command.label, "mica_gamma2_free")

    def test_gamma_sed_uses_sorted_full_posterior_and_case_driver(self):
        before = deepcopy(self.cfg)
        overlay = runner.make_sed_config(self.cfg, "gamma")
        self.assertEqual(self.cfg, before)
        fit = overlay["mica_round1"]
        self.assertEqual(fit["component_order"], "moment1")
        self.assertEqual(fit["lag_prior"], [[0, 100], [0, 100]])
        self.assertEqual(fit["posterior_branches"]["branches"]["full"]["ranges"], {"tau0": [None, None]})
        self.assertEqual(fit["posterior_branches"]["max_parallel"], 1)
        ff = overlay["fflux_comp0"]
        self.assertEqual(ff["mica"]["driver_file"], "prepared/Mrk817_a_UVW2.txt")
        self.assertEqual(ff["extinction"]["ebv"], 0)
        self.assertEqual(ff["fit"]["tau_min"], 30)
        commands = runner.sed_commands(self.case, "gamma", self.cfg, sys.executable)
        self.assertIn("--skip-complete", commands[0].argv)
        self.assertIn("--posterior-branch", commands[1].argv)
        self.assertNotIn("--mica-stage", commands[1].argv)
        self.assertNotIn("distance_flux", commands[1].argv[-1])

    def test_all_post_and_sed_dry_run_include_comp0_ffa_without_writes(self):
        for action in ("all", "post", "sed"):
            before = sorted(p.relative_to(self.case) for p in self.case.rglob("*"))
            output = io.StringIO()
            with patch.object(runner.subprocess, "run") as launch, contextlib.redirect_stdout(output):
                runner.run_case(self.case, "gamma", action, True)
            launch.assert_not_called()
            self.assertEqual(before, sorted(p.relative_to(self.case) for p in self.case.rglob("*")))
            self.assertIn("run_ffa.py", output.getvalue())
            self.assertIn("--config-section fflux_comp0", output.getvalue())

    def test_incomplete_ffa_samples_are_not_overwritten(self):
        overlay = runner.make_sed_config(self.cfg, "gamma")
        sample = runner.sed_root(self.case, overlay) / "fit/sample.txt"
        sample.parent.mkdir(parents=True)
        sample.write_text("existing sample\n")
        with patch.object(runner, "execute") as execute, self.assertRaisesRegex(RuntimeError, "existing FFA samples"):
            runner.run_case(self.case, "gamma", "all", False)
        execute.assert_not_called()
        self.assertEqual(sample.read_text(), "existing sample\n")

    def test_completed_sed_is_reused_and_cli_sed_needs_no_flag(self):
        overlay = runner.make_sed_config(self.cfg, "gamma")
        for output in runner.sed_outputs(self.case, overlay):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("finished\n")
        with patch.object(runner, "execute") as execute:
            runner.run_case(self.case, "gamma", "sed", False)
        execute.assert_not_called()
        for extra in ([], ["--with-sed"]):
            with patch.object(runner, "run_case") as run_case:
                runner.main(["sed", "--case", "all_s20", "--model", "gamma", "--dry-run", *extra])
            run_case.assert_called_once_with(runner.HERE / "cases/all_s20", "gamma", "sed", True)

    def test_four_cases_two_models_include_one_post_mica_ffa_each_by_default(self):
        for name in runner.CASES:
            case = self.case / "cases" / name
            (case / "config").mkdir(parents=True)
            (case / "config/source_config.yaml").write_text(yaml.safe_dump(self.cfg))
        output = io.StringIO()
        with patch.object(runner, "HERE", self.case), patch.object(runner.subprocess, "run") as launch:
            with contextlib.redirect_stdout(output):
                runner.main(["all", "--case", "all", "--model", "all", "--dry-run"])
        launch.assert_not_called()
        commands = [line for line in output.getvalue().splitlines() if line.startswith("$ ")]
        self.assertEqual(len(commands), 4 * 2 * 7)
        for index, (name, model) in enumerate((n, m) for n in runner.CASES for m in runner.STAGES):
            group = commands[index * 7:(index + 1) * 7]
            self.assertTrue(all(name in command for command in group))
            self.assertIn(" mica --stage ", group[0])
            self.assertIn("collect_mica_results.py", group[1])
            self.assertIn(" corner --stage ", group[2])
            self.assertIn(" transfer --stage ", group[3])
            self.assertIn("decompose_mica_branches.py", group[4])
            self.assertIn("run_ffa.py", group[5])
            self.assertIn("--config-section fflux_comp0", group[5])
            self.assertIn(f"{model}_sed.yaml", group[5])
            self.assertIn(" lagfit --stage ", group[6])
        self.assertEqual(output.getvalue().count("run_ffa.py"), 8)
        self.assertFalse(any(self.case.rglob("logs")))
        self.assertFalse(any(self.case.rglob("*_sed.yaml")))

    def test_alternative_cases_directory_routes_to_selected_case(self):
        with patch.object(runner, "run_case") as run_case:
            runner.main(["post", "--cases-dir", str(self.case), "--case", "all_s20",
                         "--model", "gamma", "--dry-run"])
        run_case.assert_called_once_with(self.case / "all_s20", "gamma", "post", True)

    def test_failures_stop_before_later_pipeline_steps(self):
        with patch.object(runner, "execute", side_effect=subprocess.CalledProcessError(2, "mica")) as execute:
            with self.assertRaises(subprocess.CalledProcessError):
                runner.run_case(self.case, "gaussian", "all", False)
        self.assertEqual(execute.call_count, 1)

    def test_all_models_run_gamma_before_gaussian(self):
        with patch.object(runner, "run_case") as run_case:
            runner.main(["all", "--case", "responses_s20", "--dry-run"])
        self.assertEqual([call.args[1] for call in run_case.call_args_list],
                         ["gamma", "gaussian"])

    def test_post_lock_is_released_after_failure_and_mica_does_not_lock(self):
        lock = self.case / "monitoring/postprocess.lock"
        with patch.dict(runner.os.environ, {"MICA_SMOOTH_POST_LOCK": str(lock)}):
            with patch.object(runner.fcntl, "flock") as flock:
                with runner.postprocess_slot(runner.Command("mica", ())):
                    pass
                flock.assert_not_called()
                self.assertFalse(lock.exists())
                with self.assertRaisesRegex(RuntimeError, "post failed"):
                    with runner.postprocess_slot(runner.Command("corner", ())):
                        raise RuntimeError("post failed")
                self.assertEqual([call.args[1] for call in flock.call_args_list],
                                 [runner.fcntl.LOCK_EX, runner.fcntl.LOCK_UN])

    def test_command_logs_append_and_subprocess_is_synchronous(self):
        command = runner.Command("mica", (sys.executable, "-c", "pass"))
        with patch.object(runner.subprocess, "run") as launch:
            runner.execute(command, self.case, "gaussian", False)
            runner.execute(command, self.case, "gaussian", False)
        self.assertEqual(launch.call_count, 2)
        for call in launch.call_args_list:
            self.assertTrue(call.kwargs["check"])
            self.assertEqual(call.kwargs["env"]["OMP_NUM_THREADS"], "1")
        self.assertEqual((self.case / "logs/gaussian_mica.log").read_text().count("-c pass"), 2)


if __name__ == "__main__":
    unittest.main()
