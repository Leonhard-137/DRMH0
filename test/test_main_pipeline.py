"""Small orchestration checks for the root pipeline dispatcher."""
from pathlib import Path
import sys
import unittest
from unittest.mock import call, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as pipeline


class MainPipelineTests(unittest.TestCase):
    def test_corner_runs_once_immediately_after_collection_in_both_flows(self):
        for flow in (pipeline.FLOW, pipeline.STAGE_FLOW):
            with self.subTest(flow=flow):
                self.assertEqual(flow.count("corner"), 1)
                self.assertEqual(flow[flow.index("collect") + 1], "corner")

    @patch.object(pipeline, "python")
    @patch.object(pipeline, "read_config")
    def test_corner_uses_all_configured_stages(self, read_config, run_python):
        read_config.return_value = {
            "mica_stages": ["mica_gaussian2_free", "mica_gamma2_free100_host"],
            "mica_gaussian2_free": {"type_tf": "gaussian"},
            "mica_gamma2_free100_host": {"type_tf": "gamma"},
        }
        source = Path("/project/AGN")
        pipeline.execute(source, "corner")
        self.assertEqual(run_python.call_args_list, [
            call("plot_mica_corner.py", source,
                 "--stage", "mica_gaussian2_free"),
            call("plot_mica_corner.py", source,
                 "--stage", "mica_gamma2_free100_host"),
        ])

    @patch.object(pipeline, "run")
    @patch.object(pipeline, "read_config")
    def test_corner_cli_selects_stage_without_rerunning_sampler(self, read_config, run):
        read_config.return_value = {
            "mica_stages": ["mica_gaussian2_free"],
            "mica_gaussian2_free": {"type_tf": "gaussian"},
            "mica_gamma2_free100_host": {"type_tf": "gamma"},
        }
        source = Path("/project/AGN")
        pipeline.main([str(source), "corner", "--stage", "mica_gamma2_free100_host"])
        run.assert_called_once_with([
            sys.executable, pipeline.SRC / "plot_mica_corner.py", source,
            "--stage", "mica_gamma2_free100_host",
        ])

    def test_transfer_is_in_default_flow_and_stage_diagnostics(self):
        self.assertIn("transfer", pipeline.FLOW)
        self.assertIn("diagnostics", pipeline.STAGE_FLOW)
        self.assertLess(pipeline.FLOW.index("collect"), pipeline.FLOW.index("transfer"))

    @patch.object(pipeline, "python")
    @patch.object(pipeline, "read_config")
    def test_transfer_writes_both_representations(self, read_config, run_python):
        read_config.return_value = {"mica_gamma2_phys": {"type_tf": "2"}}
        source = Path("/project/AGN")
        pipeline.execute(source, "transfer", "mica_gamma2_phys")
        self.assertEqual(run_python.call_args_list, [
            call("plot_mica_transfer.py", source,
                 "--stage", "mica_gamma2_phys"),
            call("plot_mica_transfer.py", source,
                 "--stage", "mica_gamma2_phys", "--median-parameters"),
        ])

    @patch.object(pipeline, "python")
    @patch.object(pipeline, "read_config")
    def test_diagnostics_includes_transfer_products(self, read_config, run_python):
        read_config.return_value = {"mica_gamma2_phys": {"type_tf": "2"}}
        source = Path("/project/AGN")
        pipeline.execute(source, "diagnostics", "mica_gamma2_phys")
        self.assertEqual(run_python.call_args_list, [
            call("plot_mica_transfer.py", source,
                 "--stage", "mica_gamma2_phys"),
            call("plot_mica_transfer.py", source,
                 "--stage", "mica_gamma2_phys", "--median-parameters"),
            call("results_diagnostics.py", source,
                 "--stage", "mica_gamma2_phys"),
            call("sed_u_diagnostics.py", source,
                 "--stage", "mica_gamma2_phys"),
        ])

    @patch.object(pipeline, "run")
    def test_hubble_command_delegates_to_dedicated_script(self, run):
        pipeline.main(["hubble", "--config", "custom.yaml", "--dry-run"])
        run.assert_called_once_with([
            sys.executable,
            pipeline.SRC / "joint_hubble.py",
            "--config", "custom.yaml", "--dry-run",
        ])


if __name__ == "__main__":
    unittest.main()
