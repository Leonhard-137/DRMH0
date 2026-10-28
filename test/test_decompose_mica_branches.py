from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import decompose_mica_branches as decompose  # noqa: E402
from decompose_mica_branches import (  # noqa: E402
    DNEST_STR_MAX_LENGTH,
    DNestOptions,
    decomposition_settings,
    file_digest,
    posterior_column_count,
    systematic_resample,
    validate_decomposition_file,
    validate_run_outputs,
)
from collect_mica_results import (  # noqa: E402
    component_parameter_arrays,
    configured_run_name,
    configured_stages,
    mica_analysis_dir,
    mica_run_root,
    model_kinds,
    model_label,
    ncomp_from_config,
    read_applied_priors,
    reorder_component_columns,
)
import ctypes


class DecompositionConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name)
        self.relative = Path("config/selected.yaml")
        self.selected = self.source / self.relative
        self.selected.parent.mkdir()
        self.selected.write_text(
            "mica_round1:\n"
            "  type_tf: gamma\n"
            "  component_order: moment1\n"
            "  posterior_branches:\n"
            "    enabled: true\n"
            "    branches:\n"
            "      full:\n"
            "        ranges: {tau0: [null, null]}\n",
            encoding="utf-8",
        )

    def test_config_path_preserves_default_and_resolves_source_relative(self):
        self.assertEqual(decompose.config_path(self.source),
                         self.source / "config/source_config.yaml")
        self.assertEqual(decompose.config_path(self.source, self.relative), self.selected)
        self.assertEqual(decompose.config_path(self.source, self.selected), self.selected)

    def test_cli_reads_selected_definitions_and_forwards_config(self):
        argv = ["decompose_mica_branches.py", str(self.source), "--config", str(self.relative),
                "--branch", "full", "--stage-only"]
        with patch.object(sys, "argv", argv), patch.object(decompose, "controller") as controller:
            decompose.main()
        controller.assert_called_once_with(
            self.source, ["full"], None, True, False, config_file=self.relative,
        )

    def test_worker_cli_forwards_config(self):
        argv = ["decompose_mica_branches.py", str(self.source), "--config", str(self.relative),
                "--worker", "--branch", "full", "--band", "U"]
        with patch.object(sys, "argv", argv), patch.object(decompose, "worker") as worker:
            decompose.main()
        worker.assert_called_once_with(self.source, "full", "U", config_file=self.relative)

    def test_controller_uses_selected_config_without_default_file(self):
        with patch.object(decompose, "run_dirs", return_value=[]) as runs:
            decompose.controller(self.source, [], None, True, False, config_file=self.relative)
        self.assertEqual(runs.call_args.args[1]["type_tf"], "gamma")
        self.assertEqual(runs.call_args.args[1]["component_order"], "moment1")

    def test_worker_reads_selected_config_before_native_work(self):
        fit_config = Mock(return_value={"type_tf": "gamma"})
        mpi = SimpleNamespace(COMM_WORLD=SimpleNamespace(Get_rank=lambda: 0))
        modules = {
            "mpi4py": SimpleNamespace(MPI=mpi),
            "pymica": SimpleNamespace(),
            "run_mica": SimpleNamespace(fit_config=fit_config, setup_kwargs=Mock()),
        }
        with patch.dict(sys.modules, modules), patch.object(
            decompose, "branch_run_dir", return_value=self.source / "missing_staged_run"
        ), self.assertRaises(FileNotFoundError):
            decompose.worker(self.source, "full", "U", config_file=self.relative)
        self.assertEqual(fit_config.call_args.args[0]["mica_round1"]["component_order"], "moment1")

    def test_launch_forwards_absolute_config_and_default_omits_flag(self):
        mpi = self.source / "mpiexec"
        mpi.touch()
        process = Mock()
        process.wait.return_value = 0
        for selected in (None, self.relative):
            with self.subTest(config=selected), patch.object(
                decompose, "branch_root", return_value=self.source / "branch"
            ), patch.object(decompose.subprocess, "Popen", return_value=process) as launch:
                decompose.launch_group(self.source, {}, "full", ["U"],
                                       {"mpiexec": str(mpi)}, config_file=selected)
            command = launch.call_args.args[0]
            if selected is None:
                self.assertNotIn("--config", command)
            else:
                self.assertEqual(command[command.index("--config") + 1], str(self.selected))


class SystematicResamplingTests(unittest.TestCase):
    def test_resampling_is_deterministic_for_a_seed(self):
        weights = np.array([0.1, 0.2, 0.7])
        first = systematic_resample(weights, 1000, np.random.default_rng(1234))
        second = systematic_resample(weights, 1000, np.random.default_rng(1234))
        np.testing.assert_array_equal(first, second)

    def test_resampling_counts_follow_weights(self):
        weights = np.array([0.1, 0.2, 0.7])
        draws = systematic_resample(weights, 1000, np.random.default_rng(9876))
        counts = np.bincount(draws, minlength=3)
        np.testing.assert_allclose(counts, [100, 200, 700], atol=1)

    def test_invalid_sample_size_is_rejected(self):
        with self.assertRaises(ValueError):
            systematic_resample([1.0], 0, np.random.default_rng(1))


class MicaLagParameterizationTests(unittest.TestCase):
    @staticmethod
    def names(location: str) -> list[str]:
        return [
            "con_systematic_error",
            "con_sigma",
            "con_tau",
            "line_systematic_error",
            "0-th_component_amplitude",
            f"0-th_component_{location}",
            "0-th_component_sigma",
            "1-th_component_amplitude",
            f"1-th_component_{location}",
            "1-th_component_sigma",
        ]

    def test_old_center_and_new_centroid_encode_same_gamma(self):
        old = np.array([[0, 0, 0, 0, np.log(3), 1, np.log(2), np.log(1), 20, 0]])
        new = old.copy()
        new[0, 5] = 5.0
        new[0, 8] = 22.0

        old_values = component_parameter_arrays(
            old, self.names("center"), 2, "gamma"
        )
        new_values = component_parameter_arrays(
            new, self.names("centroid"), 2, "gamma"
        )
        for old_value, new_value in zip(old_values, new_values):
            np.testing.assert_allclose(old_value, new_value)

    def test_new_gamma_components_are_reordered_by_centroid(self):
        sample = np.array([
            [0, 0, 0, 0, np.log(2), 30, np.log(10), np.log(5), 22, 0]
        ])
        names = self.names("centroid")
        ordered, swapped = reorder_component_columns(
            sample,
            names,
            2,
            "gamma",
            {"component_order": "moment1"},
        )
        _, _, amplitudes, centroids = component_parameter_arrays(
            ordered, names, 2, "gamma"
        )
        np.testing.assert_array_equal(swapped, [True])
        np.testing.assert_allclose(centroids[:, 0], [22.0, 30.0])
        np.testing.assert_allclose(amplitudes[:, 0], [5.0, 2.0])


class MixedMicaCollectionTests(unittest.TestCase):
    def test_native_mmap_code_has_canonical_label_and_path(self):
        fit = {
            "output_root": "runs/mica/mix220/phys12_free3",
            "type_tf": "220",
            "number_component": [3, 3],
        }
        self.assertEqual(model_label("220"), "mix220")
        self.assertEqual(model_label("mix220"), "mix220")
        self.assertEqual(
            model_kinds("mix220", 3),
            ["gamma", "gamma", "gaussian"],
        )
        self.assertEqual(ncomp_from_config(fit), 3)
        root = ROOT / "temporary_source"
        self.assertEqual(
            mica_run_root(root, fit),
            root / "runs/mica/mix220/phys12_free3",
        )
        self.assertEqual(
            mica_analysis_dir(root, fit),
            root / "runs/mica/mix220/phys12_free3/result/comp0",
        )
        self.assertEqual(
            configured_run_name({
                **fit,
                "driver": "W2",
                "run_template": "run_{driver}_to_{band}_{ncomp}comp_{type_tf}",
            }, "U"),
            "run_W2_to_U_3comp_mix220",
        )

    def test_active_stages_are_explicit_and_ordered(self):
        cfg = {
            "mica_stages": ["mica_gamma2_phys", "mica_mix220"],
            "mica_gamma2_phys": {"type_tf": "gamma"},
            "mica_mix220": {"type_tf": "220"},
        }
        self.assertEqual(
            configured_stages(cfg),
            ["mica_gamma2_phys", "mica_mix220"],
        )

    def test_new_mixed_centroids_are_not_shifted_again(self):
        names = [
            "con_systematic_error", "con_sigma", "con_tau", "line_systematic_error",
            "0-th_component_amplitude", "0-th_component_centroid", "0-th_component_sigma",
            "1-th_component_amplitude", "1-th_component_centroid", "1-th_component_sigma",
            "2-th_component_amplitude", "2-th_component_centroid", "2-th_component_sigma",
        ]
        sample = np.array([[
            0, 0, 0, 0,
            np.log(2), 0.4, np.log(0.1),
            np.log(3), 6.0, np.log(0.5),
            np.log(5), 80.0, np.log(10.0),
        ]])
        centers, _, amplitudes, centroids = component_parameter_arrays(
            sample, names, 3, "mix220"
        )
        np.testing.assert_allclose(centroids[:, 0], [0.4, 6.0, 80.0])
        np.testing.assert_allclose(centers[:, 0], [0.2, 5.0, 80.0])
        np.testing.assert_allclose(amplitudes[:, 0], [2.0, 3.0, 5.0])

    def test_native_centroid_priors_are_collected(self):
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / "para_names_line.txt_3"
            native.write_text(
                " 4 0-th_component_amplitude LOG -11.5 2.3 0 0\n"
                " 5 0-th_component_centroid UNI 0.0 1.0 0 0\n"
                " 6 0-th_component_sigma LOG -3.0 3.0 0 0\n"
                " 8 1-th_component_centroid UNI 3.0 40.0 0 0\n"
                "11 2-th_component_centroid UNI 40.0 120.0 0 0\n",
                encoding="utf-8",
            )
            rows = read_applied_priors(
                Path(directory) / "missing.csv",
                "U",
                native_path=native,
            )
        tau = [row for row in rows if row["quantity"] == "tau"]
        self.assertEqual(
            [(row["component"], row["prior_low"], row["prior_high"]) for row in tau],
            [("tf0", 0.0, 1.0), ("tf1", 3.0, 40.0), ("tf2", 40.0, 120.0)],
        )


class NativeLayoutTests(unittest.TestCase):
    def test_dnest_options_layout_matches_64_bit_c_struct(self):
        self.assertEqual(ctypes.sizeof(DNestOptions), 1176)
        self.assertEqual(DNestOptions.posterior_sample_file.offset, 704)
        self.assertEqual(
            DNestOptions.limits_file.offset + DNEST_STR_MAX_LENGTH,
            1172,
        )

    def test_posterior_column_count_ignores_header(self):
        path = (
            ROOT
            / "Mrk142/runs/mica_gaussian/2comp"
            / "run_W2_to_M2_2comp_gaussian/data/posterior_sample1d.txt_2"
        )
        if not path.is_file():
            self.skipTest("Mrk142 MICA posterior is not available")
        self.assertGreater(posterior_column_count(path), 3)

    def test_settings_fingerprint_is_order_independent(self):
        left = decomposition_settings({"ncomp": 2, "type_tf": "gaussian"})
        right = decomposition_settings({"type_tf": "gaussian", "ncomp": 2})
        self.assertEqual(left, right)

    def test_missing_file_has_no_digest(self):
        self.assertIsNone(file_digest(ROOT / "this_file_does_not_exist"))


class ExistingDecompositionValidationTests(unittest.TestCase):
    def test_existing_mrk142_decomposition_is_valid(self):
        path = (
            ROOT
            / "Mrk142/runs/mica_gaussian/2comp"
            / "run_W2_to_M2_2comp_gaussian/data/pline.txt_2_comp0"
        )
        if not path.is_file():
            self.skipTest("Mrk142 MICA decomposition is not available")
        n_driver, n_response, n_total = validate_decomposition_file(path)
        self.assertEqual(n_driver, 200)
        self.assertEqual(n_response, 200)
        self.assertEqual(n_total, 400)

    def test_existing_conditional_run_is_complete_when_available(self):
        target = (
            ROOT
            / "Mrk142/runs/mica_gaussian/2comp/result/branches/q0_lt_3"
            / "decomposition/run_W2_to_M2_2comp_gaussian"
        )
        if not (target / "data/fig_line_decomp_2.pdf").is_file():
            self.skipTest("conditional Mrk142 decomposition is not available")
        result = validate_run_outputs(target, 2, require_plot=True)
        self.assertEqual(result["total_rows"], 400)
        self.assertEqual(result["comp0_rows"], 400)
        self.assertEqual(result["comp1_rows"], 400)


if __name__ == "__main__":
    unittest.main()
