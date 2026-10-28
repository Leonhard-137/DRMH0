"""Paired-ablation fit/plot contracts; no MICA sampling or image files."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent / "mica_recovery/longterm_test"
sys.path.insert(0, str(HERE))
import fit
import plot_inputs
import plot_results
import prepare


class PairedAblationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = prepare.load_config(HERE / "experiment_ablation.yaml")
        self.path = self.root / "experiment.yaml"
        self.path.write_text(yaml.safe_dump(self.config))
        anchor = prepare.load_config(HERE / "experiment_one_shared_blr.yaml")
        self.parameters = next(iter(prepare.real_disk_mismatch_cases(anchor).values()))
        self.saved = {}
        for name, switches in self.config["paired_ablation"]["cases"].items():
            case = self.root / "cases" / name
            (case / "input").mkdir(parents=True)
            (case / "components").mkdir()
            p = {**self.parameters, **switches}
            if not switches["blr"]:
                p["response_blr_gain"] = 0.0
            self.saved[name] = (case, p, {})
            for band in ("UVW2", "U"):
                t = np.arange(4.0)
                disk = np.array([.01, -.02, .03, -.02])
                blr = .25 * disk if switches["blr"] and band == "U" else np.zeros(4)
                shared_on = switches["shared_driver" if band == "UVW2" else "shared_response"]
                shared = .5 * disk if shared_on else np.zeros(4)
                noise = np.array([.001, -.002, .001, .002])
                values = np.column_stack((t, disk + blr, shared, np.zeros(4), noise, disk, blr))
                curve = np.column_stack((t, 1 + values[:, 1:5].sum(axis=1), np.full(4, .01)))
                np.savetxt(case / "input" / f"LTSIM_{band}.txt", curve)
                header = ("time illumination_variation shared_long_variation shared_short_variation "
                          "measurement_noise disk_illumination_variation blr_illumination_variation\n")
                header += "; ".join(f"{k}={int(v) if isinstance(v, bool) else v}" for k, v in p.items())
                np.savetxt(case / "components" / f"{band}_components.txt", values, header=header)
                self.saved[name][2][band] = curve

    def test_exactly_ten_jobs_and_reuse_never_refitted(self):
        jobs = fit.build_plan(self.path, suite="paired_ablation")
        self.assertEqual(len(jobs), 10)
        self.assertEqual(len({job["run_dir"] for job in jobs}), 10)
        self.assertEqual({job["case"].name for job in jobs}, set(self.saved))
        for job in jobs:
            settings = job["fit"]
            self.assertEqual(settings["lag_prior"], [[0.0, 100.0], [0.0, 100.0]])
            np.testing.assert_allclose(settings["width_prior"], [[.14195358, 152.84]] * 2)
            self.assertEqual((settings["num_particles"], settings["ptol"], settings["mpi_per_band"]), (1, .01, 2))
            self.assertFalse(settings["fit_only"])
            self.assertTrue(settings["plot_decomp"])
        with self.assertRaisesRegex(ValueError, "Unknown cases"):
            fit.build_plan(self.path, cases=["abl_disk_blr_shared"])

    def test_diagnostic_subparts_are_not_added_twice(self):
        case, p, curves = self.saved["abl_disk_blr_uonly"]
        values = plot_inputs.load_components(case, "U", curves["U"], real_case=p)
        run = dict(case=case, bands=("UVW2", "U"), observed=[curves["UVW2"], curves["U"]])
        truth = plot_results.truth_components(run, 1)
        np.testing.assert_allclose(truth["total"], curves["U"][:, 1] - values[:, 4])
        self.assertEqual(truth["values"].shape[1], 3)
        self.assertIsNone(plot_results.input_reference(self.config, run))
        path = case / "components/U_components.txt"
        lines = path.read_text().splitlines()
        values[:, 6] += .1
        np.savetxt(path, values, header="\n".join(line[2:] for line in lines if line.startswith("# ")))
        with self.assertRaises(AssertionError):
            plot_results.truth_components(run, 1)

    def test_u_only_signal_labels_and_disabled_components(self):
        case, p, _ = self.saved["abl_disk_uonly"]
        with patch.object(plot_inputs, "save_figure") as save, redirect_stdout(io.StringIO()):
            plot_inputs.plot_case(case, "LTSIM", ("UVW2", "U"), real_case=p)
        fig = save.call_args.args[0]
        self.addCleanup(plot_inputs.plt.close, fig)
        driver_labels = fig.axes[0].get_legend_handles_labels()[1]
        response_labels = fig.axes[1].get_legend_handles_labels()[1]
        self.assertFalse(any("process" in label for label in driver_labels))
        self.assertIn("Independent U process (+1)", response_labels)
        self.assertFalse(any("short" in label.lower() for label in response_labels))
        self.assertIn(r"$aD_U*I_0$ (+1)", response_labels)

    def test_input_kernel_is_physical_and_omits_disabled_blr(self):
        _, p, _ = self.saved["abl_disk_only"]
        lag = np.array([.5, 1.5, 2.5])
        kernel = np.array([.2, .5, .087])
        stub = types.SimpleNamespace(ablation_cases=lambda path, cfg: {"abl_disk_only": p})
        with patch.dict(sys.modules, prepare_ablation=stub), patch.object(
            plot_inputs, "real_disk_mismatch_input_kernels",
            return_value=(lag, {"disk": kernel, "blr": np.zeros(3), "total": kernel}),
        ):
            _, curves = plot_inputs.build_transfer_functions(self.config, self.path)
        definition = curves["abl_disk_only"]
        self.assertNotIn("blr", definition)
        self.assertIn("Physical", definition["labels"]["total"])
        self.assertIn("not a total W2-to-U", definition["title"])


if __name__ == "__main__":
    unittest.main()
