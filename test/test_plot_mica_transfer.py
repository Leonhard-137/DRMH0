"""Single-band transfer-figure regressions; no MICA/FFA runs."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import plot_mica_transfer as transfer


def make_band(name="U", kind="gaussian"):
    lag = np.linspace(-10, 100, 1001)
    disk = transfer.kernel(lag, [1.0], [0.3], [0.5], kind)[0]
    total = disk + transfer.kernel(lag, [0.2], [15.0], [4.0], "gaussian")[0]
    def curve(median):
        return np.column_stack((lag, median * 0.5, median, median * 1.5))
    return transfer.BandTransfer(name, 3465, kind, 100, 0, Path("posterior"),
                                 Path("native_total"), curve(disk), curve(total))


class TransferPlotTests(unittest.TestCase):
    def test_median_parameter_curves_insert_tau_and_width_into_each_kernel(self):
        amplitudes = np.array([[1.0, 3.0, 2.0], [0.2, 0.4, 0.3]])
        centroids = np.array([[4.0, 6.0, 5.0], [20.0, 24.0, 22.0]])
        widths = np.array([[1.0, 3.0, 2.0], [4.0, 6.0, 5.0]])
        parameters = transfer.median_kernels(
            amplitudes, centroids, widths, ["gamma", "gaussian"],
        )
        self.assertEqual(len(parameters), 2)
        self.assertEqual(parameters[0].amplitude, 2.0)
        self.assertEqual(parameters[0].centroid, 5.0)
        self.assertEqual(parameters[0].width, 2.0)
        # Gamma's analytic shift is its onset, not its physical centroid.
        self.assertEqual(parameters[0].center, 1.0)
        self.assertEqual(parameters[1].center, 22.0)

        comp0, total = transfer.median_parameter_curves(parameters)
        np.testing.assert_array_equal(comp0[:, 0], total[:, 0])
        lag = comp0[:, 0]
        expected0 = transfer.kernel(lag, [2.0], [1.0], [2.0], "gamma")[0]
        expected1 = transfer.kernel(lag, [0.3], [22.0], [5.0], "gaussian")[0]
        np.testing.assert_allclose(comp0[:, 2], expected0, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(total[:, 2], expected0 + expected1, rtol=1e-13, atol=0.0)
        np.testing.assert_array_equal(comp0[:, 1], comp0[:, 2])
        np.testing.assert_array_equal(comp0[:, 2], comp0[:, 3])

    def test_legacy_gamma_alias_reuses_free_metadata_with_saved_output_root(self):
        cfg = {
            "mica_gamma2_free": {
                "output_root": "runs/mica/gamma/free",
                "driver": "W2",
                "responses": ["U"],
                "ncomp": 2,
                "type_tf": "gamma",
                "component_order": "moment1",
            }
        }
        fit = transfer.transfer_stage_config(cfg, transfer.LEGACY_GAMMA_STAGE)
        self.assertEqual(fit["output_root"], "runs/mica/gamma2")
        self.assertEqual(fit["component_order"], "moment1")
        self.assertEqual(cfg["mica_gamma2_free"]["output_root"], "runs/mica/gamma/free")

    def test_y_limits_match_approved_example_and_are_paired(self):
        band = make_band()
        band.comp0[:, 3] = 1.5362630669524653
        band.total[:, 3] = 1.5211242
        self.assertEqual(transfer.amplitude_limits(band), (1e-6, 3.0))

    def test_lag_window_retains_negative_support_and_all_visible_peaks(self):
        lag = np.linspace(-30, 100, 1301)
        upper = np.exp(-0.5 * ((lag + 5) / 0.8) ** 2)
        upper += 0.01 * np.exp(-0.5 * ((lag - 60) / 2) ** 2)
        curve = np.column_stack((lag, upper * 0.5, upper * 0.75, upper))
        before = curve.copy()
        low, high = transfer.lag_limits(curve, 1e-6)
        visible_lag = lag[upper > 1e-6]
        self.assertLess(low, 0)
        self.assertLessEqual(low, visible_lag.min())
        self.assertGreaterEqual(high, visible_lag.max())
        self.assertTrue(np.all(curve[(lag < low) | (lag > high), 3] <= 1e-6))
        np.testing.assert_array_equal(curve, before)

    def test_native_edges_and_no_visible_signal_keep_grid_bounds(self):
        curve = np.column_stack((np.linspace(0, 100, 11), np.ones((11, 3))))
        self.assertEqual(transfer.lag_limits(curve, 1e-6), (0, 100))
        curve[:, 1:] = 0
        self.assertEqual(transfer.lag_limits(curve, 1e-6), (0, 100))

    def test_each_band_gets_two_monochrome_png_pdf_pairs_without_data_changes(self):
        bands = [make_band("U"), make_band("B", "gamma")]
        # Native rounding may yield a negative interval lower bound.
        bands[0].total[0, 1] = -1e-12
        original = [(b.comp0.copy(), b.total.copy()) for b in bands]
        views = []
        original_savefig = Figure.savefig

        def savefig(fig, path, **kwargs):
            path = Path(path)
            if path.stem.endswith("_transfer_functions_overview"):
                self.assertEqual(len(fig.axes), 2 * len(bands))
                np.testing.assert_allclose(
                    fig.get_size_inches(), np.array([180, 235]) / 25.4,
                )
                for row, band in enumerate(bands):
                    left, right = fig.axes[2 * row:2 * row + 2]
                    self.assertEqual(left.get_title(loc="left"), band.band)
                    self.assertEqual(right.get_title(loc="left"), band.band)
                    self.assertEqual(left.get_title(loc="right"), "comp0")
                    self.assertEqual(right.get_title(loc="right"), "total")
                    self.assertEqual(left.get_ylim(), right.get_ylim())
                    for ax in (left, right):
                        self.assertEqual(ax.get_xscale(), "linear")
                        self.assertEqual(ax.get_yscale(), "log")
                        self.assertIsNone(ax.get_legend())
            else:
                ax = fig.axes[0]
                self.assertEqual(len(fig.axes), 1)
                np.testing.assert_allclose(
                    fig.get_size_inches(), np.array([89, 67]) / 25.4,
                )
                self.assertEqual(ax.get_xlabel(), r"$\tau$ (day)")
                self.assertNotIn("native", ax.get_ylabel())
                self.assertIn(ax.get_title(loc="right"), ("comp0", "total"))
                self.assertEqual(ax.get_xscale(), "linear")
                self.assertEqual(ax.get_yscale(), "log")
                self.assertIsNone(ax.get_legend())
                views.append((ax.get_title(loc="left"), ax.get_ylim()))
            original_savefig(fig, path, **kwargs)

        with tempfile.TemporaryDirectory(prefix="mica-transfer-test-") as directory:
            output = Path(directory)
            with mock.patch.object(Figure, "savefig", savefig):
                transfer.plot_stage("Synthetic", "mica_mix220", {"driver": "W2"}, bands, output)
            expected = {f"Synthetic_{b.band}_{c}_transfer.{ext}" for b in bands
                        for c in ("comp0", "total") for ext in ("png", "pdf")}
            expected |= {f"Synthetic_transfer_functions_overview.{ext}"
                         for ext in ("png", "pdf")}
            self.assertEqual({p.name for p in output.iterdir()}, expected)
            for path in output.glob("*.png"):
                with Image.open(path) as img:
                    expected_size = ((2125, 2775)
                                     if path.stem.endswith("_transfer_functions_overview")
                                     else (1051, 791))
                    self.assertEqual(img.size, expected_size)
                    pixels = np.asarray(img.convert("RGB"))
                np.testing.assert_array_equal(pixels[:, :, 0], pixels[:, :, 1])
                np.testing.assert_array_equal(pixels[:, :, 1], pixels[:, :, 2])
            for path in output.glob("*.pdf"):
                self.assertTrue(path.read_bytes().startswith(b"%PDF"))
        for title in {view[0] for view in views}:
            self.assertEqual(len({view[1] for view in views if view[0] == title}), 1)
        for band, before in zip(bands, original):
            np.testing.assert_array_equal(band.comp0, before[0])
            np.testing.assert_array_equal(band.total, before[1])

    def test_median_parameter_plot_uses_new_suffix_and_preserves_old_plot(self):
        band = make_band("U", "gamma")
        with tempfile.TemporaryDirectory(prefix="mica-transfer-median-test-") as directory:
            output = Path(directory)
            old_plot = output / "Synthetic_U_comp0_transfer.png"
            old_plot.write_bytes(b"old plot stays")
            transfer.plot_stage(
                "Synthetic", "mica_gamma2_free", {"driver": "W2"}, [band], output,
                show_interval=False, filename_suffix="_median_parameters",
            )
            self.assertEqual(old_plot.read_bytes(), b"old plot stays")
            expected = {
                f"Synthetic_U_{component}_transfer_median_parameters.{extension}"
                for component in ("comp0", "total")
                for extension in ("png", "pdf")
            }
            expected |= {
                f"Synthetic_transfer_functions_overview_median_parameters.{extension}"
                for extension in ("png", "pdf")
            }
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))


if __name__ == "__main__":
    unittest.main()
