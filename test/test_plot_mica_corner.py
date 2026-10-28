"""Scientific data-path checks for corners; no inference or real-run writes."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import plot_mica_corner as corner


def write_native_run(run, *, location="center"):
    (run / "data").mkdir(parents=True)
    (run / "param").mkdir()
    (run / "param/param_input").write_text(
        "TypeModel 0\nTypeTF 2\nNumCompLow 2\nNumCompUpp 2\nFlagNegativeResp 0\n")
    names = ["sys_err_con", "sigmad", "taud", "sys_err_line"]
    for i in range(2):
        names.extend((f"{i}-th_component_amplitude", f"{i}-th_component_{location}",
                      f"{i}-th_component_sigma"))
    (run / "data/para_names_line.txt_2").write_text(
        "\n".join(f"{i} {name}" for i, name in enumerate(names)) + "\n")
    # Gamma centroids 8 and 3 in row 0 must swap as whole parameter groups.
    samples = np.array([
        [0, -1.3, 4.2, 0, np.log(2), 4, np.log(2), np.log(5), 1, np.log(1)],
        [0, -0.7, 5.6, 0, np.log(3), 1, np.log(1), np.log(7), 3, np.log(2)],
    ], dtype=float)
    if location == "centroid":
        samples[:, 5] += 2 * np.exp(samples[:, 6])
        samples[:, 8] += 2 * np.exp(samples[:, 9])
    # Preserve duplicate posterior rows: resampling multiplicity is probability.
    samples = np.vstack((samples, samples[0]))
    np.savetxt(run / "data/posterior_sample1d.txt_2", samples)
    # A deliberately unusable raw-weight file must never be read by this plotter.
    (run / "data/weights.txt").write_text("do not read raw weights\n")
    return samples


class CornerTests(unittest.TestCase):
    def test_atomic_save_keeps_previous_file_after_failed_render(self):
        class BrokenFigure:
            @staticmethod
            def savefig(path, **_kwargs):
                Path(path).write_bytes(b"partial")
                raise RuntimeError("render failed")

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "corner.pdf"
            destination.write_bytes(b"previous complete file")
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                corner.save_figure_atomic(BrokenFigure(), destination)
            self.assertEqual(destination.read_bytes(), b"previous complete file")
            self.assertEqual(list(destination.parent.glob(".corner.*.pdf")), [])

    def test_gamma_sorts_whole_groups_and_keeps_drw_rows_and_native_logs(self):
        # Sorting is required for Gamma even if the stage requests native labels.
        fit = {"ncomp": 2, "type_tf": "gamma", "component_order": "native"}
        with tempfile.TemporaryDirectory() as directory:
            for location in ("center", "centroid"):
                run = Path(directory) / location
                original = write_native_run(run, location=location)
                expected = original[:, [1, 2, 4, 5, 6, 7, 8, 9]].copy()
                expected[[0, 2], 2:5] = original[[0, 2], 7:10]
                expected[[0, 2], 5:8] = original[[0, 2], 4:7]
                values, _, _, info = corner.read_corner_samples(run, fit)
                np.testing.assert_array_equal(values, expected)
                self.assertEqual(info["nposterior"], 3)
                self.assertEqual(info["nunique_joint_draws"], 2)
                self.assertEqual(info["nswapped"], 2)
                self.assertEqual(info["component_order"], "moment1")
                self.assertEqual(info["parameter_names"][:2], ["ln_sigmad", "ln_taud"])

    def test_gaussian_retains_native_component_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            original = write_native_run(run, location="centroid")
            p = run / "param/param_input"
            p.write_text(p.read_text().replace("TypeTF 2", "TypeTF 0"))
            values, _, _, info = corner.read_corner_samples(run, {"ncomp": 2, "type_tf": "gaussian"})
            np.testing.assert_array_equal(values, original[:, [1, 2, 4, 5, 6, 7, 8, 9]])
            self.assertEqual(info["nswapped"], 0)

    def test_ranges_include_extreme_draws_and_handle_constants(self):
        values = np.array([[0, 2], [1, 2], [10000, 2]], dtype=float)
        bounds = corner.sample_ranges(values)
        for i, (low, high) in enumerate(bounds):
            self.assertLess(low, values[:, i].min())
            self.assertGreater(high, values[:, i].max())

    def test_contours_use_enclosed_mass_not_one_dimensional_sigma(self):
        counts = np.array([[40, 30], [20, 10]], dtype=float)
        np.testing.assert_array_equal(corner.contour_thresholds(counts), [10, 30])
        self.assertEqual(len(corner.contour_thresholds(np.array([[100.0]]))), 0)

    def test_sparse_or_repeated_draws_remain_visible_without_contours(self):
        samples = np.tile([[0.0, 2.0], [1.0, 2.0]], (80, 1))
        original = samples.copy()
        fig, stats = corner.make_corner(samples, ["x", "y"], ["$x$", "$y$"])
        try:
            self.assertEqual(stats["lower_triangle"], "scatter_only")
            self.assertEqual(len(fig.axes[1].collections), 1)
            self.assertEqual(len(fig.axes[1].collections[0].get_offsets()), len(samples))
            np.testing.assert_allclose(stats["quantiles"], np.quantile(samples, corner.QUANTILES, axis=0).T)
            np.testing.assert_array_equal(samples, original)
            self.assertEqual(fig.texts, [])
        finally:
            corner.plt.close(fig)

    def test_existing_posterior_writes_only_pdf_png_in_result_corner(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "AGN"
            (source / "config").mkdir(parents=True)
            (source / "config/source_config.yaml").write_text(
                "source: AGN\nmica_stages: [mica_gamma2]\nmica_gamma2:\n"
                "  output_root: runs/mica/gamma/test\n  type_tf: gamma\n  ncomp: 2\n"
                "  driver: W2\n  responses: [U]\n  component_order: moment1\n")
            run = source / "runs/mica/gamma/test/2comp/run_W2_to_U_2comp_gamma"
            write_native_run(run)
            original = (run / "data/posterior_sample1d.txt_2").read_bytes()
            records = corner.plot_source_stage(source, "mica_gamma2")
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(Path(record["pdf"]).parent, run.parent / "result" / "corner")
            self.assertEqual(record["selection"], "all")
            self.assertEqual(record["weights"], "equal")
            self.assertTrue(Path(record["pdf"]).read_bytes().startswith(b"%PDF"))
            with Image.open(record["png"]) as image:
                self.assertGreaterEqual(image.info["dpi"][0], 299.9)
            self.assertFalse(Path(record["pdf"]).with_suffix(".json").exists())
            self.assertEqual(original, (run / "data/posterior_sample1d.txt_2").read_bytes())

    def test_single_constant_dimension_has_no_artificial_density_spread(self):
        samples = np.column_stack((np.linspace(0, 1, 150), np.full(150, 2.0)))
        fig, stats = corner.make_corner(samples, ["x", "y"], ["$x$", "$y$"])
        try:
            self.assertEqual(stats["constant_dimensions"], [1])
            self.assertEqual(len(fig.axes[1].collections), 1)
            points = fig.axes[1].collections[0].get_offsets()
            np.testing.assert_array_equal(points[:, 1], np.full(150, 2.0))
        finally:
            corner.plt.close(fig)

    def test_dense_panels_keep_every_draw_and_raw_quantiles(self):
        rng = np.random.default_rng(37)
        original = rng.normal(size=(600, 2))
        samples = np.vstack((original, original[:50]))
        before = samples.copy()
        fig, stats = corner.make_corner(samples, ["x", "y"], ["x", "y"])
        try:
            self.assertGreater(len(fig.axes[1].collections), 1)
            points = fig.axes[1].collections[0].get_offsets()
            np.testing.assert_array_equal(points, samples)
            np.testing.assert_array_equal(samples, before)
            np.testing.assert_allclose(stats["quantiles"], np.quantile(samples, corner.QUANTILES, axis=0).T)
            self.assertEqual(stats["density_omitted_dimensions"], [])
            self.assertEqual(fig.texts, [])
        finally:
            corner.plt.close(fig)

    def test_smoothing_preserves_mass_and_peak_at_hard_wall(self):
        x = np.arange(40.0)
        counts = np.exp(-x[:, None] / 2) * np.exp(-0.5 * ((x[None, :] - 20) / 4)**2)
        density = corner.smooth_histogram(counts, [1, 1], [(True, False), (False, False)])
        self.assertTrue(np.isfinite(density).all())
        self.assertTrue((density >= 0).all())
        self.assertAlmostEqual(density.sum(), counts.sum())
        self.assertEqual(density.sum(axis=1).argmax(), 0)
        # Left/right boundary handling must be symmetric under axis reversal.
        flipped = corner.smooth_histogram(counts[::-1], [1, 1], [(False, True), (False, False)])
        np.testing.assert_allclose(density, flipped[::-1])
        unsmoothed = corner.smooth_histogram(counts, [0, 0], [(True, False), (False, False)])
        np.testing.assert_array_equal(unsmoothed, counts)

    def test_ordered_density_does_not_leak_into_forbidden_triangle(self):
        allowed = np.arange(20)[:, None] <= np.arange(20)[None, :]
        counts = np.eye(20)
        density = corner.smooth_histogram(counts, [1, 1], [(False, False)]*2, allowed)
        self.assertTrue((density[~allowed] == 0).all())
        self.assertAlmostEqual(density.sum(), counts.sum())

    def test_saved_prior_bounds_preserve_drw_and_follow_relabelled_components(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            write_native_run(run, location="centroid")
            p = run / "data/para_names_line.txt_2"
            rows = []
            for line in p.read_text().splitlines():
                index, name = line.split()
                limits = (0, 0) if name in ("sigmad", "taud") else (-5, 5)
                if name.endswith("centroid"):
                    limits = (0, 10) if name.startswith("0-") else (-2, 20)
                rows.append(f"{index} {name} UNI {limits[0]} {limits[1]} 0 0")
            p.write_text("\n".join(rows) + "\n")
            _, _, _, info = corner.read_corner_samples(run, {"ncomp": 2, "type_tf": "gamma"})
            self.assertEqual(info["prior_bounds"][:2], [None, None])
            self.assertEqual(info["prior_bounds"][3], (-2, 20))
            self.assertEqual(info["prior_bounds"][6], (-2, 20))
            self.assertEqual(info["ordered_pairs"], [(3, 6)])
            # Match the shared reader's normalization of an explicit label rule.
            native = run / "param/param_input"
            native.write_text(native.read_text().replace("TypeTF 2", "TypeTF 0"))
            _, _, _, info = corner.read_corner_samples(
                run, {"ncomp": 2, "type_tf": "gaussian", "component_order": " MOMENT1 "})
            self.assertEqual(info["prior_bounds"][3], (-2, 20))
            self.assertEqual(info["prior_bounds"][6], (-2, 20))
            self.assertEqual(info["ordered_pairs"], [(3, 6)])

    def test_full_plot_respects_hard_bounds_without_dropping_draws(self):
        rng = np.random.default_rng(9)
        samples = np.column_stack((rng.exponential(.3, 800), rng.normal(size=800)))
        samples[0, 0] = 0
        fig, stats = corner.make_corner(samples, ["x", "y"], ["x", "y"], prior_bounds=[(0, 10), None])
        try:
            self.assertEqual(fig.axes[1].get_xlim()[0], 0)
            mesh = fig.axes[1].collections[1]
            self.assertGreaterEqual(mesh.get_coordinates()[..., 0].min(), 0)
            np.testing.assert_array_equal(fig.axes[1].collections[0].get_offsets(), samples)
            self.assertGreaterEqual(stats["ranges"][0][1], samples[:, 0].max())
        finally:
            corner.plt.close(fig)

    def test_long_tail_unresolved_core_keeps_scatter_and_caps_bandwidth(self):
        rng = np.random.default_rng(19)
        samples = rng.normal(size=(400, 2))
        samples[0, 0] = 1000
        fig, stats = corner.make_corner(samples, ["x", "y"], ["x", "y"])
        try:
            self.assertIn(0, stats["density_omitted_dimensions"])
            self.assertEqual(len(fig.axes[1].collections), 1)
            self.assertGreater(stats["ranges"][0][1], 1000)
            core = np.diff(np.quantile(samples[:, 0], [corner.QUANTILES[0], corner.QUANTILES[2]]))[0]
            self.assertLessEqual(stats["smoothing_bandwidths"][0], .25 * core + 1e-12)
            np.testing.assert_array_equal(fig.axes[1].collections[0].get_offsets(), samples)
        finally:
            corner.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
