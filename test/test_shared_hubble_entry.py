"""Integration boundaries for the frozen-distance comparison entry."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import pandas as pd
import yaml
from src.shared_sed_hubble.config import load_comparison_config
from src.shared_sed_hubble.comparison import assemble_inputs


class ComparisonEntryTests(unittest.TestCase):
    def make_config(self, root, sources=("NGC4151_a",)):
        path = root / "comparison.yaml"
        path.write_text(yaml.safe_dump({
            "sources": list(sources), "distance_summary": "frozen/distance.csv",
            "redshift_inputs": "external/redshifts.csv", "output_dir": "results",
        }))
        return path

    def test_paths_relative_to_explicit_root_and_frozen_input_guard(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_config(root)
            config = load_comparison_config(path, root=root)
            self.assertEqual(config.distance_summary, root / "frozen/distance.csv")
            self.assertEqual(config.output_dir, root / "results")
            with self.assertRaisesRegex(ValueError, "must not contain"):
                load_comparison_config(path, root=root, output=root / "frozen")
            with self.assertRaisesRegex(ValueError, "must not contain"):
                load_comparison_config(path, root=root, output=root)

    def test_repeated_epochs_are_not_independent_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_config(root, ("NGC4151_a", "NGC4151_b"))
            with self.assertRaisesRegex(ValueError, "distinct AGN"):
                load_comparison_config(path, root=root)

    def test_saved_redshift_is_retained_and_source_order_is_exact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_config(root)
            config = load_comparison_config(path, root=root)
            config.distance_summary.parent.mkdir()
            pd.DataFrame([{"source": "NGC4151_a", "z": .0032,
                           "D_mean": 32.509, "D_std": 13.051}]).to_csv(
                               config.distance_summary, index=False)
            external = pd.DataFrame([{"source": "NGC4151_a", "z_cmb": .00415}])
            with patch("src.shared_sed_hubble.redshift.load_redshift_inputs",
                       return_value=external):
                merged = assemble_inputs(config)
            self.assertEqual(merged.source.tolist(), ["NGC4151_a"])
            self.assertEqual(merged.z_saved.iloc[0], .0032)
            self.assertEqual(merged.z_cmb.iloc[0], .00415)
            self.assertEqual(merged.D_std.iloc[0], 13.051)
            self.assertNotIn("z", merged)

    def test_missing_or_duplicate_distance_source_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_config(root)
            config = load_comparison_config(path, root=root)
            config.distance_summary.parent.mkdir()
            for rows, message in (
                ([{"source": "Other", "z": .01, "D_mean": 40., "D_std": 4.}], "missing sources"),
                ([{"source": "NGC4151_a", "z": .0032, "D_mean": 32., "D_std": 13.}]*2,
                 "duplicate source"),
            ):
                pd.DataFrame(rows).to_csv(config.distance_summary, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    assemble_inputs(config)


if __name__ == "__main__":
    unittest.main()
