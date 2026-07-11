from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.build_milton_hurricane_manifests import _add_manifest_columns


class MiltonManifestBuilderTest(unittest.TestCase):
    def test_source_val_rows_and_optional_pre_view_are_representable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for pair_id, split_name in enumerate(("train", "val", "test")):
                damage = ("mild_damage", "moderate_damage", "severe_damage")[pair_id]
                (root / damage / "post").mkdir(parents=True)
                (root / damage / "pre").mkdir(parents=True)
                (root / "post_sat").mkdir(exist_ok=True)
                Image.new("RGB", (4, 4)).save(root / damage / "post" / f"{pair_id}.png")
                Image.new("RGB", (4, 4)).save(root / "post_sat" / f"{pair_id}.png")
                if split_name != "val":
                    Image.new("RGB", (4, 4)).save(root / damage / "pre" / f"{pair_id}.png")
                rows.append(
                    {
                        "pair_id": pair_id,
                        "damage_level": damage,
                        "prompt": damage,
                        "pre_disaster_image_path": f"/source/{pair_id}.png",
                        "post_disaster_image_path": f"/source/{pair_id}.png",
                        "post_sat_image_path": f"/source/{pair_id}.png",
                        "set": split_name,
                        "lat": 29.0 + pair_id,
                        "lon": -83.0,
                    }
                )

            manifest = _add_manifest_columns(pd.DataFrame(rows), root, verify_images=True)
            self.assertEqual(len(manifest), 3)
            self.assertEqual(set(manifest["split_source"]), {"train", "val", "test"})
            val = manifest.loc[manifest["split_source"] == "val"].iloc[0]
            self.assertFalse(bool(val["pre_street_view_available"]))
            self.assertTrue(bool(val["post_street_view_available"]))
            self.assertTrue(bool(val["post_overhead_view_available"]))

    def test_missing_required_post_view_fails_instead_of_silent_drop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = pd.DataFrame(
                [
                    {
                        "pair_id": 7,
                        "damage_level": "mild_damage",
                        "prompt": "mild",
                        "pre_disaster_image_path": "/source/7.png",
                        "post_disaster_image_path": "/source/7.png",
                        "post_sat_image_path": "/source/7.png",
                        "set": "train",
                        "lat": 29.0,
                        "lon": -83.0,
                    }
                ]
            )
            with self.assertRaises(FileNotFoundError):
                _add_manifest_columns(row, root, verify_images=True)


if __name__ == "__main__":
    unittest.main()
