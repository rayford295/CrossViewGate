from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from crossview_conflict.data.cvian import (
    build_cvi_an_image_id_map,
    georeference_cvi_an_pairs,
    load_cvi_an_positions,
    local_position_geojson,
)


class CVIANGeoreferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "images").mkdir()
        self.pairs = pd.DataFrame(
            [
                {
                    "sat_path": "/content/images/000000_sat.png",
                    "svi_path": "/content/images/000000_svi.png",
                    "severity": "0_MinorDamage",
                },
                {
                    "sat_path": "/content/images/000001_sat.png",
                    "svi_path": "/content/images/000001_svi.png",
                    "severity": "1_ModerateDamage",
                },
            ]
        )
        records = [
            ("000000", "111", "0_MinorDamage", -82.1, 26.4, "sequence-a"),
            ("000001", "222", "1_ModerateDamage", -82.0, 26.5, "sequence-b"),
        ]
        checksum_lines: list[str] = []
        features: list[dict[str, object]] = []
        for sample_id, mapillary_id, severity, lon, lat, sequence_id in records:
            for view, official_dir in (("sat", "01_Satellite"), ("svi", "00_SVI")):
                payload = f"{sample_id}-{view}".encode()
                local = self.root / "images" / f"{sample_id}_{view}.png"
                local.write_bytes(payload)
                checksum = hashlib.sha512(payload).hexdigest()
                checksum_lines.append(
                    f"{checksum}  ./CVIAN/{official_dir}/{severity}/{mapillary_id}.png"
                )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "id": mapillary_id,
                        "captured_a": 1660000000000,
                        "compass_an": 90.0,
                        "creator_id": 123.0,
                        "is_pano": 1,
                        "sequence_i": sequence_id,
                        "lon": lon,
                        "lat": lat,
                    },
                }
            )
        self.checksums = self.root / "checksums.sha512"
        self.checksums.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
        self.geojson = self.root / "positions.geojson"
        self.geojson.write_text(
            json.dumps({"type": "FeatureCollection", "features": list(reversed(features))}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_exact_hash_mapping_does_not_depend_on_feature_order(self) -> None:
        image_map = build_cvi_an_image_id_map(self.root, self.pairs, self.checksums)
        self.assertEqual(image_map["mapillary_id"].tolist(), ["111", "222"])
        positions = load_cvi_an_positions(self.geojson)
        enriched = georeference_cvi_an_pairs(self.pairs, image_map, positions, 0.005)
        self.assertEqual(enriched["latitude"].tolist(), [26.4, 26.5])
        self.assertEqual(enriched["sequence_id"].tolist(), ["sequence-a", "sequence-b"])
        self.assertTrue(enriched["spatial_block_id"].notna().all())

        output = local_position_geojson(enriched)
        self.assertEqual(len(output["features"]), 2)
        self.assertEqual(
            output["features"][0]["geometry"]["coordinates"],
            [-82.1, 26.4],
        )
        self.assertEqual(output["features"][0]["properties"]["sample_id"], "000000")

    def test_pair_views_must_map_to_the_same_official_id(self) -> None:
        text = self.checksums.read_text(encoding="utf-8").replace(
            "/00_SVI/0_MinorDamage/111.png",
            "/00_SVI/0_MinorDamage/999.png",
        )
        self.checksums.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source ids disagree"):
            build_cvi_an_image_id_map(self.root, self.pairs, self.checksums)


if __name__ == "__main__":
    unittest.main()
