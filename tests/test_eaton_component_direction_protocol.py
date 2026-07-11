from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image

from scripts.build_eaton_component_direction_protocol import (
    ROLE_NAMES,
    assign_grouped_roles,
    build_source_cohort,
    build_spatial_protocol,
    load_protocol,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "eaton_component_direction_v1.json"


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color=(120, 80, 40))
    for x in range(max(1, size[0] // 4)):
        for y in range(max(1, size[1] // 4)):
            image.putpixel((x, y), (240, 220, 200))
    image.save(path)


class EatonComponentDirectionProtocolTest(unittest.TestCase):
    def test_frozen_protocol_has_one_question_and_four_roles(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        self.assertEqual(protocol["protocol_version"], "eaton-component-direction-v1")
        self.assertEqual(
            tuple(item["name"] for item in protocol["spatial_protocol"]["roles"]),
            ROLE_NAMES,
        )
        self.assertIn("signed disagreement direction", protocol["research_question"])
        self.assertTrue(
            protocol["confirmation_rule"]["supportive_estimands_cannot_rescue_primary"]
        )

    def test_media_only_canonicalization_chooses_largest_readable_street_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_image(root / "dataset" / "a" / "street.jpg", (64, 64))
            _write_image(root / "dataset" / "a" / "remote.jpg", (32, 32))
            _write_image(root / "dataset" / "b" / "street.jpg", (128, 64))
            _write_image(root / "dataset" / "b" / "remote.jpg", (32, 32))
            _write_image(root / "dataset" / "c" / "street.jpg", (64, 64))
            joined = pd.DataFrame(
                [
                    {
                        "pair_id": "a",
                        "street_view_relative_path": r"dataset\a\street.jpg",
                        "remote_sensing_relative_path": r"dataset\a\remote.jpg",
                        "category": "Destroyed (>50%)",
                        "attachment_id": 2,
                        "latitude": 34.1,
                        "longitude": -118.1,
                        "dins_source_globalid": "structure-1",
                        "remote_tile_filename": "tile-a.tif",
                        "remote_crop_box": "0,0,32,32",
                    },
                    {
                        "pair_id": "b",
                        "street_view_relative_path": r"dataset\b\street.jpg",
                        "remote_sensing_relative_path": r"dataset\b\remote.jpg",
                        "category": "Destroyed (>50%)",
                        "attachment_id": 3,
                        "latitude": 34.1,
                        "longitude": -118.1,
                        "dins_source_globalid": "structure-1",
                        "remote_tile_filename": "tile-a.tif",
                        "remote_crop_box": "64,0,96,32",
                    },
                    {
                        "pair_id": "c",
                        "street_view_relative_path": r"dataset\c\street.jpg",
                        "remote_sensing_relative_path": "",
                        "category": "No Damage",
                        "attachment_id": 4,
                        "latitude": 34.2,
                        "longitude": -118.2,
                        "dins_source_globalid": "structure-2",
                        "remote_tile_filename": "tile-b.tif",
                        "remote_crop_box": "0,0,32,32",
                    },
                ]
            )
            cohort, exclusions, ledger = build_source_cohort(joined, root)

        self.assertEqual(cohort["pair_id"].tolist(), ["b"])
        self.assertEqual(int(cohort.iloc[0]["street_pixel_area"]), 128 * 64)
        self.assertEqual(len(ledger), 1)
        reasons = " ".join(exclusions["exclusion_reason"].astype(str))
        self.assertIn("noncanonical_structure_attachment", reasons)
        self.assertIn("unusable_media", reasons)
        self.assertEqual(len(str(ledger.iloc[0]["street_sha256"])), 64)

    def test_group_assignment_and_buffer_are_role_isolated(self) -> None:
        rows = []
        for block in range(16):
            base_latitude = 33.0 + block * 0.01
            for label in range(3):
                pair_id = f"pair-{block:02d}-{label}"
                rows.append(
                    {
                        "sample_id": pair_id,
                        "pair_id": pair_id,
                        "dependency_group_id": f"structure-{block:02d}-{label}",
                        "label": label,
                        "label_name": str(label),
                        "category": str(label),
                        "latitude": base_latitude,
                        "longitude": -118.0,
                        "street_sha256": f"{block * 3 + label + 1:064x}",
                        "remote_sha256": f"{10_000 + block * 3 + label:064x}",
                        "remote_tile_filename": "shared-tile.tif",
                        "remote_crop_x0": block * 2048 + label * 512,
                        "remote_crop_y0": 0,
                        "remote_crop_x1": block * 2048 + label * 512 + 512,
                        "remote_crop_y1": 512,
                    }
                )
        cohort = pd.DataFrame(rows)
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        roles, exclusions, assignment, audit = build_spatial_protocol(cohort, protocol)

        self.assertEqual(tuple(roles), ROLE_NAMES)
        self.assertTrue(exclusions.empty)
        self.assertEqual(assignment["split_dependency_group_id"].nunique(), 16)
        self.assertTrue((audit["pair_id_overlap"] == 0).all())
        self.assertTrue((audit["dependency_group_overlap"] == 0).all())
        self.assertTrue((audit["spatial_block_overlap"] == 0).all())
        self.assertTrue((audit["street_media_sha_overlap"] == 0).all())
        self.assertTrue((audit["remote_media_sha_overlap"] == 0).all())
        self.assertTrue((audit["overlapping_remote_crop_pairs"] == 0).all())
        self.assertTrue(audit["strictly_beyond_buffer"].all())
        for role in ROLE_NAMES:
            self.assertEqual(set(roles[role]["label"]), {0, 1, 2})

    def test_group_assignment_is_deterministic(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "pair_id": f"p-{block}-{label}",
                    "spatial_block_id": f"b-{block}",
                    "label": label,
                }
                for block in range(12)
                for label in range(3)
            ]
        )
        fractions = (0.5, 0.1, 0.1, 0.3)
        first, first_assignment = assign_grouped_roles(
            frame,
            roles=ROLE_NAMES,
            fractions=fractions,
            group_column="spatial_block_id",
            label_column="label",
            seed=20260711,
        )
        second, second_assignment = assign_grouped_roles(
            frame,
            roles=ROLE_NAMES,
            fractions=fractions,
            group_column="spatial_block_id",
            label_column="label",
            seed=20260711,
        )
        self.assertEqual(
            first_assignment.to_dict(orient="records"),
            second_assignment.to_dict(orient="records"),
        )
        for role in ROLE_NAMES:
            self.assertEqual(first[role]["pair_id"].tolist(), second[role]["pair_id"].tolist())

    def test_constant_media_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_image(root / "dataset" / "a" / "street.jpg", (32, 32))
            constant = root / "dataset" / "a" / "remote.png"
            constant.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color=(0, 0, 0)).save(constant)
            joined = pd.DataFrame(
                [
                    {
                        "pair_id": "a",
                        "street_view_relative_path": r"dataset\a\street.jpg",
                        "remote_sensing_relative_path": r"dataset\a\remote.png",
                        "category": "No Damage",
                        "attachment_id": 1,
                        "latitude": 34.1,
                        "longitude": -118.1,
                        "dins_source_globalid": "structure-a",
                        "remote_tile_filename": "tile-a.tif",
                        "remote_crop_box": "0,0,32,32",
                    }
                ]
            )
            with self.assertRaisesRegex(ValueError, "No model-usable"):
                build_source_cohort(joined, root)

    def test_non_upright_exif_orientation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            street = root / "dataset" / "a" / "street.jpg"
            remote = root / "dataset" / "a" / "remote.jpg"
            _write_image(street, (32, 32))
            _write_image(remote, (32, 32))
            with Image.open(street) as image:
                exif = image.getexif()
                exif[274] = 6
                image.save(street, exif=exif)
            joined = pd.DataFrame(
                [
                    {
                        "pair_id": "a",
                        "street_view_relative_path": r"dataset\a\street.jpg",
                        "remote_sensing_relative_path": r"dataset\a\remote.jpg",
                        "category": "No Damage",
                        "attachment_id": 1,
                        "latitude": 34.1,
                        "longitude": -118.1,
                        "dins_source_globalid": "structure-a",
                        "remote_tile_filename": "tile-a.tif",
                        "remote_crop_box": "0,0,32,32",
                    }
                ]
            )
            with self.assertRaisesRegex(ValueError, "No model-usable"):
                build_source_cohort(joined, root)


if __name__ == "__main__":
    unittest.main()
