from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import pandas as pd

from scripts.build_risk_protocol_manifests import (
    _read_source_manifest,
    build_protocol_roles,
)


class RiskProtocolManifestTests(unittest.TestCase):
    def test_source_reader_preserves_leading_zero_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            path.write_text("sample_id,label\n000033,2\n", encoding="utf-8")
            self.assertEqual(_read_source_manifest(path)["sample_id"].tolist(), ["000033"])

    def test_roles_are_sample_and_group_disjoint(self) -> None:
        validation_rows = []
        for group_index in range(12):
            for row_index in range(2):
                validation_rows.append(
                    {
                        "sample_id": f"v{group_index}_{row_index}",
                        "spatial_block_id": f"g{group_index}",
                        "label": group_index % 3,
                        "latitude": 26.0 + group_index * 0.01,
                        "longitude": -82.0 - row_index * 0.00001,
                    }
                )
        final_test = pd.DataFrame(
            {
                "sample_id": ["t0", "t1"],
                "spatial_block_id": ["test0", "test1"],
                "label": [0, 2],
                "latitude": [27.0, 27.1],
                "longitude": [-82.0, -82.1],
            }
        )

        roles, exclusions = build_protocol_roles(
            pd.DataFrame(validation_rows),
            final_test,
            group_col="spatial_block_id",
            label_col="label",
            gate_fit_fraction=0.5,
            spatial_buffer_m=0.0,
            seed=7,
            event_id="ian",
        )

        self.assertTrue(exclusions.empty)
        self.assertEqual(set(roles), {"gate_fit", "risk_calibration", "final_test"})
        sample_sets = {
            name: set(frame["sample_id"].astype(str)) for name, frame in roles.items()
        }
        group_sets = {
            name: set(frame["spatial_block_id"].astype(str))
            for name, frame in roles.items()
        }
        for left, right in (
            ("gate_fit", "risk_calibration"),
            ("gate_fit", "final_test"),
            ("risk_calibration", "final_test"),
        ):
            self.assertFalse(sample_sets[left] & sample_sets[right])
            self.assertFalse(group_sets[left] & group_sets[right])
        self.assertEqual(
            len(roles["gate_fit"]) + len(roles["risk_calibration"]),
            len(validation_rows),
        )
        self.assertTrue(
            (roles["risk_calibration"]["protocol_role"] == "risk_calibration").all()
        )
        self.assertEqual(set(roles["risk_calibration"]["event_id"]), {"ian"})

    def test_spatial_buffer_protects_calibration_and_drops_gate_fit_boundary(self) -> None:
        validation = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c", "d", "e", "f"],
                "spatial_block_id": ["a", "b", "c", "d", "e", "f"],
                "label": [0, 0, 1, 1, 2, 2],
                "latitude": [26.0, 26.00001, 26.1, 26.2, 26.3, 26.4],
                "longitude": [-82.0] * 6,
            }
        )
        final_test = pd.DataFrame(
            {
                "sample_id": ["t"],
                "spatial_block_id": ["test"],
                "label": [2],
                "latitude": [27.0],
                "longitude": [-82.0],
            }
        )
        roles, exclusions = build_protocol_roles(
            validation,
            final_test,
            group_col="spatial_block_id",
            label_col="label",
            gate_fit_fraction=0.5,
            spatial_buffer_m=5.0,
            seed=3,
        )
        self.assertEqual(len(roles["final_test"]), 1)
        self.assertTrue(set(exclusions.columns) >= {"excluded_from", "near_split"})

    def test_invalid_fraction_and_overlapping_ids_fail(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "spatial_block_id": ["a", "b", "c"],
                "label": [0, 1, 2],
                "latitude": [1.0, 2.0, 3.0],
                "longitude": [1.0, 2.0, 3.0],
            }
        )
        with self.assertRaises(ValueError):
            build_protocol_roles(
                frame,
                frame.iloc[[0]].copy(),
                group_col="spatial_block_id",
                label_col="label",
                gate_fit_fraction=0.5,
                spatial_buffer_m=0.0,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
