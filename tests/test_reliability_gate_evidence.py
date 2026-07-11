from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.train_reliability_gate import (
    ALL_FEATURES,
    SplitData,
    _parse_split_manifest_map,
    _canonical_sample_id,
    _subset_split_data,
    build_evidence_frame,
)


class ReliabilityGateEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        features = np.zeros((2, len(ALL_FEATURES)), dtype=np.float64)
        features[:, :4] = np.array(
            [
                [0.10, 0.20, 0.10, 0.40],
                [0.50, 0.30, -0.20, 0.25],
            ]
        )
        self.data = SplitData(
            features=features,
            street_probs=np.array(
                [[0.70, 0.10, 0.10, 0.10], [0.10, 0.10, 0.70, 0.10]]
            ),
            remote_probs=np.array(
                [[0.10, 0.70, 0.10, 0.10], [0.10, 0.20, 0.60, 0.10]]
            ),
            crossview_probs=np.array(
                [[0.20, 0.20, 0.50, 0.10], [0.60, 0.10, 0.20, 0.10]]
            ),
            targets=np.array([2, 2]),
            sample_ids=np.array(["sample_001", "sample_002"]),
        )
        self.two_view_weights = np.array([[0.75, 0.25], [0.30, 0.70]])
        self.three_view_weights = np.array(
            [[0.20, 0.20, 0.60], [0.20, 0.30, 0.50]]
        )

    def test_complete_dynamic_schema_and_manifest_join(self) -> None:
        manifest = pd.DataFrame(
            {
                "sample_id": ["sample_002", "sample_001"],
                "object_id": [202, 101],
                "latitude": [2.0, 1.0],
                "longitude": [-82.0, -81.0],
                "spatial_block_id": ["tile-b", "tile-a"],
                "sequence_id": ["seq-b", "seq-a"],
                "protocol_role": ["final_test", "final_test"],
            }
        )

        evidence = build_evidence_frame(
            self.data,
            self.two_view_weights,
            self.three_view_weights,
            dataset="synthetic_4class",
            seed=42,
            manifest=manifest,
        )

        self.assertEqual(evidence["sample_id"].tolist(), ["1", "2"])
        self.assertEqual(evidence["objectid"].tolist(), [101, 202])
        self.assertEqual(evidence["tile_id"].tolist(), ["tile-a", "tile-b"])
        self.assertEqual(evidence["spatial_block_id"].tolist(), ["tile-a", "tile-b"])
        self.assertEqual(evidence["sequence_id"].tolist(), ["seq-a", "seq-b"])
        self.assertEqual(evidence["protocol_role"].tolist(), ["final_test", "final_test"])
        self.assertEqual(evidence["location"].tolist(), ["POINT (-81 1)", "POINT (-82 2)"])
        self.assertEqual(evidence["latitude"].tolist(), [1.0, 2.0])
        self.assertEqual(evidence["num_classes"].tolist(), [4, 4])
        self.assertEqual(evidence["hard_conflict"].tolist(), [1, 0])
        self.assertEqual(evidence["gate2_selected_prediction"].tolist(), [0, 2])
        self.assertEqual(evidence["gate3_selected_prediction"].tolist(), [2, 2])
        self.assertTrue(
            evidence["gate_linear_prediction"].equals(
                evidence["gate2_selected_prediction"]
            )
        )
        self.assertTrue(
            evidence["gate3_linear_prediction"].equals(
                evidence["gate3_selected_prediction"]
            )
        )

        for prefix in ("street", "remote", "crossview", "gate2", "gate3"):
            probability_columns = [
                column
                for column in evidence.columns
                if column.startswith(f"{prefix}_probability_")
            ]
            self.assertEqual(len(probability_columns), 4)
        np.testing.assert_allclose(
            evidence[["gate2_weight_street", "gate2_weight_remote"]],
            self.two_view_weights,
        )
        np.testing.assert_allclose(
            evidence[
                [
                    "gate3_weight_street",
                    "gate3_weight_remote",
                    "gate3_weight_crossview",
                ]
            ],
            self.three_view_weights,
        )
        np.testing.assert_allclose(evidence["building_ratio"], [0.10, 0.50])
        np.testing.assert_allclose(evidence["street_confidence"], [0.70, 0.70])
        self.assertTrue((evidence["js_divergence"] >= 0.0).all())

    def test_metadata_columns_exist_without_manifest(self) -> None:
        evidence = build_evidence_frame(
            self.data, self.two_view_weights, self.three_view_weights
        )
        for column in ("objectid", "location", "latitude", "longitude", "tile_id"):
            self.assertIn(column, evidence.columns)
            self.assertTrue(evidence[column].isna().all())

    def test_split_manifest_mapping_parser(self) -> None:
        mapping = _parse_split_manifest_map(
            "ian_original=data/splits/ian/test.csv,milton_original=data/splits/milton"
        )
        self.assertEqual(mapping["ian_original"], Path("data/splits/ian/test.csv"))
        self.assertEqual(mapping["milton_original"], Path("data/splits/milton"))
        with self.assertRaises(ValueError):
            _parse_split_manifest_map("missing_equals")

    def test_sample_id_canonicalization_only_accepts_known_whole_patterns(self) -> None:
        self.assertEqual(_canonical_sample_id("sample_001"), "1")
        self.assertEqual(_canonical_sample_id("milton_002"), "2")
        self.assertEqual(_canonical_sample_id("001.0"), "1")
        self.assertEqual(_canonical_sample_id("tensor(33)"), "33")
        self.assertEqual(_canonical_sample_id("property-123-west"), "property-123-west")

    def test_protocol_role_subset_requires_manifest_coverage(self) -> None:
        subset = _subset_split_data(self.data, {"2"}, role="risk_calibration")
        self.assertEqual(subset.sample_ids.tolist(), ["2"])
        self.assertEqual(len(subset.targets), 1)
        with self.assertRaisesRegex(ValueError, "absent from predictions"):
            _subset_split_data(self.data, {"999"}, role="final_test")


if __name__ == "__main__":
    unittest.main()
