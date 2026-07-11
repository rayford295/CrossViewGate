from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from crossview_conflict.decision.attestation import (
    ANNOTATION_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    evaluate_attestation_dependency_gates,
    expected_attestation_coverage_gain,
    load_attestation_contract,
    load_dependency_inventory,
    validate_attestation_annotations,
    validate_attestation_predictions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "configs" / "crossviewguard_attestation_contract_v1.json"
INVENTORY_PATH = (
    REPO_ROOT
    / "configs"
    / "crossviewguard_attestation_dependency_inventory_20260710.json"
)
SHA_A = "0" * 64
SHA_B = "1" * 64


def synthetic_annotation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": [ANNOTATION_SCHEMA_VERSION],
            "entity_id": ["synthetic-entity-1"],
            "event_id": ["synthetic-event"],
            "dependency_group_id": ["synthetic-group-1"],
            "split_role": ["attestation_fit"],
            "view_id": ["synthetic-street-0"],
            "view_type": ["street_sector"],
            "field_id": ["roof_damage"],
            "reference_semantics": ["component_damage_state"],
            "reference_state": ["damaged"],
            "view_observation_state": ["damaged"],
            "observability": ["visible"],
            "reference_source_type": ["synthetic_fixture"],
            "reference_source_field": ["synthetic_roof_damage"],
            "reference_provenance": ["synthetic fixture; no empirical claim"],
            "view_annotation_provenance": ["synthetic fixture; no human label"],
            "record_origin": ["synthetic_fixture"],
            "source_artifact_sha256": [SHA_A],
            "media_sha256": [SHA_B],
        }
    )


def synthetic_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": [PREDICTION_SCHEMA_VERSION],
            "entity_id": ["synthetic-entity-1"],
            "event_id": ["synthetic-event"],
            "dependency_group_id": ["synthetic-group-1"],
            "split_role": ["validation"],
            "seed": [42],
            "view_id": ["synthetic-street-0"],
            "view_type": ["street_sector"],
            "field_id": ["roof_damage"],
            "attestation_probability": [0.7],
            "abstain_probability": [0.2],
            "predicted_observation_state": ["damaged"],
            "record_origin": ["synthetic_fixture"],
            "model_artifact_sha256": [SHA_A],
            "annotation_artifact_sha256": [SHA_B],
        }
    )


class DependencyGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_attestation_contract(CONTRACT_PATH)
        cls.inventory = load_dependency_inventory(INVENTORY_PATH, cls.contract)

    def test_current_cvian_and_eaton_are_strict_no_go(self) -> None:
        for dataset in ("cvian", "eaton"):
            decision = evaluate_attestation_dependency_gates(
                self.contract, self.inventory, dataset
            )
            self.assertFalse(decision.rq2_matrix_go)
            self.assertFalse(decision.rq2_geometry_go)
            self.assertFalse(decision.rq3_coverage_go)
            self.assertIn(
                "rq1_attestability_signal_supported",
                decision.blockers["rq2_matrix"],
            )
            self.assertIn(
                "real_damage_reference_available",
                decision.blockers["rq2_matrix"],
            )

    def test_rq3_fails_closed_on_upstream_rq2(self) -> None:
        decision = evaluate_attestation_dependency_gates(
            self.contract, self.inventory, "cvian"
        )
        self.assertIn("upstream_rq2_matrix_no_go", decision.blockers["rq3_coverage"])
        self.assertIn(
            "calibrated_attestation_predictions_available",
            decision.blockers["rq3_coverage"],
        )


class SyntheticAnnotationGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_attestation_contract(CONTRACT_PATH)

    def test_synthetic_fixture_only_passes_synthetic_test_mode(self) -> None:
        validated = validate_attestation_annotations(
            synthetic_annotation_frame(), self.contract, mode="synthetic_test"
        )
        self.assertEqual(len(validated), 1)
        with self.assertRaisesRegex(ValueError, "Synthetic fixtures are forbidden"):
            validate_attestation_annotations(
                synthetic_annotation_frame(), self.contract, mode="claim"
            )

    def test_material_attribute_cannot_be_damage_reference(self) -> None:
        bad = synthetic_annotation_frame()
        bad.loc[0, "reference_source_field"] = "dins_roofconstruction"
        with self.assertRaisesRegex(ValueError, "Construction/material"):
            validate_attestation_annotations(bad, self.contract, mode="synthetic_test")

    def test_fire_origin_cannot_be_damage_reference(self) -> None:
        bad = synthetic_annotation_frame()
        bad.loc[0, "reference_source_field"] = "dins_wherefirestartedonstructure"
        with self.assertRaisesRegex(ValueError, "fire-origin"):
            validate_attestation_annotations(bad, self.contract, mode="synthetic_test")

    def test_field_semantics_must_be_damage_not_material(self) -> None:
        bad = synthetic_annotation_frame()
        bad.loc[0, "reference_semantics"] = "roof_construction_material"
        with self.assertRaisesRegex(ValueError, "invalid reference semantics"):
            validate_attestation_annotations(bad, self.contract, mode="synthetic_test")

    def test_not_visible_view_must_abstain(self) -> None:
        bad = synthetic_annotation_frame()
        bad.loc[0, "observability"] = "not_visible"
        with self.assertRaisesRegex(ValueError, "must abstain"):
            validate_attestation_annotations(bad, self.contract, mode="synthetic_test")

    def test_dependency_group_cannot_cross_roles(self) -> None:
        bad = pd.concat(
            [synthetic_annotation_frame(), synthetic_annotation_frame()],
            ignore_index=True,
        )
        bad.loc[1, "entity_id"] = "synthetic-entity-2"
        bad.loc[1, "view_id"] = "synthetic-street-1"
        bad.loc[1, "split_role"] = "validation"
        with self.assertRaisesRegex(ValueError, "dependency group"):
            validate_attestation_annotations(bad, self.contract, mode="synthetic_test")


class SyntheticPredictionAndCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_attestation_contract(CONTRACT_PATH)

    def test_synthetic_prediction_schema_is_plumbing_only(self) -> None:
        validated = validate_attestation_predictions(
            synthetic_prediction_frame(), self.contract, mode="synthetic_test"
        )
        self.assertEqual(len(validated), 1)
        with self.assertRaisesRegex(ValueError, "Synthetic fixtures are forbidden"):
            validate_attestation_predictions(
                synthetic_prediction_frame(), self.contract, mode="claim"
            )

    def test_prediction_probabilities_fail_closed(self) -> None:
        bad = synthetic_prediction_frame()
        bad.loc[0, "attestation_probability"] = 0.9
        bad.loc[0, "abstain_probability"] = 0.2
        with self.assertRaisesRegex(ValueError, "cannot sum above one"):
            validate_attestation_predictions(bad, self.contract, mode="synthetic_test")

    def test_expected_coverage_gain_counts_only_new_fields(self) -> None:
        current = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=bool)
        candidate = np.asarray([[0.9, 0.5, 0.2], [0.1, 0.8, 1.0]])
        weights = np.asarray([2.0, 1.0, 3.0])
        gain = expected_attestation_coverage_gain(
            current, candidate, weights, acquisition_cost=0.5
        )
        np.testing.assert_allclose(gain, [0.6, 2.7])


if __name__ == "__main__":
    unittest.main()
