from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from scripts.analyze_eaton_component_anatomy import (
    add_component_direction_columns,
    analyze_component_direction,
    audit_reference_fields,
    best_case_two_proportion_mde,
    load_and_validate_whitelist,
    validate_component_references,
    validate_predictions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WHITELIST_PATH = REPO_ROOT / "configs" / "eaton_image_visible_fields_v1.json"


def reference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": ["roof_a", "roof_b", "facade_a", "facade_b"],
            "component_dominance": ["roof", "roof", "facade", "facade"],
            "reference_semantics": ["damage_dominance"] * 4,
            "annotation_provenance": ["manual protocol v1"] * 4,
        }
    )


def prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": ["roof_a", "roof_b", "facade_a", "facade_b"],
            "seed": [42] * 4,
            "street_prediction": [0, 1, 2, 1],
            "remote_prediction": [1, 2, 1, 0],
            "spatial_block_id": ["a", "b", "a", "b"],
            "protocol_role": ["development"] * 4,
        }
    )


class WhitelistTests(unittest.TestCase):
    def test_frozen_whitelist_marks_every_inspector_attribute_non_damage(self) -> None:
        whitelist = load_and_validate_whitelist(WHITELIST_PATH)
        self.assertTrue(whitelist["fields"])
        self.assertTrue(
            all(field["component_damage_reference"] is False for field in whitelist["fields"])
        )
        vent = next(
            field for field in whitelist["fields"] if field["column"] == "dins_ventscreen"
        )
        self.assertEqual(vent["street"]["status"], "excluded")
        self.assertEqual(vent["remote"]["status"], "excluded")

    def test_field_audit_does_not_turn_unknown_into_negative_damage(self) -> None:
        whitelist = load_and_validate_whitelist(WHITELIST_PATH)
        rows = []
        for pair_id, value, source in (
            ("a", "Unknown", "source-1"),
            ("b", "Asphalt", "source-2"),
        ):
            row = {
                "pair_id": pair_id,
                "dins_join_status": "matched_spatial",
                "dins_source_globalid": source,
                "category": "No Damage",
                "dins_damage": "No Damage",
            }
            for field in whitelist["fields"]:
                row[field["column"]] = value
            rows.append(row)
        audit, summary = audit_reference_fields(pd.DataFrame(rows), whitelist)
        roof = audit.set_index("field").loc["dins_roofconstruction"]
        self.assertEqual(roof["attachment_rows_unknown"], 1)
        self.assertEqual(roof["attachment_rows_usable_reference"], 1)
        self.assertFalse(bool(roof["component_damage_reference"]))
        self.assertEqual(summary["component_damage_reference_columns_found"], [])


class InputGuardTests(unittest.TestCase):
    def test_material_reference_is_rejected_as_damage_dominance(self) -> None:
        bad = reference_frame()
        bad["reference_semantics"] = "roof_construction_material"
        with self.assertRaisesRegex(ValueError, "damage_dominance"):
            validate_component_references(bad)

    def test_final_test_predictions_are_refused(self) -> None:
        bad = prediction_frame()
        bad["protocol_role"] = "final_test"
        with self.assertRaisesRegex(ValueError, "final_test"):
            validate_predictions(bad)

    def test_missing_spatial_blocks_are_refused(self) -> None:
        bad = prediction_frame().drop(columns="spatial_block_id")
        with self.assertRaisesRegex(ValueError, "spatial_block_id"):
            validate_predictions(bad)


class DirectionAnalysisTests(unittest.TestCase):
    def test_signed_direction_and_consistency_follow_view_order(self) -> None:
        joined = prediction_frame().merge(reference_frame(), on="pair_id")
        anatomy = add_component_direction_columns(joined)
        self.assertListEqual(anatomy["signed_direction"].tolist(), [1, 1, -1, -1])
        self.assertTrue(anatomy["direction_consistent"].all())

    def test_h_b_statistics_use_real_supplied_predictions(self) -> None:
        stats = analyze_component_direction(
            prediction_frame(),
            reference_frame(),
            bootstrap_replicates=50,
            bootstrap_seed=7,
        )
        self.assertEqual(len(stats), 1)
        row = stats.iloc[0]
        self.assertEqual(row["roof_positive_direction_share"], 1.0)
        self.assertEqual(row["facade_positive_direction_share"], 0.0)
        self.assertEqual(row["h_b1_directional_contrast"], 1.0)
        self.assertEqual(row["h_b2_direction_consistent_share"], 1.0)

    def test_best_case_power_is_explicitly_structure_count_based(self) -> None:
        mde = best_case_two_proportion_mde(33, 583)
        self.assertIsNotNone(mde)
        self.assertGreater(mde, 0.24)
        self.assertLess(mde, 0.26)
        self.assertIsNone(best_case_two_proportion_mde(0, 583))


if __name__ == "__main__":
    unittest.main()
