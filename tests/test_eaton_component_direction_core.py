from __future__ import annotations

import json
import math
import unittest

import pandas as pd

from crossview_conflict.analysis.eaton_component_direction import (
    ANNOTATION_FIELDS,
    PROBABILITY_COLUMNS,
    REFERENCE_SEMANTICS,
    SEVERITY_CLASS_ORDER,
    analyze_component_direction,
    build_adjudicated_reference,
    compute_annotation_reliability,
    ensemble_seed_predictions,
    validate_annotation_packet,
    validate_component_reference,
    validate_seed_predictions,
)


PROTOCOL_VERSION = "eaton-component-direction-v1"
PROTOCOL_SHA256 = "a" * 64


def annotation_packet() -> pd.DataFrame:
    rows = []
    labels = {
        "roof-0": (
            "roof",
            "not_assessable",
            "indeterminate",
            "assessable",
            "indeterminate",
        ),
        "roof-1": (
            "roof",
            "not_assessable",
            "indeterminate",
            "assessable",
            "indeterminate",
        ),
        "facade-0": (
            "facade",
            "indeterminate",
            "assessable",
            "indeterminate",
            "not_assessable",
        ),
        "facade-1": (
            "facade",
            "indeterminate",
            "assessable",
            "indeterminate",
            "not_assessable",
        ),
    }
    for index, (pair_id, values) in enumerate(labels.items()):
        for rater_id in ("rater-a", "rater-b"):
            component = values[0]
            if pair_id == "facade-1" and rater_id == "rater-b":
                component = "mixed"
            rows.append(
                {
                    "pair_id": pair_id,
                    "rater_id": rater_id,
                    "component_dominance": component,
                    **dict(zip(ANNOTATION_FIELDS[1:], values[1:])),
                    "protocol_version": PROTOCOL_VERSION,
                    "protocol_sha256": PROTOCOL_SHA256,
                    "blinded_to_model_outputs": True,
                    "street_media_sha256": f"{index + 1:x}" * 64,
                    "overhead_media_sha256": f"{index + 5:x}" * 64,
                }
            )
    return pd.DataFrame(rows)


def adjudication_frame() -> pd.DataFrame:
    pair = annotation_packet().query("pair_id == 'facade-1'").iloc[0]
    return pd.DataFrame(
        [
            {
                "pair_id": "facade-1",
                "adjudicator_id": "adjudicator-c",
                "component_dominance": "facade",
                **{field: pair[field] for field in ANNOTATION_FIELDS[1:]},
                "protocol_version": PROTOCOL_VERSION,
                "protocol_sha256": PROTOCOL_SHA256,
                "blinded_to_model_outputs": True,
                "street_media_sha256": pair["street_media_sha256"],
                "overhead_media_sha256": pair["overhead_media_sha256"],
                "adjudication_note": "Facade damage is the dominant observed component.",
            }
        ]
    )


def prediction_frame() -> pd.DataFrame:
    specs = {
        "roof-0": (0, "block-a", 0, 1),
        "roof-1": (1, "block-b", 1, 2),
        "facade-0": (0, "block-a", 1, 0),
        "facade-1": (1, "block-b", 2, 1),
    }
    rows = []
    for pair_id, (target, block, street_class, overhead_class) in specs.items():
        for seed in (42, 123):
            row = {
                "pair_id": pair_id,
                "seed": seed,
                "spatial_block_id": block,
                "protocol_role": "study_development",
                "target": target,
                "class_order": json.dumps(list(SEVERITY_CLASS_ORDER)),
            }
            for column in PROBABILITY_COLUMNS:
                row[column] = 0.05
            for view, predicted in (
                ("street", street_class),
                ("overhead", overhead_class),
            ):
                for class_index, class_name in enumerate(SEVERITY_CLASS_ORDER):
                    row[f"{view}_prob_{class_name}"] = (
                        0.90 if class_index == predicted else 0.05
                    )
            rows.append(row)
    return pd.DataFrame(rows)


class AnnotationPacketTest(unittest.TestCase):
    def test_two_blinded_raters_and_media_hashes_are_validated(self) -> None:
        validated = validate_annotation_packet(
            annotation_packet(),
            expected_protocol_version=PROTOCOL_VERSION,
            expected_protocol_sha256=PROTOCOL_SHA256,
        )
        self.assertEqual(len(validated), 8)
        self.assertEqual(validated["pair_id"].nunique(), 4)

        unblinded = annotation_packet()
        unblinded.loc[0, "blinded_to_model_outputs"] = False
        with self.assertRaisesRegex(ValueError, "blinded_to_model_outputs"):
            validate_annotation_packet(unblinded)

        leaked = annotation_packet().assign(source_label="Destroyed")
        with self.assertRaisesRegex(ValueError, "cannot contain model/selection fields"):
            validate_annotation_packet(leaked)

        mismatched = annotation_packet()
        mismatched.loc[0, "street_media_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "inconsistent street_media_sha256"):
            validate_annotation_packet(mismatched)

    def test_nominal_reliability_and_required_adjudication(self) -> None:
        reliability = compute_annotation_reliability(annotation_packet()).set_index("field")
        dominance = reliability.loc["component_dominance"]
        self.assertEqual(dominance["observed_agreement"], 0.75)
        self.assertTrue(math.isfinite(float(dominance["cohen_kappa"])))

        with self.assertRaisesRegex(ValueError, "Adjudication is required"):
            build_adjudicated_reference(annotation_packet())

        reference = build_adjudicated_reference(
            annotation_packet(), adjudication_frame()
        )
        self.assertEqual(len(reference), 4)
        self.assertFalse(reference["pair_id"].duplicated().any())
        self.assertEqual(set(reference["reference_semantics"]), {REFERENCE_SEMANTICS})
        self.assertTrue(
            bool(reference.set_index("pair_id").loc["facade-1", "adjudicated"])
        )
        self.assertEqual(reference["reference_sha256"].str.len().unique().tolist(), [64])
        validate_component_reference(reference)

        tampered = reference.copy()
        provenance = json.loads(tampered.loc[0, "annotation_provenance"])
        provenance["raw_annotations"][0]["component_dominance"] = "roof"
        tampered.loc[0, "annotation_provenance"] = json.dumps(provenance)
        with self.assertRaisesRegex(ValueError, "two agreeing ratings"):
            validate_component_reference(tampered)


class PredictionEnsembleTest(unittest.TestCase):
    def test_exact_order_role_probability_and_seed_completeness(self) -> None:
        validated = validate_seed_predictions(prediction_frame())
        self.assertEqual(validated["seed"].nunique(), 2)

        bad_order = prediction_frame()
        bad_order["class_order"] = json.dumps(list(reversed(SEVERITY_CLASS_ORDER)))
        with self.assertRaisesRegex(ValueError, "exact ordinal order"):
            validate_seed_predictions(bad_order)

        bad_sum = prediction_frame()
        bad_sum.loc[0, "street_prob_destroyed"] = 0.50
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_seed_predictions(bad_sum)

        bad_role = prediction_frame()
        bad_role.loc[0, "protocol_role"] = "spatial_confirmation"
        with self.assertRaisesRegex(ValueError, "study_development"):
            validate_seed_predictions(bad_role)

        incomplete = prediction_frame().drop(index=0)
        with self.assertRaisesRegex(ValueError, "exact same pair_id set"):
            validate_seed_predictions(incomplete)

    def test_seed_rows_are_ensembled_before_direction_analysis(self) -> None:
        predictions = prediction_frame()
        predictions.loc[
            (predictions["pair_id"] == "roof-0") & (predictions["seed"] == 123),
            [
                "street_prob_no_or_trace_damage",
                "street_prob_damaged_repairable",
                "street_prob_destroyed",
            ],
        ] = [0.80, 0.10, 0.10]
        ensemble = ensemble_seed_predictions(predictions)
        self.assertEqual(len(ensemble), 4)
        self.assertTrue((ensemble["seed_count"] == 2).all())
        roof = ensemble.set_index("pair_id").loc["roof-0"]
        self.assertAlmostEqual(roof["street_prob_no_or_trace_damage"], 0.85)

    def test_confirmation_role_requires_explicit_opt_in(self) -> None:
        predictions = prediction_frame()
        predictions["protocol_role"] = "spatial_confirmation"
        with self.assertRaisesRegex(ValueError, "study_development"):
            ensemble_seed_predictions(predictions)
        ensemble = ensemble_seed_predictions(
            predictions, expected_role="spatial_confirmation"
        )
        self.assertEqual(set(ensemble["protocol_role"]), {"spatial_confirmation"})


class DirectionAnalysisTest(unittest.TestCase):
    def test_h_b1_h_b2_use_eligible_ensembles_and_spatial_bootstrap(self) -> None:
        reference = build_adjudicated_reference(
            annotation_packet(), adjudication_frame()
        )
        result = analyze_component_direction(
            prediction_frame(),
            reference,
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )
        self.assertEqual(len(result.pair_level), 4)
        self.assertTrue(result.pair_level["mechanism_eligible"].all())
        row = result.statistics.iloc[0]
        self.assertEqual(row["ensemble_pair_count"], 4)
        self.assertEqual(row["mechanism_eligible_disagreement_count"], 4)
        self.assertEqual(row["mechanism_eligibility_coverage_among_disagreements"], 1.0)
        self.assertEqual(row["target_0_standardization_weight"], 0.5)
        self.assertEqual(row["target_1_standardization_weight"], 0.5)
        self.assertTrue(bool(row["severity_standardization_common_support"]))
        self.assertEqual(row["roof_target_0_eligible_disagreement_count"], 1)
        self.assertEqual(row["facade_target_1_eligible_disagreement_count"], 1)
        self.assertEqual(row["h_b1_directional_contrast"], 1.0)
        self.assertEqual(row["h_b2_direction_consistent_share"], 1.0)
        self.assertGreater(row["h_b1_directional_contrast_bootstrap_finite_replicates"], 0)

    def test_pair_mismatch_is_never_silently_inner_joined(self) -> None:
        reference = build_adjudicated_reference(
            annotation_packet(), adjudication_frame()
        ).query("pair_id != 'roof-0'")
        with self.assertRaisesRegex(ValueError, "pair sets must match exactly"):
            analyze_component_direction(
                prediction_frame(), reference, bootstrap_replicates=10
            )

    def test_full_prediction_population_requires_exactly_all_disagreements_annotated(self) -> None:
        predictions = prediction_frame()
        extra_rows = []
        for seed in (42, 123):
            row = {
                "pair_id": "agreement-extra",
                "seed": seed,
                "spatial_block_id": "block-c",
                "protocol_role": "study_development",
                "target": 0,
                "class_order": json.dumps(list(SEVERITY_CLASS_ORDER)),
            }
            for column in PROBABILITY_COLUMNS:
                row[column] = 0.05
            for view in ("street", "overhead"):
                row[f"{view}_prob_no_or_trace_damage"] = 0.90
            extra_rows.append(row)
        predictions = pd.concat([predictions, pd.DataFrame(extra_rows)], ignore_index=True)
        reference = build_adjudicated_reference(
            annotation_packet(), adjudication_frame()
        )
        result = analyze_component_direction(
            predictions, reference, bootstrap_replicates=20, bootstrap_seed=3
        )
        row = result.statistics.iloc[0]
        self.assertEqual(row["ensemble_pair_count"], 5)
        self.assertEqual(row["disagreement_pair_count"], 4)
        self.assertEqual(len(result.pair_level), 4)


if __name__ == "__main__":
    unittest.main()
