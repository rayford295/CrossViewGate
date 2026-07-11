from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from crossview_conflict.analysis.eaton_component_direction import (
    ANNOTATION_FIELDS,
    PROBABILITY_COLUMNS,
    SEVERITY_CLASS_ORDER,
    build_adjudicated_reference,
    compute_annotation_reliability,
    analyze_component_direction,
)
from crossview_conflict.analysis.eaton_component_gate import (
    CONTINUE_TO_BLINDED_ANNOTATION,
    GO_CONFIRMATION,
    NON_PROTOCOL_TEST_OVERRIDE,
    NO_GO_CONFIRMATION,
    PASS_PRE_UNBLIND_GATE,
    PROTOCOL_EXECUTION,
    STOP_PRE_ANNOTATION_FUTILITY,
    STOP_PRE_UNBLIND_GATE,
    build_confirmation_analysis_artifact,
    build_confirmation_authorization_artifact,
    confirmation_decision,
    pre_annotation_futility_gate,
    pre_unblinding_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "eaton_component_direction_v1.json"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: str) -> str:
    return _sha_bytes(value.encode("utf-8"))


def _resign(payload: dict[str, object], digest_field: str) -> None:
    unsigned = {key: value for key, value in payload.items() if key != digest_field}
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    payload[digest_field] = _sha_bytes(encoded)


def protocol_config(*, small_count_thresholds: bool = False) -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if small_count_thresholds:
        gate = config["pre_unblinding_gate"]
        gate["mechanism_eligible_disagreements_min"] = 1
        gate["roof_eligible_min"] = 1
        gate["facade_eligible_min"] = 1
        gate["eligible_spatial_blocks_min"] = 1
        gate["eligible_blocks_per_stratum_min"] = 1
    return config


def _config_bytes(config: dict) -> bytes:
    return (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def annotation_packet(
    protocol_hash: str,
    *,
    facade_target_two_mixed: bool = False,
    low_reliability: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stratum in ("roof", "facade"):
        for target in range(3):
            pair_id = f"{stratum}-{target}"
            final_dominance = (
                "mixed"
                if facade_target_two_mixed and stratum == "facade" and target == 2
                else stratum
            )
            values = {
                "component_dominance": final_dominance,
                "street_roof_assessability": (
                    "not_assessable" if stratum == "roof" else "assessable"
                ),
                "street_facade_assessability": (
                    "assessable" if stratum == "facade" else "not_assessable"
                ),
                "overhead_roof_assessability": (
                    "assessable" if stratum == "roof" else "not_assessable"
                ),
                "overhead_facade_assessability": (
                    "not_assessable" if stratum == "facade" else "assessable"
                ),
            }
            for rater_id in ("rater-a", "rater-b"):
                rater_values = dict(values)
                if (
                    low_reliability
                    and rater_id == "rater-b"
                    and stratum == "roof"
                    and target in (0, 1)
                ):
                    rater_values["component_dominance"] = "mixed"
                rows.append(
                    {
                        "pair_id": pair_id,
                        "rater_id": rater_id,
                        **rater_values,
                        "protocol_version": "eaton-component-direction-v1",
                        "protocol_sha256": protocol_hash,
                        "blinded_to_model_outputs": True,
                        "street_media_sha256": _sha(pair_id + "-street"),
                        "overhead_media_sha256": _sha(pair_id + "-overhead"),
                    }
                )
    return pd.DataFrame(rows)


def adjudications_for(
    raw: pd.DataFrame, protocol_hash: str
) -> pd.DataFrame | None:
    rows: list[dict[str, object]] = []
    for pair_id, group in raw.groupby("pair_id", sort=True):
        if not any(group[field].nunique() > 1 for field in ANNOTATION_FIELDS):
            continue
        first = group.iloc[0]
        rows.append(
            {
                "pair_id": pair_id,
                "adjudicator_id": "adjudicator-c",
                **{field: first[field] for field in ANNOTATION_FIELDS},
                "protocol_version": "eaton-component-direction-v1",
                "protocol_sha256": protocol_hash,
                "blinded_to_model_outputs": True,
                "street_media_sha256": first["street_media_sha256"],
                "overhead_media_sha256": first["overhead_media_sha256"],
                "adjudication_note": "Resolved from the paired image evidence.",
            }
        )
    return pd.DataFrame(rows) if rows else None


def predictions(
    *,
    reverse_order: bool = False,
    mechanism_consistent: bool = False,
    role: str = "study_development",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in (42, 123, 456, 789, 1011):
        for stratum in ("roof", "facade"):
            for target in range(3):
                pair_id = f"{stratum}-{target}"
                row: dict[str, object] = {
                    "pair_id": pair_id,
                    "seed": seed,
                    "spatial_block_id": "block-shared",
                    "protocol_role": role,
                    "target": target,
                    "class_order": json.dumps(list(SEVERITY_CLASS_ORDER)),
                }
                for column in PROBABILITY_COLUMNS:
                    row[column] = 0.05
                if mechanism_consistent and stratum == "facade":
                    row["street_prob_damaged_repairable"] = 0.90
                    row["overhead_prob_no_or_trace_damage"] = 0.90
                elif reverse_order:
                    row["street_prob_damaged_repairable"] = 0.90
                    row["overhead_prob_no_or_trace_damage"] = 0.90
                else:
                    row["street_prob_no_or_trace_damage"] = 0.90
                    row["overhead_prob_damaged_repairable"] = 0.90
                rows.append(row)
    return pd.DataFrame(rows)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def gate_bundle(
    *,
    config: dict | None = None,
    prediction_frame: pd.DataFrame | None = None,
    role: str = "study_development",
    facade_target_two_mixed: bool = False,
    low_reliability: bool = False,
) -> dict[str, object]:
    config = config or protocol_config()
    config_bytes = _config_bytes(config)
    protocol_hash = _sha_bytes(config_bytes)
    pred = prediction_frame if prediction_frame is not None else predictions(role=role)
    raw = annotation_packet(
        protocol_hash,
        facade_target_two_mixed=facade_target_two_mixed,
        low_reliability=low_reliability,
    )
    adjudications = adjudications_for(raw, protocol_hash)
    reference = build_adjudicated_reference(
        raw,
        adjudications,
        expected_protocol_version="eaton-component-direction-v1",
        expected_protocol_sha256=protocol_hash,
    )
    reliability = compute_annotation_reliability(
        raw,
        expected_protocol_version="eaton-component-direction-v1",
        expected_protocol_sha256=protocol_hash,
    )
    one_seed = pred.drop_duplicates("pair_id")
    manifest = one_seed[
        ["pair_id", "spatial_block_id", "protocol_role", "target"]
    ].rename(columns={"target": "label"})
    prediction_bytes = _csv_bytes(pred)
    manifest_bytes = _csv_bytes(manifest)
    metadata = {
        "schema_version": "eaton-component-direction-predictions-v1",
        "protocol_version": "eaton-component-direction-v1",
        "status": (
            "PROTOCOL_DEVELOPMENT_RUN"
            if role == "study_development"
            else "PROTOCOL_CONFIRMATION_RUN"
        ),
        "protocol_config_sha256": protocol_hash,
        "role_manifest_sha256": {role: _sha_bytes(manifest_bytes)},
        "prediction_role": role,
        "spatial_confirmation_read_or_scored": False,
        "spatial_confirmation_scored_once": role == "spatial_confirmation",
        "class_order": list(SEVERITY_CLASS_ORDER),
        "seeds": [42, 123, 456, 789, 1011],
        "prediction_csv_sha256": _sha_bytes(prediction_bytes),
    }
    summary = {
        "schema_version": "eaton-component-direction-spatial-summary-v1",
        "protocol_version": "eaton-component-direction-v1",
        "protocol_config_sha256": protocol_hash,
        "analysis_status": "PROTOCOL_READY_CONFIRMATION_UNTOUCHED",
        "role_manifest_sha256": {
            "study_development": (
                _sha_bytes(manifest_bytes) if role == "study_development" else "d" * 64
            ),
            "spatial_confirmation": (
                _sha_bytes(manifest_bytes) if role == "spatial_confirmation" else "f" * 64
            ),
        },
        "role_rows": {
            "study_development": len(manifest) if role == "study_development" else 6,
            "spatial_confirmation": len(manifest) if role == "spatial_confirmation" else 6,
        },
        "confirmation_commitment": {
            "status": "UNSCORED_RESERVED_SAME_EVENT_SPATIAL_CONFIRMATION",
            "manifest_sha256": (
                _sha_bytes(manifest_bytes)
                if role == "spatial_confirmation"
                else "f" * 64
            ),
        },
    }
    summary_bytes = json.dumps(summary).encode("utf-8")
    return {
        "predictions": pred,
        "reference": reference,
        "reliability": reliability,
        "config": config,
        "expected_protocol_sha256": protocol_hash,
        "expected_protocol_summary_sha256": _sha_bytes(summary_bytes),
        "expected_prediction_csv_sha256": _sha_bytes(prediction_bytes),
        "expected_prediction_metadata_sha256": _sha_bytes(
            json.dumps(metadata).encode("utf-8")
        ),
        "raw_annotations": raw,
        "adjudications": adjudications,
        "protocol_config_bytes": config_bytes,
        "prediction_csv_bytes": prediction_bytes,
        "prediction_metadata_bytes": json.dumps(metadata).encode("utf-8"),
        "role_manifest_csv_bytes": manifest_bytes,
        "protocol_summary_bytes": summary_bytes,
        "expected_role": role,
    }


def paired_role_bundles(
    *, mechanism_consistent: bool
) -> tuple[dict[str, object], dict[str, object]]:
    """Build development/confirmation inputs sharing one trusted split summary."""

    config = protocol_config(small_count_thresholds=True)
    development = gate_bundle(config=config, role="study_development")
    confirmation = gate_bundle(
        config=config,
        prediction_frame=predictions(
            role="spatial_confirmation",
            mechanism_consistent=mechanism_consistent,
        ),
        role="spatial_confirmation",
    )
    development_manifest_hash = _sha_bytes(development["role_manifest_csv_bytes"])
    confirmation_manifest_hash = _sha_bytes(confirmation["role_manifest_csv_bytes"])
    summary = {
        "schema_version": "eaton-component-direction-spatial-summary-v1",
        "protocol_version": "eaton-component-direction-v1",
        "protocol_config_sha256": development["expected_protocol_sha256"],
        "analysis_status": "PROTOCOL_READY_CONFIRMATION_UNTOUCHED",
        "role_manifest_sha256": {
            "study_development": development_manifest_hash,
            "spatial_confirmation": confirmation_manifest_hash,
        },
        "role_rows": {
            "study_development": 6,
            "spatial_confirmation": 6,
        },
        "confirmation_commitment": {
            "status": "UNSCORED_RESERVED_SAME_EVENT_SPATIAL_CONFIRMATION",
            "manifest_sha256": confirmation_manifest_hash,
        },
    }
    summary_bytes = json.dumps(summary).encode("utf-8")
    summary_hash = _sha_bytes(summary_bytes)
    for bundle in (development, confirmation):
        bundle["protocol_summary_bytes"] = summary_bytes
        bundle["expected_protocol_summary_sha256"] = summary_hash
    return development, confirmation


def run_gate(
    bundle: dict[str, object],
    *,
    overrides: tuple[int, int] | None = None,
) -> dict[str, object]:
    kwargs = dict(bundle)
    pred = kwargs.pop("predictions")
    ref = kwargs.pop("reference")
    reliability_frame = kwargs.pop("reliability")
    config = kwargs.pop("config")
    if overrides is not None:
        kwargs["test_simulation_replicates_override"] = overrides[0]
        kwargs["test_bootstrap_replicates_override"] = overrides[1]
    return pre_unblinding_gate(pred, ref, reliability_frame, config, **kwargs)


def _fake_power(*_args: object, **_kwargs: object) -> dict[str, object]:
    return {
        "executed": True,
        "simulation_replicates": 1000,
        "whole_block_bootstrap_replicates_per_simulation": 500,
        "simulation_seed": 20260711,
        "dgp_means": {"roof": 0.6, "facade": 0.4},
        "within_block_intraclass_correlation": 0.05,
        "observed_layout_pair_count": 6,
        "observed_layout_block_count": 6,
        "successful_simulations": 1000,
        "estimated_power": 1.0,
        "finite_bootstrap_replicates_min": 500,
        "finite_bootstrap_replicates_max": 500,
    }


def passing_confirmation_context(
    *, mechanism_consistent: bool
) -> tuple[dict[str, object], dict[str, object]]:
    development, bundle = paired_role_bundles(
        mechanism_consistent=mechanism_consistent
    )
    with patch(
        "crossview_conflict.analysis.eaton_component_gate._simulate_frozen_power",
        side_effect=_fake_power,
    ):
        development_gate = run_gate(development)
    authorization = build_confirmation_authorization_artifact(
        development_gate,
        expected_protocol_summary_sha256=development[
            "expected_protocol_summary_sha256"
        ],
        expected_prediction_csv_sha256=development[
            "expected_prediction_csv_sha256"
        ],
        expected_prediction_metadata_sha256=development[
            "expected_prediction_metadata_sha256"
        ],
        protocol_summary_bytes=development["protocol_summary_bytes"],
    )
    bundle["development_pass_authorization"] = authorization
    with patch(
        "crossview_conflict.analysis.eaton_component_gate._simulate_frozen_power",
        side_effect=_fake_power,
    ):
        gate = run_gate(bundle)
    assert gate["gate_status"] == PASS_PRE_UNBLIND_GATE
    return gate, bundle


def build_bound_analysis(
    gate: dict[str, object], bundle: dict[str, object]
) -> dict[str, object]:
    # Exercise the real estimator on a small frozen bootstrap, then inject that
    # row at the analysis boundary.  The gate still verifies the registered
    # 10k request, seed, counts, weights, identity, and artifact bindings.
    small_analysis = analyze_component_direction(
        bundle["predictions"],
        bundle["reference"],
        bootstrap_replicates=20,
        bootstrap_seed=20_260_711,
        expected_role="spatial_confirmation",
    )
    small_analysis.statistics.loc[0, "bootstrap_replicates_requested"] = 10_000
    with patch(
        "crossview_conflict.analysis.eaton_component_gate.analyze_component_direction",
        return_value=small_analysis,
    ):
        return build_confirmation_analysis_artifact(
            gate,
            bundle["predictions"],
            bundle["reference"],
            bundle["config"],
            expected_protocol_sha256=bundle["expected_protocol_sha256"],
            expected_protocol_summary_sha256=bundle[
                "expected_protocol_summary_sha256"
            ],
            expected_prediction_csv_sha256=bundle[
                "expected_prediction_csv_sha256"
            ],
            expected_prediction_metadata_sha256=bundle[
                "expected_prediction_metadata_sha256"
            ],
            protocol_config_bytes=bundle["protocol_config_bytes"],
            prediction_csv_bytes=bundle["prediction_csv_bytes"],
            prediction_metadata_bytes=bundle["prediction_metadata_bytes"],
            role_manifest_csv_bytes=bundle["role_manifest_csv_bytes"],
            protocol_summary_bytes=bundle["protocol_summary_bytes"],
        )


def decision_kwargs(bundle: dict[str, object]) -> dict[str, str]:
    return {
        "expected_protocol_summary_sha256": str(
            bundle["expected_protocol_summary_sha256"]
        ),
        "expected_prediction_csv_sha256": str(
            bundle["expected_prediction_csv_sha256"]
        ),
        "expected_prediction_metadata_sha256": str(
            bundle["expected_prediction_metadata_sha256"]
        ),
    }


def futility_predictions(*, block_count: int = 28, pair_count: int = 280) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in (42, 123, 456, 789, 1011):
        for index in range(pair_count):
            row: dict[str, object] = {
                "pair_id": f"futility-{index:04d}",
                "seed": seed,
                "spatial_block_id": f"block-{index % block_count:03d}",
                "protocol_role": "study_development",
                "target": index % 3,
                "class_order": json.dumps(list(SEVERITY_CLASS_ORDER)),
            }
            for column in PROBABILITY_COLUMNS:
                row[column] = 0.05
            row["street_prob_no_or_trace_damage"] = 0.90
            row["overhead_prob_damaged_repairable"] = 0.90
            rows.append(row)
    return pd.DataFrame(rows)


def run_futility(bundle: dict[str, object]) -> dict[str, object]:
    return pre_annotation_futility_gate(
        bundle["predictions"],
        bundle["config"],
        expected_protocol_sha256=bundle["expected_protocol_sha256"],
        expected_protocol_summary_sha256=bundle[
            "expected_protocol_summary_sha256"
        ],
        expected_prediction_csv_sha256=bundle["expected_prediction_csv_sha256"],
        expected_prediction_metadata_sha256=bundle[
            "expected_prediction_metadata_sha256"
        ],
        protocol_config_bytes=bundle["protocol_config_bytes"],
        prediction_csv_bytes=bundle["prediction_csv_bytes"],
        prediction_metadata_bytes=bundle["prediction_metadata_bytes"],
        role_manifest_csv_bytes=bundle["role_manifest_csv_bytes"],
        protocol_summary_bytes=bundle["protocol_summary_bytes"],
    )


class PreAnnotationFutilityTests(unittest.TestCase):
    def test_twenty_eight_blocks_stops_before_annotation(self) -> None:
        bundle = gate_bundle(prediction_frame=futility_predictions())
        result = run_futility(bundle)
        self.assertEqual(result["phase_status"], STOP_PRE_ANNOTATION_FUTILITY)
        self.assertEqual(result["gate_status"], STOP_PRE_UNBLIND_GATE)
        self.assertFalse(result["annotation_required"])
        self.assertIsNone(result["h_b1_result"])
        self.assertIsNone(result["h_b2_result"])
        self.assertFalse(result["directional_hypotheses_evaluated"])
        self.assertEqual(result["unsigned_disagreement_spatial_block_count"], 28)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn(
            "role_spatial_block_capacity_below_registered_minimum", codes
        )
        self.assertIn(
            "disagreement_block_capacity_below_registered_minimum", codes
        )

    def test_thirty_blocks_can_continue_to_blinded_annotation(self) -> None:
        bundle = gate_bundle(
            prediction_frame=futility_predictions(block_count=30, pair_count=300)
        )
        result = run_futility(bundle)
        self.assertEqual(result["phase_status"], CONTINUE_TO_BLINDED_ANNOTATION)
        self.assertTrue(result["annotation_required"])

    def test_truncation_and_all_internal_resigning_cannot_move_trust_root(self) -> None:
        bundle = gate_bundle(prediction_frame=futility_predictions())
        truncated = bundle["predictions"].query(
            "pair_id != 'futility-0000'"
        ).copy()
        manifest = pd.read_csv(io.BytesIO(bundle["role_manifest_csv_bytes"])).query(
            "pair_id != 'futility-0000'"
        )
        prediction_bytes = _csv_bytes(truncated)
        manifest_bytes = _csv_bytes(manifest)
        metadata = json.loads(bundle["prediction_metadata_bytes"].decode("utf-8"))
        metadata["prediction_csv_sha256"] = _sha_bytes(prediction_bytes)
        metadata["role_manifest_sha256"]["study_development"] = _sha_bytes(
            manifest_bytes
        )
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        summary = json.loads(bundle["protocol_summary_bytes"].decode("utf-8"))
        summary["role_manifest_sha256"]["study_development"] = _sha_bytes(
            manifest_bytes
        )
        summary["role_rows"]["study_development"] = len(manifest)
        bundle.update(
            {
                "predictions": truncated,
                "prediction_csv_bytes": prediction_bytes,
                "prediction_metadata_bytes": metadata_bytes,
                "role_manifest_csv_bytes": manifest_bytes,
                "protocol_summary_bytes": json.dumps(summary).encode("utf-8"),
            }
        )
        with self.assertRaisesRegex(ValueError, "expected_prediction_csv_sha256"):
            run_futility(bundle)

    def test_summary_resigning_is_rejected_by_external_commitment(self) -> None:
        bundle = gate_bundle(prediction_frame=futility_predictions())
        summary = json.loads(bundle["protocol_summary_bytes"].decode("utf-8"))
        summary["role_rows"]["spatial_confirmation"] += 1
        bundle["protocol_summary_bytes"] = json.dumps(summary).encode("utf-8")
        with self.assertRaisesRegex(
            ValueError, "expected_protocol_summary_sha256"
        ):
            run_futility(bundle)

    def test_prediction_and_metadata_resigning_is_rejected(self) -> None:
        bundle = gate_bundle(prediction_frame=futility_predictions())
        altered = bundle["predictions"].copy()
        mask = altered["pair_id"].eq("futility-0000")
        altered.loc[mask, "street_prob_no_or_trace_damage"] = 0.05
        altered.loc[mask, "street_prob_damaged_repairable"] = 0.90
        prediction_bytes = _csv_bytes(altered)
        metadata = json.loads(bundle["prediction_metadata_bytes"].decode("utf-8"))
        metadata["prediction_csv_sha256"] = _sha_bytes(prediction_bytes)
        bundle["predictions"] = altered
        bundle["prediction_csv_bytes"] = prediction_bytes
        bundle["prediction_metadata_bytes"] = json.dumps(metadata).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "expected_prediction_csv_sha256"):
            run_futility(bundle)


class PreUnblindingGateTests(unittest.TestCase):
    def test_gate_is_invariant_to_observed_prediction_order(self) -> None:
        config = protocol_config(small_count_thresholds=True)
        forward = run_gate(gate_bundle(config=config), overrides=(12, 30))
        reverse = run_gate(
            gate_bundle(
                config=config,
                prediction_frame=predictions(reverse_order=True),
            ),
            overrides=(12, 30),
        )
        # Artifact hashes differ because probabilities differ; all signless audits/power do not.
        for key in ("annotation_reliability", "eligible_layout", "power_simulation"):
            self.assertEqual(forward[key], reverse[key])
        self.assertFalse(forward["observed_outcome_accessed"])

    def test_precomputed_sign_leakage_column_is_rejected(self) -> None:
        leaked = predictions()
        leaked["signed_direction"] = 1
        bundle = gate_bundle(prediction_frame=leaked)
        with self.assertRaisesRegex(ValueError, "forbidden before unblinding"):
            run_gate(bundle)

    def test_count_failure_is_stop_not_no_go(self) -> None:
        result = run_gate(gate_bundle(), overrides=(5, 10))
        self.assertEqual(result["gate_status"], STOP_PRE_UNBLIND_GATE)
        codes = {reason["code"] for reason in result["reasons"]}
        self.assertIn("mechanism_eligible_count_below_minimum", codes)
        self.assertFalse(result["power_simulation"]["executed"])

    def test_genuine_raw_reliability_failure_is_stop(self) -> None:
        result = run_gate(
            gate_bundle(low_reliability=True), overrides=(5, 10)
        )
        self.assertEqual(result["gate_status"], STOP_PRE_UNBLIND_GATE)
        self.assertFalse(result["annotation_reliability"]["passed"])

    def test_free_reliability_table_tamper_is_rejected(self) -> None:
        bundle = gate_bundle()
        tampered = bundle["reliability"].copy()
        tampered.loc[0, "observed_agreement"] = 0.123
        bundle["reliability"] = tampered
        with self.assertRaisesRegex(ValueError, "exactly reproduce"):
            run_gate(bundle, overrides=(5, 10))

    def test_protocol_mapping_must_match_bound_raw_hash(self) -> None:
        bundle = gate_bundle()
        bundle["config"] = copy.deepcopy(bundle["config"])
        bundle["config"]["event_id"] = "tampered"
        with self.assertRaisesRegex(ValueError, "does not match the bound raw config"):
            run_gate(bundle, overrides=(5, 10))

    def test_truncated_prediction_population_is_rejected_even_if_metadata_updated(self) -> None:
        bundle = gate_bundle()
        truncated = bundle["predictions"].query("pair_id != 'roof-0'").copy()
        truncated_bytes = _csv_bytes(truncated)
        metadata = json.loads(bundle["prediction_metadata_bytes"].decode("utf-8"))
        metadata["prediction_csv_sha256"] = _sha_bytes(truncated_bytes)
        bundle["predictions"] = truncated
        bundle["prediction_csv_bytes"] = truncated_bytes
        bundle["prediction_metadata_bytes"] = json.dumps(metadata).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "expected_prediction_csv_sha256"):
            run_gate(bundle, overrides=(5, 10))

    def test_arbitrary_truncated_cohort_cannot_replace_frozen_summary(self) -> None:
        bundle = gate_bundle()
        truncated = bundle["predictions"].query("pair_id != 'roof-0'").copy()
        manifest = pd.read_csv(io.BytesIO(bundle["role_manifest_csv_bytes"])).query(
            "pair_id != 'roof-0'"
        )
        prediction_bytes = _csv_bytes(truncated)
        manifest_bytes = _csv_bytes(manifest)
        metadata = json.loads(bundle["prediction_metadata_bytes"].decode("utf-8"))
        metadata["prediction_csv_sha256"] = _sha_bytes(prediction_bytes)
        metadata["role_manifest_sha256"]["study_development"] = _sha_bytes(
            manifest_bytes
        )
        bundle["predictions"] = truncated
        bundle["prediction_csv_bytes"] = prediction_bytes
        bundle["role_manifest_csv_bytes"] = manifest_bytes
        bundle["prediction_metadata_bytes"] = json.dumps(metadata).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "expected_prediction_csv_sha256"):
            run_gate(bundle, overrides=(5, 10))

    def test_reference_and_raw_annotation_aggregate_are_bound(self) -> None:
        bundle = gate_bundle()
        bundle["reference"] = bundle["reference"].query("pair_id != 'roof-0'")
        with self.assertRaisesRegex(ValueError, "exactly reproduce"):
            run_gate(bundle, overrides=(5, 10))

    def test_test_repetition_override_can_never_pass(self) -> None:
        result = run_gate(
            gate_bundle(config=protocol_config(small_count_thresholds=True)),
            overrides=(5, 10),
        )
        self.assertEqual(result["execution_mode"], NON_PROTOCOL_TEST_OVERRIDE)
        self.assertEqual(result["gate_status"], STOP_PRE_UNBLIND_GATE)


class ConfirmationDecisionTests(unittest.TestCase):
    def test_confirmation_gate_refuses_access_without_development_authorization(self) -> None:
        _development, confirmation = paired_role_bundles(
            mechanism_consistent=True
        )
        with patch(
            "crossview_conflict.analysis.eaton_component_gate._simulate_frozen_power",
            side_effect=_fake_power,
        ):
            with self.assertRaisesRegex(ValueError, "authorization artifact is required"):
                run_gate(confirmation)

    def test_stopped_development_status_and_digest_resigning_cannot_authorize(self) -> None:
        development, _confirmation = paired_role_bundles(
            mechanism_consistent=True
        )
        stopped = run_gate(development, overrides=(5, 10))
        forged = copy.deepcopy(stopped)
        forged["gate_status"] = PASS_PRE_UNBLIND_GATE
        _resign(forged, "gate_artifact_sha256")
        with self.assertRaisesRegex(ValueError, "non-protocol test override"):
            build_confirmation_authorization_artifact(
                forged,
                expected_protocol_summary_sha256=development[
                    "expected_protocol_summary_sha256"
                ],
                expected_prediction_csv_sha256=development[
                    "expected_prediction_csv_sha256"
                ],
                expected_prediction_metadata_sha256=development[
                    "expected_prediction_metadata_sha256"
                ],
                protocol_summary_bytes=development["protocol_summary_bytes"],
            )

    def test_outer_resigning_cannot_hide_revoked_development_parent(self) -> None:
        gate, bundle = passing_confirmation_context(mechanism_consistent=True)
        analysis = build_bound_analysis(gate, bundle)
        forged = copy.deepcopy(gate)
        authorization = forged["development_pass_authorization"]
        parent = authorization["development_gate"]
        parent["gate_status"] = STOP_PRE_UNBLIND_GATE
        _resign(parent, "gate_artifact_sha256")
        authorization["development_gate_artifact_sha256"] = parent[
            "gate_artifact_sha256"
        ]
        _resign(authorization, "authorization_artifact_sha256")
        forged["development_authorization_artifact_sha256"] = authorization[
            "authorization_artifact_sha256"
        ]
        _resign(forged, "gate_artifact_sha256")
        with self.assertRaisesRegex(ValueError, "requires PASS_PRE_UNBLIND_GATE"):
            confirmation_decision(forged, analysis, **decision_kwargs(bundle))

    def test_decision_rechecks_external_protocol_summary_anchor(self) -> None:
        gate, bundle = passing_confirmation_context(mechanism_consistent=True)
        analysis = build_bound_analysis(gate, bundle)
        kwargs = decision_kwargs(bundle)
        kwargs["expected_protocol_summary_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "expected_protocol_summary_sha256"):
            confirmation_decision(gate, analysis, **kwargs)

    def test_failed_gate_status_flip_cannot_forge_go(self) -> None:
        bundle = gate_bundle(config=protocol_config(small_count_thresholds=True))
        stopped = run_gate(bundle, overrides=(5, 10))
        forged = copy.deepcopy(stopped)
        forged["gate_status"] = PASS_PRE_UNBLIND_GATE
        with self.assertRaisesRegex(ValueError, "digest"):
            confirmation_decision(forged, {}, **decision_kwargs(bundle))

    def test_development_gate_cannot_authorize_confirmation(self) -> None:
        bundle = gate_bundle(config=protocol_config(small_count_thresholds=True))
        with patch(
            "crossview_conflict.analysis.eaton_component_gate._simulate_frozen_power",
            side_effect=_fake_power,
        ):
            gate = run_gate(bundle)
        self.assertEqual(gate["gate_status"], PASS_PRE_UNBLIND_GATE)
        with self.assertRaisesRegex(ValueError, "spatial_confirmation gate"):
            confirmation_decision(gate, {}, **decision_kwargs(bundle))

    def test_legacy_raw_statistics_are_rejected(self) -> None:
        gate, bundle = passing_confirmation_context(mechanism_consistent=True)
        with self.assertRaisesRegex(ValueError, "wrong schema"):
            confirmation_decision(
                gate,
                {
                    "protocol_role": "spatial_confirmation",
                    "h_b1_directional_contrast": 1.0,
                },
                **decision_kwargs(bundle),
            )

    def test_bootstrap_seed_and_repetitions_are_frozen(self) -> None:
        config = protocol_config(small_count_thresholds=True)
        config["primary_estimand"]["bootstrap_seed"] = 7
        with self.assertRaisesRegex(ValueError, "10000 whole-block bootstraps"):
            run_gate(gate_bundle(config=config))

    def test_all_registered_conditions_produce_go(self) -> None:
        gate, bundle = passing_confirmation_context(mechanism_consistent=True)
        analysis = build_bound_analysis(gate, bundle)
        result = confirmation_decision(gate, analysis, **decision_kwargs(bundle))
        self.assertEqual(result["decision"], GO_CONFIRMATION)
        self.assertEqual(result["failed_criteria"], [])

    def test_any_failed_primary_condition_produces_no_go(self) -> None:
        gate, bundle = passing_confirmation_context(mechanism_consistent=False)
        analysis = build_bound_analysis(gate, bundle)
        result = confirmation_decision(gate, analysis, **decision_kwargs(bundle))
        self.assertEqual(result["decision"], NO_GO_CONFIRMATION)
        self.assertIn("standardized_delta_at_least_minimum", result["failed_criteria"])

    def test_analysis_binding_tamper_is_rejected(self) -> None:
        gate, bundle = passing_confirmation_context(mechanism_consistent=True)
        analysis = build_bound_analysis(gate, bundle)
        forged = copy.deepcopy(analysis)
        forged["artifact_binding"]["reference_table_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            confirmation_decision(gate, forged, **decision_kwargs(bundle))


if __name__ == "__main__":
    unittest.main()
