"""Fail-closed contracts for view-conditional attestation research.

This module validates provenance-bearing annotation and prediction tables.  It
does not infer labels, train a model, or convert inspector material/construction
attributes into damage targets.  Synthetic fixtures are accepted only through
the explicit ``synthetic_test`` path and can never pass claim validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


CONTRACT_SCHEMA_VERSION = "crossviewguard-attestation-contract-v1"
ANNOTATION_SCHEMA_VERSION = "crossviewguard-attestation-annotation-v1"
PREDICTION_SCHEMA_VERSION = "crossviewguard-attestation-prediction-v1"
INVENTORY_SCHEMA_VERSION = "crossviewguard-attestation-dependency-inventory-v1"
VALIDATION_MODES = frozenset({"claim", "synthetic_test"})
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AttestationGateDecision:
    """Go/No-Go decisions and unmet requirements for one dataset."""

    dataset: str
    decisions: Mapping[str, bool]
    blockers: Mapping[str, tuple[str, ...]]

    @property
    def rq2_matrix_go(self) -> bool:
        return bool(self.decisions["rq2_matrix"])

    @property
    def rq2_geometry_go(self) -> bool:
        return bool(self.decisions["rq2_geometry"])

    @property
    def rq3_coverage_go(self) -> bool:
        return bool(self.decisions["rq3_coverage"])


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _require_string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{key} must be a non-empty list of unique strings")
    return value


def validate_attestation_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the versioned annotation/prediction and dependency contract."""

    payload = dict(contract)
    if payload.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"Expected contract schema {CONTRACT_SCHEMA_VERSION!r}, got "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("annotation_schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise ValueError("Unexpected annotation schema version")
    if payload.get("prediction_schema_version") != PREDICTION_SCHEMA_VERSION:
        raise ValueError("Unexpected prediction schema version")

    for key in (
        "allowed_record_origins",
        "allowed_split_roles",
        "allowed_view_types",
        "allowed_observability_states",
        "allowed_reference_source_types",
        "annotation_required_columns",
        "prediction_required_columns",
        "forbidden_reference_source_fields",
        "forbidden_reference_tokens",
    ):
        _require_string_list(payload, key)
    required_origins = {"real_annotation", "real_model_prediction", "synthetic_fixture"}
    if not required_origins.issubset(payload["allowed_record_origins"]):
        raise ValueError(
            "allowed_record_origins must include real annotations, real predictions, "
            "and synthetic fixtures"
        )

    fields = payload.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("fields must be a non-empty list")
    field_ids: list[str] = []
    allowed_semantics = {"component_damage_state", "ordinal_damage_severity"}
    for field in fields:
        if not isinstance(field, dict):
            raise ValueError("Every field definition must be an object")
        field_id = field.get("field_id")
        if not isinstance(field_id, str) or not field_id.strip():
            raise ValueError("Every field requires a non-empty field_id")
        field_ids.append(field_id)
        if field.get("reference_semantics") not in allowed_semantics:
            raise ValueError(f"Unsupported reference semantics for {field_id}")
        reference_states = _require_string_list(field, "reference_states")
        observation_states = _require_string_list(field, "view_observation_states")
        if "unknown" not in reference_states:
            raise ValueError(f"{field_id} must preserve an explicit unknown state")
        if "abstain" not in observation_states:
            raise ValueError(f"{field_id} must support abstention")
    if len(set(field_ids)) != len(field_ids):
        raise ValueError("field_id values must be unique")

    required_annotation = set(payload["annotation_required_columns"])
    annotation_minimum = {
        "schema_version",
        "entity_id",
        "event_id",
        "dependency_group_id",
        "split_role",
        "view_id",
        "view_type",
        "field_id",
        "reference_semantics",
        "reference_state",
        "view_observation_state",
        "observability",
        "reference_source_type",
        "reference_source_field",
        "reference_provenance",
        "view_annotation_provenance",
        "record_origin",
        "source_artifact_sha256",
        "media_sha256",
    }
    if missing := annotation_minimum - required_annotation:
        raise ValueError(f"Contract omits required annotation columns: {sorted(missing)}")

    required_prediction = set(payload["prediction_required_columns"])
    prediction_minimum = {
        "schema_version",
        "entity_id",
        "event_id",
        "dependency_group_id",
        "split_role",
        "seed",
        "view_id",
        "view_type",
        "field_id",
        "attestation_probability",
        "abstain_probability",
        "predicted_observation_state",
        "record_origin",
        "model_artifact_sha256",
        "annotation_artifact_sha256",
    }
    if missing := prediction_minimum - required_prediction:
        raise ValueError(f"Contract omits required prediction columns: {sorted(missing)}")

    gates = payload.get("dependency_gates")
    if not isinstance(gates, dict) or set(gates) != {
        "rq2_matrix",
        "rq2_geometry",
        "rq3_coverage",
    }:
        raise ValueError("dependency_gates must define exactly RQ2 matrix/geometry and RQ3")
    for gate_name, requirements in gates.items():
        if (
            not isinstance(requirements, list)
            or not requirements
            or any(not isinstance(item, str) or not item for item in requirements)
            or len(set(requirements)) != len(requirements)
        ):
            raise ValueError(f"Invalid dependency requirements for {gate_name}")

    forbidden_fields = {value.casefold() for value in payload["forbidden_reference_source_fields"]}
    if "dins_wherefirestartedonstructure" not in forbidden_fields:
        raise ValueError("The fire-origin source field must be explicitly forbidden")
    return payload


def load_attestation_contract(path: str | Path) -> dict[str, Any]:
    return validate_attestation_contract(_load_json(path))


def validate_dependency_inventory(
    inventory: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate an evidence ledger without interpreting missing inputs as false labels."""

    contract_payload = validate_attestation_contract(contract)
    payload = dict(inventory)
    if payload.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise ValueError(
            f"Expected dependency inventory schema {INVENTORY_SCHEMA_VERSION!r}"
        )
    evidence_basis = payload.get("evidence_basis")
    if not isinstance(evidence_basis, list) or not evidence_basis:
        raise ValueError("Dependency inventory requires an evidence_basis list")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Dependency inventory requires at least one dataset")
    required_flags = {
        requirement
        for requirements in contract_payload["dependency_gates"].values()
        for requirement in requirements
    }
    for dataset, entry in datasets.items():
        if not isinstance(dataset, str) or not dataset or not isinstance(entry, dict):
            raise ValueError("Invalid dataset entry in dependency inventory")
        requirements = entry.get("requirements")
        if not isinstance(requirements, dict):
            raise ValueError(f"{dataset} is missing a requirements object")
        if missing := required_flags - set(requirements):
            raise ValueError(f"{dataset} is missing dependency flags: {sorted(missing)}")
        non_boolean = [key for key in required_flags if not isinstance(requirements[key], bool)]
        if non_boolean:
            raise ValueError(f"{dataset} has non-boolean dependency flags: {non_boolean}")
    return payload


def load_dependency_inventory(
    path: str | Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    return validate_dependency_inventory(_load_json(path), contract)


def evaluate_attestation_dependency_gates(
    contract: Mapping[str, Any], inventory: Mapping[str, Any], dataset: str
) -> AttestationGateDecision:
    """Evaluate strict upstream dependencies for RQ2 and RQ3."""

    contract_payload = validate_attestation_contract(contract)
    inventory_payload = validate_dependency_inventory(inventory, contract_payload)
    if dataset not in inventory_payload["datasets"]:
        raise KeyError(f"Dataset {dataset!r} is not present in the dependency inventory")
    evidence = inventory_payload["datasets"][dataset]["requirements"]
    decisions: dict[str, bool] = {}
    blockers: dict[str, tuple[str, ...]] = {}
    for gate_name in ("rq2_matrix", "rq2_geometry", "rq3_coverage"):
        unmet = [
            requirement
            for requirement in contract_payload["dependency_gates"][gate_name]
            if not evidence[requirement]
        ]
        if gate_name != "rq2_matrix" and not decisions["rq2_matrix"]:
            unmet.insert(0, "upstream_rq2_matrix_no_go")
        blockers[gate_name] = tuple(dict.fromkeys(unmet))
        decisions[gate_name] = not blockers[gate_name]
    return AttestationGateDecision(dataset, decisions, blockers)


def _as_frame(records: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(records, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if records.empty:
        raise ValueError(f"{name} must contain at least one row")
    return records.copy()


def _require_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    if missing := set(required) - set(frame.columns):
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _require_nonempty_text(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        values = frame[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"{name}.{column} contains null or blank values")


def _require_hashes(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        invalid = ~frame[column].astype(str).map(
            lambda value: bool(HEX_SHA256.fullmatch(value))
        )
        if invalid.any():
            raise ValueError(f"{name}.{column} must contain lowercase SHA-256 values")


def _validate_mode_origins(frame: pd.DataFrame, mode: str, expected_real: str) -> None:
    if mode not in VALIDATION_MODES:
        raise ValueError(f"mode must be one of {sorted(VALIDATION_MODES)}")
    origins = set(frame["record_origin"].astype(str))
    expected = {"synthetic_fixture"} if mode == "synthetic_test" else {expected_real}
    if origins != expected:
        if mode == "claim" and "synthetic_fixture" in origins:
            raise ValueError("Synthetic fixtures are forbidden in claim mode")
        raise ValueError(f"{mode} mode requires record_origin={sorted(expected)}")


def _validate_group_isolation(frame: pd.DataFrame, name: str) -> None:
    entity_roles = frame.groupby("entity_id", dropna=False)["split_role"].nunique()
    if (entity_roles > 1).any():
        raise ValueError(f"{name} assigns an entity to multiple split roles")
    group_roles = frame.groupby("dependency_group_id", dropna=False)["split_role"].nunique()
    if (group_roles > 1).any():
        raise ValueError(f"{name} assigns a dependency group to multiple split roles")
    entity_events = frame.groupby("entity_id", dropna=False)["event_id"].nunique()
    if (entity_events > 1).any():
        raise ValueError(f"{name} assigns an entity to multiple events")


def _field_definitions(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {field["field_id"]: field for field in contract["fields"]}


def validate_attestation_annotations(
    records: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    mode: str = "claim",
) -> pd.DataFrame:
    """Validate long-form view-by-field annotations.

    ``synthetic_test`` is the only mode accepting synthetic fixtures.  Claim
    mode requires provenance-bearing real annotations and rejects source fields
    that encode construction/material/exposure or fire origin.
    """

    contract_payload = validate_attestation_contract(contract)
    frame = _as_frame(records, "annotations")
    _require_columns(frame, contract_payload["annotation_required_columns"], "annotations")
    text_columns = [
        column
        for column in contract_payload["annotation_required_columns"]
        if column not in set()
    ]
    _require_nonempty_text(frame, text_columns, "annotations")
    if not frame["schema_version"].eq(ANNOTATION_SCHEMA_VERSION).all():
        raise ValueError("annotations contain an unexpected schema_version")
    _validate_mode_origins(frame, mode, "real_annotation")
    _require_hashes(frame, ["source_artifact_sha256", "media_sha256"], "annotations")

    if frame.duplicated(["entity_id", "view_id", "field_id"]).any():
        raise ValueError("annotations must be unique on (entity_id, view_id, field_id)")
    if not frame["split_role"].isin(contract_payload["allowed_split_roles"]).all():
        raise ValueError("annotations contain an unsupported split_role")
    if not frame["view_type"].isin(contract_payload["allowed_view_types"]).all():
        raise ValueError("annotations contain an unsupported view_type")
    if not frame["observability"].isin(
        contract_payload["allowed_observability_states"]
    ).all():
        raise ValueError("annotations contain an unsupported observability state")
    if not frame["reference_source_type"].isin(
        contract_payload["allowed_reference_source_types"]
    ).all():
        raise ValueError("annotations contain an unsupported reference_source_type")

    if mode == "synthetic_test":
        if not frame["reference_source_type"].eq("synthetic_fixture").all():
            raise ValueError("Synthetic fixtures require reference_source_type=synthetic_fixture")
    elif frame["reference_source_type"].eq("synthetic_fixture").any():
        raise ValueError("Synthetic reference sources are forbidden in claim mode")

    forbidden_fields = {
        value.casefold()
        for value in contract_payload["forbidden_reference_source_fields"]
    }
    source_fields = frame["reference_source_field"].astype(str).str.casefold()
    if source_fields.isin(forbidden_fields).any():
        raise ValueError(
            "Construction/material or fire-origin fields cannot be damage references"
        )
    forbidden_tokens = tuple(
        token.casefold() for token in contract_payload["forbidden_reference_tokens"]
    )
    if source_fields.map(lambda value: any(token in value for token in forbidden_tokens)).any():
        raise ValueError(
            "Construction/material or fire-origin source semantics are forbidden"
        )

    definitions = _field_definitions(contract_payload)
    if not frame["field_id"].isin(definitions).all():
        raise ValueError("annotations contain a field outside the frozen field whitelist")
    for field_id, subset in frame.groupby("field_id", sort=False):
        definition = definitions[str(field_id)]
        if not subset["reference_semantics"].eq(
            definition["reference_semantics"]
        ).all():
            raise ValueError(f"{field_id} has invalid reference semantics")
        if not subset["reference_state"].isin(definition["reference_states"]).all():
            raise ValueError(f"{field_id} has an invalid reference_state")
        if not subset["view_observation_state"].isin(
            definition["view_observation_states"]
        ).all():
            raise ValueError(f"{field_id} has an invalid view_observation_state")

    non_visible = ~frame["observability"].eq("visible")
    if not frame.loc[non_visible, "view_observation_state"].eq("abstain").all():
        raise ValueError("Not-visible or indeterminate views must abstain")
    unknown_reference = frame["reference_state"].eq("unknown")
    if not frame.loc[unknown_reference, "view_observation_state"].eq("abstain").all():
        raise ValueError("Unknown references cannot supervise a non-abstaining observation")
    _validate_group_isolation(frame, "annotations")
    return frame


def validate_attestation_predictions(
    records: pd.DataFrame,
    contract: Mapping[str, Any],
    *,
    mode: str = "claim",
) -> pd.DataFrame:
    """Validate calibrated RQ2 matrix outputs before any RQ3 consumer reads them."""

    contract_payload = validate_attestation_contract(contract)
    frame = _as_frame(records, "predictions")
    _require_columns(frame, contract_payload["prediction_required_columns"], "predictions")
    numeric_columns = {"seed", "attestation_probability", "abstain_probability"}
    text_columns = [
        column
        for column in contract_payload["prediction_required_columns"]
        if column not in numeric_columns
    ]
    _require_nonempty_text(frame, text_columns, "predictions")
    if not frame["schema_version"].eq(PREDICTION_SCHEMA_VERSION).all():
        raise ValueError("predictions contain an unexpected schema_version")
    _validate_mode_origins(frame, mode, "real_model_prediction")
    _require_hashes(
        frame,
        ["model_artifact_sha256", "annotation_artifact_sha256"],
        "predictions",
    )
    if frame.duplicated(["seed", "entity_id", "view_id", "field_id"]).any():
        raise ValueError(
            "predictions must be unique on (seed, entity_id, view_id, field_id)"
        )
    if not frame["split_role"].isin(contract_payload["allowed_split_roles"]).all():
        raise ValueError("predictions contain an unsupported split_role")
    if not frame["view_type"].isin(contract_payload["allowed_view_types"]).all():
        raise ValueError("predictions contain an unsupported view_type")
    definitions = _field_definitions(contract_payload)
    if not frame["field_id"].isin(definitions).all():
        raise ValueError("predictions contain a field outside the frozen field whitelist")
    for field_id, subset in frame.groupby("field_id", sort=False):
        allowed = definitions[str(field_id)]["view_observation_states"]
        if not subset["predicted_observation_state"].isin(allowed).all():
            raise ValueError(f"{field_id} has an invalid predicted_observation_state")

    seed = pd.to_numeric(frame["seed"], errors="coerce")
    if (
        seed.isna().any()
        or not np.isfinite(seed).all()
        or not np.equal(seed, np.floor(seed)).all()
    ):
        raise ValueError("prediction seeds must be finite integers")
    attestation = pd.to_numeric(frame["attestation_probability"], errors="coerce")
    abstain = pd.to_numeric(frame["abstain_probability"], errors="coerce")
    if (
        attestation.isna().any()
        or abstain.isna().any()
        or (~attestation.between(0.0, 1.0)).any()
        or (~abstain.between(0.0, 1.0)).any()
    ):
        raise ValueError("Attestation and abstain probabilities must lie in [0, 1]")
    if ((attestation + abstain) > 1.0 + 1e-8).any():
        raise ValueError("Attestation and abstain probabilities cannot sum above one")
    _validate_group_isolation(frame, "predictions")
    return frame


def expected_attestation_coverage_gain(
    current_attested: np.ndarray,
    candidate_attestation_probability: np.ndarray,
    field_weights: np.ndarray,
    *,
    acquisition_cost: float,
) -> np.ndarray:
    """Compute one-step expected *new* weighted coverage for plumbing tests.

    This pure function does not make an RQ3 result valid.  Real use is gated on
    calibrated RQ2 predictions and frozen decision-field weights.
    """

    current = np.asarray(current_attested)
    candidate = np.asarray(candidate_attestation_probability, dtype=np.float64)
    weights = np.asarray(field_weights, dtype=np.float64).reshape(-1)
    if current.ndim != 2 or candidate.shape != current.shape:
        raise ValueError("current and candidate coverage must be matching 2-D arrays")
    if weights.shape != (current.shape[1],):
        raise ValueError("field_weights must have one entry per field")
    if not np.isin(current, [0, 1, False, True]).all():
        raise ValueError("current_attested must be binary")
    if (
        not np.isfinite(candidate).all()
        or (candidate < 0).any()
        or (candidate > 1).any()
    ):
        raise ValueError("candidate attestation probabilities must lie in [0, 1]")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("field_weights must be finite and non-negative")
    if not np.isfinite(acquisition_cost) or acquisition_cost < 0:
        raise ValueError("acquisition_cost must be finite and non-negative")
    newly_attestable = (~current.astype(bool)).astype(np.float64) * candidate
    return newly_attestable @ weights - float(acquisition_cost)
