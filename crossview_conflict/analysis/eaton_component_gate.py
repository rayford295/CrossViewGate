"""Fail-closed pre-unblinding gate for the focused Eaton experiment.

The gate is deliberately outcome-blind.  It is allowed to determine whether
the two seed-ensembled views disagree, but it never orders the two predictions
and never computes an observed directional outcome.  Its only pair-level
inputs to the frozen power calculation are target, spatial block, and the
adjudicated mechanism-eligibility stratum.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .eaton_component_direction import (
    ANNOTATION_FIELDS,
    PROTOCOL_ROLE,
    SEVERITY_CLASS_ORDER,
    analyze_component_direction,
    build_adjudicated_reference,
    compute_annotation_reliability,
    ensemble_seed_predictions,
    validate_annotation_packet,
    validate_component_reference,
    validate_seed_predictions,
)


PASS_PRE_UNBLIND_GATE = "PASS_PRE_UNBLIND_GATE"
STOP_PRE_UNBLIND_GATE = "STOP_UNDERPOWERED_OR_UNRELIABLE_CONSTRUCT"
CONTINUE_TO_BLINDED_ANNOTATION = "CONTINUE_TO_BLINDED_ANNOTATION"
STOP_PRE_ANNOTATION_FUTILITY = "STOP_PRE_ANNOTATION_STRUCTURAL_FUTILITY"
GO_CONFIRMATION = "GO_FOR_REGISTERED_EATON_IMAGE_MECHANISM"
NO_GO_CONFIRMATION = "NO_GO_FOR_REGISTERED_EATON_IMAGE_MECHANISM"
PROTOCOL_EXECUTION = "PROTOCOL"
NON_PROTOCOL_TEST_OVERRIDE = "NON_PROTOCOL_TEST_OVERRIDE"

_GATE_SCHEMA = "eaton-component-pre-unblinding-gate-v1"
_FUTILITY_SCHEMA = "eaton-component-pre-annotation-futility-v1"
_FUTILITY_BINDING_SCHEMA = "eaton-component-pre-annotation-binding-v1"
_DECISION_SCHEMA = "eaton-component-confirmation-decision-v1"
_BINDING_SCHEMA = "eaton-component-artifact-binding-v1"
_ANALYSIS_SCHEMA = "eaton-component-bound-confirmation-analysis-v1"
_AUTHORIZATION_SCHEMA = "eaton-component-confirmation-authorization-v1"
_AUTHORIZATION_STATUS = "AUTHORIZED_AFTER_STUDY_DEVELOPMENT_PASS"
_PREDICTION_METADATA_SCHEMA = "eaton-component-direction-predictions-v1"
_EXPECTED_PROTOCOL_SIMULATIONS = 1_000
_EXPECTED_PROTOCOL_BOOTSTRAPS = 500
_EXPECTED_ANALYSIS_BOOTSTRAPS = 10_000
_EXPECTED_ANALYSIS_BOOTSTRAP_SEED = 20_260_711
_EXPECTED_ROOF_DGP_MEAN = 0.60
_EXPECTED_FACADE_DGP_MEAN = 0.40
_EXPECTED_DGP_ICC = 0.05
_EXPECTED_POWER_THRESHOLD = 0.80
_MAX_TEST_SIMULATIONS = 100
_MAX_TEST_BOOTSTRAPS = 200
_FORBIDDEN_OBSERVED_COLUMNS = (
    "signed_direction",
    "direction_consistent",
    "positive_direction_share",
    "directional_contrast",
    "h_b1",
    "h_b2",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _protocol_sha256(config: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _trusted_sha256(value: object, name: str) -> str:
    digest = str(value).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{name} must be a complete SHA-256 value")
    return digest


def _artifact_digest(payload: Mapping[str, object], digest_field: str) -> str:
    unsigned = {key: value for key, value in payload.items() if key != digest_field}
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _json_scalar(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (bool, int, str)):
        return value
    if pd.isna(value):
        return None
    return str(value)


def _frame_digest(
    frame: pd.DataFrame,
    *,
    sort_by: tuple[str, ...],
) -> str:
    missing = [column for column in sort_by if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot digest table without sort columns: {missing}")
    columns = sorted(str(column) for column in frame.columns)
    ordered = frame.loc[:, columns].sort_values(list(sort_by), kind="mergesort")
    rows = [
        {column: _json_scalar(row[column]) for column in columns}
        for row in ordered.to_dict(orient="records")
    ]
    payload = {"columns": columns, "rows": rows}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _id_set_digest(values: pd.Series) -> str:
    identifiers = sorted(values.astype("string").str.strip().astype(str).tolist())
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Identifier set digest requires unique identifiers")
    return hashlib.sha256(_canonical_json(identifiers).encode("utf-8")).hexdigest()


def _artifact_bytes(
    *,
    data: bytes | None,
    path: str | Path | None,
    name: str,
) -> bytes:
    if data is None and path is None:
        raise ValueError(f"{name} bytes or path is required")
    file_bytes = None
    if path is not None:
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        file_bytes = artifact_path.read_bytes()
    if data is not None and not isinstance(data, bytes):
        raise TypeError(f"{name} bytes must be literal bytes")
    if data is not None and file_bytes is not None and data != file_bytes:
        raise ValueError(f"{name} bytes do not match the supplied path")
    return data if data is not None else bytes(file_bytes)


def _json_object_bytes(value: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(value.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _csv_frame_bytes(value: bytes, name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(value), dtype={"pair_id": str})
    except Exception as error:
        raise ValueError(f"{name} must be a readable CSV") from error


def _reject_observed_outcomes(frame: pd.DataFrame, artifact_name: str) -> None:
    leaked = [
        str(column)
        for column in frame.columns
        if any(
            marker in str(column).strip().casefold()
            for marker in _FORBIDDEN_OBSERVED_COLUMNS
        )
    ]
    if leaked:
        raise ValueError(
            f"{artifact_name} contains fields forbidden before unblinding: "
            f"{sorted(leaked)}"
        )


def _required_mapping(
    parent: Mapping[str, object], key: str
) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"protocol config requires an object at {key!r}")
    return value


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: object, name: str) -> int:
    numeric = _finite_float(value, name)
    if not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _nonnegative_int(value: object, name: str) -> int:
    numeric = _finite_float(value, name)
    if not numeric.is_integer() or numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(numeric)


def _parse_confirmation_thresholds(
    config: Mapping[str, object], gate_config: Mapping[str, object]
) -> dict[str, float]:
    confirmation = _required_mapping(config, "confirmation_rule")
    rules = confirmation.get("go_requires_all")
    if not isinstance(rules, list) or not all(isinstance(item, str) for item in rules):
        raise ValueError("confirmation_rule.go_requires_all must be a list of strings")

    def number_from_rule(marker: str, operator: str) -> float:
        candidates = [item for item in rules if marker in item.casefold()]
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one confirmation rule for {marker!r}")
        pattern = re.escape(operator) + r"\s*([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, candidates[0])
        if match is None:
            raise ValueError(
                f"Could not parse threshold from confirmation rule {candidates[0]!r}"
            )
        return float(match.group(1))

    delta_min = _finite_float(
        gate_config.get("minimum_meaningful_delta"),
        "pre_unblinding_gate.minimum_meaningful_delta",
    )
    parsed_delta = number_from_rule("directional contrast", ">=")
    if not math.isclose(delta_min, parsed_delta, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Gate and confirmation minimum-delta rules disagree")
    return {
        "delta_min": delta_min,
        "ci_lower_strict_min": number_from_rule("bootstrap lower bound", ">"),
        "roof_share_strict_min": number_from_rule("roof positive-direction share", ">"),
        "facade_share_strict_max": number_from_rule(
            "facade positive-direction share", "<"
        ),
        "coverage_min": number_from_rule("mechanism eligibility coverage", ">="),
    }


def _validate_protocol_config(config: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise TypeError("protocol_config must be a mapping")
    gate = _required_mapping(config, "pre_unblinding_gate")
    power = _required_mapping(config, "power_simulation")
    models = _required_mapping(config, "models")

    expected_seeds_raw = models.get("seeds")
    if not isinstance(expected_seeds_raw, list):
        raise ValueError("models.seeds must be a list")
    expected_seed_values = [
        _finite_float(seed, "models.seeds item") for seed in expected_seeds_raw
    ]
    if any(not seed.is_integer() for seed in expected_seed_values):
        raise ValueError("models.seeds must contain integers")
    expected_seeds = tuple(int(seed) for seed in expected_seed_values)
    if len(expected_seeds) != 5 or len(set(expected_seeds)) != 5:
        raise ValueError("The frozen protocol must specify exactly five unique seeds")

    thresholds = {
        "raw_agreement_min": _finite_float(
            gate.get("raw_rater_agreement_min"),
            "pre_unblinding_gate.raw_rater_agreement_min",
        ),
        "kappa_min": _finite_float(
            gate.get("cohen_kappa_min"), "pre_unblinding_gate.cohen_kappa_min"
        ),
        "eligible_min": _nonnegative_int(
            gate.get("mechanism_eligible_disagreements_min"),
            "pre_unblinding_gate.mechanism_eligible_disagreements_min",
        ),
        "roof_min": _nonnegative_int(
            gate.get("roof_eligible_min"), "pre_unblinding_gate.roof_eligible_min"
        ),
        "facade_min": _nonnegative_int(
            gate.get("facade_eligible_min"),
            "pre_unblinding_gate.facade_eligible_min",
        ),
        "blocks_min": _nonnegative_int(
            gate.get("eligible_spatial_blocks_min"),
            "pre_unblinding_gate.eligible_spatial_blocks_min",
        ),
        "blocks_per_stratum_min": _nonnegative_int(
            gate.get("eligible_blocks_per_stratum_min"),
            "pre_unblinding_gate.eligible_blocks_per_stratum_min",
        ),
        "power_min": _finite_float(
            gate.get("simulated_power_min"),
            "pre_unblinding_gate.simulated_power_min",
        ),
    }
    if any(
        not 0.0 <= float(thresholds[name]) <= 1.0
        for name in ("raw_agreement_min", "kappa_min", "power_min")
    ):
        raise ValueError("Agreement, kappa, and power thresholds must lie in [0, 1]")
    if not math.isclose(
        float(thresholds["power_min"]),
        _EXPECTED_POWER_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("The frozen protocol power threshold must remain 0.80")
    if gate.get("failure_status") != STOP_PRE_UNBLIND_GATE:
        raise ValueError("The protocol has an unexpected pre-unblinding failure status")

    roof_mean = _finite_float(
        power.get("roof_positive_direction_probability"),
        "power_simulation.roof_positive_direction_probability",
    )
    facade_mean = _finite_float(
        power.get("facade_positive_direction_probability"),
        "power_simulation.facade_positive_direction_probability",
    )
    icc = _finite_float(
        power.get("within_block_intraclass_correlation"),
        "power_simulation.within_block_intraclass_correlation",
    )
    minimum_delta = _finite_float(
        power.get("minimum_delta"), "power_simulation.minimum_delta"
    )
    if not (0.0 < roof_mean < 1.0 and 0.0 < facade_mean < 1.0):
        raise ValueError("Frozen beta-binomial means must lie strictly inside (0, 1)")
    if not 0.0 < icc < 1.0:
        raise ValueError("Frozen beta-binomial ICC must lie strictly inside (0, 1)")
    if not (
        math.isclose(
            roof_mean, _EXPECTED_ROOF_DGP_MEAN, rel_tol=0.0, abs_tol=1e-12
        )
        and math.isclose(
            facade_mean,
            _EXPECTED_FACADE_DGP_MEAN,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(icc, _EXPECTED_DGP_ICC, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("The frozen beta-binomial DGP must remain 0.60/0.40 with ICC 0.05")
    if not math.isclose(
        roof_mean - facade_mean, minimum_delta, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Frozen DGP means do not encode power_simulation.minimum_delta")
    if not math.isclose(
        minimum_delta,
        _finite_float(
            gate.get("minimum_meaningful_delta"),
            "pre_unblinding_gate.minimum_meaningful_delta",
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Power and gate minimum-delta commitments disagree")
    simulations = _positive_int(
        power.get("simulation_replicates"),
        "power_simulation.simulation_replicates",
    )
    bootstraps = _positive_int(
        power.get("cluster_bootstrap_replicates_per_simulation"),
        "power_simulation.cluster_bootstrap_replicates_per_simulation",
    )
    if simulations != _EXPECTED_PROTOCOL_SIMULATIONS or bootstraps != _EXPECTED_PROTOCOL_BOOTSTRAPS:
        raise ValueError(
            "Frozen protocol power repetitions must remain 1000 simulations by "
            "500 whole-block bootstraps"
        )
    if power.get("signed_observed_predictions_forbidden") is not True:
        raise ValueError("The frozen protocol must forbid observed prediction ordering")

    primary = _required_mapping(config, "primary_estimand")
    analysis_bootstraps = _positive_int(
        primary.get("bootstrap_replicates"),
        "primary_estimand.bootstrap_replicates",
    )
    analysis_bootstrap_seed = _nonnegative_int(
        primary.get("bootstrap_seed"),
        "primary_estimand.bootstrap_seed",
    )
    if (
        analysis_bootstraps != _EXPECTED_ANALYSIS_BOOTSTRAPS
        or analysis_bootstrap_seed != _EXPECTED_ANALYSIS_BOOTSTRAP_SEED
    ):
        raise ValueError(
            "Frozen confirmation analysis must remain 10000 whole-block bootstraps "
            "with seed 20260711"
        )

    return {
        "expected_seeds": expected_seeds,
        "thresholds": thresholds,
        "roof_mean": roof_mean,
        "facade_mean": facade_mean,
        "icc": icc,
        "simulations": simulations,
        "bootstraps": bootstraps,
        "simulation_seed": _nonnegative_int(
            power.get("simulation_seed"), "power_simulation.simulation_seed"
        ),
        "confirmation_thresholds": _parse_confirmation_thresholds(config, gate),
        "analysis_contract": {
            "schema_version": _ANALYSIS_SCHEMA,
            "bootstrap_replicates": analysis_bootstraps,
            "bootstrap_seed": analysis_bootstrap_seed,
        },
    }


def _resolve_protocol_artifact(
    protocol_config: Mapping[str, object],
    *,
    expected_protocol_sha256: str,
    protocol_config_bytes: bytes | None,
    protocol_config_path: str | Path | None,
) -> tuple[dict[str, Any], bytes, str]:
    expected = str(expected_protocol_sha256).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("expected_protocol_sha256 must be a complete SHA-256 value")
    raw = _artifact_bytes(
        data=protocol_config_bytes,
        path=protocol_config_path,
        name="protocol config",
    )
    if _sha256_bytes(raw) != expected:
        raise ValueError("Raw protocol config SHA-256 does not match expected_protocol_sha256")
    parsed = _json_object_bytes(raw, "protocol config")
    if _canonical_json(parsed) != _canonical_json(dict(protocol_config)):
        raise ValueError("protocol_config mapping does not match the bound raw config bytes")
    return parsed, raw, expected


def _validated_prediction_population(
    predictions: pd.DataFrame,
    *,
    expected_role: str,
    expected_protocol_sha256: str,
    expected_protocol_summary_sha256: str,
    expected_prediction_csv_sha256: str,
    expected_prediction_metadata_sha256: str,
    protocol_version: str,
    expected_seeds: tuple[int, ...],
    prediction_csv_bytes: bytes | None,
    prediction_csv_path: str | Path | None,
    prediction_metadata_bytes: bytes | None,
    prediction_metadata_path: str | Path | None,
    role_manifest_csv_bytes: bytes | None,
    role_manifest_path: str | Path | None,
    protocol_summary_bytes: bytes | None,
    protocol_summary_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw_predictions = _artifact_bytes(
        data=prediction_csv_bytes,
        path=prediction_csv_path,
        name="prediction CSV",
    )
    raw_metadata = _artifact_bytes(
        data=prediction_metadata_bytes,
        path=prediction_metadata_path,
        name="prediction metadata",
    )
    raw_manifest = _artifact_bytes(
        data=role_manifest_csv_bytes,
        path=role_manifest_path,
        name="role manifest CSV",
    )
    raw_summary = _artifact_bytes(
        data=protocol_summary_bytes,
        path=protocol_summary_path,
        name="protocol summary",
    )
    trusted_prediction_sha256 = _trusted_sha256(
        expected_prediction_csv_sha256, "expected_prediction_csv_sha256"
    )
    trusted_metadata_sha256 = _trusted_sha256(
        expected_prediction_metadata_sha256,
        "expected_prediction_metadata_sha256",
    )
    trusted_summary_sha256 = _trusted_sha256(
        expected_protocol_summary_sha256, "expected_protocol_summary_sha256"
    )
    if _sha256_bytes(raw_predictions) != trusted_prediction_sha256:
        raise ValueError(
            "Raw prediction CSV SHA-256 does not match expected_prediction_csv_sha256"
        )
    if _sha256_bytes(raw_metadata) != trusted_metadata_sha256:
        raise ValueError(
            "Raw prediction metadata SHA-256 does not match "
            "expected_prediction_metadata_sha256"
        )
    if _sha256_bytes(raw_summary) != trusted_summary_sha256:
        raise ValueError(
            "Raw protocol summary SHA-256 does not match "
            "expected_protocol_summary_sha256"
        )
    metadata = _json_object_bytes(raw_metadata, "prediction metadata")
    summary = _json_object_bytes(raw_summary, "protocol summary")
    if metadata.get("schema_version") != _PREDICTION_METADATA_SCHEMA:
        raise ValueError("Prediction metadata has the wrong schema_version")
    expected_status = (
        "PROTOCOL_DEVELOPMENT_RUN"
        if expected_role == PROTOCOL_ROLE
        else "PROTOCOL_CONFIRMATION_RUN"
        if expected_role == "spatial_confirmation"
        else None
    )
    if expected_status is None or metadata.get("status") != expected_status:
        raise ValueError("Prediction metadata is not a complete registered role run")
    if metadata.get("protocol_version") != protocol_version:
        raise ValueError("Prediction metadata protocol_version mismatch")
    if metadata.get("protocol_config_sha256") != expected_protocol_sha256:
        raise ValueError("Prediction metadata protocol config hash mismatch")
    if metadata.get("prediction_role") != expected_role:
        raise ValueError("Prediction metadata role mismatch")
    if expected_role == PROTOCOL_ROLE and metadata.get(
        "spatial_confirmation_read_or_scored"
    ) is not False:
        raise ValueError("Development predictions must attest untouched confirmation")
    if expected_role == "spatial_confirmation" and metadata.get(
        "spatial_confirmation_scored_once"
    ) is not True:
        raise ValueError("Confirmation predictions must attest one-time confirmation scoring")
    if tuple(metadata.get("seeds", ())) != expected_seeds:
        raise ValueError("Prediction metadata seeds do not match the frozen protocol")
    if metadata.get("prediction_csv_sha256") != _sha256_bytes(raw_predictions):
        raise ValueError("Prediction CSV hash does not match prediction metadata")
    role_hashes = metadata.get("role_manifest_sha256")
    if not isinstance(role_hashes, Mapping) or role_hashes.get(expected_role) != _sha256_bytes(
        raw_manifest
    ):
        raise ValueError("Role manifest hash does not match prediction metadata")
    if summary.get("schema_version") != "eaton-component-direction-spatial-summary-v1":
        raise ValueError("Protocol summary has the wrong schema_version")
    if summary.get("protocol_version") != protocol_version:
        raise ValueError("Protocol summary protocol_version mismatch")
    if summary.get("protocol_config_sha256") != expected_protocol_sha256:
        raise ValueError("Protocol summary config hash mismatch")
    if summary.get("analysis_status") != "PROTOCOL_READY_CONFIRMATION_UNTOUCHED":
        raise ValueError("Protocol summary does not bind a ready untouched protocol")
    summary_role_hashes = summary.get("role_manifest_sha256")
    if (
        not isinstance(summary_role_hashes, Mapping)
        or summary_role_hashes.get(expected_role) != _sha256_bytes(raw_manifest)
        or role_hashes.get(expected_role) != summary_role_hashes.get(expected_role)
    ):
        raise ValueError("Role manifest does not match the frozen protocol summary")
    role_rows = summary.get("role_rows")
    if not isinstance(role_rows, Mapping):
        raise ValueError("Protocol summary is missing frozen role row counts")
    commitment = summary.get("confirmation_commitment")
    if not isinstance(commitment, Mapping) or commitment.get("status") != (
        "UNSCORED_RESERVED_SAME_EVENT_SPATIAL_CONFIRMATION"
    ):
        raise ValueError("Protocol summary confirmation commitment is not reserved")
    if expected_role == "spatial_confirmation" and commitment.get(
        "manifest_sha256"
    ) != _sha256_bytes(raw_manifest):
        raise ValueError("Confirmation manifest does not match its frozen commitment")

    supplied = validate_seed_predictions(predictions, expected_role=expected_role)
    from_file = validate_seed_predictions(
        _csv_frame_bytes(raw_predictions, "prediction CSV"),
        expected_role=expected_role,
    )
    if _frame_digest(supplied, sort_by=("pair_id", "seed")) != _frame_digest(
        from_file, sort_by=("pair_id", "seed")
    ):
        raise ValueError("Supplied prediction table does not equal the bound prediction CSV")
    actual_seeds = tuple(sorted(supplied["seed"].unique()))
    if actual_seeds != tuple(sorted(expected_seeds)):
        raise ValueError("Prediction seeds must exactly equal the five frozen protocol seeds")

    manifest = _csv_frame_bytes(raw_manifest, "role manifest CSV")
    required_manifest = {"pair_id", "spatial_block_id", "protocol_role", "label"}
    missing = sorted(required_manifest - set(manifest.columns))
    if missing:
        raise ValueError(f"Role manifest is missing columns: {missing}")
    manifest = manifest.copy()
    manifest["pair_id"] = manifest["pair_id"].astype("string").str.strip()
    manifest["spatial_block_id"] = manifest["spatial_block_id"].astype("string").str.strip()
    if (
        manifest["pair_id"].isna().any()
        or manifest["pair_id"].eq("").any()
        or manifest["pair_id"].duplicated().any()
    ):
        raise ValueError("Role manifest pair_id must be nonblank and unique")
    roles = manifest["protocol_role"].astype("string").str.strip()
    if not roles.eq(expected_role).all():
        raise ValueError("Role manifest contains an unexpected protocol role")
    labels = pd.to_numeric(manifest["label"], errors="coerce")
    if labels.isna().any() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Role manifest label must contain integer targets")
    manifest["label"] = labels.astype(int)

    prediction_meta = (
        supplied.groupby("pair_id", sort=True)
        .agg(
            spatial_block_id=("spatial_block_id", "first"),
            target=("target", "first"),
        )
        .sort_index()
    )
    manifest_meta = manifest.set_index("pair_id").sort_index()
    if set(prediction_meta.index.astype(str)) != set(manifest_meta.index.astype(str)):
        raise ValueError(
            "Prediction population must exactly equal the complete bound role manifest"
        )
    manifest_meta = manifest_meta.loc[prediction_meta.index]
    if not prediction_meta["spatial_block_id"].astype(str).eq(
        manifest_meta["spatial_block_id"].astype(str)
    ).all():
        raise ValueError("Prediction spatial blocks do not match the role manifest")
    if not prediction_meta["target"].astype(int).eq(
        manifest_meta["label"].astype(int)
    ).all():
        raise ValueError("Prediction targets do not match the role manifest")
    if int(role_rows.get(expected_role, -1)) != len(manifest):
        raise ValueError("Role manifest row count does not match protocol summary")

    identity = {
        "prediction_csv_sha256": _sha256_bytes(raw_predictions),
        "prediction_metadata_sha256": _sha256_bytes(raw_metadata),
        "trusted_prediction_csv_sha256": trusted_prediction_sha256,
        "trusted_prediction_metadata_sha256": trusted_metadata_sha256,
        "prediction_table_sha256": _frame_digest(
            supplied, sort_by=("pair_id", "seed")
        ),
        "prediction_row_count": int(len(supplied)),
        "prediction_pair_count": int(supplied["pair_id"].nunique()),
        "prediction_spatial_block_count": int(
            supplied["spatial_block_id"].nunique()
        ),
        "prediction_pair_ids_sha256": _id_set_digest(
            supplied.drop_duplicates("pair_id")["pair_id"]
        ),
        "role_manifest_csv_sha256": _sha256_bytes(raw_manifest),
        "role_manifest_table_sha256": _frame_digest(
            manifest, sort_by=("pair_id",)
        ),
        "role_manifest_row_count": int(len(manifest)),
        "protocol_summary_sha256": _sha256_bytes(raw_summary),
        "trusted_protocol_summary_sha256": trusted_summary_sha256,
    }
    return supplied, manifest, identity


def _validate_reliability(
    reliability: pd.DataFrame,
    *,
    reference_pair_count: int,
    agreement_min: float,
    kappa_min: float,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    required = {"field", "observed_agreement", "cohen_kappa"}
    missing = sorted(required - set(reliability.columns))
    if missing:
        raise ValueError(f"Annotation reliability table missing columns: {missing}")
    if reliability.empty:
        raise ValueError("Annotation reliability table cannot be empty")
    _reject_observed_outcomes(reliability, "annotation reliability table")

    frame = reliability.copy()
    frame["field"] = frame["field"].astype("string").str.strip()
    if frame["field"].isna().any() or frame["field"].eq("").any():
        raise ValueError("Reliability field names cannot be null or blank")
    if frame["field"].duplicated().any():
        raise ValueError("Reliability table must contain one row per annotation field")
    supplied = set(frame["field"])
    expected = set(ANNOTATION_FIELDS)
    if supplied != expected:
        raise ValueError(
            "Reliability table must cover exactly every registered annotation field; "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
        )

    frame["observed_agreement"] = pd.to_numeric(
        frame["observed_agreement"], errors="coerce"
    )
    frame["cohen_kappa"] = pd.to_numeric(frame["cohen_kappa"], errors="coerce")
    pair_count_ok = True
    if "pair_count" in frame.columns:
        pair_counts = pd.to_numeric(frame["pair_count"], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        pair_count_ok = bool(
            np.isfinite(pair_counts).all()
            and np.equal(pair_counts, np.floor(pair_counts)).all()
            and np.equal(pair_counts, reference_pair_count).all()
        )

    rows: list[dict[str, object]] = []
    reasons: list[dict[str, str]] = []
    all_fields_pass = pair_count_ok
    for field in ANNOTATION_FIELDS:
        row = frame.loc[frame["field"].eq(field)].iloc[0]
        agreement = (
            float(row["observed_agreement"])
            if pd.notna(row["observed_agreement"])
            else float("nan")
        )
        kappa = (
            float(row["cohen_kappa"])
            if pd.notna(row["cohen_kappa"])
            else float("nan")
        )
        agreement_pass = (
            math.isfinite(agreement)
            and 0.0 <= agreement <= 1.0
            and agreement >= agreement_min
        )
        kappa_pass = (
            math.isfinite(kappa)
            and -1.0 <= kappa <= 1.0
            and kappa >= kappa_min
        )
        field_pass = agreement_pass and kappa_pass
        all_fields_pass = all_fields_pass and field_pass
        rows.append(
            {
                "field": field,
                "observed_agreement": agreement if math.isfinite(agreement) else None,
                "cohen_kappa": kappa if math.isfinite(kappa) else None,
                "agreement_pass": agreement_pass,
                "kappa_pass": kappa_pass,
                "passed": field_pass,
            }
        )
        if not field_pass:
            reasons.append(
                {
                    "code": "annotation_reliability_below_threshold",
                    "detail": (
                        f"{field} failed raw agreement >= {agreement_min} and/or "
                        f"Cohen kappa >= {kappa_min}."
                    ),
                }
            )
    if not pair_count_ok:
        reasons.append(
            {
                "code": "reliability_pair_count_mismatch",
                "detail": (
                    "Reliability pair_count does not equal the complete final-reference "
                    "pair count for every annotation field."
                ),
            }
        )
    return (
        {
            "passed": all_fields_pass,
            "raw_agreement_min": agreement_min,
            "cohen_kappa_min": kappa_min,
            "pair_count_matches_reference": pair_count_ok,
            "fields": rows,
        },
        reasons,
    )


def _mechanism_layout(
    ensemble: pd.DataFrame, references: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    # Equality is the only operation permitted on the two observed predictions.
    disagreement = ensemble["street_prediction"].ne(ensemble["overhead_prediction"])
    disagreement_metadata = ensemble.loc[
        disagreement, ["pair_id", "spatial_block_id", "target"]
    ].copy()
    prediction_ids = set(disagreement_metadata["pair_id"].astype(str))
    reference_ids = set(references["pair_id"].astype(str))
    if prediction_ids != reference_ids:
        raise ValueError(
            "Final reference coverage must equal exactly all ensemble disagreements; "
            f"missing={sorted(prediction_ids - reference_ids)[:5]}, "
            f"unexpected={sorted(reference_ids - prediction_ids)[:5]}"
        )
    joined = disagreement_metadata.merge(
        references[["pair_id", *ANNOTATION_FIELDS]],
        on="pair_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all() or len(joined) != len(references):
        raise RuntimeError("Outer-join coverage audit detected unexpected pair loss")
    joined = joined.drop(columns="_merge")

    roof = (
        joined["component_dominance"].eq("roof")
        & joined["overhead_roof_assessability"].eq("assessable")
        & joined["street_roof_assessability"].eq("not_assessable")
    ).fillna(False).to_numpy(dtype=bool)
    facade = (
        joined["component_dominance"].eq("facade")
        & joined["street_facade_assessability"].eq("assessable")
        & joined["overhead_facade_assessability"].eq("not_assessable")
    ).fillna(False).to_numpy(dtype=bool)
    joined["mechanism_stratum"] = np.select(
        [roof, facade], ["roof", "facade"], default="ineligible"
    )
    eligible = joined.loc[
        joined["mechanism_stratum"].ne("ineligible"),
        ["pair_id", "spatial_block_id", "target", "mechanism_stratum"],
    ].copy()
    return eligible.reset_index(drop=True), int(len(disagreement_metadata))


def _layout_audit(
    layout: pd.DataFrame, disagreement_count: int, thresholds: Mapping[str, object]
) -> tuple[dict[str, object], list[dict[str, str]]]:
    eligible_count = int(len(layout))
    stratum_counts = {
        stratum: int(layout["mechanism_stratum"].eq(stratum).sum())
        for stratum in ("roof", "facade")
    }
    total_blocks = int(layout["spatial_block_id"].nunique())
    stratum_blocks = {
        stratum: int(
            layout.loc[
                layout["mechanism_stratum"].eq(stratum), "spatial_block_id"
            ].nunique()
        )
        for stratum in ("roof", "facade")
    }
    target_counts = {
        target: int(layout["target"].eq(target).sum())
        for target in range(len(SEVERITY_CLASS_ORDER))
    }
    cell_counts = {
        stratum: {
            target: int(
                (
                    layout["mechanism_stratum"].eq(stratum)
                    & layout["target"].eq(target)
                ).sum()
            )
            for target in range(len(SEVERITY_CLASS_ORDER))
        }
        for stratum in ("roof", "facade")
    }
    common_support = all(
        count == 0
        or all(cell_counts[stratum][target] > 0 for stratum in ("roof", "facade"))
        for target, count in target_counts.items()
    )
    weights = {
        str(target): (float(count / eligible_count) if eligible_count else 0.0)
        for target, count in target_counts.items()
    }

    checks = {
        "eligible_total": eligible_count >= int(thresholds["eligible_min"]),
        "eligible_roof": stratum_counts["roof"] >= int(thresholds["roof_min"]),
        "eligible_facade": stratum_counts["facade"] >= int(thresholds["facade_min"]),
        "eligible_blocks": total_blocks >= int(thresholds["blocks_min"]),
        "roof_blocks": stratum_blocks["roof"]
        >= int(thresholds["blocks_per_stratum_min"]),
        "facade_blocks": stratum_blocks["facade"]
        >= int(thresholds["blocks_per_stratum_min"]),
        "severity_common_support": common_support,
    }
    reason_details = {
        "eligible_total": (
            "mechanism_eligible_count_below_minimum",
            f"Eligible count {eligible_count} is below {thresholds['eligible_min']}.",
        ),
        "eligible_roof": (
            "roof_eligible_count_below_minimum",
            f"Roof count {stratum_counts['roof']} is below {thresholds['roof_min']}.",
        ),
        "eligible_facade": (
            "facade_eligible_count_below_minimum",
            f"Facade count {stratum_counts['facade']} is below {thresholds['facade_min']}.",
        ),
        "eligible_blocks": (
            "eligible_block_count_below_minimum",
            f"Eligible block count {total_blocks} is below {thresholds['blocks_min']}.",
        ),
        "roof_blocks": (
            "roof_block_count_below_minimum",
            f"Roof block count {stratum_blocks['roof']} is below {thresholds['blocks_per_stratum_min']}.",
        ),
        "facade_blocks": (
            "facade_block_count_below_minimum",
            f"Facade block count {stratum_blocks['facade']} is below {thresholds['blocks_per_stratum_min']}.",
        ),
        "severity_common_support": (
            "severity_common_support_absent",
            "At least one occupied target level is absent from one mechanism stratum.",
        ),
    }
    reasons = [
        {"code": reason_details[name][0], "detail": reason_details[name][1]}
        for name, passed in checks.items()
        if not passed
    ]
    return (
        {
            "passed": all(checks.values()),
            "disagreement_pair_count": disagreement_count,
            "mechanism_eligible_count": eligible_count,
            "roof_eligible_count": stratum_counts["roof"],
            "facade_eligible_count": stratum_counts["facade"],
            "eligible_spatial_block_count": total_blocks,
            "roof_eligible_spatial_block_count": stratum_blocks["roof"],
            "facade_eligible_spatial_block_count": stratum_blocks["facade"],
            "target_weights": weights,
            "stratum_target_counts": {
                stratum: {str(target): count for target, count in counts.items()}
                for stratum, counts in cell_counts.items()
            },
            "severity_standardization_common_support": common_support,
            "threshold_checks": checks,
        },
        reasons,
    )


def _simulate_frozen_power(
    layout: pd.DataFrame,
    *,
    roof_mean: float,
    facade_mean: float,
    icc: float,
    simulation_replicates: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    """Simulate the registered beta-binomial DGP on the observed layout."""

    blocks = tuple(sorted(layout["spatial_block_id"].astype(str).unique()))
    block_lookup = {block: index for index, block in enumerate(blocks)}
    block_index = layout["spatial_block_id"].astype(str).map(block_lookup).to_numpy(int)
    stratum_index = layout["mechanism_stratum"].map({"roof": 0, "facade": 1}).to_numpy(int)
    target_index = layout["target"].to_numpy(int)
    block_count = len(blocks)
    target_count = len(SEVERITY_CLASS_ORDER)

    cell_counts = np.zeros((block_count, 2, target_count), dtype=np.int64)
    np.add.at(cell_counts, (block_index, stratum_index, target_index), 1)
    target_weights = np.bincount(target_index, minlength=target_count).astype(float)
    target_weights /= float(len(layout))
    active_targets = target_weights > 0

    group_keys = tuple(sorted(set(zip(block_index.tolist(), stratum_index.tolist()))))
    group_lookup = {key: index for index, key in enumerate(group_keys)}
    row_group = np.asarray(
        [group_lookup[(block, stratum)] for block, stratum in zip(block_index, stratum_index)],
        dtype=int,
    )
    group_strata = np.asarray([stratum for _, stratum in group_keys], dtype=int)
    group_means = np.where(group_strata == 0, roof_mean, facade_mean)
    concentration = (1.0 / icc) - 1.0
    alpha = group_means * concentration
    beta = (1.0 - group_means) * concentration

    rng = np.random.default_rng(seed)
    successes = 0
    finite_bootstrap_counts: list[int] = []
    block_probabilities = np.full(block_count, 1.0 / block_count, dtype=float)
    flat_counts = cell_counts.reshape(block_count, -1)
    for _ in range(simulation_replicates):
        group_probability = rng.beta(alpha, beta)
        simulated_outcome = rng.binomial(1, group_probability[row_group]).astype(np.int64)
        outcome_counts = np.zeros_like(cell_counts)
        np.add.at(
            outcome_counts,
            (block_index, stratum_index, target_index),
            simulated_outcome,
        )
        multipliers = rng.multinomial(
            block_count, block_probabilities, size=bootstrap_replicates
        )
        denominators = (multipliers @ flat_counts).reshape(
            bootstrap_replicates, 2, target_count
        )
        numerators = (multipliers @ outcome_counts.reshape(block_count, -1)).reshape(
            bootstrap_replicates, 2, target_count
        )
        finite = np.all(denominators[:, :, active_targets] > 0, axis=(1, 2))
        finite_count = int(finite.sum())
        finite_bootstrap_counts.append(finite_count)
        if finite_count == 0:
            continue
        selected_num = numerators[finite].astype(float)
        selected_den = denominators[finite].astype(float)
        shares = np.divide(
            selected_num,
            selected_den,
            out=np.zeros_like(selected_num),
            where=selected_den > 0,
        )
        simulated_delta = (shares[:, 0, :] - shares[:, 1, :]) @ target_weights
        lower = float(np.percentile(simulated_delta, 2.5))
        if lower > 0.0:
            successes += 1

    estimated_power = float(successes / simulation_replicates)
    return {
        "executed": True,
        "simulation_replicates": simulation_replicates,
        "whole_block_bootstrap_replicates_per_simulation": bootstrap_replicates,
        "simulation_seed": seed,
        "dgp_means": {"roof": roof_mean, "facade": facade_mean},
        "within_block_intraclass_correlation": icc,
        "observed_layout_pair_count": int(len(layout)),
        "observed_layout_block_count": block_count,
        "successful_simulations": successes,
        "estimated_power": estimated_power,
        "finite_bootstrap_replicates_min": min(finite_bootstrap_counts),
        "finite_bootstrap_replicates_max": max(finite_bootstrap_counts),
    }


def pre_unblinding_gate(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    annotation_reliability: pd.DataFrame,
    protocol_config: Mapping[str, object],
    *,
    expected_protocol_sha256: str,
    expected_protocol_summary_sha256: str,
    expected_prediction_csv_sha256: str,
    expected_prediction_metadata_sha256: str,
    raw_annotations: pd.DataFrame,
    adjudications: pd.DataFrame | None,
    development_pass_authorization: Mapping[str, object] | None = None,
    protocol_config_bytes: bytes | None = None,
    protocol_config_path: str | Path | None = None,
    prediction_csv_bytes: bytes | None = None,
    prediction_csv_path: str | Path | None = None,
    prediction_metadata_bytes: bytes | None = None,
    prediction_metadata_path: str | Path | None = None,
    role_manifest_csv_bytes: bytes | None = None,
    role_manifest_path: str | Path | None = None,
    protocol_summary_bytes: bytes | None = None,
    protocol_summary_path: str | Path | None = None,
    expected_role: str = PROTOCOL_ROLE,
    test_simulation_replicates_override: int | None = None,
    test_bootstrap_replicates_override: int | None = None,
) -> dict[str, object]:
    """Evaluate the registered gate without accessing observed outcome signs.

    The two ``test_*_override`` arguments exist only to keep unit tests small.
    Supplying either marks the artifact ``NON_PROTOCOL_TEST_OVERRIDE`` and makes
    a passing gate impossible, irrespective of the simulated power estimate.
    """

    bound_protocol, _protocol_bytes, expected_protocol_sha256 = _resolve_protocol_artifact(
        protocol_config,
        expected_protocol_sha256=expected_protocol_sha256,
        protocol_config_bytes=protocol_config_bytes,
        protocol_config_path=protocol_config_path,
    )
    parsed = _validate_protocol_config(bound_protocol)
    protocol_version = str(bound_protocol.get("protocol_version", "")).strip()
    if not protocol_version:
        raise ValueError("protocol_config.protocol_version cannot be blank")
    authorization: Mapping[str, object] | None = None
    if expected_role == "spatial_confirmation":
        authorization = _verify_confirmation_authorization_artifact(
            development_pass_authorization,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_protocol_summary_sha256=expected_protocol_summary_sha256,
        )
    elif expected_role == PROTOCOL_ROLE:
        if development_pass_authorization is not None:
            raise ValueError(
                "study_development cannot consume a confirmation authorization"
            )
    else:
        raise ValueError(f"Unsupported protocol role for gate: {expected_role!r}")
    _reject_observed_outcomes(predictions, "prediction table")
    _reject_observed_outcomes(references, "component reference")
    _reject_observed_outcomes(raw_annotations, "raw annotations")
    if adjudications is not None:
        _reject_observed_outcomes(adjudications, "adjudications")
    validated_predictions, role_manifest, prediction_identity = (
        _validated_prediction_population(
            predictions,
            expected_role=expected_role,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_protocol_summary_sha256=expected_protocol_summary_sha256,
            expected_prediction_csv_sha256=expected_prediction_csv_sha256,
            expected_prediction_metadata_sha256=expected_prediction_metadata_sha256,
            protocol_version=protocol_version,
            expected_seeds=tuple(parsed["expected_seeds"]),
            prediction_csv_bytes=prediction_csv_bytes,
            prediction_csv_path=prediction_csv_path,
            prediction_metadata_bytes=prediction_metadata_bytes,
            prediction_metadata_path=prediction_metadata_path,
            role_manifest_csv_bytes=role_manifest_csv_bytes,
            role_manifest_path=role_manifest_path,
            protocol_summary_bytes=protocol_summary_bytes,
            protocol_summary_path=protocol_summary_path,
        )
    )
    if authorization is not None:
        if prediction_identity["protocol_summary_sha256"] != authorization.get(
            "protocol_summary_sha256"
        ):
            raise ValueError(
                "Confirmation protocol summary is not the development-authorized summary"
            )
        if prediction_identity["role_manifest_csv_sha256"] != authorization.get(
            "confirmation_manifest_sha256"
        ):
            raise ValueError(
                "Confirmation role manifest is not the development-authorized manifest"
            )
    actual_seeds = tuple(sorted(validated_predictions["seed"].unique()))
    ensemble = ensemble_seed_predictions(
        validated_predictions, expected_role=expected_role
    )
    reference = validate_component_reference(references)
    if not reference["protocol_version"].eq(protocol_version).all():
        raise ValueError("Reference protocol_version does not match protocol config")
    if not reference["protocol_sha256"].eq(expected_protocol_sha256).all():
        raise ValueError("Reference protocol_sha256 does not match the frozen config file")

    validated_raw = validate_annotation_packet(
        raw_annotations,
        expected_protocol_version=protocol_version,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    recomputed_reliability = compute_annotation_reliability(
        validated_raw,
        expected_protocol_version=protocol_version,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    if _frame_digest(
        annotation_reliability, sort_by=("field",)
    ) != _frame_digest(recomputed_reliability, sort_by=("field",)):
        raise ValueError(
            "Annotation reliability must exactly reproduce the bound raw annotations"
        )
    recomputed_reference = build_adjudicated_reference(
        validated_raw,
        adjudications,
        expected_protocol_version=protocol_version,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    if _frame_digest(reference, sort_by=("pair_id",)) != _frame_digest(
        recomputed_reference, sort_by=("pair_id",)
    ):
        raise ValueError(
            "Component reference must exactly reproduce raw annotations/adjudications"
        )

    layout, disagreement_count = _mechanism_layout(ensemble, reference)
    thresholds = parsed["thresholds"]
    reliability_audit, reliability_reasons = _validate_reliability(
        annotation_reliability,
        reference_pair_count=len(reference),
        agreement_min=float(thresholds["raw_agreement_min"]),
        kappa_min=float(thresholds["kappa_min"]),
    )
    layout_audit, layout_reasons = _layout_audit(
        layout, disagreement_count, thresholds
    )
    reasons = [*reliability_reasons, *layout_reasons]

    overrides = (
        test_simulation_replicates_override,
        test_bootstrap_replicates_override,
    )
    override_active = any(value is not None for value in overrides)
    if override_active and not all(value is not None for value in overrides):
        raise ValueError("Both explicit test repetition overrides must be supplied together")
    if override_active:
        simulations = _positive_int(
            test_simulation_replicates_override,
            "test_simulation_replicates_override",
        )
        bootstraps = _positive_int(
            test_bootstrap_replicates_override,
            "test_bootstrap_replicates_override",
        )
        if simulations > _MAX_TEST_SIMULATIONS or bootstraps > _MAX_TEST_BOOTSTRAPS:
            raise ValueError(
                "Test repetition overrides must be small (<=100 simulations and "
                "<=200 bootstraps)"
            )
        execution_mode = NON_PROTOCOL_TEST_OVERRIDE
    else:
        simulations = int(parsed["simulations"])
        bootstraps = int(parsed["bootstraps"])
        execution_mode = PROTOCOL_EXECUTION

    preliminary_pass = bool(reliability_audit["passed"] and layout_audit["passed"])
    if preliminary_pass:
        power = _simulate_frozen_power(
            layout,
            roof_mean=float(parsed["roof_mean"]),
            facade_mean=float(parsed["facade_mean"]),
            icc=float(parsed["icc"]),
            simulation_replicates=simulations,
            bootstrap_replicates=bootstraps,
            seed=int(parsed["simulation_seed"]),
        )
        power_pass = float(power["estimated_power"]) >= float(thresholds["power_min"])
        power["required_power"] = float(thresholds["power_min"])
        power["passed"] = power_pass
        if not power_pass:
            reasons.append(
                {
                    "code": "simulated_power_below_threshold",
                    "detail": (
                        f"Frozen simulated power {power['estimated_power']:.6f} is below "
                        f"{thresholds['power_min']}."
                    ),
                }
            )
    else:
        power_pass = False
        power = {
            "executed": False,
            "reason": "Skipped because at least one non-power gate condition failed.",
            "simulation_replicates": simulations,
            "whole_block_bootstrap_replicates_per_simulation": bootstraps,
            "simulation_seed": int(parsed["simulation_seed"]),
            "required_power": float(thresholds["power_min"]),
            "passed": False,
        }

    if override_active:
        reasons.append(
            {
                "code": "non_protocol_test_override",
                "detail": (
                    "Explicit reduced repetitions were used; this artifact can never "
                    "pass the registered gate."
                ),
            }
        )
    all_pass = preliminary_pass and power_pass and not override_active
    status = PASS_PRE_UNBLIND_GATE if all_pass else STOP_PRE_UNBLIND_GATE
    disagreement_ids = ensemble.loc[
        ensemble["street_prediction"].ne(ensemble["overhead_prediction"]), "pair_id"
    ]
    artifact_binding: dict[str, object] = {
        "schema_version": _BINDING_SCHEMA,
        "protocol_version": protocol_version,
        "protocol_role": expected_role,
        "protocol_config_sha256": expected_protocol_sha256,
        "protocol_config_canonical_sha256": _protocol_sha256(bound_protocol),
        **prediction_identity,
        "reference_table_sha256": _frame_digest(reference, sort_by=("pair_id",)),
        "reference_pair_count": int(len(reference)),
        "reference_pair_ids_sha256": _id_set_digest(reference["pair_id"]),
        "raw_annotations_table_sha256": _frame_digest(
            validated_raw, sort_by=("pair_id", "rater_id")
        ),
        "raw_annotation_row_count": int(len(validated_raw)),
        "adjudications_table_sha256": (
            _frame_digest(adjudications, sort_by=("pair_id",))
            if adjudications is not None and not adjudications.empty
            else _sha256_bytes(b"null-adjudications")
        ),
        "adjudication_row_count": int(
            len(adjudications) if adjudications is not None else 0
        ),
        "annotation_reliability_table_sha256": _frame_digest(
            recomputed_reliability, sort_by=("field",)
        ),
        "disagreement_pair_count": int(disagreement_count),
        "disagreement_pair_ids_sha256": _id_set_digest(disagreement_ids),
    }
    artifact_binding["binding_sha256"] = _artifact_digest(
        artifact_binding, "binding_sha256"
    )
    result = {
        "schema_version": _GATE_SCHEMA,
        "protocol_version": protocol_version,
        "protocol_role": expected_role,
        "protocol_config_sha256": expected_protocol_sha256,
        "protocol_config_canonical_sha256": _protocol_sha256(bound_protocol),
        "reference_protocol_sha256": expected_protocol_sha256,
        "gate_status": status,
        "execution_mode": execution_mode,
        "observed_outcome_accessed": False,
        "prediction_seed_count": len(actual_seeds),
        "reference_covers_all_disagreements": True,
        "annotation_reliability": reliability_audit,
        "eligible_layout": layout_audit,
        "power_simulation": power,
        "confirmation_thresholds": parsed["confirmation_thresholds"],
        "analysis_contract": parsed["analysis_contract"],
        "development_pass_authorization": (
            dict(authorization) if authorization is not None else None
        ),
        "development_authorization_artifact_sha256": (
            authorization.get("authorization_artifact_sha256")
            if authorization is not None
            else None
        ),
        "artifact_binding": artifact_binding,
        "reasons": reasons,
    }
    result["gate_artifact_sha256"] = _artifact_digest(
        result, "gate_artifact_sha256"
    )
    # The gate artifact itself is safe to serialize before unblinding.
    _canonical_json(result)
    return result


def pre_annotation_futility_gate(
    predictions: pd.DataFrame,
    protocol_config: Mapping[str, object],
    *,
    expected_protocol_sha256: str,
    expected_protocol_summary_sha256: str,
    expected_prediction_csv_sha256: str,
    expected_prediction_metadata_sha256: str,
    protocol_config_bytes: bytes | None = None,
    protocol_config_path: str | Path | None = None,
    prediction_csv_bytes: bytes | None = None,
    prediction_csv_path: str | Path | None = None,
    prediction_metadata_bytes: bytes | None = None,
    prediction_metadata_path: str | Path | None = None,
    role_manifest_csv_bytes: bytes | None = None,
    role_manifest_path: str | Path | None = None,
    protocol_summary_bytes: bytes | None = None,
    protocol_summary_path: str | Path | None = None,
    expected_role: str = PROTOCOL_ROLE,
) -> dict[str, object]:
    """Stop before annotation when the registered sample size is impossible.

    This audit uses only the equality of the two seed-ensembled predictions.
    It does not consume annotations, construct a signed direction, or evaluate
    H-B1/H-B2.  Since mechanism-eligible disagreements are necessarily a subset
    of all disagreements, fewer unsigned disagreements than the registered
    eligible minimum is a conclusive structural futility condition.
    """

    if expected_role != PROTOCOL_ROLE:
        raise ValueError(
            "The pre-annotation futility audit is restricted to "
            "study_development so spatial_confirmation remains untouched"
        )
    bound_protocol, _protocol_bytes, expected_hash = _resolve_protocol_artifact(
        protocol_config,
        expected_protocol_sha256=expected_protocol_sha256,
        protocol_config_bytes=protocol_config_bytes,
        protocol_config_path=protocol_config_path,
    )
    parsed = _validate_protocol_config(bound_protocol)
    protocol_version = str(bound_protocol.get("protocol_version", "")).strip()
    if not protocol_version:
        raise ValueError("protocol_config.protocol_version cannot be blank")
    _reject_observed_outcomes(predictions, "prediction table")
    validated_predictions, _role_manifest, prediction_identity = (
        _validated_prediction_population(
            predictions,
            expected_role=expected_role,
            expected_protocol_sha256=expected_hash,
            expected_protocol_summary_sha256=expected_protocol_summary_sha256,
            expected_prediction_csv_sha256=expected_prediction_csv_sha256,
            expected_prediction_metadata_sha256=expected_prediction_metadata_sha256,
            protocol_version=protocol_version,
            expected_seeds=tuple(parsed["expected_seeds"]),
            prediction_csv_bytes=prediction_csv_bytes,
            prediction_csv_path=prediction_csv_path,
            prediction_metadata_bytes=prediction_metadata_bytes,
            prediction_metadata_path=prediction_metadata_path,
            role_manifest_csv_bytes=role_manifest_csv_bytes,
            role_manifest_path=role_manifest_path,
            protocol_summary_bytes=protocol_summary_bytes,
            protocol_summary_path=protocol_summary_path,
        )
    )
    ensemble = ensemble_seed_predictions(
        validated_predictions, expected_role=expected_role
    )
    # Equality is deliberately the only operation on the observed view outputs.
    disagreement = ensemble["street_prediction"].ne(
        ensemble["overhead_prediction"]
    )
    disagreement_ids = ensemble.loc[disagreement, "pair_id"]
    disagreement_count = int(disagreement.sum())
    disagreement_blocks = ensemble.loc[
        disagreement, "spatial_block_id"
    ].drop_duplicates()
    disagreement_block_count = int(len(disagreement_blocks))
    thresholds = parsed["thresholds"]
    eligible_min = int(thresholds["eligible_min"])
    roof_min = int(thresholds["roof_min"])
    facade_min = int(thresholds["facade_min"])
    blocks_min = int(thresholds["blocks_min"])
    blocks_per_stratum_min = int(thresholds["blocks_per_stratum_min"])
    role_pair_count = int(prediction_identity["prediction_pair_count"])
    role_block_count = int(prediction_identity["prediction_spatial_block_count"])

    capacity_checks = {
        "role_pairs_can_reach_eligible_minimum": role_pair_count >= eligible_min,
        "role_pairs_can_reach_roof_minimum": role_pair_count >= roof_min,
        "role_pairs_can_reach_facade_minimum": role_pair_count >= facade_min,
        "role_pairs_can_reach_disjoint_stratum_minima": role_pair_count
        >= roof_min + facade_min,
        "role_blocks_can_reach_eligible_minimum": role_block_count >= blocks_min,
        "role_blocks_can_reach_each_stratum_minimum": role_block_count
        >= blocks_per_stratum_min,
        "unsigned_disagreements_can_reach_eligible_minimum": disagreement_count
        >= eligible_min,
        "unsigned_disagreements_can_reach_roof_minimum": disagreement_count
        >= roof_min,
        "unsigned_disagreements_can_reach_facade_minimum": disagreement_count
        >= facade_min,
        "unsigned_disagreements_can_reach_disjoint_stratum_minima": disagreement_count
        >= roof_min + facade_min,
        "disagreement_blocks_can_reach_eligible_minimum": disagreement_block_count
        >= blocks_min,
        "disagreement_blocks_can_reach_each_stratum_minimum": disagreement_block_count
        >= blocks_per_stratum_min,
    }
    reason_specs = {
        "role_pairs_can_reach_eligible_minimum": (
            "role_pair_capacity_below_registered_eligible_minimum",
            f"The complete role has {role_pair_count} pairs, below the registered "
            f"eligible minimum {eligible_min}.",
        ),
        "role_pairs_can_reach_roof_minimum": (
            "role_pair_capacity_below_registered_roof_minimum",
            f"The complete role has {role_pair_count} pairs, below the registered "
            f"roof minimum {roof_min}.",
        ),
        "role_pairs_can_reach_facade_minimum": (
            "role_pair_capacity_below_registered_facade_minimum",
            f"The complete role has {role_pair_count} pairs, below the registered "
            f"facade minimum {facade_min}.",
        ),
        "role_pairs_can_reach_disjoint_stratum_minima": (
            "role_pair_capacity_below_combined_stratum_minima",
            f"The complete role has {role_pair_count} pairs, below the combined "
            f"disjoint roof/facade minima {roof_min + facade_min}.",
        ),
        "role_blocks_can_reach_eligible_minimum": (
            "role_spatial_block_capacity_below_registered_minimum",
            f"The complete role has {role_block_count} spatial blocks, below the "
            f"registered eligible-block minimum {blocks_min}.",
        ),
        "role_blocks_can_reach_each_stratum_minimum": (
            "role_spatial_block_capacity_below_per_stratum_minimum",
            f"The complete role has {role_block_count} spatial blocks, below the "
            f"registered per-stratum block minimum {blocks_per_stratum_min}.",
        ),
        "unsigned_disagreements_can_reach_eligible_minimum": (
            "unsigned_disagreement_capacity_below_registered_eligible_minimum",
            f"The complete frozen role has {disagreement_count} unsigned "
            f"disagreements, below the registered eligible minimum {eligible_min}.",
        ),
        "unsigned_disagreements_can_reach_roof_minimum": (
            "unsigned_disagreement_capacity_below_registered_roof_minimum",
            f"The complete frozen role has {disagreement_count} unsigned "
            f"disagreements, below the registered roof minimum {roof_min}.",
        ),
        "unsigned_disagreements_can_reach_facade_minimum": (
            "unsigned_disagreement_capacity_below_registered_facade_minimum",
            f"The complete frozen role has {disagreement_count} unsigned "
            f"disagreements, below the registered facade minimum {facade_min}.",
        ),
        "unsigned_disagreements_can_reach_disjoint_stratum_minima": (
            "unsigned_disagreement_capacity_below_combined_stratum_minima",
            f"The complete frozen role has {disagreement_count} unsigned "
            f"disagreements, below the combined disjoint roof/facade minima "
            f"{roof_min + facade_min}.",
        ),
        "disagreement_blocks_can_reach_eligible_minimum": (
            "disagreement_block_capacity_below_registered_minimum",
            f"Unsigned disagreements occupy {disagreement_block_count} spatial "
            f"blocks, below the registered eligible-block minimum {blocks_min}.",
        ),
        "disagreement_blocks_can_reach_each_stratum_minimum": (
            "disagreement_block_capacity_below_per_stratum_minimum",
            f"Unsigned disagreements occupy {disagreement_block_count} spatial "
            f"blocks, below the registered per-stratum block minimum "
            f"{blocks_per_stratum_min}.",
        ),
    }
    reasons = [
        {"code": reason_specs[name][0], "detail": reason_specs[name][1]}
        for name, passed in capacity_checks.items()
        if not passed
    ]
    structurally_futile = bool(reasons)

    artifact_binding: dict[str, object] = {
        "schema_version": _FUTILITY_BINDING_SCHEMA,
        "protocol_version": protocol_version,
        "protocol_role": expected_role,
        "protocol_config_sha256": expected_hash,
        "protocol_config_canonical_sha256": _protocol_sha256(bound_protocol),
        **prediction_identity,
        "unsigned_disagreement_pair_count": disagreement_count,
        "unsigned_disagreement_pair_ids_sha256": _id_set_digest(
            disagreement_ids
        ),
        "unsigned_disagreement_spatial_block_count": disagreement_block_count,
        "unsigned_disagreement_spatial_block_ids_sha256": _id_set_digest(
            disagreement_blocks
        ),
        "registered_mechanism_eligible_minimum": eligible_min,
    }
    artifact_binding["binding_sha256"] = _artifact_digest(
        artifact_binding, "binding_sha256"
    )
    result: dict[str, object] = {
        "schema_version": _FUTILITY_SCHEMA,
        "protocol_version": protocol_version,
        "protocol_role": expected_role,
        "protocol_config_sha256": expected_hash,
        "protocol_config_canonical_sha256": _protocol_sha256(bound_protocol),
        "phase_status": (
            STOP_PRE_ANNOTATION_FUTILITY
            if structurally_futile
            else CONTINUE_TO_BLINDED_ANNOTATION
        ),
        "gate_status": (
            STOP_PRE_UNBLIND_GATE
            if structurally_futile
            else CONTINUE_TO_BLINDED_ANNOTATION
        ),
        "annotation_required": not structurally_futile,
        "observed_outcome_accessed": False,
        "signed_direction_constructed": False,
        "directional_hypotheses_evaluated": False,
        "h_b1_result": None,
        "h_b2_result": None,
        "unsigned_disagreement_pair_count": disagreement_count,
        "unsigned_disagreement_spatial_block_count": disagreement_block_count,
        "mechanism_eligible_pair_count_upper_bound": disagreement_count,
        "mechanism_eligible_spatial_block_count_upper_bound": disagreement_block_count,
        "registered_mechanism_eligible_minimum": eligible_min,
        "registered_roof_eligible_minimum": roof_min,
        "registered_facade_eligible_minimum": facade_min,
        "registered_eligible_spatial_block_minimum": blocks_min,
        "registered_blocks_per_stratum_minimum": blocks_per_stratum_min,
        "capacity_audit": {
            "complete_role_pair_count": role_pair_count,
            "complete_role_spatial_block_count": role_block_count,
            "checks": capacity_checks,
            "passed": not structurally_futile,
        },
        "structurally_futile": structurally_futile,
        "artifact_binding": artifact_binding,
        "reasons": reasons,
    }
    result["futility_artifact_sha256"] = _artifact_digest(
        result, "futility_artifact_sha256"
    )
    _canonical_json(result)
    return result


def _verify_gate_artifact(
    gate: Mapping[str, object],
    *,
    require_pass: bool,
    expected_protocol_summary_sha256: str | None = None,
    expected_prediction_csv_sha256: str | None = None,
    expected_prediction_metadata_sha256: str | None = None,
) -> Mapping[str, object]:
    if not isinstance(gate, Mapping):
        raise TypeError("gate must be a mapping returned by pre_unblinding_gate")
    if gate.get("schema_version") != _GATE_SCHEMA:
        raise ValueError("Gate artifact has the wrong schema_version")
    recorded_digest = str(gate.get("gate_artifact_sha256", "")).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", recorded_digest) or recorded_digest != (
        _artifact_digest(gate, "gate_artifact_sha256")
    ):
        raise ValueError("Gate artifact digest is missing or invalid")
    binding = gate.get("artifact_binding")
    if not isinstance(binding, Mapping) or binding.get("schema_version") != _BINDING_SCHEMA:
        raise ValueError("Gate artifact binding is missing or has the wrong schema")
    binding_digest = str(binding.get("binding_sha256", "")).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", binding_digest) or binding_digest != (
        _artifact_digest(binding, "binding_sha256")
    ):
        raise ValueError("Gate artifact binding digest is invalid")
    for field in ("protocol_version", "protocol_role", "protocol_config_sha256"):
        if gate.get(field) != binding.get(field):
            raise ValueError(f"Gate artifact and binding disagree on {field}")
    if gate.get("reference_protocol_sha256") != binding.get(
        "protocol_config_sha256"
    ):
        raise ValueError("Gate reference/config identity is inconsistent")
    summary_sha256 = str(binding.get("protocol_summary_sha256", "")).casefold()
    trusted_summary_sha256 = str(
        binding.get("trusted_protocol_summary_sha256", "")
    ).casefold()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", summary_sha256)
        or summary_sha256 != trusted_summary_sha256
    ):
        raise ValueError("Gate binding lacks a trusted protocol-summary anchor")
    if expected_protocol_summary_sha256 is not None:
        expected_summary = _trusted_sha256(
            expected_protocol_summary_sha256,
            "expected_protocol_summary_sha256",
        )
        if summary_sha256 != expected_summary:
            raise ValueError("Gate does not match expected_protocol_summary_sha256")
    trusted_prediction_sha256 = str(
        binding.get("trusted_prediction_csv_sha256", "")
    ).casefold()
    trusted_metadata_sha256 = str(
        binding.get("trusted_prediction_metadata_sha256", "")
    ).casefold()
    if (
        binding.get("prediction_csv_sha256") != trusted_prediction_sha256
        or not re.fullmatch(r"[0-9a-f]{64}", trusted_prediction_sha256)
        or binding.get("prediction_metadata_sha256") != trusted_metadata_sha256
        or not re.fullmatch(r"[0-9a-f]{64}", trusted_metadata_sha256)
    ):
        raise ValueError("Gate binding lacks trusted prediction artifact anchors")
    if expected_prediction_csv_sha256 is not None and trusted_prediction_sha256 != (
        _trusted_sha256(
            expected_prediction_csv_sha256, "expected_prediction_csv_sha256"
        )
    ):
        raise ValueError("Gate does not match expected_prediction_csv_sha256")
    if expected_prediction_metadata_sha256 is not None and trusted_metadata_sha256 != (
        _trusted_sha256(
            expected_prediction_metadata_sha256,
            "expected_prediction_metadata_sha256",
        )
    ):
        raise ValueError("Gate does not match expected_prediction_metadata_sha256")
    if int(gate.get("prediction_seed_count", -1)) != 5:
        raise ValueError("Gate artifact does not bind the frozen five-seed ensemble")
    if int(binding.get("reference_pair_count", -1)) != int(
        binding.get("disagreement_pair_count", -2)
    ):
        raise ValueError("Gate binding does not cover exactly all disagreements")
    contract = gate.get("analysis_contract")
    if not isinstance(contract, Mapping) or contract.get("schema_version") != _ANALYSIS_SCHEMA:
        raise ValueError("Gate artifact has no valid frozen analysis contract")
    if (
        int(contract.get("bootstrap_replicates", -1)) != _EXPECTED_ANALYSIS_BOOTSTRAPS
        or int(contract.get("bootstrap_seed", -1))
        != _EXPECTED_ANALYSIS_BOOTSTRAP_SEED
    ):
        raise ValueError("Gate artifact changed the frozen confirmation bootstrap")
    role = gate.get("protocol_role")
    if role == "spatial_confirmation":
        authorization = _verify_confirmation_authorization_artifact(
            gate.get("development_pass_authorization"),
            expected_protocol_sha256=str(gate.get("protocol_config_sha256", "")),
            expected_protocol_summary_sha256=summary_sha256,
        )
        if gate.get("development_authorization_artifact_sha256") != authorization.get(
            "authorization_artifact_sha256"
        ):
            raise ValueError("Confirmation gate authorization digest is inconsistent")
        if binding.get("role_manifest_csv_sha256") != authorization.get(
            "confirmation_manifest_sha256"
        ):
            raise ValueError(
                "Confirmation gate manifest is not bound by development authorization"
            )
    elif role == PROTOCOL_ROLE:
        if gate.get("development_pass_authorization") is not None or gate.get(
            "development_authorization_artifact_sha256"
        ) is not None:
            raise ValueError(
                "Development gate cannot carry a confirmation authorization"
            )
    else:
        raise ValueError("Gate artifact has an unsupported protocol role")
    if require_pass:
        if gate.get("gate_status") != PASS_PRE_UNBLIND_GATE:
            raise ValueError(
                "Confirmation decision requires PASS_PRE_UNBLIND_GATE; a STOP gate "
                "cannot be converted to NO_GO"
            )
        if gate.get("execution_mode") != PROTOCOL_EXECUTION:
            raise ValueError("A non-protocol test override can never authorize confirmation")
        if gate.get("observed_outcome_accessed") is not False:
            raise ValueError("Gate artifact accessed an observed directional outcome")
        if gate.get("reference_covers_all_disagreements") is not True:
            raise ValueError("Gate artifact lacks complete disagreement references")
        reliability = gate.get("annotation_reliability")
        layout = gate.get("eligible_layout")
        power = gate.get("power_simulation")
        if not all(
            isinstance(item, Mapping) and item.get("passed") is True
            for item in (reliability, layout, power)
        ):
            raise ValueError("Gate status is inconsistent with failed gate subchecks")
        if power.get("executed") is not True:
            raise ValueError("A passing gate must execute frozen power simulation")
        if gate.get("reasons") not in ([], ()):  # no STOP/override reason may remain
            raise ValueError("A passing gate artifact cannot contain failure reasons")
    return gate


def _verify_confirmation_authorization_artifact(
    authorization: object,
    *,
    expected_protocol_sha256: str,
    expected_protocol_summary_sha256: str,
) -> Mapping[str, object]:
    """Verify the development PASS that alone unlocks confirmation access."""

    if not isinstance(authorization, Mapping):
        raise ValueError(
            "A bound study_development PASS authorization artifact is required "
            "before spatial_confirmation may be read or scored"
        )
    if authorization.get("schema_version") != _AUTHORIZATION_SCHEMA:
        raise ValueError("Confirmation authorization has the wrong schema_version")
    recorded = str(
        authorization.get("authorization_artifact_sha256", "")
    ).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", recorded) or recorded != _artifact_digest(
        authorization, "authorization_artifact_sha256"
    ):
        raise ValueError("Confirmation authorization digest is missing or invalid")
    if authorization.get("authorization_status") != _AUTHORIZATION_STATUS:
        raise ValueError("Confirmation authorization status is invalid")
    expected_protocol = _trusted_sha256(
        expected_protocol_sha256, "expected_protocol_sha256"
    )
    expected_summary = _trusted_sha256(
        expected_protocol_summary_sha256, "expected_protocol_summary_sha256"
    )
    if authorization.get("protocol_config_sha256") != expected_protocol:
        raise ValueError("Confirmation authorization protocol hash mismatch")
    if authorization.get("protocol_summary_sha256") != expected_summary:
        raise ValueError("Confirmation authorization summary hash mismatch")
    confirmation_manifest_sha256 = str(
        authorization.get("confirmation_manifest_sha256", "")
    ).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", confirmation_manifest_sha256):
        raise ValueError("Confirmation authorization has no frozen manifest hash")

    development_gate = authorization.get("development_gate")
    verified_development = _verify_gate_artifact(
        development_gate,
        require_pass=True,
        expected_protocol_summary_sha256=expected_summary,
    )
    if verified_development.get("protocol_role") != PROTOCOL_ROLE:
        raise ValueError(
            "Confirmation authorization must descend from study_development"
        )
    if verified_development.get("protocol_config_sha256") != expected_protocol:
        raise ValueError("Development authorization protocol hash mismatch")
    binding = verified_development["artifact_binding"]
    required_parent_fields = {
        "development_gate_artifact_sha256": verified_development.get(
            "gate_artifact_sha256"
        ),
        "development_binding_sha256": binding.get("binding_sha256"),
        "development_prediction_csv_sha256": binding.get(
            "prediction_csv_sha256"
        ),
        "development_prediction_metadata_sha256": binding.get(
            "prediction_metadata_sha256"
        ),
    }
    for field, expected in required_parent_fields.items():
        if authorization.get(field) != expected:
            raise ValueError(
                f"Confirmation authorization does not match development {field}"
            )
    return authorization


def build_confirmation_authorization_artifact(
    development_gate: Mapping[str, object],
    *,
    expected_protocol_summary_sha256: str,
    expected_prediction_csv_sha256: str,
    expected_prediction_metadata_sha256: str,
    protocol_summary_bytes: bytes | None = None,
    protocol_summary_path: str | Path | None = None,
) -> dict[str, object]:
    """Authorize one frozen confirmation manifest after a genuine dev PASS.

    The full verified development gate is embedded so a caller cannot turn a
    STOP into authorization merely by editing ``gate_status`` and recomputing
    an unkeyed JSON digest.  Confirmation remains bound to the same externally
    committed protocol summary that the development gate consumed.
    """

    trusted_summary = _trusted_sha256(
        expected_protocol_summary_sha256, "expected_protocol_summary_sha256"
    )
    verified = _verify_gate_artifact(
        development_gate,
        require_pass=True,
        expected_protocol_summary_sha256=trusted_summary,
        expected_prediction_csv_sha256=expected_prediction_csv_sha256,
        expected_prediction_metadata_sha256=expected_prediction_metadata_sha256,
    )
    if verified.get("protocol_role") != PROTOCOL_ROLE:
        raise ValueError(
            "Only a genuine study_development PASS can authorize confirmation"
        )
    raw_summary = _artifact_bytes(
        data=protocol_summary_bytes,
        path=protocol_summary_path,
        name="protocol summary",
    )
    if _sha256_bytes(raw_summary) != trusted_summary:
        raise ValueError(
            "Raw protocol summary SHA-256 does not match "
            "expected_protocol_summary_sha256"
        )
    binding = verified["artifact_binding"]
    if binding.get("protocol_summary_sha256") != trusted_summary:
        raise ValueError("Development gate consumed a different protocol summary")
    summary = _json_object_bytes(raw_summary, "protocol summary")
    if summary.get("schema_version") != (
        "eaton-component-direction-spatial-summary-v1"
    ):
        raise ValueError("Protocol summary has the wrong schema_version")
    if summary.get("protocol_config_sha256") != verified.get(
        "protocol_config_sha256"
    ):
        raise ValueError("Protocol summary/config identity mismatch")
    if summary.get("analysis_status") != "PROTOCOL_READY_CONFIRMATION_UNTOUCHED":
        raise ValueError("Protocol summary no longer attests untouched confirmation")
    commitment = summary.get("confirmation_commitment")
    if not isinstance(commitment, Mapping) or commitment.get("status") != (
        "UNSCORED_RESERVED_SAME_EVENT_SPATIAL_CONFIRMATION"
    ):
        raise ValueError("Protocol summary has no untouched confirmation commitment")
    confirmation_manifest_sha256 = _trusted_sha256(
        commitment.get("manifest_sha256"),
        "confirmation_commitment.manifest_sha256",
    )
    role_hashes = summary.get("role_manifest_sha256")
    if (
        not isinstance(role_hashes, Mapping)
        or role_hashes.get("spatial_confirmation")
        != confirmation_manifest_sha256
    ):
        raise ValueError("Confirmation commitment and role manifest hash disagree")

    artifact: dict[str, object] = {
        "schema_version": _AUTHORIZATION_SCHEMA,
        "authorization_status": _AUTHORIZATION_STATUS,
        "protocol_version": verified.get("protocol_version"),
        "protocol_config_sha256": verified.get("protocol_config_sha256"),
        "protocol_summary_sha256": trusted_summary,
        "confirmation_manifest_sha256": confirmation_manifest_sha256,
        "development_gate_artifact_sha256": verified.get(
            "gate_artifact_sha256"
        ),
        "development_binding_sha256": binding.get("binding_sha256"),
        "development_prediction_csv_sha256": binding.get(
            "prediction_csv_sha256"
        ),
        "development_prediction_metadata_sha256": binding.get(
            "prediction_metadata_sha256"
        ),
        "development_gate": dict(verified),
    }
    artifact["authorization_artifact_sha256"] = _artifact_digest(
        artifact, "authorization_artifact_sha256"
    )
    return artifact


def _one_statistics_row(stats: object) -> Mapping[str, object]:
    if hasattr(stats, "statistics"):
        stats = getattr(stats, "statistics")
    if isinstance(stats, pd.DataFrame):
        if len(stats) != 1:
            raise ValueError("Confirmation statistics must contain exactly one row")
        return stats.iloc[0].to_dict()
    if isinstance(stats, pd.Series):
        return stats.to_dict()
    if isinstance(stats, Mapping):
        return stats
    raise TypeError("stats must be a one-row DataFrame, Series, mapping, or analysis result")


def build_confirmation_analysis_artifact(
    gate: Mapping[str, object],
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    protocol_config: Mapping[str, object],
    *,
    expected_protocol_sha256: str,
    expected_protocol_summary_sha256: str,
    expected_prediction_csv_sha256: str,
    expected_prediction_metadata_sha256: str,
    protocol_config_bytes: bytes | None = None,
    protocol_config_path: str | Path | None = None,
    prediction_csv_bytes: bytes | None = None,
    prediction_csv_path: str | Path | None = None,
    prediction_metadata_bytes: bytes | None = None,
    prediction_metadata_path: str | Path | None = None,
    role_manifest_csv_bytes: bytes | None = None,
    role_manifest_path: str | Path | None = None,
    protocol_summary_bytes: bytes | None = None,
    protocol_summary_path: str | Path | None = None,
) -> dict[str, object]:
    """Recompute and bind the frozen confirmation analysis from exact gate inputs."""

    verified_gate = _verify_gate_artifact(
        gate,
        require_pass=True,
        expected_protocol_summary_sha256=expected_protocol_summary_sha256,
        expected_prediction_csv_sha256=expected_prediction_csv_sha256,
        expected_prediction_metadata_sha256=expected_prediction_metadata_sha256,
    )
    if verified_gate.get("protocol_role") != "spatial_confirmation":
        raise ValueError(
            "Only a spatial_confirmation gate can bind a confirmation analysis"
        )
    binding = verified_gate["artifact_binding"]
    contract = verified_gate["analysis_contract"]
    layout = verified_gate["eligible_layout"]
    bound_protocol, _raw_protocol, expected_hash = _resolve_protocol_artifact(
        protocol_config,
        expected_protocol_sha256=expected_protocol_sha256,
        protocol_config_bytes=protocol_config_bytes,
        protocol_config_path=protocol_config_path,
    )
    parsed = _validate_protocol_config(bound_protocol)
    if expected_hash != verified_gate.get("protocol_config_sha256"):
        raise ValueError("Confirmation protocol does not match the gate")
    if parsed["analysis_contract"] != contract:
        raise ValueError("Confirmation analysis contract does not match the gate")
    protocol_version = str(bound_protocol.get("protocol_version", "")).strip()
    validated_predictions, _manifest, prediction_identity = (
        _validated_prediction_population(
            predictions,
            expected_role="spatial_confirmation",
            expected_protocol_sha256=expected_hash,
            expected_protocol_summary_sha256=expected_protocol_summary_sha256,
            expected_prediction_csv_sha256=expected_prediction_csv_sha256,
            expected_prediction_metadata_sha256=expected_prediction_metadata_sha256,
            protocol_version=protocol_version,
            expected_seeds=tuple(parsed["expected_seeds"]),
            prediction_csv_bytes=prediction_csv_bytes,
            prediction_csv_path=prediction_csv_path,
            prediction_metadata_bytes=prediction_metadata_bytes,
            prediction_metadata_path=prediction_metadata_path,
            role_manifest_csv_bytes=role_manifest_csv_bytes,
            role_manifest_path=role_manifest_path,
            protocol_summary_bytes=protocol_summary_bytes,
            protocol_summary_path=protocol_summary_path,
        )
    )
    for field, value in prediction_identity.items():
        if binding.get(field) != value:
            raise ValueError(
                f"Confirmation prediction artifact does not match gate binding: {field}"
            )
    reference = validate_component_reference(references)
    if not reference["protocol_version"].eq(protocol_version).all() or not reference[
        "protocol_sha256"
    ].eq(expected_hash).all():
        raise ValueError("Confirmation reference protocol identity mismatch")
    if _frame_digest(reference, sort_by=("pair_id",)) != binding.get(
        "reference_table_sha256"
    ):
        raise ValueError("Confirmation reference does not match gate binding")
    ensemble = ensemble_seed_predictions(
        validated_predictions, expected_role="spatial_confirmation"
    )
    disagreement_ids = ensemble.loc[
        ensemble["street_prediction"].ne(ensemble["overhead_prediction"]), "pair_id"
    ]
    if (
        _id_set_digest(disagreement_ids)
        != binding.get("disagreement_pair_ids_sha256")
        or set(disagreement_ids.astype(str)) != set(reference["pair_id"].astype(str))
    ):
        raise ValueError("Confirmation disagreement population does not match gate binding")

    analysis_result = analyze_component_direction(
        validated_predictions,
        reference,
        bootstrap_replicates=int(contract["bootstrap_replicates"]),
        bootstrap_seed=int(contract["bootstrap_seed"]),
        expected_role="spatial_confirmation",
    )
    row = dict(_one_statistics_row(analysis_result))
    row.update(
        {
            "protocol_role": "spatial_confirmation",
            "protocol_config_sha256": expected_hash,
            "prediction_table_sha256": binding["prediction_table_sha256"],
            "reference_table_sha256": binding["reference_table_sha256"],
            "bootstrap_seed": contract["bootstrap_seed"],
        }
    )
    required_identity = {
        "protocol_role": "spatial_confirmation",
        "protocol_config_sha256": verified_gate["protocol_config_sha256"],
        "prediction_table_sha256": binding["prediction_table_sha256"],
        "reference_table_sha256": binding["reference_table_sha256"],
        "bootstrap_seed": contract["bootstrap_seed"],
        "bootstrap_replicates_requested": contract["bootstrap_replicates"],
    }
    for field, expected in required_identity.items():
        if row.get(field) != expected:
            raise ValueError(
                f"Confirmation statistics {field} does not match the frozen gate"
            )
    expected_counts = {
        "ensemble_pair_count": binding["prediction_pair_count"],
        "seed_count": 5,
        "spatial_block_count": binding["prediction_spatial_block_count"],
        "disagreement_pair_count": binding["disagreement_pair_count"],
        "mechanism_eligible_pair_count": layout["mechanism_eligible_count"],
        "mechanism_eligible_disagreement_count": layout[
            "mechanism_eligible_count"
        ],
        "mechanism_eligible_spatial_block_count": layout[
            "eligible_spatial_block_count"
        ],
        "roof_eligible_disagreement_count": layout["roof_eligible_count"],
        "facade_eligible_disagreement_count": layout["facade_eligible_count"],
        "roof_eligible_spatial_block_count": layout[
            "roof_eligible_spatial_block_count"
        ],
        "facade_eligible_spatial_block_count": layout[
            "facade_eligible_spatial_block_count"
        ],
    }
    for field, expected in expected_counts.items():
        try:
            actual = int(row.get(field, -1))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Confirmation statistics {field} must be an integer") from error
        if actual != int(expected):
            raise ValueError(
                f"Confirmation statistics {field} does not match the gate population"
            )
    support_value = row.get("severity_standardization_common_support")
    expected_support = layout.get("severity_standardization_common_support")
    if (
        not isinstance(support_value, (bool, np.bool_))
        or bool(support_value) != bool(expected_support)
    ):
        raise ValueError("Confirmation common-support state does not match the gate")
    expected_coverage = float(
        int(layout["mechanism_eligible_count"])
        / int(binding["disagreement_pair_count"])
    )
    supplied_coverage = _finite_float(
        row.get("mechanism_eligibility_coverage_among_disagreements"),
        "mechanism_eligibility_coverage_among_disagreements",
    )
    if not math.isclose(
        supplied_coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Confirmation mechanism coverage does not match the gate layout")
    target_weights = layout.get("target_weights", {})
    stratum_counts = layout.get("stratum_target_counts", {})
    for target in range(len(SEVERITY_CLASS_ORDER)):
        weight_field = f"target_{target}_standardization_weight"
        actual_weight = _finite_float(row.get(weight_field), weight_field)
        expected_weight = _finite_float(target_weights.get(str(target)), weight_field)
        if not math.isclose(actual_weight, expected_weight, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Confirmation target weights do not match the gate layout")
        for stratum in ("roof", "facade"):
            count_field = f"{stratum}_target_{target}_eligible_disagreement_count"
            if int(row.get(count_field, -1)) != int(
                stratum_counts.get(stratum, {}).get(str(target), -2)
            ):
                raise ValueError(
                    "Confirmation stratum/target counts do not match the gate layout"
                )

    normalized_row = {str(key): _json_scalar(value) for key, value in row.items()}
    statistics_sha256 = hashlib.sha256(
        _canonical_json(normalized_row).encode("utf-8")
    ).hexdigest()
    artifact: dict[str, object] = {
        "schema_version": _ANALYSIS_SCHEMA,
        "protocol_version": verified_gate["protocol_version"],
        "protocol_role": "spatial_confirmation",
        "protocol_config_sha256": verified_gate["protocol_config_sha256"],
        "gate_artifact_sha256": verified_gate["gate_artifact_sha256"],
        "development_authorization_artifact_sha256": verified_gate[
            "development_authorization_artifact_sha256"
        ],
        "artifact_binding": dict(binding),
        "analysis_contract": dict(contract),
        "statistics": normalized_row,
        "statistics_sha256": statistics_sha256,
    }
    artifact["analysis_artifact_sha256"] = _artifact_digest(
        artifact, "analysis_artifact_sha256"
    )
    return artifact


def _verify_analysis_artifact(
    gate: Mapping[str, object], analysis: Mapping[str, object]
) -> Mapping[str, object]:
    if not isinstance(analysis, Mapping) or analysis.get("schema_version") != _ANALYSIS_SCHEMA:
        raise ValueError("Confirmation analysis artifact has the wrong schema")
    recorded = str(analysis.get("analysis_artifact_sha256", "")).strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", recorded) or recorded != _artifact_digest(
        analysis, "analysis_artifact_sha256"
    ):
        raise ValueError("Confirmation analysis artifact digest is invalid")
    if analysis.get("protocol_role") != "spatial_confirmation":
        raise ValueError("Confirmation analysis must have protocol_role=spatial_confirmation")
    for field in (
        "protocol_version",
        "protocol_config_sha256",
        "gate_artifact_sha256",
        "development_authorization_artifact_sha256",
        "analysis_contract",
        "artifact_binding",
    ):
        expected = (
            gate.get(field)
            if field != "gate_artifact_sha256"
            else gate.get("gate_artifact_sha256")
        )
        if analysis.get(field) != expected:
            raise ValueError(f"Gate and confirmation analysis disagree on {field}")
    statistics = analysis.get("statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("Confirmation analysis artifact has no statistics row")
    expected_stats_digest = hashlib.sha256(
        _canonical_json(dict(statistics)).encode("utf-8")
    ).hexdigest()
    if analysis.get("statistics_sha256") != expected_stats_digest:
        raise ValueError("Confirmation statistics digest is invalid")
    return analysis


def confirmation_decision(
    gate: Mapping[str, object],
    stats: object,
    *,
    expected_protocol_summary_sha256: str,
    expected_prediction_csv_sha256: str,
    expected_prediction_metadata_sha256: str,
) -> dict[str, object]:
    """Return GO/NO-GO only after a genuine registered gate pass.

    A failed or test-override gate is a STOP state, not a negative mechanism
    result, so this function refuses to turn it into ``NO_GO``.
    """

    verified_gate = _verify_gate_artifact(
        gate,
        require_pass=True,
        expected_protocol_summary_sha256=expected_protocol_summary_sha256,
        expected_prediction_csv_sha256=expected_prediction_csv_sha256,
        expected_prediction_metadata_sha256=expected_prediction_metadata_sha256,
    )
    if verified_gate.get("protocol_role") != "spatial_confirmation":
        raise ValueError(
            "Confirmation decision requires a spatial_confirmation gate, not "
            "study_development"
        )
    if not isinstance(stats, Mapping):
        raise TypeError("stats must be a bound confirmation analysis artifact")
    analysis = _verify_analysis_artifact(verified_gate, stats)
    thresholds = verified_gate.get("confirmation_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("Gate artifact is missing frozen confirmation thresholds")
    required_thresholds = {
        "delta_min",
        "ci_lower_strict_min",
        "roof_share_strict_min",
        "facade_share_strict_max",
        "coverage_min",
    }
    if not required_thresholds.issubset(thresholds):
        raise ValueError("Gate artifact has incomplete confirmation thresholds")

    row = analysis["statistics"]
    required_stats = {
        "severity_standardization_common_support",
        "h_b1_directional_contrast",
        "h_b1_directional_contrast_ci_low",
        "roof_standardized_positive_direction_share",
        "facade_standardized_positive_direction_share",
        "mechanism_eligibility_coverage_among_disagreements",
    }
    missing = sorted(required_stats - set(row))
    if missing:
        raise ValueError(f"Confirmation statistics missing fields: {missing}")

    support_value = row["severity_standardization_common_support"]
    common_support = bool(
        isinstance(support_value, (bool, np.bool_)) and bool(support_value)
    )

    def numeric_or_none(name: str) -> float | None:
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    delta = numeric_or_none("h_b1_directional_contrast")
    ci_low = numeric_or_none("h_b1_directional_contrast_ci_low")
    roof_share = numeric_or_none("roof_standardized_positive_direction_share")
    facade_share = numeric_or_none("facade_standardized_positive_direction_share")
    coverage = numeric_or_none("mechanism_eligibility_coverage_among_disagreements")
    criteria = {
        "severity_standardization_common_support": common_support,
        "standardized_delta_at_least_minimum": (
            delta is not None and delta >= float(thresholds["delta_min"])
        ),
        "bootstrap_lower_bound_above_zero": (
            ci_low is not None
            and ci_low > float(thresholds["ci_lower_strict_min"])
        ),
        "roof_share_above_half": (
            roof_share is not None
            and 0.0 <= roof_share <= 1.0
            and roof_share > float(thresholds["roof_share_strict_min"])
        ),
        "facade_share_below_half": (
            facade_share is not None
            and 0.0 <= facade_share <= 1.0
            and facade_share < float(thresholds["facade_share_strict_max"])
        ),
        "mechanism_coverage_at_least_minimum": (
            coverage is not None
            and 0.0 <= coverage <= 1.0
            and coverage >= float(thresholds["coverage_min"])
        ),
    }
    passed = all(criteria.values())
    decision = GO_CONFIRMATION if passed else NO_GO_CONFIRMATION
    return {
        "schema_version": _DECISION_SCHEMA,
        "protocol_version": verified_gate.get("protocol_version"),
        "protocol_role": "spatial_confirmation",
        "protocol_config_sha256": verified_gate.get("protocol_config_sha256"),
        "protocol_config_canonical_sha256": verified_gate.get(
            "protocol_config_canonical_sha256"
        ),
        "reference_protocol_sha256": verified_gate.get("reference_protocol_sha256"),
        "gate_artifact_sha256": verified_gate.get("gate_artifact_sha256"),
        "development_authorization_artifact_sha256": verified_gate.get(
            "development_authorization_artifact_sha256"
        ),
        "analysis_artifact_sha256": analysis.get("analysis_artifact_sha256"),
        "artifact_binding_sha256": verified_gate["artifact_binding"].get(
            "binding_sha256"
        ),
        "gate_status": PASS_PRE_UNBLIND_GATE,
        "decision": decision,
        "criteria": criteria,
        "failed_criteria": [name for name, value in criteria.items() if not value],
        "supportive_estimands_used_to_rescue_primary": False,
    }


__all__ = [
    "CONTINUE_TO_BLINDED_ANNOTATION",
    "GO_CONFIRMATION",
    "NON_PROTOCOL_TEST_OVERRIDE",
    "NO_GO_CONFIRMATION",
    "PASS_PRE_UNBLIND_GATE",
    "PROTOCOL_EXECUTION",
    "STOP_PRE_UNBLIND_GATE",
    "STOP_PRE_ANNOTATION_FUTILITY",
    "build_confirmation_authorization_artifact",
    "build_confirmation_analysis_artifact",
    "confirmation_decision",
    "pre_annotation_futility_gate",
    "pre_unblinding_gate",
]
