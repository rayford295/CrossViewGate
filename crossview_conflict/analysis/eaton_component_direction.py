"""Protocol-bound Eaton damaged-component observability analysis.

This module deliberately stops at one research question: whether component
damage dominance and view-specific assessability explain the signed direction
of street--overhead ordinal-severity disagreement.  It does not implement a
selector, router, or deployment policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Callable, Mapping

import numpy as np
import pandas as pd


COMPONENT_DOMINANCE_STATES = (
    "roof",
    "facade",
    "mixed",
    "none_detected",
    "unknown",
)
ASSESSABILITY_STATES = ("assessable", "not_assessable", "indeterminate")
ASSESSABILITY_FIELDS = (
    "street_roof_assessability",
    "street_facade_assessability",
    "overhead_roof_assessability",
    "overhead_facade_assessability",
)
ANNOTATION_FIELDS = ("component_dominance", *ASSESSABILITY_FIELDS)
SEVERITY_CLASS_ORDER = (
    "no_or_trace_damage",
    "damaged_repairable",
    "destroyed",
)
REFERENCE_SEMANTICS = (
    "damage_dominance_and_view_specific_damage_assessability"
)
PROTOCOL_ROLE = "study_development"

PROBABILITY_COLUMNS = tuple(
    f"{view}_prob_{class_name}"
    for view in ("street", "overhead")
    for class_name in SEVERITY_CLASS_ORDER
)
RAW_ANNOTATION_COLUMNS = {
    "pair_id",
    "rater_id",
    *ANNOTATION_FIELDS,
    "protocol_version",
    "protocol_sha256",
    "blinded_to_model_outputs",
    "street_media_sha256",
    "overhead_media_sha256",
}
ADJUDICATION_COLUMNS = {
    "pair_id",
    "adjudicator_id",
    *ANNOTATION_FIELDS,
    "protocol_version",
    "protocol_sha256",
    "blinded_to_model_outputs",
    "street_media_sha256",
    "overhead_media_sha256",
    "adjudication_note",
}
PREDICTION_COLUMNS = {
    "pair_id",
    "seed",
    "spatial_block_id",
    "protocol_role",
    "target",
    "class_order",
    *PROBABILITY_COLUMNS,
}
REFERENCE_COLUMNS = {
    "pair_id",
    *ANNOTATION_FIELDS,
    "reference_semantics",
    "annotation_provenance",
    "reference_sha256",
    "protocol_version",
    "protocol_sha256",
    "street_media_sha256",
    "overhead_media_sha256",
    "adjudicated",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_RAW_MARKERS = (
    "prediction",
    "probability",
    "confidence",
    "disagreement",
    "signed_direction",
    "spatial_block",
    "protocol_role",
    "native_severity",
    "target",
    "source_label",
    "derived_label",
    "category",
    "other_rater",
    "selection_reason",
)
_PROVENANCE_SCHEMA = "eaton-component-annotation-provenance-v1"


@dataclass(frozen=True)
class ComponentDirectionAnalysis:
    """One-row statistics plus the auditable one-row-per-pair analysis table."""

    pair_level: pd.DataFrame
    statistics: pd.DataFrame


def _normalized_identifier(series: pd.Series, name: str) -> pd.Series:
    result = series.astype("string").str.strip()
    if result.isna().any() or result.eq("").any():
        raise ValueError(f"{name} cannot be null or blank")
    return result


def _normalized_category(
    series: pd.Series, name: str, allowed: tuple[str, ...]
) -> pd.Series:
    result = series.astype("string").str.strip().str.casefold()
    if result.isna().any() or result.eq("").any():
        raise ValueError(f"{name} cannot be null or blank")
    invalid = sorted(set(result.unique()) - set(allowed))
    if invalid:
        raise ValueError(f"Invalid {name} values: {invalid}")
    return result


def _normalized_sha256(series: pd.Series, name: str) -> pd.Series:
    result = series.astype("string").str.strip().str.casefold()
    invalid = result.isna() | ~result.str.fullmatch(_SHA256)
    if invalid.any():
        raise ValueError(f"{name} must contain complete 64-character SHA-256 values")
    return result


def _strict_true(series: pd.Series, name: str) -> pd.Series:
    def is_true(value: object) -> bool:
        return bool(
            isinstance(value, (bool, np.bool_))
            and bool(value)
            or isinstance(value, str)
            and value.strip().casefold() == "true"
        )

    result = series.map(is_true)
    if not result.all():
        raise ValueError(f"Every row must attest {name}=true")
    return result.astype(bool)


def _validate_expected_protocol(
    frame: pd.DataFrame,
    *,
    expected_protocol_version: str | None,
    expected_protocol_sha256: str | None,
) -> pd.DataFrame:
    result = frame.copy()
    result["protocol_version"] = _normalized_identifier(
        result["protocol_version"], "protocol_version"
    )
    result["protocol_sha256"] = _normalized_sha256(
        result["protocol_sha256"], "protocol_sha256"
    )
    if result["protocol_version"].nunique() != 1:
        raise ValueError("An annotation artifact must use one frozen protocol_version")
    if result["protocol_sha256"].nunique() != 1:
        raise ValueError("An annotation artifact must use one frozen protocol_sha256")
    if (
        expected_protocol_version is not None
        and result["protocol_version"].iat[0] != expected_protocol_version
    ):
        raise ValueError("protocol_version does not match the expected frozen protocol")
    if expected_protocol_sha256 is not None:
        expected_hash = str(expected_protocol_sha256).strip().casefold()
        if not _SHA256.fullmatch(expected_hash):
            raise ValueError("expected_protocol_sha256 is not a SHA-256 value")
        if result["protocol_sha256"].iat[0] != expected_hash:
            raise ValueError("protocol_sha256 does not match the expected frozen protocol")
    return result


def _normalize_annotation_values(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["component_dominance"] = _normalized_category(
        result["component_dominance"],
        "component_dominance",
        COMPONENT_DOMINANCE_STATES,
    )
    for column in ASSESSABILITY_FIELDS:
        result[column] = _normalized_category(
            result[column], column, ASSESSABILITY_STATES
        )
    return result


def _reject_model_outputs(frame: pd.DataFrame, allowed: set[str]) -> None:
    leaked = []
    for column in frame.columns:
        normalized = str(column).strip().casefold()
        if column in allowed:
            continue
        if any(marker in normalized for marker in _FORBIDDEN_RAW_MARKERS):
            leaked.append(str(column))
    if leaked:
        raise ValueError(
            "Blinded annotation artifacts cannot contain model/selection fields: "
            f"{sorted(leaked)}"
        )


def validate_annotation_packet(
    annotations: pd.DataFrame,
    *,
    expected_protocol_version: str | None = None,
    expected_protocol_sha256: str | None = None,
) -> pd.DataFrame:
    """Validate a long-form packet rated independently by exactly two raters.

    The two raters must be the same across the packet so nominal Cohen's kappa
    has its usual two-rater interpretation.  The blindness flag is an explicit
    protocol attestation; model-output columns are also rejected defensively.
    """

    missing = sorted(RAW_ANNOTATION_COLUMNS - set(annotations.columns))
    if missing:
        raise ValueError(f"Annotation packet missing required columns: {missing}")
    if annotations.empty:
        raise ValueError("Annotation packet cannot be empty")
    _reject_model_outputs(annotations, RAW_ANNOTATION_COLUMNS)
    result = annotations.copy()
    result["pair_id"] = _normalized_identifier(result["pair_id"], "pair_id")
    result["rater_id"] = _normalized_identifier(result["rater_id"], "rater_id")
    result = _normalize_annotation_values(result)
    result = _validate_expected_protocol(
        result,
        expected_protocol_version=expected_protocol_version,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    result["blinded_to_model_outputs"] = _strict_true(
        result["blinded_to_model_outputs"], "blinded_to_model_outputs"
    )
    for column in ("street_media_sha256", "overhead_media_sha256"):
        result[column] = _normalized_sha256(result[column], column)

    if result.duplicated(["pair_id", "rater_id"]).any():
        raise ValueError("Annotation rows must be unique by (pair_id, rater_id)")
    raters = tuple(sorted(result["rater_id"].unique()))
    if len(raters) != 2:
        raise ValueError("The packet must contain exactly two independent raters")
    expected_raters = set(raters)
    for pair_id, group in result.groupby("pair_id", sort=False):
        if set(group["rater_id"]) != expected_raters or len(group) != 2:
            raise ValueError(
                f"pair_id {pair_id!r} must have one row from each of the two raters"
            )
        for column in (
            "protocol_version",
            "protocol_sha256",
            "street_media_sha256",
            "overhead_media_sha256",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"pair_id {pair_id!r} has inconsistent {column} across raters"
                )
    return result.sort_values(["pair_id", "rater_id"]).reset_index(drop=True)


def compute_annotation_reliability(
    annotations: pd.DataFrame,
    *,
    expected_protocol_version: str | None = None,
    expected_protocol_sha256: str | None = None,
) -> pd.DataFrame:
    """Return observed agreement and nominal Cohen's kappa for every field."""

    frame = validate_annotation_packet(
        annotations,
        expected_protocol_version=expected_protocol_version,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    raters = tuple(sorted(frame["rater_id"].unique()))
    rows: list[dict[str, object]] = []
    for field in ANNOTATION_FIELDS:
        pivot = frame.pivot(index="pair_id", columns="rater_id", values=field).sort_index()
        left = pivot[raters[0]].astype(str)
        right = pivot[raters[1]].astype(str)
        observed = float(left.eq(right).mean())
        categories = (
            COMPONENT_DOMINANCE_STATES
            if field == "component_dominance"
            else ASSESSABILITY_STATES
        )
        expected = float(
            sum(
                float(left.eq(category).mean()) * float(right.eq(category).mean())
                for category in categories
            )
        )
        kappa = (
            float((observed - expected) / (1.0 - expected))
            if not math.isclose(expected, 1.0, abs_tol=1e-15)
            else float("nan")
        )
        rows.append(
            {
                "field": field,
                "pair_count": int(len(pivot)),
                "rater_a": raters[0],
                "rater_b": raters[1],
                "observed_agreement": observed,
                "expected_agreement": expected,
                "cohen_kappa": kappa,
            }
        )
    return pd.DataFrame(rows)


def _annotation_disagreement_pairs(frame: pd.DataFrame) -> set[str]:
    result: set[str] = set()
    for pair_id, group in frame.groupby("pair_id", sort=False):
        if any(group[column].nunique(dropna=False) > 1 for column in ANNOTATION_FIELDS):
            result.add(str(pair_id))
    return result


def _validate_adjudications(
    adjudications: pd.DataFrame | None,
    raw: pd.DataFrame,
    disagreement_pairs: set[str],
) -> pd.DataFrame:
    if adjudications is None:
        adjudications = pd.DataFrame(columns=sorted(ADJUDICATION_COLUMNS))
    missing = sorted(ADJUDICATION_COLUMNS - set(adjudications.columns))
    if missing and (disagreement_pairs or not adjudications.empty):
        raise ValueError(f"Adjudications missing required columns: {missing}")
    if adjudications.empty:
        if disagreement_pairs:
            raise ValueError(
                "Adjudication is required for every pair with any rater disagreement"
            )
        return pd.DataFrame(columns=sorted(ADJUDICATION_COLUMNS))
    _reject_model_outputs(adjudications, ADJUDICATION_COLUMNS)
    result = adjudications.copy()
    result["pair_id"] = _normalized_identifier(result["pair_id"], "pair_id")
    result["adjudicator_id"] = _normalized_identifier(
        result["adjudicator_id"], "adjudicator_id"
    )
    if result["pair_id"].duplicated().any():
        raise ValueError("Adjudication must contain exactly one row per disputed pair")
    supplied = set(result["pair_id"])
    if supplied != disagreement_pairs:
        raise ValueError(
            "Adjudication pair set must exactly equal the rater-disagreement pair set; "
            f"missing={sorted(disagreement_pairs - supplied)}, "
            f"unexpected={sorted(supplied - disagreement_pairs)}"
        )
    result = _normalize_annotation_values(result)
    result = _validate_expected_protocol(
        result,
        expected_protocol_version=raw["protocol_version"].iat[0],
        expected_protocol_sha256=raw["protocol_sha256"].iat[0],
    )
    result["blinded_to_model_outputs"] = _strict_true(
        result["blinded_to_model_outputs"], "blinded_to_model_outputs"
    )
    result["adjudication_note"] = _normalized_identifier(
        result["adjudication_note"], "adjudication_note"
    )
    for column in ("street_media_sha256", "overhead_media_sha256"):
        result[column] = _normalized_sha256(result[column], column)

    raw_by_pair = {pair_id: group for pair_id, group in raw.groupby("pair_id")}
    for row in result.itertuples(index=False):
        source = raw_by_pair[str(row.pair_id)]
        if str(row.adjudicator_id) in set(source["rater_id"]):
            raise ValueError("The adjudicator must be distinct from both independent raters")
        for column in (
            "protocol_version",
            "protocol_sha256",
            "street_media_sha256",
            "overhead_media_sha256",
        ):
            if str(getattr(row, column)) != str(source[column].iat[0]):
                raise ValueError(
                    f"Adjudication {column} does not match the blinded source packet"
                )
    return result.sort_values("pair_id").reset_index(drop=True)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reference_digest(row: Mapping[str, object]) -> str:
    payload = {
        "pair_id": str(row["pair_id"]),
        **{field: str(row[field]) for field in ANNOTATION_FIELDS},
        "reference_semantics": str(row["reference_semantics"]),
        "annotation_provenance": str(row["annotation_provenance"]),
        "protocol_version": str(row["protocol_version"]),
        "protocol_sha256": str(row["protocol_sha256"]),
        "street_media_sha256": str(row["street_media_sha256"]),
        "overhead_media_sha256": str(row["overhead_media_sha256"]),
        "adjudicated": bool(row["adjudicated"]),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_adjudicated_reference(
    annotations: pd.DataFrame,
    adjudications: pd.DataFrame | None = None,
    *,
    expected_protocol_version: str | None = None,
    expected_protocol_sha256: str | None = None,
) -> pd.DataFrame:
    """Resolve all rater disagreements and emit one provenance-bearing reference."""

    raw = validate_annotation_packet(
        annotations,
        expected_protocol_version=expected_protocol_version,
        expected_protocol_sha256=expected_protocol_sha256,
    )
    disagreement_pairs = _annotation_disagreement_pairs(raw)
    adjudicated = _validate_adjudications(adjudications, raw, disagreement_pairs)
    adjudicated_by_pair = adjudicated.set_index("pair_id", drop=False)
    rows: list[dict[str, object]] = []
    for pair_id, group in raw.groupby("pair_id", sort=True):
        rater_rows = []
        for rater in group.sort_values("rater_id").itertuples(index=False):
            rater_rows.append(
                {
                    "rater_id": str(rater.rater_id),
                    **{field: str(getattr(rater, field)) for field in ANNOTATION_FIELDS},
                }
            )
        was_adjudicated = str(pair_id) in disagreement_pairs
        if was_adjudicated:
            resolution = adjudicated_by_pair.loc[str(pair_id)]
            final_values = {field: str(resolution[field]) for field in ANNOTATION_FIELDS}
            adjudication_payload: dict[str, object] | None = {
                "adjudicator_id": str(resolution["adjudicator_id"]),
                "adjudication_note": str(resolution["adjudication_note"]),
                **final_values,
            }
        else:
            final_values = {field: str(group[field].iat[0]) for field in ANNOTATION_FIELDS}
            adjudication_payload = None
        provenance = {
            "schema_version": _PROVENANCE_SCHEMA,
            "pair_id": str(pair_id),
            "protocol_version": str(group["protocol_version"].iat[0]),
            "protocol_sha256": str(group["protocol_sha256"].iat[0]),
            "street_media_sha256": str(group["street_media_sha256"].iat[0]),
            "overhead_media_sha256": str(group["overhead_media_sha256"].iat[0]),
            "raw_annotations": rater_rows,
            "adjudication": adjudication_payload,
        }
        row: dict[str, object] = {
            "pair_id": str(pair_id),
            **final_values,
            "reference_semantics": REFERENCE_SEMANTICS,
            "annotation_provenance": _canonical_json(provenance),
            "protocol_version": str(group["protocol_version"].iat[0]),
            "protocol_sha256": str(group["protocol_sha256"].iat[0]),
            "street_media_sha256": str(group["street_media_sha256"].iat[0]),
            "overhead_media_sha256": str(group["overhead_media_sha256"].iat[0]),
            "adjudicated": was_adjudicated,
        }
        row["reference_sha256"] = _reference_digest(row)
        rows.append(row)
    return validate_component_reference(pd.DataFrame(rows))


def validate_component_reference(references: pd.DataFrame) -> pd.DataFrame:
    """Validate the unique adjudicated reference and its cryptographic provenance."""

    missing = sorted(REFERENCE_COLUMNS - set(references.columns))
    if missing:
        raise ValueError(f"Component reference missing required columns: {missing}")
    if references.empty:
        raise ValueError("Component reference cannot be empty")
    result = references.copy()
    result["pair_id"] = _normalized_identifier(result["pair_id"], "pair_id")
    if result["pair_id"].duplicated().any():
        raise ValueError("Component reference must be unique by pair_id")
    result = _normalize_annotation_values(result)
    result = _validate_expected_protocol(
        result,
        expected_protocol_version=None,
        expected_protocol_sha256=None,
    )
    for column in (
        "street_media_sha256",
        "overhead_media_sha256",
        "reference_sha256",
    ):
        result[column] = _normalized_sha256(result[column], column)
    if result["reference_sha256"].duplicated().any():
        raise ValueError("reference_sha256 must be unique for every pair reference")
    semantics = result["reference_semantics"].astype("string").str.strip()
    if not semantics.eq(REFERENCE_SEMANTICS).all():
        raise ValueError(f"reference_semantics must be exactly {REFERENCE_SEMANTICS!r}")
    result["reference_semantics"] = semantics
    if not result["adjudicated"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("adjudicated must contain literal booleans")
    result["adjudicated"] = result["adjudicated"].astype(bool)

    for row in result.to_dict(orient="records"):
        try:
            provenance = json.loads(str(row["annotation_provenance"]))
        except json.JSONDecodeError as error:
            raise ValueError("annotation_provenance must be valid JSON") from error
        if provenance.get("schema_version") != _PROVENANCE_SCHEMA:
            raise ValueError("annotation_provenance has the wrong schema_version")
        for column in (
            "pair_id",
            "protocol_version",
            "protocol_sha256",
            "street_media_sha256",
            "overhead_media_sha256",
        ):
            if str(provenance.get(column)) != str(row[column]):
                raise ValueError(f"annotation_provenance does not match {column}")
        raw_rows = provenance.get("raw_annotations")
        if not isinstance(raw_rows, list) or len(raw_rows) != 2:
            raise ValueError("annotation_provenance must retain exactly two raw ratings")
        raw_frame = pd.DataFrame(raw_rows)
        required_raw = {"rater_id", *ANNOTATION_FIELDS}
        if not required_raw.issubset(raw_frame.columns):
            raise ValueError("annotation_provenance raw ratings are incomplete")
        raw_frame["rater_id"] = _normalized_identifier(
            raw_frame["rater_id"], "provenance rater_id"
        )
        raw_frame = _normalize_annotation_values(raw_frame)
        if raw_frame["rater_id"].nunique() != 2:
            raise ValueError("annotation_provenance raw rater IDs must be distinct")
        adjudication = provenance.get("adjudication")
        if bool(adjudication is not None) != bool(row["adjudicated"]):
            raise ValueError("annotation_provenance adjudication state is inconsistent")
        if adjudication is None:
            for field in ANNOTATION_FIELDS:
                if raw_frame[field].nunique() != 1 or str(raw_frame[field].iat[0]) != str(
                    row[field]
                ):
                    raise ValueError(
                        "An unadjudicated reference must reproduce two agreeing ratings"
                    )
        else:
            if not isinstance(adjudication, dict):
                raise ValueError("annotation_provenance adjudication must be an object")
            adjudicator_id = str(adjudication.get("adjudicator_id", "")).strip()
            note = str(adjudication.get("adjudication_note", "")).strip()
            if (
                not adjudicator_id
                or adjudicator_id in set(raw_frame["rater_id"])
                or not note
            ):
                raise ValueError("annotation_provenance adjudicator identity is invalid")
            for field in ANNOTATION_FIELDS:
                if str(adjudication.get(field)) != str(row[field]):
                    raise ValueError(
                        "Adjudicated reference values must reproduce the adjudication"
                    )
        if _reference_digest(row) != row["reference_sha256"]:
            raise ValueError("reference_sha256 does not match the reference content")
    return result.sort_values("pair_id").reset_index(drop=True)


def _parse_class_order(value: object) -> tuple[str, ...] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return None


def validate_seed_predictions(
    predictions: pd.DataFrame,
    *,
    probability_tolerance: float = 1e-6,
    expected_role: str = PROTOCOL_ROLE,
) -> pd.DataFrame:
    """Validate complete, role-isolated per-seed probability predictions."""

    missing = sorted(PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")
    if predictions.empty:
        raise ValueError("Predictions cannot be empty")
    if probability_tolerance <= 0:
        raise ValueError("probability_tolerance must be positive")
    result = predictions.copy()
    result["pair_id"] = _normalized_identifier(result["pair_id"], "pair_id")
    result["spatial_block_id"] = _normalized_identifier(
        result["spatial_block_id"], "spatial_block_id"
    )
    expected_role = str(expected_role).strip()
    if not expected_role:
        raise ValueError("expected_role cannot be blank")
    roles = result["protocol_role"].astype("string").str.strip()
    if roles.isna().any() or not roles.eq(expected_role).all():
        raise ValueError(f"Every prediction row must have protocol_role={expected_role!r}")
    result["protocol_role"] = roles
    parsed_orders = result["class_order"].map(_parse_class_order)
    if not parsed_orders.map(lambda value: value == SEVERITY_CLASS_ORDER).all():
        raise ValueError(
            "class_order must be the exact ordinal order "
            f"{list(SEVERITY_CLASS_ORDER)}"
        )
    result["class_order"] = _canonical_json(list(SEVERITY_CLASS_ORDER))

    seeds = pd.to_numeric(result["seed"], errors="coerce")
    if seeds.isna().any() or not np.equal(seeds, np.floor(seeds)).all():
        raise ValueError("seed must contain integer values")
    result["seed"] = seeds.astype(int)
    if result["seed"].nunique() < 2:
        raise ValueError("At least two complete seeds are required for an ensemble")
    targets = pd.to_numeric(result["target"], errors="coerce")
    if targets.isna().any() or not np.equal(targets, np.floor(targets)).all():
        raise ValueError("target must contain integer ordinal class indices")
    result["target"] = targets.astype(int)
    if not result["target"].isin(range(len(SEVERITY_CLASS_ORDER))).all():
        raise ValueError("target must be one of 0, 1, or 2")
    if result.duplicated(["seed", "pair_id"]).any():
        raise ValueError("Predictions must be unique by (seed, pair_id)")

    for column in PROBABILITY_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite probabilities")
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} must lie in [0, 1]")
        result[column] = values.astype(float)
    for view in ("street", "overhead"):
        columns = [f"{view}_prob_{name}" for name in SEVERITY_CLASS_ORDER]
        totals = result[columns].sum(axis=1).to_numpy(dtype=float)
        if not np.isclose(totals, 1.0, rtol=0.0, atol=probability_tolerance).all():
            raise ValueError(f"{view} probabilities must sum to one on every row")

    seed_pair_sets = [
        frozenset(group["pair_id"])
        for _, group in result.groupby("seed", sort=True)
    ]
    if any(pair_set != seed_pair_sets[0] for pair_set in seed_pair_sets[1:]):
        raise ValueError("Every seed must contain the exact same pair_id set")
    for column in ("spatial_block_id", "target", "protocol_role", "class_order"):
        inconsistent = result.groupby("pair_id")[column].nunique(dropna=False) > 1
        if inconsistent.any():
            raise ValueError(f"{column} must be invariant across seeds for each pair_id")
    return result.sort_values(["pair_id", "seed"]).reset_index(drop=True)


def ensemble_seed_predictions(
    predictions: pd.DataFrame, *, expected_role: str = PROTOCOL_ROLE
) -> pd.DataFrame:
    """Average probabilities over seeds, retaining exactly one row per pair."""

    frame = validate_seed_predictions(predictions, expected_role=expected_role)
    metadata = frame.groupby("pair_id", sort=True).agg(
        spatial_block_id=("spatial_block_id", "first"),
        protocol_role=("protocol_role", "first"),
        target=("target", "first"),
        class_order=("class_order", "first"),
        seed_count=("seed", "nunique"),
    )
    probabilities = frame.groupby("pair_id", sort=True)[list(PROBABILITY_COLUMNS)].mean()
    result = metadata.join(probabilities).reset_index()
    for view in ("street", "overhead"):
        columns = [f"{view}_prob_{name}" for name in SEVERITY_CLASS_ORDER]
        matrix = result[columns].to_numpy(dtype=float)
        result[f"{view}_prediction"] = np.argmax(matrix, axis=1).astype(int)
        result[f"{view}_prediction_name"] = result[f"{view}_prediction"].map(
            dict(enumerate(SEVERITY_CLASS_ORDER))
        )
    if result["pair_id"].duplicated().any() or len(result) != frame["pair_id"].nunique():
        raise RuntimeError("Seed ensembling failed to produce one row per pair")
    return result


def _pair_sets_must_match(predictions: pd.DataFrame, references: pd.DataFrame) -> None:
    prediction_ids = set(predictions["pair_id"].astype(str))
    reference_ids = set(references["pair_id"].astype(str))
    if prediction_ids != reference_ids:
        only_predictions = sorted(prediction_ids - reference_ids)
        only_references = sorted(reference_ids - prediction_ids)
        raise ValueError(
            "Prediction/reference pair sets must match exactly; no silent inner-join "
            f"loss is allowed (only_predictions={only_predictions[:5]}, "
            f"only_references={only_references[:5]})"
        )


def _add_direction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["disagree"] = (
        result["street_prediction"] != result["overhead_prediction"]
    )
    result["signed_direction"] = (
        result["overhead_prediction"] - result["street_prediction"]
    )
    roof_eligible = (
        result["component_dominance"].eq("roof")
        & result["overhead_roof_assessability"].eq("assessable")
        & result["street_roof_assessability"].eq("not_assessable")
    ).fillna(False).astype(bool)
    facade_eligible = (
        result["component_dominance"].eq("facade")
        & result["street_facade_assessability"].eq("assessable")
        & result["overhead_facade_assessability"].eq("not_assessable")
    ).fillna(False).astype(bool)
    result["mechanism_stratum"] = np.select(
        [roof_eligible, facade_eligible], ["roof", "facade"], default="ineligible"
    )
    result["mechanism_eligible"] = roof_eligible | facade_eligible
    expected = result["mechanism_stratum"].map({"roof": 1, "facade": -1})
    result["direction_consistent"] = (
        result["disagree"]
        & result["mechanism_eligible"]
        & (result["signed_direction"] * expected > 0)
    )
    return result


def _positive_share(frame: pd.DataFrame) -> float:
    return float((frame["signed_direction"] > 0).mean()) if len(frame) else float("nan")


def _standardized_positive_share(
    frame: pd.DataFrame, target_weights: Mapping[int, float]
) -> float:
    value = 0.0
    for target, weight in target_weights.items():
        if weight <= 0:
            continue
        target_frame = frame[frame["target"] == target]
        if target_frame.empty:
            return float("nan")
        value += float(weight) * _positive_share(target_frame)
    return float(value)


def _metric_values(
    frame: pd.DataFrame, target_weights: Mapping[int, float]
) -> dict[str, float]:
    eligible = frame[frame["mechanism_eligible"] & frame["disagree"]]
    roof = eligible[eligible["mechanism_stratum"] == "roof"]
    facade = eligible[eligible["mechanism_stratum"] == "facade"]
    roof_raw = _positive_share(roof)
    facade_raw = _positive_share(facade)
    roof_standardized = _standardized_positive_share(roof, target_weights)
    facade_standardized = _standardized_positive_share(facade, target_weights)
    return {
        "roof_raw_positive_direction_share": roof_raw,
        "facade_raw_positive_direction_share": facade_raw,
        "roof_standardized_positive_direction_share": roof_standardized,
        "facade_standardized_positive_direction_share": facade_standardized,
        "h_b1_raw_directional_contrast": roof_raw - facade_raw,
        "h_b1_directional_contrast": roof_standardized - facade_standardized,
        "h_b2_direction_consistent_share": (
            float(eligible["direction_consistent"].mean())
            if len(eligible)
            else float("nan")
        ),
    }


def _bootstrap_intervals(
    frame: pd.DataFrame,
    target_weights: Mapping[int, float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    metric_names = (
        "h_b1_directional_contrast",
        "h_b1_raw_directional_contrast",
        "h_b2_direction_consistent_share",
    )
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    blocks = np.asarray(sorted(frame["spatial_block_id"].unique()), dtype=object)
    groups = {
        block: group.copy()
        for block, group in frame.groupby("spatial_block_id", sort=False)
    }
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        chosen = rng.choice(blocks, size=len(blocks), replace=True)
        sampled = pd.concat([groups[block] for block in chosen], ignore_index=True)
        metrics = _metric_values(sampled, target_weights)
        for name in metric_names:
            value = metrics[name]
            if math.isfinite(value):
                values[name].append(float(value))
    result: dict[str, float | int] = {}
    for name in metric_names:
        finite = values[name]
        result[f"{name}_bootstrap_finite_replicates"] = len(finite)
        if finite:
            low, high = np.percentile(finite, [2.5, 97.5])
            result[f"{name}_ci_low"] = float(low)
            result[f"{name}_ci_high"] = float(high)
        else:
            result[f"{name}_ci_low"] = float("nan")
            result[f"{name}_ci_high"] = float("nan")
    return result


def analyze_component_direction(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_260_711,
    expected_role: str = PROTOCOL_ROLE,
) -> ComponentDirectionAnalysis:
    """Run H-B1/H-B2 once on seed-ensembled, mechanism-eligible pairs.

    The primary H-B1 contrast standardizes both component strata to the pooled
    frozen target distribution among all mechanism-eligible disagreements.
    Seed rows are averaged before any direction or inferential calculation.
    """

    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    ensemble = ensemble_seed_predictions(predictions, expected_role=expected_role)
    reference = validate_component_reference(references)
    disagreement_predictions = ensemble.loc[
        ensemble["street_prediction"].ne(ensemble["overhead_prediction"])
    ].copy()
    _pair_sets_must_match(disagreement_predictions, reference)
    joined = disagreement_predictions.merge(
        reference,
        on="pair_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if (
        not joined["_merge"].eq("both").all()
        or len(joined) != len(disagreement_predictions)
    ):
        raise RuntimeError("Outer-join audit detected unexpected pair loss")
    joined = joined.drop(columns="_merge")
    joined = _add_direction_columns(joined)
    eligible_disagreements = joined[
        joined["mechanism_eligible"] & joined["disagree"]
    ]
    target_counts = eligible_disagreements["target"].value_counts().to_dict()
    denominator = int(len(eligible_disagreements))
    target_weights = {
        target: (float(target_counts.get(target, 0)) / denominator if denominator else 0.0)
        for target in range(len(SEVERITY_CLASS_ORDER))
    }
    metrics = _metric_values(joined, target_weights)
    disagreements = int(joined["disagree"].sum())
    eligible = int(len(eligible_disagreements))
    consistent = int(eligible_disagreements["direction_consistent"].sum())
    common_support = all(
        weight <= 0
        or all(
            (
                (eligible_disagreements["mechanism_stratum"] == stratum)
                & (eligible_disagreements["target"] == target)
            ).any()
            for stratum in ("roof", "facade")
        )
        for target, weight in target_weights.items()
    )
    row: dict[str, object] = {
        "ensemble_pair_count": int(len(ensemble)),
        "seed_count": int(ensemble["seed_count"].iat[0]),
        "spatial_block_count": int(ensemble["spatial_block_id"].nunique()),
        "disagreement_pair_count": disagreements,
        "disagreement_spatial_block_count": int(
            joined["spatial_block_id"].nunique()
        ),
        "mechanism_eligible_pair_count": int(joined["mechanism_eligible"].sum()),
        "mechanism_eligible_disagreement_count": eligible,
        "mechanism_eligible_spatial_block_count": int(
            eligible_disagreements["spatial_block_id"].nunique()
        ),
        "direction_consistent_count": consistent,
        "mechanism_eligibility_coverage_among_disagreements": (
            float(eligible / disagreements) if disagreements else float("nan")
        ),
        "full_disagreement_explained_fraction": (
            float(consistent / disagreements) if disagreements else float("nan")
        ),
        "roof_eligible_disagreement_count": int(
            eligible_disagreements["mechanism_stratum"].eq("roof").sum()
        ),
        "facade_eligible_disagreement_count": int(
            eligible_disagreements["mechanism_stratum"].eq("facade").sum()
        ),
        "roof_eligible_spatial_block_count": int(
            eligible_disagreements.loc[
                eligible_disagreements["mechanism_stratum"].eq("roof"),
                "spatial_block_id",
            ].nunique()
        ),
        "facade_eligible_spatial_block_count": int(
            eligible_disagreements.loc[
                eligible_disagreements["mechanism_stratum"].eq("facade"),
                "spatial_block_id",
            ].nunique()
        ),
        **{
            f"target_{target}_standardization_weight": weight
            for target, weight in target_weights.items()
        },
        **{
            f"{stratum}_target_{target}_eligible_disagreement_count": int(
                (
                    eligible_disagreements["mechanism_stratum"].eq(stratum)
                    & eligible_disagreements["target"].eq(target)
                ).sum()
            )
            for stratum in ("roof", "facade")
            for target in range(len(SEVERITY_CLASS_ORDER))
        },
        "severity_standardization_common_support": common_support,
        "bootstrap_replicates_requested": int(bootstrap_replicates),
        **metrics,
    }
    row.update(
        _bootstrap_intervals(
            joined,
            target_weights,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    )
    pair_level = joined.sort_values("pair_id").reset_index(drop=True)
    return ComponentDirectionAnalysis(
        pair_level=pair_level,
        statistics=pd.DataFrame([row]),
    )
