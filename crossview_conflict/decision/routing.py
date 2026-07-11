"""Auditable priority scoring and geodesic route simulation.

This module turns per-sample CrossViewGate evidence into a tile-level field
inspection queue.  Distances are great-circle (Haversine) distances between
coordinates.  They are a deterministic straight-line proxy and are *not* a
road-network route, travel-time estimate, or claim that a tile is accessible.

The implementation deliberately has no label requirement.  When ``target`` is
available, callers may request descriptive severe-discovery, recall, and NDCG
metrics; targets never affect the route itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088
ROUTING_SCHEMA_VERSION = "routing-mvp-v1"
DEFAULT_RANDOM_SEED = 42

POLICIES = (
    "random",
    "nearest_neighbor",
    "severity_only",
    "uncertainty_only",
    "conflict_only",
    "reliability_aware",
    "joint",
)

_LATITUDE_CANDIDATES = ("latitude", "lat")
_LONGITUDE_CANDIDATES = ("longitude", "lon", "lng")
_TILE_CANDIDATES = ("tile_id", "spatial_tile_id", "tile", "geohash")
_RISK_CANDIDATES = ("risk_score", "uncertainty")
_DISPOSITION_CANDIDATES = ("disposition", "decision_disposition")
_SEVERITY_PROBABILITY_CANDIDATES = (
    "severity_probability",
    "severe_probability",
    "severity_prob",
    "severe_prob",
    "p_severe",
    "prob_severe",
    "predicted_severity_probability",
    "predicted_severe_probability",
    "highest_severity_probability",
    "damage_severity_probability",
)
_HARD_CONFLICT_CANDIDATES = ("hard_conflict", "views_disagree")
_JS_CANDIDATES = ("js_divergence", "street_remote_js_divergence")
_PROBABILITY_FAMILIES = (
    "gate3_probability_",
    "gate2_probability_",
    "crossview_probability_",
    "remote_probability_",
    "overhead_probability_",
    "street_probability_",
)


@dataclass(frozen=True)
class RoutingInputInfo:
    """Resolved input fields and any evidence-only fallbacks."""

    latitude_source: str
    longitude_source: str
    tile_source: str
    severity_probability_source: str
    risk_score_source: str
    disposition_source: str
    conflict_source: str
    target_source: str | None
    severe_labels: tuple[Any, ...]
    labels_available: bool
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "latitude_source": self.latitude_source,
            "longitude_source": self.longitude_source,
            "tile_source": self.tile_source,
            "severity_probability_source": self.severity_probability_source,
            "risk_score_source": self.risk_score_source,
            "disposition_source": self.disposition_source,
            "conflict_source": self.conflict_source,
            "target_source": self.target_source,
            "severe_labels": list(self.severe_labels),
            "labels_available": self.labels_available,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RouteResult:
    """A route table plus its declared origin and resource limits."""

    policy: str
    stops: pd.DataFrame
    start_latitude: float
    start_longitude: float
    max_stops: int | None
    distance_budget_km: float | None
    random_seed: int

    @property
    def total_distance_km(self) -> float:
        if self.stops.empty:
            return 0.0
        return float(self.stops["cumulative_distance_km"].iloc[-1])


def haversine_km(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return great-circle distance in kilometers between two WGS84 points."""

    values = (latitude_a, longitude_a, latitude_b, longitude_b)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Haversine coordinates must be finite")
    if not -90.0 <= float(latitude_a) <= 90.0 or not -90.0 <= float(latitude_b) <= 90.0:
        raise ValueError("Latitude must be in [-90, 90]")
    if not -180.0 <= float(longitude_a) <= 180.0 or not -180.0 <= float(longitude_b) <= 180.0:
        raise ValueError("Longitude must be in [-180, 180]")

    lat_a, lon_a, lat_b, lon_b = map(math.radians, map(float, values))
    d_lat = lat_b - lat_a
    d_lon = lon_b - lon_a
    hav = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(d_lon / 2.0) ** 2
    )
    return float(2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, hav))))


def _resolve_column(
    frame: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
    field_name: str,
    *,
    required: bool,
) -> str | None:
    if requested and requested != "auto":
        if requested not in frame.columns:
            raise ValueError(f"Requested {field_name} column {requested!r} is absent")
        return requested
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValueError(
            f"Could not resolve {field_name}; expected one of {', '.join(candidates)}"
        )
    return None


def _probability_columns(frame: pd.DataFrame) -> tuple[str, list[str]] | None:
    for prefix in _PROBABILITY_FAMILIES:
        columns = [column for column in frame.columns if column.startswith(prefix)]
        if columns:
            return prefix, sorted(columns, key=_class_column_sort_key)
    return None


def _class_column_sort_key(column: str) -> tuple[int, float, str]:
    suffix = column.rsplit("probability_", 1)[-1]
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", suffix)
    if match:
        return (1, float(match.group(1)), suffix)
    severe_keyword = bool(re.search(r"destroy|severe|catastroph|major", suffix, re.I))
    return (1 if severe_keyword else 0, 0.0, suffix)


def _severity_probability_column(frame: pd.DataFrame) -> str | None:
    explicit = _resolve_column(
        frame,
        "auto",
        _SEVERITY_PROBABILITY_CANDIDATES,
        "severity probability",
        required=False,
    )
    if explicit:
        return explicit
    family = _probability_columns(frame)
    if family is None:
        return None
    return family[1][-1]


def _numeric_series(frame: pd.DataFrame, column: str, field_name: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        bad_count = int(values.isna().sum())
        raise ValueError(f"{field_name} must be finite numeric values ({bad_count} invalid rows)")
    return values.astype(float)


def _unit_interval(values: pd.Series, *, field_name: str) -> pd.Series:
    """Map a score to [0, 1] without changing valid probability-like scores."""

    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must be finite")
    minimum = float(array.min(initial=0.0))
    maximum = float(array.max(initial=0.0))
    if minimum >= 0.0 and maximum <= 1.0:
        return pd.Series(array, index=values.index, dtype=float)
    actual_min = float(array.min())
    actual_max = float(array.max())
    if math.isclose(actual_min, actual_max):
        return pd.Series(np.zeros(len(array), dtype=float), index=values.index)
    return pd.Series((array - actual_min) / (actual_max - actual_min), index=values.index)


def _parse_boolean_score(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.clip(0.0, 1.0).astype(float)
    truthy = {"1", "true", "yes", "y", "conflict", "disagree"}
    falsy = {"0", "false", "no", "n", "agree"}
    lowered = series.astype(str).str.strip().str.lower()
    unknown = ~lowered.isin(truthy | falsy)
    if unknown.any():
        raise ValueError("hard_conflict contains values that are not boolean-like")
    return lowered.isin(truthy).astype(float)


def _disposition_score(series: pd.Series) -> pd.Series:
    escalation_pattern = re.compile(
        r"defer|human|acquir|inspect|review|escalat|verify|field", re.I
    )
    acceptance_pattern = re.compile(r"accept|automatic|auto_accept|monitor", re.I)

    def score(value: Any) -> float:
        text = "" if pd.isna(value) else str(value).strip()
        if escalation_pattern.search(text):
            return 1.0
        if acceptance_pattern.search(text):
            return 0.0
        return 0.0

    return series.map(score).astype(float)


def _accept_disposition_mask(series: pd.Series) -> pd.Series:
    """Match every low-escalation acceptance spelling recognized by routing."""
    normalized = series.fillna("").astype(str).str.strip()
    escalation = normalized.str.contains(
        r"defer|human|acquir|inspect|review|escalat|verify|field",
        case=False,
        regex=True,
    )
    acceptance = normalized.str.contains(
        r"accept|automatic|auto_accept|monitor", case=False, regex=True
    )
    return acceptance & ~escalation


def infer_severe_labels(target: pd.Series) -> tuple[Any, ...]:
    """Infer the highest severity label for descriptive metrics.

    Explicit labels supplied by a caller are preferred.  This fallback chooses
    the maximum numeric value, the label with the largest numeric prefix (for
    example ``2_Destroyed``), or a destroyed/severe keyword label.
    """

    observed = target.dropna()
    if observed.empty:
        return ()
    numeric = pd.to_numeric(observed, errors="coerce")
    if numeric.notna().all():
        maximum = numeric.max()
        labels = observed[numeric == maximum].drop_duplicates().tolist()
        return tuple(labels)

    labels = observed.astype(str)
    prefixes = labels.str.extract(r"^\s*(-?\d+(?:\.\d+)?)", expand=False)
    numeric_prefixes = pd.to_numeric(prefixes, errors="coerce")
    if numeric_prefixes.notna().any():
        maximum = numeric_prefixes.max()
        return tuple(observed[numeric_prefixes == maximum].drop_duplicates().tolist())

    strongest = labels.str.contains(r"destroy|catastroph|total[_ -]?loss", case=False, regex=True)
    if strongest.any():
        return tuple(observed[strongest].drop_duplicates().tolist())
    severe = labels.str.contains(r"severe|major", case=False, regex=True)
    if severe.any():
        return tuple(observed[severe].drop_duplicates().tolist())
    # Last-resort deterministic ordinal assumption, disclosed in provenance.
    maximum_text = max(labels.tolist())
    return tuple(observed[labels == maximum_text].drop_duplicates().tolist())


def _matches_labels(target: pd.Series, labels: Sequence[Any]) -> pd.Series:
    if not labels:
        return pd.Series(False, index=target.index, dtype=bool)
    direct = target.isin(labels)
    text_labels = {str(value).strip() for value in labels}
    matches = direct | target.astype(str).str.strip().isin(text_labels)
    numeric_labels = pd.to_numeric(pd.Series(list(labels)), errors="coerce")
    if numeric_labels.notna().all():
        numeric_target = pd.to_numeric(target, errors="coerce")
        matches = matches | numeric_target.isin(numeric_labels.tolist())
    return matches


def normalize_routing_input(
    frame: pd.DataFrame,
    *,
    latitude_col: str = "auto",
    longitude_col: str = "auto",
    tile_col: str = "auto",
    risk_score_col: str = "auto",
    disposition_col: str = "auto",
    severity_probability_col: str = "auto",
    hard_conflict_col: str = "auto",
    js_divergence_col: str = "auto",
    target_col: str = "target",
    severe_labels: Sequence[Any] | None = None,
) -> tuple[pd.DataFrame, RoutingInputInfo]:
    """Resolve P0.7 decisions or P0.2 evidence into a stable routing schema.

    P0.2 evidence lacks the P0.7 ``risk_score`` and ``disposition`` fields.  In
    that compatibility mode uncertainty is derived as ``1 - max(probability)``
    from the best available gate/view family, and disposition is recorded as
    ``evidence_only``.  These fallbacks are explicit in :class:`RoutingInputInfo`.
    """

    if frame.empty:
        raise ValueError("Routing input must contain at least one row")
    source = frame.copy()
    latitude_source = _resolve_column(
        source, latitude_col, _LATITUDE_CANDIDATES, "latitude", required=True
    )
    longitude_source = _resolve_column(
        source, longitude_col, _LONGITUDE_CANDIDATES, "longitude", required=True
    )
    tile_source = _resolve_column(source, tile_col, _TILE_CANDIDATES, "tile id", required=True)
    assert latitude_source is not None and longitude_source is not None and tile_source is not None

    output = pd.DataFrame(index=source.index)
    if "sample_id" in source.columns:
        output["sample_id"] = source["sample_id"].astype(str)
    else:
        output["sample_id"] = [f"row_{index}" for index in range(len(source))]
    output["latitude"] = _numeric_series(source, latitude_source, "latitude")
    output["longitude"] = _numeric_series(source, longitude_source, "longitude")
    if (~output["latitude"].between(-90.0, 90.0)).any():
        raise ValueError("latitude must be in [-90, 90]")
    if (~output["longitude"].between(-180.0, 180.0)).any():
        raise ValueError("longitude must be in [-180, 180]")
    if source[tile_source].isna().any():
        raise ValueError("tile_id must be non-null for every row")
    output["tile_id"] = source[tile_source].astype(str)

    warnings: list[str] = []
    probability_family = _probability_columns(source)
    severity_source = (
        _severity_probability_column(source)
        if severity_probability_col == "auto"
        else _resolve_column(
            source,
            severity_probability_col,
            (),
            "severity probability",
            required=True,
        )
    )
    if severity_source is None:
        output["severity_probability"] = 0.0
        severity_source_name = "missing_default_zero"
        warnings.append(
            "No severity-probability field was found; severity-only priority is identically zero."
        )
    else:
        severity_values = _numeric_series(source, severity_source, "severity probability")
        if (~severity_values.between(0.0, 1.0)).any():
            raise ValueError("severity probability must be in [0, 1]")
        output["severity_probability"] = severity_values
        severity_source_name = severity_source

    risk_source = _resolve_column(
        source, risk_score_col, _RISK_CANDIDATES, "risk score", required=False
    )
    if risk_source is not None:
        risk_raw = _numeric_series(source, risk_source, "risk score")
        output["risk_score"] = risk_raw
        output["uncertainty_score"] = _unit_interval(risk_raw, field_name="risk score")
        risk_source_name = risk_source
    elif probability_family is not None and len(probability_family[1]) >= 2:
        probabilities = source[probability_family[1]].apply(pd.to_numeric, errors="coerce")
        if probabilities.isna().any().any():
            raise ValueError("Probability family used for P0.2 uncertainty contains invalid values")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any().any():
            raise ValueError("Probability family used for P0.2 uncertainty must be in [0, 1]")
        risk_raw = 1.0 - probabilities.max(axis=1)
        output["risk_score"] = risk_raw
        output["uncertainty_score"] = risk_raw.clip(0.0, 1.0)
        risk_source_name = f"derived_1_minus_max:{probability_family[0]}*"
        warnings.append("risk_score was derived from P0.2 probabilities; it is not risk-calibrated.")
    elif "crossview_entropy" in source.columns:
        risk_raw = _numeric_series(source, "crossview_entropy", "crossview entropy")
        output["risk_score"] = risk_raw
        output["uncertainty_score"] = _unit_interval(
            risk_raw, field_name="crossview entropy"
        )
        risk_source_name = "crossview_entropy"
        warnings.append("risk_score was substituted by P0.2 entropy; it is not risk-calibrated.")
    else:
        output["risk_score"] = 0.0
        output["uncertainty_score"] = 0.0
        risk_source_name = "missing_default_zero"
        warnings.append("No risk score or probability family was found; uncertainty is zero.")

    disposition_source = _resolve_column(
        source,
        disposition_col,
        _DISPOSITION_CANDIDATES,
        "disposition",
        required=False,
    )
    if disposition_source is None:
        output["disposition"] = "evidence_only"
        disposition_source_name = "P0.2_default:evidence_only"
        warnings.append("disposition was absent and marked evidence_only (not a calibrated action).")
    else:
        output["disposition"] = source[disposition_source].fillna("unspecified").astype(str)
        disposition_source_name = disposition_source

    # Defense in depth: an evaluated policy that did not retain a finite-sample
    # risk-control claim must never reach routing as an authorized ``accept``.
    # Current P0.7 writers already fail closed; this also protects reports built
    # from older decision artifacts.
    if "risk_control_status" in source.columns:
        statuses = source["risk_control_status"].fillna("missing").astype(str)
        output["risk_control_status"] = statuses
        unsafe_accept = (statuses != "risk-controlled") & _accept_disposition_mask(
            output["disposition"]
        )
        if unsafe_accept.any():
            output.loc[unsafe_accept, "disposition"] = "defer_human"
            warnings.append(
                f"Fail-closed override: {int(unsafe_accept.sum())} accept disposition(s) "
                "came from a non-risk-controlled policy and were changed to defer_human."
            )
    if "policy_authorized" in source.columns:
        authorized = pd.to_numeric(source["policy_authorized"], errors="coerce") == 1
        unsafe_accept = ~authorized & _accept_disposition_mask(output["disposition"])
        if unsafe_accept.any():
            output.loc[unsafe_accept, "disposition"] = "defer_human"
            warnings.append(
                f"Fail-closed override: {int(unsafe_accept.sum())} unauthorized accept "
                "disposition(s) were changed to defer_human."
            )
    elif "risk_control_status" not in source.columns:
        unaudited_accept = _accept_disposition_mask(output["disposition"])
        if unaudited_accept.any():
            output.loc[unaudited_accept, "disposition"] = "defer_human"
            warnings.append(
                f"Fail-closed override: {int(unaudited_accept.sum())} accept disposition(s) "
                "lacked both risk_control_status and policy_authorized metadata and "
                "were changed to defer_human."
            )
    output["escalation_score"] = _disposition_score(output["disposition"])

    hard_source = _resolve_column(
        source,
        hard_conflict_col,
        _HARD_CONFLICT_CANDIDATES,
        "hard conflict",
        required=False,
    )
    js_source = _resolve_column(
        source,
        js_divergence_col,
        _JS_CANDIDATES,
        "JS divergence",
        required=False,
    )
    conflict_parts: list[pd.Series] = []
    conflict_sources: list[str] = []
    if hard_source is not None:
        output["hard_conflict"] = _parse_boolean_score(source[hard_source])
        conflict_parts.append(output["hard_conflict"])
        conflict_sources.append(hard_source)
    else:
        output["hard_conflict"] = 0.0
    if js_source is not None:
        js_values = _numeric_series(source, js_source, "JS divergence")
        if (js_values < 0.0).any():
            raise ValueError("JS divergence must be non-negative")
        output["js_divergence"] = js_values
        # Natural-log JS has maximum ln(2); clipping also supports already-normalized JS.
        js_score = (js_values / math.log(2.0)).clip(0.0, 1.0)
        conflict_parts.append(js_score)
        conflict_sources.append(js_source)
    else:
        output["js_divergence"] = 0.0
    if conflict_parts:
        output["conflict_score"] = pd.concat(conflict_parts, axis=1).max(axis=1)
        conflict_source_name = "+".join(conflict_sources)
    else:
        output["conflict_score"] = 0.0
        conflict_source_name = "missing_default_zero"
        warnings.append("No hard-conflict or JS field was found; conflict priority is zero.")

    labels_available = target_col in source.columns and source[target_col].notna().any()
    resolved_severe_labels: tuple[Any, ...] = ()
    if labels_available:
        output["target"] = source[target_col]
        resolved_severe_labels = tuple(severe_labels or infer_severe_labels(source[target_col]))
        output["target_available"] = source[target_col].notna()
        output["is_severe"] = _matches_labels(source[target_col], resolved_severe_labels)
    else:
        output["target_available"] = False
        output["is_severe"] = False

    info = RoutingInputInfo(
        latitude_source=latitude_source,
        longitude_source=longitude_source,
        tile_source=tile_source,
        severity_probability_source=severity_source_name,
        risk_score_source=risk_source_name,
        disposition_source=disposition_source_name,
        conflict_source=conflict_source_name,
        target_source=target_col if labels_available else None,
        severe_labels=resolved_severe_labels,
        labels_available=labels_available,
        warnings=tuple(warnings),
    )
    return output.reset_index(drop=True), info


def _join_dispositions(values: Iterable[Any]) -> str:
    unique = sorted({str(value) for value in values if not pd.isna(value)})
    return "|".join(unique)


def _descending_rank(values: pd.Series, tile_ids: pd.Series) -> pd.Series:
    order = pd.DataFrame(
        {"value": values.astype(float), "tile_id": tile_ids.astype(str)}, index=values.index
    ).sort_values(["value", "tile_id"], ascending=[False, True], kind="mergesort")
    result = pd.Series(index=values.index, dtype="int64")
    result.loc[order.index] = np.arange(1, len(order) + 1, dtype=int)
    return result.astype(int)


def build_tile_priority(
    normalized: pd.DataFrame,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Aggregate sample evidence by tile and compute auditable priority scores."""

    required = {
        "tile_id",
        "latitude",
        "longitude",
        "severity_probability",
        "risk_score",
        "uncertainty_score",
        "conflict_score",
        "hard_conflict",
        "js_divergence",
        "disposition",
        "escalation_score",
        "target_available",
        "is_severe",
    }
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"Normalized routing input is missing: {', '.join(missing)}")

    grouped = normalized.groupby("tile_id", sort=True, dropna=False)
    tiles = grouped.agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        sample_count=("sample_id", "size"),
        severity_probability=("severity_probability", "max"),
        risk_score=("risk_score", "max"),
        uncertainty_score=("uncertainty_score", "max"),
        conflict_score=("conflict_score", "max"),
        hard_conflict=("hard_conflict", "max"),
        js_divergence=("js_divergence", "max"),
        escalation_score=("escalation_score", "max"),
        disposition=("disposition", _join_dispositions),
        labeled_sample_count=("target_available", "sum"),
        severe_sample_count=("is_severe", "sum"),
    ).reset_index()
    tiles["sample_count"] = tiles["sample_count"].astype(int)
    tiles["labeled_sample_count"] = tiles["labeled_sample_count"].astype(int)
    tiles["severe_sample_count"] = tiles["severe_sample_count"].astype(int)
    tiles["has_severe"] = (tiles["severe_sample_count"] > 0).astype(int)

    rng = np.random.default_rng(int(random_seed))
    tiles["priority_random"] = rng.random(len(tiles))
    tiles["priority_severity_only"] = tiles["severity_probability"]
    tiles["priority_uncertainty_only"] = tiles["uncertainty_score"]
    tiles["priority_conflict_only"] = tiles["conflict_score"]
    tiles["priority_reliability_aware"] = (
        0.50 * tiles["severity_probability"]
        + 0.30 * tiles["uncertainty_score"]
        + 0.20 * tiles["escalation_score"]
    )
    tiles["priority_joint"] = (
        0.40 * tiles["severity_probability"]
        + 0.25 * tiles["uncertainty_score"]
        + 0.25 * tiles["conflict_score"]
        + 0.10 * tiles["escalation_score"]
    )

    for policy in POLICIES:
        score_column = f"priority_{policy}"
        if score_column in tiles.columns:
            tiles[f"rank_{policy}"] = _descending_rank(tiles[score_column], tiles["tile_id"])
    return tiles.sort_values("tile_id", kind="mergesort").reset_index(drop=True)


def _default_start(tiles: pd.DataFrame) -> tuple[float, float]:
    return float(tiles["latitude"].mean()), float(tiles["longitude"].mean())


def _policy_score_column(policy: str) -> str | None:
    if policy == "nearest_neighbor":
        return None
    return f"priority_{policy}"


def simulate_route(
    tiles: pd.DataFrame,
    *,
    policy: str = "joint",
    max_stops: int | None = None,
    distance_budget_km: float | None = None,
    start_latitude: float | None = None,
    start_longitude: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> RouteResult:
    """Greedily simulate a deterministic route under stop and/or distance limits.

    Score policies visit the highest-priority feasible tile next.  The nearest
    neighbor baseline visits the geographically closest feasible tile next.
    Ties are broken by ``tile_id``.  A tile whose next leg cannot fit is skipped
    while other feasible tiles remain.
    """

    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; expected one of {', '.join(POLICIES)}")
    if max_stops is not None and max_stops < 0:
        raise ValueError("max_stops must be non-negative")
    if distance_budget_km is not None:
        if not math.isfinite(float(distance_budget_km)) or distance_budget_km < 0.0:
            raise ValueError("distance_budget_km must be a finite non-negative value")
        distance_budget_km = float(distance_budget_km)
    if (start_latitude is None) != (start_longitude is None):
        raise ValueError("start_latitude and start_longitude must be supplied together")
    if tiles.empty:
        if start_latitude is None:
            raise ValueError("An explicit start is required for an empty tile table")
        empty = pd.DataFrame()
        return RouteResult(
            policy, empty, float(start_latitude), float(start_longitude),
            max_stops, distance_budget_km, int(random_seed)
        )

    required = {"tile_id", "latitude", "longitude"}
    missing = required - set(tiles.columns)
    if missing:
        raise ValueError(f"Tile table is missing: {', '.join(sorted(missing))}")
    score_column = _policy_score_column(policy)
    route_tiles = tiles.copy().sort_values("tile_id", kind="mergesort").reset_index(drop=True)
    if policy == "random":
        # The route-level seed is authoritative even if the table was scored by
        # an earlier caller with a different seed.
        rng = np.random.default_rng(int(random_seed))
        assert score_column is not None
        route_tiles[score_column] = rng.random(len(route_tiles))
    elif score_column is not None and score_column not in route_tiles.columns:
        raise ValueError(f"Tile table lacks score column {score_column!r}")

    if start_latitude is None:
        start_latitude, start_longitude = _default_start(route_tiles)
    assert start_longitude is not None
    # Validate the origin with the same bounds used for all route legs.
    haversine_km(start_latitude, start_longitude, start_latitude, start_longitude)

    stop_limit = len(route_tiles) if max_stops is None else min(int(max_stops), len(route_tiles))
    remaining = set(route_tiles.index.tolist())
    current_latitude = float(start_latitude)
    current_longitude = float(start_longitude)
    cumulative_distance = 0.0
    records: list[dict[str, Any]] = []

    while remaining and len(records) < stop_limit:
        candidates: list[tuple[int, float]] = []
        for index in remaining:
            row = route_tiles.loc[index]
            leg = haversine_km(
                current_latitude,
                current_longitude,
                float(row["latitude"]),
                float(row["longitude"]),
            )
            if distance_budget_km is None or cumulative_distance + leg <= distance_budget_km + 1e-9:
                candidates.append((index, leg))
        if not candidates:
            break

        if policy == "nearest_neighbor":
            chosen_index, leg_distance = min(
                candidates,
                key=lambda item: (item[1], str(route_tiles.at[item[0], "tile_id"])),
            )
            priority_score = float("nan")
        else:
            assert score_column is not None
            chosen_index, leg_distance = min(
                candidates,
                key=lambda item: (
                    -float(route_tiles.at[item[0], score_column]),
                    item[1],
                    str(route_tiles.at[item[0], "tile_id"]),
                ),
            )
            priority_score = float(route_tiles.at[chosen_index, score_column])

        row = route_tiles.loc[chosen_index].to_dict()
        cumulative_distance += leg_distance
        remaining_budget = (
            float(distance_budget_km - cumulative_distance)
            if distance_budget_km is not None
            else float("nan")
        )
        row.update(
            {
                "route_policy": policy,
                "stop_order": len(records) + 1,
                "priority_score": priority_score,
                "leg_distance_km": float(leg_distance),
                "cumulative_distance_km": float(cumulative_distance),
                "remaining_distance_budget_km": remaining_budget,
            }
        )
        records.append(row)
        current_latitude = float(row["latitude"])
        current_longitude = float(row["longitude"])
        remaining.remove(chosen_index)

    stops = pd.DataFrame.from_records(records)
    return RouteResult(
        policy=policy,
        stops=stops,
        start_latitude=float(start_latitude),
        start_longitude=float(start_longitude),
        max_stops=max_stops,
        distance_budget_km=distance_budget_km,
        random_seed=int(random_seed),
    )


def annotate_reference_route_ranks(
    tiles: pd.DataFrame,
    *,
    start_latitude: float | None = None,
    start_longitude: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Add unbudgeted route-order ranks for every baseline to the tile table."""

    output = tiles.copy()
    for policy in POLICIES:
        route = simulate_route(
            output,
            policy=policy,
            max_stops=len(output),
            start_latitude=start_latitude,
            start_longitude=start_longitude,
            random_seed=random_seed,
        )
        rank_map = dict(zip(route.stops["tile_id"].astype(str), route.stops["stop_order"]))
        output[f"route_rank_{policy}"] = output["tile_id"].astype(str).map(rank_map).astype(int)
    return output


def evaluate_route(route: RouteResult, tiles: pd.DataFrame) -> dict[str, Any]:
    """Return resource metrics and optional label-based discovery metrics."""

    selected_ids = set(route.stops.get("tile_id", pd.Series(dtype=str)).astype(str))
    selected = tiles[tiles["tile_id"].astype(str).isin(selected_ids)]
    total_samples = int(tiles["sample_count"].sum()) if "sample_count" in tiles else len(tiles)
    selected_samples = (
        int(selected["sample_count"].sum()) if "sample_count" in selected else len(selected)
    )
    result: dict[str, Any] = {
        "policy": route.policy,
        "stop_count": int(len(route.stops)),
        "distance_km": route.total_distance_km,
        "tile_coverage": float(len(selected) / len(tiles)) if len(tiles) else 0.0,
        "sample_coverage": float(selected_samples / total_samples) if total_samples else 0.0,
        "labels_available": bool(
            "labeled_sample_count" in tiles and int(tiles["labeled_sample_count"].sum()) > 0
        ),
        "severe_discoveries": None,
        "severe_tile_discoveries": None,
        "severe_recall_at_budget": None,
        "ndcg_at_budget": None,
        "first_severe_distance_km": None,
    }
    if not result["labels_available"]:
        return result

    total_severe = int(tiles["severe_sample_count"].sum())
    selected_severe = int(selected["severe_sample_count"].sum())
    result["severe_discoveries"] = selected_severe
    result["severe_tile_discoveries"] = int((selected["severe_sample_count"] > 0).sum())
    result["severe_recall_at_budget"] = (
        float(selected_severe / total_severe) if total_severe else 0.0
    )

    if route.stops.empty:
        result["ndcg_at_budget"] = 0.0
        return result
    relevance = route.stops["severe_sample_count"].to_numpy(dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, len(relevance) + 2, dtype=float))
    dcg = float(np.sum(relevance * discounts))
    ideal_relevance = np.sort(tiles["severe_sample_count"].to_numpy(dtype=float))[::-1][
        : len(relevance)
    ]
    ideal_dcg = float(np.sum(ideal_relevance * discounts))
    result["ndcg_at_budget"] = float(dcg / ideal_dcg) if ideal_dcg > 0.0 else 0.0
    severe_stops = route.stops[route.stops["severe_sample_count"] > 0]
    if not severe_stops.empty:
        result["first_severe_distance_km"] = float(
            severe_stops["cumulative_distance_km"].iloc[0]
        )
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def route_to_geojson(route: RouteResult) -> dict[str, Any]:
    """Serialize a route as GeoJSON using the required ``[longitude, latitude]`` order."""

    features: list[dict[str, Any]] = []
    audit_columns = (
        "route_policy",
        "stop_order",
        "tile_id",
        "sample_count",
        "disposition",
        "priority_score",
        "severity_probability",
        "uncertainty_score",
        "conflict_score",
        "escalation_score",
        "leg_distance_km",
        "cumulative_distance_km",
        "remaining_distance_budget_km",
    )
    for _, row in route.stops.iterrows():
        properties = {
            column: _json_value(row[column]) for column in audit_columns if column in row.index
        }
        features.append(
            {
                "type": "Feature",
                "id": f"stop-{int(row['stop_order'])}",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["longitude"]), float(row["latitude"])],
                },
                "properties": properties,
            }
        )

    if not route.stops.empty:
        coordinates = [[route.start_longitude, route.start_latitude]] + [
            [float(row["longitude"]), float(row["latitude"])]
            for _, row in route.stops.iterrows()
        ]
        # A one-stop route still has two positions because the declared origin is included.
        features.append(
            {
                "type": "Feature",
                "id": "route-line",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {
                    "policy": route.policy,
                    "distance_model": "haversine_geodesic_straight_line_proxy",
                    "road_network_routing": False,
                    "stop_count": len(route.stops),
                    "total_distance_km": route.total_distance_km,
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "name": "inspection_route",
        "schema_version": ROUTING_SCHEMA_VERSION,
        "properties": {
            "policy": route.policy,
            "distance_model": "haversine_geodesic_straight_line_proxy",
            "road_network_routing": False,
            "coordinate_order": "longitude,latitude",
            "start": [route.start_longitude, route.start_latitude],
            "max_stops": route.max_stops,
            "distance_budget_km": route.distance_budget_km,
            "random_seed": route.random_seed,
        },
        "features": features,
    }


def dumps_geojson(route: RouteResult, *, indent: int = 2) -> str:
    """Return deterministic UTF-8-ready GeoJSON text."""

    return json.dumps(route_to_geojson(route), ensure_ascii=False, indent=indent) + "\n"


def metrics_frame(metrics: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Create a stable baseline-comparison table."""

    columns = (
        "policy",
        "stop_count",
        "distance_km",
        "tile_coverage",
        "sample_coverage",
        "labels_available",
        "severe_discoveries",
        "severe_tile_discoveries",
        "severe_recall_at_budget",
        "ndcg_at_budget",
        "first_severe_distance_km",
    )
    return pd.DataFrame([{column: row.get(column) for column in columns} for row in metrics])
