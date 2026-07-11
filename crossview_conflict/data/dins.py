"""Auditable CAL FIRE DINS retrieval and manifest enrichment.

The join policy in this module is deliberately conservative.  Stable business
keys are tried first; ArcGIS ``OBJECTID``/``FID`` values are never accepted as
cross-service keys.  When no unique business-key match is available, the code
uses an explicit nearest-point match and records its tolerance, distance,
candidate count, runner-up distance, and ambiguity decision on every row.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


EATON_DINS_FIELDS_SERVICE_URL = (
    "https://services.arcgis.com/HdtZMT2FmI4wPzTM/arcgis/rest/services/"
    "Eaton_Palisades_DINS/FeatureServer/2"
)
EATON_DINS_PUBLIC_SERVICE_URL = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "DINS_2025_Eaton_Public_View/FeatureServer/0"
)
DEFAULT_EATON_WHERE = "INCIDENTNAME = 'Eaton'"
DEFAULT_BUSINESS_KEY_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("GLOBALID",),
    ("INCIDENTNUM", "APN", "STRUCTURETYPE"),
    ("APN", "STRUCTURETYPE"),
    ("SITEADDRESS", "STRUCTURETYPE"),
)

_OBJECT_ID_NAMES = {"objectid", "oid", "fid", "esriobjectid"}
_EARTH_RADIUS_M = 6_371_008.8

_VISUALLY_OBSERVABLE_CANDIDATES = {
    "STRUCTURETYPE",
    "STRUCTURECATEGORY",
    "ROOFCONSTRUCTION",
    "EAVES",
    "VENTSCREEN",
    "EXTERIORSIDING",
    "WINDOWPANE",
    "DECKPORCHONGRADE",
    "DECKPORCHELEVATED",
    "PATIOCOVERCARPORT",
    "FENCEATTACHEDTOSTRUCTURE",
    "PROPANETANKDISTANCE",
    "UTILITYMISCSTRUCTUREDISTANCE",
}
_INSPECTOR_ASSESSMENTS = {
    "DAMAGE",
    "WHEREFIRESTARTEDONSTRUCTURE",
    "WHATDIDFIRESTARTFROM",
    "DEFENSIVEACTIONS",
    "NUMBEROFUNITPERSTRUCTURE",
    "NOOUTBUILDINGSDAMAGED",
    "NOOUTBUILDINGSNOTDAMAGED",
}
_ADMINISTRATIVE_OR_PARCEL = {
    "STREETNUMBER",
    "STREETNAME",
    "STREETTYPE",
    "STREETSUFFIX",
    "CITY",
    "STATE",
    "ZIPCODE",
    "CALFIREUNIT",
    "COUNTY",
    "COMMUNITY",
    "BATTALION",
    "INCIDENTNAME",
    "INCIDENTNUM",
    "INCIDENTSTARTDATE",
    "HAZARDTYPE",
    "FIRENAME",
    "APN",
    "ASSESSEDIMPROVEDVALUE",
    "YEARBUILT",
    "SITEADDRESS",
    "LATITUDE",
    "LONGITUDE",
}


JsonRequester = Callable[[str, Mapping[str, Any]], dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def epoch_to_utc(value: object) -> str | None:
    """Convert an ArcGIS epoch value (milliseconds or seconds) to UTC ISO-8601."""

    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp):
        return None
    if abs(timestamp) > 100_000_000_000:
        timestamp /= 1000.0
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
    except (OverflowError, OSError, ValueError):
        return None


def service_last_edit_utc(metadata: Mapping[str, Any]) -> str | None:
    editing_info = metadata.get("editingInfo") or {}
    if not isinstance(editing_info, Mapping):
        return None
    return epoch_to_utc(editing_info.get("lastEditDate"))


def _request_json(
    url: str,
    params: Mapping[str, Any],
    *,
    timeout: float = 30.0,
    retries: int = 3,
) -> dict[str, Any]:
    query = urlencode({key: value for key, value in params.items() if value is not None})
    request = Request(
        f"{url}?{query}",
        headers={"User-Agent": "CrossViewGate-DINS-Provenance/0.1"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS endpoint by default
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError(f"ArcGIS returned non-object JSON from {url}")
            if "error" in payload:
                raise RuntimeError(f"ArcGIS error from {url}: {payload['error']}")
            return payload
        except Exception as exc:  # network/JSON errors receive bounded retries
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Could not retrieve ArcGIS JSON from {url}") from last_error


def fetch_layer_metadata(
    service_url: str,
    *,
    timeout: float = 30.0,
    requester: JsonRequester | None = None,
) -> dict[str, Any]:
    """Fetch one ArcGIS FeatureServer layer's complete public metadata."""

    service_url = service_url.rstrip("/")
    if requester is None:
        return _request_json(service_url, {"f": "pjson"}, timeout=timeout)
    payload = requester(service_url, {"f": "pjson"})
    if "error" in payload:
        raise RuntimeError(f"ArcGIS metadata error: {payload['error']}")
    return payload


def fetch_dins_layer(
    service_url: str = EATON_DINS_FIELDS_SERVICE_URL,
    *,
    where: str = DEFAULT_EATON_WHERE,
    page_size: int | None = None,
    timeout: float = 30.0,
    requester: JsonRequester | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fetch a DINS layer as ordered, validated GeoJSON pages.

    Returns ``(metadata, feature_collection, fetch_provenance)``.  The initial
    count and final feature count must agree, which prevents a partial page from
    silently becoming a seemingly complete local snapshot.
    """

    service_url = service_url.rstrip("/")

    def get(url: str, params: Mapping[str, Any]) -> dict[str, Any]:
        if requester is not None:
            payload = requester(url, params)
            if "error" in payload:
                raise RuntimeError(f"ArcGIS error from {url}: {payload['error']}")
            return payload
        return _request_json(url, params, timeout=timeout)

    retrieved_at = _utc_now()
    metadata = get(service_url, {"f": "pjson"})
    query_url = f"{service_url}/query"
    count_payload = get(
        query_url,
        {"where": where, "returnCountOnly": "true", "f": "json"},
    )
    expected_count = int(count_payload.get("count", -1))
    if expected_count < 0:
        raise ValueError("ArcGIS count response did not contain a valid count")

    max_record_count = int(metadata.get("maxRecordCount") or 2000)
    effective_page_size = min(page_size or max_record_count, max_record_count)
    if effective_page_size <= 0:
        raise ValueError("page_size must be positive")
    object_id_field = str(metadata.get("objectIdField") or "OBJECTID")

    features: list[dict[str, Any]] = []
    seen_object_ids: set[object] = set()
    offsets: list[int] = []
    while len(features) < expected_count:
        offset = len(features)
        offsets.append(offset)
        page = get(
            query_url,
            {
                "where": where,
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": 4326,
                "orderByFields": f"{object_id_field} ASC",
                "resultOffset": offset,
                "resultRecordCount": effective_page_size,
                "f": "geojson",
            },
        )
        page_features = page.get("features")
        if not isinstance(page_features, list):
            raise ValueError(f"ArcGIS page at offset {offset} has no features list")
        if not page_features:
            raise RuntimeError(
                f"ArcGIS pagination stopped at {offset}/{expected_count} records"
            )
        for feature in page_features:
            if not isinstance(feature, dict):
                raise ValueError(f"Non-object feature at offset {offset}")
            properties = feature.get("properties") or {}
            object_id = properties.get(object_id_field)
            if object_id is not None:
                if object_id in seen_object_ids:
                    raise RuntimeError(
                        f"Duplicate {object_id_field}={object_id!r} across ArcGIS pages"
                    )
                seen_object_ids.add(object_id)
            features.append(feature)
        if len(features) > expected_count:
            raise RuntimeError(
                "ArcGIS feature count changed during retrieval; rerun for a consistent snapshot"
            )

    if len(features) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} features but fetched {len(features)}"
        )

    collection = {
        "type": "FeatureCollection",
        "properties": {
            "source_service_url": service_url,
            "query_where": where,
            "retrieved_at_utc": retrieved_at,
        },
        "features": features,
    }
    provenance = {
        "source_mode": "arcgis_rest_paginated",
        "service_url": service_url,
        "query_where": where,
        "retrieved_at_utc": retrieved_at,
        "service_last_edit_utc": service_last_edit_utc(metadata),
        "service_last_edit_epoch_ms": (metadata.get("editingInfo") or {}).get(
            "lastEditDate"
        ),
        "object_id_field": object_id_field,
        "global_id_field": metadata.get("globalIdField"),
        "record_count_expected": expected_count,
        "record_count_fetched": len(features),
        "page_size": effective_page_size,
        "page_count": len(offsets),
        "result_offsets": offsets,
        "pagination_order": f"{object_id_field} ASC",
    }
    return metadata, collection, provenance


def load_geojson(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ValueError(f"Expected a GeoJSON FeatureCollection: {path}")
    if not isinstance(payload.get("features"), list):
        raise ValueError(f"GeoJSON has no features list: {path}")
    return payload


def load_service_metadata(path: str | Path) -> dict[str, Any]:
    """Load raw metadata or unwrap a previously written metadata artifact."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object metadata: {path}")
    for key in ("field_service_metadata", "metadata"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
    return payload


def infer_metadata_from_geojson(geojson: Mapping[str, Any]) -> dict[str, Any]:
    """Create minimal metadata for a fully offline local GeoJSON test fixture."""

    field_names: list[str] = []
    seen: set[str] = set()
    for feature in geojson.get("features", []):
        properties = feature.get("properties") or {}
        for name in properties:
            if name not in seen:
                seen.add(name)
                field_names.append(name)
    fields = [
        {"name": name, "alias": name, "type": "unknown", "domain": None}
        for name in field_names
    ]
    object_id_field = next(
        (name for name in field_names if _is_object_id_name(name)), None
    )
    global_id_field = next(
        (name for name in field_names if _compact_name(name) == "globalid"), None
    )
    return {
        "name": "local GeoJSON fixture",
        "fields": fields,
        "objectIdField": object_id_field,
        "globalIdField": global_id_field,
        "editingInfo": {},
    }


def features_to_frame(geojson: Mapping[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for position, feature in enumerate(geojson.get("features", [])):
        properties = dict(feature.get("properties") or {})
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        longitude: float | None = None
        latitude: float | None = None
        if geometry.get("type") == "Point" and isinstance(coordinates, list) and len(coordinates) >= 2:
            longitude = _finite_float(coordinates[0])
            latitude = _finite_float(coordinates[1])
        if longitude is None:
            for name in ("Longitude", "longitude", "lon", "x"):
                if name in properties:
                    longitude = _finite_float(properties[name])
                    if longitude is not None:
                        break
        if latitude is None:
            for name in ("Latitude", "latitude", "lat", "y"):
                if name in properties:
                    latitude = _finite_float(properties[name])
                    if latitude is not None:
                        break
        properties["_dins_geometry_longitude"] = longitude
        properties["_dins_geometry_latitude"] = latitude
        properties["_dins_feature_id"] = feature.get("id", position)
        records.append(properties)
    return pd.DataFrame.from_records(records)


def classify_field_semantics(field_name: str) -> str:
    name = str(field_name).upper()
    if _is_object_id_name(name) or _compact_name(name) == "globalid":
        return "source_identifier"
    if name in _VISUALLY_OBSERVABLE_CANDIDATES:
        return "visually_observable_candidate"
    if name in _INSPECTOR_ASSESSMENTS:
        return "inspector_assessment_or_observation"
    if name in _ADMINISTRATIVE_OR_PARCEL:
        return "administrative_or_parcel"
    return "unclassified_inspector_recorded"


def extract_field_domains(
    metadata: Mapping[str, Any], *, service_url: str | None = None
) -> dict[str, Any]:
    """Preserve published field and subtype domains without inferring new ones."""

    layer_types = metadata.get("types") or []
    fields: list[dict[str, Any]] = []
    for field in metadata.get("fields") or []:
        name = str(field.get("name"))
        subtype_domains: dict[str, Any] = {}
        for layer_type in layer_types:
            domains = layer_type.get("domains") or {}
            if name in domains:
                subtype_domains[str(layer_type.get("id"))] = domains[name]
        fields.append(
            {
                "name": name,
                "alias": field.get("alias"),
                "type": field.get("type"),
                "length": field.get("length"),
                "nullable": field.get("nullable"),
                "editable": field.get("editable"),
                "domain": field.get("domain"),
                "subtype_domains": subtype_domains,
                "semantic_group": classify_field_semantics(name),
            }
        )
    return {
        "service_url": service_url,
        "service_last_edit_utc": service_last_edit_utc(metadata),
        "type_id_field": metadata.get("typeIdField"),
        "published_layer_types": [
            {"id": layer_type.get("id"), "name": layer_type.get("name")}
            for layer_type in layer_types
        ],
        "fields_with_direct_domain": sum(field["domain"] is not None for field in fields),
        "fields_with_subtype_domain": sum(bool(field["subtype_domains"]) for field in fields),
        "fields": fields,
        "policy": "Published domains are preserved verbatim; observed values are not promoted to domains.",
    }


def compute_field_statistics(
    source: pd.DataFrame,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute explicit blank/null and exact ``Unknown`` statistics per field."""

    metadata_fields = {
        str(field.get("name")): field for field in (metadata or {}).get("fields", [])
    }
    columns = [column for column in source.columns if not str(column).startswith("_dins_")]
    stats: list[dict[str, Any]] = []
    row_count = len(source)
    for column in columns:
        series = source[column]
        blank = series.isna() | series.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        unknown = series.map(
            lambda value: isinstance(value, str)
            and value.strip().casefold() == "unknown"
        )
        null_count = int(blank.sum())
        unknown_count = int(unknown.sum())
        non_null_count = row_count - null_count
        field = metadata_fields.get(str(column), {})
        stats.append(
            {
                "name": str(column),
                "alias": field.get("alias"),
                "type": field.get("type"),
                "semantic_group": classify_field_semantics(str(column)),
                "row_count": row_count,
                "null_or_blank_count": null_count,
                "null_or_blank_rate": null_count / row_count if row_count else None,
                "unknown_count": unknown_count,
                "unknown_rate_all_rows": unknown_count / row_count if row_count else None,
                "unknown_rate_non_null": (
                    unknown_count / non_null_count if non_null_count else None
                ),
                "unique_non_null_count": int(series[~blank].nunique(dropna=True)),
                "published_domain_available": field.get("domain") is not None,
            }
        )
    return {
        "generated_at_utc": _utc_now(),
        "record_count": row_count,
        "null_definition": "JSON null, NaN, empty string, or whitespace-only string",
        "unknown_definition": "case-insensitive exact string 'Unknown' after trimming",
        "fields": stats,
    }


def haversine_m(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    lat_a = math.radians(latitude_a)
    lat_b = math.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(value)))


class _SpatialIndex:
    def __init__(
        self,
        source: pd.DataFrame,
        latitude_column: str,
        longitude_column: str,
        tolerance_m: float,
    ) -> None:
        if tolerance_m <= 0:
            raise ValueError("tolerance_m must be positive")
        self.source = source
        self.latitude_column = latitude_column
        self.longitude_column = longitude_column
        self.tolerance_m = float(tolerance_m)
        valid_latitudes = [
            value
            for value in (_finite_float(v) for v in source[latitude_column])
            if value is not None
        ]
        self.reference_latitude = (
            sum(valid_latitudes) / len(valid_latitudes) if valid_latitudes else 0.0
        )
        self.cos_reference = max(1e-8, math.cos(math.radians(self.reference_latitude)))
        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for position, row in source.iterrows():
            latitude = _finite_float(row[latitude_column])
            longitude = _finite_float(row[longitude_column])
            if latitude is None or longitude is None:
                continue
            self.buckets[self._cell(latitude, longitude)].append(int(position))

    def _xy(self, latitude: float, longitude: float) -> tuple[float, float]:
        x = _EARTH_RADIUS_M * math.radians(longitude) * self.cos_reference
        y = _EARTH_RADIUS_M * math.radians(latitude)
        return x, y

    def _cell(self, latitude: float, longitude: float) -> tuple[int, int]:
        x, y = self._xy(latitude, longitude)
        return math.floor(x / self.tolerance_m), math.floor(y / self.tolerance_m)

    def candidates(self, latitude: float, longitude: float) -> list[tuple[float, int]]:
        cell_x, cell_y = self._cell(latitude, longitude)
        candidates: list[tuple[float, int]] = []
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                for position in self.buckets.get((cell_x + delta_x, cell_y + delta_y), []):
                    source_row = self.source.iloc[position]
                    source_lat = float(source_row[self.latitude_column])
                    source_lon = float(source_row[self.longitude_column])
                    distance = haversine_m(latitude, longitude, source_lat, source_lon)
                    if distance <= self.tolerance_m:
                        candidates.append((distance, position))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates


def join_dins_fields(
    manifest: pd.DataFrame,
    source: pd.DataFrame,
    *,
    business_key_candidates: Sequence[Sequence[str]] = DEFAULT_BUSINESS_KEY_CANDIDATES,
    manifest_latitude_column: str = "latitude",
    manifest_longitude_column: str = "longitude",
    source_latitude_column: str = "_dins_geometry_latitude",
    source_longitude_column: str = "_dins_geometry_longitude",
    tolerance_m: float = 10.0,
    ambiguity_margin_m: float = 1.0,
    service_url: str | None = None,
    service_last_edit: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join DINS fields and return an enriched copy plus an audit summary.

    Spatial ambiguity means that the runner-up is within
    ``ambiguity_margin_m`` of the nearest candidate.  Ambiguous rows are not
    enriched; their audit columns retain both distances and candidate counts.
    """

    if ambiguity_margin_m < 0:
        raise ValueError("ambiguity_margin_m must be non-negative")
    source = source.reset_index(drop=True).copy()
    enriched = manifest.reset_index(drop=True).copy()

    manifest_latitude_column = _resolve_column(enriched, manifest_latitude_column)
    manifest_longitude_column = _resolve_column(enriched, manifest_longitude_column)
    source_latitude_column = _resolve_column(source, source_latitude_column)
    source_longitude_column = _resolve_column(source, source_longitude_column)

    normalized_candidates: list[tuple[str, ...]] = []
    for candidate in business_key_candidates:
        candidate = tuple(str(field) for field in candidate)
        if not candidate:
            continue
        if any(_is_object_id_name(field) for field in candidate):
            raise ValueError(
                "ArcGIS OBJECTID/OID/FID cannot be used as a cross-service business key"
            )
        normalized_candidates.append(candidate)

    key_indices: list[
        tuple[tuple[str, ...], tuple[str, ...], dict[tuple[str, ...], list[int]]]
    ] = []
    for candidate in normalized_candidates:
        try:
            manifest_columns = tuple(_resolve_column(enriched, field) for field in candidate)
            source_columns = tuple(_resolve_column(source, field) for field in candidate)
        except KeyError:
            continue
        index: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for position, source_row in source.iterrows():
            key = _row_key(source_row, source_columns)
            if key is not None:
                index[key].append(int(position))
        key_indices.append((manifest_columns, source_columns, dict(index)))

    spatial_index = _SpatialIndex(
        source,
        source_latitude_column,
        source_longitude_column,
        tolerance_m,
    )

    source_property_columns = [
        str(column) for column in source.columns if not str(column).startswith("_dins_")
    ]
    source_output_columns = _source_output_column_map(source_property_columns)
    audit_columns = [
        "dins_join_status",
        "dins_join_method",
        "dins_join_key",
        "dins_join_distance_m",
        "dins_join_second_distance_m",
        "dins_join_tolerance_m",
        "dins_join_ambiguity_margin_m",
        "dins_join_candidate_count",
        "dins_join_ambiguous",
        "dins_join_business_key_ambiguous",
        "dins_join_business_key_candidate_count",
        "dins_join_distance_warning",
        "dins_source_service_url",
        "dins_source_last_edit_utc",
    ]
    collisions = (set(source_output_columns.values()) | set(audit_columns)) & set(
        enriched.columns
    )
    if collisions:
        raise ValueError(
            "Refusing to overwrite existing manifest columns: "
            + ", ".join(sorted(collisions))
        )
    for output_column in source_output_columns.values():
        enriched[output_column] = pd.NA

    audit_records: list[dict[str, Any]] = []
    selected_positions: list[int | None] = []
    for _, manifest_row in enriched.iterrows():
        selected: int | None = None
        status = "unmatched"
        method = "none"
        key_label: str | None = None
        distance: float | None = None
        second_distance: float | None = None
        candidate_count = 0
        ambiguous = False
        business_key_ambiguous = False
        business_key_candidate_count = 0

        for manifest_columns, source_columns, index in key_indices:
            key = _row_key(manifest_row, manifest_columns)
            if key is None:
                continue
            matches = index.get(key, [])
            if len(matches) == 1:
                selected = matches[0]
                status = "matched_business_key"
                method = "business_key"
                key_label = "+".join(source_columns)
                candidate_count = 1
                break
            if len(matches) > 1:
                business_key_ambiguous = True
                business_key_candidate_count = max(
                    business_key_candidate_count, len(matches)
                )

        manifest_latitude = _finite_float(manifest_row[manifest_latitude_column])
        manifest_longitude = _finite_float(manifest_row[manifest_longitude_column])
        if selected is not None:
            source_row = source.iloc[selected]
            source_latitude = _finite_float(source_row[source_latitude_column])
            source_longitude = _finite_float(source_row[source_longitude_column])
            if (
                manifest_latitude is not None
                and manifest_longitude is not None
                and source_latitude is not None
                and source_longitude is not None
            ):
                distance = haversine_m(
                    manifest_latitude,
                    manifest_longitude,
                    source_latitude,
                    source_longitude,
                )
        elif manifest_latitude is None or manifest_longitude is None:
            status = (
                "business_key_ambiguous"
                if business_key_ambiguous
                else "missing_coordinates"
            )
            method = "business_key" if business_key_ambiguous else "none"
            ambiguous = business_key_ambiguous
            candidate_count = business_key_candidate_count
        else:
            spatial_candidates = spatial_index.candidates(
                manifest_latitude, manifest_longitude
            )
            candidate_count = len(spatial_candidates)
            if spatial_candidates:
                distance, nearest_position = spatial_candidates[0]
                if len(spatial_candidates) > 1:
                    second_distance = spatial_candidates[1][0]
                    ambiguous = second_distance - distance <= ambiguity_margin_m
                if ambiguous:
                    status = "spatial_ambiguous"
                    method = "spatial_nearest"
                else:
                    selected = nearest_position
                    status = "matched_spatial"
                    method = "spatial_nearest"
            elif business_key_ambiguous:
                status = "business_key_ambiguous"
                method = "business_key_then_spatial"
                ambiguous = True

        selected_positions.append(selected)
        audit_records.append(
            {
                "dins_join_status": status,
                "dins_join_method": method,
                "dins_join_key": key_label,
                "dins_join_distance_m": distance,
                "dins_join_second_distance_m": second_distance,
                "dins_join_tolerance_m": float(tolerance_m),
                "dins_join_ambiguity_margin_m": float(ambiguity_margin_m),
                "dins_join_candidate_count": candidate_count,
                "dins_join_ambiguous": bool(ambiguous),
                "dins_join_business_key_ambiguous": bool(business_key_ambiguous),
                "dins_join_business_key_candidate_count": business_key_candidate_count,
                "dins_join_distance_warning": bool(
                    selected is not None
                    and method == "business_key"
                    and distance is not None
                    and distance > tolerance_m
                ),
                "dins_source_service_url": service_url,
                "dins_source_last_edit_utc": service_last_edit,
            }
        )

    for manifest_position, source_position in enumerate(selected_positions):
        if source_position is None:
            continue
        source_row = source.iloc[source_position]
        for source_column, output_column in source_output_columns.items():
            enriched.at[manifest_position, output_column] = source_row[source_column]
    audit_frame = pd.DataFrame.from_records(audit_records)
    for column in audit_columns:
        enriched[column] = audit_frame[column]

    status_counts = enriched["dins_join_status"].value_counts(dropna=False).to_dict()
    matched_mask = enriched["dins_join_status"].isin(
        ["matched_business_key", "matched_spatial"]
    )
    source_use_counts = Counter(
        position for position in selected_positions if position is not None
    )
    spatial_distances = pd.to_numeric(
        enriched.loc[
            enriched["dins_join_status"] == "matched_spatial",
            "dins_join_distance_m",
        ],
        errors="coerce",
    ).dropna()
    spatial_distance_summary = {
        "count": int(len(spatial_distances)),
        "p50_m": float(spatial_distances.quantile(0.50)) if len(spatial_distances) else None,
        "p95_m": float(spatial_distances.quantile(0.95)) if len(spatial_distances) else None,
        "p99_m": float(spatial_distances.quantile(0.99)) if len(spatial_distances) else None,
        "max_m": float(spatial_distances.max()) if len(spatial_distances) else None,
    }
    summary = {
        "manifest_rows": int(len(enriched)),
        "source_rows": int(len(source)),
        "matched_rows": int(matched_mask.sum()),
        "matched_rate": float(matched_mask.mean()) if len(enriched) else None,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "matched_unique_source_rows": len(source_use_counts),
        "unmatched_source_rows": int(len(source) - len(source_use_counts)),
        "source_rows_used_by_multiple_manifest_rows": sum(
            count > 1 for count in source_use_counts.values()
        ),
        "business_key_priority": [list(candidate) for candidate in normalized_candidates],
        "available_business_key_candidates": [
            list(source_columns) for _, source_columns, _ in key_indices
        ],
        "objectid_policy": (
            "OBJECTID/OID/FID are retained only as source provenance and are never "
            "cross-service join keys."
        ),
        "spatial_policy": {
            "distance": "haversine metres",
            "tolerance_m": float(tolerance_m),
            "ambiguity_margin_m": float(ambiguity_margin_m),
            "ambiguous_rows_enriched": False,
            "matched_distance_summary": spatial_distance_summary,
        },
        "source_field_output_columns": source_output_columns,
        "semantic_policy": {
            "all_fields_origin": "CAL FIRE DINS inspector/service record",
            "visually_observable_candidate": (
                "A field may be visually testable only when the relevant structure part "
                "is visible; this tag is not image-level visibility ground truth."
            ),
            "unsupported_ground_truth": ["recommended_action", "rationale"],
        },
        "source_service_url": service_url,
        "source_last_edit_utc": service_last_edit,
    }
    return enriched, summary


def build_join_provenance(
    *,
    fetch_provenance: Mapping[str, Any],
    join_summary: Mapping[str, Any],
    manifest_path: str | Path,
    source_geojson: Mapping[str, Any] | None = None,
    source_geojson_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    payload: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256_file(manifest_path),
        },
        "fetch": dict(fetch_provenance),
        "join": dict(join_summary),
        "guardrails": {
            "cross_service_objectid_equality_assumed": False,
            "ambiguous_spatial_matches_enriched": False,
            "recommended_action_ground_truth_created": False,
            "rationale_ground_truth_created": False,
        },
    }
    if source_geojson is not None:
        payload["source_snapshot"] = {
            "feature_count": len(source_geojson.get("features", [])),
            "sha256_canonical_json": _sha256_json(source_geojson),
        }
    if source_geojson_path is not None:
        source_path = Path(source_geojson_path)
        payload["local_source_geojson"] = {
            "path": str(source_path.resolve()),
            "sha256": _sha256_file(source_path),
        }
    return payload


def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "features": root / "dins_fields.geojson",
        "metadata": root / "service_metadata.json",
        "domains": root / "field_domains.json",
        "statistics": root / "field_statistics.json",
        "joined_manifest": root / "eaton_manifest_with_dins.csv",
        "provenance": root / "join_provenance.json",
    }


def write_dins_artifacts(
    output_dir: str | Path,
    *,
    geojson: Mapping[str, Any],
    service_metadata_artifact: Mapping[str, Any],
    domains: Mapping[str, Any],
    statistics: Mapping[str, Any],
    joined_manifest: pd.DataFrame,
    provenance: Mapping[str, Any],
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write the snapshot only after an all-path non-overwrite preflight."""

    paths = artifact_paths(output_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing DINS artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _write_json(paths["features"], geojson)
    _write_json(paths["metadata"], service_metadata_artifact)
    _write_json(paths["domains"], domains)
    _write_json(paths["statistics"], statistics)
    joined_manifest.to_csv(paths["joined_manifest"], index=False)
    _write_json(paths["provenance"], provenance)
    return paths


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _compact_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _is_object_id_name(value: object) -> bool:
    return _compact_name(value) in _OBJECT_ID_NAMES


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _resolve_column(frame: pd.DataFrame, requested: str) -> str:
    if requested in frame.columns:
        return str(requested)
    matches = [str(column) for column in frame.columns if str(column).casefold() == requested.casefold()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Ambiguous case-insensitive column {requested!r}: {matches}")
    raise KeyError(f"Column {requested!r} not found")


def _normalized_key_value(value: object) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        normalized = " ".join(value.strip().casefold().split())
        if not normalized:
            return None
        return normalized.strip("{}")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip().casefold() or None


def _row_key(row: pd.Series, columns: Sequence[str]) -> tuple[str, ...] | None:
    values = tuple(_normalized_key_value(row[column]) for column in columns)
    if any(value is None for value in values):
        return None
    return tuple(value for value in values if value is not None)


def _snake_case(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return value or "field"


def _source_output_column_map(columns: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for column in columns:
        compact = _compact_name(column)
        if _is_object_id_name(column):
            base = "dins_source_objectid"
        elif compact == "globalid":
            base = "dins_source_globalid"
        else:
            base = f"dins_{_snake_case(column)}"
        output = base
        suffix = 2
        while output in used:
            output = f"{base}_{suffix}"
            suffix += 1
        used.add(output)
        result[column] = output
    return result
