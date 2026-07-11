from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


CVIAN_DOI = "https://doi.org/10.14459/2024mp1749324"
CVIAN_POSITION_RELATIVE_PATH = Path("02_Position") / "CVIAN_position.geojson"
CVIAN_CHECKSUM_RELATIVE_PATH = Path("checksums.sha512")
CVIAN_POSITION_SHA512 = (
    "579e4e9ee2d130a1922fd7e91600292737f694cf06473d66b257beb5fd0a60c"
    "4ec1e9a5d90a49f4f9f5f7e9caf46ff8977d37b0aa2b38f2d471cdced1625960a"
)

_OFFICIAL_IMAGE_RE = re.compile(
    r"(?:^|/)CVIAN/(?P<view>00_SVI|01_Satellite)/"
    r"(?P<severity>[012]_[^/]+)/(?P<mapillary_id>\d+)\.png$"
)
_LOCAL_IMAGE_RE = re.compile(r"(?P<sample_id>\d+)_(?P<view>sat|svi)\.png$")


def sha512_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha512()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha512_manifest(path: str | Path) -> dict[str, list[str]]:
    """Return checksum -> official relative paths without assuming hash uniqueness."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{128}", parts[0]):
            raise ValueError(f"Malformed SHA-512 line {line_number} in {path}")
        relative_path = parts[1].strip().lstrip("*").replace("\\", "/")
        by_hash[parts[0].lower()].append(relative_path)
    if not by_hash:
        raise ValueError(f"No SHA-512 records found in {path}")
    return dict(by_hash)


def load_cvi_an_positions(path: str | Path) -> pd.DataFrame:
    """Load and validate the official CVIAN Point feature collection."""
    position_path = Path(path)
    document = json.loads(position_path.read_text(encoding="utf-8"))
    if document.get("type") != "FeatureCollection":
        raise ValueError(f"Expected a GeoJSON FeatureCollection: {position_path}")

    rows: list[dict[str, object]] = []
    for index, feature in enumerate(document.get("features", [])):
        properties = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Point":
            raise ValueError(f"Feature {index} is not a Point")
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 2:
            raise ValueError(f"Feature {index} has invalid coordinates")

        mapillary_id = str(properties.get("id", "")).strip()
        if not mapillary_id:
            raise ValueError(f"Feature {index} is missing properties.id")
        longitude = float(properties.get("lon", coordinates[0]))
        latitude = float(properties.get("lat", coordinates[1]))
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise ValueError(f"Feature {mapillary_id} has invalid lon/lat")
        # The property values retain more precision than the rounded geometry.
        if abs(float(coordinates[0]) - longitude) > 1e-4 or abs(float(coordinates[1]) - latitude) > 1e-4:
            raise ValueError(f"Feature {mapillary_id} geometry disagrees with properties.lon/lat")

        captured_value = properties.get("captured_a")
        captured_at_ms = int(float(captured_value)) if captured_value is not None else None
        captured_at_utc = (
            datetime.fromtimestamp(captured_at_ms / 1000.0, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
            if captured_at_ms is not None
            else None
        )
        creator_value = properties.get("creator_id")
        creator_id = str(int(float(creator_value))) if creator_value is not None else ""
        rows.append(
            {
                "mapillary_id": mapillary_id,
                "latitude": latitude,
                "longitude": longitude,
                "sequence_id": str(properties.get("sequence_i", "")).strip(),
                "captured_at_ms": captured_at_ms,
                "captured_at_utc": captured_at_utc,
                "compass_angle_deg": float(properties["compass_an"])
                if properties.get("compass_an") is not None
                else np.nan,
                "creator_id": creator_id,
                "is_pano": int(properties.get("is_pano", 0)),
            }
        )

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError(f"No position features found in {position_path}")
    if frame["mapillary_id"].duplicated().any():
        duplicates = frame.loc[frame["mapillary_id"].duplicated(), "mapillary_id"].head().tolist()
        raise ValueError(f"Duplicate Mapillary ids in GeoJSON: {duplicates}")
    if (frame["sequence_id"] == "").any():
        raise ValueError("CVIAN position features contain empty sequence ids")
    return frame


def _sample_id_from_pair(row: pd.Series | dict[str, object]) -> str:
    sat_name = Path(str(row["sat_path"])).name
    svi_name = Path(str(row["svi_path"])).name
    sat_match = _LOCAL_IMAGE_RE.fullmatch(sat_name)
    svi_match = _LOCAL_IMAGE_RE.fullmatch(svi_name)
    if not sat_match or sat_match.group("view") != "sat":
        raise ValueError(f"Unexpected local satellite filename: {sat_name}")
    if not svi_match or svi_match.group("view") != "svi":
        raise ValueError(f"Unexpected local street-view filename: {svi_name}")
    if sat_match.group("sample_id") != svi_match.group("sample_id"):
        raise ValueError(f"Pair ids disagree: {sat_name} vs {svi_name}")
    return sat_match.group("sample_id")


def _match_official_path(
    candidates: Iterable[str], *, view: str, severity: str, local_path: Path
) -> tuple[str, str]:
    official_view = "01_Satellite" if view == "sat" else "00_SVI"
    matches: list[tuple[str, str]] = []
    for candidate in candidates:
        parsed = _OFFICIAL_IMAGE_RE.search(candidate.replace("\\", "/"))
        if not parsed:
            continue
        if parsed.group("view") == official_view and parsed.group("severity") == severity:
            matches.append((parsed.group("mapillary_id"), candidate))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one official checksum match for {local_path} ({severity}, {view}); "
            f"found {len(matches)}"
        )
    return matches[0]


def build_cvi_an_image_id_map(
    dataset_root: str | Path,
    pairs: pd.DataFrame,
    checksums_path: str | Path,
) -> pd.DataFrame:
    """Recover original Mapillary ids for renamed local pairs via exact SHA-512."""
    root = Path(dataset_root)
    image_root = root / "images"
    checksum_paths = parse_sha512_manifest(checksums_path)
    rows: list[dict[str, object]] = []
    seen_samples: set[str] = set()

    for pair in pairs.to_dict(orient="records"):
        sample_id = _sample_id_from_pair(pair)
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate sample id in pairs CSV: {sample_id}")
        seen_samples.add(sample_id)
        severity = str(pair["severity"])
        view_matches: dict[str, tuple[str, str, str]] = {}
        for view, column in (("sat", "sat_path"), ("svi", "svi_path")):
            local_path = image_root / Path(str(pair[column])).name
            if not local_path.is_file():
                raise FileNotFoundError(f"Missing local image: {local_path}")
            checksum = sha512_file(local_path)
            candidates = checksum_paths.get(checksum, [])
            if not candidates:
                raise ValueError(f"No official checksum match for {local_path}")
            mapillary_id, official_path = _match_official_path(
                candidates, view=view, severity=severity, local_path=local_path
            )
            view_matches[view] = (mapillary_id, official_path, checksum)

        if view_matches["sat"][0] != view_matches["svi"][0]:
            raise ValueError(
                f"Satellite/SVI source ids disagree for {sample_id}: "
                f"{view_matches['sat'][0]} vs {view_matches['svi'][0]}"
            )
        rows.append(
            {
                "sample_id": sample_id,
                "mapillary_id": view_matches["svi"][0],
                "severity": severity,
                "sat_sha512": view_matches["sat"][2],
                "svi_sha512": view_matches["svi"][2],
                "official_sat_path": view_matches["sat"][1],
                "official_svi_path": view_matches["svi"][1],
            }
        )

    frame = pd.DataFrame.from_records(rows)
    if frame["mapillary_id"].duplicated().any():
        raise ValueError("Multiple local samples map to the same official Mapillary id")
    return frame.sort_values("sample_id").reset_index(drop=True)


def add_spatial_block_ids(frame: pd.DataFrame, grid_degrees: float = 0.005) -> pd.DataFrame:
    if not math.isfinite(grid_degrees) or grid_degrees <= 0:
        raise ValueError("grid_degrees must be finite and positive")
    if frame.empty:
        raise ValueError("Cannot assign spatial blocks to an empty frame")
    missing_columns = [column for column in ("latitude", "longitude") if column not in frame]
    if missing_columns:
        raise KeyError(f"Missing coordinate columns: {missing_columns}")
    if "sample_id" in frame:
        if frame["sample_id"].isna().any() or frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("sample_id must be non-null and unique before spatial blocking")
    result = frame.copy()
    latitude = pd.to_numeric(result["latitude"], errors="raise").to_numpy(dtype=float)
    longitude = pd.to_numeric(result["longitude"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("Spatial coordinates must all be finite")
    if ((latitude < -90.0) | (latitude > 90.0)).any():
        raise ValueError("latitude must lie in [-90, 90]")
    if ((longitude < -180.0) | (longitude > 180.0)).any():
        raise ValueError("longitude must lie in [-180, 180]")
    lat_index = np.floor((latitude + 90.0) / grid_degrees).astype(np.int64)
    lon_index = np.floor((longitude + 180.0) / grid_degrees).astype(np.int64)
    prefix = f"grid_{grid_degrees:g}"
    result["spatial_block_id"] = [
        f"{prefix}_{lat_value}_{lon_value}"
        for lat_value, lon_value in zip(lat_index, lon_index)
    ]
    result["tile_id"] = result["spatial_block_id"]
    return result


def georeference_cvi_an_pairs(
    pairs: pd.DataFrame,
    image_id_map: pd.DataFrame,
    positions: pd.DataFrame,
    grid_degrees: float = 0.005,
) -> pd.DataFrame:
    """Join local pair ids to official positions and fail on any incomplete join."""
    frame = pairs.copy()
    frame["sample_id"] = [_sample_id_from_pair(row) for row in frame.to_dict(orient="records")]
    frame = frame.merge(
        image_id_map,
        on=["sample_id", "severity"],
        how="left",
        validate="one_to_one",
    )
    frame = frame.merge(positions, on="mapillary_id", how="left", validate="one_to_one")
    required = ["mapillary_id", "latitude", "longitude", "sequence_id"]
    missing = frame[required].isna().any(axis=1)
    if missing.any():
        sample_ids = frame.loc[missing, "sample_id"].head().tolist()
        raise ValueError(f"Incomplete CVIAN georeference join for samples: {sample_ids}")
    if len(frame) != len(pairs):
        raise ValueError("CVIAN georeference join changed the pair count")
    return add_spatial_block_ids(frame, grid_degrees=grid_degrees)


def local_position_geojson(frame: pd.DataFrame) -> dict[str, object]:
    features: list[dict[str, object]] = []
    property_columns = [
        "sample_id",
        "mapillary_id",
        "severity",
        "sequence_id",
        "spatial_block_id",
        "tile_id",
        "captured_at_ms",
        "captured_at_utc",
        "compass_angle_deg",
        "creator_id",
        "is_pano",
        "sat_sha512",
        "svi_sha512",
    ]
    for row in frame.sort_values("sample_id").to_dict(orient="records"):
        properties = {
            column: _json_scalar(row.get(column))
            for column in property_columns
            if column in row
        }
        properties["latitude"] = float(row["latitude"])
        properties["longitude"] = float(row["longitude"])
        properties["position_source"] = CVIAN_DOI
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["longitude"]), float(row["latitude"])],
                },
                "properties": properties,
            }
        )
    return {
        "type": "FeatureCollection",
        "name": "CVIAN_position_local_ids",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }


def _json_scalar(value: object) -> object:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def split_grouped_frame(
    frame: pd.DataFrame,
    *,
    group_col: str,
    label_col: str,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Greedily stratify indivisible groups into deterministic train/val/test sets."""
    test_fraction = 1.0 - train_fraction - val_fraction
    fractions = np.array([train_fraction, val_fraction, test_fraction], dtype=float)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train_fraction and val_fraction must leave a positive test fraction")
    if group_col not in frame or label_col not in frame:
        raise KeyError(f"Missing split column: {group_col if group_col not in frame else label_col}")
    if frame[group_col].isna().any() or frame[label_col].isna().any():
        raise ValueError("Group and label columns cannot contain null values")

    split_names = ["train", "val", "test"]
    labels = sorted(frame[label_col].astype(str).unique().tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    group_rows: list[tuple[str, int, np.ndarray, float]] = []
    rng = np.random.default_rng(seed)
    for group_name, group in frame.groupby(group_col, sort=True):
        counts = np.zeros(len(labels), dtype=float)
        for label, count in group[label_col].astype(str).value_counts().items():
            counts[label_to_index[label]] = float(count)
        group_rows.append((str(group_name), len(group), counts, float(rng.random())))
    if len(group_rows) < 3:
        raise ValueError(f"Need at least three {group_col} groups to create three splits")

    # Large and label-concentrated groups are placed first; jitter only breaks ties.
    group_rows.sort(key=lambda item: (-item[1], -float(item[2].max()), item[3], item[0]))
    target_total = fractions * len(frame)
    global_labels = np.array(
        [(frame[label_col].astype(str) == label).sum() for label in labels], dtype=float
    )
    target_labels = fractions[:, None] * global_labels[None, :]
    assigned_total = np.zeros(3, dtype=float)
    assigned_labels = np.zeros((3, len(labels)), dtype=float)
    assignments: dict[str, str] = {}

    for position, (group_name, group_size, group_labels, _) in enumerate(group_rows):
        candidate_indices = list(range(3))
        # Seed every split with a group, preventing an empty low-fraction split.
        if position < 3:
            candidate_indices = [position]
        best: tuple[float, int] | None = None
        for split_index in candidate_indices:
            totals = assigned_total.copy()
            label_counts = assigned_labels.copy()
            totals[split_index] += group_size
            label_counts[split_index] += group_labels
            total_error = np.square((totals - target_total) / np.maximum(target_total, 1.0)).mean()
            label_error = np.square(
                (label_counts - target_labels) / np.maximum(target_labels, 1.0)
            ).mean()
            overflow = np.square(
                np.maximum(totals - target_total, 0.0) / np.maximum(target_total, 1.0)
            ).mean()
            score = float(total_error + label_error + 2.0 * overflow)
            candidate = (score, split_index)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        chosen = best[1]
        assignments[group_name] = split_names[chosen]
        assigned_total[chosen] += group_size
        assigned_labels[chosen] += group_labels

    split_series = frame[group_col].astype(str).map(assignments)
    if split_series.isna().any():
        raise RuntimeError("Internal error: one or more groups were not assigned")
    result = {
        name: frame.loc[split_series == name].copy().reset_index(drop=True)
        for name in split_names
    }
    assert_group_isolation(result, group_col)
    return result


def assert_group_isolation(splits: dict[str, pd.DataFrame], group_col: str) -> None:
    names = list(splits)
    for left_index, left_name in enumerate(names):
        left = set(splits[left_name][group_col].dropna().astype(str))
        for right_name in names[left_index + 1 :]:
            right = set(splits[right_name][group_col].dropna().astype(str))
            overlap = left & right
            if overlap:
                raise ValueError(
                    f"{group_col} leakage between {left_name} and {right_name}: "
                    f"{sorted(overlap)[:5]}"
                )


def _minimum_haversine_distances(
    source: np.ndarray,
    target: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    """Minimum source-to-target great-circle distances in metres."""
    if len(source) == 0:
        return np.empty(0, dtype=float)
    if len(target) == 0:
        return np.full(len(source), np.inf, dtype=float)
    target_radians = np.radians(target)
    target_lat = target_radians[:, 0][None, :]
    target_lon = target_radians[:, 1][None, :]
    output = np.full(len(source), np.inf, dtype=float)
    earth_radius_m = 6_371_008.8
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        source_radians = np.radians(source[start:stop])
        source_lat = source_radians[:, 0][:, None]
        source_lon = source_radians[:, 1][:, None]
        delta_lat = target_lat - source_lat
        delta_lon = target_lon - source_lon
        a = (
            np.sin(delta_lat / 2.0) ** 2
            + np.cos(source_lat) * np.cos(target_lat) * np.sin(delta_lon / 2.0) ** 2
        )
        distances = 2.0 * earth_radius_m * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        output[start:stop] = distances.min(axis=1)
    return output


def purge_spatial_buffer(
    splits: dict[str, pd.DataFrame],
    threshold_m: float,
    *,
    priority: tuple[str, ...] = ("test", "val", "train"),
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Drop lower-priority boundary samples that are too close to a held-out split.

    The default preserves test first, then validation, and finally training.
    Every retained cross-split point pair is strictly farther than
    ``threshold_m``. Exclusions are returned as an auditable sidecar.
    """
    if threshold_m < 0:
        raise ValueError("threshold_m cannot be negative")
    if set(priority) != set(splits) or len(priority) != len(splits):
        raise ValueError("priority must name every split exactly once")
    copied = {name: frame.copy().reset_index(drop=True) for name, frame in splits.items()}
    if threshold_m == 0:
        return copied, pd.DataFrame(
            columns=["sample_id", "excluded_from", "near_split", "minimum_distance_m"]
        )

    kept: dict[str, pd.DataFrame] = {}
    exclusions: list[pd.DataFrame] = []
    for split_name in priority:
        current = copied[split_name]
        if not kept:
            kept[split_name] = current
            continue
        coordinates = current[["latitude", "longitude"]].to_numpy(dtype=float)
        nearest_distance = np.full(len(current), np.inf, dtype=float)
        nearest_split = np.full(len(current), "", dtype=object)
        for protected_name, protected in kept.items():
            protected_coordinates = protected[["latitude", "longitude"]].to_numpy(dtype=float)
            distances = _minimum_haversine_distances(coordinates, protected_coordinates)
            closer = distances < nearest_distance
            nearest_distance[closer] = distances[closer]
            nearest_split[closer] = protected_name
        exclude_mask = nearest_distance <= threshold_m
        if exclude_mask.any():
            excluded = current.loc[exclude_mask, ["sample_id"]].copy()
            excluded["excluded_from"] = split_name
            excluded["near_split"] = nearest_split[exclude_mask]
            excluded["minimum_distance_m"] = nearest_distance[exclude_mask]
            exclusions.append(excluded)
        kept[split_name] = current.loc[~exclude_mask].reset_index(drop=True)

    ordered = {name: kept[name] for name in splits}
    exclusions_frame = (
        pd.concat(exclusions, ignore_index=True)
        if exclusions
        else pd.DataFrame(
            columns=["sample_id", "excluded_from", "near_split", "minimum_distance_m"]
        )
    )
    return ordered, exclusions_frame
