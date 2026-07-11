#!/usr/bin/env python
"""Build the frozen spatial protocol for the focused Eaton component study."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.data.cvian import add_spatial_block_ids, purge_spatial_buffer
from crossview_conflict.labels import to_wildfire_3class_label, to_wildfire_3class_name


PROTOCOL_SCHEMA = "eaton-component-direction-protocol-v1"
SUMMARY_SCHEMA = "eaton-component-direction-spatial-summary-v1"
ROLE_NAMES = (
    "model_fit",
    "model_validation",
    "study_development",
    "spatial_confirmation",
)
REQUIRED_SOURCE_COLUMNS = {
    "pair_id",
    "street_view_relative_path",
    "remote_sensing_relative_path",
    "category",
    "attachment_id",
    "latitude",
    "longitude",
    "dins_source_globalid",
    "remote_tile_filename",
    "remote_crop_box",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_lf(path: Path, payload: Mapping[str, object]) -> None:
    data = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(data)


def _count_map(values: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in values.value_counts(dropna=False).sort_index().items()
    }


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError(f"Expected protocol schema {PROTOCOL_SCHEMA!r}")
    spatial = payload.get("spatial_protocol", {})
    roles = spatial.get("roles", [])
    names = tuple(item.get("name") for item in roles)
    if names != ROLE_NAMES:
        raise ValueError(f"Protocol roles must be exactly {ROLE_NAMES}")
    fractions = np.asarray(
        [item.get("fraction_before_buffer") for item in roles], dtype=float
    )
    if not np.isfinite(fractions).all() or (fractions <= 0).any():
        raise ValueError("Role fractions must be finite and positive")
    if not math.isclose(float(fractions.sum()), 1.0, abs_tol=1e-9):
        raise ValueError("Role fractions must sum to one")
    priority = tuple(spatial.get("buffer_priority", []))
    if set(priority) != set(ROLE_NAMES) or len(priority) != len(ROLE_NAMES):
        raise ValueError("buffer_priority must name every role exactly once")
    if payload.get("label_scheme", {}).get("class_order") != [
        "no_or_trace_damage",
        "damaged_repairable",
        "destroyed",
    ]:
        raise ValueError("The frozen class order changed")
    return payload


def _dependency_group(row: pd.Series) -> str:
    value = row.get("dins_source_globalid")
    if pd.notna(value) and str(value).strip():
        return f"dins:{str(value).strip().casefold()}"
    return f"unmatched:{str(row['pair_id']).strip()}"


def _parse_crop_box(value: object, pair_id: str) -> tuple[int, int, int, int]:
    parts = [item.strip() for item in str(value).split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid remote_crop_box for {pair_id}: {value!r}")
    try:
        x0, y0, x1, y1 = (int(item) for item in parts)
    except ValueError as error:
        raise ValueError(f"Invalid remote_crop_box for {pair_id}: {value!r}") from error
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Non-positive remote crop for {pair_id}: {value!r}")
    return x0, y0, x1, y1


def _media_path(dataset_root: Path, value: object) -> Path:
    if pd.isna(value) or not str(value).strip():
        raise FileNotFoundError("blank media path")
    candidate = (dataset_root / str(value).replace("\\", os.sep)).resolve()
    root = dataset_root.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Media path escapes dataset root: {value!r}")
    return candidate


def inspect_media(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        width, height = image.size
        image_format = str(image.format or "unknown")
        orientation = int(image.getexif().get(274, 1) or 1)
        if orientation != 1:
            raise ValueError(
                f"non-upright EXIF Orientation={orientation} is excluded: {path}"
            )
        image.load()
        extrema = image.convert("RGB").getextrema()
    channel_ranges = tuple(int(high) - int(low) for low, high in extrema)
    if not any(channel_ranges):
        raise ValueError(f"constant-content media is excluded: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "width": int(width),
        "height": int(height),
        "pixel_area": int(width * height),
        "format": image_format,
        "bytes": int(path.stat().st_size),
        "exif_orientation": orientation,
        "channel_range_r": channel_ranges[0],
        "channel_range_g": channel_ranges[1],
        "channel_range_b": channel_ranges[2],
    }


def build_source_cohort(
    joined: pd.DataFrame,
    dataset_root: Path,
    *,
    media_inspector: Callable[[Path], Mapping[str, object]] = inspect_media,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create one readable, media-only canonical pair per DINS structure."""

    missing = sorted(REQUIRED_SOURCE_COLUMNS - set(joined.columns))
    if missing:
        raise ValueError(f"Joined manifest is missing columns: {missing}")
    if joined["pair_id"].duplicated().any():
        raise ValueError("pair_id must be unique")

    candidates: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for row in joined.to_dict(orient="records"):
        pair_id = str(row["pair_id"]).strip()
        category = str(row["category"]).strip()
        base_exclusion = {"pair_id": pair_id, "category": category}
        if category == "Inaccessible":
            exclusions.append({**base_exclusion, "exclusion_reason": "non_ordinal_inaccessible"})
            continue
        try:
            label = to_wildfire_3class_label(category)
        except (KeyError, ValueError) as error:
            exclusions.append(
                {**base_exclusion, "exclusion_reason": f"invalid_source_label:{error}"}
            )
            continue
        if label is None:
            exclusions.append({**base_exclusion, "exclusion_reason": "non_ordinal_label"})
            continue
        try:
            street_path = _media_path(dataset_root, row["street_view_relative_path"])
            overhead_path = _media_path(dataset_root, row["remote_sensing_relative_path"])
            street = dict(media_inspector(street_path))
            overhead = dict(media_inspector(overhead_path))
        except (FileNotFoundError, OSError, ValueError) as error:
            exclusions.append(
                {
                    **base_exclusion,
                    "exclusion_reason": f"unusable_media:{type(error).__name__}:{error}",
                }
            )
            continue
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError(f"Invalid WGS84 coordinates for {pair_id}")
        dependency_group_id = _dependency_group(pd.Series(row))
        remote_tile_filename = str(row["remote_tile_filename"]).strip()
        if not remote_tile_filename:
            raise ValueError(f"Missing remote_tile_filename for {pair_id}")
        crop_x0, crop_y0, crop_x1, crop_y1 = _parse_crop_box(
            row["remote_crop_box"], pair_id
        )
        attachment = pd.to_numeric(pd.Series([row["attachment_id"]]), errors="coerce").iloc[0]
        candidates.append(
            {
                "sample_id": pair_id,
                "pair_id": pair_id,
                "dependency_group_id": dependency_group_id,
                "category": category,
                "label": int(label),
                "label_name": to_wildfire_3class_name(category),
                "latitude": latitude,
                "longitude": longitude,
                "attachment_id": int(attachment) if pd.notna(attachment) else 2**63 - 1,
                "remote_tile_filename": remote_tile_filename,
                "remote_crop_x0": crop_x0,
                "remote_crop_y0": crop_y0,
                "remote_crop_x1": crop_x1,
                "remote_crop_y1": crop_y1,
                "street_view_path": str(street["path"]),
                "remote_sensing_path": str(overhead["path"]),
                "street_sha256": str(street["sha256"]),
                "remote_sha256": str(overhead["sha256"]),
                "street_width": int(street["width"]),
                "street_height": int(street["height"]),
                "street_pixel_area": int(street["pixel_area"]),
                "remote_width": int(overhead["width"]),
                "remote_height": int(overhead["height"]),
                "remote_pixel_area": int(overhead["pixel_area"]),
                "street_format": str(street["format"]),
                "remote_format": str(overhead["format"]),
                "street_bytes": int(street["bytes"]),
                "remote_bytes": int(overhead["bytes"]),
                "street_exif_orientation": int(street["exif_orientation"]),
                "remote_exif_orientation": int(overhead["exif_orientation"]),
                "street_channel_range_r": int(street["channel_range_r"]),
                "street_channel_range_g": int(street["channel_range_g"]),
                "street_channel_range_b": int(street["channel_range_b"]),
                "remote_channel_range_r": int(overhead["channel_range_r"]),
                "remote_channel_range_g": int(overhead["channel_range_g"]),
                "remote_channel_range_b": int(overhead["channel_range_b"]),
            }
        )

    candidate_frame = pd.DataFrame(candidates)
    if candidate_frame.empty:
        raise ValueError("No model-usable Eaton media pairs remain")
    inconsistent = candidate_frame.groupby("dependency_group_id")["label"].nunique()
    if (inconsistent > 1).any():
        examples = inconsistent[inconsistent > 1].index[:5].tolist()
        raise ValueError(f"A DINS structure has conflicting ordinal labels: {examples}")

    ordered = candidate_frame.sort_values(
        ["dependency_group_id", "street_pixel_area", "attachment_id", "pair_id"],
        ascending=[True, False, True, True],
    )
    canonical = ordered.drop_duplicates("dependency_group_id", keep="first").copy()
    noncanonical = ordered.loc[~ordered.index.isin(canonical.index)].copy()
    if not noncanonical.empty:
        exclusions.extend(
            {
                "pair_id": str(row.pair_id),
                "category": str(row.category),
                "exclusion_reason": "noncanonical_structure_attachment",
            }
            for row in noncanonical.itertuples(index=False)
        )
    canonical = canonical.sort_values("pair_id").reset_index(drop=True)
    if canonical["dependency_group_id"].duplicated().any():
        raise RuntimeError("Canonicalization did not produce unique structures")
    media_ledger = canonical[
        [
            "pair_id",
            "dependency_group_id",
            "remote_tile_filename",
            "remote_crop_x0",
            "remote_crop_y0",
            "remote_crop_x1",
            "remote_crop_y1",
            "street_view_path",
            "street_sha256",
            "street_width",
            "street_height",
            "street_format",
            "street_bytes",
            "street_exif_orientation",
            "street_channel_range_r",
            "street_channel_range_g",
            "street_channel_range_b",
            "remote_sensing_path",
            "remote_sha256",
            "remote_width",
            "remote_height",
            "remote_format",
            "remote_bytes",
            "remote_exif_orientation",
            "remote_channel_range_r",
            "remote_channel_range_g",
            "remote_channel_range_b",
        ]
    ].copy()
    return canonical, pd.DataFrame(exclusions), media_ledger


def assign_grouped_roles(
    frame: pd.DataFrame,
    *,
    roles: Sequence[str],
    fractions: Sequence[float],
    group_column: str,
    label_column: str,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Greedily balance indivisible spatial blocks across four roles."""

    if tuple(roles) != ROLE_NAMES:
        raise ValueError(f"roles must be exactly {ROLE_NAMES}")
    fraction_array = np.asarray(fractions, dtype=float)
    if len(fraction_array) != len(roles) or (fraction_array <= 0).any():
        raise ValueError("Need one positive fraction per role")
    if not math.isclose(float(fraction_array.sum()), 1.0, abs_tol=1e-9):
        raise ValueError("Role fractions must sum to one")
    labels = sorted(frame[label_column].astype(int).unique().tolist())
    label_index = {label: index for index, label in enumerate(labels)}
    rng = np.random.default_rng(seed)
    groups: list[tuple[str, int, np.ndarray, float]] = []
    for group_name, group in frame.groupby(group_column, sort=True):
        counts = np.zeros(len(labels), dtype=float)
        for label, count in group[label_column].astype(int).value_counts().items():
            counts[label_index[int(label)]] = float(count)
        groups.append((str(group_name), len(group), counts, float(rng.random())))
    if len(groups) < len(roles):
        raise ValueError("Too few spatial blocks for the declared roles")
    groups.sort(key=lambda item: (-item[1], -float(item[2].max()), item[3], item[0]))

    target_total = fraction_array * len(frame)
    global_labels = np.asarray(
        [(frame[label_column].astype(int) == label).sum() for label in labels], dtype=float
    )
    target_labels = fraction_array[:, None] * global_labels[None, :]
    assigned_total = np.zeros(len(roles), dtype=float)
    assigned_labels = np.zeros((len(roles), len(labels)), dtype=float)
    assignment: dict[str, int] = {}
    for position, (group_name, size, counts, _) in enumerate(groups):
        empty_roles = [index for index in range(len(roles)) if index not in assignment.values()]
        remaining = len(groups) - position
        candidates = empty_roles if remaining == len(empty_roles) else list(range(len(roles)))
        best: tuple[float, int] | None = None
        for role_index in candidates:
            totals = assigned_total.copy()
            by_label = assigned_labels.copy()
            totals[role_index] += size
            by_label[role_index] += counts
            total_error = np.square(
                (totals - target_total) / np.maximum(target_total, 1.0)
            ).mean()
            label_error = np.square(
                (by_label - target_labels) / np.maximum(target_labels, 1.0)
            ).mean()
            overflow = np.square(
                np.maximum(totals - target_total, 0.0) / np.maximum(target_total, 1.0)
            ).mean()
            candidate = (float(total_error + label_error + 2.0 * overflow), role_index)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        selected = best[1]
        assignment[group_name] = selected
        assigned_total[selected] += size
        assigned_labels[selected] += counts

    role_series = frame[group_column].astype(str).map(assignment)
    role_frames = {
        role: frame.loc[role_series == index].copy().reset_index(drop=True)
        for index, role in enumerate(roles)
    }
    expected_labels = set(labels)
    for role, role_frame in role_frames.items():
        if set(role_frame[label_column].astype(int)) != expected_labels:
            raise ValueError(f"Grouped assignment produced class-incomplete role {role}")
    assignment_rows = []
    for group_name, group in frame.groupby(group_column, sort=True):
        assignment_rows.append(
            {
                group_column: str(group_name),
                "protocol_role": roles[assignment[str(group_name)]],
                "rows": int(len(group)),
                **{
                    f"label_{int(label)}_rows": int(count)
                    for label, count in group[label_column].value_counts().sort_index().items()
                },
            }
        )
    return role_frames, pd.DataFrame(assignment_rows).fillna(0)


def _haversine_minimum(left: pd.DataFrame, right: pd.DataFrame) -> float:
    if left.empty or right.empty:
        return float("inf")
    lat1 = np.radians(left["latitude"].to_numpy(dtype=float))[:, None]
    lon1 = np.radians(left["longitude"].to_numpy(dtype=float))[:, None]
    lat2_all = np.radians(right["latitude"].to_numpy(dtype=float))
    lon2_all = np.radians(right["longitude"].to_numpy(dtype=float))
    best = float("inf")
    for start in range(0, len(left), 256):
        stop = min(start + 256, len(left))
        lat2 = lat2_all[None, :]
        lon2 = lon2_all[None, :]
        dlat = lat2 - lat1[start:stop]
        dlon = lon2 - lon1[start:stop]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1[start:stop]) * np.cos(lat2) * np.sin(
            dlon / 2
        ) ** 2
        distance = 2 * 6_371_008.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        best = min(best, float(distance.min()))
    return best


def add_media_dependency_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Join spatial blocks linked by byte-identical media into split components."""

    required = {"spatial_block_id", "street_sha256", "remote_sha256"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Media dependency construction is missing columns: {missing}")
    blocks = sorted(frame["spatial_block_id"].astype(str).unique())
    parent = {block: block for block in blocks}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for column in ("street_sha256", "remote_sha256"):
        for _, group in frame.groupby(column, sort=True):
            linked = sorted(group["spatial_block_id"].astype(str).unique())
            for block in linked[1:]:
                union(linked[0], block)
    roots = sorted({find(block) for block in blocks})
    root_to_id = {root: f"media_component_{index:04d}" for index, root in enumerate(roots)}
    result = frame.copy()
    result["split_dependency_group_id"] = result["spatial_block_id"].astype(str).map(
        lambda block: root_to_id[find(block)]
    )
    return result


def _remote_crop_overlap_audit(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[int, float | None]:
    """Count positive-area cross-role crop intersections on the same source tile."""

    required = {
        "remote_tile_filename",
        "remote_crop_x0",
        "remote_crop_y0",
        "remote_crop_x1",
        "remote_crop_y1",
    }
    missing = sorted(required - set(left.columns) | (required - set(right.columns)))
    if missing:
        raise ValueError(f"Remote crop audit is missing columns: {missing}")
    shared_tiles = sorted(
        set(left["remote_tile_filename"].astype(str))
        & set(right["remote_tile_filename"].astype(str))
    )
    overlap_count = 0
    minimum_center_distance = float("inf")
    columns = [
        "remote_crop_x0",
        "remote_crop_y0",
        "remote_crop_x1",
        "remote_crop_y1",
    ]
    for tile in shared_tiles:
        left_boxes = left.loc[
            left["remote_tile_filename"].astype(str).eq(tile), columns
        ].to_numpy(dtype=np.int64)
        right_boxes = right.loc[
            right["remote_tile_filename"].astype(str).eq(tile), columns
        ].to_numpy(dtype=np.int64)
        right_centers = np.column_stack(
            (
                (right_boxes[:, 0] + right_boxes[:, 2]) / 2.0,
                (right_boxes[:, 1] + right_boxes[:, 3]) / 2.0,
            )
        )
        for start in range(0, len(left_boxes), 256):
            boxes = left_boxes[start : start + 256]
            intersects = (
                (boxes[:, None, 0] < right_boxes[None, :, 2])
                & (boxes[:, None, 2] > right_boxes[None, :, 0])
                & (boxes[:, None, 1] < right_boxes[None, :, 3])
                & (boxes[:, None, 3] > right_boxes[None, :, 1])
            )
            overlap_count += int(intersects.sum())
            left_centers = np.column_stack(
                (
                    (boxes[:, 0] + boxes[:, 2]) / 2.0,
                    (boxes[:, 1] + boxes[:, 3]) / 2.0,
                )
            )
            distances = np.sqrt(
                np.square(left_centers[:, None, 0] - right_centers[None, :, 0])
                + np.square(left_centers[:, None, 1] - right_centers[None, :, 1])
            )
            minimum_center_distance = min(
                minimum_center_distance, float(distances.min())
            )
    return overlap_count, (
        minimum_center_distance if math.isfinite(minimum_center_distance) else None
    )


def pairwise_role_audit(
    roles: Mapping[str, pd.DataFrame], buffer_m: float
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left, right in combinations(ROLE_NAMES, 2):
        left_frame, right_frame = roles[left], roles[right]
        minimum = _haversine_minimum(left_frame, right_frame)
        crop_overlaps, minimum_crop_center_distance = _remote_crop_overlap_audit(
            left_frame, right_frame
        )
        rows.append(
            {
                "left_role": left,
                "right_role": right,
                "pair_id_overlap": len(
                    set(left_frame["pair_id"].astype(str))
                    & set(right_frame["pair_id"].astype(str))
                ),
                "dependency_group_overlap": len(
                    set(left_frame["dependency_group_id"].astype(str))
                    & set(right_frame["dependency_group_id"].astype(str))
                ),
                "spatial_block_overlap": len(
                    set(left_frame["spatial_block_id"].astype(str))
                    & set(right_frame["spatial_block_id"].astype(str))
                ),
                "street_media_sha_overlap": len(
                    set(left_frame["street_sha256"].astype(str))
                    & set(right_frame["street_sha256"].astype(str))
                ),
                "remote_media_sha_overlap": len(
                    set(left_frame["remote_sha256"].astype(str))
                    & set(right_frame["remote_sha256"].astype(str))
                ),
                "overlapping_remote_crop_pairs": crop_overlaps,
                "minimum_same_tile_crop_center_distance_px": minimum_crop_center_distance,
                "minimum_distance_m": minimum,
                "strictly_beyond_buffer": bool(minimum > buffer_m),
            }
        )
    return pd.DataFrame(rows)


def build_spatial_protocol(
    cohort: pd.DataFrame, protocol: Mapping[str, Any]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spatial = protocol["spatial_protocol"]
    blocked = add_spatial_block_ids(
        cohort.rename(columns={"pair_id": "_pair_id"}).assign(
            sample_id=cohort["pair_id"].astype(str).to_numpy()
        ),
        grid_degrees=float(spatial["grid_degrees"]),
    ).rename(columns={"_pair_id": "pair_id"})
    spans = blocked.groupby("dependency_group_id")["spatial_block_id"].nunique()
    if (spans > 1).any():
        raise ValueError("A dependency group spans multiple frozen spatial blocks")
    blocked = add_media_dependency_components(blocked)
    fractions = [item["fraction_before_buffer"] for item in spatial["roles"]]
    raw_roles, assignment = assign_grouped_roles(
        blocked,
        roles=ROLE_NAMES,
        fractions=fractions,
        group_column="split_dependency_group_id",
        label_column="label",
        seed=int(spatial["split_seed"]),
    )
    roles, buffer_exclusions = purge_spatial_buffer(
        raw_roles,
        float(spatial["cross_role_buffer_m"]),
        priority=tuple(spatial["buffer_priority"]),
    )
    expected_labels = set(blocked["label"].astype(int))
    for role, frame in roles.items():
        if frame.empty or set(frame["label"].astype(int)) != expected_labels:
            raise ValueError(f"Buffer purge made {role} empty or class-incomplete")
        frame["protocol_role"] = role
    audit = pairwise_role_audit(roles, float(spatial["cross_role_buffer_m"]))
    zero_columns = [
        "pair_id_overlap",
        "dependency_group_overlap",
        "spatial_block_overlap",
        "street_media_sha_overlap",
        "remote_media_sha_overlap",
        "overlapping_remote_crop_pairs",
    ]
    if audit[zero_columns].to_numpy().any() or not audit["strictly_beyond_buffer"].all():
        raise RuntimeError("Spatial protocol failed its leakage audit")
    return roles, buffer_exclusions, assignment, audit


def write_protocol_artifacts(
    *,
    output_dir: Path,
    roles: Mapping[str, pd.DataFrame],
    source_exclusions: pd.DataFrame,
    buffer_exclusions: pd.DataFrame,
    assignment: pd.DataFrame,
    pairwise_audit: pd.DataFrame,
    media_ledger: pd.DataFrame,
    joined_manifest: Path,
    dataset_root: Path,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    canonical_rows_before_buffer: int,
    overwrite: bool,
) -> dict[str, object]:
    filenames = [f"{role}.csv" for role in ROLE_NAMES] + [
        "source_exclusions.csv",
        "buffer_exclusions.csv",
        "spatial_block_assignment.csv",
        "pairwise_role_audit.csv",
        "media_hash_ledger.csv",
        "confirmation_commitment.json",
        "protocol_summary.json",
    ]
    existing = [output_dir / name for name in filenames if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {existing[0]}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    for role in ROLE_NAMES:
        roles[role].sort_values("pair_id").to_csv(output_dir / f"{role}.csv", index=False)
    source_exclusions.to_csv(output_dir / "source_exclusions.csv", index=False)
    buffer_exclusions.to_csv(output_dir / "buffer_exclusions.csv", index=False)
    assignment.to_csv(output_dir / "spatial_block_assignment.csv", index=False)
    pairwise_audit.to_csv(output_dir / "pairwise_role_audit.csv", index=False)
    media_ledger.to_csv(output_dir / "media_hash_ledger.csv", index=False)

    role_hashes = {
        role: sha256_file(output_dir / f"{role}.csv") for role in ROLE_NAMES
    }
    confirmation = roles["spatial_confirmation"]
    commitment = {
        "schema_version": "eaton-component-direction-confirmation-commitment-v1",
        "protocol_version": protocol["protocol_version"],
        "role": "spatial_confirmation",
        "status": "UNSCORED_RESERVED_SAME_EVENT_SPATIAL_CONFIRMATION",
        "external_event_confirmation": False,
        "manifest_sha256": role_hashes["spatial_confirmation"],
        "row_count": int(len(confirmation)),
        "spatial_block_count": int(confirmation["spatial_block_id"].nunique()),
        "class_counts": _count_map(confirmation["label"]),
        "use_policy": (
            "Do not infer or annotate signed confirmation disagreements until the model, "
            "ensemble, annotation codebook, power gate, estimator, and hashes are frozen."
        ),
    }
    _write_json_lf(output_dir / "confirmation_commitment.json", commitment)

    summary: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA,
        "protocol_version": protocol["protocol_version"],
        "event_id": protocol["event_id"],
        "research_question": protocol["research_question"],
        "joined_manifest": str(joined_manifest.resolve()),
        "joined_manifest_sha256": sha256_file(joined_manifest),
        "dataset_root": str(dataset_root.resolve()),
        "protocol_config": str(protocol_path.resolve()),
        "protocol_config_sha256": sha256_file(protocol_path),
        "canonical_rows_before_buffer": int(canonical_rows_before_buffer),
        "source_exclusion_rows": int(len(source_exclusions)),
        "buffer_exclusion_rows": int(len(buffer_exclusions)),
        "role_rows": {role: int(len(roles[role])) for role in ROLE_NAMES},
        "role_structures": {
            role: int(roles[role]["dependency_group_id"].nunique()) for role in ROLE_NAMES
        },
        "role_spatial_blocks": {
            role: int(roles[role]["spatial_block_id"].nunique()) for role in ROLE_NAMES
        },
        "role_class_counts": {
            role: _count_map(roles[role]["label"]) for role in ROLE_NAMES
        },
        "role_manifest_sha256": role_hashes,
        "media_hash_ledger_sha256": sha256_file(output_dir / "media_hash_ledger.csv"),
        "pairwise_role_audit": pairwise_audit.to_dict(orient="records"),
        "confirmation_commitment": commitment,
        "analysis_status": "PROTOCOL_READY_CONFIRMATION_UNTOUCHED",
    }
    _write_json_lf(output_dir / "protocol_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=REPO_ROOT / "configs" / "eaton_component_direction_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "splits" / "eaton_component_direction_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for path in (args.joined_manifest, args.protocol_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.dataset_root.is_dir():
        raise NotADirectoryError(args.dataset_root)
    protocol = load_protocol(args.protocol_config)
    joined = pd.read_csv(args.joined_manifest, low_memory=False)
    cohort, source_exclusions, media_ledger = build_source_cohort(
        joined, args.dataset_root
    )
    roles, buffer_exclusions, assignment, audit = build_spatial_protocol(cohort, protocol)
    summary = write_protocol_artifacts(
        output_dir=args.output_dir,
        roles=roles,
        source_exclusions=source_exclusions,
        buffer_exclusions=buffer_exclusions,
        assignment=assignment,
        pairwise_audit=audit,
        media_ledger=media_ledger,
        joined_manifest=args.joined_manifest,
        dataset_root=args.dataset_root,
        protocol_path=args.protocol_config,
        protocol=protocol,
        canonical_rows_before_buffer=len(cohort),
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
