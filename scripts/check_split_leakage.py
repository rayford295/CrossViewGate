"""Audit train/validation/test splits for group and spatial leakage.

The command supports one or more datasets.  The preferred form is repeatable::

    python scripts/check_split_leakage.py \
        --dataset wildfire train.csv val.csv test.csv \
        --dataset hurricane h_train.csv h_val.csv h_test.csv \
        --group-cols objectid sequence_id spatial_block_id \
        --spatial-threshold-m 100

For a single dataset, ``--train``, ``--val`` and ``--test`` are also accepted.
The historical ``--wildfire-*`` and ``--hurricane-*`` arguments remain
supported so existing experiment commands continue to work.

Programmatic callers can use :func:`audit_split_frames` (already-loaded data)
or :func:`audit_split_paths` (CSV paths).  Both return a JSON-serialisable
mapping.  The historical :func:`check_leakage` function still returns a plain
boolean so existing callers remain source-compatible.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SPLIT_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))
EARTH_RADIUS_M = 6_371_008.8
DEFAULT_GROUP_COLUMNS = ("objectid",)


class LeakageAuditResult(dict[str, Any]):
    """A JSON-compatible result whose truth value means "leakage found".

    This makes structured callers concise (``if result:``) while the separate
    compatibility wrapper still returns a plain boolean.
    """

    def __bool__(self) -> bool:
        return bool(self.get("leakage_found", False))


def _pair_name(left: str, right: str) -> str:
    return f"{left}_{right}"


def _normalise_group_value(value: object) -> str | None:
    """Create stable group keys across common CSV type-inference differences."""

    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, ".15g")
    text = str(value).strip()
    return text or None


def _group_keys(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].map(_normalise_group_value)


def _audit_group_pair(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    left_name: str,
    right_name: str,
    column: str,
) -> dict[str, Any]:
    missing = [
        split_name
        for split_name, frame in ((left_name, left_frame), (right_name, right_frame))
        if column not in frame.columns
    ]
    base: dict[str, Any] = {
        "splits": [left_name, right_name],
        "available": not missing,
        "missing_splits": missing,
        "empty_splits": [],
        "valid_group_sample_count_by_split": {left_name: 0, right_name: 0},
        "overlap_group_count": 0,
        "overlap_sample_count": 0,
        "overlap_sample_count_by_split": {left_name: 0, right_name: 0},
        "example_groups": [],
        "leakage_found": False,
    }
    if missing:
        return base

    left_keys = _group_keys(left_frame, column)
    right_keys = _group_keys(right_frame, column)
    valid_counts = {
        left_name: int(left_keys.notna().sum()),
        right_name: int(right_keys.notna().sum()),
    }
    empty_splits = [name for name, count in valid_counts.items() if count == 0]
    base["valid_group_sample_count_by_split"] = valid_counts
    base["empty_splits"] = empty_splits
    base["available"] = not empty_splits
    if empty_splits:
        return base

    left_groups = set(left_keys.dropna())
    right_groups = set(right_keys.dropna())
    overlap = left_groups & right_groups
    left_sample_count = int(left_keys.isin(overlap).sum())
    right_sample_count = int(right_keys.isin(overlap).sum())
    base.update(
        {
            "overlap_group_count": len(overlap),
            # Samples means all rows affected by a shared group, across both
            # sides of the pair.  Per-split counts make the definition explicit.
            "overlap_sample_count": left_sample_count + right_sample_count,
            "overlap_sample_count_by_split": {
                left_name: left_sample_count,
                right_name: right_sample_count,
            },
            "example_groups": sorted(overlap)[:5],
            "leakage_found": bool(overlap),
        }
    )
    return base


def _audit_group_column(
    splits: Mapping[str, pd.DataFrame], column: str
) -> dict[str, Any]:
    pairs = {
        _pair_name(left, right): _audit_group_pair(
            splits[left], splits[right], left, right, column
        )
        for left, right in SPLIT_PAIRS
    }
    split_metadata: dict[str, dict[str, Any]] = {}
    for name, frame in splits.items():
        if column not in frame.columns:
            split_metadata[name] = {
                "row_count": len(frame),
                "valid_group_sample_count": 0,
                "missing_group_sample_count": len(frame),
                "unique_group_count": 0,
                "column_missing": True,
            }
            continue
        keys = _group_keys(frame, column)
        valid_count = int(keys.notna().sum())
        split_metadata[name] = {
            "row_count": len(frame),
            "valid_group_sample_count": valid_count,
            "missing_group_sample_count": len(frame) - valid_count,
            "unique_group_count": int(keys.dropna().nunique()),
            "column_missing": False,
        }
    missing_splits = [
        name for name, metadata in split_metadata.items() if metadata["column_missing"]
    ]
    empty_splits = [
        name
        for name, metadata in split_metadata.items()
        if not metadata["column_missing"] and metadata["valid_group_sample_count"] == 0
    ]
    available = not missing_splits and not empty_splits
    complete = available and all(
        metadata["missing_group_sample_count"] == 0
        for metadata in split_metadata.values()
    )
    return {
        "column": column,
        "available": available,
        "complete": complete,
        "missing_splits": missing_splits,
        "empty_splits": empty_splits,
        "splits": split_metadata,
        "pairs": pairs,
        "leakage_found": any(pair["leakage_found"] for pair in pairs.values()),
    }


def _extract_coordinates(
    frame: pd.DataFrame, latitude_column: str, longitude_column: str
) -> tuple[np.ndarray | None, dict[str, Any]]:
    missing_columns = [
        column
        for column in (latitude_column, longitude_column)
        if column not in frame.columns
    ]
    metadata: dict[str, Any] = {
        "row_count": len(frame),
        "valid_coordinate_count": 0,
        "invalid_coordinate_count": len(frame),
        "missing_columns": missing_columns,
    }
    if missing_columns:
        return None, metadata

    latitude = pd.to_numeric(frame[latitude_column], errors="coerce").to_numpy(dtype=float)
    longitude = pd.to_numeric(frame[longitude_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(latitude) & np.isfinite(longitude)
    valid &= (latitude >= -90.0) & (latitude <= 90.0)
    valid &= (longitude >= -180.0) & (longitude <= 180.0)
    coordinates = np.column_stack((latitude[valid], longitude[valid]))
    metadata["valid_coordinate_count"] = int(valid.sum())
    metadata["invalid_coordinate_count"] = int((~valid).sum())
    return coordinates, metadata


def _distance_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum_m": float(np.min(values)),
        "median_m": float(np.median(values)),
        "maximum_m": float(np.max(values)),
    }


def _audit_spatial_pair(
    left_coordinates: np.ndarray | None,
    right_coordinates: np.ndarray | None,
    left_name: str,
    right_name: str,
    threshold_m: float | None,
    chunk_size: int,
) -> dict[str, Any]:
    left_count = 0 if left_coordinates is None else len(left_coordinates)
    right_count = 0 if right_coordinates is None else len(right_coordinates)
    available = left_count > 0 and right_count > 0
    result: dict[str, Any] = {
        "splits": [left_name, right_name],
        "available": available,
        "valid_coordinate_count_by_split": {
            left_name: left_count,
            right_name: right_count,
        },
        "minimum_distance_m": None,
        "nearest_distance_by_split": {left_name: None, right_name: None},
        "threshold_m": threshold_m,
        "within_threshold_pair_count": 0,
        "within_threshold_sample_count": 0,
        "within_threshold_sample_count_by_split": {left_name: 0, right_name: 0},
        "leakage_found": False,
    }
    if not available:
        return result

    # Haversine blocks keep memory bounded and require only NumPy.  Exact
    # nearest distances are retained for every point in both directions.
    left_radians = np.radians(left_coordinates)
    right_radians = np.radians(right_coordinates)
    left_nearest = np.full(left_count, np.inf, dtype=float)
    right_nearest = np.full(right_count, np.inf, dtype=float)
    left_within = np.zeros(left_count, dtype=bool)
    right_within = np.zeros(right_count, dtype=bool)
    close_pair_count = 0

    for left_start in range(0, left_count, chunk_size):
        left_stop = min(left_start + chunk_size, left_count)
        left_block = left_radians[left_start:left_stop]
        lat_left = left_block[:, 0, None]
        lon_left = left_block[:, 1, None]
        for right_start in range(0, right_count, chunk_size):
            right_stop = min(right_start + chunk_size, right_count)
            right_block = right_radians[right_start:right_stop]
            lat_right = right_block[None, :, 0]
            lon_right = right_block[None, :, 1]

            delta_latitude = lat_right - lat_left
            delta_longitude = lon_right - lon_left
            haversine = np.sin(delta_latitude / 2.0) ** 2
            haversine += (
                np.cos(lat_left)
                * np.cos(lat_right)
                * np.sin(delta_longitude / 2.0) ** 2
            )
            np.clip(haversine, 0.0, 1.0, out=haversine)
            distances = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(haversine))

            left_nearest[left_start:left_stop] = np.minimum(
                left_nearest[left_start:left_stop], distances.min(axis=1)
            )
            right_nearest[right_start:right_stop] = np.minimum(
                right_nearest[right_start:right_stop], distances.min(axis=0)
            )
            if threshold_m is not None:
                within = distances <= threshold_m
                close_pair_count += int(within.sum())
                left_within[left_start:left_stop] |= within.any(axis=1)
                right_within[right_start:right_stop] |= within.any(axis=0)

    left_within_count = int(left_within.sum())
    right_within_count = int(right_within.sum())
    result.update(
        {
            "minimum_distance_m": float(min(left_nearest.min(), right_nearest.min())),
            "nearest_distance_by_split": {
                left_name: _distance_summary(left_nearest),
                right_name: _distance_summary(right_nearest),
            },
            "within_threshold_pair_count": close_pair_count,
            "within_threshold_sample_count": left_within_count + right_within_count,
            "within_threshold_sample_count_by_split": {
                left_name: left_within_count,
                right_name: right_within_count,
            },
            "leakage_found": threshold_m is not None and close_pair_count > 0,
        }
    )
    return result


def _audit_spatial(
    splits: Mapping[str, pd.DataFrame],
    latitude_column: str,
    longitude_column: str,
    threshold_m: float | None,
    chunk_size: int,
) -> dict[str, Any]:
    coordinates: dict[str, np.ndarray | None] = {}
    split_metadata: dict[str, dict[str, Any]] = {}
    for name, frame in splits.items():
        coordinates[name], split_metadata[name] = _extract_coordinates(
            frame, latitude_column, longitude_column
        )

    pairs = {
        _pair_name(left, right): _audit_spatial_pair(
            coordinates[left],
            coordinates[right],
            left,
            right,
            threshold_m,
            chunk_size,
        )
        for left, right in SPLIT_PAIRS
    }
    return {
        "enabled": True,
        "coordinate_columns": {
            "latitude": latitude_column,
            "longitude": longitude_column,
        },
        "threshold_m": threshold_m,
        "splits": split_metadata,
        "pairs": pairs,
        "available": all(pair["available"] for pair in pairs.values()),
        "complete": all(
            metadata["valid_coordinate_count"] == metadata["row_count"]
            for metadata in split_metadata.values()
        )
        and all(pair["available"] for pair in pairs.values()),
        # Distances alone are diagnostic.  Spatial leakage is asserted only
        # when the caller supplies an explicit operational threshold.
        "leakage_found": any(pair["leakage_found"] for pair in pairs.values()),
    }


def audit_split_frames(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label: str = "Dataset",
    *,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    audit_coordinates: bool = False,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    spatial_threshold_m: float | None = None,
    spatial_chunk_size: int = 1024,
) -> LeakageAuditResult:
    """Audit loaded split frames and return a structured result.

    Group leakage is evaluated independently for each requested column and
    each split pair.  ``overlap_sample_count`` is the total number of rows in
    both splits belonging to a shared group.

    When coordinate auditing is enabled, the exact cross-split nearest-neighbor
    distance is reported in both directions.  Spatial leakage contributes to
    ``leakage_found`` only when ``spatial_threshold_m`` is supplied.
    """

    if isinstance(group_columns, str):
        group_columns = [group_columns]
    columns = [str(column).strip() for column in group_columns if str(column).strip()]
    columns = list(dict.fromkeys(columns))
    if not columns:
        raise ValueError("At least one group column is required.")
    if spatial_threshold_m is not None:
        if not math.isfinite(spatial_threshold_m) or spatial_threshold_m < 0:
            raise ValueError("spatial_threshold_m must be a finite, non-negative number.")
        audit_coordinates = True
    if spatial_chunk_size <= 0:
        raise ValueError("spatial_chunk_size must be positive.")

    splits = {"train": train, "val": val, "test": test}
    group_audits = {
        column: _audit_group_column(splits, column) for column in columns
    }
    spatial_audit: dict[str, Any]
    if audit_coordinates:
        spatial_audit = _audit_spatial(
            splits,
            latitude_column,
            longitude_column,
            spatial_threshold_m,
            spatial_chunk_size,
        )
    else:
        spatial_audit = {
            "enabled": False,
            "coordinate_columns": {
                "latitude": latitude_column,
                "longitude": longitude_column,
            },
            "threshold_m": spatial_threshold_m,
            "splits": {},
            "pairs": {},
            "available": False,
            "complete": False,
            "leakage_found": False,
        }

    leakage_found = any(audit["leakage_found"] for audit in group_audits.values())
    leakage_found |= bool(spatial_audit["leakage_found"])
    audit_complete = all(audit["complete"] for audit in group_audits.values())
    if audit_coordinates:
        audit_complete &= bool(spatial_audit["complete"])

    return LeakageAuditResult(
        {
            "label": label,
            "split_row_counts": {name: len(frame) for name, frame in splits.items()},
            "group_columns": columns,
            "group_audits": group_audits,
            "spatial_audit": spatial_audit,
            "audit_complete": audit_complete,
            "leakage_found": leakage_found,
        }
    )


def audit_split_paths(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    label: str = "Dataset",
    *,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    audit_coordinates: bool = False,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    spatial_threshold_m: float | None = None,
    spatial_chunk_size: int = 1024,
    print_report: bool = True,
) -> LeakageAuditResult:
    """Load three split CSVs, audit them, and return the structured result."""

    paths = {
        "train": Path(train_path),
        "val": Path(val_path),
        "test": Path(test_path),
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    result = audit_split_frames(
        frames["train"],
        frames["val"],
        frames["test"],
        label,
        group_columns=group_columns,
        audit_coordinates=audit_coordinates,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        spatial_threshold_m=spatial_threshold_m,
        spatial_chunk_size=spatial_chunk_size,
    )
    result["split_paths"] = {name: str(path) for name, path in paths.items()}
    if print_report:
        print_audit_report(result)
    return result


def check_leakage(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    label: str = "Dataset",
    *,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    audit_coordinates: bool = False,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    spatial_threshold_m: float | None = None,
    spatial_chunk_size: int = 1024,
    print_report: bool = True,
) -> bool:
    """Compatibility wrapper returning only whether leakage was found.

    New code should call :func:`audit_split_paths` to retain the structured
    evidence behind the decision.
    """

    result = audit_split_paths(
        train_path,
        val_path,
        test_path,
        label,
        group_columns=group_columns,
        audit_coordinates=audit_coordinates,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        spatial_threshold_m=spatial_threshold_m,
        spatial_chunk_size=spatial_chunk_size,
        print_report=print_report,
    )
    return bool(result)


def _format_distance(distance_m: float | None) -> str:
    if distance_m is None:
        return "unavailable"
    if distance_m < 1_000:
        return f"{distance_m:.2f} m"
    return f"{distance_m / 1_000:.3f} km"


def print_audit_report(result: Mapping[str, Any]) -> None:
    """Print a compact human-readable rendering of a structured audit."""

    print(f"\n{'=' * 64}")
    print(f"  {result['label']}")
    counts = result["split_row_counts"]
    print(f"  train={counts['train']}  val={counts['val']}  test={counts['test']}")
    print(f"{'=' * 64}")

    for column, audit in result["group_audits"].items():
        print(f"  Group column: {column}")
        if audit["missing_splits"]:
            missing = ", ".join(audit["missing_splits"])
            print(f"    WARNING: column missing from split(s): {missing}")
        if audit["empty_splits"]:
            empty = ", ".join(audit["empty_splits"])
            print(f"    WARNING: no usable group values in split(s): {empty}")
        for split_name, metadata in audit["splits"].items():
            if metadata["column_missing"]:
                continue
            if metadata["missing_group_sample_count"]:
                print(
                    f"    WARNING: {split_name} has "
                    f"{metadata['missing_group_sample_count']} sample(s) without a group value"
                )
        for pair_name, pair in audit["pairs"].items():
            left, right = pair["splits"]
            if not pair["available"]:
                print(f"    [SKIP] {left} / {right}: usable group values unavailable")
                continue
            status = "LEAK" if pair["leakage_found"] else "OK"
            samples = pair["overlap_sample_count_by_split"]
            print(
                f"    [{status}] {left} / {right}: "
                f"{pair['overlap_group_count']} overlapping group(s), "
                f"{pair['overlap_sample_count']} affected sample(s) "
                f"({left}={samples[left]}, {right}={samples[right]})"
            )
            if pair["example_groups"]:
                print(f"      example groups: {pair['example_groups']}")

    spatial = result["spatial_audit"]
    if spatial["enabled"]:
        latitude = spatial["coordinate_columns"]["latitude"]
        longitude = spatial["coordinate_columns"]["longitude"]
        print(f"  Spatial nearest-neighbor audit: {latitude}, {longitude}")
        for split_name, metadata in spatial["splits"].items():
            if metadata["missing_columns"]:
                missing = ", ".join(metadata["missing_columns"])
                print(f"    WARNING: {split_name} missing coordinate column(s): {missing}")
            elif metadata["invalid_coordinate_count"]:
                print(
                    f"    WARNING: {split_name} has "
                    f"{metadata['invalid_coordinate_count']} invalid/missing coordinate row(s)"
                )
        for pair in spatial["pairs"].values():
            left, right = pair["splits"]
            if not pair["available"]:
                print(f"    [SKIP] {left} / {right}: no valid coordinate pair")
                continue
            minimum = _format_distance(pair["minimum_distance_m"])
            if pair["threshold_m"] is None:
                print(f"    [INFO] {left} / {right}: minimum distance {minimum}")
                continue
            status = "LEAK" if pair["leakage_found"] else "OK"
            samples = pair["within_threshold_sample_count_by_split"]
            print(
                f"    [{status}] {left} / {right}: minimum distance {minimum}; "
                f"{pair['within_threshold_pair_count']} pair(s) and "
                f"{pair['within_threshold_sample_count']} sample(s) within "
                f"{pair['threshold_m']:g} m "
                f"({left}={samples[left]}, {right}={samples[right]})"
            )

    if result["leakage_found"]:
        print("  RESULT: leakage found; regenerate the affected split(s).")
    elif not result["audit_complete"]:
        print("  RESULT: no leakage found, but the requested audit was incomplete.")
    else:
        print("  RESULT: no leakage found.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit one or more train/val/test datasets for group and spatial leakage."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=4,
        metavar=("NAME", "TRAIN", "VAL", "TEST"),
        help="Dataset name and its train/val/test CSVs. Repeat for multiple datasets.",
    )
    parser.add_argument("--train", help="Train CSV for a single generic dataset.")
    parser.add_argument("--val", help="Validation CSV for a single generic dataset.")
    parser.add_argument("--test", help="Test CSV for a single generic dataset.")
    parser.add_argument("--label", default="Dataset", help="Label used with --train/--val/--test.")

    # Historical arguments are intentionally optional now: either legacy
    # dataset can be checked alone, or both can be checked as before.
    for dataset_name in ("wildfire", "hurricane"):
        parser.add_argument(f"--{dataset_name}-train")
        parser.add_argument(f"--{dataset_name}-val")
        parser.add_argument(f"--{dataset_name}-test")

    parser.add_argument(
        "--group-cols",
        "--group-columns",
        "--group-col",
        dest="group_column_groups",
        action="append",
        nargs="+",
        metavar="COLUMN",
        help=(
            "Group columns to audit (space- or comma-separated). May be repeated; "
            "default: objectid. Common choices: objectid, sequence_id, spatial_block_id."
        ),
    )
    parser.add_argument(
        "--audit-coordinates",
        "--spatial-audit",
        "--nearest-neighbor",
        dest="audit_coordinates",
        action="store_true",
        help="Report cross-split nearest-neighbor distances using latitude/longitude.",
    )
    parser.add_argument(
        "--spatial-threshold-m",
        "--distance-threshold-m",
        type=float,
        help="Flag cross-split coordinate pairs at or below this distance in metres.",
    )
    parser.add_argument("--latitude-col", "--lat-col", default="latitude")
    parser.add_argument("--longitude-col", "--lon-col", default="longitude")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally save the full structured audit as JSON.",
    )
    return parser


def _collect_dataset_specs(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[tuple[str, str, str, str]]:
    specs: list[tuple[str, str, str, str]] = []
    if args.dataset:
        specs.extend(tuple(values) for values in args.dataset)

    generic = (args.train, args.val, args.test)
    if any(generic):
        if not all(generic):
            parser.error("--train, --val and --test must be provided together.")
        specs.append((args.label, args.train, args.val, args.test))

    for dataset_name, label in (("wildfire", "Wildfire"), ("hurricane", "Hurricane")):
        legacy = tuple(
            getattr(args, f"{dataset_name}_{split_name}")
            for split_name in ("train", "val", "test")
        )
        if any(legacy):
            if not all(legacy):
                parser.error(
                    f"--{dataset_name}-train, --{dataset_name}-val and "
                    f"--{dataset_name}-test must be provided together."
                )
            specs.append((label, legacy[0], legacy[1], legacy[2]))

    if not specs:
        parser.error(
            "provide at least one --dataset, a --train/--val/--test set, "
            "or a complete legacy wildfire/hurricane set"
        )
    return specs


def _parse_group_columns(groups: Sequence[Sequence[str]] | None) -> list[str]:
    if not groups:
        return list(DEFAULT_GROUP_COLUMNS)
    columns: list[str] = []
    for group in groups:
        for value in group:
            columns.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(columns))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    specs = _collect_dataset_specs(parser, args)
    group_columns = _parse_group_columns(args.group_column_groups)
    if not group_columns:
        parser.error("at least one non-empty group column is required")
    if args.spatial_threshold_m is not None and (
        not math.isfinite(args.spatial_threshold_m) or args.spatial_threshold_m < 0
    ):
        parser.error("--spatial-threshold-m must be a finite, non-negative number")

    results: list[LeakageAuditResult] = []
    try:
        for label, train_path, val_path, test_path in specs:
            results.append(
                audit_split_paths(
                    train_path,
                    val_path,
                    test_path,
                    label,
                    group_columns=group_columns,
                    audit_coordinates=args.audit_coordinates,
                    latitude_column=args.latitude_col,
                    longitude_column=args.longitude_col,
                    spatial_threshold_m=args.spatial_threshold_m,
                    print_report=True,
                )
            )
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    any_leakage = any(result["leakage_found"] for result in results)
    audit_complete = all(result["audit_complete"] for result in results)
    if args.json_output:
        payload = {
            # ``LeakageAuditResult.__bool__`` intentionally means "leakage
            # found" for programmatic callers.  The stdlib JSON encoder uses
            # mapping truthiness as an empty-mapping fast path, so a clean
            # result would otherwise be emitted as ``{}``.  Convert each
            # result to a plain dict at the serialization boundary.
            "datasets": [dict(result) for result in results],
            "leakage_found": any_leakage,
            "audit_complete": audit_complete,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nSaved structured audit to {args.json_output}")

    print(f"\n{'=' * 64}")
    if any_leakage:
        print("OVERALL: leakage found")
        return 1
    if not audit_complete:
        print("OVERALL: audit incomplete; no clean-split claim is authorized")
        return 2
    print("OVERALL: no leakage found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
