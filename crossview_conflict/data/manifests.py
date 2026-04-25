from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from crossview_conflict.labels import normalize_category, to_binary_label, to_binary_name
from crossview_conflict.utils.io import ensure_dir, save_json


def _as_relative_path(raw_path: object) -> Path:
    return Path(str(raw_path).replace("\\", "/"))


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_readable_image(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def summarize_manifest(df: pd.DataFrame) -> dict[str, Any]:
    usable = df[df["binary_label"].notna()]
    summary = {
        "rows": int(len(df)),
        "usable_binary_rows": int(len(usable)),
        "unique_objectids": int(df["objectid"].nunique(dropna=True)) if "objectid" in df else 0,
        "unique_remote_tiles": int(df["remote_tile_filename"].nunique(dropna=True))
        if "remote_tile_filename" in df
        else 0,
        "category_counts": df["category"].value_counts(dropna=False).to_dict() if "category" in df else {},
        "binary_counts": usable["binary_name"].value_counts(dropna=False).to_dict()
        if "binary_name" in usable
        else {},
    }
    return summary


def build_altadena_manifest(
    index_csv: str | Path,
    dataset_root: str | Path,
    output_csv: str | Path,
    binary_scheme: str = "operational",
    require_complete_pairs: bool = True,
    verify_images: bool = True,
) -> pd.DataFrame:
    index_csv = Path(index_csv)
    dataset_root = Path(dataset_root)
    output_csv = Path(output_csv)
    ensure_dir(output_csv.parent)

    df = pd.read_csv(index_csv)
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        category = normalize_category(row["category"])
        binary_label = to_binary_label(category, scheme=binary_scheme)
        street_rel = _as_relative_path(row["street_view_relative_path"])
        remote_rel = _as_relative_path(row["remote_sensing_relative_path"])
        street_abs = dataset_root / street_rel
        remote_abs = dataset_root / remote_rel
        street_exists = street_abs.exists()
        remote_exists = remote_abs.exists()
        street_readable = _is_readable_image(street_abs) if verify_images else street_exists
        remote_readable = _is_readable_image(remote_abs) if verify_images else remote_exists
        if require_complete_pairs and not (street_exists and remote_exists and street_readable and remote_readable):
            continue
        record = {
            "sample_id": row["pair_id"],
            "sample_folder": str(_as_relative_path(row["sample_folder"])),
            "objectid": _to_int(row.get("objectid")),
            "attachment_id": _to_int(row.get("attachment_id")),
            "category": category,
            "binary_label": binary_label,
            "binary_name": to_binary_name(category, scheme=binary_scheme),
            "latitude": _to_float(row.get("latitude")),
            "longitude": _to_float(row.get("longitude")),
            "street_view_path": str(street_abs),
            "remote_sensing_path": str(remote_abs),
            "street_view_relative_path": str(street_rel),
            "remote_sensing_relative_path": str(remote_rel),
            "street_view_exists": street_exists,
            "remote_sensing_exists": remote_exists,
            "street_view_readable": street_readable,
            "remote_sensing_readable": remote_readable,
            "remote_tile_filename": row.get("remote_tile_filename", ""),
            "remote_tile_path": row.get("remote_tile_path", ""),
            "remote_match_status": row.get("remote_match_status", ""),
            "remote_pixel_x": _to_float(row.get("remote_pixel_x")),
            "remote_pixel_y": _to_float(row.get("remote_pixel_y")),
            "remote_distance_to_coverage_m": _to_float(row.get("remote_distance_to_coverage_m")),
            "group_id": str(_to_int(row.get("objectid")) or row["pair_id"]),
            "fire": row.get("fire", "Eaton_Fire"),
        }
        records.append(record)

    manifest = pd.DataFrame.from_records(records).sort_values(["objectid", "attachment_id", "sample_id"])
    manifest.to_csv(output_csv, index=False)
    save_json(summarize_manifest(manifest), output_csv.with_suffix(".summary.json"))
    return manifest


def build_eaton_manifest(
    attachments_index_csv: str | Path,
    output_csv: str | Path,
    binary_scheme: str = "operational",
) -> pd.DataFrame:
    attachments_index_csv = Path(attachments_index_csv)
    output_csv = Path(output_csv)
    ensure_dir(output_csv.parent)

    df = pd.read_csv(attachments_index_csv)
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        category = normalize_category(row["category"])
        records.append(
            {
                "fire": row.get("fire", "Eaton_Fire"),
                "objectid": _to_int(row.get("objectid")),
                "attachment_id": _to_int(row.get("attachment_id")),
                "category": category,
                "binary_label": to_binary_label(category, scheme=binary_scheme),
                "binary_name": to_binary_name(category, scheme=binary_scheme),
                "latitude": _to_float(row.get("lat")),
                "longitude": _to_float(row.get("lon")),
                "file": row.get("file", ""),
                "url": row.get("url", ""),
                "group_id": str(_to_int(row.get("objectid")) or row.get("attachment_id")),
            }
        )

    manifest = pd.DataFrame.from_records(records).sort_values(["objectid", "attachment_id"])
    manifest.to_csv(output_csv, index=False)
    save_json(summarize_manifest(manifest), output_csv.with_suffix(".summary.json"))
    return manifest
