from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Number
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.data.panorama import (
    absolute_sector_azimuth,
    build_panorama_sectors,
    crop_equirectangular_sector,
    sector_pixel_geometry,
    validate_equirectangular_size,
)


SPLIT_NAMES = ("train", "val", "test")
MANIFEST_SCHEMA_VERSION = "cvian-active-view-manifest-v2"


def _number_token(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def crop_version(
    num_sectors: int,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> str:
    """Return a geometry-specific crop identifier safe for paths and manifests."""
    build_panorama_sectors(
        num_sectors=num_sectors,
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=vertical_fov_deg,
    )
    overlap_deg = max(0.0, float(horizontal_fov_deg) - 360.0 / num_sectors)
    return (
        f"cvian-{num_sectors}sector-h{_number_token(horizontal_fov_deg)}-"
        f"v{_number_token(vertical_fov_deg)}-"
        f"overlap{_number_token(overlap_deg)}deg-v1"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the CVIAN multi-azimuth offline sequential-reveal benchmark. "
            "Each 2:1 panorama becomes eight overlapping 90-degree sectors by default."
        )
    )
    parser.add_argument("--split-dir", default="data/splits/ian_hurricane_original")
    parser.add_argument("--output-dir", default="data/active_view/cvian_spatial_v1")
    parser.add_argument("--num-sectors", type=int, default=8)
    parser.add_argument("--horizontal-fov-deg", type=float, default=90.0)
    parser.add_argument("--vertical-fov-deg", type=float, default=90.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--materialize-crops",
        action="store_true",
        help="Cache JPEG sector crops. The default manifest uses deterministic lazy crops.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f"Invalid CVIAN sample_id: {value!r}")
    return f"{int(text):06d}"


def _read_source_splits(split_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    seen_samples: set[str] = set()
    block_sets: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        path = split_dir / f"{split}.csv"
        frame = pd.read_csv(path, dtype={"sample_id": str, "mapillary_id": str})
        required = {
            "sample_id",
            "label",
            "street_view_path",
            "remote_sensing_path",
            "spatial_block_id",
            "compass_angle_deg",
            "is_pano",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        frame["sample_id"] = frame["sample_id"].map(_sample_id)
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample_id in {path}")
        overlap = seen_samples & set(frame["sample_id"])
        if overlap:
            raise ValueError(f"Cross-split sample leakage: {sorted(overlap)[:3]}")
        seen_samples.update(frame["sample_id"])
        block_sets[split] = set(frame["spatial_block_id"].astype(str))
        if not (pd.to_numeric(frame["is_pano"], errors="raise") == 1).all():
            raise ValueError(f"Every retained CVIAN street image must be panoramic: {path}")
        frames[split] = frame
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if block_sets[left] & block_sets[right]:
            raise ValueError(f"Spatial-block leakage between {left} and {right}")
    return frames


def build_sector_rows(
    source: pd.DataFrame,
    *,
    split: str,
    image_dir: Path,
    num_sectors: int,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    jpeg_quality: int,
    materialize_crops: bool,
    overwrite: bool,
) -> pd.DataFrame:
    sectors = build_panorama_sectors(
        num_sectors=num_sectors,
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=vertical_fov_deg,
    )
    version = crop_version(
        num_sectors,
        horizontal_fov_deg,
        vertical_fov_deg,
    )
    rows: list[dict[str, object]] = []
    # JPEG quality changes encoded pixels even when crop geometry is identical. Keep
    # materialized variants in separate directories so a no-overwrite rebuild cannot
    # silently reuse pixels produced with another quality setting.
    version_dir = image_dir / version / f"jpeg-q{int(jpeg_quality)}"
    if materialize_crops:
        version_dir.mkdir(parents=True, exist_ok=True)
    for source_row in source.to_dict(orient="records"):
        parent_sample_id = _sample_id(source_row["sample_id"])
        panorama_path = Path(str(source_row["street_view_path"]))
        if not panorama_path.is_file():
            raise FileNotFoundError(panorama_path)
        with Image.open(panorama_path) as panorama:
            source_media_format = str(panorama.format or "unknown")
            panorama = panorama.convert("RGB")
            validate_equirectangular_size(panorama.size)
            for sector in sectors:
                geometry = sector_pixel_geometry(panorama.size, sector)
                sector_sample_id = f"{parent_sample_id}_sector{sector.sector_id:02d}"
                output_path = version_dir / f"{sector_sample_id}.jpg"
                if materialize_crops and (overwrite or not output_path.is_file()):
                    crop = crop_equirectangular_sector(panorama, sector)
                    crop.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)
                sector_path = str(output_path.resolve()) if materialize_crops else ""
                sector_hash = _sha256(output_path) if materialize_crops else ""
                center_x = float(geometry["center_x_px"])
                crop_width = int(geometry["crop_width_px"])
                wraps_seam = (
                    center_x - crop_width / 2.0 < 0.0
                    or center_x + crop_width / 2.0 > panorama.width
                )
                crop_height = int(geometry["crop_height_px"])
                crop_y0 = max(0, (panorama.height - crop_height) // 2)
                row = dict(source_row)
                row.update(
                    {
                        "sample_id": sector_sample_id,
                        "parent_sample_id": parent_sample_id,
                        "split": split,
                        "sector_id": sector.sector_id,
                        "relative_azimuth_deg": sector.relative_azimuth_deg,
                        "absolute_azimuth_deg": absolute_sector_azimuth(
                            float(source_row["compass_angle_deg"]),
                            sector.relative_azimuth_deg,
                        ),
                        "horizontal_fov_deg": sector.horizontal_fov_deg,
                        "vertical_fov_deg": sector.vertical_fov_deg,
                        "sector_count": num_sectors,
                        "event_id": "hurricane_ian_cvian",
                        "split_role": split,
                        "initial_sector_id": 0,
                        "source_width": panorama.width,
                        "source_height": panorama.height,
                        "source_extension": panorama_path.suffix.lower(),
                        "source_media_format": source_media_format,
                        "compass_source": "official_CVIAN_GeoJSON",
                        "orientation_quality": "sequence_bearing_validated_with_outliers",
                        "projection": "equirectangular_window",
                        "wraps_seam": int(wraps_seam),
                        "crop_y0_px": crop_y0,
                        "crop_y1_px": crop_y0 + crop_height,
                        "crop_version": version,
                        "source_panorama_path": str(panorama_path.resolve()),
                        "sector_path": sector_path,
                        "sector_sha256": sector_hash,
                        "sector_materialized": int(materialize_crops),
                        # A lazy row must not expose the source panorama through the
                        # generic street-view field. Consumers must explicitly crop
                        # source_panorama_path with the recorded geometry.
                        "street_view_path": sector_path,
                        **geometry,
                    }
                )
                rows.append(row)
    output = pd.DataFrame(rows)
    expected = len(source) * num_sectors
    if len(output) != expected or output["sample_id"].duplicated().any():
        raise ValueError(f"Expected {expected} unique sector rows, got {len(output)}")
    counts = output.groupby("parent_sample_id")["sector_id"].nunique()
    if not (counts == num_sectors).all():
        raise ValueError("Every sample must contain every sector exactly once")
    return output


_DERIVED_MANIFEST_COLUMNS = {
    "sample_id",
    "street_view_path",
    "parent_sample_id",
    "split",
    "sector_id",
    "relative_azimuth_deg",
    "absolute_azimuth_deg",
    "horizontal_fov_deg",
    "vertical_fov_deg",
    "sector_count",
    "event_id",
    "split_role",
    "initial_sector_id",
    "source_width",
    "source_height",
    "source_extension",
    "source_media_format",
    "compass_source",
    "orientation_quality",
    "projection",
    "wraps_seam",
    "crop_y0_px",
    "crop_y1_px",
    "crop_version",
    "source_panorama_path",
    "sector_path",
    "sector_sha256",
    "sector_materialized",
    "center_x_px",
    "center_y_px",
    "crop_width_px",
    "crop_height_px",
}


def _is_missing(value: object) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _values_equal(actual: object, expected: object) -> bool:
    if _is_missing(actual) or _is_missing(expected):
        return _is_missing(actual) and _is_missing(expected)
    if isinstance(actual, Number) and isinstance(expected, Number):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)
    return str(actual) == str(expected)


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def _assert_value(actual: object, expected: object, *, context: str) -> None:
    if not _values_equal(actual, expected):
        raise ValueError(f"{context}: expected {expected!r}, got {actual!r}")


def _assert_float(actual: object, expected: float, *, context: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: expected numeric {expected!r}, got {actual!r}") from exc
    if not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{context}: expected {expected!r}, got {actual!r}")


def _verify_split_manifest(
    *,
    path: Path,
    split: str,
    source: pd.DataFrame,
    num_sectors: int,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    materialized: bool,
) -> pd.DataFrame:
    string_columns = {
        "sample_id": str,
        "parent_sample_id": str,
        "mapillary_id": str,
        "source_panorama_path": str,
        "sector_path": str,
        "sector_sha256": str,
        "street_view_path": str,
    }
    frame = pd.read_csv(path, dtype=string_columns, keep_default_na=False)
    required = _DERIVED_MANIFEST_COLUMNS | {
        column for column in source.columns if column not in {"sample_id", "street_view_path"}
    }
    _require_columns(frame, required, path)
    expected_rows = len(source) * num_sectors
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path}, got {len(frame)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sector sample_id in {path}")
    if set(frame["parent_sample_id"]) != set(source["sample_id"]):
        raise ValueError(f"Parent sample coverage mismatch in {path}")

    sectors = build_panorama_sectors(
        num_sectors=num_sectors,
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=vertical_fov_deg,
    )
    sector_by_id = {sector.sector_id: sector for sector in sectors}
    expected_sector_ids = list(range(num_sectors))
    expected_version = crop_version(
        num_sectors,
        horizontal_fov_deg,
        vertical_fov_deg,
    )
    source_by_id = source.set_index("sample_id", drop=False)

    for parent_sample_id, group in frame.groupby("parent_sample_id", sort=False):
        numeric_sector_ids = pd.to_numeric(group["sector_id"], errors="raise")
        if any(not float(value).is_integer() for value in numeric_sector_ids):
            raise ValueError(f"{path}: parent {parent_sample_id} has non-integer sector ids")
        sector_ids = numeric_sector_ids.astype(int)
        if sorted(sector_ids.tolist()) != expected_sector_ids:
            raise ValueError(
                f"{path}: parent {parent_sample_id} must contain sector ids "
                f"{expected_sector_ids}, got {sorted(sector_ids.tolist())}"
            )
        parent = source_by_id.loc[parent_sample_id]
        panorama_path = Path(str(parent["street_view_path"])).resolve()
        if not panorama_path.is_file():
            raise FileNotFoundError(panorama_path)
        with Image.open(panorama_path) as image:
            source_size = image.size
            source_media_format = str(image.format or "unknown")
            validate_equirectangular_size(source_size)

        for row in group.to_dict(orient="records"):
            sector_id = int(row["sector_id"])
            sector = sector_by_id[sector_id]
            context = f"{path}: parent {parent_sample_id}, sector {sector_id}"
            _assert_value(
                row["sample_id"],
                f"{parent_sample_id}_sector{sector_id:02d}",
                context=f"{context} sample_id",
            )
            for column in source.columns:
                if column in _DERIVED_MANIFEST_COLUMNS:
                    continue
                _assert_value(
                    row[column],
                    parent[column],
                    context=f"{context} inherited {column}",
                )
            _assert_value(row["split"], split, context=f"{context} split")
            _assert_value(row["split_role"], split, context=f"{context} split_role")
            _assert_value(
                row["sector_count"], num_sectors, context=f"{context} sector_count"
            )
            _assert_value(row["initial_sector_id"], 0, context=f"{context} initial sector")
            _assert_value(
                row["projection"],
                "equirectangular_window",
                context=f"{context} projection",
            )
            _assert_value(row["crop_version"], expected_version, context=f"{context} crop version")
            _assert_value(
                Path(str(row["source_panorama_path"])).resolve(),
                panorama_path,
                context=f"{context} source panorama",
            )
            _assert_value(row["source_width"], source_size[0], context=f"{context} source width")
            _assert_value(row["source_height"], source_size[1], context=f"{context} source height")
            _assert_value(
                row["source_extension"],
                panorama_path.suffix.lower(),
                context=f"{context} source extension",
            )
            _assert_value(
                row["source_media_format"],
                source_media_format,
                context=f"{context} source media format",
            )
            _assert_float(
                row["relative_azimuth_deg"],
                sector.relative_azimuth_deg,
                context=f"{context} relative azimuth",
            )
            _assert_float(
                row["absolute_azimuth_deg"],
                absolute_sector_azimuth(
                    float(parent["compass_angle_deg"]), sector.relative_azimuth_deg
                ),
                context=f"{context} absolute azimuth",
            )
            _assert_float(
                row["horizontal_fov_deg"],
                sector.horizontal_fov_deg,
                context=f"{context} horizontal FOV",
            )
            _assert_float(
                row["vertical_fov_deg"],
                sector.vertical_fov_deg,
                context=f"{context} vertical FOV",
            )
            geometry = sector_pixel_geometry(source_size, sector)
            for field in ("center_x_px", "center_y_px", "crop_width_px", "crop_height_px"):
                _assert_float(row[field], float(geometry[field]), context=f"{context} {field}")
            crop_width = int(geometry["crop_width_px"])
            crop_height = int(geometry["crop_height_px"])
            center_x = float(geometry["center_x_px"])
            wraps_seam = int(
                center_x - crop_width / 2.0 < 0.0
                or center_x + crop_width / 2.0 > source_size[0]
            )
            crop_y0 = max(0, (source_size[1] - crop_height) // 2)
            _assert_value(row["wraps_seam"], wraps_seam, context=f"{context} seam flag")
            _assert_value(row["crop_y0_px"], crop_y0, context=f"{context} crop y0")
            _assert_value(row["crop_y1_px"], crop_y0 + crop_height, context=f"{context} crop y1")
            _assert_value(
                row["sector_materialized"],
                int(materialized),
                context=f"{context} materialization flag",
            )

            sector_path_text = str(row["sector_path"]).strip()
            street_path_text = str(row["street_view_path"]).strip()
            sector_hash = str(row["sector_sha256"]).strip().lower()
            if not materialized:
                if sector_path_text or street_path_text or sector_hash:
                    raise ValueError(
                        f"{context}: lazy rows must leave sector_path, street_view_path, "
                        "and sector_sha256 empty"
                    )
                continue
            if not sector_path_text or not street_path_text or not sector_hash:
                raise ValueError(f"{context}: materialized sector provenance is incomplete")
            sector_path = Path(sector_path_text).resolve()
            if not sector_path.is_file():
                raise FileNotFoundError(sector_path)
            _assert_value(
                Path(street_path_text).resolve(),
                sector_path,
                context=f"{context} street sector path",
            )
            actual_hash = _sha256(sector_path)
            _assert_value(actual_hash, sector_hash, context=f"{context} sector SHA-256")
            with Image.open(sector_path) as crop_image:
                _assert_value(
                    crop_image.size,
                    (crop_width, crop_height),
                    context=f"{context} crop dimensions",
                )
    return frame


def _verify_episodes(
    episodes_path: Path,
    source_frames: dict[str, pd.DataFrame],
    *,
    num_sectors: int,
) -> None:
    episodes = pd.read_csv(
        episodes_path,
        dtype={"episode_id": str, "parent_sample_id": str},
        keep_default_na=False,
    )
    required = {
        "episode_id",
        "parent_sample_id",
        "split_role",
        "spatial_block_id",
        "sequence_id",
        "initial_state",
        "initial_sector_id",
        "candidate_sector_ids",
        "max_additional_reveals",
    }
    _require_columns(episodes, required, episodes_path)
    expected_episode_count = sum(len(frame) for frame in source_frames.values())
    if len(episodes) != expected_episode_count:
        raise ValueError(
            f"Expected {expected_episode_count} active-view episodes, got {len(episodes)}"
        )
    if episodes["episode_id"].duplicated().any() or episodes["parent_sample_id"].duplicated().any():
        raise ValueError("Active-view episodes must be unique by episode and parent sample")
    expected_parents = {
        _sample_id(value)
        for frame in source_frames.values()
        for value in frame["sample_id"]
    }
    if set(episodes["parent_sample_id"]) != expected_parents:
        raise ValueError("Active-view episode parent coverage is inconsistent")
    episode_by_parent = episodes.set_index("parent_sample_id", drop=False)
    expected_candidates = ",".join(str(index) for index in range(1, num_sectors))
    for split, source in source_frames.items():
        for parent in source.to_dict(orient="records"):
            parent_sample_id = _sample_id(parent["sample_id"])
            episode = episode_by_parent.loc[parent_sample_id]
            context = f"{episodes_path}: parent {parent_sample_id}"
            expected = {
                "episode_id": f"cvian_{parent_sample_id}",
                "split_role": split,
                "spatial_block_id": parent["spatial_block_id"],
                "sequence_id": parent.get("sequence_id", ""),
                "initial_state": "post_overhead+sector_0",
                "initial_sector_id": 0,
                "candidate_sector_ids": expected_candidates,
                "max_additional_reveals": num_sectors - 1,
            }
            for field, value in expected.items():
                _assert_value(episode[field], value, context=f"{context} {field}")


def verify_outputs(
    output_dir: Path,
    source_frames: dict[str, pd.DataFrame],
    *,
    num_sectors: int,
    materialized: bool,
    horizontal_fov_deg: float = 90.0,
    vertical_fov_deg: float = 90.0,
) -> dict[str, object]:
    split_counts: dict[str, int] = {}
    artifact_hashes: dict[str, str] = {}
    for split, source in source_frames.items():
        path = output_dir / f"{split}.csv"
        frame = _verify_split_manifest(
            path=path,
            split=split,
            source=source,
            num_sectors=num_sectors,
            horizontal_fov_deg=horizontal_fov_deg,
            vertical_fov_deg=vertical_fov_deg,
            materialized=materialized,
        )
        split_counts[split] = len(frame)
        artifact_hashes[path.name] = _sha256(path)
    episodes_path = output_dir / "active_view_episodes.csv"
    _verify_episodes(episodes_path, source_frames, num_sectors=num_sectors)
    artifact_hashes[episodes_path.name] = _sha256(episodes_path)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "num_sectors": num_sectors,
        "materialized_crops": materialized,
        "split_rows": split_counts,
        "manifest_sha256": artifact_hashes,
        "crop_version": crop_version(
            num_sectors, horizontal_fov_deg, vertical_fov_deg
        ),
    }


def _summary_payload(
    core: dict[str, object],
    *,
    split_dir: Path,
    output_dir: Path,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    jpeg_quality: int,
) -> dict[str, object]:
    summary = dict(core)
    summary.update(
        {
            "source_split_dir": str(split_dir),
            "source_manifest_sha256": {
                split: _sha256(split_dir / f"{split}.csv") for split in SPLIT_NAMES
            },
            "output_dir": str(output_dir),
            "horizontal_fov_deg": horizontal_fov_deg,
            "vertical_fov_deg": vertical_fov_deg,
            "jpeg_quality": jpeg_quality,
            "overlap_degrees": max(
                0.0, horizontal_fov_deg - 360.0 / int(core["num_sectors"])
            ),
            "initial_state": "post_overhead+sector_0",
            "claim_scope": "offline sequential evidence reveal",
            "independent_evaluation_unit": "parent panorama grouped by spatial_block_id",
            "orientation_note": (
                "relative yaw is exact for the crop grid and normalized to [-180, 180); "
                "absolute azimuth is derived from official compass metadata and must not "
                "be treated as camera-intrinsic truth"
            ),
        }
    )
    return summary


def _json_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return all(
            key in actual and _json_equal(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_equal(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(actual, Number) and isinstance(expected, Number):
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)
    return actual == expected


def verify_existing_summary(summary_path: Path, expected: dict[str, object]) -> dict[str, object]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing manifest summary: {summary_path}")
    existing = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise ValueError(f"Manifest summary must be a JSON object: {summary_path}")
    for field, expected_value in expected.items():
        if field not in existing:
            raise ValueError(f"Manifest summary is missing field {field!r}")
        if not _json_equal(existing[field], expected_value):
            raise ValueError(
                f"Manifest summary field {field!r} does not match current artifacts/parameters"
            )
    return existing


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    if not split_dir.is_absolute():
        split_dir = REPO_ROOT / split_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    split_dir = split_dir.resolve()
    output_dir = output_dir.resolve()
    source_frames = _read_source_splits(split_dir)
    if not args.verify_only:
        image_dir = output_dir / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        for split, source in source_frames.items():
            frame = build_sector_rows(
                source,
                split=split,
                image_dir=image_dir,
                num_sectors=args.num_sectors,
                horizontal_fov_deg=args.horizontal_fov_deg,
                vertical_fov_deg=args.vertical_fov_deg,
                jpeg_quality=args.jpeg_quality,
                materialize_crops=args.materialize_crops,
                overwrite=args.overwrite,
            )
            frame.to_csv(output_dir / f"{split}.csv", index=False)
            print(f"wrote {split}: {len(frame)} rows", flush=True)
        episodes = []
        for split, source in source_frames.items():
            for row in source.to_dict(orient="records"):
                episodes.append(
                    {
                        "episode_id": f"cvian_{_sample_id(row['sample_id'])}",
                        "parent_sample_id": _sample_id(row["sample_id"]),
                        "split_role": split,
                        "spatial_block_id": row["spatial_block_id"],
                        "sequence_id": row.get("sequence_id", ""),
                        "initial_state": "post_overhead+sector_0",
                        "initial_sector_id": 0,
                        "candidate_sector_ids": ",".join(
                            str(index) for index in range(1, args.num_sectors)
                        ),
                        "max_additional_reveals": args.num_sectors - 1,
                        "reveal_cost": 0.5,
                        "stop_cost": 0.0,
                        "defer_human_cost": 1.5,
                    }
                )
        pd.DataFrame(episodes).to_csv(output_dir / "active_view_episodes.csv", index=False)
    summary = verify_outputs(
        output_dir,
        source_frames,
        num_sectors=args.num_sectors,
        horizontal_fov_deg=args.horizontal_fov_deg,
        vertical_fov_deg=args.vertical_fov_deg,
        materialized=args.materialize_crops,
    )
    summary = _summary_payload(
        summary,
        split_dir=split_dir,
        output_dir=output_dir,
        horizontal_fov_deg=args.horizontal_fov_deg,
        vertical_fov_deg=args.vertical_fov_deg,
        jpeg_quality=args.jpeg_quality,
    )
    summary_path = output_dir / "manifest_summary.json"
    if args.verify_only:
        existing = verify_existing_summary(summary_path, summary)
        print(json.dumps(existing, indent=2))
        return
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
