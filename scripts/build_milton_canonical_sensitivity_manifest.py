from __future__ import annotations

"""Build a non-overwriting, sensitivity-only canonical Milton manifest.

This builder reconciles the legacy ``GenDisasterSVI`` package with the local
``Bi-temporal_hurricane`` canonical metadata by exact pre-image SHA-256.  It is
deliberately *not* a train/validation/test splitter and does not make Milton an
untouched confirmatory dataset.  The main output contains one representative
per byte-identical post panorama and excludes post panoramas whose canonical
human labels conflict.  Every excluded or missing row remains in an audit
artifact.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any
from uuid import uuid4

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_VERSION = "milton-canonical-sensitivity-v1"
PAIR_PATTERN = re.compile(
    r"^(?P<pre_image_id>\d+)_vs_(?P<post_image_id>\d+)\((?P<pair_index>\d+)\)$"
)
CANONICAL_LABELS = {
    "mild": (0, "mild_damage"),
    "moderate": (1, "moderate_damage"),
    "severe": (2, "severe_damage"),
}
LEGACY_LABELS = {
    "mild_damage": 0,
    "moderate_damage": 1,
    "severe_damage": 2,
}
OUTPUT_FILENAMES = (
    "canonical_all_mapped.csv",
    "main_sensitivity_cohort.csv",
    "conflicting_post_label_audit.csv",
    "unanimous_post_duplicate_audit.csv",
    "missing_crossview_rows.csv",
    "provenance_audit.json",
)


@dataclass(frozen=True)
class ExpectedCounts:
    canonical_rows: int = 2556
    mapped_rows: int = 2555
    missing_rows: int = 1
    label_mismatches: int = 182
    conflict_post_groups: int = 159
    main_rows: int = 1707


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a canonical-label Milton sensitivity manifest without "
            "overwriting the legacy Milton manifests."
        )
    )
    parser.add_argument(
        "--gendisaster-root",
        required=True,
        help="Local hurrican-milton-GenDisasterSVI directory.",
    )
    parser.add_argument(
        "--canonical-root",
        required=True,
        help="Local Bi-temporal_hurricane directory containing folder_0..2.",
    )
    parser.add_argument(
        "--location-csv",
        default="",
        help="Canonical Location.csv; defaults to <canonical-root>/Location.csv.",
    )
    parser.add_argument(
        "--source-csv",
        default="dataset_with_post_sat.csv",
        help="CSV inside the GenDisasterSVI root.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits/milton_canonical_sensitivity_v1",
        help=(
            "New output directory. It must not already exist; this builder has "
            "no overwrite mode."
        ),
    )
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-8)
    parser.add_argument("--expected-canonical-rows", type=int, default=2556)
    parser.add_argument("--expected-mapped-rows", type=int, default=2555)
    parser.add_argument("--expected-missing-rows", type=int, default=1)
    parser.add_argument("--expected-label-mismatches", type=int, default=182)
    parser.add_argument("--expected-conflict-post-groups", type=int, default=159)
    parser.add_argument("--expected-main-rows", type=int, default=1707)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_basename(value: object) -> str:
    return Path(str(value).replace("\\", "/")).name


def _parse_pair_id(sample_id: str) -> dict[str, object]:
    match = PAIR_PATTERN.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"Unrecognized canonical Milton pair id: {sample_id!r}")
    return {
        "pre_image_id": match.group("pre_image_id"),
        "post_image_id": match.group("post_image_id"),
        "pair_index": int(match.group("pair_index")),
    }


def _canonical_label(value: object) -> tuple[int, str, str]:
    normalized = str(value).strip().lower()
    if normalized not in CANONICAL_LABELS:
        raise ValueError(f"Unknown canonical human_damage_perception: {value!r}")
    label, label_name = CANONICAL_LABELS[normalized]
    return label, label_name, normalized


def _image_metadata(path: Path) -> tuple[int, int, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        width, height = image.size
        image_format = str(image.format or "unknown")
        image.verify()
    if width != 2 * height:
        raise ValueError(
            f"Milton street image is not a 2:1 panorama: {path} has {width}x{height}"
        )
    return width, height, image_format


def _find_pair_directories(canonical_root: Path) -> dict[str, Path]:
    pairs: dict[str, Path] = {}
    for folder in sorted(canonical_root.glob("folder_*")):
        if not folder.is_dir():
            continue
        for pair_dir in sorted(folder.iterdir()):
            if not pair_dir.is_dir():
                continue
            if pair_dir.name in pairs:
                raise ValueError(f"Duplicate canonical pair directory {pair_dir.name!r}")
            pairs[pair_dir.name] = pair_dir
    if not pairs:
        raise RuntimeError(f"No canonical pair directories found under {canonical_root}")
    return pairs


def _single_year_image(pair_dir: Path, year: int) -> Path:
    candidates = sorted(pair_dir.glob(f"*_{year}.png"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one *_{year}.png in {pair_dir}, found {len(candidates)}"
        )
    return candidates[0]


def _load_canonical_inventory(
    canonical_root: Path,
    location_csv: Path,
) -> pd.DataFrame:
    locations = pd.read_csv(location_csv)
    required = {"root", "lat", "lon", "human_damage_perception"}
    missing = sorted(required - set(locations.columns))
    if missing:
        raise ValueError(f"{location_csv} is missing canonical fields: {missing}")
    locations = locations.copy()
    locations["canonical_sample_id"] = locations["root"].map(_portable_basename)
    if locations["canonical_sample_id"].duplicated().any():
        duplicated = locations.loc[
            locations["canonical_sample_id"].duplicated(keep=False),
            "canonical_sample_id",
        ].tolist()
        raise ValueError(f"Duplicate canonical sample ids in Location.csv: {duplicated[:5]}")

    pair_dirs = _find_pair_directories(canonical_root)
    location_ids = set(locations["canonical_sample_id"])
    missing_dirs = sorted(location_ids - set(pair_dirs))
    extra_dirs = sorted(set(pair_dirs) - location_ids)
    if missing_dirs or extra_dirs:
        raise ValueError(
            "Canonical Location.csv/pair-directory mismatch: "
            f"missing_dirs={missing_dirs[:5]}, extra_dirs={extra_dirs[:5]}"
        )

    records: list[dict[str, object]] = []
    for row in locations.to_dict(orient="records"):
        sample_id = str(row["canonical_sample_id"])
        identifiers = _parse_pair_id(sample_id)
        pair_dir = pair_dirs[sample_id]
        pre_path = _single_year_image(pair_dir, 2023)
        post_path = _single_year_image(pair_dir, 2024)
        pre_width, pre_height, pre_format = _image_metadata(pre_path)
        post_width, post_height, post_format = _image_metadata(post_path)
        canonical_label, canonical_label_name, canonical_label_source_value = (
            _canonical_label(row["human_damage_perception"])
        )
        records.append(
            {
                "canonical_sample_id": sample_id,
                **identifiers,
                "canonical_label": canonical_label,
                "canonical_label_name": canonical_label_name,
                "canonical_label_source_value": canonical_label_source_value,
                "latitude": float(row["lat"]),
                "longitude": float(row["lon"]),
                "pre_street_view_path": str(pre_path.resolve()),
                "street_view_path": str(post_path.resolve()),
                "canonical_pre_sha256": _sha256(pre_path),
                "post_panorama_sha256": _sha256(post_path),
                "pre_width": pre_width,
                "pre_height": pre_height,
                "pre_format": pre_format,
                "post_width": post_width,
                "post_height": post_height,
                "post_format": post_format,
            }
        )
    inventory = pd.DataFrame.from_records(records)
    if inventory["canonical_pre_sha256"].duplicated().any():
        raise ValueError(
            "Canonical pre-image SHA-256 values are not unique; exact reconciliation "
            "would be ambiguous"
        )
    post_id_hash_counts = inventory.groupby("post_image_id")[
        "post_panorama_sha256"
    ].nunique()
    if (post_id_hash_counts != 1).any():
        raise ValueError("A canonical post_image_id maps to multiple panorama contents")
    # Do not require the reverse mapping.  The real snapshot contains one
    # byte-identical panorama whose numeric post id appears once with a missing
    # trailing digit.  Content SHA-256 is therefore the authoritative group key
    # while every source id remains preserved for audit.
    return inventory.sort_values("canonical_sample_id").reset_index(drop=True)


def _legacy_image_path(
    root: Path,
    damage_level: str,
    raw_path: object,
    view: str,
) -> Path:
    filename = _portable_basename(raw_path)
    if view == "pre":
        return root / damage_level / "pre" / filename
    if view == "post":
        return root / damage_level / "post" / filename
    if view == "overhead":
        return root / "post_sat" / filename
    raise ValueError(f"Unknown Milton view {view!r}")


def _reconcile_source(
    source_root: Path,
    source_csv: Path,
    canonical: pd.DataFrame,
    *,
    coordinate_tolerance: float,
) -> pd.DataFrame:
    source = pd.read_csv(source_csv)
    required = {
        "pair_id",
        "damage_level",
        "prompt",
        "pre_disaster_image_path",
        "post_disaster_image_path",
        "post_sat_image_path",
        "set",
        "lat",
        "lon",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"{source_csv} is missing legacy fields: {missing}")
    if source["pair_id"].duplicated().any():
        raise ValueError("Legacy Milton pair_id values are not unique")

    canonical_by_pre_hash = canonical.set_index("canonical_pre_sha256", drop=False)
    records: list[dict[str, object]] = []
    mapped_ids: set[str] = set()
    for row in source.to_dict(orient="records"):
        damage_level = str(row["damage_level"]).strip().lower()
        if damage_level not in LEGACY_LABELS:
            raise ValueError(f"Unknown legacy Milton damage_level: {damage_level!r}")
        pre_path = _legacy_image_path(
            source_root, damage_level, row["pre_disaster_image_path"], "pre"
        )
        post_path = _legacy_image_path(
            source_root, damage_level, row["post_disaster_image_path"], "post"
        )
        overhead_path = _legacy_image_path(
            source_root, damage_level, row["post_sat_image_path"], "overhead"
        )
        for path in (pre_path, post_path, overhead_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        pre_hash = _sha256(pre_path)
        if pre_hash not in canonical_by_pre_hash.index:
            raise ValueError(
                f"Legacy pair {row['pair_id']} pre-image SHA-256 has no canonical match"
            )
        canonical_row = canonical_by_pre_hash.loc[pre_hash]
        if isinstance(canonical_row, pd.DataFrame):
            raise ValueError("Ambiguous canonical pre-image SHA-256 reconciliation")
        sample_id = str(canonical_row["canonical_sample_id"])
        if sample_id in mapped_ids:
            raise ValueError(f"Multiple legacy rows map to canonical sample {sample_id}")
        mapped_ids.add(sample_id)

        packaged_post_hash = _sha256(post_path)
        canonical_post_hash = str(canonical_row["post_panorama_sha256"])
        if packaged_post_hash != canonical_post_hash:
            raise ValueError(
                f"Legacy pair {row['pair_id']} post panorama is not byte-identical "
                f"to canonical sample {sample_id}"
            )
        for coordinate, source_field, canonical_field in (
            ("latitude", row["lat"], canonical_row["latitude"]),
            ("longitude", row["lon"], canonical_row["longitude"]),
        ):
            if not math.isclose(
                float(source_field),
                float(canonical_field),
                rel_tol=0.0,
                abs_tol=coordinate_tolerance,
            ):
                raise ValueError(
                    f"Legacy pair {row['pair_id']} {coordinate} does not match "
                    f"canonical sample {sample_id}"
                )
        legacy_label = LEGACY_LABELS[damage_level]
        remote_hash = _sha256(overhead_path)
        records.append(
            {
                "sample_id": sample_id,
                "canonical_sample_id": sample_id,
                "legacy_pair_id": int(row["pair_id"]),
                "pre_image_id": str(canonical_row["pre_image_id"]),
                "post_image_id": str(canonical_row["post_image_id"]),
                "pair_index": int(canonical_row["pair_index"]),
                "label": int(canonical_row["canonical_label"]),
                "label_name": str(canonical_row["canonical_label_name"]),
                "canonical_label_source_value": str(
                    canonical_row["canonical_label_source_value"]
                ),
                "legacy_label": legacy_label,
                "legacy_label_name": damage_level,
                "label_agrees_with_legacy": bool(
                    legacy_label == int(canonical_row["canonical_label"])
                ),
                "legacy_prompt": str(row["prompt"]),
                "legacy_source_split": str(row["set"]).strip().lower(),
                "latitude": float(canonical_row["latitude"]),
                "longitude": float(canonical_row["longitude"]),
                "pre_street_view_path": str(canonical_row["pre_street_view_path"]),
                "street_view_path": str(canonical_row["street_view_path"]),
                "packaged_post_view_path": str(post_path.resolve()),
                "remote_sensing_path": str(overhead_path.resolve()),
                "canonical_pre_sha256": pre_hash,
                "post_panorama_sha256": canonical_post_hash,
                "remote_content_sha256": remote_hash,
                "post_panorama_group_id": f"post_sha256:{canonical_post_hash}",
                "remote_content_group_id": f"remote_sha256:{remote_hash}",
                "sequence_id": "",
                "compass_angle_deg": "",
                "sequence_metadata_available": False,
                "compass_metadata_available": False,
                "absolute_azimuth_available": False,
                "is_pano": 1,
                "panorama_geometry_verified": True,
                "post_width": int(canonical_row["post_width"]),
                "post_height": int(canonical_row["post_height"]),
                "post_format": str(canonical_row["post_format"]),
                "claim_scope": "sensitivity_only_not_confirmatory",
                "post_street_provenance": (
                    "observed_2024_local_canonical_pair_byte_identical"
                ),
                "overhead_provenance": (
                    "local_package_only_source_date_license_unresolved"
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("sample_id").reset_index(
        drop=True
    )


def _add_group_audit(
    mapped: pd.DataFrame,
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    label_sets = (
        canonical.groupby("post_panorama_sha256")["canonical_label_name"]
        .agg(lambda values: "|".join(sorted(set(map(str, values)))))
        .rename("canonical_post_label_set")
    )
    label_counts = (
        canonical.groupby("post_panorama_sha256")["canonical_label_name"]
        .nunique()
        .rename("canonical_post_label_count")
    )
    total_counts = (
        canonical.groupby("post_panorama_sha256")
        .size()
        .rename("canonical_post_group_rows_total")
    )
    post_id_sets = (
        canonical.groupby("post_panorama_sha256")["post_image_id"]
        .agg(lambda values: "|".join(sorted(set(map(str, values)))))
        .rename("canonical_post_image_id_set")
    )
    post_id_counts = (
        canonical.groupby("post_panorama_sha256")["post_image_id"]
        .nunique()
        .rename("canonical_post_image_id_count")
    )
    mapped_counts = (
        mapped.groupby("post_panorama_sha256")
        .size()
        .rename("mapped_post_group_rows")
    )
    remote_counts = (
        mapped.groupby("remote_content_sha256")
        .size()
        .rename("mapped_remote_content_rows")
    )
    enriched = mapped.join(
        pd.concat(
            [
                label_sets,
                label_counts,
                total_counts,
                post_id_sets,
                post_id_counts,
                mapped_counts,
            ],
            axis=1,
        ),
        on="post_panorama_sha256",
    ).join(remote_counts, on="remote_content_sha256")
    enriched["post_label_conflict"] = enriched["canonical_post_label_count"] > 1
    remote_per_post = enriched.groupby("post_panorama_sha256")[
        "remote_content_sha256"
    ].nunique()
    if (remote_per_post != 1).any():
        raise ValueError("A post panorama maps to multiple overhead contents")
    return enriched


def _build_cohorts(
    mapped: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows = mapped.sort_values("sample_id").reset_index(drop=True).copy()
    conflicting = all_rows[all_rows["post_label_conflict"]].copy()
    conflicting["cohort_status"] = "excluded_conflicting_canonical_post_labels"
    conflicting["exclusion_reason"] = (
        "canonical_human_labels_conflict_for_byte_identical_post_panorama"
    )

    unanimous = all_rows[~all_rows["post_label_conflict"]].copy()
    representatives = (
        unanimous.sort_values("sample_id")
        .drop_duplicates("post_panorama_sha256", keep="first")
        .copy()
    )
    representatives["cohort_status"] = "main_unanimous_post_representative"
    representatives["selection_rule"] = (
        "smallest_canonical_sample_id_per_unanimous_post_sha256"
    )
    representative_ids = set(representatives["sample_id"])
    duplicates = unanimous[~unanimous["sample_id"].isin(representative_ids)].copy()
    duplicates["cohort_status"] = "excluded_unanimous_post_duplicate"
    duplicates["exclusion_reason"] = (
        "byte_identical_post_panorama_already_has_deterministic_representative"
    )

    status = pd.Series("", index=all_rows.index, dtype=object)
    status.loc[all_rows["post_label_conflict"]] = (
        "excluded_conflicting_canonical_post_labels"
    )
    status.loc[all_rows["sample_id"].isin(representative_ids)] = (
        "main_unanimous_post_representative"
    )
    status.loc[
        (~all_rows["post_label_conflict"])
        & (~all_rows["sample_id"].isin(representative_ids))
    ] = "excluded_unanimous_post_duplicate"
    all_rows["cohort_status"] = status
    all_rows["main_sensitivity_cohort"] = all_rows["sample_id"].isin(
        representative_ids
    )
    return (
        all_rows,
        representatives.reset_index(drop=True),
        conflicting.reset_index(drop=True),
        duplicates.reset_index(drop=True),
    )


def _missing_rows(canonical: pd.DataFrame, mapped: pd.DataFrame) -> pd.DataFrame:
    missing = canonical[
        ~canonical["canonical_sample_id"].isin(set(mapped["canonical_sample_id"]))
    ].copy()
    missing = missing.rename(
        columns={
            "canonical_label": "label",
            "canonical_label_name": "label_name",
            "canonical_sample_id": "sample_id",
        }
    )
    missing["missing_required_view"] = "post_overhead"
    missing["missing_reason"] = "no_legacy_crossview_row_or_post_sat_mapping"
    missing["claim_scope"] = "sensitivity_only_not_confirmatory"
    keep = [
        "sample_id",
        "pre_image_id",
        "post_image_id",
        "pair_index",
        "label",
        "label_name",
        "canonical_label_source_value",
        "latitude",
        "longitude",
        "pre_street_view_path",
        "street_view_path",
        "canonical_pre_sha256",
        "post_panorama_sha256",
        "missing_required_view",
        "missing_reason",
        "claim_scope",
    ]
    return missing[keep].sort_values("sample_id").reset_index(drop=True)


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts(dropna=False).sort_index().items()
    }


def _validate_expected_counts(actual: dict[str, int], expected: ExpectedCounts) -> None:
    for key, expected_value in asdict(expected).items():
        if expected_value < 0:
            continue
        if actual[key] != expected_value:
            raise ValueError(
                f"Milton canonical contract mismatch for {key}: "
                f"expected {expected_value}, observed {actual[key]}"
            )


def _write_atomic_output(
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    audit: dict[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Milton output directory: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    temporary.mkdir()
    try:
        output_hashes: dict[str, str] = {}
        for filename, frame in frames.items():
            path = temporary / filename
            frame.to_csv(path, index=False)
            output_hashes[filename] = _sha256(path)
        audit["output_sha256"] = output_hashes
        (temporary / "provenance_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_canonical_sensitivity_manifest(
    *,
    gendisaster_root: Path,
    canonical_root: Path,
    location_csv: Path,
    source_csv: Path,
    output_dir: Path,
    expected_counts: ExpectedCounts = ExpectedCounts(),
    coordinate_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Reconcile Milton and atomically write a sensitivity-only artifact set."""

    gendisaster_root = gendisaster_root.resolve()
    canonical_root = canonical_root.resolve()
    location_csv = location_csv.resolve()
    source_csv = source_csv.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Milton output directory: {output_dir}"
        )
    if coordinate_tolerance < 0:
        raise ValueError("coordinate_tolerance must be non-negative")

    canonical = _load_canonical_inventory(canonical_root, location_csv)
    mapped = _reconcile_source(
        gendisaster_root,
        source_csv,
        canonical,
        coordinate_tolerance=coordinate_tolerance,
    )
    mapped = _add_group_audit(mapped, canonical)
    all_rows, main, conflicts, duplicates = _build_cohorts(mapped)
    missing = _missing_rows(canonical, mapped)
    label_mismatches = int((~mapped["label_agrees_with_legacy"]).sum())
    conflict_post_groups = int(
        conflicts["post_panorama_sha256"].nunique()
    )
    actual_counts = {
        "canonical_rows": int(len(canonical)),
        "mapped_rows": int(len(mapped)),
        "missing_rows": int(len(missing)),
        "label_mismatches": label_mismatches,
        "conflict_post_groups": conflict_post_groups,
        "main_rows": int(len(main)),
    }
    _validate_expected_counts(actual_counts, expected_counts)

    canonical_post_groups = int(canonical["post_panorama_sha256"].nunique())
    mapped_post_groups = int(mapped["post_panorama_sha256"].nunique())
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Hurricane Milton / local bi-temporal Horseshoe Beach collection",
        "claim_scope": "sensitivity_only_not_confirmatory",
        "confirmatory_eligible": False,
        "main_cohort_label": "canonical_unanimous_unique_post_sensitivity_only",
        "confirmatory_blockers": [
            "Milton data and FOV results were used in prior CrossViewGate hypothesis development.",
            "No sequence_id metadata is available in the reconciled local source.",
            "No compass angle or absolute panorama azimuth metadata is available.",
            "The canonical collection contains conflicting human labels for some byte-identical post panoramas.",
            "One canonical sample has no legacy cross-view row or post-overhead mapping.",
            "The post-overhead source, acquisition date, and redistribution license remain unresolved in the local package.",
            "A single held-out disaster event cannot support population-level inference across events.",
        ],
        "metadata_availability": {
            "sequence_id": False,
            "compass_angle_deg": False,
            "absolute_azimuth": False,
            "relative_panorama_sector_geometry": True,
        },
        "provenance": {
            "post_street": (
                "Every mapped packaged post panorama is byte-identical to its "
                "local canonical 2024 observed-image counterpart. Upstream capture "
                "platform and redistribution rights are not established here."
            ),
            "post_overhead": (
                "Retained from dataset_with_post_sat.csv; source, acquisition date, "
                "and license are not established by the local files."
            ),
            "canonical_label": (
                "Location.csv human_damage_perception; legacy damage_level is "
                "preserved only for disagreement audit."
            ),
        },
        "inputs": {
            "gendisaster_root": str(gendisaster_root),
            "canonical_root": str(canonical_root),
            "location_csv": str(location_csv),
            "location_csv_sha256": _sha256(location_csv),
            "source_csv": str(source_csv),
            "source_csv_sha256": _sha256(source_csv),
        },
        "expected_counts": asdict(expected_counts),
        "counts": {
            **actual_counts,
            "canonical_post_panorama_groups": canonical_post_groups,
            "mapped_post_panorama_groups": mapped_post_groups,
            "mapped_remote_content_groups": int(
                mapped["remote_content_sha256"].nunique()
            ),
            "conflict_rows": int(len(conflicts)),
            "unanimous_duplicate_rows_excluded": int(len(duplicates)),
            "main_post_panorama_groups": int(
                main["post_panorama_sha256"].nunique()
            ),
            "main_remote_content_groups": int(
                main["remote_content_sha256"].nunique()
            ),
        },
        "label_distributions": {
            "canonical_all": _value_counts(canonical, "canonical_label_name"),
            "canonical_mapped": _value_counts(mapped, "label_name"),
            "legacy_mapped": _value_counts(mapped, "legacy_label_name"),
            "main_sensitivity_cohort": _value_counts(main, "label_name"),
        },
        "label_disagreement": {
            "rows": label_mismatches,
            "rate": float(label_mismatches / max(len(mapped), 1)),
            "canonical_by_legacy": {
                str(canonical_name): {
                    str(legacy_name): int(value)
                    for legacy_name, value in row.items()
                }
                for canonical_name, row in pd.crosstab(
                    mapped["label_name"], mapped["legacy_label_name"]
                ).to_dict(orient="index").items()
            },
        },
        "cohort_policy": {
            "main": (
                "Exclude every post panorama with more than one canonical human "
                "label, then choose the lexicographically smallest canonical "
                "sample_id per remaining post SHA-256."
            ),
            "outcome_informed_curation": True,
            "interpretation": (
                "The label-consistency exclusion makes this a sensitivity cohort, "
                "not a confirmatory test. Conflicts and duplicates are never dropped "
                "silently and remain in separate audit CSVs."
            ),
        },
        "output_files": list(OUTPUT_FILENAMES),
    }
    frames = {
        "canonical_all_mapped.csv": all_rows,
        "main_sensitivity_cohort.csv": main,
        "conflicting_post_label_audit.csv": conflicts,
        "unanimous_post_duplicate_audit.csv": duplicates,
        "missing_crossview_rows.csv": missing,
    }
    _write_atomic_output(output_dir, frames, audit)
    return audit


def main() -> None:
    args = parse_args()
    gendisaster_root = Path(args.gendisaster_root)
    canonical_root = Path(args.canonical_root)
    location_csv = (
        Path(args.location_csv)
        if args.location_csv
        else canonical_root / "Location.csv"
    )
    source_csv = gendisaster_root / args.source_csv
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    audit = build_canonical_sensitivity_manifest(
        gendisaster_root=gendisaster_root,
        canonical_root=canonical_root,
        location_csv=location_csv,
        source_csv=source_csv,
        output_dir=output_dir,
        expected_counts=ExpectedCounts(
            canonical_rows=args.expected_canonical_rows,
            mapped_rows=args.expected_mapped_rows,
            missing_rows=args.expected_missing_rows,
            label_mismatches=args.expected_label_mismatches,
            conflict_post_groups=args.expected_conflict_post_groups,
            main_rows=args.expected_main_rows,
        ),
        coordinate_tolerance=args.coordinate_tolerance,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"Wrote sensitivity-only Milton artifacts to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
