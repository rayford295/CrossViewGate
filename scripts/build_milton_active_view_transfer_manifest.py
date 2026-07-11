from __future__ import annotations

"""Build the frozen, sensitivity-only Milton active-view transfer manifest.

This is intentionally a transfer manifest rather than a train/validation/test
split.  It retains the canonical Milton labels for later *post-hoc* scoring,
but it does not authorize fitting, calibration, model selection, or a
confirmatory claim on Milton.
"""

import argparse
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any
from uuid import uuid4

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_VERSION = "milton-active-view-transfer-manifest-v1"
CONTRACT_SCHEMA_VERSION = "milton-active-view-transfer-contract-v1"
SOURCE_SCHEMA_VERSION = "milton-canonical-sensitivity-v1"
CLAIM_SCOPE = "sensitivity_only_not_confirmatory"
DEFAULT_SOURCE_SHA256 = (
    "e401e9a8015aa08f3c5d6bbad5bc132a28de288734b0899e976b5e1e43fc4bf9"
)
DEFAULT_EXPECTED_ROWS = 1707
GRID_SIZE_DEG = Decimal("0.001")
OUTPUT_FILENAME = "milton_transfer_sensitivity.csv"
CONTRACT_FILENAME = "manifest_contract.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a frozen Milton manifest for forward-only active-view transfer "
            "sensitivity analysis."
        )
    )
    parser.add_argument(
        "--source-csv",
        default=(
            "data/splits/milton_canonical_sensitivity_v1/"
            "main_sensitivity_cohort.csv"
        ),
    )
    parser.add_argument(
        "--provenance-audit",
        default="data/splits/milton_canonical_sensitivity_v1/provenance_audit.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits/milton_active_view_transfer_v1",
    )
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _strict_false(series: pd.Series, *, field: str) -> None:
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"false", "0"}).all():
        raise ValueError(f"Milton {field} must be explicitly false for every row")


def _grid_index(value: object) -> int:
    numeric = Decimal(str(value))
    if not numeric.is_finite():
        raise ValueError(f"Non-finite coordinate: {value!r}")
    return int((numeric / GRID_SIZE_DEG).to_integral_value(rounding=ROUND_FLOOR))


def spatial_block_id(latitude: object, longitude: object) -> str:
    """Return a deterministic half-open 0.001-degree grid-cell identifier."""

    lat_index = _grid_index(latitude)
    lon_index = _grid_index(longitude)
    return f"milton_grid_0p001deg:{lat_index:+07d}:{lon_index:+08d}"


def missing_sequence_id(sample_id: str) -> str:
    """Create a unique sentinel without pretending an observed sequence exists."""

    return f"sequence_unavailable_for_sample:{sample_id}"


def _validate_source_contract(
    *,
    source_csv: Path,
    provenance_audit: Path,
    expected_source_sha256: str,
    expected_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any], str, str]:
    if not source_csv.is_file():
        raise FileNotFoundError(source_csv)
    if not provenance_audit.is_file():
        raise FileNotFoundError(provenance_audit)
    source_sha256 = _sha256(source_csv)
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            "Canonical Milton sensitivity cohort hash mismatch: "
            f"expected {expected_source_sha256}, got {source_sha256}"
        )
    audit_sha256 = _sha256(provenance_audit)
    audit = _load_json(provenance_audit)
    if audit.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("Unsupported canonical Milton provenance schema")
    if audit.get("claim_scope") != CLAIM_SCOPE or audit.get("confirmatory_eligible") is not False:
        raise ValueError("Canonical Milton source is not locked to sensitivity-only use")
    recorded_hash = audit.get("output_sha256", {}).get(source_csv.name)
    if recorded_hash != source_sha256:
        raise ValueError("Canonical Milton provenance does not attest the source CSV hash")
    if int(audit.get("counts", {}).get("main_rows", -1)) != expected_rows:
        raise ValueError("Canonical Milton provenance row count changed")
    metadata = audit.get("metadata_availability", {})
    if any(metadata.get(field) is not False for field in ("sequence_id", "compass_angle_deg", "absolute_azimuth")):
        raise ValueError("Canonical Milton metadata availability contract changed")

    frame = pd.read_csv(
        source_csv,
        dtype={
            "sample_id": str,
            "canonical_sample_id": str,
            "pre_image_id": str,
            "post_image_id": str,
            "post_panorama_group_id": str,
            "remote_content_group_id": str,
            "sequence_id": str,
        },
    ).reset_index(drop=True)
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} Milton rows, found {len(frame)}")
    required = {
        "sample_id",
        "canonical_sample_id",
        "label",
        "label_name",
        "latitude",
        "longitude",
        "street_view_path",
        "remote_sensing_path",
        "post_panorama_sha256",
        "remote_content_sha256",
        "post_panorama_group_id",
        "remote_content_group_id",
        "sequence_id",
        "compass_angle_deg",
        "sequence_metadata_available",
        "compass_metadata_available",
        "absolute_azimuth_available",
        "panorama_geometry_verified",
        "post_width",
        "post_height",
        "claim_scope",
        "cohort_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical Milton cohort is missing columns: {missing}")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Canonical Milton sample_id values must be non-missing and unique")
    if not frame["canonical_sample_id"].astype(str).equals(frame["sample_id"].astype(str)):
        raise ValueError("Canonical and transfer sample identifiers diverge")
    if set(frame["label"].astype(int)) != {0, 1, 2}:
        raise ValueError("Canonical Milton transfer cohort is not three-class complete")
    if not frame["claim_scope"].eq(CLAIM_SCOPE).all():
        raise ValueError("Milton row-level claim scope changed")
    if not frame["cohort_status"].eq("main_unanimous_post_representative").all():
        raise ValueError("Milton transfer source is not the canonical main cohort")
    if frame["post_panorama_sha256"].duplicated().any():
        raise ValueError("Milton transfer cohort must retain one row per post panorama")
    if frame["sequence_id"].notna().any() or frame["compass_angle_deg"].notna().any():
        raise ValueError("Milton sequence and compass values must remain unavailable")
    for field in (
        "sequence_metadata_available",
        "compass_metadata_available",
        "absolute_azimuth_available",
    ):
        _strict_false(frame[field], field=field)
    if not frame["panorama_geometry_verified"].astype(str).str.lower().isin({"true", "1"}).all():
        raise ValueError("Milton source does not attest panorama geometry")
    return frame, audit, source_sha256, audit_sha256


def _validate_media(frame: pd.DataFrame) -> dict[str, int]:
    panorama_hashes: dict[Path, str] = {}
    overhead_hashes: dict[Path, str] = {}
    for row in frame.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        street_path = Path(str(row["street_view_path"]))
        overhead_path = Path(str(row["remote_sensing_path"]))
        for path, kind in ((street_path, "street"), (overhead_path, "overhead")):
            if not path.is_file():
                raise FileNotFoundError(f"Missing Milton {kind} media for {sample_id}: {path}")

        expected_street_hash = str(row["post_panorama_sha256"])
        if street_path not in panorama_hashes:
            panorama_hashes[street_path] = _sha256(street_path)
        actual_street_hash = panorama_hashes[street_path]
        if actual_street_hash != expected_street_hash:
            raise ValueError(f"Milton post-panorama hash mismatch for {sample_id}")
        expected_overhead_hash = str(row["remote_content_sha256"])
        if overhead_path not in overhead_hashes:
            overhead_hashes[overhead_path] = _sha256(overhead_path)
        actual_overhead_hash = overhead_hashes[overhead_path]
        if actual_overhead_hash != expected_overhead_hash:
            raise ValueError(f"Milton overhead hash mismatch for {sample_id}")
        if str(row["post_panorama_group_id"]) != f"post_sha256:{expected_street_hash}":
            raise ValueError(f"Milton post content group mismatch for {sample_id}")
        if str(row["remote_content_group_id"]) != f"remote_sha256:{expected_overhead_hash}":
            raise ValueError(f"Milton overhead content group mismatch for {sample_id}")

        with Image.open(street_path) as image:
            width, height = image.size
            image.verify()
        if width != 2 * height:
            raise ValueError(
                f"Milton street image is not a 2:1 panorama: {street_path} "
                f"has {width}x{height}"
            )
        if width != int(row["post_width"]) or height != int(row["post_height"]):
            raise ValueError(f"Milton recorded panorama dimensions changed for {sample_id}")
        with Image.open(overhead_path) as image:
            image.verify()
    return {
        "unique_post_panorama_files_verified": len(panorama_hashes),
        "unique_overhead_files_verified": len(overhead_hashes),
    }


def _build_output_frame(source: pd.DataFrame) -> pd.DataFrame:
    output = source.copy()
    output["spatial_block_id"] = [
        spatial_block_id(latitude, longitude)
        for latitude, longitude in zip(output["latitude"], output["longitude"])
    ]
    output["dependency_group_id"] = output["remote_content_group_id"].astype(str)
    output["sequence_id"] = output["sample_id"].astype(str).map(missing_sequence_id)
    output["sequence_metadata_available"] = False
    output["sequence_id_is_observed"] = False
    output["sequence_id_semantics"] = "unique_missing-metadata_sentinel_not_observed_sequence"
    output["compass_angle_deg"] = math.nan
    output["compass_metadata_available"] = False
    output["compass_available"] = False
    output["absolute_azimuth_available"] = False
    output["spatial_block_grid_degrees"] = float(GRID_SIZE_DEG)
    output["transfer_operation"] = "forward_only_no_fit_calibration_or_selection"
    output = output.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if output["sequence_id"].duplicated().any():
        raise ValueError("Missing-sequence sentinels unexpectedly collapse samples")
    if not output["dependency_group_id"].equals(output["remote_content_group_id"].astype(str)):
        raise AssertionError("Milton dependency groups diverged from remote content groups")
    return output


def build_transfer_manifest(
    *,
    source_csv: Path,
    provenance_audit: Path,
    output_dir: Path,
    expected_source_sha256: str = DEFAULT_SOURCE_SHA256,
    expected_rows: int = DEFAULT_EXPECTED_ROWS,
) -> dict[str, Any]:
    source_csv = source_csv.resolve()
    provenance_audit = provenance_audit.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    source, audit, source_sha256, audit_sha256 = _validate_source_contract(
        source_csv=source_csv,
        provenance_audit=provenance_audit,
        expected_source_sha256=expected_source_sha256,
        expected_rows=expected_rows,
    )
    media_audit = _validate_media(source)
    output = _build_output_frame(source)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        manifest_path = temporary / OUTPUT_FILENAME
        output.to_csv(manifest_path, index=False)
        manifest_sha256 = _sha256(manifest_path)
        contract: dict[str, Any] = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "manifest_schema_version": SCHEMA_VERSION,
            "dataset": "Hurricane Milton canonical sensitivity cohort",
            "claim_scope": CLAIM_SCOPE,
            "confirmatory_eligible": False,
            "operation_scope": "forward_only_transfer_sensitivity",
            "fitting_permitted": False,
            "calibration_permitted": False,
            "model_or_policy_selection_permitted": False,
            "source": {
                "path": str(source_csv),
                "sha256": source_sha256,
                "required_sha256": expected_source_sha256,
                "provenance_audit_path": str(provenance_audit),
                "provenance_audit_sha256": audit_sha256,
                "provenance_schema_version": audit["schema_version"],
            },
            "expected_rows": expected_rows,
            "rows": len(output),
            "label_distribution": {
                str(int(label)): int(count)
                for label, count in output["label"].astype(int).value_counts().sort_index().items()
            },
            "manifest": {
                "filename": OUTPUT_FILENAME,
                "sha256": manifest_sha256,
            },
            "spatial_block": {
                "method": "floor(latitude/0.001),floor(longitude/0.001)",
                "grid_degrees": float(GRID_SIZE_DEG),
                "unique_blocks": int(output["spatial_block_id"].nunique()),
            },
            "dependency_group": {
                "source_field": "remote_content_group_id",
                "exact_copy": True,
                "unique_groups": int(output["dependency_group_id"].nunique()),
            },
            "missing_metadata": {
                "observed_sequence_available": False,
                "sequence_id_storage": "unique_per-sample_missing-metadata_sentinel",
                "observed_compass_available": False,
                "absolute_azimuth_available": False,
            },
            "media_audit": media_audit,
            "entrypoint_sha256": _sha256(Path(__file__).resolve()),
        }
        (temporary / CONTRACT_FILENAME).write_text(
            json.dumps(contract, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return contract


def main() -> None:
    args = parse_args()
    contract = build_transfer_manifest(
        source_csv=_resolve(args.source_csv),
        provenance_audit=_resolve(args.provenance_audit),
        output_dir=_resolve(args.output_dir),
    )
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
