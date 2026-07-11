from __future__ import annotations

"""Cache forward-only Milton sector embeddings from frozen CVIAN encoders.

The script never opens a CVIAN role manifest.  Its only CVIAN inputs are the
five already-attested base-model artifact directories.  Milton labels are
copied into the cache solely for later fixed-policy sensitivity scoring; they
are never passed to a model and no fitting, calibration, thresholding, or
policy selection occurs here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.data.panorama import (
    PanoramaSector,
    build_panorama_sectors,
    crop_equirectangular_sector,
    validate_equirectangular_size,
)
from crossview_conflict.factory import load_triage_from_checkpoint
from crossview_conflict.models.backbones import get_backbone_normalization
from crossview_conflict.utils.image import build_transform
from scripts.build_milton_active_view_transfer_manifest import (
    CLAIM_SCOPE,
    CONTRACT_FILENAME,
    CONTRACT_SCHEMA_VERSION,
    DEFAULT_EXPECTED_ROWS,
    DEFAULT_SOURCE_SHA256,
    OUTPUT_FILENAME,
)


SCHEMA_VERSION = "milton-active-view-forward-cache-v1"
SUMMARY_SCHEMA_VERSION = "milton-active-view-forward-cache-summary-v1"
BASE_ATTESTATION_SCHEMA_VERSION = "cvian-sequence-role-isolated-base-v1"
SEEDS = (42, 123, 456, 789, 1011)
NUM_SECTORS = 8
HORIZONTAL_FOV_DEG = 90.0
VERTICAL_FOV_DEG = 90.0
IMAGE_SIZE = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache fixed CVIAN-encoder features for Milton transfer sensitivity. "
            "This command performs forward passes only."
        )
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/splits/milton_active_view_transfer_v1",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="outputs/cvian_sequence_active_v2/base_encoders",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/cvian_sequence_active_v2/milton_transfer_cache",
    )
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
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


def _unicode_array(values: Iterable[object]) -> np.ndarray:
    """Encode strings without any fixed U32/U64 truncation ceiling."""

    strings = [str(value) for value in values]
    width = max((len(value) for value in strings), default=1)
    return np.asarray(strings, dtype=f"<U{max(width, 1)}")


def _assert_fields(
    payload: dict[str, Any], expected: dict[str, Any], *, context: str
) -> None:
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{context} mismatch for {field!r}")


def _load_transfer_manifest(
    manifest_dir: Path, *, enforce_registered_snapshot: bool = True
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    contract_path = manifest_dir / CONTRACT_FILENAME
    if not contract_path.is_file():
        raise FileNotFoundError(contract_path)
    contract = _load_json(contract_path)
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError("Unsupported Milton transfer contract")
    expected_contract = {
        "claim_scope": CLAIM_SCOPE,
        "confirmatory_eligible": False,
        "operation_scope": "forward_only_transfer_sensitivity",
        "fitting_permitted": False,
        "calibration_permitted": False,
        "model_or_policy_selection_permitted": False,
    }
    _assert_fields(contract, expected_contract, context="Milton transfer contract")
    if enforce_registered_snapshot:
        if int(contract.get("expected_rows", -1)) != DEFAULT_EXPECTED_ROWS:
            raise ValueError("Milton registered transfer row count changed")
        if contract.get("source", {}).get("required_sha256") != DEFAULT_SOURCE_SHA256:
            raise ValueError("Milton registered source SHA-256 changed")

    manifest_info = contract.get("manifest", {})
    if manifest_info.get("filename") != OUTPUT_FILENAME:
        raise ValueError("Milton transfer manifest filename changed")
    manifest_path = manifest_dir / OUTPUT_FILENAME
    if not manifest_path.is_file() or _sha256(manifest_path) != manifest_info.get("sha256"):
        raise ValueError("Milton transfer manifest hash mismatch")
    source_info = contract.get("source", {})
    source_path = Path(str(source_info.get("path", "")))
    audit_path = Path(str(source_info.get("provenance_audit_path", "")))
    if not source_path.is_file() or _sha256(source_path) != source_info.get("sha256"):
        raise ValueError("Milton canonical source cohort hash mismatch")
    if not audit_path.is_file() or _sha256(audit_path) != source_info.get("provenance_audit_sha256"):
        raise ValueError("Milton canonical provenance-audit hash mismatch")

    frame = pd.read_csv(
        manifest_path,
        dtype={
            "sample_id": str,
            "canonical_sample_id": str,
            "pre_image_id": str,
            "post_image_id": str,
            "spatial_block_id": str,
            "dependency_group_id": str,
            "remote_content_group_id": str,
            "sequence_id": str,
        },
    ).reset_index(drop=True)
    required = {
        "sample_id",
        "label",
        "street_view_path",
        "remote_sensing_path",
        "post_panorama_sha256",
        "remote_content_sha256",
        "latitude",
        "longitude",
        "spatial_block_id",
        "dependency_group_id",
        "remote_content_group_id",
        "sequence_id",
        "sequence_metadata_available",
        "sequence_id_is_observed",
        "compass_angle_deg",
        "compass_metadata_available",
        "compass_available",
        "absolute_azimuth_available",
        "claim_scope",
        "transfer_operation",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Milton transfer manifest is missing columns: {missing}")
    if len(frame) != int(contract.get("rows", -1)):
        raise ValueError("Milton transfer manifest row count mismatch")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Milton transfer sample identifiers are invalid")
    if frame["sequence_id"].isna().any() or frame["sequence_id"].duplicated().any():
        raise ValueError("Missing Milton sequences collapsed into a shared identifier")
    if not frame["dependency_group_id"].equals(frame["remote_content_group_id"]):
        raise ValueError("Milton dependency groups must equal remote-content groups")
    if frame["compass_angle_deg"].notna().any():
        raise ValueError("Milton compass values must remain missing")
    for field in (
        "sequence_metadata_available",
        "sequence_id_is_observed",
        "compass_metadata_available",
        "compass_available",
        "absolute_azimuth_available",
    ):
        if not frame[field].astype(str).str.lower().isin({"false", "0"}).all():
            raise ValueError(f"Milton {field} must remain false")
    if not frame["claim_scope"].eq(CLAIM_SCOPE).all():
        raise ValueError("Milton transfer claim scope changed")
    if not frame["transfer_operation"].eq("forward_only_no_fit_calibration_or_selection").all():
        raise ValueError("Milton transfer operation scope changed")
    if set(frame["label"].astype(int)) != {0, 1, 2}:
        raise ValueError("Milton transfer manifest is not three-class complete")
    hashes = {
        "manifest_contract_sha256": _sha256(contract_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_cohort_sha256": _sha256(source_path),
        "source_provenance_audit_sha256": _sha256(audit_path),
    }
    return frame, contract, hashes


def _validate_media(frame: pd.DataFrame) -> dict[str, int]:
    verified_street: dict[Path, str] = {}
    verified_overhead: dict[Path, str] = {}
    for row in frame.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        street_path = Path(str(row["street_view_path"]))
        overhead_path = Path(str(row["remote_sensing_path"]))
        if not street_path.is_file() or not overhead_path.is_file():
            raise FileNotFoundError(f"Missing Milton transfer media for {sample_id}")
        if street_path not in verified_street:
            verified_street[street_path] = _sha256(street_path)
            with Image.open(street_path) as image:
                validate_equirectangular_size(image.size, aspect_tolerance=0.0)
                image.verify()
        if verified_street[street_path] != str(row["post_panorama_sha256"]):
            raise ValueError(f"Milton street media hash mismatch for {sample_id}")
        if overhead_path not in verified_overhead:
            verified_overhead[overhead_path] = _sha256(overhead_path)
            with Image.open(overhead_path) as image:
                image.verify()
        if verified_overhead[overhead_path] != str(row["remote_content_sha256"]):
            raise ValueError(f"Milton overhead media hash mismatch for {sample_id}")
    return {
        "unique_post_panorama_files_verified": len(verified_street),
        "unique_overhead_files_verified": len(verified_overhead),
    }


def _validate_attested_checkpoints(checkpoint_root: Path) -> dict[int, dict[str, Any]]:
    """Validate model artifacts without opening any CVIAN role CSV."""

    summary_path = checkpoint_root / "training_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = _load_json(summary_path)
    if summary.get("schema_version") != BASE_ATTESTATION_SCHEMA_VERSION:
        raise ValueError("Unsupported CVIAN base-training summary")
    runs = summary.get("runs")
    if not isinstance(runs, list):
        raise ValueError("CVIAN base-training summary has no run inventory")
    by_seed: dict[int, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict) or "seed" not in run:
            raise ValueError("Invalid CVIAN base-training run entry")
        seed = int(run["seed"])
        if seed in by_seed:
            raise ValueError(f"Duplicate CVIAN base-training seed {seed}")
        by_seed[seed] = run
    if tuple(sorted(by_seed)) != tuple(sorted(SEEDS)):
        raise ValueError("CVIAN base-training seed inventory changed")

    inventory: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        run_dir = checkpoint_root / f"crossview_seed{seed}"
        completion_path = run_dir / "training_complete.json"
        checkpoint_path = run_dir / "triage_best.pt"
        completion = _load_json(completion_path)
        if completion != by_seed[seed]:
            raise ValueError(f"CVIAN seed {seed} completion differs from training summary")
        expected = {
            "schema_version": BASE_ATTESTATION_SCHEMA_VERSION,
            "seed": seed,
            "mode": "crossview",
            "train_role": "base_fit",
            "validation_role": "validation",
            "image_size": IMAGE_SIZE,
            "test_evaluated": False,
        }
        _assert_fields(completion, expected, context=f"CVIAN seed {seed} completion")
        artifacts = {
            "checkpoint": (checkpoint_path, "checkpoint_sha256"),
            "history": (run_dir / "triage_history.json", "history_sha256"),
            "train_log": (run_dir / "train.log", "train_log_sha256"),
        }
        artifact_hashes: dict[str, str] = {}
        for name, (path, field) in artifacts.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = _sha256(path)
            if actual != completion.get(field):
                raise ValueError(f"CVIAN seed {seed} {name} hash mismatch")
            artifact_hashes[name] = actual
        inventory[seed] = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": artifact_hashes["checkpoint"],
            "completion_path": str(completion_path),
            "completion_sha256": _sha256(completion_path),
            "training_fingerprint_sha256": completion.get("fingerprint_sha256"),
            "history_sha256": artifact_hashes["history"],
            "train_log_sha256": artifact_hashes["train_log"],
            "training_summary_path": str(summary_path),
            "training_summary_sha256": _sha256(summary_path),
        }
    return inventory


class MiltonSectorBagDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        sectors: tuple[PanoramaSector, ...],
        *,
        street_backbone: str,
        overhead_backbone: str,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.sectors = sectors
        street_mean, street_std = get_backbone_normalization(street_backbone)
        overhead_mean, overhead_std = get_backbone_normalization(overhead_backbone)
        self.street_transform = build_transform(
            IMAGE_SIZE, domain="street", mean=street_mean, std=street_std
        )
        self.overhead_transform = build_transform(
            IMAGE_SIZE, domain="overhead", mean=overhead_mean, std=overhead_std
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(row["street_view_path"]) as source:
            panorama = source.convert("RGB")
        validate_equirectangular_size(panorama.size, aspect_tolerance=0.0)
        sectors = torch.stack(
            [
                self.street_transform(crop_equirectangular_sector(panorama, sector))
                for sector in self.sectors
            ]
        )
        panorama_tensor = self.street_transform(panorama)
        with Image.open(row["remote_sensing_path"]) as source:
            overhead = self.overhead_transform(source.convert("RGB"))
        return {
            "sample_id": str(row["sample_id"]),
            "spatial_block_id": str(row["spatial_block_id"]),
            "sequence_id": str(row["sequence_id"]),
            "dependency_group_id": str(row["dependency_group_id"]),
            "target": int(row["label"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "sectors": sectors,
            "panorama": panorama_tensor,
            "overhead": overhead,
        }


def _cache_forward(
    *,
    model: torch.nn.Module,
    dataset: MiltonSectorBagDataset,
    sectors: tuple[PanoramaSector, ...],
    output_path: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.startswith("cuda"),
    )
    sample_ids: list[str] = []
    spatial_blocks: list[str] = []
    sequence_ids: list[str] = []
    dependency_groups: list[str] = []
    targets: list[np.ndarray] = []
    latitudes: list[np.ndarray] = []
    longitudes: list[np.ndarray] = []
    street_embeddings: list[np.ndarray] = []
    overhead_embeddings: list[np.ndarray] = []
    sector_logits: list[np.ndarray] = []
    panorama_logits: list[np.ndarray] = []
    model.eval()
    use_amp = device.startswith("cuda")
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            sector_batch = batch["sectors"].to(device, non_blocking=True)
            overhead = batch["overhead"].to(device, non_blocking=True)
            panorama = batch["panorama"].to(device, non_blocking=True)
            actual_batch, sector_count = sector_batch.shape[:2]
            flat_sectors = sector_batch.reshape(
                actual_batch * sector_count, *sector_batch.shape[2:]
            )
            repeated_overhead = (
                overhead[:, None]
                .expand(actual_batch, sector_count, *overhead.shape[1:])
                .reshape(actual_batch * sector_count, *overhead.shape[1:])
            )
            with torch.autocast(
                device_type="cuda" if use_amp else "cpu",
                dtype=torch.float16 if use_amp else torch.bfloat16,
                enabled=use_amp,
            ):
                flat_logits, flat_street, flat_overhead = model(
                    street=flat_sectors,
                    overhead=repeated_overhead,
                    return_embeddings=True,
                )
                full_logits, _, _ = model(
                    street=panorama,
                    overhead=overhead,
                    return_embeddings=True,
                )
            if flat_street is None or flat_overhead is None:
                raise RuntimeError("CVIAN encoder did not return transfer embeddings")
            sector_logits.append(
                flat_logits.reshape(actual_batch, sector_count, -1)
                .float().cpu().numpy().astype(np.float16)
            )
            street_embeddings.append(
                flat_street.reshape(actual_batch, sector_count, -1)
                .float().cpu().numpy().astype(np.float16)
            )
            overhead_embeddings.append(
                flat_overhead.reshape(actual_batch, sector_count, -1)[:, 0]
                .float().cpu().numpy().astype(np.float16)
            )
            panorama_logits.append(full_logits.float().cpu().numpy().astype(np.float16))
            sample_ids.extend(map(str, batch["sample_id"]))
            spatial_blocks.extend(map(str, batch["spatial_block_id"]))
            sequence_ids.extend(map(str, batch["sequence_id"]))
            dependency_groups.extend(map(str, batch["dependency_group_id"]))
            targets.append(batch["target"].numpy().astype(np.int8))
            latitudes.append(batch["latitude"].numpy().astype(np.float64))
            longitudes.append(batch["longitude"].numpy().astype(np.float64))
            if batch_index % 25 == 0:
                print(f"cached {len(sample_ids)}/{len(dataset)}", flush=True)

    arrays = {
        "sample_id": _unicode_array(sample_ids),
        "spatial_block_id": _unicode_array(spatial_blocks),
        "sequence_id": _unicode_array(sequence_ids),
        "dependency_group_id": _unicode_array(dependency_groups),
        "target": np.concatenate(targets),
        "latitude": np.concatenate(latitudes),
        "longitude": np.concatenate(longitudes),
        "compass_angle_deg": np.full(len(dataset), np.nan, dtype=np.float32),
        "compass_available": np.zeros(len(dataset), dtype=bool),
        "sequence_metadata_available": np.zeros(len(dataset), dtype=bool),
        "sector_id": np.asarray([sector.sector_id for sector in sectors], dtype=np.int8),
        "relative_azimuth_deg": np.asarray(
            [sector.relative_azimuth_deg for sector in sectors], dtype=np.float32
        ),
        "street_embedding": np.concatenate(street_embeddings),
        "overhead_embedding": np.concatenate(overhead_embeddings),
        "sector_logits": np.concatenate(sector_logits),
        "panorama_logits": np.concatenate(panorama_logits),
    }
    if len(np.unique(arrays["sample_id"])) != len(dataset):
        raise ValueError("Cached Milton sample identifiers are not unique")
    if len(np.unique(arrays["sequence_id"])) != len(dataset):
        raise ValueError("Missing Milton sequence sentinels collapsed in cache")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(output_path)
    return {
        "rows": len(dataset),
        "sectors_per_row": len(sectors),
        "embedding_dim": int(arrays["street_embedding"].shape[-1]),
        "classes": int(arrays["sector_logits"].shape[-1]),
        "output": str(output_path),
        "sha256": _sha256(output_path),
        "unicode_widths": {
            field: int(arrays[field].dtype.itemsize // np.dtype("U1").itemsize)
            for field in (
                "sample_id",
                "spatial_block_id",
                "sequence_id",
                "dependency_group_id",
            )
        },
    }


def _validate_cache(
    path: Path,
    frame: pd.DataFrame,
    sectors: tuple[PanoramaSector, ...],
    summary: dict[str, Any],
) -> None:
    if not path.is_file() or _sha256(path) != summary.get("sha256"):
        raise ValueError(f"Milton cache hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "sample_id", "spatial_block_id", "sequence_id", "dependency_group_id",
            "target", "latitude", "longitude", "compass_angle_deg",
            "compass_available", "sequence_metadata_available", "sector_id",
            "relative_azimuth_deg", "street_embedding", "overhead_embedding",
            "sector_logits", "panorama_logits",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Milton cache is missing arrays: {missing}")
        for field in ("sample_id", "spatial_block_id", "sequence_id", "dependency_group_id"):
            if not np.array_equal(payload[field].astype(str), frame[field].astype(str).to_numpy()):
                raise ValueError(f"Milton cache {field} order mismatch")
        if not np.array_equal(payload["target"], frame["label"].astype(int).to_numpy()):
            raise ValueError("Milton cache target mismatch")
        for field in ("latitude", "longitude"):
            if not np.allclose(
                payload[field].astype(np.float64),
                frame[field].astype(float).to_numpy(),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"Milton cache {field} mismatch")
        if not np.isnan(payload["compass_angle_deg"]).all():
            raise ValueError("Milton cache unexpectedly contains compass angles")
        if payload["compass_available"].any() or payload["sequence_metadata_available"].any():
            raise ValueError("Milton cache unexpectedly marks missing metadata available")
        rows = len(frame)
        count = len(sectors)
        expected_sector_id = np.asarray(
            [sector.sector_id for sector in sectors], dtype=np.int8
        )
        expected_azimuth = np.asarray(
            [sector.relative_azimuth_deg for sector in sectors], dtype=np.float32
        )
        if not np.array_equal(payload["sector_id"], expected_sector_id):
            raise ValueError("Milton cache sector-id geometry mismatch")
        if not np.allclose(
            payload["relative_azimuth_deg"],
            expected_azimuth,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("Milton cache azimuth geometry mismatch")
        street = payload["street_embedding"]
        if street.ndim != 3 or street.shape[:2] != (rows, count):
            raise ValueError("Milton street-embedding cache shape mismatch")
        dim = street.shape[-1]
        shapes = {
            "overhead_embedding": (rows, dim),
            "sector_logits": (rows, count, 3),
            "panorama_logits": (rows, 3),
        }
        for field, shape in shapes.items():
            if payload[field].shape != shape or not np.isfinite(payload[field]).all():
                raise ValueError(f"Milton cache {field} shape/finite check failed")
    expected_summary = {
        "rows": len(frame),
        "sectors_per_row": len(sectors),
        "output": str(path),
        "sha256": _sha256(path),
    }
    _assert_fields(summary, expected_summary, context="Milton cache summary")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and num-workers non-negative")
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if seeds != SEEDS:
        raise ValueError(f"Milton transfer fixes the five registered seeds to {SEEDS}")
    manifest_dir = _resolve(args.manifest_dir)
    checkpoint_root = _resolve(args.checkpoint_root)
    output_root = _resolve(args.output_root)
    frame, contract, manifest_hashes = _load_transfer_manifest(manifest_dir)
    media_audit = _validate_media(frame)
    inventory = _validate_attested_checkpoints(checkpoint_root)
    sectors = build_panorama_sectors(
        num_sectors=NUM_SECTORS,
        horizontal_fov_deg=HORIZONTAL_FOV_DEG,
        vertical_fov_deg=VERTICAL_FOV_DEG,
    )
    source_paths = {
        "cache_script": Path(__file__).resolve(),
        "manifest_builder": REPO_ROOT / "scripts" / "build_milton_active_view_transfer_manifest.py",
        "panorama_geometry": REPO_ROOT / "crossview_conflict" / "data" / "panorama.py",
        "triage_model": REPO_ROOT / "crossview_conflict" / "models" / "triage.py",
        "model_factory": REPO_ROOT / "crossview_conflict" / "factory.py",
        "image_transforms": REPO_ROOT / "crossview_conflict" / "utils" / "image.py",
    }
    source_sha256 = {name: _sha256(path) for name, path in source_paths.items()}
    summaries: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = output_root / f"seed{seed}"
        cache_path = seed_dir / "milton_transfer.npz"
        metadata_path = seed_dir / "cache_metadata.json"
        attestation = inventory[seed]
        expected_provenance: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "claim_scope": CLAIM_SCOPE,
            "confirmatory_eligible": False,
            "operation_scope": "forward_only_transfer_sensitivity",
            "forward_only": True,
            "parameters_updated": False,
            "calibration_performed": False,
            "selector_or_policy_applied": False,
            "labels_used_as_model_inputs": False,
            "labels_cached_for_fixed_posthoc_scoring": True,
            "cvians_role_manifest_files_read": [],
            "cvian_prospective_test_read": False,
            "manifest_dir": str(manifest_dir),
            "manifest_hashes": manifest_hashes,
            "source_contract_claim_scope": contract["claim_scope"],
            "base_model_attestation": attestation,
            "num_sectors": NUM_SECTORS,
            "horizontal_fov_deg": HORIZONTAL_FOV_DEG,
            "vertical_fov_deg": VERTICAL_FOV_DEG,
            "image_size": IMAGE_SIZE,
            "sector_id": [sector.sector_id for sector in sectors],
            "relative_azimuth_deg": [float(sector.relative_azimuth_deg) for sector in sectors],
            "compass_metadata": "unavailable; NaN plus explicit false availability mask",
            "absolute_azimuth_available": False,
            "sequence_metadata": "unavailable; unique per-sample missing-data sentinels",
            "dependency_group_source": "remote_content_group_id",
            "media_audit": media_audit,
            "entrypoint_sha256": source_sha256,
        }
        if cache_path.exists() or metadata_path.exists():
            if args.overwrite:
                pass
            elif not cache_path.is_file() or not metadata_path.is_file():
                raise ValueError(f"Partial Milton cache for seed {seed}; use --overwrite")
            else:
                metadata = _load_json(metadata_path)
                _assert_fields(metadata, expected_provenance, context=f"Milton seed {seed} metadata")
                _validate_cache(cache_path, frame, sectors, metadata.get("cache", {}))
                summaries.append(metadata)
                continue

        checkpoint_path = Path(attestation["checkpoint_path"])
        model, checkpoint = load_triage_from_checkpoint(checkpoint_path, device=args.device)
        config = checkpoint.get("config", {})
        if config.get("mode") != "crossview" or int(config.get("num_classes", 0)) != 3:
            raise ValueError(f"Expected a three-class crossview checkpoint: {checkpoint_path}")
        if int(config.get("image_size", IMAGE_SIZE)) != IMAGE_SIZE:
            raise ValueError(f"CVIAN checkpoint image-size contract changed: {checkpoint_path}")
        street_backbone = str(config.get("street_backbone", "resnet18"))
        overhead_backbone = str(config.get("overhead_backbone", "resnet18"))
        dataset = MiltonSectorBagDataset(
            frame,
            sectors,
            street_backbone=street_backbone,
            overhead_backbone=overhead_backbone,
        )
        cache_summary = _cache_forward(
            model=model,
            dataset=dataset,
            sectors=sectors,
            output_path=cache_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        metadata = {
            **expected_provenance,
            "street_backbone": street_backbone,
            "overhead_backbone": overhead_backbone,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "cache": cache_summary,
        }
        seed_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        _validate_cache(cache_path, frame, sectors, cache_summary)
        summaries.append(metadata)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cache_summary.json").write_text(
        json.dumps(
            {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "claim_scope": CLAIM_SCOPE,
                "confirmatory_eligible": False,
                "forward_only": True,
                "runs": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
