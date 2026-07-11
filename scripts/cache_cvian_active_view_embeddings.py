from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

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


SPLIT_NAMES = ("train", "val", "test")
EMBEDDING_SCHEMA_VERSION = "cvian-active-view-cache-v3"
EMBEDDING_SUMMARY_SCHEMA_VERSION = "cvian-active-view-cache-summary-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache frozen CVIAN panorama-sector embeddings for the offline "
            "sequential-reveal benchmark. Hidden sectors are cached for the "
            "environment but never exposed to an online policy observation."
        )
    )
    parser.add_argument("--split-dir", default="data/splits/ian_hurricane_original")
    parser.add_argument(
        "--checkpoint-root",
        default="outputs/multiseed_ian_spatial_v1/ian_original",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/analysis/active_view_cvian_spatial_v1/cache",
    )
    parser.add_argument("--seeds", default="42,123,456,789,1011")
    parser.add_argument(
        "--splits",
        default=",".join(SPLIT_NAMES),
        help="Comma-separated role CSV stems to cache (for example base_fit,selector_fit,validation,prospective_test).",
    )
    parser.add_argument("--num-sectors", type=int, default=8)
    parser.add_argument("--horizontal-fov-deg", type=float, default=90.0)
    parser.add_argument("--vertical-fov-deg", type=float, default=90.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--require-training-attestation",
        action="store_true",
        help="Require each checkpoint directory to contain a matching role-isolated training_complete.json.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_frames(
    split_dir: Path,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    required = {
        "sample_id",
        "label",
        "street_view_path",
        "remote_sensing_path",
        "spatial_block_id",
        "sequence_id",
        "latitude",
        "longitude",
        "compass_angle_deg",
        "compass_angle_deg",
    }
    for split in split_names:
        path = split_dir / f"{split}.csv"
        frame = pd.read_csv(
            path,
            dtype={"sample_id": str, "mapillary_id": str},
        ).reset_index(drop=True)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample_id in {path}")
        frames[split] = frame
        hashes[split] = _sha256(path)
    return frames, hashes


def _assert_provenance(
    metadata: dict[str, Any],
    expected: dict[str, Any],
    *,
    context: str,
) -> None:
    for field, expected_value in expected.items():
        if field not in metadata or metadata[field] != expected_value:
            raise ValueError(
                f"{context} provenance mismatch for {field!r}; "
                "rerun with --overwrite"
            )


def _validate_training_attestation(
    checkpoint_path: Path,
    *,
    seed: int,
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    path = checkpoint_path.parent / "training_complete.json"
    if not path.is_file():
        raise ValueError(f"Missing role-isolated training attestation: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid training attestation: {path}")
    if payload.get("schema_version") != "cvian-sequence-role-isolated-base-v1":
        raise ValueError(f"Unsupported training attestation schema: {path}")
    if int(payload.get("seed", -1)) != seed or payload.get("test_evaluated") is not False:
        raise ValueError(f"Training attestation seed/test boundary mismatch: {path}")
    if payload.get("role_sha256") != source_hashes:
        raise ValueError(f"Training attestation role hashes mismatch: {path}")
    if payload.get("checkpoint_sha256") != _sha256(checkpoint_path):
        raise ValueError(f"Training attestation checkpoint hash mismatch: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "fingerprint_sha256": payload.get("fingerprint_sha256"),
    }


def _validate_embedding_npz(
    path: Path,
    frame: pd.DataFrame,
    sectors: tuple[PanoramaSector, ...],
    *,
    num_classes: int,
    summary: dict[str, Any],
) -> None:
    actual_hash = _sha256(path)
    if summary.get("sha256") != actual_hash:
        raise ValueError(f"Embedding cache hash mismatch for {path}; rerun with --overwrite")
    required = {
        "sample_id",
        "spatial_block_id",
        "sequence_id",
        "target",
        "latitude",
        "longitude",
        "compass_angle_deg",
        "sector_id",
        "relative_azimuth_deg",
        "street_embedding",
        "overhead_embedding",
        "sector_logits",
        "panorama_logits",
    }
    with np.load(path, allow_pickle=False) as arrays:
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"{path} is missing arrays {missing}; rerun with --overwrite")
        values = {name: arrays[name] for name in required}
    expected_sample_ids = frame["sample_id"].astype(str).to_numpy()
    if not np.array_equal(values["sample_id"].astype(str), expected_sample_ids):
        raise ValueError(
            f"Embedding cache sample order mismatch for {path}; rerun with --overwrite"
        )
    if not np.array_equal(
        values["spatial_block_id"].astype(str),
        frame["spatial_block_id"].astype(str).to_numpy(),
    ):
        raise ValueError(
            f"Embedding cache block order mismatch for {path}; rerun with --overwrite"
        )
    if not np.array_equal(
        values["sequence_id"].astype(str),
        frame["sequence_id"].astype(str).to_numpy(),
    ):
        raise ValueError(
            f"Embedding cache sequence order mismatch for {path}; rerun with --overwrite"
        )
    if not np.array_equal(values["target"], frame["label"].astype(int).to_numpy()):
        raise ValueError(f"Embedding cache targets mismatch for {path}; rerun with --overwrite")
    for field in ("latitude", "longitude", "compass_angle_deg"):
        if not np.allclose(
            values[field].astype(np.float64),
            frame[field].astype(float).to_numpy(),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                f"Embedding cache {field} mismatch for {path}; rerun with --overwrite"
            )
    sector_ids = np.asarray([sector.sector_id for sector in sectors], dtype=np.int8)
    relative_azimuth = np.asarray(
        [sector.relative_azimuth_deg for sector in sectors], dtype=np.float32
    )
    if not np.array_equal(values["sector_id"], sector_ids) or not np.allclose(
        values["relative_azimuth_deg"], relative_azimuth, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            f"Embedding cache sector geometry mismatch for {path}; rerun with --overwrite"
        )

    row_count = len(frame)
    sector_count = len(sectors)
    street = values["street_embedding"]
    overhead = values["overhead_embedding"]
    sector_logits = values["sector_logits"]
    panorama_logits = values["panorama_logits"]
    if street.ndim != 3 or street.shape[:2] != (row_count, sector_count):
        raise ValueError(
            f"Embedding cache street shape mismatch for {path}; rerun with --overwrite"
        )
    embedding_dim = int(street.shape[-1])
    expected_shapes = {
        "overhead_embedding": (row_count, embedding_dim),
        "sector_logits": (row_count, sector_count, num_classes),
        "panorama_logits": (row_count, num_classes),
    }
    for field, expected_shape in expected_shapes.items():
        if values[field].shape != expected_shape:
            raise ValueError(
                f"Embedding cache {field} shape mismatch for {path}; rerun with --overwrite"
            )
    for field in (
        "street_embedding",
        "overhead_embedding",
        "sector_logits",
        "panorama_logits",
    ):
        if not np.isfinite(values[field]).all():
            raise ValueError(f"Embedding cache {field} contains non-finite values: {path}")
    expected_summary = {
        "rows": row_count,
        "sectors_per_row": sector_count,
        "embedding_dim": embedding_dim,
        "classes": num_classes,
        "output": str(path),
        "sha256": actual_hash,
    }
    _assert_provenance(summary, expected_summary, context=f"embedding split {path.stem}")


def validate_reusable_embedding_cache(
    *,
    metadata_path: Path,
    seed_dir: Path,
    frames: dict[str, pd.DataFrame],
    sectors: tuple[PanoramaSector, ...],
    num_classes: int,
    expected_provenance: dict[str, Any],
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> dict[str, Any]:
    if not metadata_path.is_file():
        raise ValueError(
            f"Embedding cache exists without provenance metadata {metadata_path}; "
            "rerun with --overwrite"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid embedding cache metadata: {metadata_path}")
    _assert_provenance(metadata, expected_provenance, context=f"embedding cache {seed_dir.name}")
    summaries = metadata.get("splits")
    if not isinstance(summaries, dict) or set(summaries) != set(split_names):
        raise ValueError("Embedding cache split metadata is incomplete; rerun with --overwrite")
    for split in split_names:
        path = seed_dir / f"{split}.npz"
        if not path.is_file():
            raise ValueError(f"Missing embedding cache {path}; rerun with --overwrite")
        if not isinstance(summaries[split], dict):
            raise ValueError(f"Invalid embedding metadata for {split}; rerun with --overwrite")
        _validate_embedding_npz(
            path,
            frames[split],
            sectors,
            num_classes=num_classes,
            summary=summaries[split],
        )
    return metadata


class CVIANSectorBagDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        sectors: tuple[PanoramaSector, ...],
        *,
        image_size: int,
        street_backbone: str,
        overhead_backbone: str,
    ) -> None:
        self.frame = pd.read_csv(
            split_csv,
            dtype={"sample_id": str, "mapillary_id": str},
        ).reset_index(drop=True)
        required = {
            "sample_id",
            "label",
            "street_view_path",
            "remote_sensing_path",
            "spatial_block_id",
            "sequence_id",
            "latitude",
            "longitude",
            "compass_angle_deg",
        }
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise ValueError(f"{split_csv} is missing columns: {missing}")
        if self.frame["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample_id in {split_csv}")
        self.sectors = sectors
        street_mean, street_std = get_backbone_normalization(street_backbone)
        overhead_mean, overhead_std = get_backbone_normalization(overhead_backbone)
        self.street_transform = build_transform(
            image_size,
            domain="street",
            mean=street_mean,
            std=street_std,
        )
        self.overhead_transform = build_transform(
            image_size,
            domain="overhead",
            mean=overhead_mean,
            std=overhead_std,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(row["street_view_path"]) as source:
            panorama = source.convert("RGB")
        validate_equirectangular_size(panorama.size)
        sector_tensors = torch.stack(
            [
                self.street_transform(crop_equirectangular_sector(panorama, sector))
                for sector in self.sectors
            ]
        )
        panorama_tensor = self.street_transform(panorama)
        with Image.open(row["remote_sensing_path"]) as source:
            overhead_tensor = self.overhead_transform(source.convert("RGB"))
        return {
            "sample_id": str(row["sample_id"]),
            "target": int(row["label"]),
            "spatial_block_id": str(row["spatial_block_id"]),
            "sequence_id": str(row["sequence_id"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "compass_angle_deg": float(row["compass_angle_deg"]),
            "sectors": sector_tensors,
            "panorama": panorama_tensor,
            "overhead": overhead_tensor,
        }


def _cache_split(
    *,
    model: torch.nn.Module,
    dataset: CVIANSectorBagDataset,
    sectors: tuple[PanoramaSector, ...],
    output_path: Path,
    batch_size: int,
    num_workers: int,
    device: str,
) -> dict[str, object]:
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
    targets: list[np.ndarray] = []
    latitudes: list[np.ndarray] = []
    longitudes: list[np.ndarray] = []
    compass_angles: list[np.ndarray] = []
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
            batch_size_actual, sector_count = sector_batch.shape[:2]
            flat_sectors = sector_batch.reshape(
                batch_size_actual * sector_count, *sector_batch.shape[2:]
            )
            repeated_overhead = (
                overhead[:, None]
                .expand(batch_size_actual, sector_count, *overhead.shape[1:])
                .reshape(batch_size_actual * sector_count, *overhead.shape[1:])
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
            assert flat_street is not None and flat_overhead is not None
            sector_logits.append(
                flat_logits.reshape(batch_size_actual, sector_count, -1)
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            street_embeddings.append(
                flat_street.reshape(batch_size_actual, sector_count, -1)
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            overhead_embeddings.append(
                flat_overhead.reshape(batch_size_actual, sector_count, -1)[:, 0]
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            panorama_logits.append(full_logits.float().cpu().numpy().astype(np.float16))
            sample_ids.extend(str(value) for value in batch["sample_id"])
            spatial_blocks.extend(str(value) for value in batch["spatial_block_id"])
            sequence_ids.extend(str(value) for value in batch["sequence_id"])
            targets.append(batch["target"].numpy().astype(np.int8))
            latitudes.append(batch["latitude"].numpy().astype(np.float64))
            longitudes.append(batch["longitude"].numpy().astype(np.float64))
            compass_angles.append(
                batch["compass_angle_deg"].numpy().astype(np.float32)
            )
            if batch_index % 25 == 0:
                print(
                    f"cached {min(len(dataset), len(sample_ids))}/{len(dataset)}",
                    flush=True,
                )
    arrays = {
        "sample_id": np.asarray(sample_ids, dtype="U32"),
        "spatial_block_id": np.asarray(spatial_blocks, dtype="U64"),
        "sequence_id": np.asarray(sequence_ids, dtype="U64"),
        "target": np.concatenate(targets),
        "latitude": np.concatenate(latitudes),
        "longitude": np.concatenate(longitudes),
        "compass_angle_deg": np.concatenate(compass_angles),
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
        raise ValueError("Cached sample ids are not unique")
    if arrays["street_embedding"].shape[:2] != (len(dataset), len(sectors)):
        raise ValueError("Cached sector embedding shape is inconsistent")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    return {
        "rows": len(dataset),
        "sectors_per_row": len(sectors),
        "embedding_dim": int(arrays["street_embedding"].shape[-1]),
        "classes": int(arrays["sector_logits"].shape[-1]),
        "output": str(output_path),
        "sha256": _sha256(output_path),
    }


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    checkpoint_root = Path(args.checkpoint_root)
    output_root = Path(args.output_root)
    for name, value in (
        ("split_dir", split_dir),
        ("checkpoint_root", checkpoint_root),
        ("output_root", output_root),
    ):
        if not value.is_absolute():
            resolved = (REPO_ROOT / value).resolve()
            if name == "split_dir":
                split_dir = resolved
            elif name == "checkpoint_root":
                checkpoint_root = resolved
            else:
                output_root = resolved
    sectors = build_panorama_sectors(
        num_sectors=args.num_sectors,
        horizontal_fov_deg=args.horizontal_fov_deg,
        vertical_fov_deg=args.vertical_fov_deg,
    )
    split_names = tuple(
        value.strip() for value in args.splits.split(",") if value.strip()
    )
    if not split_names or len(set(split_names)) != len(split_names):
        raise ValueError("--splits must contain unique, non-empty role names")
    if any(not value.replace("_", "").isalnum() for value in split_names):
        raise ValueError("--splits role names may contain only letters, digits, and underscores")
    frames, source_hashes = _read_source_frames(split_dir, split_names)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    summaries = []
    built_any = False
    for seed in seeds:
        checkpoint_path = checkpoint_root / f"crossview_seed{seed}" / "triage_best.pt"
        seed_dir = output_root / f"seed{seed}"
        output_paths = {split: seed_dir / f"{split}.npz" for split in split_names}
        existing = {split: path.is_file() for split, path in output_paths.items()}
        if not args.overwrite and any(existing.values()) and not all(existing.values()):
            raise ValueError(
                f"Embedding cache for seed {seed} is only partially present; "
                "rerun with --overwrite"
            )
        training_attestation = (
            _validate_training_attestation(
                checkpoint_path,
                seed=seed,
                source_hashes=source_hashes,
            )
            if args.require_training_attestation
            else None
        )
        model, checkpoint = load_triage_from_checkpoint(checkpoint_path, device=args.device)
        config = checkpoint.get("config", {})
        if config.get("mode") != "crossview" or int(config.get("num_classes", 0)) != 3:
            raise ValueError(f"Expected a 3-class crossview checkpoint: {checkpoint_path}")
        num_classes = int(config["num_classes"])
        street_backbone = str(config.get("street_backbone", "resnet18"))
        overhead_backbone = str(config.get("overhead_backbone", "resnet18"))
        expected_provenance: dict[str, Any] = {
            "schema_version": EMBEDDING_SCHEMA_VERSION,
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "training_attestation": training_attestation,
            "training_attestation_required": args.require_training_attestation,
            "split_dir": str(split_dir),
            "source_manifest_sha256": source_hashes,
            "split_roles": list(split_names),
            "protocol_artifact_sha256": {
                name: _sha256(split_dir / name)
                for name in ("protocol_summary.json", "test_commitment.json")
                if (split_dir / name).is_file()
            },
            "entrypoint_sha256": {
                "cache_script": _sha256(Path(__file__).resolve()),
                "panorama_geometry": _sha256(
                    REPO_ROOT / "crossview_conflict" / "data" / "panorama.py"
                ),
                "triage_model": _sha256(
                    REPO_ROOT / "crossview_conflict" / "models" / "triage.py"
                ),
            },
            "num_sectors": args.num_sectors,
            "horizontal_fov_deg": args.horizontal_fov_deg,
            "vertical_fov_deg": args.vertical_fov_deg,
            "image_size": args.image_size,
            "street_backbone": street_backbone,
            "overhead_backbone": overhead_backbone,
            "model_mode": str(config["mode"]),
            "num_classes": num_classes,
            "sector_id": [sector.sector_id for sector in sectors],
            "relative_azimuth_deg": [
                float(sector.relative_azimuth_deg) for sector in sectors
            ],
            "initial_state": "post_overhead+sector_0",
            "claim_scope": "frozen panorama-to-sector transfer cache",
            "compass_metadata": "sensor metadata cached for candidate absolute-yaw features",
            "prospective_test_scored": False,
        }
        metadata_path = seed_dir / "cache_metadata.json"
        if not args.overwrite and all(existing.values()):
            metadata = validate_reusable_embedding_cache(
                metadata_path=metadata_path,
                seed_dir=seed_dir,
                frames=frames,
                sectors=sectors,
                num_classes=num_classes,
                expected_provenance=expected_provenance,
                split_names=split_names,
            )
            summaries.append(metadata)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        split_summaries = {}
        for split in split_names:
            dataset = CVIANSectorBagDataset(
                split_dir / f"{split}.csv",
                sectors,
                image_size=args.image_size,
                street_backbone=street_backbone,
                overhead_backbone=overhead_backbone,
            )
            split_summaries[split] = _cache_split(
                model=model,
                dataset=dataset,
                sectors=sectors,
                output_path=output_paths[split],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
            )
        metadata = {
            **expected_provenance,
            "splits": split_summaries,
        }
        seed_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append(metadata)
        built_any = True
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if built_any:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "cache_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": EMBEDDING_SUMMARY_SCHEMA_VERSION,
                    "runs": summaries,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
