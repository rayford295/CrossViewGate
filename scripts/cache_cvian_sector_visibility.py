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
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.data.panorama import (
    build_panorama_sectors,
    crop_equirectangular_sector,
    validate_equirectangular_size,
)


SPLIT_NAMES = ("train", "val", "test")
VISIBILITY_SCHEMA_VERSION = "cvian-sector-visibility-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache CVIAN per-sector building ratios from a full-panorama "
            "SegFormer mask. These ratios are privileged/offline baselines and "
            "must not enter an online policy observation."
        )
    )
    parser.add_argument("--split-dir", default="data/splits/ian_hurricane_original")
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/active_view_cvian_spatial_v1/visibility",
    )
    parser.add_argument("--num-sectors", type=int, default=8)
    parser.add_argument(
        "--splits",
        default=",".join(SPLIT_NAMES),
        help="Comma-separated role CSV stems to cache.",
    )
    parser.add_argument("--horizontal-fov-deg", type=float, default=90.0)
    parser.add_argument("--vertical-fov-deg", type=float, default=90.0)
    parser.add_argument("--model-id", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional Hugging Face revision; the resolved immutable commit is recorded.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_model(
    model_id: str,
    device: str,
    model_revision: str | None = None,
):
    kwargs = {"revision": model_revision} if model_revision else {}
    processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id, **kwargs).to(device)
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None) or getattr(
        processor, "_commit_hash", None
    )
    if not resolved_revision:
        raise ValueError(
            "Could not resolve an immutable model revision; pass --model-revision "
            "as a commit SHA before creating a reusable visibility cache"
        )
    for index, label in model.config.id2label.items():
        if str(label).lower() == "building":
            return processor, model, int(index), str(resolved_revision)
    raise KeyError(f"No building class in {model_id}")


def _read_source_frames(
    split_dir: Path,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for split in split_names:
        path = split_dir / f"{split}.csv"
        frame = pd.read_csv(path, dtype={"sample_id": str}).reset_index(drop=True)
        missing = sorted({"sample_id", "street_view_path"} - set(frame.columns))
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


def _validate_visibility_npz(
    path: Path,
    frame: pd.DataFrame,
    *,
    sector_ids: np.ndarray,
    summary: dict[str, Any],
) -> None:
    actual_hash = _sha256(path)
    if summary.get("sha256") != actual_hash:
        raise ValueError(f"Visibility cache hash mismatch for {path}; rerun with --overwrite")
    with np.load(path, allow_pickle=False) as arrays:
        required = {"sample_id", "sector_id", "building_ratio"}
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(f"{path} is missing arrays {missing}; rerun with --overwrite")
        sample_ids = arrays["sample_id"].astype(str)
        cached_sector_ids = arrays["sector_id"]
        ratios = arrays["building_ratio"]
    expected_ids = frame["sample_id"].astype(str).to_numpy()
    if not np.array_equal(sample_ids, expected_ids):
        raise ValueError(
            f"Visibility cache sample order mismatch for {path}; rerun with --overwrite"
        )
    if not np.array_equal(cached_sector_ids, sector_ids):
        raise ValueError(
            f"Visibility cache sector geometry mismatch for {path}; rerun with --overwrite"
        )
    if ratios.shape != (len(frame), len(sector_ids)):
        raise ValueError(f"Visibility cache shape mismatch for {path}; rerun with --overwrite")
    if not np.isfinite(ratios).all() or np.any(ratios < 0.0) or np.any(ratios > 1.0):
        raise ValueError(f"Visibility cache contains invalid ratios: {path}")
    expected_summary = {
        "rows": len(frame),
        "sectors_per_row": len(sector_ids),
        "output": str(path),
        "sha256": actual_hash,
        "no_building_panorama_count": int((ratios.max(axis=1) == 0).sum()),
    }
    _assert_provenance(summary, expected_summary, context=f"visibility split {path.stem}")


def validate_reusable_visibility_cache(
    *,
    metadata_path: Path,
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    expected_provenance: dict[str, Any],
    sector_ids: np.ndarray,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> dict[str, Any]:
    if not metadata_path.is_file():
        raise ValueError(
            f"Visibility cache exists without provenance metadata {metadata_path}; "
            "rerun with --overwrite"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid visibility metadata: {metadata_path}")
    _assert_provenance(metadata, expected_provenance, context="visibility cache")
    summaries = metadata.get("splits")
    if not isinstance(summaries, dict) or set(summaries) != set(split_names):
        raise ValueError("Visibility cache split metadata is incomplete; rerun with --overwrite")
    for split in split_names:
        output_path = output_dir / f"{split}.npz"
        if not output_path.is_file():
            raise ValueError(f"Missing visibility cache {output_path}; rerun with --overwrite")
        if not isinstance(summaries[split], dict):
            raise ValueError(f"Invalid visibility metadata for {split}; rerun with --overwrite")
        _validate_visibility_npz(
            output_path,
            frames[split],
            sector_ids=sector_ids,
            summary=summaries[split],
        )
    return metadata


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    if not split_dir.is_absolute():
        split_dir = (REPO_ROOT / split_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
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
    output_paths = {split: output_dir / f"{split}.npz" for split in split_names}
    existing = {split: path.is_file() for split, path in output_paths.items()}
    if not args.overwrite and any(existing.values()) and not all(existing.values()):
        raise ValueError(
            "Visibility cache is only partially present; rerun with --overwrite "
            "to rebuild one coherent cache"
        )
    processor, model, building_id, resolved_revision = _prepare_model(
        args.model_id, args.device, args.model_revision
    )
    sector_ids = np.asarray([sector.sector_id for sector in sectors], dtype=np.int8)
    expected_provenance: dict[str, Any] = {
        "schema_version": VISIBILITY_SCHEMA_VERSION,
        "model_id": args.model_id,
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": resolved_revision,
        "building_class_id": building_id,
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
        },
        "num_sectors": args.num_sectors,
        "horizontal_fov_deg": args.horizontal_fov_deg,
        "vertical_fov_deg": args.vertical_fov_deg,
        "sector_id": sector_ids.astype(int).tolist(),
        "relative_azimuth_deg": [
            float(sector.relative_azimuth_deg) for sector in sectors
        ],
        "prospective_test_scored": False,
    }
    metadata_path = output_dir / "visibility_metadata.json"
    if not args.overwrite and all(existing.values()):
        metadata = validate_reusable_visibility_cache(
            metadata_path=metadata_path,
            output_dir=output_dir,
            frames=frames,
            expected_provenance=expected_provenance,
            sector_ids=sector_ids,
            split_names=split_names,
        )
        print(json.dumps(metadata, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split in split_names:
        output_path = output_paths[split]
        frame = frames[split]
        ratios = np.zeros((len(frame), len(sectors)), dtype=np.float32)
        sample_ids: list[str] = []
        for start in range(0, len(frame), args.batch_size):
            batch = frame.iloc[start : start + args.batch_size]
            images = []
            for path in batch["street_view_path"]:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                validate_equirectangular_size(image.size)
                images.append(image)
            inputs = processor(images=images, return_tensors="pt").to(args.device)
            with torch.no_grad():
                logits = model(**inputs).logits
            for local_index, image in enumerate(images):
                upsampled = torch.nn.functional.interpolate(
                    logits[local_index : local_index + 1],
                    size=image.size[::-1],
                    mode="bilinear",
                    align_corners=False,
                )
                mask = (upsampled.argmax(dim=1)[0] == building_id).cpu().numpy()
                mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
                for sector in sectors:
                    sector_mask = np.asarray(
                        crop_equirectangular_sector(mask_image, sector)
                    )[..., 0]
                    ratios[start + local_index, sector.sector_id] = float(
                        (sector_mask > 0).mean()
                    )
            sample_ids.extend(str(value) for value in batch["sample_id"])
            if start % (args.batch_size * 25) == 0:
                print(f"{split}: {min(len(frame), start + len(batch))}/{len(frame)}", flush=True)
        cached_ratios = ratios.astype(np.float16)
        np.savez(
            output_path,
            sample_id=np.asarray(sample_ids, dtype="U32"),
            sector_id=np.asarray([sector.sector_id for sector in sectors], dtype=np.int8),
            building_ratio=cached_ratios,
        )
        summaries[split] = {
            "rows": len(frame),
            "sectors_per_row": len(sectors),
            "output": str(output_path),
            "sha256": _sha256(output_path),
            "no_building_panorama_count": int((cached_ratios.max(axis=1) == 0).sum()),
        }
    metadata = {
        **expected_provenance,
        "policy_scope": "privileged_offline_baseline_only",
        "warning": (
            "Full-panorama segmentation reads hidden sectors and is forbidden in "
            "online policy observations."
        ),
        "splits": summaries,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
