from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from crossview_conflict.data.panorama import build_panorama_sectors
from scripts import build_cvian_active_view_manifests as manifest_builder
from scripts import cache_cvian_active_view_embeddings as embedding_cache
from scripts import cache_cvian_sector_visibility as visibility_cache


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_manifest_artifacts(
    tmp_path: Path,
    *,
    materialized: bool,
) -> tuple[Path, Path, dict[str, pd.DataFrame]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    split_dir = tmp_path / "splits"
    output_dir = tmp_path / "active_view"
    split_dir.mkdir()
    output_dir.mkdir()
    frames: dict[str, pd.DataFrame] = {}
    episodes: list[dict[str, object]] = []
    for index, split in enumerate(manifest_builder.SPLIT_NAMES, start=1):
        sample_id = f"{index:06d}"
        panorama_path = tmp_path / f"{split}_panorama.png"
        overhead_path = tmp_path / f"{split}_overhead.png"
        Image.new("RGB", (16, 8), color=(index, index + 1, index + 2)).save(
            panorama_path
        )
        Image.new("RGB", (8, 8), color=(index + 3, index + 4, index + 5)).save(
            overhead_path
        )
        source = pd.DataFrame(
            [
                {
                    "sample_id": sample_id,
                    "mapillary_id": f"123456789012345{index}",
                    "label": index - 1,
                    "severity": f"severity_{index}",
                    "spatial_block_id": f"block_{split}",
                    "sequence_id": f"sequence_{split}",
                    "compass_angle_deg": 350.0 + index,
                    "is_pano": 1,
                    "street_view_path": str(panorama_path),
                    "remote_sensing_path": str(overhead_path),
                }
            ]
        )
        source.to_csv(split_dir / f"{split}.csv", index=False)
        frames[split] = source
        sectors = manifest_builder.build_sector_rows(
            source,
            split=split,
            image_dir=output_dir / "images",
            num_sectors=8,
            horizontal_fov_deg=90.0,
            vertical_fov_deg=90.0,
            jpeg_quality=92,
            materialize_crops=materialized,
            overwrite=False,
        )
        sectors.to_csv(output_dir / f"{split}.csv", index=False)
        episodes.append(
            {
                "episode_id": f"cvian_{sample_id}",
                "parent_sample_id": sample_id,
                "split_role": split,
                "spatial_block_id": f"block_{split}",
                "sequence_id": f"sequence_{split}",
                "initial_state": "post_overhead+sector_0",
                "initial_sector_id": 0,
                "candidate_sector_ids": "1,2,3,4,5,6,7",
                "max_additional_reveals": 7,
                "reveal_cost": 0.5,
                "stop_cost": 0.0,
                "defer_human_cost": 1.5,
            }
        )
    pd.DataFrame(episodes).to_csv(
        output_dir / "active_view_episodes.csv", index=False
    )
    core = manifest_builder.verify_outputs(
        output_dir,
        frames,
        num_sectors=8,
        horizontal_fov_deg=90.0,
        vertical_fov_deg=90.0,
        materialized=materialized,
    )
    summary = manifest_builder._summary_payload(
        core,
        split_dir=split_dir.resolve(),
        output_dir=output_dir.resolve(),
        horizontal_fov_deg=90.0,
        vertical_fov_deg=90.0,
        jpeg_quality=92,
    )
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return split_dir, output_dir, frames


def test_verify_only_is_read_only_and_checks_existing_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_dir, output_dir, _ = _build_manifest_artifacts(
        tmp_path, materialized=False
    )
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        manifest_builder,
        "parse_args",
        lambda: argparse.Namespace(
            split_dir=str(split_dir),
            output_dir=str(output_dir),
            num_sectors=8,
            horizontal_fov_deg=90.0,
            vertical_fov_deg=90.0,
            jpeg_quality=92,
            materialize_crops=False,
            overwrite=False,
            verify_only=True,
        ),
    )
    manifest_builder.main()
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert after == before

    summary_path = output_dir / "manifest_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["horizontal_fov_deg"] = 45.0
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tampered = summary_path.read_bytes()
    with pytest.raises(ValueError, match="horizontal_fov_deg"):
        manifest_builder.main()
    assert summary_path.read_bytes() == tampered


def test_manifest_verifier_rejects_geometry_and_materialized_hash_tampering(
    tmp_path: Path,
) -> None:
    _, output_dir, frames = _build_manifest_artifacts(
        tmp_path / "lazy", materialized=False
    )
    train_path = output_dir / "train.csv"
    train = pd.read_csv(
        train_path,
        dtype={"sample_id": str, "parent_sample_id": str, "mapillary_id": str},
    )
    train.loc[0, "relative_azimuth_deg"] = 12.0
    train.to_csv(train_path, index=False)
    with pytest.raises(ValueError, match="relative azimuth"):
        manifest_builder.verify_outputs(
            output_dir,
            frames,
            num_sectors=8,
            horizontal_fov_deg=90.0,
            vertical_fov_deg=90.0,
            materialized=False,
        )

    _, materialized_dir, materialized_frames = _build_manifest_artifacts(
        tmp_path / "materialized", materialized=True
    )
    materialized_train = pd.read_csv(materialized_dir / "train.csv")
    sector_path = Path(materialized_train.loc[0, "sector_path"])
    with sector_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="sector SHA-256"):
        manifest_builder.verify_outputs(
            materialized_dir,
            materialized_frames,
            num_sectors=8,
            horizontal_fov_deg=90.0,
            vertical_fov_deg=90.0,
            materialized=True,
        )

    _, contract_dir, contract_frames = _build_manifest_artifacts(
        tmp_path / "contract", materialized=False
    )
    contract_train_path = contract_dir / "train.csv"
    contract_train = pd.read_csv(
        contract_train_path,
        dtype={"sample_id": str, "parent_sample_id": str, "mapillary_id": str},
    )
    original_severity = contract_train.loc[0, "severity"]
    contract_train.loc[0, "severity"] = "not-the-parent-value"
    contract_train.to_csv(contract_train_path, index=False)
    with pytest.raises(ValueError, match="inherited severity"):
        manifest_builder.verify_outputs(
            contract_dir,
            contract_frames,
            num_sectors=8,
            materialized=False,
        )
    contract_train.loc[0, "severity"] = original_severity
    contract_train.to_csv(contract_train_path, index=False)
    episode_path = contract_dir / "active_view_episodes.csv"
    episodes = pd.read_csv(
        episode_path, dtype={"episode_id": str, "parent_sample_id": str}
    )
    episodes.loc[0, "split_role"] = "val"
    episodes.to_csv(episode_path, index=False)
    with pytest.raises(ValueError, match="split_role"):
        manifest_builder.verify_outputs(
            contract_dir,
            contract_frames,
            num_sectors=8,
            materialized=False,
        )


def test_visibility_cache_reuse_validates_provenance_and_npz(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "visibility"
    output_dir.mkdir()
    frames: dict[str, pd.DataFrame] = {}
    split_summaries: dict[str, dict[str, object]] = {}
    sector_ids = np.asarray([0, 1], dtype=np.int8)
    for index, split in enumerate(visibility_cache.SPLIT_NAMES):
        sample_id = f"{index + 1:06d}"
        frames[split] = pd.DataFrame([{"sample_id": sample_id}])
        path = output_dir / f"{split}.npz"
        ratios = np.asarray([[0.0, 0.5]], dtype=np.float16)
        np.savez(
            path,
            sample_id=np.asarray([sample_id]),
            sector_id=sector_ids,
            building_ratio=ratios,
        )
        split_summaries[split] = {
            "rows": 1,
            "sectors_per_row": 2,
            "output": str(path),
            "sha256": _sha256(path),
            "no_building_panorama_count": 0,
        }
    provenance = {
        "schema_version": visibility_cache.VISIBILITY_SCHEMA_VERSION,
        "model_id": "model",
        "model_revision_resolved": "commit-a",
        "source_manifest_sha256": {split: f"hash-{split}" for split in frames},
        "num_sectors": 2,
    }
    metadata_path = output_dir / "visibility_metadata.json"
    metadata_path.write_text(
        json.dumps({**provenance, "splits": split_summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    visibility_cache.validate_reusable_visibility_cache(
        metadata_path=metadata_path,
        output_dir=output_dir,
        frames=frames,
        expected_provenance=provenance,
        sector_ids=sector_ids,
    )
    before = metadata_path.read_bytes()
    mismatched = {**provenance, "model_revision_resolved": "commit-b"}
    with pytest.raises(ValueError, match="model_revision_resolved"):
        visibility_cache.validate_reusable_visibility_cache(
            metadata_path=metadata_path,
            output_dir=output_dir,
            frames=frames,
            expected_provenance=mismatched,
            sector_ids=sector_ids,
        )
    assert metadata_path.read_bytes() == before


def test_embedding_cache_reuse_validates_checkpoint_and_split_content(
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "seed42"
    seed_dir.mkdir()
    sectors = build_panorama_sectors(
        num_sectors=2,
        horizontal_fov_deg=180.0,
        vertical_fov_deg=90.0,
    )
    frames: dict[str, pd.DataFrame] = {}
    split_summaries: dict[str, dict[str, object]] = {}
    for index, split in enumerate(embedding_cache.SPLIT_NAMES):
        sample_id = f"{index + 1:06d}"
        frames[split] = pd.DataFrame(
            [
                {
                    "sample_id": sample_id,
                    "label": index % 3,
                    "spatial_block_id": f"block-{split}",
                    "sequence_id": f"sequence-{split}",
                    "latitude": 26.0 + index,
                    "longitude": -82.0 - index,
                }
            ]
        )
        path = seed_dir / f"{split}.npz"
        np.savez(
            path,
            sample_id=np.asarray([sample_id]),
            spatial_block_id=np.asarray([f"block-{split}"]),
            sequence_id=np.asarray([f"sequence-{split}"]),
            target=np.asarray([index % 3], dtype=np.int8),
            latitude=np.asarray([26.0 + index]),
            longitude=np.asarray([-82.0 - index]),
            sector_id=np.asarray([0, 1], dtype=np.int8),
            relative_azimuth_deg=np.asarray([0.0, -180.0], dtype=np.float32),
            street_embedding=np.ones((1, 2, 4), dtype=np.float16),
            overhead_embedding=np.ones((1, 4), dtype=np.float16),
            sector_logits=np.ones((1, 2, 3), dtype=np.float16),
            panorama_logits=np.ones((1, 3), dtype=np.float16),
        )
        split_summaries[split] = {
            "rows": 1,
            "sectors_per_row": 2,
            "embedding_dim": 4,
            "classes": 3,
            "output": str(path),
            "sha256": _sha256(path),
        }
    provenance = {
        "schema_version": embedding_cache.EMBEDDING_SCHEMA_VERSION,
        "checkpoint_sha256": "checkpoint-a",
        "source_manifest_sha256": {split: f"hash-{split}" for split in frames},
        "num_sectors": 2,
        "image_size": 224,
    }
    metadata_path = seed_dir / "cache_metadata.json"
    metadata_path.write_text(
        json.dumps({**provenance, "splits": split_summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    embedding_cache.validate_reusable_embedding_cache(
        metadata_path=metadata_path,
        seed_dir=seed_dir,
        frames=frames,
        sectors=sectors,
        num_classes=3,
        expected_provenance=provenance,
    )
    before = metadata_path.read_bytes()
    mismatched = {**provenance, "checkpoint_sha256": "checkpoint-b"}
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        embedding_cache.validate_reusable_embedding_cache(
            metadata_path=metadata_path,
            seed_dir=seed_dir,
            frames=frames,
            sectors=sectors,
            num_classes=3,
            expected_provenance=mismatched,
        )
    assert metadata_path.read_bytes() == before
