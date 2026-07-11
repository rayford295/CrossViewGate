from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from scripts.build_milton_active_view_transfer_manifest import (
    CLAIM_SCOPE,
    CONTRACT_FILENAME,
    OUTPUT_FILENAME,
    build_transfer_manifest,
    spatial_block_id,
)
from scripts.cache_milton_active_view_embeddings import (
    BASE_ATTESTATION_SCHEMA_VERSION,
    SEEDS,
    _load_transfer_manifest,
    _unicode_array,
    _validate_attested_checkpoints,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="PNG")


def _canonical_fixture(
    tmp_path: Path, *, non_panorama_index: int | None = None
) -> tuple[Path, Path]:
    media = tmp_path / "media"
    long_id = (
        "1000066497969277_vs_2716734655175175(4)-"
        + "identifier-kept-beyond-thirty-two-characters"
    )
    rows: list[dict[str, object]] = []
    for index, (sample_id, label, latitude, longitude) in enumerate(
        (
            (long_id, 0, "29.43910", "-83.29110"),
            ("second_missing_sequence_sample", 1, "29.43980", "-83.29180"),
            ("third_missing_sequence_sample", 2, "29.44100", "-83.29400"),
        )
    ):
        size = (9, 4) if non_panorama_index == index else (8, 4)
        street = media / f"street-{index}.png"
        # The first two rows intentionally share overhead content and therefore
        # the same dependency group while retaining different missing-sequence
        # sentinels.
        overhead_index = 0 if index < 2 else 1
        overhead = media / f"overhead-{overhead_index}.png"
        _save_image(street, size, (20 + index, 30, 40))
        if not overhead.exists():
            _save_image(overhead, (4, 4), (50 + overhead_index, 60, 70))
        street_hash = _sha256(street)
        overhead_hash = _sha256(overhead)
        rows.append(
            {
                "sample_id": sample_id,
                "canonical_sample_id": sample_id,
                "pre_image_id": f"pre-{index}",
                "post_image_id": f"post-{index}",
                "label": label,
                "label_name": ("mild_damage", "moderate_damage", "severe_damage")[label],
                "latitude": latitude,
                "longitude": longitude,
                "street_view_path": str(street),
                "remote_sensing_path": str(overhead),
                "post_panorama_sha256": street_hash,
                "remote_content_sha256": overhead_hash,
                "post_panorama_group_id": f"post_sha256:{street_hash}",
                "remote_content_group_id": f"remote_sha256:{overhead_hash}",
                "sequence_id": "",
                "compass_angle_deg": "",
                "sequence_metadata_available": False,
                "compass_metadata_available": False,
                "absolute_azimuth_available": False,
                "panorama_geometry_verified": True,
                "post_width": size[0],
                "post_height": size[1],
                "claim_scope": CLAIM_SCOPE,
                "cohort_status": "main_unanimous_post_representative",
            }
        )
    source_csv = tmp_path / "main_sensitivity_cohort.csv"
    pd.DataFrame(rows).to_csv(source_csv, index=False)
    audit = {
        "schema_version": "milton-canonical-sensitivity-v1",
        "claim_scope": CLAIM_SCOPE,
        "confirmatory_eligible": False,
        "counts": {"main_rows": 3},
        "metadata_availability": {
            "sequence_id": False,
            "compass_angle_deg": False,
            "absolute_azimuth": False,
        },
        "output_sha256": {source_csv.name: _sha256(source_csv)},
    }
    audit_path = tmp_path / "provenance_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return source_csv, audit_path


def _build_fixture_manifest(tmp_path: Path) -> Path:
    source_csv, audit_path = _canonical_fixture(tmp_path)
    output_dir = tmp_path / "transfer"
    build_transfer_manifest(
        source_csv=source_csv,
        provenance_audit=audit_path,
        output_dir=output_dir,
        expected_source_sha256=_sha256(source_csv),
        expected_rows=3,
    )
    return output_dir


def test_transfer_manifest_preserves_long_ids_and_does_not_collapse_missing_sequences(
    tmp_path: Path,
) -> None:
    output_dir = _build_fixture_manifest(tmp_path)
    frame = pd.read_csv(output_dir / OUTPUT_FILENAME, dtype=str)
    contract = json.loads((output_dir / CONTRACT_FILENAME).read_text())

    assert len(frame.iloc[0]["sample_id"]) > 32
    assert frame.iloc[0]["sample_id"].endswith("beyond-thirty-two-characters")
    assert frame["sequence_id"].nunique() == len(frame)
    assert frame["sequence_id"].str.startswith("sequence_unavailable_for_sample:").all()
    assert set(frame["sequence_metadata_available"].str.lower()) == {"false"}
    assert set(frame["sequence_id_is_observed"].str.lower()) == {"false"}
    assert set(frame["compass_available"].str.lower()) == {"false"}
    assert frame["compass_angle_deg"].isna().all()
    assert frame["dependency_group_id"].equals(frame["remote_content_group_id"])
    assert frame.iloc[0]["spatial_block_id"] == frame.iloc[1]["spatial_block_id"]
    assert frame.iloc[0]["dependency_group_id"] == frame.iloc[1]["dependency_group_id"]
    assert frame.iloc[0]["sequence_id"] != frame.iloc[1]["sequence_id"]
    assert contract["operation_scope"] == "forward_only_transfer_sensitivity"
    assert contract["model_or_policy_selection_permitted"] is False
    assert contract["rows"] == 3

    loaded, _, _ = _load_transfer_manifest(
        output_dir, enforce_registered_snapshot=False
    )
    assert loaded["sample_id"].tolist() == frame["sample_id"].tolist()


def test_spatial_block_is_half_open_and_deterministic_for_negative_longitude() -> None:
    assert spatial_block_id("29.43910", "-83.29110") == spatial_block_id(
        "29.43999", "-83.29199"
    )
    assert spatial_block_id("29.44000", "-83.29110") != spatial_block_id(
        "29.43999", "-83.29110"
    )
    assert spatial_block_id("29.43910", "-83.29100") != spatial_block_id(
        "29.43910", "-83.29101"
    )


def test_transfer_manifest_fails_closed_on_non_2_to_1_panorama(tmp_path: Path) -> None:
    source_csv, audit_path = _canonical_fixture(tmp_path, non_panorama_index=1)
    with pytest.raises(ValueError, match="not a 2:1 panorama"):
        build_transfer_manifest(
            source_csv=source_csv,
            provenance_audit=audit_path,
            output_dir=tmp_path / "transfer",
            expected_source_sha256=_sha256(source_csv),
            expected_rows=3,
        )


def test_unicode_cache_encoding_never_truncates_long_identifiers() -> None:
    values = ["short", "x" * 173, "sequence_unavailable_for_sample:" + "y" * 90]
    encoded = _unicode_array(values)
    assert encoded.tolist() == values
    assert encoded.dtype.kind == "U"
    assert encoded.dtype.itemsize // np.dtype("U1").itemsize == 173


def _attested_checkpoint_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "base_encoders"
    runs = []
    for seed in SEEDS:
        run_dir = root / f"crossview_seed{seed}"
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "triage_best.pt"
        history = run_dir / "triage_history.json"
        log = run_dir / "train.log"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        history.write_text(f"{{\"seed\": {seed}}}", encoding="utf-8")
        log.write_text(f"seed={seed}\n", encoding="utf-8")
        completion = {
            "schema_version": BASE_ATTESTATION_SCHEMA_VERSION,
            "seed": seed,
            "mode": "crossview",
            "train_role": "base_fit",
            "validation_role": "validation",
            "image_size": 224,
            "test_evaluated": False,
            "fingerprint_sha256": f"fingerprint-{seed}",
            "checkpoint_sha256": _sha256(checkpoint),
            "history_sha256": _sha256(history),
            "train_log_sha256": _sha256(log),
        }
        (run_dir / "training_complete.json").write_text(
            json.dumps(completion, indent=2) + "\n", encoding="utf-8"
        )
        runs.append(completion)
    (root / "training_summary.json").write_text(
        json.dumps(
            {"schema_version": BASE_ATTESTATION_SCHEMA_VERSION, "runs": runs},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_checkpoint_inventory_validates_all_five_completion_and_artifact_hashes(
    tmp_path: Path,
) -> None:
    root = _attested_checkpoint_fixture(tmp_path)
    inventory = _validate_attested_checkpoints(root)
    assert tuple(sorted(inventory)) == tuple(sorted(SEEDS))
    assert inventory[42]["checkpoint_sha256"] == _sha256(
        root / "crossview_seed42" / "triage_best.pt"
    )

    (root / "crossview_seed456" / "triage_best.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="seed 456 checkpoint hash mismatch"):
        _validate_attested_checkpoints(root)
