from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crossview_conflict.decision import artifact_provenance as provenance


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_repo_sources(repo_root: Path) -> None:
    source_paths = set(provenance._BASE_SOURCE_FILES.values())
    source_paths.update(provenance._EMBEDDING_ENTRYPOINT_FILES.values())
    source_paths.update(provenance._VISIBILITY_ENTRYPOINT_FILES.values())
    for index, relative in enumerate(sorted(source_paths)):
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic source {index}\n", encoding="utf-8")


def _source_hashes(repo_root: Path, sources: dict[str, str]) -> dict[str, str]:
    return {name: _sha256(repo_root / relative) for name, relative in sources.items()}


def _build_protocol(protocol_dir: Path) -> dict[str, Any]:
    protocol_dir.mkdir(parents=True)
    role_hashes: dict[str, str] = {}
    role_rows: dict[str, int] = {}
    for index, role in enumerate(provenance.ROLE_NAMES):
        rows = index + 1
        path = protocol_dir / f"{role}.csv"
        lines = ["sample_id,label"]
        lines.extend(f"{role}-{row},{row % 3}" for row in range(rows))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        role_hashes[f"{role}.csv"] = _sha256(path)
        role_rows[role] = rows
    summary = {
        "schema_version": "cvian-sequence-four-role-summary-v1",
        "protocol_version": "cvian-sequence-four-role-v1",
        "role_rows": role_rows,
        "record_overlap_zero": True,
        "sequence_overlap_zero": True,
        "spatial_block_overlap_zero": True,
        "spatial_buffer_clear": True,
        "role_manifest_sha256": role_hashes,
        "test_status": {
            "status": "selector_selection_holdout_with_historical_base_exposure",
            "new_protocol_test_scored": False,
            "old_base_or_checkpoint_reuse_permitted": False,
        },
    }
    commitment = {
        "schema_version": "cvian-sequence-test-commitment-v1",
        "protocol_version": "cvian-sequence-four-role-v1",
        "role": "prospective_test",
        "status": "selector_selection_holdout_with_historical_base_exposure",
        "historical_checkpoint_reuse_permitted": False,
        "row_count": role_rows["prospective_test"],
        "manifest_sha256": role_hashes["prospective_test.csv"],
    }
    _write_json(protocol_dir / "protocol_summary.json", summary)
    _write_json(protocol_dir / "test_commitment.json", commitment)
    return {
        "summary_sha256": _sha256(protocol_dir / "protocol_summary.json"),
        "commitment_sha256": _sha256(protocol_dir / "test_commitment.json"),
        "role_hashes": {
            role: role_hashes[f"{role}.csv"] for role in provenance.ROLE_NAMES
        },
        "role_rows": role_rows,
    }


def _build_base_and_cache(
    repo_root: Path,
    protocol_dir: Path,
    base_root: Path,
    cache_root: Path,
    protocol: dict[str, Any],
    *,
    seed: int,
) -> None:
    base_dir = base_root / f"crossview_seed{seed}"
    base_dir.mkdir(parents=True)
    artifacts = {
        "checkpoint_sha256": base_dir / "triage_best.pt",
        "history_sha256": base_dir / "triage_history.json",
        "train_log_sha256": base_dir / "train.log",
    }
    for field, path in artifacts.items():
        path.write_bytes(f"{field}-{seed}".encode("ascii"))
    completion: dict[str, Any] = {
        "schema_version": "cvian-sequence-role-isolated-base-v1",
        "seed": seed,
        "mode": "crossview",
        "train_role": "base_fit",
        "validation_role": "validation",
        "epochs": 2,
        "patience": 1,
        "batch_size": 4,
        "num_workers": 0,
        "learning_rate": 0.001,
        "image_size": 224,
        "class_weighting": "balanced",
        "street_augment": True,
        "overhead_augment": True,
        "device": "cpu",
        "protocol_summary_sha256": protocol["summary_sha256"],
        "test_commitment_sha256": protocol["commitment_sha256"],
        "role_sha256": protocol["role_hashes"],
        "source_sha256": _source_hashes(repo_root, provenance._BASE_SOURCE_FILES),
        "test_evaluated": False,
    }
    completion["fingerprint_sha256"] = _canonical_hash(completion)
    for field, path in artifacts.items():
        completion[field] = _sha256(path)
    completion_path = base_dir / "training_complete.json"
    _write_json(completion_path, completion)

    seed_dir = cache_root / f"seed{seed}"
    seed_dir.mkdir(parents=True)
    split_metadata: dict[str, Any] = {}
    for role in provenance.ROLE_NAMES:
        path = seed_dir / f"{role}.npz"
        path.write_bytes(f"opaque embedding cache {seed} {role}".encode("ascii"))
        split_metadata[role] = {
            "rows": protocol["role_rows"][role],
            "sectors_per_row": 8,
            "embedding_dim": 256,
            "classes": 3,
            "output": str(path.resolve()),
            "sha256": _sha256(path),
        }
    metadata = {
        "schema_version": "cvian-active-view-cache-v3",
        "seed": seed,
        "checkpoint": str((base_dir / "triage_best.pt").resolve()),
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "checkpoint_epoch": 1,
        "training_attestation": {
            "path": str(completion_path.resolve()),
            "sha256": _sha256(completion_path),
            "fingerprint_sha256": completion["fingerprint_sha256"],
        },
        "training_attestation_required": True,
        "split_dir": str(protocol_dir.resolve()),
        "source_manifest_sha256": protocol["role_hashes"],
        "split_roles": list(provenance.ROLE_NAMES),
        "protocol_artifact_sha256": {
            "protocol_summary.json": protocol["summary_sha256"],
            "test_commitment.json": protocol["commitment_sha256"],
        },
        "entrypoint_sha256": _source_hashes(
            repo_root, provenance._EMBEDDING_ENTRYPOINT_FILES
        ),
        "num_sectors": 8,
        "horizontal_fov_deg": 90.0,
        "vertical_fov_deg": 90.0,
        "image_size": 224,
        "street_backbone": "resnet18",
        "overhead_backbone": "resnet18",
        "model_mode": "crossview",
        "num_classes": 3,
        "sector_id": list(provenance.EXPECTED_SECTOR_IDS),
        "relative_azimuth_deg": list(provenance.EXPECTED_RELATIVE_AZIMUTH_DEG),
        "initial_state": "post_overhead+sector_0",
        "prospective_test_scored": False,
        "splits": split_metadata,
    }
    _write_json(seed_dir / "cache_metadata.json", metadata)


def _build_visibility(
    repo_root: Path,
    protocol_dir: Path,
    visibility_dir: Path,
    protocol: dict[str, Any],
) -> None:
    visibility_dir.mkdir(parents=True)
    split_metadata: dict[str, Any] = {}
    for role in provenance.ROLE_NAMES:
        path = visibility_dir / f"{role}.npz"
        path.write_bytes(f"opaque visibility cache {role}".encode("ascii"))
        split_metadata[role] = {
            "rows": protocol["role_rows"][role],
            "sectors_per_row": 8,
            "output": str(path.resolve()),
            "sha256": _sha256(path),
            "no_building_panorama_count": 0,
        }
    metadata = {
        "schema_version": "cvian-sector-visibility-v3",
        "model_id": provenance.SEGFORMER_MODEL_ID,
        "model_revision_requested": None,
        "model_revision_resolved": provenance.SEGFORMER_REVISION,
        "building_class_id": 1,
        "split_dir": str(protocol_dir.resolve()),
        "source_manifest_sha256": protocol["role_hashes"],
        "split_roles": list(provenance.ROLE_NAMES),
        "protocol_artifact_sha256": {
            "protocol_summary.json": protocol["summary_sha256"],
            "test_commitment.json": protocol["commitment_sha256"],
        },
        "entrypoint_sha256": _source_hashes(
            repo_root, provenance._VISIBILITY_ENTRYPOINT_FILES
        ),
        "num_sectors": 8,
        "horizontal_fov_deg": 90.0,
        "vertical_fov_deg": 90.0,
        "sector_id": list(provenance.EXPECTED_SECTOR_IDS),
        "relative_azimuth_deg": list(provenance.EXPECTED_RELATIVE_AZIMUTH_DEG),
        "prospective_test_scored": False,
        "policy_scope": "privileged_offline_baseline_only",
        "splits": split_metadata,
    }
    _write_json(visibility_dir / "visibility_metadata.json", metadata)


@pytest.fixture
def synthetic_artifacts(tmp_path: Path) -> dict[str, Any]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_repo_sources(repo_root)
    protocol_dir = repo_root / "protocol"
    base_root = repo_root / "base"
    cache_root = repo_root / "cache"
    visibility_dir = repo_root / "visibility"
    protocol = _build_protocol(protocol_dir)
    _build_base_and_cache(
        repo_root,
        protocol_dir,
        base_root,
        cache_root,
        protocol,
        seed=42,
    )
    _build_visibility(repo_root, protocol_dir, visibility_dir, protocol)
    return {
        "repo_root": repo_root,
        "protocol_dir": protocol_dir,
        "base_root": base_root,
        "cache_root": cache_root,
        "visibility_dir": visibility_dir,
    }


def test_runtime_lock_validates_sources_and_exact_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    source = repo_root / "crossview_conflict" / "data" / "datasets.py"
    source.parent.mkdir(parents=True)
    source.write_text("locked source\n", encoding="utf-8")
    runtime = {
        "python": "3.10.4",
        "numpy": "1.26.1",
        "scipy": "1.15.3",
        "scikit_learn": "1.7.2",
        "torch": "2.1.0+cu118",
        "torchvision": "0.16.0+cu118",
        "cuda": "11.8",
        "cudnn": 8700,
    }
    lock_path = repo_root / "configs" / "runtime.json"
    _write_json(
        lock_path,
        {
            "schema_version": "cvian-sequence-base-runtime-lock-v1",
            "source_sha256": {
                "crossview_conflict/data/datasets.py": _sha256(source)
            },
            "runtime_versions": runtime,
        },
    )
    monkeypatch.setattr(provenance, "_current_runtime_versions", lambda: runtime)
    result = provenance.validate_supplemental_runtime_lock(
        repo_root=repo_root, lock_path=lock_path
    )
    assert set(result) == {"runtime_lock_sha256", "runtime_identity_sha256"}
    assert all(len(value) == 64 for value in result.values())

    source.write_text("tampered source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_sha256"):
        provenance.validate_supplemental_runtime_lock(
            repo_root=repo_root, lock_path=lock_path
        )


def test_runtime_lock_rejects_version_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    source = repo_root / "source.py"
    source.parent.mkdir(parents=True)
    source.write_text("source\n", encoding="utf-8")
    locked = {
        "python": "1",
        "numpy": "2",
        "scipy": "3",
        "scikit_learn": "4",
        "torch": "5",
        "torchvision": "6",
        "cuda": "7",
        "cudnn": 8,
    }
    lock_path = repo_root / "lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": "cvian-sequence-base-runtime-lock-v1",
            "source_sha256": {"source.py": _sha256(source)},
            "runtime_versions": locked,
        },
    )
    monkeypatch.setattr(
        provenance,
        "_current_runtime_versions",
        lambda: {**locked, "numpy": "different"},
    )
    with pytest.raises(ValueError, match="runtime_versions"):
        provenance.validate_supplemental_runtime_lock(
            repo_root=repo_root, lock_path=lock_path
        )


def test_seed_validator_is_hash_only_and_returns_compact_fingerprint_inputs(
    synthetic_artifacts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        np,
        "load",
        lambda *args, **kwargs: pytest.fail("provenance validation called np.load"),
    )
    result = provenance.validate_seed_base_and_embedding_provenance(
        repo_root=synthetic_artifacts["repo_root"],
        protocol_dir=synthetic_artifacts["protocol_dir"],
        base_root=synthetic_artifacts["base_root"],
        cache_root=synthetic_artifacts["cache_root"],
        seed=42,
    )
    assert result["prospective_test_cache_sha256"] == _sha256(
        synthetic_artifacts["cache_root"] / "seed42" / "prospective_test.npz"
    )
    assert result["validated_roles"] == list(provenance.ROLE_NAMES)
    assert all(
        len(value) == 64
        for key, value in result.items()
        if key != "validated_roles"
    )


def test_seed_fit_role_validation_never_touches_prospective_files(
    synthetic_artifacts: dict[str, Any],
) -> None:
    prospective_manifest = (
        synthetic_artifacts["protocol_dir"] / "prospective_test.csv"
    )
    prospective_cache = (
        synthetic_artifacts["cache_root"] / "seed42" / "prospective_test.npz"
    )
    prospective_manifest.unlink()
    prospective_cache.unlink()

    fit_roles = ("base_fit", "selector_fit", "validation")
    result = provenance.validate_seed_base_and_embedding_provenance(
        repo_root=synthetic_artifacts["repo_root"],
        protocol_dir=synthetic_artifacts["protocol_dir"],
        base_root=synthetic_artifacts["base_root"],
        cache_root=synthetic_artifacts["cache_root"],
        seed=42,
        roles=fit_roles,
    )
    assert result["validated_roles"] == list(fit_roles)
    assert "prospective_test_cache_sha256" not in result


@pytest.mark.parametrize(
    ("target", "field", "replacement", "message"),
    [
        ("completion", "role_sha256", {}, "role_sha256"),
        ("completion", "source_sha256", {}, "source_sha256"),
        ("cache", "training_attestation_required", False, "training_attestation_required"),
        ("cache", "relative_azimuth_deg", [0.0] * 8, "relative_azimuth_deg"),
        ("cache", "entrypoint_sha256", {}, "entrypoint_sha256"),
    ],
)
def test_seed_validator_rejects_missing_or_drifted_provenance(
    synthetic_artifacts: dict[str, Any],
    target: str,
    field: str,
    replacement: Any,
    message: str,
) -> None:
    path = (
        synthetic_artifacts["base_root"]
        / "crossview_seed42"
        / "training_complete.json"
        if target == "completion"
        else synthetic_artifacts["cache_root"] / "seed42" / "cache_metadata.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = replacement
    _write_json(path, payload)
    with pytest.raises(ValueError, match=message):
        provenance.validate_seed_base_and_embedding_provenance(
            repo_root=synthetic_artifacts["repo_root"],
            protocol_dir=synthetic_artifacts["protocol_dir"],
            base_root=synthetic_artifacts["base_root"],
            cache_root=synthetic_artifacts["cache_root"],
            seed=42,
        )


def test_seed_validator_rejects_artifact_hash_tampering(
    synthetic_artifacts: dict[str, Any],
) -> None:
    path = synthetic_artifacts["cache_root"] / "seed42" / "prospective_test.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="splits.prospective_test.sha256"):
        provenance.validate_seed_base_and_embedding_provenance(
            repo_root=synthetic_artifacts["repo_root"],
            protocol_dir=synthetic_artifacts["protocol_dir"],
            base_root=synthetic_artifacts["base_root"],
            cache_root=synthetic_artifacts["cache_root"],
            seed=42,
        )


def test_visibility_validator_checks_revision_and_test_hash_before_load(
    synthetic_artifacts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        np,
        "load",
        lambda *args, **kwargs: pytest.fail("provenance validation called np.load"),
    )
    result = provenance.validate_visibility_provenance(
        repo_root=synthetic_artifacts["repo_root"],
        protocol_dir=synthetic_artifacts["protocol_dir"],
        visibility_dir=synthetic_artifacts["visibility_dir"],
    )
    test_path = synthetic_artifacts["visibility_dir"] / "prospective_test.npz"
    assert result["prospective_test_visibility_sha256"] == _sha256(test_path)
    assert all(len(value) == 64 for value in result.values())

    test_path.write_bytes(test_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="splits.prospective_test.sha256"):
        provenance.validate_visibility_provenance(
            repo_root=synthetic_artifacts["repo_root"],
            protocol_dir=synthetic_artifacts["protocol_dir"],
            visibility_dir=synthetic_artifacts["visibility_dir"],
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_revision_resolved", "mutable-main"),
        ("num_sectors", 7),
        ("relative_azimuth_deg", [0.0] * 8),
        ("entrypoint_sha256", {}),
    ],
)
def test_visibility_validator_rejects_metadata_drift(
    synthetic_artifacts: dict[str, Any], field: str, replacement: Any
) -> None:
    path = synthetic_artifacts["visibility_dir"] / "visibility_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = replacement
    _write_json(path, payload)
    with pytest.raises(ValueError, match=field):
        provenance.validate_visibility_provenance(
            repo_root=synthetic_artifacts["repo_root"],
            protocol_dir=synthetic_artifacts["protocol_dir"],
            visibility_dir=synthetic_artifacts["visibility_dir"],
        )


def test_visibility_validator_rejects_registered_row_count_drift(
    synthetic_artifacts: dict[str, Any],
) -> None:
    path = synthetic_artifacts["visibility_dir"] / "visibility_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["splits"]["prospective_test"]["rows"] += 1
    _write_json(path, payload)
    with pytest.raises(ValueError, match="splits.prospective_test.rows"):
        provenance.validate_visibility_provenance(
            repo_root=synthetic_artifacts["repo_root"],
            protocol_dir=synthetic_artifacts["protocol_dir"],
            visibility_dir=synthetic_artifacts["visibility_dir"],
        )
