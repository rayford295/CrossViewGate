"""Fail-closed provenance checks for the CVIAN sequence utility experiment.

The checks in this module intentionally operate on JSON metadata and raw file
hashes only.  In particular, they never call ``numpy.load``.  They are meant to
run before any prospective-test cache is deserialized.
"""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
from typing import Any, Mapping


ROLE_NAMES = ("base_fit", "selector_fit", "validation", "prospective_test")
EXPECTED_SECTOR_IDS = tuple(range(8))
EXPECTED_RELATIVE_AZIMUTH_DEG = (
    0.0,
    45.0,
    90.0,
    135.0,
    -180.0,
    -135.0,
    -90.0,
    -45.0,
)
SEGFORMER_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
SEGFORMER_REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"

_BASE_SOURCE_FILES = {
    "orchestrator": "scripts/run_cvian_sequence_base_multiseed.py",
    "trainer": "scripts/train_triage.py",
    "training_loops": "crossview_conflict/training/loops.py",
    "triage_model": "crossview_conflict/models/triage.py",
}
_EMBEDDING_ENTRYPOINT_FILES = {
    "cache_script": "scripts/cache_cvian_active_view_embeddings.py",
    "panorama_geometry": "crossview_conflict/data/panorama.py",
    "triage_model": "crossview_conflict/models/triage.py",
}
_VISIBILITY_ENTRYPOINT_FILES = {
    "cache_script": "scripts/cache_cvian_sector_visibility.py",
    "panorama_geometry": "crossview_conflict/data/panorama.py",
}


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Required provenance artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid provenance JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require_equal(
    actual: Any,
    expected: Any,
    *,
    field: str,
    context: str,
) -> None:
    if actual != expected:
        raise ValueError(f"{context} provenance mismatch for {field}")


def _require_exact_mapping(
    actual: Any,
    expected: Mapping[str, Any],
    *,
    field: str,
    context: str,
) -> None:
    if not isinstance(actual, dict) or actual != dict(expected):
        raise ValueError(f"{context} provenance mismatch for {field}")


def _same_path(recorded: Any, expected: Path) -> bool:
    if not isinstance(recorded, str) or not recorded:
        return False
    try:
        return Path(recorded).resolve() == expected.resolve()
    except (OSError, RuntimeError):
        return False


def _hash_source_map(repo_root: Path, sources: Mapping[str, str]) -> dict[str, str]:
    root = repo_root.resolve()
    hashes: dict[str, str] = {}
    for name, relative in sources.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe registered source path: {relative}")
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Registered source escapes repository: {relative}") from error
        hashes[name] = _sha256(path)
    return hashes


def _current_runtime_versions() -> dict[str, Any]:
    package_names = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "torch": "torch",
        "torchvision": "torchvision",
    }
    versions: dict[str, Any] = {"python": platform.python_version()}
    try:
        for field, distribution in package_names.items():
            versions[field] = importlib_metadata.version(distribution)
        import torch
    except (ImportError, importlib_metadata.PackageNotFoundError) as error:
        raise ValueError("A runtime-lock dependency is not installed") from error
    versions["cuda"] = torch.version.cuda
    versions["cudnn"] = torch.backends.cudnn.version()
    return versions


def validate_supplemental_runtime_lock(
    *,
    repo_root: Path,
    lock_path: Path,
) -> dict[str, str]:
    """Validate the post-training source/runtime lock against this process.

    Returns only stable hashes so callers can bind the validated identity into
    a larger fit or evaluation fingerprint.
    """

    repo_root = repo_root.resolve()
    lock_path = lock_path.resolve()
    payload = _load_json(lock_path)
    _require_equal(
        payload.get("schema_version"),
        "cvian-sequence-base-runtime-lock-v1",
        field="schema_version",
        context="runtime lock",
    )
    recorded_sources = payload.get("source_sha256")
    if not isinstance(recorded_sources, dict) or not recorded_sources:
        raise ValueError("Runtime lock has no source_sha256 mapping")
    actual_sources = _hash_source_map(
        repo_root,
        {str(relative): str(relative) for relative in recorded_sources},
    )
    _require_exact_mapping(
        recorded_sources,
        actual_sources,
        field="source_sha256",
        context="runtime lock",
    )
    recorded_runtime = payload.get("runtime_versions")
    if not isinstance(recorded_runtime, dict) or set(recorded_runtime) != {
        "python",
        "numpy",
        "scipy",
        "scikit_learn",
        "torch",
        "torchvision",
        "cuda",
        "cudnn",
    }:
        raise ValueError("Runtime lock has an incomplete runtime_versions mapping")
    actual_runtime = _current_runtime_versions()
    _require_exact_mapping(
        recorded_runtime,
        actual_runtime,
        field="runtime_versions",
        context="runtime lock",
    )
    identity = {
        "source_sha256": actual_sources,
        "runtime_versions": actual_runtime,
    }
    return {
        "runtime_lock_sha256": _sha256(lock_path),
        "runtime_identity_sha256": _canonical_hash(identity),
    }


def _normalize_roles(roles: tuple[str, ...]) -> tuple[str, ...]:
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("Validated roles must be unique and non-empty")
    unknown = sorted(set(roles) - set(ROLE_NAMES))
    if unknown:
        raise ValueError(f"Unknown validated roles: {unknown}")
    return tuple(role for role in ROLE_NAMES if role in roles)


def _validate_protocol_registration(
    protocol_dir: Path,
    *,
    roles_to_hash: tuple[str, ...] = ROLE_NAMES,
) -> dict[str, Any]:
    protocol_dir = protocol_dir.resolve()
    validated_roles = _normalize_roles(roles_to_hash)
    summary_path = protocol_dir / "protocol_summary.json"
    commitment_path = protocol_dir / "test_commitment.json"
    summary = _load_json(summary_path)
    commitment = _load_json(commitment_path)
    _require_equal(
        summary.get("schema_version"),
        "cvian-sequence-four-role-summary-v1",
        field="schema_version",
        context="protocol summary",
    )
    _require_equal(
        commitment.get("schema_version"),
        "cvian-sequence-test-commitment-v1",
        field="schema_version",
        context="test commitment",
    )
    status = summary.get("test_status")
    if not isinstance(status, dict):
        raise ValueError("Protocol summary has no test_status object")
    expected_status = "selector_selection_holdout_with_historical_base_exposure"
    for actual, field in (
        (status.get("status"), "test_status.status"),
        (commitment.get("status"), "commitment.status"),
    ):
        _require_equal(
            actual,
            expected_status,
            field=field,
            context="registered protocol",
        )
    for actual, field in (
        (status.get("new_protocol_test_scored"), "new_protocol_test_scored"),
        (
            status.get("old_base_or_checkpoint_reuse_permitted"),
            "old_base_or_checkpoint_reuse_permitted",
        ),
        (
            commitment.get("historical_checkpoint_reuse_permitted"),
            "historical_checkpoint_reuse_permitted",
        ),
    ):
        _require_equal(actual, False, field=field, context="registered protocol")
    for field in (
        "record_overlap_zero",
        "sequence_overlap_zero",
        "spatial_block_overlap_zero",
        "spatial_buffer_clear",
    ):
        _require_equal(
            summary.get(field), True, field=field, context="registered protocol"
        )

    role_rows = summary.get("role_rows")
    if (
        not isinstance(role_rows, dict)
        or set(role_rows) != set(ROLE_NAMES)
        or any(type(role_rows[role]) is not int or role_rows[role] <= 0 for role in ROLE_NAMES)
    ):
        raise ValueError("Protocol role_rows are incomplete or invalid")
    registered = summary.get("role_manifest_sha256")
    expected_filenames = {f"{role}.csv" for role in ROLE_NAMES}
    if not isinstance(registered, dict) or set(registered) != expected_filenames:
        raise ValueError("Protocol role_manifest_sha256 is incomplete")
    role_hashes: dict[str, str] = {
        role: str(registered[f"{role}.csv"]) for role in ROLE_NAMES
    }
    validated_manifest_hashes: dict[str, str] = {}
    for role in ROLE_NAMES:
        if role not in validated_roles:
            continue
        path = protocol_dir / f"{role}.csv"
        actual = _sha256(path)
        _require_equal(
            registered.get(path.name),
            actual,
            field=f"role_manifest_sha256.{path.name}",
            context="registered protocol",
        )
        validated_manifest_hashes[role] = actual

    _require_equal(
        commitment.get("role"),
        "prospective_test",
        field="role",
        context="test commitment",
    )
    _require_equal(
        commitment.get("manifest_sha256"),
        role_hashes["prospective_test"],
        field="manifest_sha256",
        context="test commitment",
    )
    _require_equal(
        commitment.get("row_count"),
        role_rows["prospective_test"],
        field="row_count",
        context="test commitment",
    )
    summary_sha256 = _sha256(summary_path)
    commitment_sha256 = _sha256(commitment_path)
    compact = {
        "protocol_summary_sha256": summary_sha256,
        "test_commitment_sha256": commitment_sha256,
        "role_manifest_sha256": role_hashes,
        "role_rows": role_rows,
        "validated_manifest_roles": list(validated_roles),
        "validated_manifest_sha256": validated_manifest_hashes,
    }
    return {
        **compact,
        "protocol_registration_sha256": _canonical_hash(compact),
    }


def _validate_base_completion(
    *,
    repo_root: Path,
    protocol: Mapping[str, Any],
    base_dir: Path,
    seed: int,
) -> dict[str, Any]:
    completion_path = base_dir / "training_complete.json"
    completion = _load_json(completion_path)
    exact_fields = {
        "schema_version": "cvian-sequence-role-isolated-base-v1",
        "seed": seed,
        "mode": "crossview",
        "train_role": "base_fit",
        "validation_role": "validation",
        "image_size": 224,
        "class_weighting": "balanced",
        "test_evaluated": False,
        "protocol_summary_sha256": protocol["protocol_summary_sha256"],
        "test_commitment_sha256": protocol["test_commitment_sha256"],
    }
    for field, expected in exact_fields.items():
        _require_equal(
            completion.get(field), expected, field=field, context=f"base seed {seed}"
        )
    _require_exact_mapping(
        completion.get("role_sha256"),
        protocol["role_manifest_sha256"],
        field="role_sha256",
        context=f"base seed {seed}",
    )
    current_sources = _hash_source_map(repo_root, _BASE_SOURCE_FILES)
    _require_exact_mapping(
        completion.get("source_sha256"),
        current_sources,
        field="source_sha256",
        context=f"base seed {seed}",
    )

    fingerprint_payload = {
        key: value
        for key, value in completion.items()
        if key
        not in {
            "fingerprint_sha256",
            "checkpoint_sha256",
            "history_sha256",
            "train_log_sha256",
        }
    }
    _require_equal(
        completion.get("fingerprint_sha256"),
        _canonical_hash(fingerprint_payload),
        field="fingerprint_sha256",
        context=f"base seed {seed}",
    )
    artifact_fields = {
        "checkpoint_sha256": base_dir / "triage_best.pt",
        "history_sha256": base_dir / "triage_history.json",
        "train_log_sha256": base_dir / "train.log",
    }
    artifact_hashes: dict[str, str] = {}
    for field, path in artifact_fields.items():
        actual = _sha256(path)
        _require_equal(
            completion.get(field), actual, field=field, context=f"base seed {seed}"
        )
        artifact_hashes[field] = actual
    return {
        "completion": completion,
        "completion_path": completion_path,
        "completion_sha256": _sha256(completion_path),
        "source_sha256": current_sources,
        **artifact_hashes,
    }


def validate_seed_base_and_embedding_provenance(
    *,
    repo_root: Path,
    protocol_dir: Path,
    base_root: Path,
    cache_root: Path,
    seed: int,
    roles: tuple[str, ...] = ROLE_NAMES,
) -> dict[str, Any]:
    """Validate one role-isolated encoder and its complete v3 cache.

    All role manifests and NPZ files are checked by raw SHA-256.  No NPZ is
    deserialized, so this function is safe to call before the one-time test
    load.  The return value is deliberately compact for fingerprint embedding.
    """

    repo_root = repo_root.resolve()
    protocol_dir = protocol_dir.resolve()
    base_dir = base_root.resolve() / f"crossview_seed{seed}"
    seed_dir = cache_root.resolve() / f"seed{seed}"
    validated_roles = _normalize_roles(roles)
    protocol = _validate_protocol_registration(
        protocol_dir, roles_to_hash=validated_roles
    )
    base = _validate_base_completion(
        repo_root=repo_root,
        protocol=protocol,
        base_dir=base_dir,
        seed=seed,
    )
    completion = base["completion"]

    metadata_path = seed_dir / "cache_metadata.json"
    metadata = _load_json(metadata_path)
    exact_fields = {
        "schema_version": "cvian-active-view-cache-v3",
        "seed": seed,
        "checkpoint_sha256": base["checkpoint_sha256"],
        "training_attestation_required": True,
        "split_roles": list(ROLE_NAMES),
        "num_sectors": 8,
        "horizontal_fov_deg": 90.0,
        "vertical_fov_deg": 90.0,
        "image_size": 224,
        "street_backbone": "resnet18",
        "overhead_backbone": "resnet18",
        "model_mode": "crossview",
        "num_classes": 3,
        "sector_id": list(EXPECTED_SECTOR_IDS),
        "relative_azimuth_deg": list(EXPECTED_RELATIVE_AZIMUTH_DEG),
        "initial_state": "post_overhead+sector_0",
        "prospective_test_scored": False,
    }
    for field, expected in exact_fields.items():
        _require_equal(
            metadata.get(field),
            expected,
            field=field,
            context=f"embedding cache seed {seed}",
        )
    if not _same_path(metadata.get("checkpoint"), base_dir / "triage_best.pt"):
        raise ValueError(f"Embedding cache seed {seed} checkpoint path mismatch")
    if not _same_path(metadata.get("split_dir"), protocol_dir):
        raise ValueError(f"Embedding cache seed {seed} split_dir mismatch")
    _require_exact_mapping(
        metadata.get("source_manifest_sha256"),
        protocol["role_manifest_sha256"],
        field="source_manifest_sha256",
        context=f"embedding cache seed {seed}",
    )
    _require_exact_mapping(
        metadata.get("protocol_artifact_sha256"),
        {
            "protocol_summary.json": protocol["protocol_summary_sha256"],
            "test_commitment.json": protocol["test_commitment_sha256"],
        },
        field="protocol_artifact_sha256",
        context=f"embedding cache seed {seed}",
    )
    expected_entrypoints = _hash_source_map(repo_root, _EMBEDDING_ENTRYPOINT_FILES)
    _require_exact_mapping(
        metadata.get("entrypoint_sha256"),
        expected_entrypoints,
        field="entrypoint_sha256",
        context=f"embedding cache seed {seed}",
    )
    attestation = metadata.get("training_attestation")
    if not isinstance(attestation, dict):
        raise ValueError(f"Embedding cache seed {seed} has no training attestation")
    if not _same_path(attestation.get("path"), base["completion_path"]):
        raise ValueError(f"Embedding cache seed {seed} attestation path mismatch")
    _require_equal(
        attestation.get("sha256"),
        base["completion_sha256"],
        field="training_attestation.sha256",
        context=f"embedding cache seed {seed}",
    )
    _require_equal(
        attestation.get("fingerprint_sha256"),
        completion.get("fingerprint_sha256"),
        field="training_attestation.fingerprint_sha256",
        context=f"embedding cache seed {seed}",
    )

    split_metadata = metadata.get("splits")
    if not isinstance(split_metadata, dict) or set(split_metadata) != set(ROLE_NAMES):
        raise ValueError(f"Embedding cache seed {seed} has incomplete split metadata")
    cache_hashes: dict[str, str] = {}
    for role in validated_roles:
        entry = split_metadata.get(role)
        if not isinstance(entry, dict):
            raise ValueError(f"Embedding cache seed {seed} has invalid {role} metadata")
        path = seed_dir / f"{role}.npz"
        actual = _sha256(path)
        expected_entry = {
            "rows": protocol["role_rows"][role],
            "sectors_per_row": 8,
            "embedding_dim": 256,
            "classes": 3,
            "sha256": actual,
        }
        for field, expected in expected_entry.items():
            _require_equal(
                entry.get(field),
                expected,
                field=f"splits.{role}.{field}",
                context=f"embedding cache seed {seed}",
            )
        if not _same_path(entry.get("output"), path):
            raise ValueError(
                f"Embedding cache seed {seed} output path mismatch for {role}"
            )
        cache_hashes[role] = actual

    compact_payload = {
        "seed": seed,
        "validated_roles": list(validated_roles),
        "protocol_registration_sha256": protocol["protocol_registration_sha256"],
        "base_completion_sha256": base["completion_sha256"],
        "base_fingerprint_sha256": completion["fingerprint_sha256"],
        "checkpoint_sha256": base["checkpoint_sha256"],
        "cache_metadata_sha256": _sha256(metadata_path),
        "cache_artifact_sha256": cache_hashes,
    }
    result: dict[str, Any] = {
        "validated_roles": list(validated_roles),
        "protocol_registration_sha256": protocol["protocol_registration_sha256"],
        "base_completion_sha256": base["completion_sha256"],
        "base_fingerprint_sha256": completion["fingerprint_sha256"],
        "checkpoint_sha256": base["checkpoint_sha256"],
        "embedding_cache_metadata_sha256": compact_payload["cache_metadata_sha256"],
        "embedding_cache_artifacts_sha256": _canonical_hash(cache_hashes),
        "seed_input_provenance_sha256": _canonical_hash(compact_payload),
    }
    if "prospective_test" in cache_hashes:
        result["prospective_test_cache_sha256"] = cache_hashes["prospective_test"]
    return result


def validate_visibility_provenance(
    *,
    repo_root: Path,
    protocol_dir: Path,
    visibility_dir: Path,
) -> dict[str, str]:
    """Validate the required privileged visibility cache without loading it."""

    repo_root = repo_root.resolve()
    protocol_dir = protocol_dir.resolve()
    visibility_dir = visibility_dir.resolve()
    protocol = _validate_protocol_registration(protocol_dir)
    metadata_path = visibility_dir / "visibility_metadata.json"
    metadata = _load_json(metadata_path)
    exact_fields = {
        "schema_version": "cvian-sector-visibility-v3",
        "model_id": SEGFORMER_MODEL_ID,
        "model_revision_requested": None,
        "model_revision_resolved": SEGFORMER_REVISION,
        "building_class_id": 1,
        "split_roles": list(ROLE_NAMES),
        "num_sectors": 8,
        "horizontal_fov_deg": 90.0,
        "vertical_fov_deg": 90.0,
        "sector_id": list(EXPECTED_SECTOR_IDS),
        "relative_azimuth_deg": list(EXPECTED_RELATIVE_AZIMUTH_DEG),
        "prospective_test_scored": False,
        "policy_scope": "privileged_offline_baseline_only",
    }
    for field, expected in exact_fields.items():
        _require_equal(
            metadata.get(field), expected, field=field, context="visibility cache"
        )
    if not _same_path(metadata.get("split_dir"), protocol_dir):
        raise ValueError("Visibility cache split_dir mismatch")
    _require_exact_mapping(
        metadata.get("source_manifest_sha256"),
        protocol["role_manifest_sha256"],
        field="source_manifest_sha256",
        context="visibility cache",
    )
    _require_exact_mapping(
        metadata.get("protocol_artifact_sha256"),
        {
            "protocol_summary.json": protocol["protocol_summary_sha256"],
            "test_commitment.json": protocol["test_commitment_sha256"],
        },
        field="protocol_artifact_sha256",
        context="visibility cache",
    )
    expected_entrypoints = _hash_source_map(repo_root, _VISIBILITY_ENTRYPOINT_FILES)
    _require_exact_mapping(
        metadata.get("entrypoint_sha256"),
        expected_entrypoints,
        field="entrypoint_sha256",
        context="visibility cache",
    )

    split_metadata = metadata.get("splits")
    if not isinstance(split_metadata, dict) or set(split_metadata) != set(ROLE_NAMES):
        raise ValueError("Visibility cache has incomplete split metadata")
    cache_hashes: dict[str, str] = {}
    for role in ROLE_NAMES:
        entry = split_metadata.get(role)
        if not isinstance(entry, dict):
            raise ValueError(f"Visibility cache has invalid {role} metadata")
        path = visibility_dir / f"{role}.npz"
        actual = _sha256(path)
        expected_entry = {
            "rows": protocol["role_rows"][role],
            "sectors_per_row": 8,
            "sha256": actual,
        }
        for field, expected in expected_entry.items():
            _require_equal(
                entry.get(field),
                expected,
                field=f"splits.{role}.{field}",
                context="visibility cache",
            )
        no_building = entry.get("no_building_panorama_count")
        if type(no_building) is not int or not 0 <= no_building <= expected_entry["rows"]:
            raise ValueError(
                f"Visibility cache invalid no_building_panorama_count for {role}"
            )
        if not _same_path(entry.get("output"), path):
            raise ValueError(f"Visibility cache output path mismatch for {role}")
        cache_hashes[role] = actual

    compact_payload = {
        "protocol_registration_sha256": protocol["protocol_registration_sha256"],
        "visibility_metadata_sha256": _sha256(metadata_path),
        "visibility_artifact_sha256": cache_hashes,
        "segformer_revision": SEGFORMER_REVISION,
    }
    return {
        "protocol_registration_sha256": protocol["protocol_registration_sha256"],
        "visibility_metadata_sha256": compact_payload["visibility_metadata_sha256"],
        "visibility_artifacts_sha256": _canonical_hash(cache_hashes),
        "prospective_test_visibility_sha256": cache_hashes["prospective_test"],
        "visibility_provenance_sha256": _canonical_hash(compact_payload),
    }
