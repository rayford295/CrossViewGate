"""Fit the pre-registered CVIAN relative-geometry utility transfer policy.

Only the CVIAN ``selector_fit`` and ``validation`` manifests/caches are opened.
The classifier, its standardizer, and its temperature are read unchanged from
the already-attested CVIAN utility fit.  Milton is not loaded, scored, fitted,
calibrated, or used for model selection by this entrypoint.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.decision.relative_geometry_utility import (
    RELATIVE_GEOMETRY_FEATURE_NAMES,
    RELATIVE_GEOMETRY_FEATURE_SCHEMA,
    REGISTERED_RELATIVE_AZIMUTH_DEG,
    build_relative_geometry_utility_features,
)
from crossview_conflict.decision.active_view import validate_active_view_cache
from scripts import run_cvian_sequence_utility_experiment as frozen_runner


CONFIG_SCHEMA = "milton-zero-shot-active-view-sensitivity-v1"
ARTIFACT_SCHEMA = "cvian-relative-geometry-utility-v1"
FIT_SCHEMA = "cvian-relative-geometry-utility-fit-v1"
SUMMARY_SCHEMA = "cvian-relative-geometry-utility-summary-v1"
SEEDS = (42, 123, 456, 789, 1011)
FIT_ROLES = ("selector_fit", "validation")
ORIGINS = tuple(range(8))


@dataclass(frozen=True)
class RelativeUtilityData:
    table: frozen_runner.UtilityTable
    row_origin: np.ndarray
    state_origin_by_state: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/milton_zero_shot_active_view_sensitivity_v1.json"
    )
    parser.add_argument(
        "--protocol-dir", default="data/splits/ian_hurricane_sequence_four_role_v1"
    )
    parser.add_argument(
        "--cache-root", default="outputs/cvian_sequence_active_v2/cache"
    )
    parser.add_argument(
        "--source-fit-root",
        default="outputs/cvian_sequence_active_v2/utility_experiment",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/cvian_sequence_active_v2/milton_relative_policy",
    )
    parser.add_argument("--utility-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
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


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("Unsupported Milton sensitivity config")
    if config.get("claim_scope") != "Milton zero-shot sensitivity only; not confirmatory":
        raise ValueError("Milton claim scope changed")
    if config.get("confirmatory_eligible") is not False:
        raise ValueError("Milton must remain non-confirmatory")
    if tuple(config.get("seeds", ())) != SEEDS:
        raise ValueError("Registered model seeds changed")
    if config.get("sector_count") != 8 or tuple(
        config.get("cyclic_origin_rotations", ())
    ) != ORIGINS:
        raise ValueError("Registered cyclic origins changed")
    if config.get("fixed_budget_views") != 3:
        raise ValueError("Registered fixed view budget changed")
    if config.get("fixed_budget_views_semantics") != (
        "total revealed views including the rolled-local initial sector 0"
    ):
        raise ValueError("Registered fixed-budget semantics changed")
    if config.get("view_cost") != 0.5 or config.get("defer_human_cost") != 1.5:
        raise ValueError("Registered operational costs changed")
    if config.get("view_cost_semantics") != (
        "0.5 per additional reveal; the initial view has zero acquisition cost"
    ):
        raise ValueError("Registered view-cost semantics changed")
    expected_cost = np.asarray([[0, 1, 4], [1, 0, 1], [8, 8, 0]], dtype=float)
    if not np.array_equal(np.asarray(config.get("cost_matrix"), dtype=float), expected_cost):
        raise ValueError("Registered cost matrix changed")

    source = config.get("source_contract", {})
    expected_source = {
        "manifest_contract_sha256": "505e1d746553cc5258fbddcf995e3b9564a4bbe62ff7d65a04304daae6dbc6e0",
        "transfer_manifest_sha256": "c141c1ffcce86aefccd3a544352161e07133ab81f152e4f946ad9177053a277f",
        "canonical_source_cohort_sha256": "e401e9a8015aa08f3c5d6bbad5bc132a28de288734b0899e976b5e1e43fc4bf9",
        "cache_summary_sha256": "a5af66d5215f1c4c6c4b2a4332cdb671e95bfa73afc4a486d21cc10c14f05233",
        "cache_entrypoint_sha256": "b04d2beb7c4af593496293bcd19c99647be68ff2b059592acb8f5fa2bf5622d8",
        "manifest_builder_sha256": "46b2557059e29bbd2a3bb2588ca53309e7ce827ae1836d0b6b9707018b8bc923",
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"Frozen Milton source field {key!r} changed")

    fit = config.get("cvian_relative_policy_fit", {})
    expected_fit = {
        "permitted_roles": ["selector_fit", "validation"],
        "forbidden_roles": ["base_fit", "prospective_test"],
        "classifier_updates": False,
        "base_encoder_updates": False,
        "temperature_updates": False,
        "absolute_geometry_inputs": False,
        "feature_schema": RELATIVE_GEOMETRY_FEATURE_SCHEMA,
        "feature_width": len(RELATIVE_GEOMETRY_FEATURE_NAMES),
        "remaining_acquisitions_feature": "(4 - revealed_view_count) / 8",
        "azimuth_geometry_exact_deg": REGISTERED_RELATIVE_AZIMUTH_DEG.tolist(),
        "nearest_revealed_angular_distance_normalization_deg": 180.0,
        "entropy_log_base": "natural",
        "canonical_sector_one_hot_forbidden": True,
        "compass_and_absolute_yaw_forbidden": True,
        "hidden_candidate_image_embedding_or_logit_forbidden": True,
        "loss": "sample-k-balanced smooth_l1",
        "early_stopping_metric": "validation sequence-macro top-1 utility regret",
        "utility_epochs": 60,
        "patience": 7,
        "batch_size": 512,
        "learning_rate": 0.001,
        "num_workers": 0,
        "device": "cuda",
        "cublas_workspace_config": ":4096:8",
    }
    for key, expected in expected_fit.items():
        if fit.get(key) != expected:
            raise ValueError(f"Relative-policy fit field {key!r} changed")
    if fit.get("origin_augmentation") != (
        "for physical origin r, cyclically roll every sector-indexed cache tensor "
        "so physical r maps to local sector 0; enumerate classifier states only "
        "in the rolled local frame"
    ) or fit.get("state_enumeration") != (
        "all unique rolled-local subsets containing local sector 0 for k=1..3"
    ):
        raise ValueError("Rolled-local origin/state contract changed")
    if fit.get("feature_contract") != [
        "current frozen calibrated class probabilities",
        "confidence margin and entropy",
        "revealed-sector mask relative to initial sector",
        "normalized revealed count",
        "relative revealed-angle mean sine and cosine",
        "normalized remaining acquisitions",
        "candidate sector one-hot relative to initial sector",
        "candidate relative-yaw sine and cosine",
        "nearest revealed angular distance",
    ]:
        raise ValueError("Relative-policy feature contract changed")

    evaluation = config.get("evaluation", {})
    if evaluation.get("mode") != "frozen_zero_shot_forward_only":
        raise ValueError("Milton evaluation mode changed")
    for field in (
        "milton_parameter_fitting",
        "milton_calibration",
        "milton_model_selection",
        "milton_policy_selection",
        "stop_or_adaptive_policy_enabled",
    ):
        if evaluation.get(field) is not False:
            raise ValueError(f"Milton operation {field!r} must remain disabled")
    if evaluation.get("go_no_go_criterion") is not None:
        raise ValueError("Milton sensitivity cannot have a GO criterion")
    if evaluation.get("main_policy") != "relative_geometry_utility":
        raise ValueError("Registered main Milton policy changed")
    if evaluation.get("comparators") != ["farthest", "clockwise", "random_mc32"]:
        raise ValueError("Registered Milton comparators changed")
    if evaluation.get("ood_diagnostics") != [
        "absolute_aware_utility_ood_diagnostic"
    ] or evaluation.get("privileged_diagnostics") != [
        "max_confidence_privileged",
        "greedy_label_oracle_privileged",
    ]:
        raise ValueError("Registered Milton diagnostics changed")
    bootstrap = evaluation.get("bootstrap", {})
    if bootstrap.get("purpose") != "descriptive uncertainty only; never a GO test":
        raise ValueError("Milton bootstrap must remain descriptive")

    implementation = config.get("implementation_contract", {})
    expected_implementation_paths = {
        "fit_entrypoint_sha256": Path(__file__).resolve(),
        "feature_interface_sha256": (
            REPO_ROOT
            / "crossview_conflict"
            / "decision"
            / "relative_geometry_utility.py"
        ),
        "frozen_cvian_runner_sha256": (
            REPO_ROOT / "scripts" / "run_cvian_sequence_utility_experiment.py"
        ),
        "frozen_active_view_sha256": (
            REPO_ROOT / "crossview_conflict" / "decision" / "active_view.py"
        ),
        "frozen_active_view_utility_sha256": (
            REPO_ROOT
            / "crossview_conflict"
            / "decision"
            / "active_view_utility.py"
        ),
    }
    for field, path in expected_implementation_paths.items():
        if implementation.get(field) != _sha256(path):
            raise ValueError(f"Registered implementation hash {field!r} changed")
    if implementation.get("attestation_chain") != (
        "config hash + implementation hashes + source completion/artifact/cache/manifest "
        "hashes -> per-seed fit fingerprint -> relative artifact hash -> fit completion "
        "hash -> summary"
    ):
        raise ValueError("Relative-policy artifact attestation chain changed")


def _validate_args(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    fit = config["cvian_relative_policy_fit"]
    expected = {
        "utility_epochs": args.utility_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "device": args.device,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    registered = {key: fit[key] for key in expected}
    if expected != registered:
        raise ValueError("CLI hyperparameters do not match the registered config")
    locked_paths = {
        _resolve(args.protocol_dir): _resolve(fit["source_protocol_dir"]),
        _resolve(args.cache_root): _resolve(fit["source_cache_root"]),
        _resolve(args.source_fit_root): _resolve(fit["source_fit_root"]),
        _resolve(args.output_root): _resolve(fit["output_root"]),
    }
    if any(actual != expected_path for actual, expected_path in locked_paths.items()):
        raise ValueError("CLI paths do not match the registered config")


def _read_fit_manifests(protocol_dir: Path) -> dict[str, pd.DataFrame]:
    """Read exactly the two registered CVIAN policy-fit roles."""
    return {
        role: pd.read_csv(
            protocol_dir / f"{role}.csv",
            dtype={
                "sample_id": str,
                "sequence_id": str,
                "spatial_block_id": str,
            },
            usecols=[
                "sample_id",
                "sequence_id",
                "spatial_block_id",
                "label",
                "latitude",
                "longitude",
                "compass_angle_deg",
            ],
        ).reset_index(drop=True)
        for role in FIT_ROLES
    }


def _load_fit_roles(
    cache_root: Path, seed: int, manifests: Mapping[str, pd.DataFrame]
) -> dict[str, frozen_runner.RoleData]:
    """Load exactly selector_fit and validation, then remove absolute geometry."""
    roles: dict[str, frozen_runner.RoleData] = {}
    for role in FIT_ROLES:
        loaded = frozen_runner._load_role_data(
            cache_root / f"seed{seed}" / f"{role}.npz", manifests[role]
        )
        roles[role] = frozen_runner.RoleData(
            cache=loaded.cache,
            compass_angle_deg=np.full(loaded.cache.sample_count, np.nan),
            compass_available=np.zeros(loaded.cache.sample_count, dtype=bool),
        )
    return roles


def _load_frozen_classifier(
    source_fit_root: Path, seed: int, *, device: str
) -> tuple[frozen_runner.ClassifierResult, float, dict[str, Any]]:
    seed_dir = source_fit_root / f"seed{seed}"
    completion_path = seed_dir / "fit_complete.json"
    artifact_path = seed_dir / "fit_artifact.pt"
    completion = _load_json(completion_path)
    completion_sha = _sha256(completion_path)
    artifact_sha = _sha256(artifact_path)
    if completion.get("schema_version") != frozen_runner.FIT_SCHEMA:
        raise ValueError(f"Unsupported source CVIAN fit completion: {completion_path}")
    if completion.get("fit_artifact_sha256") != artifact_sha:
        raise ValueError(f"Source CVIAN fit artifact hash mismatch: {artifact_path}")
    if completion.get("prospective_test_loaded") is not False:
        raise ValueError("Source classifier fit must attest no prospective-test load")
    payload = torch.load(artifact_path, map_location=device, weights_only=False)
    if payload.get("schema_version") != frozen_runner.POLICY_SCHEMA:
        raise ValueError(f"Unsupported source CVIAN fit artifact: {artifact_path}")
    metadata = payload.get("metadata", {})
    if metadata.get("prospective_test_loaded") is not False:
        raise ValueError("Source classifier metadata does not attest fit-role isolation")
    if metadata.get("fit_fingerprint_sha256") != completion.get(
        "fit_fingerprint_sha256"
    ):
        raise ValueError("Source artifact metadata/completion fingerprint mismatch")
    classifier = frozen_runner.ActiveViewMLP(
        int(payload["classifier_input_dim"]), 3
    ).to(device)
    classifier.load_state_dict(payload["classifier_state_dict"])
    classifier.eval()
    classifier.requires_grad_(False)
    result = frozen_runner.ClassifierResult(
        model=classifier,
        standardizer=frozen_runner.Standardizer(
            mean=np.asarray(payload["classifier_mean"], dtype=np.float32),
            scale=np.asarray(payload["classifier_scale"], dtype=np.float32),
        ),
        best_epoch=int(metadata["classifier_best_epoch"]),
        best_validation_nll=float(metadata["classifier_validation_nll"]),
        history=[],
    )
    temperature = float(payload["temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Frozen classifier temperature is invalid")
    provenance = {
        "fit_completion_sha256": completion_sha,
        "fit_artifact_sha256": artifact_sha,
        "fit_fingerprint_sha256": completion["fit_fingerprint_sha256"],
        "classifier_input_dim": int(payload["classifier_input_dim"]),
        "classifier_best_epoch": int(metadata["classifier_best_epoch"]),
        "classifier_validation_nll": float(metadata["classifier_validation_nll"]),
        "temperature": temperature,
        "classifier_parameters_updated": False,
        "temperature_updated": False,
        "protocol_summary_sha256": completion["protocol_summary_sha256"],
        "attested_fit_manifest_sha256": {
            role: completion["fit_manifest_sha256"][role] for role in FIT_ROLES
        },
        "attested_cache_sha256": {
            role: completion["input_provenance"]["cache_sha256"][role]
            for role in FIT_ROLES
        },
        "attested_cache_metadata_sha256": completion["input_provenance"][
            "cache_metadata_sha256"
        ],
    }
    return result, temperature, provenance


def _candidate_rows(
    states: frozen_runner.StateTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parent_rows: list[int] = []
    candidate_sectors: list[int] = []
    updated_masks: list[np.ndarray] = []
    for state_id, mask in enumerate(states.revealed_mask):
        for candidate in np.flatnonzero(~mask):
            updated = mask.copy()
            updated[int(candidate)] = True
            parent_rows.append(state_id)
            candidate_sectors.append(int(candidate))
            updated_masks.append(updated)
    return (
        np.asarray(parent_rows, dtype=np.int64),
        np.asarray(candidate_sectors, dtype=np.int64),
        np.stack(updated_masks),
    )


def roll_cache_to_local_origin(
    cache: frozen_runner.ActiveViewCache, physical_origin: int
) -> frozen_runner.ActiveViewCache:
    """Roll sector-indexed content so ``physical_origin`` becomes local zero."""
    sector_count = cache.sector_count
    if sector_count != 8 or not 0 <= physical_origin < sector_count:
        raise ValueError("The registered origin roll requires one of eight sectors")
    if not np.array_equal(cache.relative_azimuth_deg, REGISTERED_RELATIVE_AZIMUTH_DEG):
        raise ValueError("Cache does not use the registered 45-degree geometry")
    physical_order = (
        np.arange(sector_count, dtype=np.int64) + int(physical_origin)
    ) % sector_count
    rolled = frozen_runner.ActiveViewCache(
        sample_id=cache.sample_id,
        spatial_block_id=cache.spatial_block_id,
        sequence_id=cache.sequence_id,
        target=cache.target,
        latitude=cache.latitude,
        longitude=cache.longitude,
        sector_id=np.arange(sector_count, dtype=np.int64),
        relative_azimuth_deg=REGISTERED_RELATIVE_AZIMUTH_DEG.copy(),
        street_embedding=np.take(cache.street_embedding, physical_order, axis=1),
        overhead_embedding=cache.overhead_embedding,
        sector_logits=np.take(cache.sector_logits, physical_order, axis=1),
        panorama_logits=cache.panorama_logits,
    )
    validate_active_view_cache(rolled)
    return rolled


def _build_relative_utility_table(
    role: frozen_runner.RoleData,
    classifier: frozen_runner.ClassifierResult,
    *,
    temperature: float,
    view_cost: float,
    cost_matrix: np.ndarray,
    device: str,
) -> RelativeUtilityData:
    source_cache = role.cache
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    sequence_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    candidate_parts: list[np.ndarray] = []
    row_origins: list[np.ndarray] = []
    state_origins: list[np.ndarray] = []
    state_offset = 0

    for origin in ORIGINS:
        cache = roll_cache_to_local_origin(source_cache, origin)
        states = frozen_runner.enumerate_subset_states(
            cache,
            maximum_revealed=3,
            maximum_adaptive_views=4,
            initial_sector_id=0,
        )
        state_features = frozen_runner._state_features(cache, states)
        current_logits = frozen_runner._predict(
            classifier.model,
            classifier.standardizer,
            state_features,
            device=device,
        )
        current_probabilities = frozen_runner.softmax(current_logits, temperature)
        parent, candidates, updated_masks = _candidate_rows(states)
        after_states = frozen_runner.StateTable(
            sample_indices=states.sample_indices[parent],
            revealed_mask=updated_masks,
            revealed_count=states.revealed_count[parent] + 1,
            remaining_acquisitions=np.maximum(
                states.remaining_acquisitions[parent] - 1, 0
            ).astype(np.float32),
        )
        after_features = frozen_runner._state_features(cache, after_states)
        after_logits = frozen_runner._predict(
            classifier.model,
            classifier.standardizer,
            after_features,
            device=device,
        )
        after_probabilities = frozen_runner.softmax(after_logits, temperature)
        sample_index = states.sample_indices[parent]
        labels = cache.target[sample_index]
        before_loss = frozen_runner.target_conditioned_soft_loss(
            current_probabilities[parent], labels, cost_matrix
        )
        after_loss = frozen_runner.target_conditioned_soft_loss(
            after_probabilities, labels, cost_matrix
        )
        targets = (before_loss - after_loss - view_cost).astype(np.float32)
        features = build_relative_geometry_utility_features(
            current_probabilities[parent],
            states.revealed_mask[parent],
            candidates,
            cache.relative_azimuth_deg,
            origin_sector=0,
            remaining_acquisitions=states.remaining_acquisitions[parent],
        )
        feature_parts.append(features)
        target_parts.append(targets)
        state_parts.append(parent + state_offset)
        sample_parts.append(sample_index)
        sequence_parts.append(cache.sequence_id[sample_index])
        count_parts.append(states.revealed_count[parent])
        candidate_parts.append(candidates)
        row_origins.append(np.full(len(parent), origin, dtype=np.int8))
        state_origins.append(np.full(len(states.sample_indices), origin, dtype=np.int8))
        state_offset += len(states.sample_indices)

    sample_index = np.concatenate(sample_parts)
    revealed_count = np.concatenate(count_parts)
    table = frozen_runner.UtilityTable(
        features=np.concatenate(feature_parts),
        targets=np.concatenate(target_parts),
        weights=frozen_runner._balanced_candidate_weights(
            sample_index, revealed_count
        ),
        state_id=np.concatenate(state_parts),
        sample_index=sample_index,
        sequence_id=np.concatenate(sequence_parts),
        revealed_count=revealed_count,
        candidate_sector=np.concatenate(candidate_parts),
    )
    if table.features.shape[1] != len(RELATIVE_GEOMETRY_FEATURE_NAMES):
        raise ValueError("Relative feature width does not match the registered interface")
    row_origin = np.concatenate(row_origins)
    state_origin_by_state = np.concatenate(state_origins)
    if len(row_origin) != len(table.features):
        raise RuntimeError("Candidate-row origin mapping is not row-aligned")
    if len(state_origin_by_state) != state_offset:
        raise RuntimeError("State-origin lookup is not state-aligned")
    if not np.array_equal(row_origin, state_origin_by_state[table.state_id]):
        raise RuntimeError("Candidate-row origins do not match their parent states")
    return RelativeUtilityData(
        table=table,
        row_origin=row_origin,
        state_origin_by_state=state_origin_by_state,
    )


def _source_provenance(
    *,
    seed: int,
    protocol_dir: Path,
    cache_root: Path,
    classifier_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    protocol_summary_path = protocol_dir / "protocol_summary.json"
    protocol_summary = _load_json(protocol_summary_path)
    if _sha256(protocol_summary_path) != classifier_provenance[
        "protocol_summary_sha256"
    ]:
        raise ValueError("Protocol summary hash differs from the source fit attestation")
    role_hashes: dict[str, dict[str, str]] = {}
    for role in FIT_ROLES:
        manifest_path = protocol_dir / f"{role}.csv"
        cache_path = cache_root / f"seed{seed}" / f"{role}.npz"
        manifest_sha = _sha256(manifest_path)
        cache_sha = _sha256(cache_path)
        expected_manifest = protocol_summary["role_manifest_sha256"][f"{role}.csv"]
        if manifest_sha != expected_manifest or manifest_sha != classifier_provenance[
            "attested_fit_manifest_sha256"
        ][role]:
            raise ValueError(f"{role} manifest is not the attested source manifest")
        if cache_sha != classifier_provenance["attested_cache_sha256"][role]:
            raise ValueError(f"{role} cache is not the cache attested by the source fit")
        role_hashes[role] = {
            "manifest_sha256": manifest_sha,
            "cache_sha256": cache_sha,
        }
    metadata_path = cache_root / f"seed{seed}" / "cache_metadata.json"
    metadata_sha = _sha256(metadata_path)
    if metadata_sha != classifier_provenance["attested_cache_metadata_sha256"]:
        raise ValueError("Cache metadata differs from the source fit attestation")
    cache_metadata = _load_json(metadata_path)
    if cache_metadata.get("schema_version") != "cvian-active-view-cache-v3":
        raise ValueError("Unsupported CVIAN cache metadata schema")
    for role in FIT_ROLES:
        if cache_metadata["source_manifest_sha256"][role] != role_hashes[role][
            "manifest_sha256"
        ] or cache_metadata["splits"][role]["sha256"] != role_hashes[role][
            "cache_sha256"
        ]:
            raise ValueError(f"Cache metadata provenance mismatch for {role}")
    return {
        "roles_loaded": list(FIT_ROLES),
        "forbidden_roles_loaded": [],
        "prospective_test_loaded": False,
        "milton_manifest_or_cache_loaded": False,
        "compass_angles_forced_nan": True,
        "compass_availability_forced_false": True,
        "role_hashes": role_hashes,
        "protocol_summary_sha256": _sha256(protocol_summary_path),
        "cache_metadata_sha256": metadata_sha,
        "frozen_classifier": dict(classifier_provenance),
    }


def _fit_fingerprint(
    *,
    args: argparse.Namespace,
    seed: int,
    config_path: Path,
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": FIT_SCHEMA,
        "seed": seed,
        "claim_scope": "CVIAN-only relative-policy fit for Milton sensitivity",
        "config_sha256": _sha256(config_path),
        "source_provenance": source_provenance,
        "source_sha256": {
            "fit_entrypoint": _sha256(Path(__file__).resolve()),
            "feature_interface": _sha256(
                REPO_ROOT
                / "crossview_conflict"
                / "decision"
                / "relative_geometry_utility.py"
            ),
            "frozen_cvian_runner": _sha256(
                REPO_ROOT / "scripts" / "run_cvian_sequence_utility_experiment.py"
            ),
            "frozen_active_view": _sha256(
                REPO_ROOT / "crossview_conflict" / "decision" / "active_view.py"
            ),
            "frozen_active_view_utility": _sha256(
                REPO_ROOT
                / "crossview_conflict"
                / "decision"
                / "active_view_utility.py"
            ),
        },
        "feature_schema": RELATIVE_GEOMETRY_FEATURE_SCHEMA,
        "feature_names": list(RELATIVE_GEOMETRY_FEATURE_NAMES),
        "feature_width": len(RELATIVE_GEOMETRY_FEATURE_NAMES),
        "origin_rotations": list(ORIGINS),
        "target": "target-conditioned soft loss before minus after minus 0.5",
        "hyperparameters": {
            "utility_epochs": args.utility_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_workers": args.num_workers,
            "device": args.device,
            "seed": seed + 97,
            "loss": "sample-k-balanced smooth_l1",
            "early_stopping": "validation sequence-macro top-1 utility regret",
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "classifier_parameters_updated": False,
        "base_encoder_parameters_updated": False,
        "temperature_updated": False,
        "milton_data_loaded": False,
        "prospective_test_loaded": False,
    }
    payload["fit_fingerprint_sha256"] = _canonical_hash(payload)
    return payload


def _validate_existing_completion(
    seed_dir: Path, completion: Mapping[str, Any], fingerprint: Mapping[str, Any]
) -> None:
    if completion.get("schema_version") != FIT_SCHEMA:
        raise ValueError(f"Unsupported existing relative fit: {seed_dir}")
    if completion.get("fit_fingerprint_sha256") != fingerprint.get(
        "fit_fingerprint_sha256"
    ):
        raise ValueError(f"Relative fit fingerprint mismatch: {seed_dir}")
    for field, expected in fingerprint.items():
        if completion.get(field) != expected:
            raise ValueError(f"Relative fit completion field {field!r} drifted")
    for filename, field in (
        ("relative_utility_artifact.pt", "artifact_sha256"),
        ("training_history.json", "history_sha256"),
        ("validation_top1_regret.csv", "validation_regret_sha256"),
    ):
        path = seed_dir / filename
        if not path.is_file() or _sha256(path) != completion.get(field):
            raise ValueError(f"Relative fit output hash mismatch: {path}")
    artifact = torch.load(
        seed_dir / "relative_utility_artifact.pt",
        map_location="cpu",
        weights_only=False,
    )
    if artifact.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError(f"Relative fit artifact schema mismatch: {seed_dir}")
    if artifact.get("utility_input_dim") != len(RELATIVE_GEOMETRY_FEATURE_NAMES):
        raise ValueError(f"Relative fit artifact input width mismatch: {seed_dir}")
    if np.asarray(artifact.get("utility_mean")).shape != (
        len(RELATIVE_GEOMETRY_FEATURE_NAMES),
    ) or np.asarray(artifact.get("utility_scale")).shape != (
        len(RELATIVE_GEOMETRY_FEATURE_NAMES),
    ):
        raise ValueError(f"Relative fit artifact standardizer mismatch: {seed_dir}")
    validation_model = frozen_runner.ActiveViewMLP(
        len(RELATIVE_GEOMETRY_FEATURE_NAMES), 1
    )
    validation_model.load_state_dict(artifact["utility_state_dict"])
    metadata = artifact.get("metadata", {})
    for field, expected in fingerprint.items():
        if metadata.get(field) != expected:
            raise ValueError(f"Relative fit artifact metadata field {field!r} drifted")


def _fit_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    config: Mapping[str, Any],
    config_path: Path,
    protocol_dir: Path,
    cache_root: Path,
    source_fit_root: Path,
    output_root: Path,
    manifests: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    classifier, temperature, classifier_provenance = _load_frozen_classifier(
        source_fit_root, seed, device=args.device
    )
    provenance = _source_provenance(
        seed=seed,
        protocol_dir=protocol_dir,
        cache_root=cache_root,
        classifier_provenance=classifier_provenance,
    )
    fingerprint = _fit_fingerprint(
        args=args,
        seed=seed,
        config_path=config_path,
        source_provenance=provenance,
    )
    seed_dir = output_root / f"seed{seed}"
    completion_path = seed_dir / "fit_complete.json"
    if completion_path.is_file():
        completion = _load_json(completion_path)
        _validate_existing_completion(seed_dir, completion, fingerprint)
        return completion
    protected = (
        "relative_utility_artifact.pt",
        "training_history.json",
        "validation_top1_regret.csv",
    )
    if any((seed_dir / name).exists() for name in protected):
        raise ValueError(f"Unattested partial relative fit exists: {seed_dir}")

    roles = _load_fit_roles(cache_root, seed, manifests)
    cost_matrix = np.asarray(config["cost_matrix"], dtype=np.float64)
    selector_data = _build_relative_utility_table(
        roles["selector_fit"],
        classifier,
        temperature=temperature,
        view_cost=float(config["view_cost"]),
        cost_matrix=cost_matrix,
        device=args.device,
    )
    validation_data = _build_relative_utility_table(
        roles["validation"],
        classifier,
        temperature=temperature,
        view_cost=float(config["view_cost"]),
        cost_matrix=cost_matrix,
        device=args.device,
    )
    regressor = frozen_runner._train_utility_regressor(
        selector_data.table,
        validation_data.table,
        epochs=args.utility_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=seed + 97,
        device=args.device,
        num_workers=args.num_workers,
    )
    validation_prediction = frozen_runner._predict(
        regressor.model,
        regressor.standardizer,
        validation_data.table.features,
        device=args.device,
    ).reshape(-1)
    validation_regret, validation_diagnostics = (
        frozen_runner._sequence_macro_top1_regret(
            validation_prediction, validation_data.table
        )
    )
    validation_diagnostics.insert(
        1,
        "origin_rotation",
        validation_data.state_origin_by_state[
            validation_diagnostics["state_id"].to_numpy(dtype=np.int64)
        ],
    )

    seed_dir.mkdir(parents=True, exist_ok=True)
    validation_diagnostics.to_csv(
        seed_dir / "validation_top1_regret.csv", index=False
    )
    (seed_dir / "training_history.json").write_text(
        json.dumps(
            {
                "schema_version": FIT_SCHEMA,
                "utility_regressor": regressor.history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_metadata = {
        **fingerprint,
        "utility_model_class": "ActiveViewMLP",
        "utility_hidden_dim": 256,
        "utility_dropout": 0.15,
        "utility_best_epoch": regressor.best_epoch,
        "validation_sequence_macro_top1_utility_regret": validation_regret,
        "selector_fit_candidate_rows": len(selector_data.table.targets),
        "validation_candidate_rows": len(validation_data.table.targets),
        "classifier_reference_only": True,
        "classifier_state_saved_in_artifact": False,
        "milton_fit_calibration_or_selection_performed": False,
    }
    torch.save(
        {
            "schema_version": ARTIFACT_SCHEMA,
            "utility_input_dim": int(regressor.standardizer.mean.shape[0]),
            "utility_state_dict": {
                key: value.detach().cpu()
                for key, value in regressor.model.state_dict().items()
            },
            "utility_mean": regressor.standardizer.mean,
            "utility_scale": regressor.standardizer.scale,
            "metadata": artifact_metadata,
        },
        seed_dir / "relative_utility_artifact.pt",
    )
    completion = {
        **artifact_metadata,
        "schema_version": FIT_SCHEMA,
        "artifact_schema_version": ARTIFACT_SCHEMA,
        "artifact_sha256": _sha256(seed_dir / "relative_utility_artifact.pt"),
        "history_sha256": _sha256(seed_dir / "training_history.json"),
        "validation_regret_sha256": _sha256(
            seed_dir / "validation_top1_regret.csv"
        ),
    }
    completion_path.write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    return completion


def fit_all_seeds(args: argparse.Namespace) -> None:
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    _validate_config(config)
    _validate_args(args, config)
    protocol_dir = _resolve(args.protocol_dir)
    cache_root = _resolve(args.cache_root)
    source_fit_root = _resolve(args.source_fit_root)
    output_root = _resolve(args.output_root)
    manifests = _read_fit_manifests(protocol_dir)
    completions: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        print(f"fit relative-geometry utility seed={seed}", flush=True)
        completions[str(seed)] = _fit_seed(
            args,
            seed=seed,
            config=config,
            config_path=config_path,
            protocol_dir=protocol_dir,
            cache_root=cache_root,
            source_fit_root=source_fit_root,
            output_root=output_root,
            manifests=manifests,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "claim_scope": "CVIAN-only fit for Milton zero-shot sensitivity",
        "config_sha256": _sha256(config_path),
        "fit_entrypoint_sha256": _sha256(Path(__file__).resolve()),
        "feature_interface_sha256": _sha256(
            REPO_ROOT
            / "crossview_conflict"
            / "decision"
            / "relative_geometry_utility.py"
        ),
        "feature_schema": RELATIVE_GEOMETRY_FEATURE_SCHEMA,
        "feature_width": len(RELATIVE_GEOMETRY_FEATURE_NAMES),
        "seeds": list(SEEDS),
        "prospective_test_loaded": False,
        "milton_data_loaded": False,
        "milton_fit_calibration_or_selection_performed": False,
        "seed_completion_sha256": {
            str(seed): _sha256(output_root / f"seed{seed}" / "fit_complete.json")
            for seed in SEEDS
        },
        "seed_artifact_sha256": {
            str(seed): completions[str(seed)]["artifact_sha256"] for seed in SEEDS
        },
    }
    (output_root / "fit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    fit_all_seeds(parse_args())


if __name__ == "__main__":
    main()
