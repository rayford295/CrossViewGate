from __future__ import annotations

"""Run the frozen, sensitivity-only Milton fixed-k=3 active-view study.

This entrypoint has evaluation code only.  It never updates a parameter,
temperature, threshold, or policy.  The registered main environment rolls
sector-indexed content so physical origin ``r`` becomes local sector zero;
the frozen classifier therefore remains on its trained mask support.  The
absolute-aware policy is deliberately isolated as an unrolled/off-support
diagnostic.
"""

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.decision.active_view import (
    ActiveViewCache,
    ActiveViewMLP,
    Standardizer,
    build_state_features,
    realized_operational_cost,
    softmax,
    validate_active_view_cache,
)
from crossview_conflict.decision.active_view_utility import (
    bayes_operational_action_and_risk,
    build_candidate_utility_features,
    target_conditioned_soft_loss,
)
from crossview_conflict.decision.relative_geometry_utility import (
    RELATIVE_GEOMETRY_FEATURE_NAMES,
    RELATIVE_GEOMETRY_FEATURE_SCHEMA,
    build_relative_geometry_utility_features,
)


CONFIG_SCHEMA = "milton-zero-shot-active-view-sensitivity-v1"
CACHE_SUMMARY_SCHEMA = "milton-active-view-forward-cache-summary-v1"
CACHE_SCHEMA = "milton-active-view-forward-cache-v1"
MAIN_POLICY_SCHEMA = "cvian-sequence-utility-policy-v1"
MAIN_FIT_SCHEMA = "cvian-sequence-utility-fit-v1"
RELATIVE_POLICY_SCHEMA = "cvian-relative-geometry-utility-v1"
RELATIVE_FIT_SCHEMA = "cvian-relative-geometry-utility-fit-v1"
COMMITMENT_SCHEMA = "milton-zero-shot-active-view-commitment-v1"
STARTED_SCHEMA = "milton-zero-shot-active-view-started-v1"
COMPLETION_SCHEMA = "milton-zero-shot-active-view-completion-v1"
AGGREGATE_SCHEMA = "milton-zero-shot-active-view-aggregate-v1"
EXPECTED_SEEDS = (42, 123, 456, 789, 1011)
EXPECTED_ROTATIONS = tuple(range(8))
EXPECTED_SAMPLES = 1707
EXPECTED_DEPENDENCY_GROUPS = 259
EXPECTED_SPATIAL_BLOCKS = 57
EXPECTED_JOINT_COMPONENTS = 11
EXPECTED_PER_SEED_RAW_ROWS = 518_928
EXPECTED_PER_SEED_TRAJECTORY_ROWS = 95_592
EXPECTED_TOTAL_RAW_ROWS = 2_594_640
EXPECTED_TOTAL_TRAJECTORY_ROWS = 477_960
EXPECTED_POST_ORIGIN_ROWS = 59_745
EXPECTED_POST_SEED_ROWS = 11_949
EXPECTED_AZIMUTH = np.asarray(
    [0.0, 45.0, 90.0, 135.0, -180.0, -135.0, -90.0, -45.0],
    dtype=np.float64,
)
FROZEN_CVIAN_RUNNER_SHA256 = (
    "6f7cb8d5fb6914f5a59a449655b58370ef9e245e2f140fca3afd863c2194be2b"
)
FROZEN_CONFIG_SHA256 = "c9457127d156aae00770d36196bdc1526934ca30bf5269978f75cadf088df463"
COMMITMENT_NAME = "pre_score_commitment.json"
STARTED_NAME = "evaluation_started.json"
REPORT_PATH = REPO_ROOT / "docs" / "results" / "active_view_milton_zero_shot_v1.md"


@dataclass(frozen=True)
class MiltonData:
    cache: ActiveViewCache
    dependency_group_id: np.ndarray
    joint_component_id: np.ndarray
    sorted_sample_ordinal: np.ndarray


@dataclass(frozen=True)
class FrozenModels:
    classifier: torch.nn.Module
    classifier_standardizer: Standardizer
    absolute_regressor: torch.nn.Module
    absolute_standardizer: Standardizer
    relative_regressor: torch.nn.Module
    relative_standardizer: Standardizer
    temperature: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/milton_zero_shot_active_view_sensitivity_v1.json"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
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
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _runtime_contract(*, device: str, batch_size: int) -> dict[str, Any]:
    return {
        "device": device,
        "batch_size": int(batch_size),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _configure_deterministic_inference() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must equal the frozen :4096:8 value")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    if (
        not torch.are_deterministic_algorithms_enabled()
        or not torch.backends.cudnn.deterministic
        or torch.backends.cudnn.benchmark
    ):
        raise RuntimeError("Failed to enable the registered deterministic inference runtime")


def _require_fields(payload: Mapping[str, Any], expected: Mapping[str, Any], context: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{context} changed field {key!r}")


def _validate_config(config: Mapping[str, Any]) -> None:
    _require_fields(
        config,
        {
            "schema_version": CONFIG_SCHEMA,
            "confirmatory_eligible": False,
            "seeds": list(EXPECTED_SEEDS),
            "sector_count": 8,
            "cyclic_origin_rotations": list(EXPECTED_ROTATIONS),
            "fixed_budget_views": 3,
            "fixed_budget_views_semantics": (
                "total revealed views including the rolled-local initial sector 0"
            ),
            "view_cost": 0.5,
            "view_cost_semantics": (
                "0.5 per additional reveal; the initial view has zero acquisition cost"
            ),
            "cost_matrix": [[0, 1, 4], [1, 0, 1], [8, 8, 0]],
        },
        "Milton sensitivity config",
    )
    relative = config.get("cvian_relative_policy_fit", {})
    _require_fields(
        relative,
        {
            "permitted_roles": ["selector_fit", "validation"],
            "forbidden_roles": ["base_fit", "prospective_test"],
            "classifier_updates": False,
            "base_encoder_updates": False,
            "temperature_updates": False,
            "absolute_geometry_inputs": False,
            "feature_schema": RELATIVE_GEOMETRY_FEATURE_SCHEMA,
            "feature_width": len(RELATIVE_GEOMETRY_FEATURE_NAMES),
            "remaining_acquisitions_feature": "(4 - revealed_view_count) / 8",
            "azimuth_geometry_exact_deg": EXPECTED_AZIMUTH.tolist(),
            "canonical_sector_one_hot_forbidden": True,
            "compass_and_absolute_yaw_forbidden": True,
            "hidden_candidate_image_embedding_or_logit_forbidden": True,
        },
        "relative-policy contract",
    )
    evaluation = config.get("evaluation", {})
    _require_fields(
        evaluation,
        {
            "mode": "frozen_zero_shot_forward_only",
            "milton_parameter_fitting": False,
            "milton_calibration": False,
            "milton_model_selection": False,
            "milton_policy_selection": False,
            "stop_or_adaptive_policy_enabled": False,
            "go_no_go_criterion": None,
            "pre_score_commitment_required": True,
            "evaluation_started_sentinel_required_before_first_milton_cache_load": True,
            "overwrite_or_rerun_permitted": False,
            "main_policy": "relative_geometry_utility",
            "comparators": ["farthest", "clockwise", "random_mc32"],
            "ood_diagnostics": ["absolute_aware_utility_ood_diagnostic"],
            "privileged_diagnostics": [
                "max_confidence_privileged",
                "greedy_label_oracle_privileged",
            ],
            "random_trajectories_per_seed_rotation": 32,
            "averaging_order": (
                "trajectory then origin then model seed within each sample/unit "
                "before paired policy contrasts"
            ),
        },
        "Milton evaluation contract",
    )
    expected_rng = (
        "numpy Generator(SeedSequence([model_seed, physical_origin, trajectory_index, "
        "sorted_sample_ordinal, 20260710])); sample ordinal is from ascending full "
        "sample_id, candidates are sorted rolled-local sector IDs, and Python hash "
        "is forbidden"
    )
    if evaluation.get("random_rng_derivation") != expected_rng:
        raise ValueError("Registered random RNG derivation changed")
    rotation = evaluation.get("rotation_action_contract", {})
    if rotation.get("origin_rotation_r") != (
        "cyclically roll all sector-indexed cache tensors so physical sector r is local sector 0"
    ):
        raise ValueError("Registered roll-to-local-origin contract changed")
    bootstrap = evaluation.get("bootstrap", {})
    _require_fields(
        bootstrap,
        {
            "primary_resampling_unit": "joint_dependency_spatial_component",
            "expected_joint_components": 11,
            "resamples": 10000,
            "random_seed": 42,
            "confidence_level": 0.95,
            "interval": "percentile",
        },
        "Milton bootstrap contract",
    )
    implementation = config.get("implementation_contract", {})
    implementation_paths = {
        "fit_entrypoint_sha256": REPO_ROOT
        / "scripts"
        / "fit_cvian_relative_geometry_utility.py",
        "feature_interface_sha256": REPO_ROOT
        / "crossview_conflict"
        / "decision"
        / "relative_geometry_utility.py",
        "frozen_cvian_runner_sha256": REPO_ROOT
        / "scripts"
        / "run_cvian_sequence_utility_experiment.py",
        "frozen_active_view_sha256": REPO_ROOT
        / "crossview_conflict"
        / "decision"
        / "active_view.py",
        "frozen_active_view_utility_sha256": REPO_ROOT
        / "crossview_conflict"
        / "decision"
        / "active_view_utility.py",
    }
    for field, path in implementation_paths.items():
        if implementation.get(field) != _sha256(path):
            raise ValueError(f"Registered implementation hash changed: {field}")


def roll_cache_to_local_origin(cache: ActiveViewCache, physical_origin: int) -> ActiveViewCache:
    """Roll sector content while keeping the frozen classifier's local geometry."""

    count = cache.sector_count
    if count != 8 or physical_origin not in EXPECTED_ROTATIONS:
        raise ValueError("The registered rotation requires origin 0..7 and eight sectors")
    if not np.array_equal(cache.sector_id, np.arange(count)) or not np.allclose(
        cache.relative_azimuth_deg, EXPECTED_AZIMUTH, rtol=0.0, atol=1e-7
    ):
        raise ValueError("Milton cache geometry changed")
    physical_order = (np.arange(count, dtype=np.int64) + physical_origin) % count
    return replace(
        cache,
        sector_id=np.arange(count, dtype=np.int64),
        relative_azimuth_deg=EXPECTED_AZIMUTH.copy(),
        street_embedding=cache.street_embedding[:, physical_order, :],
        sector_logits=cache.sector_logits[:, physical_order, :],
    )


def local_to_physical(local_sector: np.ndarray, physical_origin: int) -> np.ndarray:
    values = np.asarray(local_sector, dtype=np.int64)
    if ((values < 0) | (values >= 8)).any() or physical_origin not in EXPECTED_ROTATIONS:
        raise ValueError("Invalid local-to-physical sector mapping")
    return (values + int(physical_origin)) % 8


def _joint_dependency_components(
    dependency_group: np.ndarray, spatial_block: np.ndarray
) -> np.ndarray:
    dependency = np.asarray(dependency_group).astype(str)
    blocks = np.asarray(spatial_block).astype(str)
    if dependency.shape != blocks.shape or dependency.ndim != 1:
        raise ValueError("Dependency and spatial identifiers must align")
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for dependency_id, block_id in zip(dependency, blocks):
        union(f"dependency:{dependency_id}", f"block:{block_id}")
    roots = sorted({find(f"dependency:{value}") for value in dependency})
    names = {root: f"joint_component_{index:02d}" for index, root in enumerate(roots)}
    return np.asarray(
        [names[find(f"dependency:{value}")] for value in dependency], dtype="U32"
    )


def _sorted_ordinals(sample_id: np.ndarray) -> np.ndarray:
    values = np.asarray(sample_id).astype(str)
    order = np.argsort(values, kind="mergesort")
    if len(np.unique(values)) != len(values):
        raise ValueError("Sample IDs must be unique before ordinal derivation")
    ordinal = np.empty(len(values), dtype=np.int64)
    ordinal[order] = np.arange(len(values), dtype=np.int64)
    return ordinal


def _load_milton_cache(path: Path, *, expected_sha256: str, expected_components: int) -> MiltonData:
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError(f"Milton cache hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "sample_id",
            "spatial_block_id",
            "sequence_id",
            "dependency_group_id",
            "target",
            "latitude",
            "longitude",
            "compass_angle_deg",
            "compass_available",
            "sequence_metadata_available",
            "sector_id",
            "relative_azimuth_deg",
            "street_embedding",
            "overhead_embedding",
            "sector_logits",
            "panorama_logits",
        }
        if not required.issubset(payload.files):
            raise ValueError(f"Milton cache arrays are incomplete: {path}")
        dependency_group = payload["dependency_group_id"].astype(str)
        sequence = payload["sequence_id"].astype(str)
        if len(np.unique(sequence)) != len(sequence):
            raise ValueError("Missing sequence sentinels collapsed; sequence aggregation forbidden")
        if payload["sequence_metadata_available"].astype(bool).any():
            raise ValueError("Milton cache unexpectedly claims observed sequences")
        if payload["compass_available"].astype(bool).any() or not np.isnan(
            payload["compass_angle_deg"]
        ).all():
            raise ValueError("Milton cache unexpectedly contains compass metadata")
        cache = ActiveViewCache(
            sample_id=payload["sample_id"].astype(str),
            spatial_block_id=payload["spatial_block_id"].astype(str),
            sequence_id=sequence,
            target=payload["target"].astype(np.int64),
            latitude=payload["latitude"].astype(np.float64),
            longitude=payload["longitude"].astype(np.float64),
            sector_id=payload["sector_id"].astype(np.int64),
            relative_azimuth_deg=payload["relative_azimuth_deg"].astype(np.float64),
            street_embedding=payload["street_embedding"].astype(np.float32),
            overhead_embedding=payload["overhead_embedding"].astype(np.float32),
            sector_logits=payload["sector_logits"].astype(np.float32),
            panorama_logits=payload["panorama_logits"].astype(np.float32),
        )
    if cache.sample_count != 1707 or cache.sector_count != 8:
        raise ValueError("Registered Milton cache dimensions changed")
    validate_active_view_cache(cache)
    if set(cache.target.tolist()) != {0, 1, 2} or not np.array_equal(
        np.bincount(cache.target, minlength=3), np.asarray([413, 817, 477])
    ):
        raise ValueError("Registered Milton target support/distribution changed")
    if not np.array_equal(cache.sector_id, np.arange(8)) or not np.allclose(
        cache.relative_azimuth_deg, EXPECTED_AZIMUTH, rtol=0.0, atol=1e-7
    ):
        raise ValueError("Registered Milton sector geometry changed")
    components = _joint_dependency_components(dependency_group, cache.spatial_block_id)
    if len(np.unique(components)) != expected_components:
        raise ValueError("Registered joint dependency-component count changed")
    return MiltonData(
        cache=cache,
        dependency_group_id=dependency_group,
        joint_component_id=components,
        sorted_sample_ordinal=_sorted_ordinals(cache.sample_id),
    )


def _load_transfer_manifest_after_started(config: Mapping[str, Any]) -> pd.DataFrame:
    contract_path = _resolve(config["source_contract"]["manifest_contract"])
    contract = _load_json(contract_path)
    manifest_name = contract.get("manifest", {}).get("filename")
    if not isinstance(manifest_name, str):
        raise ValueError("Milton manifest contract has no registered filename")
    manifest_path = contract_path.parent / manifest_name
    if _sha256(manifest_path) != config["source_contract"]["transfer_manifest_sha256"]:
        raise ValueError("Registered Milton transfer manifest changed")
    frame = pd.read_csv(
        manifest_path,
        dtype={
            "sample_id": str,
            "dependency_group_id": str,
            "spatial_block_id": str,
            "sequence_id": str,
        },
    ).reset_index(drop=True)
    required = {
        "sample_id",
        "label",
        "dependency_group_id",
        "spatial_block_id",
        "latitude",
        "longitude",
    }
    if len(frame) != 1707 or not required.issubset(frame.columns):
        raise ValueError("Registered Milton transfer manifest shape changed")
    if frame["sample_id"].duplicated().any() or not np.array_equal(
        np.bincount(frame["label"].astype(int), minlength=3),
        np.asarray([413, 817, 477]),
    ):
        raise ValueError("Registered Milton transfer labels or IDs changed")
    return frame


def _validate_cache_against_manifest(data: MiltonData, manifest: pd.DataFrame) -> None:
    cache = data.cache
    exact = {
        "sample_id": cache.sample_id,
        "label": cache.target,
        "dependency_group_id": data.dependency_group_id,
        "spatial_block_id": cache.spatial_block_id,
    }
    for field, values in exact.items():
        expected = manifest[field].to_numpy()
        if field == "label":
            expected = expected.astype(np.int64)
        else:
            expected = expected.astype(str)
            values = np.asarray(values).astype(str)
        if not np.array_equal(values, expected):
            raise ValueError(f"Milton cache/manifest order mismatch for {field}")
    for field, values in (("latitude", cache.latitude), ("longitude", cache.longitude)):
        if not np.allclose(
            values,
            manifest[field].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Milton cache/manifest coordinate mismatch for {field}")


def _predict_batches(
    model: torch.nn.Module,
    standardizer: Standardizer,
    features: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    values = standardizer.transform(features)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            outputs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _state_and_probabilities(
    cache: ActiveViewCache,
    mask: np.ndarray,
    models: FrozenModels,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if mask.shape != (cache.sample_count, cache.sector_count):
        raise ValueError("State mask shape changed")
    revealed = mask.sum(axis=1)
    state = build_state_features(
        cache,
        mask,
        sample_indices=np.arange(cache.sample_count, dtype=np.int64),
        remaining_budget=np.maximum(4 - revealed, 0).astype(np.float32),
    )
    logits = _predict_batches(
        models.classifier,
        models.classifier_standardizer,
        state,
        device=device,
        batch_size=batch_size,
    )
    return state, softmax(logits, temperature=models.temperature)


def _candidate_index(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, candidates = np.where(~mask)
    return rows.astype(np.int64), candidates.astype(np.int64)


def _scores_to_actions(
    scores: np.ndarray, mask: np.ndarray, *, tie_origin: int = 0
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != mask.shape:
        raise ValueError("Candidate scores must match the revealed mask")
    values = np.where(mask, -np.inf, values)
    if (~mask).sum(axis=1).min() < 1 or not np.isfinite(values[~mask]).all():
        raise ValueError("Every row needs finite scores for every available candidate")
    if not 0 <= tie_origin < mask.shape[1]:
        raise ValueError("tie_origin is outside the sector range")
    order = (np.arange(mask.shape[1], dtype=np.int64) + tie_origin) % mask.shape[1]
    # Argmax in rotation-relative order provides the registered local tie break.
    return order[values[:, order].argmax(axis=1)].astype(np.int64)


def _learned_actions(
    policy: str,
    cache: ActiveViewCache,
    mask: np.ndarray,
    models: FrozenModels,
    *,
    device: str,
    batch_size: int,
    tie_origin: int = 0,
) -> np.ndarray:
    state, probabilities = _state_and_probabilities(
        cache, mask, models, device=device, batch_size=batch_size
    )
    rows, candidates = _candidate_index(mask)
    if policy == "relative_geometry_utility":
        features = build_relative_geometry_utility_features(
            probabilities[rows],
            mask[rows],
            candidates,
            cache.relative_azimuth_deg,
            origin_sector=0,
            remaining_acquisitions=np.maximum(4 - mask[rows].sum(axis=1), 0),
        )
        predicted = _predict_batches(
            models.relative_regressor,
            models.relative_standardizer,
            features,
            device=device,
            batch_size=batch_size,
        ).reshape(-1)
    elif policy == "absolute_aware_utility_ood_diagnostic":
        features = build_candidate_utility_features(
            state[rows],
            probabilities[rows],
            mask[rows],
            candidates,
            cache.relative_azimuth_deg,
            compass_angle_deg=np.full(len(rows), np.nan),
            compass_available=np.zeros(len(rows), dtype=bool),
        )
        predicted = _predict_batches(
            models.absolute_regressor,
            models.absolute_standardizer,
            features,
            device=device,
            batch_size=batch_size,
        ).reshape(-1)
    else:
        raise ValueError(f"Not a learned Milton policy: {policy}")
    scores = np.full(mask.shape, -np.inf, dtype=np.float64)
    scores[rows, candidates] = predicted
    return _scores_to_actions(scores, mask, tie_origin=tie_origin)


def _lookahead_probabilities(
    cache: ActiveViewCache,
    mask: np.ndarray,
    models: FrozenModels,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state, current = _state_and_probabilities(
        cache, mask, models, device=device, batch_size=batch_size
    )
    del state
    rows, candidates = _candidate_index(mask)
    after_mask = mask[rows].copy()
    after_mask[np.arange(len(rows)), candidates] = True
    revealed = after_mask.sum(axis=1)
    after_state = build_state_features(
        cache,
        after_mask,
        sample_indices=rows,
        remaining_budget=np.maximum(4 - revealed, 0).astype(np.float32),
    )
    after_logits = _predict_batches(
        models.classifier,
        models.classifier_standardizer,
        after_state,
        device=device,
        batch_size=batch_size,
    )
    return rows, candidates, current, softmax(after_logits, models.temperature)


def _privileged_actions(
    policy: str,
    cache: ActiveViewCache,
    mask: np.ndarray,
    models: FrozenModels,
    *,
    view_cost: float,
    cost_matrix: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    rows, candidates, current, after = _lookahead_probabilities(
        cache, mask, models, device=device, batch_size=batch_size
    )
    if policy == "max_confidence_privileged":
        predicted = after.max(axis=1)
    elif policy == "greedy_label_oracle_privileged":
        target = cache.target[rows]
        before_loss = target_conditioned_soft_loss(
            current[rows], target, cost_matrix=cost_matrix
        )
        after_loss = target_conditioned_soft_loss(
            after, target, cost_matrix=cost_matrix
        )
        predicted = before_loss - after_loss - float(view_cost)
    else:
        raise ValueError(f"Not a privileged Milton policy: {policy}")
    scores = np.full(mask.shape, -np.inf, dtype=np.float64)
    scores[rows, candidates] = predicted
    return _scores_to_actions(scores, mask)


def _farthest_actions(mask: np.ndarray) -> np.ndarray:
    sector_count = mask.shape[1]
    scores = np.full(mask.shape, -np.inf, dtype=np.float64)
    for candidate in range(sector_count):
        distance = np.minimum(
            np.abs(np.arange(sector_count) - candidate),
            sector_count - np.abs(np.arange(sector_count) - candidate),
        )
        candidate_distance = np.where(mask, distance[None, :], np.inf).min(axis=1)
        scores[:, candidate] = np.where(mask[:, candidate], -np.inf, candidate_distance)
    return _scores_to_actions(scores, mask)


def _clockwise_actions(mask: np.ndarray) -> np.ndarray:
    available = ~mask
    if not available.any(axis=1).all():
        raise ValueError("Clockwise policy has no available sector")
    return available.argmax(axis=1).astype(np.int64)


def _random_trajectory(
    sample_ordinals: np.ndarray,
    *,
    model_seed: int,
    physical_origin: int,
    trajectory: int,
    budget: int,
    sector_count: int = 8,
) -> np.ndarray:
    """Return local actions using only the registered SeedSequence derivation."""

    ordinals = np.asarray(sample_ordinals, dtype=np.int64)
    selected = np.zeros((len(ordinals), budget), dtype=np.int64)
    selected[:, 0] = 0
    for row, ordinal in enumerate(ordinals):
        generator = np.random.default_rng(
            np.random.SeedSequence(
                [model_seed, physical_origin, trajectory, int(ordinal), 20260710]
            )
        )
        available = list(range(1, sector_count))
        for step in range(1, budget):
            offset = int(generator.integers(0, len(available)))
            selected[row, step] = available.pop(offset)
    return selected


def _validate_selected_orders(
    selected_local: np.ndarray,
    *,
    physical_origin: int,
    expected_budget: int,
) -> np.ndarray:
    selected = np.asarray(selected_local, dtype=np.int64)
    if selected.ndim != 2 or selected.shape[1] != expected_budget:
        raise ValueError("Selected trajectory has the wrong fixed budget")
    if ((selected < 0) | (selected >= 8)).any() or not (selected[:, 0] == 0).all():
        raise ValueError("Rolled-local trajectories must start at valid local sector 0")
    if any(len(np.unique(row)) != expected_budget for row in selected):
        raise ValueError("A trajectory contains a duplicate sector")
    physical = local_to_physical(selected, physical_origin)
    if not (physical[:, 0] == physical_origin).all():
        raise ValueError("Local-to-physical origin mapping changed")
    return physical


def _terminal_frame(
    *,
    policy: str,
    policy_scope: str,
    seed: int,
    physical_origin: int,
    trajectory: int,
    data: MiltonData,
    environment_cache: ActiveViewCache,
    selected_local: np.ndarray,
    selected_environment: np.ndarray | None = None,
    models: FrozenModels,
    cost_matrix: np.ndarray,
    view_cost: float,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    environment_order = (
        selected_local
        if selected_environment is None
        else np.asarray(selected_environment, dtype=np.int64)
    )
    if environment_order.shape != selected_local.shape or (
        (environment_order < 0) | (environment_order >= environment_cache.sector_count)
    ).any():
        raise ValueError("Environment reveal order is invalid")
    if any(len(np.unique(row)) != environment_order.shape[1] for row in environment_order):
        raise ValueError("Environment reveal order contains a duplicate")
    mask = np.zeros((environment_cache.sample_count, environment_cache.sector_count), dtype=bool)
    rows = np.arange(environment_cache.sample_count)[:, None]
    mask[rows, environment_order] = True
    _, probabilities = _state_and_probabilities(
        environment_cache, mask, models, device=device, batch_size=batch_size
    )
    prediction, bayes_risk = bayes_operational_action_and_risk(
        probabilities, cost_matrix=cost_matrix
    )
    target = environment_cache.target.astype(np.int64)
    classification_cost = realized_operational_cost(
        prediction, target, cost_matrix=cost_matrix
    )
    spent_view_cost = (selected_local.shape[1] - 1) * float(view_cost)
    selected_physical = _validate_selected_orders(
        selected_local,
        physical_origin=physical_origin,
        expected_budget=selected_local.shape[1],
    )
    return pd.DataFrame(
        {
            "seed": seed,
            "physical_origin": physical_origin,
            "policy": policy,
            "policy_scope": policy_scope,
            "sample_id": data.cache.sample_id.astype(str),
            "dependency_group_id": data.dependency_group_id.astype(str),
            "spatial_block_id": data.cache.spatial_block_id.astype(str),
            "joint_component_id": data.joint_component_id.astype(str),
            "trajectory": trajectory,
            "target": target,
            "prediction": prediction,
            "local_revealed_order": [
                ",".join(map(str, values)) for values in selected_local
            ],
            "physical_revealed_order": [
                ",".join(map(str, values)) for values in selected_physical
            ],
            "view_count": selected_local.shape[1],
            "additional_view_count": selected_local.shape[1] - 1,
            "p0": probabilities[:, 0],
            "p1": probabilities[:, 1],
            "p2": probabilities[:, 2],
            "model_bayes_risk": bayes_risk,
            "classification_cost": classification_cost,
            "view_cost": spent_view_cost,
            "operational_cost": classification_cost + spent_view_cost,
            "severe_miss": ((target == 2) & (prediction != 2)).astype(float),
            "correct": (target == prediction).astype(float),
        }
    )


def _evaluate_policy(
    *,
    policy: str,
    seed: int,
    physical_origin: int,
    data: MiltonData,
    models: FrozenModels,
    config: Mapping[str, Any],
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    evaluation = config["evaluation"]
    budget = int(config["fixed_budget_views"])
    view_cost = float(config["view_cost"])
    cost_matrix = np.asarray(config["cost_matrix"], dtype=np.float64)
    is_ood = policy == "absolute_aware_utility_ood_diagnostic"
    environment = data.cache if is_ood else roll_cache_to_local_origin(
        data.cache, physical_origin
    )
    initial = physical_origin if is_ood else 0
    trajectories = (
        int(evaluation["random_trajectories_per_seed_rotation"])
        if policy == "random_mc32"
        else 1
    )
    scope = (
        "unrolled_classifier_off_support_ood_diagnostic"
        if is_ood
        else (
            "rolled_local_privileged_diagnostic"
            if policy in evaluation["privileged_diagnostics"]
            else "rolled_local_main_or_comparator"
        )
    )
    frames: list[pd.DataFrame] = []
    for trajectory in range(trajectories):
        if policy == "random_mc32":
            selected = _random_trajectory(
                data.sorted_sample_ordinal,
                model_seed=seed,
                physical_origin=physical_origin,
                trajectory=trajectory,
                budget=budget,
            )
        else:
            selected = np.full((environment.sample_count, budget), -1, dtype=np.int64)
            selected[:, 0] = initial
            mask = np.zeros((environment.sample_count, environment.sector_count), dtype=bool)
            mask[:, initial] = True
            for step in range(1, budget):
                if policy in (
                    "relative_geometry_utility",
                    "absolute_aware_utility_ood_diagnostic",
                ):
                    action = _learned_actions(
                        policy,
                        environment,
                        mask,
                        models,
                        device=device,
                        batch_size=batch_size,
                        tie_origin=physical_origin if is_ood else 0,
                    )
                elif policy == "farthest":
                    action = _farthest_actions(mask)
                elif policy == "clockwise":
                    action = _clockwise_actions(mask)
                elif policy in (
                    "max_confidence_privileged",
                    "greedy_label_oracle_privileged",
                ):
                    action = _privileged_actions(
                        policy,
                        environment,
                        mask,
                        models,
                        view_cost=view_cost,
                        cost_matrix=cost_matrix,
                        device=device,
                        batch_size=batch_size,
                    )
                else:
                    raise ValueError(f"Unknown Milton policy: {policy}")
                if mask[np.arange(environment.sample_count), action].any():
                    raise ValueError(f"Policy {policy} repeated a revealed sector")
                selected[:, step] = action
                mask[np.arange(environment.sample_count), action] = True
        # OOD actions are physical IDs already; convert to local before the common
        # physical mapping so the stored action audit remains exact.
        terminal_selected = (
            (selected - physical_origin) % environment.sector_count if is_ood else selected
        )
        _validate_selected_orders(
            terminal_selected,
            physical_origin=physical_origin,
            expected_budget=budget,
        )
        terminal_environment = (
            roll_cache_to_local_origin(data.cache, physical_origin) if is_ood else environment
        )
        if is_ood:
            # Terminal prediction remains in the same unrolled/off-support
            # diagnostic environment used for selection.
            frame = _terminal_frame(
                policy=policy,
                policy_scope=scope,
                seed=seed,
                physical_origin=physical_origin,
                trajectory=trajectory,
                data=data,
                environment_cache=environment,
                selected_local=terminal_selected,
                selected_environment=selected,
                models=models,
                cost_matrix=cost_matrix,
                view_cost=view_cost,
                device=device,
                batch_size=batch_size,
            )
        else:
            frame = _terminal_frame(
                policy=policy,
                policy_scope=scope,
                seed=seed,
                physical_origin=physical_origin,
                trajectory=trajectory,
                data=data,
                environment_cache=terminal_environment,
                selected_local=terminal_selected,
                models=models,
                cost_matrix=cost_matrix,
                view_cost=view_cost,
                device=device,
                batch_size=batch_size,
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _policy_inventory(config: Mapping[str, Any]) -> tuple[str, ...]:
    evaluation = config["evaluation"]
    values = (
        [evaluation["main_policy"]]
        + list(evaluation["comparators"])
        + list(evaluation["ood_diagnostics"])
        + list(evaluation["privileged_diagnostics"])
    )
    if len(values) != len(set(values)) or len(values) != 7:
        raise ValueError("Registered Milton policy inventory changed")
    return tuple(values)


def _trajectory_average(decisions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "seed",
        "physical_origin",
        "policy",
        "policy_scope",
        "sample_id",
        "dependency_group_id",
        "spatial_block_id",
        "joint_component_id",
        "target",
    ]
    values = decisions.groupby(keys, sort=True, observed=True).agg(
        operational_cost=("operational_cost", "mean"),
        severe_miss=("severe_miss", "mean"),
        correct=("correct", "mean"),
        trajectory_count=("trajectory", "nunique"),
    )
    return values.reset_index()


def _validate_decision_lattice(
    decisions: pd.DataFrame,
    *,
    policies: Sequence[str],
    expected_samples: int = EXPECTED_SAMPLES,
    expected_origins: int = len(EXPECTED_ROTATIONS),
    expected_random_trajectories: int = 32,
    expected_raw_rows: int = EXPECTED_PER_SEED_RAW_ROWS,
    expected_dependency_groups: int = EXPECTED_DEPENDENCY_GROUPS,
    expected_spatial_blocks: int = EXPECTED_SPATIAL_BLOCKS,
    expected_joint_components: int = EXPECTED_JOINT_COMPONENTS,
) -> None:
    keys = ["seed", "physical_origin", "policy", "sample_id", "trajectory"]
    if decisions.duplicated(keys).any():
        raise ValueError("Duplicate decision-lattice row detected")
    expected_inventory = set(map(str, policies))
    observed_inventory = set(decisions["policy"].astype(str))
    if (
        len(decisions) != expected_raw_rows
        or decisions["sample_id"].nunique() != expected_samples
        or decisions["physical_origin"].nunique() != expected_origins
        or observed_inventory != expected_inventory
        or decisions["seed"].nunique() != 1
        or decisions["dependency_group_id"].nunique() != expected_dependency_groups
        or decisions["spatial_block_id"].nunique() != expected_spatial_blocks
        or decisions["joint_component_id"].nunique() != expected_joint_components
    ):
        raise ValueError("Decision-lattice registered counts changed")
    multiplicity = decisions.groupby(
        ["seed", "physical_origin", "policy", "sample_id"],
        sort=False,
        observed=True,
    )["trajectory"].agg(["size", "nunique", "min", "max", "sum"])
    if len(multiplicity) != expected_samples * expected_origins * len(policies):
        raise ValueError("Decision-lattice sample/origin/policy cross product is incomplete")
    sample_group_count = multiplicity.reset_index().groupby("sample_id").size()
    if not (sample_group_count == expected_origins * len(policies)).all() or set(
        decisions["physical_origin"].astype(int)
    ) != set(range(expected_origins)):
        raise ValueError("Decision-lattice sample/origin cross product is incomplete")
    expected = np.where(
        multiplicity.index.get_level_values("policy") == "random_mc32",
        expected_random_trajectories,
        1,
    )
    if not np.array_equal(multiplicity["size"].to_numpy(), expected) or not np.array_equal(
        multiplicity["nunique"].to_numpy(), expected
    ):
        raise ValueError("Decision-lattice trajectory multiplicity is incomplete")
    random_rows = multiplicity.index.get_level_values("policy") == "random_mc32"
    if (
        not (multiplicity.loc[random_rows, "min"] == 0).all()
        or not (
            multiplicity.loc[random_rows, "max"] == expected_random_trajectories - 1
        ).all()
        or not (
            multiplicity.loc[random_rows, "sum"]
            == expected_random_trajectories * (expected_random_trajectories - 1) // 2
        ).all()
        or not (multiplicity.loc[~random_rows, "min"] == 0).all()
        or not (multiplicity.loc[~random_rows, "max"] == 0).all()
    ):
        raise ValueError("Decision-lattice trajectory IDs changed")
    if not (decisions["view_count"] == 3).all() or not (
        decisions["additional_view_count"] == 2
    ).all():
        raise ValueError("Decision-lattice fixed budget changed")


def _metrics_for_seed(
    decisions: pd.DataFrame,
    *,
    expected_trajectory_rows: int = EXPECTED_PER_SEED_TRAJECTORY_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectory = _trajectory_average(decisions)
    expected_trajectory_count = np.where(
        trajectory["policy"].eq("random_mc32"), 32, 1
    )
    if not np.array_equal(
        trajectory["trajectory_count"].to_numpy(dtype=np.int64),
        expected_trajectory_count,
    ):
        raise ValueError("Policy trajectory multiplicity changed before aggregation")
    units = ("dependency_group_id", "spatial_block_id")
    rows: list[dict[str, Any]] = []
    for (seed, origin, policy), subset in trajectory.groupby(
        ["seed", "physical_origin", "policy"], sort=True
    ):
        for unit in units:
            grouped = subset.groupby(unit, sort=True).agg(
                operational_cost=("operational_cost", "mean"),
                severe_miss=("severe_miss", "mean"),
                correct=("correct", "mean"),
            )
            rows.append(
                {
                    "seed": seed,
                    "scope": "per_rotation",
                    "physical_origin": origin,
                    "policy": policy,
                    "aggregation_unit": unit,
                    "groups": len(grouped),
                    "sample_rows": len(subset),
                    "equal_unit_macro_operational_cost": grouped["operational_cost"].mean(),
                    "equal_unit_macro_severe_miss_rate": grouped["severe_miss"].mean(),
                    "equal_unit_macro_accuracy": grouped["correct"].mean(),
                }
            )
    origin_keys = [
        "seed",
        "policy",
        "policy_scope",
        "sample_id",
        "dependency_group_id",
        "spatial_block_id",
        "joint_component_id",
        "target",
    ]
    origin_average = trajectory.groupby(origin_keys, sort=True, observed=True).agg(
        operational_cost=("operational_cost", "mean"),
        severe_miss=("severe_miss", "mean"),
        correct=("correct", "mean"),
        origins=("physical_origin", "nunique"),
    ).reset_index()
    if not (origin_average["origins"] == 8).all():
        raise ValueError("Every sample/policy/seed must retain all eight origins")
    if len(trajectory) != expected_trajectory_rows:
        raise ValueError("Per-seed post-trajectory row count changed")
    for (seed, policy), subset in origin_average.groupby(["seed", "policy"], sort=True):
        for unit in units:
            grouped = subset.groupby(unit, sort=True).agg(
                operational_cost=("operational_cost", "mean"),
                severe_miss=("severe_miss", "mean"),
                correct=("correct", "mean"),
            )
            rows.append(
                {
                    "seed": seed,
                    "scope": "origin_averaged",
                    "physical_origin": -1,
                    "policy": policy,
                    "aggregation_unit": unit,
                    "groups": len(grouped),
                    "sample_rows": len(subset),
                    "equal_unit_macro_operational_cost": grouped["operational_cost"].mean(),
                    "equal_unit_macro_severe_miss_rate": grouped["severe_miss"].mean(),
                    "equal_unit_macro_accuracy": grouped["correct"].mean(),
                }
            )
    return pd.DataFrame(rows), trajectory


def _bootstrap_interval(
    unit_values: np.ndarray, *, resamples: int, seed: int, confidence: float
) -> tuple[float, float]:
    values = np.asarray(unit_values, dtype=np.float64).reshape(-1)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires at least two finite dependence units")
    generator = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    chunk = 256
    for start in range(0, resamples, chunk):
        stop = min(resamples, start + chunk)
        index = generator.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[index].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(
        estimates, [alpha, 1.0 - alpha], method="linear"
    )
    return float(low), float(high)


def _cluster_bootstrap_interval(
    unit_frame: pd.DataFrame,
    *,
    value_column: str,
    cluster_column: str,
    resamples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    """Resample whole joint components while preserving equal-unit weighting."""

    if unit_frame[cluster_column].isna().any():
        raise ValueError("Every estimand unit must belong to a joint component")
    cluster = unit_frame.groupby(cluster_column, sort=True)[value_column].agg(
        ["sum", "count"]
    )
    if len(cluster) < 2 or not np.isfinite(cluster.to_numpy(dtype=float)).all():
        raise ValueError("Cluster bootstrap requires at least two finite components")
    sums = cluster["sum"].to_numpy(dtype=np.float64)
    counts = cluster["count"].to_numpy(dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    chunk = 256
    for start in range(0, resamples, chunk):
        stop = min(resamples, start + chunk)
        index = generator.integers(0, len(cluster), size=(stop - start, len(cluster)))
        estimates[start:stop] = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(
        estimates, [alpha, 1.0 - alpha], method="linear"
    )
    return float(low), float(high)


def _aggregate_sensitivity(
    trajectory_rows: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    raw_decision_rows: int | None = None,
    expected_raw_rows: int = EXPECTED_TOTAL_RAW_ROWS,
    expected_trajectory_rows: int = EXPECTED_TOTAL_TRAJECTORY_ROWS,
    expected_origin_rows: int = EXPECTED_POST_ORIGIN_ROWS,
    expected_seed_rows: int = EXPECTED_POST_SEED_ROWS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Apply trajectory -> origin -> seed averaging before unit macro contrasts."""

    evaluation = config["evaluation"]
    bootstrap = evaluation["bootstrap"]
    main = str(evaluation["main_policy"])
    policies = _policy_inventory(config)
    observation_keys = [
        "policy",
        "policy_scope",
        "sample_id",
        "dependency_group_id",
        "spatial_block_id",
        "joint_component_id",
        "target",
    ]
    origin_stage = trajectory_rows.groupby(
        ["seed"] + observation_keys, sort=True, observed=True
    ).agg(
        operational_cost=("operational_cost", "mean"),
        severe_miss=("severe_miss", "mean"),
        correct=("correct", "mean"),
        origins=("physical_origin", "nunique"),
        repeated_rows=("physical_origin", "size"),
    ).reset_index()
    if not (origin_stage["origins"] == 8).all() or not (
        origin_stage["repeated_rows"] == 8
    ).all():
        raise ValueError("Origin averaging did not retain exactly eight repeated measures")
    collapsed = origin_stage.groupby(observation_keys, sort=True, observed=True).agg(
        operational_cost=("operational_cost", "mean"),
        severe_miss=("severe_miss", "mean"),
        correct=("correct", "mean"),
        seeds=("seed", "nunique"),
        repeated_rows=("seed", "size"),
    ).reset_index()
    if not (collapsed["seeds"] == len(EXPECTED_SEEDS)).all() or not (
        collapsed["repeated_rows"] == len(EXPECTED_SEEDS)
    ).all():
        raise ValueError("Seed averaging did not retain exactly five repeated measures")
    if (
        (raw_decision_rows is not None and int(raw_decision_rows) != expected_raw_rows)
        or len(trajectory_rows) != expected_trajectory_rows
        or len(origin_stage) != expected_origin_rows
        or len(collapsed) != expected_seed_rows
    ):
        raise ValueError("Registered aggregate averaging-stage counts changed")
    units = ("dependency_group_id", "spatial_block_id")
    policy_summary: dict[str, Any] = {}
    for policy in policies:
        subset = collapsed[collapsed["policy"] == policy]
        policy_summary[policy] = {}
        per_rotation = trajectory_rows[trajectory_rows["policy"] == policy]
        for unit in units:
            grouped = subset.groupby(unit, sort=True).agg(
                operational_cost=("operational_cost", "mean"),
                severe_miss=("severe_miss", "mean"),
                correct=("correct", "mean"),
            )
            rotation_cost = (
                per_rotation.groupby(["physical_origin", unit], sort=True)[
                    "operational_cost"
                ]
                .mean()
                .groupby("physical_origin")
                .mean()
            )
            policy_summary[policy][unit] = {
                "units": int(len(grouped)),
                "operational_cost": float(grouped["operational_cost"].mean()),
                "severe_miss_rate": float(grouped["severe_miss"].mean()),
                "accuracy": float(grouped["correct"].mean()),
                "rotation_cost_median": float(rotation_cost.median()),
                "rotation_cost_min": float(rotation_cost.min()),
                "rotation_cost_max": float(rotation_cost.max()),
                "rotation_cost_range": float(rotation_cost.max() - rotation_cost.min()),
                "rotation_cost_worst": float(rotation_cost.max()),
            }
    contrast_rows: list[dict[str, Any]] = []
    contrast_json: dict[str, Any] = {}
    metric_columns = ["operational_cost", "severe_miss", "correct"]
    key_columns = [
        "sample_id",
        "dependency_group_id",
        "spatial_block_id",
        "joint_component_id",
    ]
    main_rows = collapsed[collapsed["policy"] == main][key_columns + metric_columns]
    for comparator in policies:
        if comparator == main:
            continue
        other = collapsed[collapsed["policy"] == comparator][key_columns + metric_columns]
        paired = main_rows.merge(
            other,
            on=key_columns,
            how="inner",
            validate="one_to_one",
            suffixes=("_main", "_comparator"),
        )
        if len(paired) != len(main_rows) or len(paired) != len(other):
            raise ValueError(f"Paired policy support changed for {comparator}")
        paired["cost_difference_comparator_minus_main"] = (
            paired["operational_cost_comparator"] - paired["operational_cost_main"]
        )
        paired["severe_difference_comparator_minus_main"] = (
            paired["severe_miss_comparator"] - paired["severe_miss_main"]
        )
        contrast_json[comparator] = {}
        per_rotation_main = trajectory_rows[trajectory_rows["policy"] == main]
        per_rotation_other = trajectory_rows[trajectory_rows["policy"] == comparator]
        rotation_pair = per_rotation_main.merge(
            per_rotation_other,
            on=["seed", "physical_origin"] + key_columns + ["target"],
            validate="one_to_one",
            suffixes=("_main", "_comparator"),
        )
        rotation_pair["cost_difference"] = (
            rotation_pair["operational_cost_comparator"]
            - rotation_pair["operational_cost_main"]
        )
        for unit in units:
            grouped = paired.groupby(unit, sort=True).agg(
                cost_difference=("cost_difference_comparator_minus_main", "mean"),
                severe_difference=("severe_difference_comparator_minus_main", "mean"),
                cluster_id=("joint_component_id", "first"),
                joint_component_memberships=("joint_component_id", "nunique"),
            )
            if not (grouped["joint_component_memberships"] == 1).all():
                raise ValueError(f"Estimand unit {unit} crosses a joint component")
            # The repeated origins and model seeds stay paired inside each unit;
            # only the registered unit vector is resampled.
            secondary_low, secondary_high = _bootstrap_interval(
                grouped["cost_difference"].to_numpy(),
                resamples=int(bootstrap["resamples"]),
                seed=int(bootstrap["random_seed"]),
                confidence=float(bootstrap["confidence_level"]),
            )
            primary_low, primary_high = _cluster_bootstrap_interval(
                grouped.reset_index(),
                value_column="cost_difference",
                cluster_column="cluster_id",
                resamples=int(bootstrap["resamples"]),
                seed=int(bootstrap["random_seed"]),
                confidence=float(bootstrap["confidence_level"]),
            )
            rotation_values = (
                rotation_pair.groupby(["physical_origin", unit], sort=True)[
                    "cost_difference"
                ]
                .mean()
                .groupby("physical_origin")
                .mean()
            )
            record = {
                "comparator": comparator,
                "comparator_scope": (
                    "comparator"
                    if comparator in evaluation["comparators"]
                    else (
                        "off_support_ood_diagnostic"
                        if comparator in evaluation["ood_diagnostics"]
                        else "privileged_diagnostic"
                    )
                ),
                "main_policy": main,
                "aggregation_unit": unit,
                "units": int(len(grouped)),
                "cost_difference_comparator_minus_main": float(
                    grouped["cost_difference"].mean()
                ),
                "primary_joint_component_bootstrap_95_ci_low": primary_low,
                "primary_joint_component_bootstrap_95_ci_high": primary_high,
                "secondary_direct_unit_bootstrap_95_ci_low": secondary_low,
                "secondary_direct_unit_bootstrap_95_ci_high": secondary_high,
                "severe_difference_comparator_minus_main": float(
                    grouped["severe_difference"].mean()
                ),
                "rotation_contrast_median": float(rotation_values.median()),
                "rotation_contrast_min": float(rotation_values.min()),
                "rotation_contrast_max": float(rotation_values.max()),
                "rotation_contrast_range": float(
                    rotation_values.max() - rotation_values.min()
                ),
                "rotation_contrast_worst_for_main": float(rotation_values.min()),
                "descriptive_only": True,
                "go_no_go_test": False,
            }
            contrast_rows.append(record)
            contrast_json[comparator][unit] = record
    membership = (
        collapsed[
            [
                "sample_id",
                "dependency_group_id",
                "spatial_block_id",
                "joint_component_id",
            ]
        ]
        .drop_duplicates()
        .sort_values("sample_id", kind="mergesort")
    )
    aggregate = {
        "schema_version": AGGREGATE_SCHEMA,
        "claim_scope": config["claim_scope"],
        "confirmatory_eligible": False,
        "go_no_go_decision_performed": False,
        "interpretation": "sensitivity_only_no_confirmation",
        "averaging_order": evaluation["averaging_order"],
        "estimator": {
            "contrast_orientation": "comparator_operational_cost_minus_relative_geometry_utility_operational_cost",
            "trajectory_stage": "mean trajectories within sample-policy-seed-origin",
            "origin_stage": "mean eight origins within sample-policy-seed",
            "seed_stage": "mean five frozen model seeds within sample-policy",
            "macro_stage": "equal arithmetic mean of registered estimand units",
        },
        "averaging_stage_rows": {
            "raw_decision_rows": (
                int(raw_decision_rows) if raw_decision_rows is not None else None
            ),
            "post_trajectory_rows": int(len(trajectory_rows)),
            "post_origin_rows": int(len(origin_stage)),
            "post_seed_rows": int(len(collapsed)),
        },
        "component_membership_sha256": _canonical_hash(
            {"rows": membership.to_dict(orient="records")}
        ),
        "bootstrap": {
            "primary_resampling_unit": bootstrap["primary_resampling_unit"],
            "joint_components": int(membership["joint_component_id"].nunique()),
            "secondary_descriptive_resampling_units": bootstrap[
                "secondary_descriptive_resampling_units"
            ],
            "resamples": int(bootstrap["resamples"]),
            "random_seed": int(bootstrap["random_seed"]),
            "confidence_level": float(bootstrap["confidence_level"]),
            "interval": bootstrap["interval"],
            "quantile_method": "linear",
            "numpy_generator": "default_rng",
            "numpy_bit_generator": "PCG64",
            "seeds_and_origins_resampled": False,
        },
        "seeds": list(EXPECTED_SEEDS),
        "physical_origins": list(EXPECTED_ROTATIONS),
        "policy_summary": policy_summary,
        "contrasts": contrast_json,
    }
    return aggregate, pd.DataFrame(contrast_rows)


def _render_report(aggregate: Mapping[str, Any], contrasts: pd.DataFrame) -> str:
    lines = [
        "# Milton zero-shot active-view sensitivity v1",
        "",
        "**Status: sensitivity only; no GO/NO-GO decision and no confirmatory claim.**",
        "",
        "The frozen CVIAN classifier and selectors were applied without Milton fitting, "
        "calibration, threshold selection, or policy selection. Milton influenced prior "
        "hypothesis development and has neither observed sequence IDs nor compass metadata.",
        "",
        "## Registered estimand",
        "",
        "The main policy rolls sector content so each physical origin maps to local sector 0. "
        "Random trajectories are averaged first, then eight origins, then five frozen model "
        "seeds within each observation before equal-unit macro aggregation. Positive contrasts "
        "below mean comparator cost minus relative-policy cost.",
        "",
        "## Descriptive contrasts",
        "",
        "| Comparator | Scope | Unit | Cost difference | 95% descriptive CI | Rotation median | Rotation range | Worst rotation |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in contrasts.to_dict(orient="records"):
        lines.append(
            "| {comparator} | {scope} | {unit} | {difference:.6f} | "
            "[{low:.6f}, {high:.6f}] | {median:.6f} | {range:.6f} | {worst:.6f} |".format(
                comparator=row["comparator"],
                scope=row["comparator_scope"],
                unit=row["aggregation_unit"],
                difference=row["cost_difference_comparator_minus_main"],
                low=row["primary_joint_component_bootstrap_95_ci_low"],
                high=row["primary_joint_component_bootstrap_95_ci_high"],
                median=row["rotation_contrast_median"],
                range=row["rotation_contrast_range"],
                worst=row["rotation_contrast_worst_for_main"],
            )
        )
    lines.extend(
        [
            "",
            "The primary descriptive bootstrap clusters the joint dependency-group/spatial-block "
            "connected components; dependency-group and spatial-block rows remain separate "
            "sensitivity estimands. Model seeds and origins are repeated measurements, not "
            "resampling units.",
            "",
            "The absolute-aware result is explicitly an unrolled classifier-off-support OOD "
            "diagnostic. Max-confidence and label-oracle rows are privileged diagnostics. "
            "Neither may be promoted to headline transfer evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_input_inventory(
    config: Mapping[str, Any], config_path: Path
) -> tuple[dict[str, Any], dict[int, dict[str, Path]]]:
    source = config["source_contract"]
    if _sha256(config_path) != FROZEN_CONFIG_SHA256:
        raise ValueError("Frozen Milton sensitivity config SHA-256 changed")
    manifest_contract_path = _resolve(source["manifest_contract"])
    if _sha256(manifest_contract_path) != source["manifest_contract_sha256"]:
        raise ValueError("Frozen Milton manifest contract changed")
    manifest_contract = _load_json(manifest_contract_path)
    transfer_manifest_path = (
        manifest_contract_path.parent / manifest_contract["manifest"]["filename"]
    )
    canonical_source_path = Path(str(manifest_contract["source"]["path"]))
    provenance_audit_path = Path(
        str(manifest_contract["source"]["provenance_audit_path"])
    )
    if _sha256(transfer_manifest_path) != source["transfer_manifest_sha256"]:
        raise ValueError("Frozen Milton transfer manifest changed")
    if _sha256(canonical_source_path) != source["canonical_source_cohort_sha256"]:
        raise ValueError("Frozen Milton canonical source cohort changed")
    if _sha256(provenance_audit_path) != manifest_contract["source"].get(
        "provenance_audit_sha256"
    ):
        raise ValueError("Frozen Milton canonical provenance audit changed")
    cache_entrypoint_path = REPO_ROOT / "scripts" / "cache_milton_active_view_embeddings.py"
    manifest_builder_path = (
        REPO_ROOT / "scripts" / "build_milton_active_view_transfer_manifest.py"
    )
    if _sha256(cache_entrypoint_path) != source["cache_entrypoint_sha256"] or _sha256(
        manifest_builder_path
    ) != source["manifest_builder_sha256"]:
        raise ValueError("Frozen Milton cache/manifest implementation changed")
    cache_summary_path = _resolve(source["cache_summary"])
    if _sha256(cache_summary_path) != source["cache_summary_sha256"]:
        raise ValueError("Frozen Milton cache summary hash changed")
    summary = _load_json(cache_summary_path)
    _require_fields(
        summary,
        {
            "schema_version": CACHE_SUMMARY_SCHEMA,
            "claim_scope": "sensitivity_only_not_confirmatory",
            "confirmatory_eligible": False,
            "forward_only": True,
        },
        "Milton cache summary",
    )
    runs = summary.get("runs", [])
    if [int(run.get("seed", -1)) for run in runs] != list(EXPECTED_SEEDS):
        raise ValueError("Milton cache seed inventory changed")
    runner_path = REPO_ROOT / "scripts" / "run_cvian_sequence_utility_experiment.py"
    if _sha256(runner_path) != FROZEN_CVIAN_RUNNER_SHA256:
        raise ValueError("Frozen CVIAN utility runner changed")
    fit_root = _resolve(config["cvian_relative_policy_fit"]["source_fit_root"])
    relative_root = _resolve(config["cvian_relative_policy_fit"]["output_root"])
    relative_summary_path = relative_root / "fit_summary.json"
    if not relative_summary_path.is_file():
        raise FileNotFoundError(relative_summary_path)
    relative_summary = _load_json(relative_summary_path)
    _require_fields(
        relative_summary,
        {
            "schema_version": "cvian-relative-geometry-utility-summary-v1",
            "config_sha256": FROZEN_CONFIG_SHA256,
            "fit_entrypoint_sha256": config["implementation_contract"][
                "fit_entrypoint_sha256"
            ],
            "feature_interface_sha256": config["implementation_contract"][
                "feature_interface_sha256"
            ],
            "seeds": list(EXPECTED_SEEDS),
            "prospective_test_loaded": False,
            "milton_data_loaded": False,
            "milton_fit_calibration_or_selection_performed": False,
        },
        "relative fit summary",
    )
    per_seed: dict[int, dict[str, Path]] = {}
    inventory_runs: dict[str, Any] = {}
    for run in runs:
        seed = int(run["seed"])
        cache_path = Path(str(run["cache"]["output"]))
        if not cache_path.is_file() or _sha256(cache_path) != run["cache"]["sha256"]:
            raise ValueError(f"Milton cache artifact changed for seed {seed}")
        if run.get("schema_version") != CACHE_SCHEMA or run.get("parameters_updated") is not False:
            raise ValueError(f"Milton forward-cache contract changed for seed {seed}")
        expected_manifest_hashes = {
            "manifest_contract_sha256": source["manifest_contract_sha256"],
            "manifest_sha256": source["transfer_manifest_sha256"],
            "source_cohort_sha256": source["canonical_source_cohort_sha256"],
        }
        if any(
            run.get("manifest_hashes", {}).get(field) != value
            for field, value in expected_manifest_hashes.items()
        ) or run.get("entrypoint_sha256", {}).get("cache_script") != source[
            "cache_entrypoint_sha256"
        ]:
            raise ValueError(f"Milton cache recursive source chain changed for seed {seed}")
        main_dir = fit_root / f"seed{seed}"
        main_completion_path = main_dir / "fit_complete.json"
        main_artifact_path = main_dir / "fit_artifact.pt"
        main_completion = _load_json(main_completion_path)
        if main_completion.get("schema_version") != MAIN_FIT_SCHEMA:
            raise ValueError(f"Main CVIAN fit schema changed for seed {seed}")
        if main_completion.get("prospective_test_loaded") is not False:
            raise ValueError(f"Main CVIAN fit role isolation changed for seed {seed}")
        if main_completion.get("source_sha256", {}).get("runner") != FROZEN_CVIAN_RUNNER_SHA256:
            raise ValueError(f"Main CVIAN runner provenance changed for seed {seed}")
        if _sha256(main_artifact_path) != main_completion.get("fit_artifact_sha256"):
            raise ValueError(f"Main CVIAN fit artifact changed for seed {seed}")
        relative_dir = relative_root / f"seed{seed}"
        relative_completion_path = relative_dir / "fit_complete.json"
        relative_artifact_path = relative_dir / "relative_utility_artifact.pt"
        relative_completion = _load_json(relative_completion_path)
        if relative_completion.get("schema_version") != RELATIVE_FIT_SCHEMA:
            raise ValueError(f"Relative fit schema changed for seed {seed}")
        if _sha256(relative_artifact_path) != relative_completion.get("artifact_sha256"):
            raise ValueError(f"Relative policy artifact changed for seed {seed}")
        if relative_summary.get("seed_completion_sha256", {}).get(str(seed)) != _sha256(
            relative_completion_path
        ) or relative_summary.get("seed_artifact_sha256", {}).get(str(seed)) != _sha256(
            relative_artifact_path
        ):
            raise ValueError(f"Relative fit summary chain changed for seed {seed}")
        _require_fields(
            relative_completion,
            {
                "seed": seed,
                "config_sha256": FROZEN_CONFIG_SHA256,
                "feature_schema": RELATIVE_GEOMETRY_FEATURE_SCHEMA,
                "feature_width": len(RELATIVE_GEOMETRY_FEATURE_NAMES),
                "classifier_parameters_updated": False,
                "base_encoder_parameters_updated": False,
                "temperature_updated": False,
                "milton_data_loaded": False,
                "prospective_test_loaded": False,
                "milton_fit_calibration_or_selection_performed": False,
            },
            f"relative fit completion seed {seed}",
        )
        for field, expected_hash in config["implementation_contract"].items():
            if not field.endswith("_sha256"):
                continue
            completion_field = {
                "fit_entrypoint_sha256": "fit_entrypoint",
                "feature_interface_sha256": "feature_interface",
                "frozen_cvian_runner_sha256": "frozen_cvian_runner",
                "frozen_active_view_sha256": "frozen_active_view",
                "frozen_active_view_utility_sha256": "frozen_active_view_utility",
            }[field]
            if relative_completion.get("source_sha256", {}).get(completion_field) != expected_hash:
                raise ValueError(
                    f"Relative completion implementation chain changed for seed {seed}: {field}"
                )
        frozen_classifier = relative_completion.get("source_provenance", {}).get(
            "frozen_classifier", {}
        )
        if frozen_classifier.get("fit_completion_sha256") != _sha256(
            main_completion_path
        ) or frozen_classifier.get("fit_artifact_sha256") != _sha256(main_artifact_path):
            raise ValueError(f"Relative/main classifier chain changed for seed {seed}")
        per_seed[seed] = {
            "cache": cache_path,
            "main_completion": main_completion_path,
            "main_artifact": main_artifact_path,
            "relative_completion": relative_completion_path,
            "relative_artifact": relative_artifact_path,
        }
        inventory_runs[str(seed)] = {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in per_seed[seed].items()
        }
    inventory = {
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "milton_source": {
            "manifest_contract": {
                "path": str(manifest_contract_path),
                "sha256": _sha256(manifest_contract_path),
            },
            "transfer_manifest": {
                "path": str(transfer_manifest_path),
                "sha256": _sha256(transfer_manifest_path),
            },
            "canonical_source_cohort": {
                "path": str(canonical_source_path),
                "sha256": _sha256(canonical_source_path),
            },
            "canonical_provenance_audit": {
                "path": str(provenance_audit_path),
                "sha256": _sha256(provenance_audit_path),
            },
            "cache_entrypoint": {
                "path": str(cache_entrypoint_path),
                "sha256": _sha256(cache_entrypoint_path),
            },
            "manifest_builder": {
                "path": str(manifest_builder_path),
                "sha256": _sha256(manifest_builder_path),
            },
        },
        "source": {
            "scorer": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
            "feature_interface": {
                "path": str(REPO_ROOT / "crossview_conflict" / "decision" / "relative_geometry_utility.py"),
                "sha256": _sha256(
                    REPO_ROOT / "crossview_conflict" / "decision" / "relative_geometry_utility.py"
                ),
            },
            "active_view": {
                "path": str(REPO_ROOT / "crossview_conflict" / "decision" / "active_view.py"),
                "sha256": _sha256(REPO_ROOT / "crossview_conflict" / "decision" / "active_view.py"),
            },
            "active_view_utility": {
                "path": str(REPO_ROOT / "crossview_conflict" / "decision" / "active_view_utility.py"),
                "sha256": _sha256(
                    REPO_ROOT / "crossview_conflict" / "decision" / "active_view_utility.py"
                ),
            },
            "frozen_cvian_runner": {"path": str(runner_path), "sha256": _sha256(runner_path)},
        },
        "cache_summary": {"path": str(cache_summary_path), "sha256": _sha256(cache_summary_path)},
        "relative_fit_summary": {
            "path": str(relative_summary_path),
            "sha256": _sha256(relative_summary_path),
        },
        "seed_inputs": inventory_runs,
    }
    return inventory, per_seed


def _load_frozen_models(
    paths: Mapping[str, Path], *, seed: int, device: str
) -> FrozenModels:
    main_completion = _load_json(paths["main_completion"])
    relative_completion = _load_json(paths["relative_completion"])
    main = torch.load(paths["main_artifact"], map_location=device, weights_only=False)
    relative = torch.load(
        paths["relative_artifact"], map_location=device, weights_only=False
    )
    if main.get("schema_version") != MAIN_POLICY_SCHEMA:
        raise ValueError(f"Unsupported main policy artifact for seed {seed}")
    if relative.get("schema_version") != RELATIVE_POLICY_SCHEMA:
        raise ValueError(f"Unsupported relative policy artifact for seed {seed}")
    if int(relative.get("utility_input_dim", -1)) != len(RELATIVE_GEOMETRY_FEATURE_NAMES):
        raise ValueError(f"Relative feature width changed for seed {seed}")
    main_metadata = main.get("metadata", {})
    relative_metadata = relative.get("metadata", {})
    if (
        main_metadata.get("prospective_test_loaded") is not False
        or main_metadata.get("fit_fingerprint_sha256")
        != main_completion.get("fit_fingerprint_sha256")
    ):
        raise ValueError(f"Main policy metadata/completion chain changed for seed {seed}")
    if relative_metadata.get("fit_fingerprint_sha256") != relative_completion.get(
        "fit_fingerprint_sha256"
    ) or relative_metadata.get("config_sha256") != FROZEN_CONFIG_SHA256:
        raise ValueError(f"Relative policy metadata/completion chain changed for seed {seed}")
    temperature = float(main["temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"Frozen classifier temperature is invalid for seed {seed}")
    classifier = ActiveViewMLP(int(main["classifier_input_dim"]), 3).to(device)
    classifier.load_state_dict(main["classifier_state_dict"])
    absolute = ActiveViewMLP(int(main["utility_input_dim"]), 1).to(device)
    absolute.load_state_dict(main["utility_state_dict"])
    relative_model = ActiveViewMLP(int(relative["utility_input_dim"]), 1).to(device)
    relative_model.load_state_dict(relative["utility_state_dict"])
    for model in (classifier, absolute, relative_model):
        model.eval()
        model.requires_grad_(False)
    return FrozenModels(
        classifier=classifier,
        classifier_standardizer=Standardizer(
            np.asarray(main["classifier_mean"], dtype=np.float32),
            np.asarray(main["classifier_scale"], dtype=np.float32),
        ),
        absolute_regressor=absolute,
        absolute_standardizer=Standardizer(
            np.asarray(main["utility_mean"], dtype=np.float32),
            np.asarray(main["utility_scale"], dtype=np.float32),
        ),
        relative_regressor=relative_model,
        relative_standardizer=Standardizer(
            np.asarray(relative["utility_mean"], dtype=np.float32),
            np.asarray(relative["utility_scale"], dtype=np.float32),
        ),
        temperature=temperature,
    )


def evaluate_registered_sensitivity(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    _configure_deterministic_inference()
    config_path = _resolve(args.config)
    if _sha256(config_path) != FROZEN_CONFIG_SHA256:
        raise ValueError("Frozen Milton sensitivity config SHA-256 changed")
    config = _load_json(config_path)
    _validate_config(config)
    inventory, paths_by_seed = _validate_input_inventory(config, config_path)
    inventory["runtime"] = _runtime_contract(
        device=str(args.device), batch_size=int(args.batch_size)
    )
    evaluation = config["evaluation"]
    output_root = _resolve(evaluation["output_root"])
    registry_root = output_root.parent / "evaluation_registry"
    registry_path = registry_root / "milton_zero_shot_active_view_sensitivity_v1_started.json"
    registry_complete_path = (
        registry_root / "milton_zero_shot_active_view_sensitivity_v1_complete.json"
    )
    if (
        output_root.exists()
        or registry_path.exists()
        or registry_complete_path.exists()
        or REPORT_PATH.exists()
    ):
        raise FileExistsError(
            "Registered Milton sensitivity is one-shot; output, report, or global sentinel already exists"
        )
    fingerprint = _canonical_hash(inventory)
    commitment = {
        "schema_version": COMMITMENT_SCHEMA,
        "claim_scope": config["claim_scope"],
        "confirmatory_eligible": False,
        "go_no_go_criterion": None,
        "milton_parameter_fitting": False,
        "milton_calibration": False,
        "stop_or_adaptive_policy_enabled": False,
        "input_inventory": inventory,
        "evaluation_fingerprint_sha256": fingerprint,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    commitment_path = output_root / COMMITMENT_NAME
    _write_json_exclusive(commitment_path, commitment)
    started = {
        "schema_version": STARTED_SCHEMA,
        "evaluation_fingerprint_sha256": fingerprint,
        "commitment_sha256": _sha256(commitment_path),
        "status": "started_before_first_semantic_milton_cache_load",
        "rerun_permitted": False,
    }
    _write_json_exclusive(registry_path, started)
    _write_json_exclusive(output_root / STARTED_NAME, started)

    # This is the first semantic read of Milton labels/rows and occurs only
    # after both local and global started sentinels exist.
    transfer_manifest = _load_transfer_manifest_after_started(config)

    staged_trajectory: list[pd.DataFrame] = []
    output_hashes: dict[str, Any] = {}
    total_raw_rows = 0
    policies = _policy_inventory(config)
    expected_components = int(evaluation["bootstrap"]["expected_joint_components"])
    reference_ids: tuple[np.ndarray, ...] | None = None
    for seed in EXPECTED_SEEDS:
        print(f"score frozen Milton sensitivity seed={seed}", flush=True)
        paths = paths_by_seed[seed]
        data = _load_milton_cache(
            paths["cache"],
            expected_sha256=inventory["seed_inputs"][str(seed)]["cache"]["sha256"],
            expected_components=expected_components,
        )
        _validate_cache_against_manifest(data, transfer_manifest)
        identifiers = (
            data.cache.sample_id,
            data.dependency_group_id,
            data.cache.spatial_block_id,
            data.joint_component_id,
            data.cache.target,
        )
        if reference_ids is None:
            reference_ids = tuple(value.copy() for value in identifiers)
        elif any(not np.array_equal(left, right) for left, right in zip(reference_ids, identifiers)):
            raise ValueError("Milton sample/group/target order differs across model-seed caches")
        models = _load_frozen_models(paths, seed=seed, device=args.device)
        frames: list[pd.DataFrame] = []
        for physical_origin in EXPECTED_ROTATIONS:
            for policy in policies:
                frames.append(
                    _evaluate_policy(
                        policy=policy,
                        seed=seed,
                        physical_origin=physical_origin,
                        data=data,
                        models=models,
                        config=config,
                        device=args.device,
                        batch_size=args.batch_size,
                    )
                )
        decisions = pd.concat(frames, ignore_index=True)
        _validate_decision_lattice(decisions, policies=policies)
        metrics, trajectory = _metrics_for_seed(decisions)
        if len(metrics) != 126:
            raise ValueError("Registered per-seed metrics row count changed")
        total_raw_rows += len(decisions)
        staged_trajectory.append(trajectory)
        decision_path = output_root / evaluation["per_seed_decisions"].format(seed=seed)
        metric_path = output_root / evaluation["per_seed_metrics"].format(seed=seed)
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decisions.to_csv(decision_path, index=False, float_format="%.17g")
        metrics.to_csv(metric_path, index=False, float_format="%.17g")
        seed_completion_path = decision_path.parent / "evaluation_complete.json"
        seed_completion = {
            "schema_version": COMPLETION_SCHEMA,
            "scope": "per_seed",
            "seed": seed,
            "evaluation_fingerprint_sha256": fingerprint,
            "commitment_sha256": _sha256(commitment_path),
            "started_sha256": _sha256(output_root / STARTED_NAME),
            "seed_input_inventory": inventory["seed_inputs"][str(seed)],
            "row_counts": {
                "raw_decision_rows": int(len(decisions)),
                "post_trajectory_rows": int(len(trajectory)),
                "origins": int(decisions["physical_origin"].nunique()),
                "policies": int(decisions["policy"].nunique()),
                "samples": int(decisions["sample_id"].nunique()),
                "random_trajectories": int(
                    decisions.loc[decisions["policy"] == "random_mc32", "trajectory"].nunique()
                ),
            },
            "decisions_sha256": _sha256(decision_path),
            "metrics_sha256": _sha256(metric_path),
            "parameters_updated": False,
            "calibration_performed": False,
            "status": "seed_sensitivity_complete",
        }
        _write_json_exclusive(seed_completion_path, seed_completion)
        output_hashes[str(seed)] = {
            "decisions": {"path": str(decision_path), "sha256": _sha256(decision_path)},
            "metrics": {"path": str(metric_path), "sha256": _sha256(metric_path)},
            "completion": {
                "path": str(seed_completion_path),
                "sha256": _sha256(seed_completion_path),
            },
        }
        del models, decisions, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate, contrasts = _aggregate_sensitivity(
        pd.concat(staged_trajectory, ignore_index=True),
        config,
        raw_decision_rows=total_raw_rows,
    )
    if len(contrasts) != 12:
        raise ValueError("Registered aggregate contrast row count changed")
    aggregate_path = output_root / evaluation["aggregate_json"]
    contrast_path = output_root / evaluation["aggregate_csv"]
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    contrasts.to_csv(contrast_path, index=False, float_format="%.17g")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_render_report(aggregate, contrasts), encoding="utf-8")
    output_hashes["aggregate"] = {
        "json": {"path": str(aggregate_path), "sha256": _sha256(aggregate_path)},
        "csv": {"path": str(contrast_path), "sha256": _sha256(contrast_path)},
        "report": {"path": str(REPORT_PATH), "sha256": _sha256(REPORT_PATH)},
    }
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "claim_scope": config["claim_scope"],
        "confirmatory_eligible": False,
        "go_no_go_decision_performed": False,
        "evaluation_fingerprint_sha256": fingerprint,
        "commitment_sha256": _sha256(commitment_path),
        "started_sha256": _sha256(output_root / STARTED_NAME),
        "global_started_sha256": _sha256(registry_path),
        "parameters_updated": False,
        "calibration_performed": False,
        "stop_or_adaptive_policy_enabled": False,
        "outputs": output_hashes,
        "row_counts": {
            "raw_decision_rows": int(total_raw_rows),
            **aggregate["averaging_stage_rows"],
            "aggregate_contrast_rows": int(len(contrasts)),
        },
        "status": "sensitivity_complete_no_confirmation",
    }
    completion_path = output_root / evaluation["completion"]
    _write_json_exclusive(completion_path, completion)
    global_completion = {
        "schema_version": COMPLETION_SCHEMA,
        "scope": "global_registry",
        "evaluation_fingerprint_sha256": fingerprint,
        "commitment_sha256": _sha256(commitment_path),
        "started_sha256": _sha256(registry_path),
        "locked_output_root": str(output_root),
        "completion_sha256": _sha256(completion_path),
        "status": "sensitivity_complete_no_confirmation",
    }
    _write_json_exclusive(registry_complete_path, global_completion)
    print(json.dumps(completion, indent=2))


def main() -> None:
    evaluate_registered_sensitivity(parse_args())


if __name__ == "__main__":
    main()
