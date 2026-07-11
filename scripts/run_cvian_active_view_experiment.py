from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
import sklearn
from sklearn.metrics import accuracy_score, f1_score, recall_score

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.decision.active_view import (
    ActiveViewCache,
    ActiveViewMLP,
    DEFAULT_OPERATIONAL_COST,
    Standardizer,
    build_selector_features,
    build_state_features,
    entropy,
    expected_operational_cost,
    farthest_available_sector,
    load_active_view_cache,
    realized_operational_cost,
    softmax,
)


ONLINE_POLICIES = (
    "clockwise",
    "random",
    "farthest",
    "learned",
)

PRIVILEGED_POLICIES = (
    "max_building_privileged",
    "max_confidence_privileged",
    "entropy_reduction_privileged",
    "oracle_cost",
)

POLICIES = ONLINE_POLICIES + PRIVILEGED_POLICIES

REGISTERED_MAIN_BUDGET = 3
REGISTERED_VIEW_COST = 0.5
EVALUATION_SCHEMA = "cvian-active-view-development-evaluation-v2"


@dataclass(frozen=True)
class Schedule:
    sample_indices: np.ndarray
    masks: np.ndarray
    last_sector: np.ndarray
    remaining_budget: np.ndarray


@dataclass(frozen=True)
class NetworkResult:
    model: ActiveViewMLP
    standardizer: Standardizer
    best_epoch: int
    best_val_macro_f1: float
    history: list[dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a mask-aware CVIAN sequential-reveal aggregator "
            "and supervised next-sector selector on frozen spatial-v1 embeddings."
        )
    )
    parser.add_argument(
        "--cache-root",
        default="outputs/analysis/active_view_cvian_spatial_v1/cache",
    )
    parser.add_argument(
        "--visibility-dir",
        default="outputs/analysis/active_view_cvian_spatial_v1/visibility",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/analysis/active_view_cvian_spatial_v1/experiment",
    )
    parser.add_argument(
        "--manifest-summary",
        default="data/active_view/cvian_spatial_v1/manifest_summary.json",
    )
    parser.add_argument("--seeds", default="42,123,456,789,1011")
    parser.add_argument("--selector-block-fraction", type=float, default=0.20)
    parser.add_argument("--classifier-epochs", type=int, default=35)
    parser.add_argument("--selector-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--view-cost", type=float, default=REGISTERED_VIEW_COST)
    parser.add_argument("--main-budget", type=int, default=REGISTERED_MAIN_BUDGET)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--development-rerun-acknowledged",
        action="store_true",
        help=(
            "Required with --overwrite. The spatial-v1 test is already consumed; "
            "reruns are development diagnostics, never new confirmatory tests."
        ),
    )
    parser.add_argument(
        "--allow-historical-cache-lineage",
        action="store_true",
        help=(
            "Explicitly allow the already-created v1 embedding/visibility metadata. "
            "The visibility model revision is unresolved, so this is development-only."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate_input_metadata(
    *,
    args: argparse.Namespace,
    seed: int,
    cache_paths: dict[str, Path],
    cache_hashes: dict[str, str],
    cache_metadata_path: Path,
    visibility_dir: Path,
    visibility_metadata_path: Path,
    manifest_summary: Path,
) -> dict[str, object]:
    manifest = _load_json_object(manifest_summary)
    cache_metadata = _load_json_object(cache_metadata_path)
    visibility_metadata = _load_json_object(visibility_metadata_path)
    manifest_schema = str(manifest.get("schema_version", ""))
    if manifest_schema not in {
        "cvian-active-view-manifest-v1",
        "cvian-active-view-manifest-v2",
    }:
        raise ValueError(f"Unsupported active-view manifest schema: {manifest_schema}")
    if int(manifest.get("num_sectors", -1)) != 8:
        raise ValueError("Active-view manifest must contain eight canonical sectors")
    for field in ("horizontal_fov_deg", "vertical_fov_deg"):
        if not math.isclose(float(manifest.get(field, -1)), 90.0, abs_tol=1e-9):
            raise ValueError(f"Active-view manifest has unexpected {field}")
    manifest_hashes = manifest.get("manifest_sha256")
    if not isinstance(manifest_hashes, dict):
        raise ValueError("Active-view manifest summary is missing artifact hashes")
    for filename in ("train.csv", "val.csv", "test.csv", "active_view_episodes.csv"):
        artifact = manifest_summary.parent / filename
        if not artifact.is_file() or _sha256(artifact) != manifest_hashes.get(filename):
            raise ValueError(f"Active-view manifest artifact hash mismatch: {artifact}")

    cache_schema = str(cache_metadata.get("schema_version", ""))
    if cache_schema not in {"cvian-active-view-cache-v1", "cvian-active-view-cache-v2"}:
        raise ValueError(f"Unsupported embedding cache schema: {cache_schema}")
    if int(cache_metadata.get("seed", -1)) != seed:
        raise ValueError("Embedding cache seed metadata mismatch")
    for field, expected in (
        ("num_sectors", 8),
        ("horizontal_fov_deg", 90.0),
        ("vertical_fov_deg", 90.0),
    ):
        if not math.isclose(float(cache_metadata.get(field, -1)), expected, abs_tol=1e-9):
            raise ValueError(f"Embedding cache metadata mismatch for {field}")
    checkpoint_path = Path(str(cache_metadata.get("checkpoint", "")))
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != cache_metadata.get(
        "checkpoint_sha256"
    ):
        raise ValueError("Embedding cache checkpoint hash mismatch")
    cache_splits = cache_metadata.get("splits")
    if not isinstance(cache_splits, dict):
        raise ValueError("Embedding cache metadata is missing split attestations")
    for split, path in cache_paths.items():
        split_summary = cache_splits.get(split)
        if not isinstance(split_summary, dict) or split_summary.get("sha256") != cache_hashes[split]:
            raise ValueError(f"Embedding cache metadata hash mismatch: {path}")

    visibility_schema = str(visibility_metadata.get("schema_version", ""))
    if visibility_schema not in {"cvian-sector-visibility-v1", "cvian-sector-visibility-v2"}:
        raise ValueError(f"Unsupported visibility cache schema: {visibility_schema}")
    for field, expected in (
        ("num_sectors", 8),
        ("horizontal_fov_deg", 90.0),
        ("vertical_fov_deg", 90.0),
    ):
        if not math.isclose(
            float(visibility_metadata.get(field, -1)), expected, abs_tol=1e-9
        ):
            raise ValueError(f"Visibility cache metadata mismatch for {field}")
    visibility_splits = visibility_metadata.get("splits")
    if not isinstance(visibility_splits, dict):
        raise ValueError("Visibility cache metadata is missing split attestations")
    for split in ("train", "val", "test"):
        path = visibility_dir / f"{split}.npz"
        split_summary = visibility_splits.get(split)
        if not path.is_file() or not isinstance(split_summary, dict):
            raise ValueError(f"Missing visibility cache attestation: {path}")
        if _sha256(path) != split_summary.get("sha256"):
            raise ValueError(f"Visibility cache metadata hash mismatch: {path}")

    source_dirs = {
        str(Path(str(value)).resolve())
        for value in (
            manifest.get("source_split_dir"),
            cache_metadata.get("split_dir"),
            visibility_metadata.get("split_dir"),
        )
        if value
    }
    if len(source_dirs) != 1:
        raise ValueError("Manifest, embedding, and visibility caches use different split dirs")
    source_dir = Path(next(iter(source_dirs)))
    actual_source_hashes = {
        split: _sha256(source_dir / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    manifest_source_hashes = manifest.get("source_manifest_sha256")
    if manifest_schema.endswith("-v2") and manifest_source_hashes != actual_source_hashes:
        raise ValueError("Active-view manifest source split hashes are stale")
    if cache_schema.endswith("-v2") and cache_metadata.get(
        "source_manifest_sha256"
    ) != actual_source_hashes:
        raise ValueError("Embedding cache source split hashes do not match the manifest")
    if visibility_schema.endswith("-v2") and visibility_metadata.get(
        "source_manifest_sha256"
    ) != actual_source_hashes:
        raise ValueError("Visibility cache source split hashes do not match the manifest")

    legacy = (
        manifest_schema.endswith("-v1")
        or cache_schema.endswith("-v1")
        or visibility_schema.endswith("-v1")
    )
    if legacy and not args.allow_historical_cache_lineage:
        raise ValueError(
            "Historical v1 cache metadata requires --allow-historical-cache-lineage; "
            "the visibility model revision is unresolved and results are development-only"
        )
    return {
        "manifest_schema": manifest_schema,
        "embedding_cache_schema": cache_schema,
        "visibility_cache_schema": visibility_schema,
        "lineage_status": (
            "historical_v1_development_only_visibility_revision_unresolved"
            if legacy
            else "v2_fail_closed"
        ),
    }


def _run_provenance(
    args: argparse.Namespace,
    seed: int,
    cache_root: Path,
    visibility_dir: Path,
    manifest_summary: Path,
) -> dict[str, object]:
    config = {
        "seed": seed,
        "selector_block_fraction": args.selector_block_fraction,
        "classifier_epochs": args.classifier_epochs,
        "selector_epochs": args.selector_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "view_cost": args.view_cost,
        "main_budget": args.main_budget,
        "device": args.device,
        "historical_cache_lineage_allowed": args.allow_historical_cache_lineage,
        "policies": list(POLICIES),
        "cost_matrix": DEFAULT_OPERATIONAL_COST.tolist(),
        "initial_state": "post_overhead+sector_0",
        "evaluation_status": "development_test_consumed",
        "claim_scope": "offline sequential evidence reveal; risk-aware",
        "deterministic_algorithms": True,
    }
    source_paths = {
        "runner": Path(__file__).resolve(),
        "active_view": (
            REPO_ROOT / "crossview_conflict" / "decision" / "active_view.py"
        ).resolve(),
    }
    cache_paths = {
        split: (cache_root / f"seed{seed}" / f"{split}.npz").resolve()
        for split in ("train", "val", "test")
    }
    cache_metadata_path = (
        cache_root / f"seed{seed}" / "cache_metadata.json"
    ).resolve()
    visibility_path = (visibility_dir / "test.npz").resolve()
    visibility_metadata_path = (
        visibility_dir / "visibility_metadata.json"
    ).resolve()
    required = [
        *source_paths.values(),
        *cache_paths.values(),
        cache_metadata_path,
        visibility_path,
        visibility_metadata_path,
        manifest_summary,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing provenance input: {missing[0]}")
    cache_hashes = {name: _sha256(path) for name, path in cache_paths.items()}
    input_semantics = _validate_input_metadata(
        args=args,
        seed=seed,
        cache_paths=cache_paths,
        cache_hashes=cache_hashes,
        cache_metadata_path=cache_metadata_path,
        visibility_dir=visibility_dir,
        visibility_metadata_path=visibility_metadata_path,
        manifest_summary=manifest_summary,
    )
    payload: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA,
        "config": config,
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "input_sha256": {
            "cache": cache_hashes,
            "cache_metadata": _sha256(cache_metadata_path),
            "visibility_test": _sha256(visibility_path),
            "visibility_metadata": _sha256(visibility_metadata_path),
            "manifest_summary": _sha256(manifest_summary),
        },
        "runtime_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "input_semantics": input_semantics,
    }
    payload["run_fingerprint"] = _canonical_hash(payload)
    return payload


def _validate_completed_artifact(
    seed_dir: Path,
    completion: dict[str, object],
    expected: dict[str, object],
) -> None:
    if completion.get("schema_version") != EVALUATION_SCHEMA:
        raise ValueError(
            f"Stale completion schema in {seed_dir}; use the explicit development rerun flag"
        )
    if completion.get("run_fingerprint") != expected["run_fingerprint"]:
        raise ValueError(
            f"Completion fingerprint mismatch in {seed_dir}; refusing stale artifact reuse"
        )
    for filename, field in (
        ("metrics.csv", "metrics_sha256"),
        ("per_sample_decisions.csv", "decisions_sha256"),
        ("policy_artifact.pt", "artifact_sha256"),
        ("training_history.json", "training_history_sha256"),
    ):
        path = seed_dir / filename
        if not path.is_file() or _sha256(path) != completion.get(field):
            raise ValueError(f"Completion artifact hash mismatch: {path}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_model_selector_blocks(
    cache: ActiveViewCache,
    *,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.05 <= fraction <= 0.5:
        raise ValueError("selector-block-fraction must be between 0.05 and 0.5")
    groups = np.unique(cache.spatial_block_id)
    target_count = max(1, int(round(len(groups) * fraction)))
    for attempt in range(1000):
        rng = np.random.default_rng(seed + 104729 * attempt)
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        selector_groups = set(shuffled[:target_count].tolist())
        selector_mask = np.asarray(
            [group in selector_groups for group in cache.spatial_block_id], dtype=bool
        )
        model_indices = np.flatnonzero(~selector_mask)
        selector_indices = np.flatnonzero(selector_mask)
        if (
            len(model_indices) > 0
            and len(selector_indices) > 0
            and len(np.unique(cache.target[model_indices])) == 3
            and len(np.unique(cache.target[selector_indices])) == 3
        ):
            return model_indices, selector_indices
    raise ValueError("Could not build a class-complete block-disjoint model/selector split")


def _orders_for_sample(sector_count: int, *, seed: int, sample_index: int) -> list[list[int]]:
    clockwise = list(range(sector_count))
    remaining = list(range(1, sector_count))
    rng = np.random.default_rng(seed * 1_000_003 + sample_index)
    rng.shuffle(remaining)
    return [clockwise, [0, *remaining]]


def _build_schedule(
    cache: ActiveViewCache,
    sample_indices: np.ndarray,
    *,
    seed: int,
    include_overhead_only: bool,
    maximum_budget: int | None = None,
) -> Schedule:
    sectors = cache.sector_count
    max_budget = sectors if maximum_budget is None else min(maximum_budget, sectors)
    indices: list[int] = []
    masks: list[np.ndarray] = []
    last: list[int] = []
    remaining_budget: list[int] = []
    for sample_index in sample_indices:
        if include_overhead_only:
            indices.append(int(sample_index))
            masks.append(np.zeros(sectors, dtype=bool))
            last.append(-1)
            remaining_budget.append(sectors)
        for order in _orders_for_sample(sectors, seed=seed, sample_index=int(sample_index)):
            mask = np.zeros(sectors, dtype=bool)
            for budget, action in enumerate(order[:max_budget], 1):
                mask = mask.copy()
                mask[action] = True
                indices.append(int(sample_index))
                masks.append(mask)
                last.append(int(action))
                remaining_budget.append(sectors - budget)
    return Schedule(
        sample_indices=np.asarray(indices, dtype=np.int64),
        masks=np.stack(masks),
        last_sector=np.asarray(last, dtype=np.int64),
        remaining_budget=np.asarray(remaining_budget, dtype=np.float32),
    )


def _schedule_features(
    cache: ActiveViewCache,
    schedule: Schedule,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(schedule.sample_indices), chunk_size):
        stop = min(len(schedule.sample_indices), start + chunk_size)
        chunks.append(
            build_state_features(
                cache,
                schedule.masks[start:stop],
                sample_indices=schedule.sample_indices[start:stop],
                last_sector=schedule.last_sector[start:stop],
                remaining_budget=schedule.remaining_budget[start:stop],
            )
        )
    return np.concatenate(chunks)


def _class_weights(targets: np.ndarray, output_dim: int) -> torch.Tensor:
    counts = np.bincount(targets.astype(np.int64), minlength=output_dim).astype(np.float64)
    weights = len(targets) / np.maximum(counts * output_dim, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def _predict_network(
    model: ActiveViewMLP,
    standardizer: Standardizer,
    features: np.ndarray,
    *,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    values = standardizer.transform(features)
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            outputs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outputs)


def _train_network(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    *,
    output_dim: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> NetworkResult:
    _seed_everything(seed)
    standardizer = Standardizer.fit(train_features)
    train_values = standardizer.transform(train_features)
    model = ActiveViewMLP(train_values.shape[1], output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    weights = _class_weights(train_targets, output_dim).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_values),
            torch.from_numpy(train_targets.astype(np.int64)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_f1 = -math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            total_rows += len(targets)
        val_logits = _predict_network(
            model, standardizer, val_features, device=device
        )
        val_predictions = val_logits.argmax(axis=1)
        val_f1 = float(f1_score(val_targets, val_predictions, average="macro"))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(total_rows, 1),
                "val_macro_f1": val_f1,
                "val_accuracy": float(accuracy_score(val_targets, val_predictions)),
            }
        )
        if val_f1 > best_f1 + 1e-8:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return NetworkResult(
        model=model,
        standardizer=standardizer,
        best_epoch=best_epoch,
        best_val_macro_f1=float(best_f1),
        history=history,
    )


def _fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    labels = np.asarray(targets, dtype=np.int64)

    def objective(log_temperature: float) -> float:
        probabilities = softmax(logits, temperature=float(np.exp(log_temperature)))
        return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean())

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def _candidate_states(
    cache: ActiveViewCache,
    sample_indices: np.ndarray,
    masks: np.ndarray,
    *,
    remaining_budget: np.ndarray | int,
) -> tuple[Schedule, np.ndarray]:
    candidate_indices = []
    candidate_masks = []
    candidate_last = []
    candidate_parent = []
    remaining_values = np.asarray(remaining_budget, dtype=np.float32)
    if remaining_values.ndim == 0:
        remaining_values = np.full(len(sample_indices), float(remaining_values))
    if len(remaining_values) != len(sample_indices):
        raise ValueError("candidate remaining_budget length is inconsistent")
    candidate_remaining = []
    for parent, (sample_index, mask) in enumerate(zip(sample_indices, masks)):
        for action in np.flatnonzero(~mask):
            updated = mask.copy()
            updated[action] = True
            candidate_indices.append(int(sample_index))
            candidate_masks.append(updated)
            candidate_last.append(int(action))
            candidate_parent.append(parent)
            candidate_remaining.append(max(float(remaining_values[parent]), 0.0))
    return (
        Schedule(
            sample_indices=np.asarray(candidate_indices, dtype=np.int64),
            masks=np.stack(candidate_masks),
            last_sector=np.asarray(candidate_last, dtype=np.int64),
            remaining_budget=np.asarray(candidate_remaining, dtype=np.float32),
        ),
        np.asarray(candidate_parent, dtype=np.int64),
    )


def _build_action_dataset(
    cache: ActiveViewCache,
    sample_indices: np.ndarray,
    classifier: NetworkResult,
    *,
    temperature: float,
    seed: int,
    device: str,
    maximum_budget: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    state_indices = []
    state_masks = []
    state_last = []
    state_remaining = []
    for sample_index in sample_indices:
        for order in _orders_for_sample(
            cache.sector_count, seed=seed + 17, sample_index=int(sample_index)
        ):
            mask = np.zeros(cache.sector_count, dtype=bool)
            mask[order[0]] = True
            for revealed_count in range(1, min(maximum_budget, cache.sector_count)):
                state_indices.append(int(sample_index))
                state_masks.append(mask.copy())
                state_last.append(int(order[revealed_count - 1]))
                state_remaining.append(cache.sector_count - revealed_count)
                mask[order[revealed_count]] = True
    schedule = Schedule(
        sample_indices=np.asarray(state_indices, dtype=np.int64),
        masks=np.stack(state_masks),
        last_sector=np.asarray(state_last, dtype=np.int64),
        remaining_budget=np.asarray(state_remaining, dtype=np.float32),
    )
    state_features = _schedule_features(cache, schedule)
    current_logits = _predict_network(
        classifier.model, classifier.standardizer, state_features, device=device
    )
    current_probs = softmax(current_logits, temperature)
    selector_features = build_selector_features(state_features, current_probs)

    candidate_schedule, candidate_parent = _candidate_states(
        cache,
        schedule.sample_indices,
        schedule.masks,
        remaining_budget=np.maximum(schedule.remaining_budget - 1, 0),
    )
    candidate_features = _schedule_features(cache, candidate_schedule)
    candidate_logits = _predict_network(
        classifier.model, classifier.standardizer, candidate_features, device=device
    )
    candidate_probs = softmax(candidate_logits, temperature)
    candidate_targets = cache.target[candidate_schedule.sample_indices]
    candidate_costs = expected_operational_cost(candidate_probs, candidate_targets)
    best_cost = np.full(len(schedule.sample_indices), np.inf)
    best_action = np.full(len(schedule.sample_indices), -1, dtype=np.int64)
    for candidate_index, parent in enumerate(candidate_parent):
        action = int(candidate_schedule.last_sector[candidate_index])
        cost = float(candidate_costs[candidate_index])
        if cost < best_cost[parent] - 1e-12 or (
            math.isclose(cost, best_cost[parent], abs_tol=1e-12)
            and action < best_action[parent]
        ):
            best_cost[parent] = cost
            best_action[parent] = action
    if (best_action < 0).any():
        raise ValueError("Failed to construct an oracle action target")
    return selector_features, best_action


def _load_visibility(path: Path, cache: ActiveViewCache) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        sample_ids = payload["sample_id"].astype(str)
        ratios = payload["building_ratio"].astype(np.float32)
    index = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    missing = [sample_id for sample_id in cache.sample_id if sample_id not in index]
    if missing:
        raise ValueError(f"Visibility cache misses sample {missing[0]}")
    aligned = ratios[[index[sample_id] for sample_id in cache.sample_id]]
    if aligned.shape != (cache.sample_count, cache.sector_count):
        raise ValueError("Visibility cache shape is inconsistent")
    return aligned


def _select_online_action(
    policy: str,
    cache: ActiveViewCache,
    masks: np.ndarray,
    last_sector: np.ndarray,
    classifier: NetworkResult,
    selector: NetworkResult,
    *,
    temperature: float,
    seed: int,
    budget: int,
    device: str,
) -> np.ndarray:
    """Select an admissible action before any privileged/oracle diagnostic is built."""
    if policy not in ONLINE_POLICIES:
        raise ValueError(f"Not an online policy: {policy}")
    count = cache.sample_count
    actions = np.full(count, -1, dtype=np.int64)
    if policy == "learned":
        state_features = build_state_features(
            cache,
            masks,
            last_sector=last_sector,
            remaining_budget=cache.sector_count - budget,
        )
        current_logits = _predict_network(
            classifier.model, classifier.standardizer, state_features, device=device
        )
        current_probs = softmax(current_logits, temperature)
        selector_features = build_selector_features(state_features, current_probs)
        action_logits = _predict_network(
            selector.model, selector.standardizer, selector_features, device=device
        )
        action_logits[masks] = -np.inf
        actions = action_logits.argmax(axis=1)
    elif policy == "clockwise":
        for row in range(count):
            actions[row] = int(np.flatnonzero(~masks[row])[0])
    elif policy == "random":
        for row in range(count):
            available = np.flatnonzero(~masks[row])
            rng = np.random.default_rng(seed * 1_000_003 + row * 101 + budget)
            actions[row] = int(rng.choice(available))
    elif policy == "farthest":
        for row in range(count):
            actions[row] = farthest_available_sector(
                np.flatnonzero(masks[row]),
                np.flatnonzero(~masks[row]),
                sector_count=cache.sector_count,
            )
    if (actions < 0).any() or np.any(masks[np.arange(count), actions]):
        raise ValueError(f"Online policy {policy} selected an illegal action")
    return actions.astype(np.int64)


def _policy_action(
    policy: str,
    cache: ActiveViewCache,
    visibility: np.ndarray,
    masks: np.ndarray,
    last_sector: np.ndarray,
    classifier: NetworkResult,
    selector: NetworkResult,
    *,
    temperature: float,
    seed: int,
    budget: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = cache.sample_count
    actions = (
        _select_online_action(
            policy,
            cache,
            masks,
            last_sector,
            classifier,
            selector,
            temperature=temperature,
            seed=seed,
            budget=budget,
            device=device,
        )
        if policy in ONLINE_POLICIES
        else np.full(count, -1, dtype=np.int64)
    )
    oracle_actions = np.full(count, -1, dtype=np.int64)
    # Everything below this line is privileged/post-hoc diagnostic computation. Online
    # actions above have already been fixed and cannot depend on targets or hidden sectors.
    candidate_schedule, candidate_parent = _candidate_states(
        cache,
        np.arange(count),
        masks,
        remaining_budget=cache.sector_count - budget - 1,
    )
    candidate_features = _schedule_features(cache, candidate_schedule)
    candidate_logits = _predict_network(
        classifier.model, classifier.standardizer, candidate_features, device=device
    )
    candidate_probs = softmax(candidate_logits, temperature)
    candidate_targets = cache.target[candidate_schedule.sample_indices]
    candidate_costs = expected_operational_cost(candidate_probs, candidate_targets)
    candidate_entropies = entropy(candidate_probs)
    best_oracle_cost = np.full(count, np.inf)
    best_entropy = np.full(count, np.inf)
    entropy_actions = np.full(count, -1, dtype=np.int64)
    action_cost_lookup: list[dict[int, float]] = [dict() for _ in range(count)]
    for candidate_index, parent in enumerate(candidate_parent):
        action = int(candidate_schedule.last_sector[candidate_index])
        cost = float(candidate_costs[candidate_index])
        action_cost_lookup[parent][action] = cost
        if cost < best_oracle_cost[parent] - 1e-12 or (
            math.isclose(cost, best_oracle_cost[parent], abs_tol=1e-12)
            and action < oracle_actions[parent]
        ):
            best_oracle_cost[parent] = cost
            oracle_actions[parent] = action
        uncertainty = float(candidate_entropies[candidate_index])
        if uncertainty < best_entropy[parent] - 1e-12 or (
            math.isclose(uncertainty, best_entropy[parent], abs_tol=1e-12)
            and action < entropy_actions[parent]
        ):
            best_entropy[parent] = uncertainty
            entropy_actions[parent] = action

    if policy in ONLINE_POLICIES:
        pass
    elif policy == "max_building_privileged":
        scores = visibility.copy()
        scores[masks] = -np.inf
        actions = scores.argmax(axis=1)
    elif policy == "max_confidence_privileged":
        scores = softmax(cache.sector_logits).max(axis=2)
        scores[masks] = -np.inf
        actions = scores.argmax(axis=1)
    elif policy == "entropy_reduction_privileged":
        actions = entropy_actions
    elif policy == "oracle_cost":
        actions = oracle_actions
    else:
        raise ValueError(f"Unknown policy: {policy}")
    if (actions < 0).any() or np.any(masks[np.arange(count), actions]):
        raise ValueError(f"Policy {policy} selected an illegal action")
    regrets = np.asarray(
        [
            action_cost_lookup[row][int(actions[row])] - best_oracle_cost[row]
            for row in range(count)
        ],
        dtype=np.float64,
    )
    return actions.astype(np.int64), oracle_actions, regrets


def _classification_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    view_count: int,
    view_cost: float,
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    severe = targets == 2
    operational = realized_operational_cost(predictions, targets)
    nll = -np.log(
        np.clip(probabilities[np.arange(len(targets)), targets], 1e-12, 1.0)
    )
    acquisition = max(0, view_count - 1) * view_cost
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "severe_recall": float(recall_score(severe, predictions == 2, zero_division=0)),
        "severe_miss_rate": float(((predictions != 2) & severe).sum() / max(severe.sum(), 1)),
        "extreme_error_rate": float((np.abs(predictions - targets) == 2).mean()),
        "operational_cost": float(operational.mean()),
        "normalized_operational_cost": float(operational.mean() / DEFAULT_OPERATIONAL_COST.max()),
        "nll": float(nll.mean()),
        "mean_entropy": float(entropy(probabilities).mean()),
        "additional_acquisition_cost": float(acquisition),
        "total_cost": float(operational.mean() + acquisition),
    }


def _evaluate_policies(
    cache: ActiveViewCache,
    visibility: np.ndarray,
    classifier: NetworkResult,
    selector: NetworkResult,
    *,
    temperature: float,
    view_cost: float,
    seed: int,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    decision_rows = []
    for policy in POLICIES:
        masks = np.zeros((cache.sample_count, cache.sector_count), dtype=bool)
        masks[:, 0] = True
        last = np.zeros(cache.sample_count, dtype=np.int64)
        cumulative_regret = np.zeros(cache.sample_count, dtype=np.float64)
        for budget in range(1, cache.sector_count + 1):
            state_features = build_state_features(
                cache,
                masks,
                last_sector=last,
                remaining_budget=cache.sector_count - budget,
            )
            logits = _predict_network(
                classifier.model, classifier.standardizer, state_features, device=device
            )
            probabilities = softmax(logits, temperature)
            metrics = _classification_metrics(
                probabilities,
                cache.target,
                view_count=budget,
                view_cost=view_cost,
            )
            metrics_rows.append(
                {
                    "policy": policy,
                    "view_count": budget,
                    "mean_cumulative_action_regret": float(cumulative_regret.mean()),
                    **metrics,
                }
            )
            predictions = probabilities.argmax(axis=1)
            for row in range(cache.sample_count):
                decision_rows.append(
                    {
                        "sample_id": cache.sample_id[row],
                        "spatial_block_id": cache.spatial_block_id[row],
                        "sequence_id": cache.sequence_id[row],
                        "target": int(cache.target[row]),
                        "policy": policy,
                        "view_count": budget,
                        "revealed_sectors": ",".join(str(value) for value in np.flatnonzero(masks[row])),
                        "last_sector": int(last[row]),
                        "prediction": int(predictions[row]),
                        "prob_0": float(probabilities[row, 0]),
                        "prob_1": float(probabilities[row, 1]),
                        "prob_2": float(probabilities[row, 2]),
                        "operational_cost": float(
                            DEFAULT_OPERATIONAL_COST[cache.target[row], predictions[row]]
                        ),
                        "cumulative_action_regret": float(cumulative_regret[row]),
                    }
                )
            if budget == cache.sector_count:
                continue
            actions, oracle_actions, regrets = _policy_action(
                policy,
                cache,
                visibility,
                masks,
                last,
                classifier,
                selector,
                temperature=temperature,
                seed=seed,
                budget=budget,
                device=device,
            )
            cumulative_regret += regrets
            masks[np.arange(cache.sample_count), actions] = True
            last = actions
    return pd.DataFrame(metrics_rows), pd.DataFrame(decision_rows)


def _panorama_reference_metrics(cache: ActiveViewCache) -> dict[str, float]:
    probabilities = softmax(cache.panorama_logits)
    return _classification_metrics(
        probabilities, cache.target, view_count=0, view_cost=0.0
    )


def _write_seed_artifact(
    path: Path,
    classifier: NetworkResult,
    selector: NetworkResult,
    *,
    temperature: float,
    metadata: dict[str, object],
) -> None:
    torch.save(
        {
            "schema_version": "cvian-active-view-policy-v1",
            "classifier_state_dict": classifier.model.state_dict(),
            "classifier_mean": classifier.standardizer.mean,
            "classifier_scale": classifier.standardizer.scale,
            "selector_state_dict": selector.model.state_dict(),
            "selector_mean": selector.standardizer.mean,
            "selector_scale": selector.standardizer.scale,
            "temperature": temperature,
            "metadata": metadata,
        },
        path,
    )


def _run_seed(
    args: argparse.Namespace,
    seed: int,
    cache_root: Path,
    visibility_dir: Path,
    output_root: Path,
    manifest_summary: Path,
) -> dict[str, object]:
    seed_dir = output_root / f"seed{seed}"
    completion_path = seed_dir / "evaluation_complete.json"
    provenance = _run_provenance(
        args, seed, cache_root, visibility_dir, manifest_summary
    )
    existing_artifacts = [
        seed_dir / "metrics.csv",
        seed_dir / "per_sample_decisions.csv",
        seed_dir / "policy_artifact.pt",
        seed_dir / "locked_test_complete.json",
    ]
    if (
        not completion_path.is_file()
        and any(path.exists() for path in existing_artifacts)
        and not args.overwrite
    ):
        raise ValueError(
            f"Unattested or legacy artifacts exist in {seed_dir}; refusing implicit "
            "development-test rerun. Use --overwrite --development-rerun-acknowledged."
        )
    if completion_path.is_file() and not args.overwrite:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        _validate_completed_artifact(seed_dir, completion, provenance)
        return completion
    if not args.development_rerun_acknowledged:
        raise ValueError(
            "Scoring the consumed spatial-v1 development test requires "
            "--development-rerun-acknowledged, including for a new output directory"
        )
    seed_dir.mkdir(parents=True, exist_ok=True)
    train = load_active_view_cache(cache_root / f"seed{seed}" / "train.npz")
    val = load_active_view_cache(cache_root / f"seed{seed}" / "val.npz")
    test = load_active_view_cache(cache_root / f"seed{seed}" / "test.npz")
    if set(train.spatial_block_id) & (set(val.spatial_block_id) | set(test.spatial_block_id)):
        raise ValueError("Active-view caches violate spatial-block disjointness")
    if set(val.spatial_block_id) & set(test.spatial_block_id):
        raise ValueError("Validation and final-test spatial blocks overlap")
    model_indices, selector_indices = _split_model_selector_blocks(
        train,
        fraction=args.selector_block_fraction,
        seed=seed,
    )
    train_schedule = _build_schedule(
        train,
        model_indices,
        seed=seed,
        include_overhead_only=True,
    )
    val_schedule = _build_schedule(
        val,
        np.arange(val.sample_count),
        seed=seed + 1,
        include_overhead_only=True,
    )
    train_features = _schedule_features(train, train_schedule)
    val_features = _schedule_features(val, val_schedule)
    classifier = _train_network(
        train_features,
        train.target[train_schedule.sample_indices],
        val_features,
        val.target[val_schedule.sample_indices],
        output_dim=3,
        epochs=args.classifier_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=seed,
        device=args.device,
    )
    val_logits = _predict_network(
        classifier.model, classifier.standardizer, val_features, device=args.device
    )
    temperature = _fit_temperature(
        val_logits, val.target[val_schedule.sample_indices]
    )
    action_features, action_targets = _build_action_dataset(
        train,
        selector_indices,
        classifier,
        temperature=temperature,
        seed=seed,
        device=args.device,
    )
    val_action_features, val_action_targets = _build_action_dataset(
        val,
        np.arange(val.sample_count),
        classifier,
        temperature=temperature,
        seed=seed + 11,
        device=args.device,
    )
    selector = _train_network(
        action_features,
        action_targets,
        val_action_features,
        val_action_targets,
        output_dim=train.sector_count,
        epochs=args.selector_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=seed + 97,
        device=args.device,
    )
    visibility = _load_visibility(visibility_dir / "test.npz", test)
    metrics, decisions = _evaluate_policies(
        test,
        visibility,
        classifier,
        selector,
        temperature=temperature,
        view_cost=args.view_cost,
        seed=seed,
        device=args.device,
    )
    metrics.insert(0, "seed", seed)
    decisions.insert(0, "seed", seed)
    metrics.to_csv(seed_dir / "metrics.csv", index=False)
    decisions.to_csv(seed_dir / "per_sample_decisions.csv", index=False)
    metadata = {
        "seed": seed,
        "model_fit_rows": int(len(model_indices)),
        "model_fit_blocks": int(len(np.unique(train.spatial_block_id[model_indices]))),
        "selector_fit_rows": int(len(selector_indices)),
        "selector_fit_blocks": int(len(np.unique(train.spatial_block_id[selector_indices]))),
        "validation_rows": val.sample_count,
        "validation_blocks": int(len(np.unique(val.spatial_block_id))),
        "development_test_rows": test.sample_count,
        "development_test_blocks": int(len(np.unique(test.spatial_block_id))),
        "classifier_best_epoch": classifier.best_epoch,
        "classifier_val_macro_f1": classifier.best_val_macro_f1,
        "selector_best_epoch": selector.best_epoch,
        "selector_val_macro_f1": selector.best_val_macro_f1,
        "temperature": temperature,
        "main_budget": args.main_budget,
        "view_cost": args.view_cost,
        "initial_state": "post_overhead+sector_0",
        "online_policies": ["clockwise", "random", "farthest", "learned"],
        "privileged_policies": [
            "max_building_privileged",
            "max_confidence_privileged",
            "entropy_reduction_privileged",
            "oracle_cost",
        ],
        "claim_scope": "offline sequential evidence reveal; risk-aware",
        "evaluation_status": "development_test_consumed",
        "test_usage_note": (
            "The spatial-v1 test was inspected during protocol implementation and is "
            "not available for a future confirmatory claim."
        ),
        "cost_matrix": DEFAULT_OPERATIONAL_COST.tolist(),
        "base_encoder_role_isolation": "not_role_disjoint",
        "base_encoder_role_note": (
            "Frozen cross-view checkpoints were trained on the full spatial-v1 train "
            "role, including later selector-fit blocks. Only the downstream mask-aware "
            "head is disjoint from selector-fit blocks."
        ),
        "state_invariance_scope": (
            "reveal-order invariant for a fixed canonical sector indexing; not sector-index "
            "permutation invariant or cyclically equivariant"
        ),
        "oracle_definition": "label-aware greedy one-step expected-cost action",
        "random_trajectories_per_model_seed": 1,
        "panorama_reference": _panorama_reference_metrics(test),
        **provenance,
    }
    _write_seed_artifact(
        seed_dir / "policy_artifact.pt",
        classifier,
        selector,
        temperature=temperature,
        metadata=metadata,
    )
    (seed_dir / "training_history.json").write_text(
        json.dumps(
            {
                "classifier": classifier.history,
                "selector": selector.history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    completion = {
        "schema_version": EVALUATION_SCHEMA,
        **metadata,
        "metrics_sha256": _sha256(seed_dir / "metrics.csv"),
        "decisions_sha256": _sha256(seed_dir / "per_sample_decisions.csv"),
        "artifact_sha256": _sha256(seed_dir / "policy_artifact.pt"),
        "training_history_sha256": _sha256(seed_dir / "training_history.json"),
    }
    completion_path.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    return completion


def main() -> None:
    args = parse_args()
    if args.main_budget != REGISTERED_MAIN_BUDGET:
        raise ValueError(
            f"This protocol fixes main-budget={REGISTERED_MAIN_BUDGET}; "
            "create a new protocol/version for another primary budget"
        )
    if not math.isclose(args.view_cost, REGISTERED_VIEW_COST, abs_tol=1e-12):
        raise ValueError(
            f"This protocol fixes view-cost={REGISTERED_VIEW_COST}; "
            "create a new protocol/version for another acquisition cost"
        )
    if args.overwrite and not args.development_rerun_acknowledged:
        raise ValueError(
            "--overwrite requires --development-rerun-acknowledged because the test "
            "is already consumed"
        )
    cache_root = Path(args.cache_root)
    visibility_dir = Path(args.visibility_dir)
    output_root = Path(args.output_root)
    manifest_summary = Path(args.manifest_summary)
    if not cache_root.is_absolute():
        cache_root = (REPO_ROOT / cache_root).resolve()
    if not visibility_dir.is_absolute():
        visibility_dir = (REPO_ROOT / visibility_dir).resolve()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    if not manifest_summary.is_absolute():
        manifest_summary = (REPO_ROOT / manifest_summary).resolve()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    summaries = []
    for seed in seeds:
        print(f"active-view seed={seed}", flush=True)
        summaries.append(
            _run_seed(
                args,
                seed,
                cache_root,
                visibility_dir,
                output_root,
                manifest_summary,
            )
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "experiment_summary.json").write_text(
        json.dumps({"runs": summaries}, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
