"""Role-isolated CVIAN sequential utility experiment.

This runner has two deliberately separate phases:

``fit``
    Reads only ``base_fit``, ``selector_fit``, and ``validation`` caches.  It
    fits a probabilistic mask classifier, calibrates its temperature, fits the
    candidate net-utility regressor, and locks a fail-closed STOP policy.  The
    adaptive secondary analysis is explicitly risk-aware, not risk-controlled.

``evaluate``
    Verifies the protocol, code/config fingerprints, test commitment, cache
    hashes, and every fit artifact before loading ``prospective_test``.  The
    holdout is then scored once.  A completed evaluation is validation-only:
    there is intentionally no overwrite switch.

The split between the phases is a scientific guardrail, not just a convenient
CLI.  In particular, no function reachable from :func:`fit_all_seeds` accepts
or opens the prospective-test cache.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
from importlib.metadata import version as package_version
from itertools import combinations
import json
import math
import os
import platform
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# PyTorch deterministic CUDA matmul requires this to be present before the
# first CUDA context is created.  The value is also locked in config and every
# fit/evaluation fingerprint.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.decision.active_view import (
    ActiveViewCache,
    ActiveViewMLP,
    Standardizer,
    build_state_features,
    farthest_available_sector,
    load_active_view_cache,
    realized_operational_cost,
    softmax,
)
from crossview_conflict.decision.active_view_utility import (
    bayes_operational_action_and_risk,
    build_candidate_utility_features,
    select_adaptive_actions,
    target_conditioned_soft_loss,
)
from crossview_conflict.decision.artifact_provenance import (
    validate_seed_base_and_embedding_provenance,
    validate_supplemental_runtime_lock,
    validate_visibility_provenance,
)


CONFIG_SCHEMA = "cvian-sequence-utility-protocol-v1"
FIT_SCHEMA = "cvian-sequence-utility-fit-v1"
POLICY_SCHEMA = "cvian-sequence-utility-policy-v1"
COMMITMENT_SCHEMA = "cvian-sequence-utility-evaluation-commitment-v1"
EVALUATION_SCHEMA = "cvian-sequence-utility-evaluation-v1"
AGGREGATE_SCHEMA = "cvian-sequence-utility-primary-aggregate-v1"
AGGREGATE_COMPLETION_SCHEMA = (
    "cvian-sequence-utility-primary-aggregate-completion-v1"
)
REGISTRY_COMPLETION_SCHEMA = "cvian-sequence-utility-registry-completion-v1"
FIT_ROLES = ("base_fit", "selector_fit", "validation")
TEST_ROLE = "prospective_test"
EXPECTED_SEEDS = (42, 123, 456, 789, 1011)
PRIMARY_UTILITY_POLICY = "utility_regression"
PRIMARY_BASELINES = ("farthest", "max_building_privileged")
AGGREGATE_JSON_NAME = "aggregate_primary_analysis.json"
AGGREGATE_CSV_NAME = "aggregate_primary_comparisons.csv"
AGGREGATE_COMPLETION_NAME = "aggregate_complete.json"
PRIMARY_REPORT_PATH = REPO_ROOT / "docs" / "results" / "cvian_sequence_utility_v1.md"


@dataclass(frozen=True)
class RoleData:
    cache: ActiveViewCache
    compass_angle_deg: np.ndarray
    compass_available: np.ndarray


@dataclass(frozen=True)
class StateTable:
    sample_indices: np.ndarray
    revealed_mask: np.ndarray
    revealed_count: np.ndarray
    remaining_acquisitions: np.ndarray


@dataclass(frozen=True)
class ClassifierResult:
    model: ActiveViewMLP
    standardizer: Standardizer
    best_epoch: int
    best_validation_nll: float
    history: list[dict[str, float]]


@dataclass(frozen=True)
class UtilityTable:
    features: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    state_id: np.ndarray
    sample_index: np.ndarray
    sequence_id: np.ndarray
    revealed_count: np.ndarray
    candidate_sector: np.ndarray


@dataclass(frozen=True)
class RegressorResult:
    model: ActiveViewMLP
    standardizer: Standardizer
    best_epoch: int
    best_validation_sequence_macro_regret: float
    history: list[dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "evaluate"), default="fit")
    parser.add_argument("--config", default="configs/cvian_sequence_utility_v1.json")
    parser.add_argument(
        "--protocol-dir", default="data/splits/ian_hurricane_sequence_four_role_v1"
    )
    parser.add_argument(
        "--cache-root", default="outputs/cvian_sequence_active_v2/cache"
    )
    parser.add_argument(
        "--base-root", default="outputs/cvian_sequence_active_v2/base_encoders"
    )
    parser.add_argument(
        "--output-root", default="outputs/cvian_sequence_active_v2/utility_experiment"
    )
    parser.add_argument(
        "--visibility-dir", default="outputs/cvian_sequence_active_v2/visibility"
    )
    parser.add_argument("--classifier-epochs", type=int, default=40)
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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _runtime_versions() -> dict[str, Any]:
    """Runtime identity bound into fit and evaluation fingerprints."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": package_version("scipy"),
        "scikit_learn": package_version("scikit-learn"),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config(config: Mapping[str, Any]) -> None:
    """Fail closed if the registered protocol is silently changed."""
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("Unsupported CVIAN sequence utility config schema")
    if tuple(config.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError(f"Registered seeds must be {EXPECTED_SEEDS}")
    if config.get("roles") != {
        "base_fit": "base_fit.csv",
        "utility_fit": "selector_fit.csv",
        "validation": "validation.csv",
        "prospective_test": "prospective_test.csv",
    }:
        raise ValueError("Registered four-role manifest mapping changed")
    locked_scalars = {
        "initial_sector_id": 0,
        "sector_count": 8,
        "main_fixed_budget": 3,
        "maximum_adaptive_views": 4,
        "view_cost": 0.5,
        "defer_human_cost": 1.5,
        "random_trajectories_per_model_seed": 32,
    }
    for field, expected in locked_scalars.items():
        if config.get(field) != expected:
            raise ValueError(f"Registered config field {field!r} changed")
    if config.get("maximum_adaptive_views_semantics") != (
        "total revealed views including the fixed initial sector_0"
    ):
        raise ValueError("Adaptive view-cap semantics changed")
    if config.get("classifier_state_enumeration") != (
        "all unique subsets containing sector_0 for k=1..4"
    ) or config.get("utility_decision_state_enumeration") != (
        "all unique subsets containing sector_0 for k=1..3; k=4 is the acquisition cap"
    ):
        raise ValueError("Classifier/utility state enumeration changed")
    expected_cost = np.asarray([[0, 1, 4], [1, 0, 1], [8, 8, 0]], dtype=float)
    if not np.array_equal(np.asarray(config.get("cost_matrix"), dtype=float), expected_cost):
        raise ValueError("Registered operational cost matrix changed")
    head = config.get("probabilistic_head", {})
    if head.get("loss") != "unweighted_cross_entropy" or head.get(
        "early_stopping_metric"
    ) != "validation_nll":
        raise ValueError("Probabilistic-head training contract changed")
    utility = config.get("utility_regressor", {})
    expected_utility = {
        "target": "target_conditioned_soft_loss_before_minus_after_minus_view_cost",
        "loss": "smooth_l1",
        "state_weighting": "equal total weight per sample and revealed-view count",
        "hidden_candidate_inputs_forbidden": True,
        "early_stopping_metric": "validation sequence-macro top-1 utility regret",
    }
    for field, expected in expected_utility.items():
        if utility.get(field) != expected:
            raise ValueError(f"Utility-regressor contract field {field!r} changed")
    adaptive = config.get("adaptive_policy", {})
    if adaptive.get("q_stop") != "model-implied cost-sensitive Bayes risk":
        raise ValueError("Adaptive Q_stop contract changed")
    if adaptive.get("q_acquire") != "q_stop - predicted net utility":
        raise ValueError("Adaptive Q_acquire contract changed")
    expected_adaptive = {
        "q_defer": 1.5,
        "claim_status": "secondary risk-aware heuristic only; not risk-controlled",
        "formal_stop_calibration_performed": False,
        "formal_guarantee": False,
        "fail_closed_stop_threshold": None,
        "fail_closed_rule": (
            "STOP disabled because no independent policy-trajectory calibration role exists"
        ),
        "q_functional_warning": (
            "candidate utility is target-conditioned soft-loss reduction while q_stop "
            "is model-implied Bayes risk; adaptive Q values are heuristic"
        ),
        "risk_aware_heuristic_stop_threshold": 1.0,
    }
    for field, expected in expected_adaptive.items():
        if adaptive.get(field) != expected:
            raise ValueError(f"Adaptive risk-aware contract field {field!r} changed")
    training = config.get("training_hyperparameters")
    if training != {
        "classifier_epochs": 40,
        "utility_epochs": 60,
        "patience": 7,
        "batch_size": 512,
        "learning_rate": 0.001,
        "num_workers": 0,
        "device": "cuda",
        "cublas_workspace_config": ":4096:8",
    }:
        raise ValueError("Registered utility training hyperparameters changed")
    execution = config.get("execution_lock")
    if execution != {
        "evaluation_output_root": "outputs/cvian_sequence_active_v2/utility_experiment",
        "evaluation_registry_root": "outputs/cvian_sequence_active_v2/evaluation_registry",
        "base_runtime_lock": "configs/cvian_sequence_base_runtime_lock_v1.json",
    }:
        raise ValueError("Registered execution lock changed")
    if config.get("privileged_building_baseline") != {
        "required_for_primary_go": True,
        "metadata_schema": "cvian-sector-visibility-v3",
        "model_id": "nvidia/segformer-b0-finetuned-ade-512-512",
        "model_revision": "489d5cd81a0b59fab9b7ea758d3548ebe99677da",
        "building_class_id": 1,
        "horizontal_fov_deg": 90.0,
        "vertical_fov_deg": 90.0,
        "scope": "privileged offline baseline only",
    }:
        raise ValueError("Privileged building-baseline contract changed")
    registered_policies = [
        "clockwise",
        "random_mc32",
        "farthest",
        "utility_regression",
        "max_building_privileged",
        "max_confidence_privileged",
        "greedy_label_oracle",
    ]
    if config.get("fixed_budget_policies") != registered_policies:
        raise ValueError("Registered fixed-budget policy list changed")
    required_policies = {
        "clockwise",
        "random_mc32",
        "farthest",
        "utility_regression",
        "max_building_privileged",
        "max_confidence_privileged",
        "greedy_label_oracle",
    }
    if not required_policies.issubset(set(config.get("fixed_budget_policies", []))):
        raise ValueError("Required fixed-budget policies are missing")
    criterion = config.get("primary_go_criterion", {})
    expected_primary = {
        "estimand": "equal dependency-component macro operational cost at k=3",
        "contrast": "baseline_minus_utility_regression",
        "severe_miss_estimand": (
            "equal dependency-component macro severe-miss rate at k=3"
        ),
        "severe_miss_definition": (
            "indicator(target=2 and prediction!=2); fixed-k3 primary policies never defer"
        ),
        "severe_miss_noninferiority_margin": 0.0,
        "severe_miss_rule": (
            "the fixed-five-seed equal-component macro baseline-minus-utility "
            "contrast must be nonnegative"
        ),
    }
    for field, expected in expected_primary.items():
        if criterion.get(field) != expected:
            raise ValueError(f"Primary GO contract field {field!r} changed")
    if criterion.get("paired_bootstrap") != {
        "resampling_unit": "dependency_component",
        "pairing": (
            "resample each shared component with all five model-seed and "
            "paired-policy values intact"
        ),
        "resamples": 10000,
        "random_seed": 42,
        "confidence_level": 0.95,
        "interval": "percentile",
    }:
        raise ValueError("Paired dependency-component bootstrap contract changed")
    expected_requirements = [
        "utility_regression lower cost than farthest in all five model seeds",
        (
            "utility_regression lower cost than max_building_privileged in all "
            "five model seeds"
        ),
        (
            "paired dependency-component bootstrap 95% CI lower bound for "
            "baseline-minus-learned is above zero for both baselines"
        ),
        "utility_regression severe-miss rate is no worse than both baselines",
    ]
    if criterion.get("requirements") != expected_requirements:
        raise ValueError("Primary GO requirements changed")
    if config.get("adaptive_results_are_secondary") is not True:
        raise ValueError("Adaptive results must remain secondary")
    if config.get("external_confirmation_required_for_strong_claim") is not True:
        raise ValueError("External confirmation requirement changed")


def _validate_args_against_config(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> tuple[Path, Path]:
    training = config["training_hyperparameters"]
    actual = {
        "classifier_epochs": args.classifier_epochs,
        "utility_epochs": args.utility_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "device": args.device,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if actual != training:
        raise ValueError("CLI training hyperparameters do not match the frozen config")
    execution = config["execution_lock"]
    output_root = _resolve(args.output_root)
    locked_output = _resolve(execution["evaluation_output_root"])
    if output_root != locked_output:
        raise ValueError("Output root does not match the frozen one-shot execution root")
    registry_root = _resolve(execution["evaluation_registry_root"])
    return locked_output, registry_root


def _source_hashes() -> dict[str, str]:
    sources = {
        "runner": Path(__file__).resolve(),
        "active_view": REPO_ROOT / "crossview_conflict" / "decision" / "active_view.py",
        "active_view_utility": REPO_ROOT
        / "crossview_conflict"
        / "decision"
        / "active_view_utility.py",
        "artifact_provenance": REPO_ROOT
        / "crossview_conflict"
        / "decision"
        / "artifact_provenance.py",
    }
    return {name: _sha256(path) for name, path in sources.items()}


def _validate_protocol_without_test_load(protocol_dir: Path) -> dict[str, Any]:
    """Validate the registered protocol while never opening test rows.

    Hashing the signed commitment and reading aggregate protocol metadata is
    allowed in fit.  The prospective-test CSV itself is intentionally neither
    read nor hashed here.
    """
    summary_path = protocol_dir / "protocol_summary.json"
    commitment_path = protocol_dir / "test_commitment.json"
    summary = _load_json(summary_path)
    commitment = _load_json(commitment_path)
    if summary.get("schema_version") != "cvian-sequence-four-role-summary-v1":
        raise ValueError("Unsupported four-role protocol summary")
    if commitment.get("schema_version") != "cvian-sequence-test-commitment-v1":
        raise ValueError("Unsupported prospective-test commitment")
    status = summary.get("test_status", {})
    if status.get("status") != "selector_selection_holdout_with_historical_base_exposure":
        raise ValueError("Unexpected prospective-test status")
    if status.get("new_protocol_test_scored") is not False:
        raise ValueError("Protocol metadata says the prospective test was already scored")
    if status.get("old_base_or_checkpoint_reuse_permitted") is not False:
        raise ValueError("Historical checkpoint reuse must remain forbidden")
    if commitment.get("historical_checkpoint_reuse_permitted") is not False:
        raise ValueError("Test commitment permits forbidden historical checkpoints")
    for field in (
        "record_overlap_zero",
        "sequence_overlap_zero",
        "spatial_block_overlap_zero",
        "spatial_buffer_clear",
    ):
        if summary.get(field) is not True:
            raise ValueError(f"Protocol audit failed: {field}")
    hashes = summary.get("role_manifest_sha256", {})
    fit_manifest_hashes: dict[str, str] = {}
    for role in FIT_ROLES:
        path = protocol_dir / f"{role}.csv"
        actual = _sha256(path)
        if hashes.get(path.name) != actual:
            raise ValueError(f"Fit-role manifest hash mismatch: {path}")
        fit_manifest_hashes[role] = actual
    return {
        "protocol_dir": str(protocol_dir),
        "summary": summary,
        "summary_sha256": _sha256(summary_path),
        "commitment": commitment,
        "commitment_sha256": _sha256(commitment_path),
        "fit_manifest_sha256": fit_manifest_hashes,
        # This is signed aggregate metadata, not a read of prospective_test.csv.
        "registered_test_manifest_sha256": hashes.get("prospective_test.csv"),
    }


def _read_fit_manifests(protocol_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for role in FIT_ROLES:
        path = protocol_dir / f"{role}.csv"
        frame = pd.read_csv(
            path,
            dtype={"sample_id": str, "sequence_id": str, "spatial_block_id": str},
            usecols=[
                "sample_id",
                "sequence_id",
                "spatial_block_id",
                "label",
                "latitude",
                "longitude",
                "compass_angle_deg",
            ],
        )
        if frame["sample_id"].duplicated().any() or frame["label"].nunique() != 3:
            raise ValueError(f"Fit role is duplicated or not class-complete: {role}")
        frames[role] = frame.reset_index(drop=True)
    for index, left in enumerate(FIT_ROLES):
        for right in FIT_ROLES[index + 1 :]:
            for field in ("sample_id", "sequence_id", "spatial_block_id"):
                if set(frames[left][field]) & set(frames[right][field]):
                    raise ValueError(f"{field} overlap between {left} and {right}")
    return frames


def _validate_base_and_cache_metadata(
    *,
    seed: int,
    base_root: Path,
    cache_root: Path,
    protocol: Mapping[str, Any],
    roles: Sequence[str],
) -> dict[str, Any]:
    strict_provenance = validate_seed_base_and_embedding_provenance(
        repo_root=REPO_ROOT,
        protocol_dir=Path(protocol["protocol_dir"]),
        base_root=base_root,
        cache_root=cache_root,
        seed=seed,
        roles=tuple(roles),
    )
    base_dir = base_root / f"crossview_seed{seed}"
    checkpoint = base_dir / "triage_best.pt"
    completion_path = base_dir / "training_complete.json"
    completion = _load_json(completion_path)
    if completion.get("schema_version") != "cvian-sequence-role-isolated-base-v1":
        raise ValueError(f"Unsupported base completion for seed {seed}")
    if completion.get("seed") != seed or completion.get("train_role") != "base_fit":
        raise ValueError(f"Base role/seed mismatch for seed {seed}")
    if completion.get("validation_role") != "validation" or completion.get(
        "test_evaluated"
    ) is not False:
        raise ValueError(f"Base completion violates holdout contract for seed {seed}")
    base_fingerprint_payload = {
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
    if _canonical_hash(base_fingerprint_payload) != completion.get(
        "fingerprint_sha256"
    ):
        raise ValueError(f"Base completion fingerprint is internally invalid: seed {seed}")
    artifacts = (
        (checkpoint, "checkpoint_sha256"),
        (base_dir / "triage_history.json", "history_sha256"),
        (base_dir / "train.log", "train_log_sha256"),
    )
    for path, field in artifacts:
        if not path.is_file() or _sha256(path) != completion.get(field):
            raise ValueError(f"Base artifact hash mismatch: {path}")
    if completion.get("protocol_summary_sha256") != protocol["summary_sha256"]:
        raise ValueError("Base encoder used a different protocol summary")
    if completion.get("test_commitment_sha256") != protocol["commitment_sha256"]:
        raise ValueError("Base encoder used a different test commitment")

    seed_dir = cache_root / f"seed{seed}"
    metadata_path = seed_dir / "cache_metadata.json"
    metadata = _load_json(metadata_path)
    if metadata.get("schema_version") != "cvian-active-view-cache-v3":
        raise ValueError(f"Unsupported cache metadata for seed {seed}")
    if metadata.get("seed") != seed or metadata.get("checkpoint_sha256") != _sha256(
        checkpoint
    ):
        raise ValueError(f"Cache/base checkpoint mismatch for seed {seed}")
    if metadata.get("num_sectors") != 8 or metadata.get("num_classes") != 3:
        raise ValueError(f"Cache geometry/classes mismatch for seed {seed}")
    split_metadata = metadata.get("splits", {})
    source_hashes = metadata.get("source_manifest_sha256", {})
    declared_roles = metadata.get("split_roles", [])
    cache_hashes: dict[str, str] = {}
    for role in roles:
        if role not in declared_roles:
            raise ValueError(f"Cache metadata does not declare role {role}")
        path = seed_dir / f"{role}.npz"
        entry = split_metadata.get(role, {})
        actual = _sha256(path)
        if entry.get("sha256") != actual:
            raise ValueError(f"Cache artifact hash mismatch: {path}")
        registered = (
            protocol["fit_manifest_sha256"].get(role)
            if role in FIT_ROLES
            else protocol["registered_test_manifest_sha256"]
        )
        if source_hashes.get(role) != registered:
            raise ValueError(f"Cache source-manifest mismatch for role {role}")
        cache_hashes[role] = actual
    return {
        "base_completion_sha256": _sha256(completion_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "cache_metadata_sha256": _sha256(metadata_path),
        "cache_sha256": cache_hashes,
        "strict_provenance": strict_provenance,
    }


def _load_role_data(path: Path, manifest: pd.DataFrame) -> RoleData:
    cache = load_active_view_cache(path)
    with np.load(path, allow_pickle=False) as payload:
        if "compass_angle_deg" not in payload.files:
            raise ValueError(f"Cache has no compass_angle_deg: {path}")
        compass = payload["compass_angle_deg"].astype(np.float64)
    if len(compass) != cache.sample_count:
        raise ValueError(f"Compass/cache row mismatch: {path}")
    expected_ids = manifest["sample_id"].astype(str).to_numpy()
    checks = {
        "sample_id": np.array_equal(cache.sample_id, expected_ids),
        "sequence_id": np.array_equal(
            cache.sequence_id, manifest["sequence_id"].astype(str).to_numpy()
        ),
        "spatial_block_id": np.array_equal(
            cache.spatial_block_id,
            manifest["spatial_block_id"].astype(str).to_numpy(),
        ),
        "target": np.array_equal(cache.target, manifest["label"].astype(int).to_numpy()),
        "latitude": np.allclose(
            cache.latitude,
            manifest["latitude"].astype(float).to_numpy(),
            rtol=0.0,
            atol=1e-12,
        ),
        "longitude": np.allclose(
            cache.longitude,
            manifest["longitude"].astype(float).to_numpy(),
            rtol=0.0,
            atol=1e-12,
        ),
        "compass_angle_deg": np.allclose(
            compass,
            manifest["compass_angle_deg"].astype(float).to_numpy(),
            rtol=0.0,
            atol=1e-4,
            equal_nan=True,
        ),
    }
    failed = [field for field, passes in checks.items() if not passes]
    if failed:
        raise ValueError(f"Cache/manifest mismatch for {path}: {failed}")
    availability = np.isfinite(compass)
    return RoleData(
        cache=cache,
        compass_angle_deg=compass,
        compass_available=availability,
    )


def _load_fit_roles(
    cache_root: Path, seed: int, manifests: Mapping[str, pd.DataFrame]
) -> dict[str, RoleData]:
    """Load only fit roles.  Keeping this list literal makes leakage auditable."""
    return {
        role: _load_role_data(cache_root / f"seed{seed}" / f"{role}.npz", manifests[role])
        for role in FIT_ROLES
    }


def enumerate_subset_states(
    cache: ActiveViewCache,
    *,
    maximum_revealed: int,
    maximum_adaptive_views: int,
    initial_sector_id: int = 0,
) -> StateTable:
    """Enumerate every unique canonical subset containing the initial sector."""
    if not 1 <= maximum_revealed <= cache.sector_count:
        raise ValueError("maximum_revealed is outside the sector range")
    if not 1 <= maximum_adaptive_views <= cache.sector_count:
        raise ValueError("maximum_adaptive_views is outside the sector range")
    if not 0 <= initial_sector_id < cache.sector_count:
        raise ValueError("initial_sector_id is invalid")
    remaining = [value for value in range(cache.sector_count) if value != initial_sector_id]
    templates: list[np.ndarray] = []
    counts: list[int] = []
    for revealed_count in range(1, maximum_revealed + 1):
        for selected in combinations(remaining, revealed_count - 1):
            mask = np.zeros(cache.sector_count, dtype=bool)
            mask[initial_sector_id] = True
            mask[list(selected)] = True
            templates.append(mask)
            counts.append(revealed_count)
    template_mask = np.stack(templates)
    template_count = np.asarray(counts, dtype=np.int16)
    sample_indices = np.repeat(np.arange(cache.sample_count, dtype=np.int64), len(templates))
    masks = np.tile(template_mask, (cache.sample_count, 1))
    revealed = np.tile(template_count, cache.sample_count)
    remaining_acquisitions = np.maximum(maximum_adaptive_views - revealed, 0).astype(
        np.float32
    )
    return StateTable(
        sample_indices=sample_indices,
        revealed_mask=masks,
        revealed_count=revealed,
        remaining_acquisitions=remaining_acquisitions,
    )


def _state_features(
    cache: ActiveViewCache, states: StateTable, *, chunk_size: int = 4096
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(states.sample_indices), chunk_size):
        stop = min(len(states.sample_indices), start + chunk_size)
        chunks.append(
            build_state_features(
                cache,
                states.revealed_mask[start:stop],
                sample_indices=states.sample_indices[start:stop],
                remaining_budget=states.remaining_acquisitions[start:stop],
            )
        )
    return np.concatenate(chunks)


def _predict(
    model: ActiveViewMLP,
    standardizer: Standardizer,
    features: np.ndarray,
    *,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    values = standardizer.transform(features)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            outputs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outputs)


def _nll(logits: np.ndarray, targets: np.ndarray, temperature: float = 1.0) -> float:
    probabilities = softmax(logits, temperature=temperature)
    labels = np.asarray(targets, dtype=np.int64)
    return float(
        -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
    )


def _fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    result = minimize_scalar(
        lambda log_t: _nll(logits, targets, float(np.exp(log_t))),
        bounds=(-3.0, 3.0),
        method="bounded",
    )
    if not result.success or not np.isfinite(result.x):
        raise ValueError("Temperature optimization failed")
    return float(np.exp(result.x))


def _train_classifier(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    num_workers: int,
) -> ClassifierResult:
    _seed_everything(seed)
    standardizer = Standardizer.fit(train_features)
    train_values = standardizer.transform(train_features)
    model = ActiveViewMLP(train_values.shape[1], 3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Registered contract: no class or state weighting in the probabilistic head.
    loss_function = torch.nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_values),
            torch.from_numpy(np.asarray(train_targets, dtype=np.int64)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_nll = math.inf
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            total_rows += len(targets)
        validation_logits = _predict(
            model, standardizer, validation_features, device=device
        )
        validation_nll = _nll(validation_logits, validation_targets)
        history.append(
            {
                "epoch": float(epoch),
                "train_unweighted_cross_entropy": total_loss / max(total_rows, 1),
                "validation_nll": validation_nll,
            }
        )
        if validation_nll < best_nll - 1e-8:
            best_nll = validation_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return ClassifierResult(
        model=model,
        standardizer=standardizer,
        best_epoch=best_epoch,
        best_validation_nll=float(best_nll),
        history=history,
    )


def _balanced_candidate_weights(
    sample_index: np.ndarray, revealed_count: np.ndarray
) -> np.ndarray:
    """Give every (sample, k) cell equal total weight."""
    samples = np.asarray(sample_index, dtype=np.int64)
    counts = np.asarray(revealed_count, dtype=np.int64)
    if samples.shape != counts.shape or samples.ndim != 1 or len(samples) == 0:
        raise ValueError("Candidate weight keys must be non-empty one-dimensional arrays")
    keys = np.stack([samples, counts], axis=1)
    _, inverse, cell_counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    weights = 1.0 / cell_counts[inverse]
    # Unit mean keeps optimizer scale independent of enumeration cardinality.
    return (weights / weights.mean()).astype(np.float32)


def _build_utility_table(
    role: RoleData,
    classifier: ClassifierResult,
    *,
    temperature: float,
    view_cost: float,
    maximum_adaptive_views: int,
    device: str,
) -> UtilityTable:
    cache = role.cache
    states = enumerate_subset_states(
        cache,
        maximum_revealed=3,
        maximum_adaptive_views=maximum_adaptive_views,
    )
    state_features = _state_features(cache, states)
    current_logits = _predict(
        classifier.model, classifier.standardizer, state_features, device=device
    )
    current_probabilities = softmax(current_logits, temperature)

    parent_rows: list[int] = []
    candidates: list[int] = []
    candidate_masks: list[np.ndarray] = []
    for state_id, mask in enumerate(states.revealed_mask):
        for candidate in np.flatnonzero(~mask):
            updated = mask.copy()
            updated[candidate] = True
            parent_rows.append(state_id)
            candidates.append(int(candidate))
            candidate_masks.append(updated)
    parent = np.asarray(parent_rows, dtype=np.int64)
    candidate_sector = np.asarray(candidates, dtype=np.int64)
    after_states = StateTable(
        sample_indices=states.sample_indices[parent],
        revealed_mask=np.stack(candidate_masks),
        revealed_count=states.revealed_count[parent] + 1,
        remaining_acquisitions=np.maximum(
            states.remaining_acquisitions[parent] - 1, 0
        ).astype(np.float32),
    )
    after_features = _state_features(cache, after_states)
    after_logits = _predict(
        classifier.model, classifier.standardizer, after_features, device=device
    )
    after_probabilities = softmax(after_logits, temperature)
    sample_index = states.sample_indices[parent]
    labels = cache.target[sample_index]
    before_loss = target_conditioned_soft_loss(current_probabilities[parent], labels)
    after_loss = target_conditioned_soft_loss(after_probabilities, labels)
    targets = (before_loss - after_loss - view_cost).astype(np.float32)
    features = build_candidate_utility_features(
        state_features[parent],
        current_probabilities[parent],
        states.revealed_mask[parent],
        candidate_sector,
        cache.relative_azimuth_deg,
        compass_angle_deg=role.compass_angle_deg[sample_index],
        compass_available=role.compass_available[sample_index],
    )
    revealed_count = states.revealed_count[parent]
    return UtilityTable(
        features=features,
        targets=targets,
        weights=_balanced_candidate_weights(sample_index, revealed_count),
        state_id=parent,
        sample_index=sample_index,
        sequence_id=cache.sequence_id[sample_index],
        revealed_count=revealed_count,
        candidate_sector=candidate_sector,
    )


def _sequence_macro_top1_regret(
    predicted_utility: np.ndarray, table: UtilityTable
) -> tuple[float, pd.DataFrame]:
    predicted = np.asarray(predicted_utility, dtype=np.float64).reshape(-1)
    if len(predicted) != len(table.targets) or not np.isfinite(predicted).all():
        raise ValueError("Predicted utility is invalid")
    rows: list[dict[str, Any]] = []
    order = np.lexsort((table.candidate_sector, table.state_id))
    ordered_state = table.state_id[order]
    unique_state, starts, counts = np.unique(
        ordered_state, return_index=True, return_counts=True
    )
    for state_id, start, count in zip(unique_state, starts, counts):
        member = order[start : start + count]
        # np.lexsort: primary key is -prediction, canonical sector breaks ties.
        chosen_local = np.lexsort(
            (table.candidate_sector[member], -predicted[member])
        )[0]
        chosen = member[chosen_local]
        best = float(np.max(table.targets[member]))
        regret = max(0.0, best - float(table.targets[chosen]))
        rows.append(
            {
                "state_id": int(state_id),
                "sample_index": int(table.sample_index[chosen]),
                "sequence_id": str(table.sequence_id[chosen]),
                "revealed_count": int(table.revealed_count[chosen]),
                "selected_sector": int(table.candidate_sector[chosen]),
                "selected_target_net_utility": float(table.targets[chosen]),
                "best_target_net_utility": best,
                "top1_utility_regret": regret,
            }
        )
    diagnostics = pd.DataFrame(rows)
    sequence_means = diagnostics.groupby("sequence_id", sort=True)[
        "top1_utility_regret"
    ].mean()
    return float(sequence_means.mean()), diagnostics


def _train_utility_regressor(
    train: UtilityTable,
    validation: UtilityTable,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    num_workers: int,
) -> RegressorResult:
    _seed_everything(seed)
    standardizer = Standardizer.fit(train.features)
    train_values = standardizer.transform(train.features)
    model = ActiveViewMLP(train_values.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_values),
            torch.from_numpy(train.targets[:, None]),
            torch.from_numpy(train.weights),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_regret = math.inf
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        for features, targets, weights in loader:
            features = features.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            row_loss = torch.nn.functional.smooth_l1_loss(
                model(features), targets, reduction="none"
            ).squeeze(1)
            loss = torch.sum(row_loss * weights) / torch.sum(weights)
            loss.backward()
            optimizer.step()
            weighted_loss_sum += float(torch.sum(row_loss.detach() * weights).item())
            weight_sum += float(torch.sum(weights).item())
        validation_prediction = _predict(
            model, standardizer, validation.features, device=device
        ).reshape(-1)
        validation_regret, _ = _sequence_macro_top1_regret(
            validation_prediction, validation
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_weighted_smooth_l1": weighted_loss_sum / max(weight_sum, 1e-12),
                "validation_sequence_macro_top1_utility_regret": validation_regret,
            }
        )
        if validation_regret < best_regret - 1e-8:
            best_regret = validation_regret
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return RegressorResult(
        model=model,
        standardizer=standardizer,
        best_epoch=best_epoch,
        best_validation_sequence_macro_regret=float(best_regret),
        history=history,
    )


def _fit_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "classifier_epochs": args.classifier_epochs,
        "utility_epochs": args.utility_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "device": args.device,
        "classifier_loss": "unweighted_cross_entropy",
        "classifier_early_stopping": "validation_nll",
        "utility_loss": "sample_k_balanced_smooth_l1",
        "utility_early_stopping": "validation_sequence_macro_top1_utility_regret",
        "formal_stop_calibration_performed": False,
        "adaptive_claim_status": "secondary_risk_aware_not_risk_controlled",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _fit_fingerprint(
    args: argparse.Namespace,
    *,
    seed: int,
    config_path: Path,
    protocol: Mapping[str, Any],
    input_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    config = _load_json(config_path)
    runtime_lock_provenance = validate_supplemental_runtime_lock(
        repo_root=REPO_ROOT,
        lock_path=_resolve(config["execution_lock"]["base_runtime_lock"]),
    )
    payload: dict[str, Any] = {
        "schema_version": FIT_SCHEMA,
        "seed": seed,
        "config_sha256": _sha256(config_path),
        "protocol_summary_sha256": protocol["summary_sha256"],
        "test_commitment_sha256": protocol["commitment_sha256"],
        "fit_manifest_sha256": protocol["fit_manifest_sha256"],
        "input_provenance": input_provenance,
        "source_sha256": _source_hashes(),
        "git_head": _git_head(),
        "runtime_versions": _runtime_versions(),
        "base_runtime_lock_provenance": runtime_lock_provenance,
        "hyperparameters": _fit_hyperparameters(args),
        "prospective_test_loaded": False,
    }
    payload["fit_fingerprint_sha256"] = _canonical_hash(payload)
    return payload


def _validate_fit_completion(
    seed_dir: Path, completion: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if completion.get("schema_version") != FIT_SCHEMA:
        raise ValueError(f"Unsupported fit completion: {seed_dir}")
    if completion.get("fit_fingerprint_sha256") != expected.get(
        "fit_fingerprint_sha256"
    ):
        raise ValueError(f"Fit fingerprint mismatch: {seed_dir}")
    for filename, field in (
        ("fit_artifact.pt", "fit_artifact_sha256"),
        ("training_history.json", "training_history_sha256"),
        ("validation_top1_regret.csv", "validation_regret_sha256"),
        ("stop_policy_status.csv", "stop_policy_status_sha256"),
    ):
        path = seed_dir / filename
        if not path.is_file() or _sha256(path) != completion.get(field):
            raise ValueError(f"Fit artifact hash mismatch: {path}")


def _fit_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    config: Mapping[str, Any],
    config_path: Path,
    protocol: Mapping[str, Any],
    manifests: Mapping[str, pd.DataFrame],
    cache_root: Path,
    base_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    input_provenance = _validate_base_and_cache_metadata(
        seed=seed,
        base_root=base_root,
        cache_root=cache_root,
        protocol=protocol,
        roles=FIT_ROLES,
    )
    fingerprint = _fit_fingerprint(
        args,
        seed=seed,
        config_path=config_path,
        protocol=protocol,
        input_provenance=input_provenance,
    )
    seed_dir = output_root / f"seed{seed}"
    completion_path = seed_dir / "fit_complete.json"
    if completion_path.is_file():
        completion = _load_json(completion_path)
        _validate_fit_completion(seed_dir, completion, fingerprint)
        return completion
    protected = (
        "fit_artifact.pt",
        "training_history.json",
        "validation_top1_regret.csv",
        "stop_policy_status.csv",
        "metrics.csv",
        "per_sample_decisions.csv",
        "evaluation_complete.json",
    )
    if any((seed_dir / name).exists() for name in protected):
        raise ValueError(f"Unattested artifacts exist in {seed_dir}; use a new output root")

    roles = _load_fit_roles(cache_root, seed, manifests)
    base_states = enumerate_subset_states(
        roles["base_fit"].cache,
        maximum_revealed=4,
        maximum_adaptive_views=int(config["maximum_adaptive_views"]),
    )
    validation_states = enumerate_subset_states(
        roles["validation"].cache,
        maximum_revealed=4,
        maximum_adaptive_views=int(config["maximum_adaptive_views"]),
    )
    base_features = _state_features(roles["base_fit"].cache, base_states)
    validation_features = _state_features(roles["validation"].cache, validation_states)
    classifier = _train_classifier(
        base_features,
        roles["base_fit"].cache.target[base_states.sample_indices],
        validation_features,
        roles["validation"].cache.target[validation_states.sample_indices],
        epochs=args.classifier_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=seed,
        device=args.device,
        num_workers=args.num_workers,
    )
    validation_logits = _predict(
        classifier.model,
        classifier.standardizer,
        validation_features,
        device=args.device,
    )
    temperature = _fit_temperature(
        validation_logits,
        roles["validation"].cache.target[validation_states.sample_indices],
    )
    utility_fit = _build_utility_table(
        roles["selector_fit"],
        classifier,
        temperature=temperature,
        view_cost=float(config["view_cost"]),
        maximum_adaptive_views=int(config["maximum_adaptive_views"]),
        device=args.device,
    )
    utility_validation = _build_utility_table(
        roles["validation"],
        classifier,
        temperature=temperature,
        view_cost=float(config["view_cost"]),
        maximum_adaptive_views=int(config["maximum_adaptive_views"]),
        device=args.device,
    )
    regressor = _train_utility_regressor(
        utility_fit,
        utility_validation,
        epochs=args.utility_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=seed + 97,
        device=args.device,
        num_workers=args.num_workers,
    )
    validation_prediction = _predict(
        regressor.model,
        regressor.standardizer,
        utility_validation.features,
        device=args.device,
    ).reshape(-1)
    validation_regret, validation_diagnostics = _sequence_macro_top1_regret(
        validation_prediction, utility_validation
    )
    # A formal STOP certificate is deliberately not fit on the reused validation
    # role.  Valid certification would require an independent role and calibration
    # on states reached by the frozen policy.  The fail-closed secondary policy
    # therefore receives threshold=None; a separate threshold=1 heuristic is
    # reported only as risk-aware/descriptive.
    stop_policy_status = pd.DataFrame(
        [
            {
                "formal_stop_calibration_performed": False,
                "formal_guarantee": False,
                "fail_closed_stop_threshold": math.nan,
                "risk_aware_heuristic_stop_threshold": float(
                    config["adaptive_policy"]["risk_aware_heuristic_stop_threshold"]
                ),
                "validation_reused_for_model_selection": True,
                "policy_trajectory_calibration_role_available": False,
                "status": "risk-aware-only",
            }
        ]
    )

    seed_dir.mkdir(parents=True, exist_ok=True)
    validation_diagnostics.to_csv(seed_dir / "validation_top1_regret.csv", index=False)
    stop_policy_status.to_csv(seed_dir / "stop_policy_status.csv", index=False)
    artifact_metadata = {
        **fingerprint,
        "temperature": temperature,
        "classifier_best_epoch": classifier.best_epoch,
        "classifier_validation_nll": classifier.best_validation_nll,
        "utility_best_epoch": regressor.best_epoch,
        "validation_sequence_macro_top1_utility_regret": validation_regret,
        "fail_closed_stop_threshold": None,
        "formal_stop_calibration_performed": False,
        "formal_stop_guarantee": False,
        "adaptive_claim_status": "secondary_risk_aware_not_risk_controlled",
        "available_sector_count_feature": "revealed_mask",
        "remaining_acquisitions_feature": "explicit_remaining_budget",
    }
    torch.save(
        {
            "schema_version": POLICY_SCHEMA,
            "classifier_input_dim": int(classifier.standardizer.mean.shape[0]),
            "classifier_state_dict": classifier.model.state_dict(),
            "classifier_mean": classifier.standardizer.mean,
            "classifier_scale": classifier.standardizer.scale,
            "utility_input_dim": int(regressor.standardizer.mean.shape[0]),
            "utility_state_dict": regressor.model.state_dict(),
            "utility_mean": regressor.standardizer.mean,
            "utility_scale": regressor.standardizer.scale,
            "temperature": temperature,
            "fail_closed_stop_threshold": None,
            "metadata": artifact_metadata,
        },
        seed_dir / "fit_artifact.pt",
    )
    (seed_dir / "training_history.json").write_text(
        json.dumps(
            {
                "schema_version": FIT_SCHEMA,
                "classifier": classifier.history,
                "utility_regressor": regressor.history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    completion = {
        **artifact_metadata,
        "schema_version": FIT_SCHEMA,
        "fit_artifact_sha256": _sha256(seed_dir / "fit_artifact.pt"),
        "training_history_sha256": _sha256(seed_dir / "training_history.json"),
        "validation_regret_sha256": _sha256(seed_dir / "validation_top1_regret.csv"),
        "stop_policy_status_sha256": _sha256(seed_dir / "stop_policy_status.csv"),
    }
    completion_path.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    return completion


def fit_all_seeds(args: argparse.Namespace) -> None:
    config_path = _resolve(args.config)
    protocol_dir = _resolve(args.protocol_dir)
    cache_root = _resolve(args.cache_root)
    base_root = _resolve(args.base_root)
    output_root = _resolve(args.output_root)
    config = _load_json(config_path)
    _validate_config(config)
    output_root, _ = _validate_args_against_config(args, config)
    protocol = _validate_protocol_without_test_load(protocol_dir)
    manifests = _read_fit_manifests(protocol_dir)
    for seed in EXPECTED_SEEDS:
        print(f"fit sequence utility seed={seed}", flush=True)
        _fit_seed(
            args,
            seed=seed,
            config=config,
            config_path=config_path,
            protocol=protocol,
            manifests=manifests,
            cache_root=cache_root,
            base_root=base_root,
            output_root=output_root,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": FIT_SCHEMA,
        "phase": "fit",
        "prospective_test_loaded": False,
        "seed_fit_completion_sha256": {
            str(seed): _sha256(output_root / f"seed{seed}" / "fit_complete.json")
            for seed in EXPECTED_SEEDS
        },
    }
    (output_root / "fit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def _load_models(path: Path, *, device: str) -> tuple[ClassifierResult, RegressorResult, float, float | None]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != POLICY_SCHEMA:
        raise ValueError(f"Unsupported policy artifact: {path}")
    classifier = ActiveViewMLP(int(payload["classifier_input_dim"]), 3).to(device)
    classifier.load_state_dict(payload["classifier_state_dict"])
    classifier.eval()
    regressor = ActiveViewMLP(int(payload["utility_input_dim"]), 1).to(device)
    regressor.load_state_dict(payload["utility_state_dict"])
    regressor.eval()
    classifier_result = ClassifierResult(
        model=classifier,
        standardizer=Standardizer(
            mean=np.asarray(payload["classifier_mean"], dtype=np.float32),
            scale=np.asarray(payload["classifier_scale"], dtype=np.float32),
        ),
        best_epoch=int(payload["metadata"]["classifier_best_epoch"]),
        best_validation_nll=float(payload["metadata"]["classifier_validation_nll"]),
        history=[],
    )
    regressor_result = RegressorResult(
        model=regressor,
        standardizer=Standardizer(
            mean=np.asarray(payload["utility_mean"], dtype=np.float32),
            scale=np.asarray(payload["utility_scale"], dtype=np.float32),
        ),
        best_epoch=int(payload["metadata"]["utility_best_epoch"]),
        best_validation_sequence_macro_regret=float(
            payload["metadata"]["validation_sequence_macro_top1_utility_regret"]
        ),
        history=[],
    )
    threshold = payload.get("fail_closed_stop_threshold")
    if threshold is not None:
        raise ValueError("Fail-closed policy artifact unexpectedly enables STOP")
    if payload["metadata"].get("formal_stop_guarantee") is not False:
        raise ValueError("Policy artifact contains an unsupported STOP guarantee")
    return classifier_result, regressor_result, float(payload["temperature"]), None


def _single_state_features(
    cache: ActiveViewCache,
    sample_index: int,
    mask: np.ndarray,
    *,
    maximum_adaptive_views: int,
) -> np.ndarray:
    revealed_count = int(mask.sum())
    return build_state_features(
        cache,
        np.asarray(mask, dtype=bool)[None, :],
        sample_indices=np.asarray([sample_index]),
        remaining_budget=np.asarray(
            [max(maximum_adaptive_views - revealed_count, 0)], dtype=np.float32
        ),
    )


def _candidate_predictions(
    role: RoleData,
    sample_index: int,
    mask: np.ndarray,
    classifier: ClassifierResult,
    regressor: RegressorResult,
    *,
    temperature: float,
    maximum_adaptive_views: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = role.cache
    candidates = np.flatnonzero(~mask).astype(np.int64)
    state = _single_state_features(
        cache,
        sample_index,
        mask,
        maximum_adaptive_views=maximum_adaptive_views,
    )
    logits = _predict(
        classifier.model, classifier.standardizer, state, device=device
    )
    probabilities = softmax(logits, temperature)
    if len(candidates) == 0:
        return candidates, np.empty(0, dtype=float), probabilities[0]
    candidate_features = build_candidate_utility_features(
        np.repeat(state, len(candidates), axis=0),
        np.repeat(probabilities, len(candidates), axis=0),
        np.repeat(mask[None, :], len(candidates), axis=0),
        candidates,
        cache.relative_azimuth_deg,
        compass_angle_deg=np.full(len(candidates), role.compass_angle_deg[sample_index]),
        compass_available=np.full(
            len(candidates), role.compass_available[sample_index], dtype=bool
        ),
    )
    predicted = _predict(
        regressor.model, regressor.standardizer, candidate_features, device=device
    ).reshape(-1)
    return candidates, predicted, probabilities[0]


def _after_probabilities(
    role: RoleData,
    sample_index: int,
    mask: np.ndarray,
    candidates: np.ndarray,
    classifier: ClassifierResult,
    *,
    temperature: float,
    maximum_adaptive_views: int,
    device: str,
) -> np.ndarray:
    features: list[np.ndarray] = []
    for candidate in candidates:
        updated = mask.copy()
        updated[int(candidate)] = True
        features.append(
            _single_state_features(
                role.cache,
                sample_index,
                updated,
                maximum_adaptive_views=maximum_adaptive_views,
            )[0]
        )
    logits = _predict(
        classifier.model,
        classifier.standardizer,
        np.stack(features),
        device=device,
    )
    return softmax(logits, temperature)


def _argmax_canonical(values: np.ndarray, sectors: np.ndarray) -> int:
    order = np.lexsort((sectors, -np.asarray(values, dtype=np.float64)))
    return int(sectors[order[0]])


def _load_visibility_if_present(path: Path, role: RoleData) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        required = {"sample_id", "sector_id", "building_ratio"}
        if not required.issubset(payload.files):
            raise ValueError(f"Visibility cache is incomplete: {path}")
        sample_ids = payload["sample_id"].astype(str)
        sector_ids = payload["sector_id"].astype(np.int64)
        ratios = payload["building_ratio"].astype(np.float32)
    if not np.array_equal(sector_ids, role.cache.sector_id):
        raise ValueError("Visibility sector geometry mismatch")
    index = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    if (
        len(index) != len(sample_ids)
        or len(sample_ids) != role.cache.sample_count
        or set(role.cache.sample_id) != set(index)
    ):
        raise ValueError("Visibility sample IDs do not exactly match the evaluation role")
    aligned = ratios[[index[value] for value in role.cache.sample_id]]
    if aligned.shape != (role.cache.sample_count, role.cache.sector_count):
        raise ValueError("Visibility cache shape mismatch")
    if not np.isfinite(aligned).all() or (aligned < 0.0).any() or (aligned > 1.0).any():
        raise ValueError("Visibility ratios must be finite values in [0, 1]")
    return aligned


def _dependency_components(cache: ActiveViewCache) -> np.ndarray:
    """Connected components of the sequence--spatial-block bipartite graph."""
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

    for sequence, block in zip(cache.sequence_id, cache.spatial_block_id):
        union(f"sequence:{sequence}", f"block:{block}")
    roots = sorted({find(f"sequence:{value}") for value in cache.sequence_id})
    names = {root: f"component_{index:03d}" for index, root in enumerate(roots)}
    return np.asarray(
        [names[find(f"sequence:{value}")] for value in cache.sequence_id], dtype="U32"
    )


def _fixed_action(
    policy: str,
    role: RoleData,
    sample_index: int,
    mask: np.ndarray,
    classifier: ClassifierResult,
    regressor: RegressorResult,
    *,
    temperature: float,
    view_cost: float,
    maximum_adaptive_views: int,
    device: str,
    rng: np.random.Generator,
    visibility: np.ndarray | None,
) -> int:
    candidates = np.flatnonzero(~mask).astype(np.int64)
    if len(candidates) == 0:
        raise ValueError(f"Policy {policy} has no available sector")
    if policy == "clockwise":
        return int(candidates.min())
    if policy == "random_mc32":
        return int(rng.choice(candidates))
    if policy == "farthest":
        return farthest_available_sector(
            np.flatnonzero(mask), candidates, sector_count=role.cache.sector_count
        )
    if policy == "max_building_privileged":
        if visibility is None:
            raise ValueError("Building policy requested without a visibility cache")
        return _argmax_canonical(visibility[sample_index, candidates], candidates)
    if policy == "utility_regression":
        utility_candidates, predicted_utility, _ = _candidate_predictions(
            role,
            sample_index,
            mask,
            classifier,
            regressor,
            temperature=temperature,
            maximum_adaptive_views=maximum_adaptive_views,
            device=device,
        )
        if not np.array_equal(utility_candidates, candidates):
            raise ValueError("Utility candidate ordering changed unexpectedly")
        return _argmax_canonical(predicted_utility, candidates)
    current_features = _single_state_features(
        role.cache,
        sample_index,
        mask,
        maximum_adaptive_views=maximum_adaptive_views,
    )
    current_probability = softmax(
        _predict(
            classifier.model,
            classifier.standardizer,
            current_features,
            device=device,
        ),
        temperature,
    )[0]
    after_probabilities = _after_probabilities(
        role,
        sample_index,
        mask,
        candidates,
        classifier,
        temperature=temperature,
        maximum_adaptive_views=maximum_adaptive_views,
        device=device,
    )
    if policy == "max_confidence_privileged":
        return _argmax_canonical(after_probabilities.max(axis=1), candidates)
    if policy == "greedy_label_oracle":
        target = np.full(len(candidates), role.cache.target[sample_index], dtype=np.int64)
        before = target_conditioned_soft_loss(
            np.repeat(current_probability[None, :], len(candidates), axis=0), target
        )
        after = target_conditioned_soft_loss(after_probabilities, target)
        return _argmax_canonical(before - after - view_cost, candidates)
    raise ValueError(f"Unknown fixed-budget policy: {policy}")


def _decision_row(
    *,
    seed: int,
    policy: str,
    scope: str,
    role: RoleData,
    sample_index: int,
    mask: np.ndarray,
    selected: Sequence[int],
    classifier: ClassifierResult,
    temperature: float,
    view_cost: float,
    maximum_adaptive_views: int,
    device: str,
    component: str,
    trajectory: int,
    terminal_action: str,
    defer_cost: float,
) -> dict[str, Any]:
    features = _single_state_features(
        role.cache,
        sample_index,
        mask,
        maximum_adaptive_views=maximum_adaptive_views,
    )
    probabilities = softmax(
        _predict(classifier.model, classifier.standardizer, features, device=device),
        temperature,
    )[0]
    bayes_action, bayes_risk = bayes_operational_action_and_risk(probabilities[None, :])
    target = int(role.cache.target[sample_index])
    spent_view_cost = (len(selected) - 1) * view_cost
    if terminal_action == "defer":
        prediction = -1
        class_cost = defer_cost
        operational_cost = defer_cost + spent_view_cost
        severe_miss = 0.0
    else:
        prediction = int(bayes_action[0])
        class_cost = float(
            realized_operational_cost(np.asarray([prediction]), np.asarray([target]))[0]
        )
        operational_cost = class_cost + spent_view_cost
        severe_miss = float(target == 2 and prediction != 2)
    return {
        "seed": seed,
        "policy": policy,
        "scope": scope,
        "sample_id": str(role.cache.sample_id[sample_index]),
        "sequence_id": str(role.cache.sequence_id[sample_index]),
        "spatial_block_id": str(role.cache.spatial_block_id[sample_index]),
        "dependency_component": component,
        "trajectory": trajectory,
        "target": target,
        "prediction": prediction,
        "terminal_action": terminal_action,
        "revealed_sector_ids": ",".join(map(str, selected)),
        "view_count": len(selected),
        "additional_view_count": len(selected) - 1,
        "probability_0": float(probabilities[0]),
        "probability_1": float(probabilities[1]),
        "probability_2": float(probabilities[2]),
        "model_bayes_risk": float(bayes_risk[0]),
        "classification_or_defer_cost": class_cost,
        "view_cost": spent_view_cost,
        "operational_cost": operational_cost,
        "severe_miss": severe_miss,
    }


def _evaluate_fixed_policies(
    *,
    seed: int,
    role: RoleData,
    classifier: ClassifierResult,
    regressor: RegressorResult,
    temperature: float,
    config: Mapping[str, Any],
    device: str,
    visibility: np.ndarray | None,
) -> list[dict[str, Any]]:
    policies = [
        "utility_regression",
        "farthest",
        "clockwise",
        "random_mc32",
        "max_building_privileged",
        "max_confidence_privileged",
        "greedy_label_oracle",
    ]
    if visibility is None:
        raise ValueError("The registered building baseline is required for primary GO")
    components = _dependency_components(role.cache)
    rows: list[dict[str, Any]] = []
    for policy in policies:
        trajectories = (
            int(config["random_trajectories_per_model_seed"])
            if policy == "random_mc32"
            else 1
        )
        for trajectory in range(trajectories):
            for sample_index in range(role.cache.sample_count):
                mask = np.zeros(role.cache.sector_count, dtype=bool)
                initial = int(config["initial_sector_id"])
                mask[initial] = True
                selected = [initial]
                rng = np.random.default_rng(
                    seed * 1_000_003 + sample_index * 10_007 + trajectory * 101
                )
                while len(selected) < int(config["main_fixed_budget"]):
                    action = _fixed_action(
                        policy,
                        role,
                        sample_index,
                        mask,
                        classifier,
                        regressor,
                        temperature=temperature,
                        view_cost=float(config["view_cost"]),
                        maximum_adaptive_views=int(config["maximum_adaptive_views"]),
                        device=device,
                        rng=rng,
                        visibility=visibility,
                    )
                    if mask[action]:
                        raise ValueError(f"Policy {policy} attempted a duplicate reveal")
                    mask[action] = True
                    selected.append(action)
                rows.append(
                    _decision_row(
                        seed=seed,
                        policy=policy,
                        scope="fixed_k3_primary",
                        role=role,
                        sample_index=sample_index,
                        mask=mask,
                        selected=selected,
                        classifier=classifier,
                        temperature=temperature,
                        view_cost=float(config["view_cost"]),
                        maximum_adaptive_views=int(config["maximum_adaptive_views"]),
                        device=device,
                        component=str(components[sample_index]),
                        trajectory=trajectory,
                        terminal_action="stop",
                        defer_cost=float(config["defer_human_cost"]),
                    )
                )
    return rows


def _evaluate_adaptive_policies(
    *,
    seed: int,
    role: RoleData,
    classifier: ClassifierResult,
    regressor: RegressorResult,
    temperature: float,
    fail_closed_threshold: float | None,
    config: Mapping[str, Any],
    device: str,
) -> list[dict[str, Any]]:
    adaptive = config["adaptive_policy"]
    if fail_closed_threshold is not None:
        raise ValueError("Fail-closed adaptive policy must not receive a STOP threshold")
    policies = {
        "utility_adaptive_fail_closed": fail_closed_threshold,
        "utility_adaptive_risk_aware_heuristic": float(
            adaptive["risk_aware_heuristic_stop_threshold"]
        ),
    }
    components = _dependency_components(role.cache)
    rows: list[dict[str, Any]] = []
    for policy, threshold in policies.items():
        for sample_index in range(role.cache.sample_count):
            mask = np.zeros(role.cache.sector_count, dtype=bool)
            initial = int(config["initial_sector_id"])
            mask[initial] = True
            selected = [initial]
            terminal = "defer"
            while True:
                candidates, predicted, probabilities = _candidate_predictions(
                    role,
                    sample_index,
                    mask,
                    classifier,
                    regressor,
                    temperature=temperature,
                    maximum_adaptive_views=int(config["maximum_adaptive_views"]),
                    device=device,
                )
                utility_by_sector = np.zeros((1, role.cache.sector_count), dtype=float)
                utility_by_sector[0, candidates] = predicted
                # The environment still has unrevealed sectors at k=4, but the
                # registered adaptive acquisition budget is exhausted.  Mark
                # every candidate unavailable for Q minimization so the final
                # choice is genuinely STOP versus DEFER, never an ACQUIRE that
                # is post-hoc relabelled as DEFER.
                action_mask = (
                    np.ones_like(mask)
                    if len(selected) >= int(config["maximum_adaptive_views"])
                    else mask
                )
                action = select_adaptive_actions(
                    utility_by_sector,
                    action_mask[None, :],
                    probabilities[None, :],
                    defer_cost=float(config["defer_human_cost"]),
                    stop_risk_threshold=threshold,
                    view_cost=float(config["view_cost"]),
                    cost_matrix=np.asarray(config["cost_matrix"], dtype=np.float64),
                )
                selected_action = str(action.action[0])
                if selected_action == "acquire":
                    sector = int(action.sector_id[0])
                    if sector < 0 or mask[sector]:
                        raise ValueError("Adaptive policy produced an invalid reveal")
                    mask[sector] = True
                    selected.append(sector)
                    continue
                # At the registered cap ACQUIRE is unavailable by construction.
                terminal = "stop" if selected_action == "stop" else "defer"
                break
            rows.append(
                _decision_row(
                    seed=seed,
                    policy=policy,
                    scope="adaptive_secondary",
                    role=role,
                    sample_index=sample_index,
                    mask=mask,
                    selected=selected,
                    classifier=classifier,
                    temperature=temperature,
                    view_cost=float(config["view_cost"]),
                    maximum_adaptive_views=int(config["maximum_adaptive_views"]),
                    device=device,
                    component=str(components[sample_index]),
                    trajectory=0,
                    terminal_action=terminal,
                    defer_cost=float(config["defer_human_cost"]),
                )
            )
    return rows


def _sample_average_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse Monte Carlo trajectories before computing sample estimands.

    The random policy has 32 rows per sample while deterministic policies have
    one.  Averaging here prevents a trajectory count from changing the weight
    of a sample or dependency component.  Identifier columns are part of the
    group key so inconsistent target/component assignments cannot be hidden by
    the reduction.
    """
    required = {
        "seed",
        "policy",
        "scope",
        "sample_id",
        "sequence_id",
        "spatial_block_id",
        "dependency_component",
        "trajectory",
        "target",
        "operational_cost",
        "severe_miss",
        "view_count",
        "terminal_action",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Decision frame lacks metric columns: {sorted(missing)}")
    numeric = frame[["operational_cost", "severe_miss", "view_count"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all():
        raise ValueError("Decision metric columns must be finite")
    if (frame["operational_cost"].to_numpy(dtype=float) < 0.0).any():
        raise ValueError("Operational cost cannot be negative")
    severe = frame["severe_miss"].to_numpy(dtype=float)
    if ((severe < 0.0) | (severe > 1.0)).any():
        raise ValueError("Severe-miss indicators must lie in [0, 1]")
    key = [
        "seed",
        "policy",
        "scope",
        "sample_id",
        "sequence_id",
        "spatial_block_id",
        "dependency_component",
        "target",
    ]
    if frame.duplicated(["seed", "policy", "scope", "sample_id", "trajectory"]).any():
        raise ValueError("A policy has duplicate sample/trajectory decision rows")
    work = frame.copy()
    work["defer_probability"] = (work["terminal_action"] == "defer").astype(float)
    collapsed = (
        work.groupby(key, sort=True, as_index=False)
        .agg(
            operational_cost=("operational_cost", "mean"),
            severe_miss=("severe_miss", "mean"),
            view_count=("view_count", "mean"),
            defer_probability=("defer_probability", "mean"),
            trajectory_count=("trajectory", "nunique"),
        )
        .reset_index(drop=True)
    )
    if len(collapsed) != frame["sample_id"].nunique():
        raise ValueError("Sample identifiers are not unique within a policy/seed")
    return collapsed


def _trajectory_averaged_classification_scores(
    frame: pd.DataFrame,
) -> tuple[float, float]:
    """Expected accuracy/F1 with equal sample weight before trajectory pooling."""
    trajectory_count = frame.groupby("sample_id", sort=False)["trajectory"].transform(
        "nunique"
    )
    if (trajectory_count <= 0).any():
        raise ValueError("A sample has no policy trajectory")
    classified = frame[frame["prediction"] >= 0].copy()
    if not len(classified):
        return math.nan, math.nan
    classified["sample_weight"] = 1.0 / trajectory_count.loc[classified.index]
    weights = classified["sample_weight"].to_numpy(dtype=float)
    target = classified["target"].to_numpy(dtype=np.int64)
    prediction = classified["prediction"].to_numpy(dtype=np.int64)
    accuracy = float(np.average(target == prediction, weights=weights))
    f1_values: list[float] = []
    for label in (0, 1, 2):
        true_positive = float(weights[(target == label) & (prediction == label)].sum())
        false_positive = float(weights[(target != label) & (prediction == label)].sum())
        false_negative = float(weights[(target == label) & (prediction != label)].sum())
        denominator = 2.0 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0.0 else 2.0 * true_positive / denominator)
    return accuracy, float(np.mean(f1_values))


def _metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (seed, policy, scope), frame in decisions.groupby(
        ["seed", "policy", "scope"], sort=True
    ):
        sample_frame = _sample_average_trajectories(frame)
        component_cost = sample_frame.groupby("dependency_component", sort=True)[
            "operational_cost"
        ].mean()
        component_severe = sample_frame.groupby("dependency_component", sort=True)[
            "severe_miss"
        ].mean()
        accuracy, macro_f1 = _trajectory_averaged_classification_scores(frame)
        rows.append(
            {
                "seed": int(seed),
                "policy": policy,
                "scope": scope,
                "decision_rows": int(len(frame)),
                "estimand_sample_rows": int(len(sample_frame)),
                "unique_samples": int(frame["sample_id"].nunique()),
                "trajectories_per_sample_min": int(
                    sample_frame["trajectory_count"].min()
                ),
                "trajectories_per_sample_max": int(
                    sample_frame["trajectory_count"].max()
                ),
                "dependency_components": int(
                    sample_frame["dependency_component"].nunique()
                ),
                "mean_operational_cost": float(
                    sample_frame["operational_cost"].mean()
                ),
                "dependency_component_macro_operational_cost": float(
                    component_cost.mean()
                ),
                "severe_miss_rate": float(sample_frame["severe_miss"].mean()),
                "dependency_component_macro_severe_miss_rate": float(
                    component_severe.mean()
                ),
                "mean_view_count": float(sample_frame["view_count"].mean()),
                "defer_rate": float(sample_frame["defer_probability"].mean()),
                "accuracy_non_deferred": accuracy,
                "macro_f1_non_deferred": macro_f1,
            }
        )
    return pd.DataFrame(rows)


def _paired_component_bootstrap(
    contrasts: pd.DataFrame,
    *,
    value_column: str,
    resamples: int,
    random_seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    """Percentile CI after resampling paired dependency components.

    A draw keeps every model-seed value attached to a sampled component.  This
    preserves both policy pairing and the repeated five-seed evaluation of the
    same prospective sample dependencies.
    """
    if resamples < 1 or not 0.0 < confidence_level < 1.0:
        raise ValueError("Invalid paired bootstrap settings")
    pivot = contrasts.pivot(
        index="dependency_component", columns="seed", values=value_column
    ).sort_index(axis=0).sort_index(axis=1)
    if tuple(int(value) for value in pivot.columns) != EXPECTED_SEEDS:
        raise ValueError("Bootstrap contrasts do not contain the five locked seeds")
    if pivot.empty or pivot.isna().any().any():
        raise ValueError("Bootstrap contrasts are incomplete across components/seeds")
    values = pivot.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    sampled_components = rng.integers(
        0, len(values), size=(resamples, len(values)), endpoint=False
    )
    replicates = values[sampled_components].mean(axis=(1, 2))
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(replicates, [tail, 1.0 - tail])
    return float(lower), float(upper)


def _aggregate_primary_analysis(
    decisions: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Apply the frozen primary GO criterion to staged five-seed decisions."""
    _validate_config(config)
    primary = decisions[decisions["scope"] == "fixed_k3_primary"].copy()
    if set(int(value) for value in primary["seed"].unique()) != set(EXPECTED_SEEDS):
        raise ValueError("Primary decisions do not contain exactly the five locked seeds")
    expected_policies = {PRIMARY_UTILITY_POLICY, *PRIMARY_BASELINES}
    if not expected_policies.issubset(set(primary["policy"].unique())):
        raise ValueError("Primary decisions are missing a registered GO policy")
    bootstrap = config["primary_go_criterion"]["paired_bootstrap"]
    csv_rows: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    common_component_set: set[str] | None = None
    sample_counts: dict[str, int] = {}

    for baseline in PRIMARY_BASELINES:
        component_rows: list[pd.DataFrame] = []
        per_seed: dict[str, Any] = {}
        for seed in EXPECTED_SEEDS:
            utility = _sample_average_trajectories(
                primary[
                    (primary["seed"] == seed)
                    & (primary["policy"] == PRIMARY_UTILITY_POLICY)
                ]
            )
            baseline_frame = _sample_average_trajectories(
                primary[
                    (primary["seed"] == seed) & (primary["policy"] == baseline)
                ]
            )
            if not (utility["trajectory_count"] == 1).all() or not (
                baseline_frame["trajectory_count"] == 1
            ).all():
                raise ValueError(
                    "Primary utility and baseline policies must have one locked "
                    "trajectory per sample"
                )
            join_keys = [
                "seed",
                "scope",
                "sample_id",
                "sequence_id",
                "spatial_block_id",
                "dependency_component",
                "target",
            ]
            utility = utility.drop(columns=["policy"]).rename(
                columns={
                    "operational_cost": "utility_operational_cost",
                    "severe_miss": "utility_severe_miss",
                }
            )
            baseline_frame = baseline_frame.drop(columns=["policy"]).rename(
                columns={
                    "operational_cost": "baseline_operational_cost",
                    "severe_miss": "baseline_severe_miss",
                }
            )
            paired = utility.merge(
                baseline_frame,
                on=join_keys,
                how="inner",
                validate="one_to_one",
                suffixes=("_utility", "_baseline"),
            )
            if len(paired) != len(utility) or len(paired) != len(baseline_frame):
                raise ValueError(f"Unpaired {baseline} decisions for seed {seed}")
            if len(paired) == 0:
                raise ValueError(f"No paired {baseline} decisions for seed {seed}")
            sample_counts[str(seed)] = int(len(paired))
            paired["cost_difference_baseline_minus_utility"] = (
                paired["baseline_operational_cost"]
                - paired["utility_operational_cost"]
            )
            paired["severe_difference_baseline_minus_utility"] = (
                paired["baseline_severe_miss"] - paired["utility_severe_miss"]
            )
            component = (
                paired.groupby("dependency_component", sort=True, as_index=False)
                .agg(
                    samples=("sample_id", "size"),
                    utility_component_mean_cost=("utility_operational_cost", "mean"),
                    baseline_component_mean_cost=("baseline_operational_cost", "mean"),
                    cost_difference_baseline_minus_utility=(
                        "cost_difference_baseline_minus_utility",
                        "mean",
                    ),
                    utility_component_mean_severe_miss=(
                        "utility_severe_miss",
                        "mean",
                    ),
                    baseline_component_mean_severe_miss=(
                        "baseline_severe_miss",
                        "mean",
                    ),
                    severe_difference_baseline_minus_utility=(
                        "severe_difference_baseline_minus_utility",
                        "mean",
                    ),
                )
                .assign(seed=seed)
            )
            component_set = set(component["dependency_component"].astype(str))
            if common_component_set is None:
                common_component_set = component_set
            elif component_set != common_component_set:
                raise ValueError("Dependency-component membership changed across seeds")
            component_rows.append(component)
            seed_cost_difference = float(
                component["cost_difference_baseline_minus_utility"].mean()
            )
            seed_severe_difference = float(
                component["severe_difference_baseline_minus_utility"].mean()
            )
            seed_result = {
                "samples": int(component["samples"].sum()),
                "dependency_components": int(len(component)),
                "utility_component_macro_operational_cost": float(
                    component["utility_component_mean_cost"].mean()
                ),
                "baseline_component_macro_operational_cost": float(
                    component["baseline_component_mean_cost"].mean()
                ),
                "cost_difference_baseline_minus_utility": seed_cost_difference,
                "utility_component_macro_severe_miss_rate": float(
                    component["utility_component_mean_severe_miss"].mean()
                ),
                "baseline_component_macro_severe_miss_rate": float(
                    component["baseline_component_mean_severe_miss"].mean()
                ),
                "severe_difference_baseline_minus_utility": seed_severe_difference,
                "utility_lower_cost": bool(seed_cost_difference > 0.0),
            }
            per_seed[str(seed)] = seed_result
            csv_rows.append(
                {
                    "row_type": "per_seed",
                    "baseline": baseline,
                    "seed": str(seed),
                    **seed_result,
                    "bootstrap_ci_lower": math.nan,
                    "bootstrap_ci_upper": math.nan,
                    "all_five_seeds_lower_cost": math.nan,
                    "severe_miss_no_worse": math.nan,
                    "comparison_go": math.nan,
                }
            )

        all_components = pd.concat(component_rows, ignore_index=True)
        ci_lower, ci_upper = _paired_component_bootstrap(
            all_components,
            value_column="cost_difference_baseline_minus_utility",
            resamples=int(bootstrap["resamples"]),
            random_seed=int(bootstrap["random_seed"]),
            confidence_level=float(bootstrap["confidence_level"]),
        )
        overall_cost_difference = float(
            all_components["cost_difference_baseline_minus_utility"].mean()
        )
        overall_severe_difference = float(
            all_components["severe_difference_baseline_minus_utility"].mean()
        )
        all_five_lower = bool(
            all(item["utility_lower_cost"] for item in per_seed.values())
        )
        severe_margin = float(
            config["primary_go_criterion"]["severe_miss_noninferiority_margin"]
        )
        severe_no_worse = bool(overall_severe_difference >= -severe_margin)
        ci_strictly_positive = bool(ci_lower > 0.0)
        comparison_go = bool(
            all_five_lower and ci_strictly_positive and severe_no_worse
        )
        overall = {
            "dependency_components": int(all_components["dependency_component"].nunique()),
            "utility_component_macro_operational_cost": float(
                all_components["utility_component_mean_cost"].mean()
            ),
            "baseline_component_macro_operational_cost": float(
                all_components["baseline_component_mean_cost"].mean()
            ),
            "cost_difference_baseline_minus_utility": overall_cost_difference,
            "bootstrap_95_percentile_ci_lower": ci_lower,
            "bootstrap_95_percentile_ci_upper": ci_upper,
            "utility_component_macro_severe_miss_rate": float(
                all_components["utility_component_mean_severe_miss"].mean()
            ),
            "baseline_component_macro_severe_miss_rate": float(
                all_components["baseline_component_mean_severe_miss"].mean()
            ),
            "severe_difference_baseline_minus_utility": overall_severe_difference,
            "all_five_seeds_lower_cost": all_five_lower,
            "bootstrap_ci_lower_strictly_positive": ci_strictly_positive,
            "severe_miss_no_worse": severe_no_worse,
            "comparison_go": comparison_go,
        }
        comparisons[baseline] = {"per_seed": per_seed, "overall": overall}
        csv_rows.append(
            {
                "row_type": "all_seed_component_macro",
                "baseline": baseline,
                "seed": "ALL",
                "samples": int(sum(sample_counts.values())),
                "dependency_components": overall["dependency_components"],
                "utility_component_macro_operational_cost": overall[
                    "utility_component_macro_operational_cost"
                ],
                "baseline_component_macro_operational_cost": overall[
                    "baseline_component_macro_operational_cost"
                ],
                "cost_difference_baseline_minus_utility": overall_cost_difference,
                "utility_component_macro_severe_miss_rate": overall[
                    "utility_component_macro_severe_miss_rate"
                ],
                "baseline_component_macro_severe_miss_rate": overall[
                    "baseline_component_macro_severe_miss_rate"
                ],
                "severe_difference_baseline_minus_utility": overall_severe_difference,
                "utility_lower_cost": bool(overall_cost_difference > 0.0),
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper,
                "all_five_seeds_lower_cost": all_five_lower,
                "severe_miss_no_worse": severe_no_worse,
                "comparison_go": comparison_go,
            }
        )

    aggregate_go = bool(
        all(comparisons[name]["overall"]["comparison_go"] for name in PRIMARY_BASELINES)
    )
    analysis = {
        "schema_version": AGGREGATE_SCHEMA,
        "claim_scope": config["claim_scope"],
        "test_status": config["test_status"],
        "primary_scope": "fixed_k3_primary",
        "utility_policy": PRIMARY_UTILITY_POLICY,
        "baselines": list(PRIMARY_BASELINES),
        "model_seeds": list(EXPECTED_SEEDS),
        "sample_count_per_seed": sample_counts,
        "dependency_components": len(common_component_set or set()),
        "estimand": config["primary_go_criterion"]["estimand"],
        "contrast": config["primary_go_criterion"]["contrast"],
        "severe_miss_estimand": config["primary_go_criterion"][
            "severe_miss_estimand"
        ],
        "severe_miss_definition": config["primary_go_criterion"][
            "severe_miss_definition"
        ],
        "severe_miss_noninferiority_margin": config["primary_go_criterion"][
            "severe_miss_noninferiority_margin"
        ],
        "severe_miss_rule": config["primary_go_criterion"]["severe_miss_rule"],
        "paired_bootstrap": copy.deepcopy(bootstrap),
        "comparisons": comparisons,
        "requirements": copy.deepcopy(
            config["primary_go_criterion"]["requirements"]
        ),
        "go": aggregate_go,
        "decision": "GO" if aggregate_go else "NO-GO",
        "adaptive_results_in_primary_decision": False,
        "external_confirmation_required_for_strong_claim": bool(
            config["external_confirmation_required_for_strong_claim"]
        ),
    }
    return analysis, pd.DataFrame(csv_rows)


def _render_primary_report(
    analysis: Mapping[str, Any], comparisons: pd.DataFrame
) -> str:
    """Render the immutable, machine-derived primary result report."""
    lines = [
        "# CVIAN sequence utility experiment v1",
        "",
        f"**Primary decision: {analysis['decision']}.**",
        "",
        f"Claim scope: {analysis['claim_scope']}.",
        "",
        (
            "This report is generated only after all five one-time prospective-test "
            "seed artifacts have been staged. The primary estimand is equal "
            "dependency-component macro operational cost at fixed k=3; adaptive "
            "results are secondary and cannot change the decision."
        ),
        "",
        "## Frozen primary comparisons",
        "",
        (
            "Positive cost and severe-miss differences favor utility_regression "
            "(baseline minus utility)."
        ),
        "",
        "| Baseline | Mean cost difference | Paired component bootstrap 95% CI | Five seeds lower | Severe miss no worse | Result |",
        "|---|---:|---:|:---:|:---:|:---:|",
    ]
    for baseline in PRIMARY_BASELINES:
        overall = analysis["comparisons"][baseline]["overall"]
        lines.append(
            "| {baseline} | {difference:.6f} | [{lower:.6f}, {upper:.6f}] | "
            "{five} | {severe} | {result} |".format(
                baseline=baseline,
                difference=overall["cost_difference_baseline_minus_utility"],
                lower=overall["bootstrap_95_percentile_ci_lower"],
                upper=overall["bootstrap_95_percentile_ci_upper"],
                five="yes" if overall["all_five_seeds_lower_cost"] else "no",
                severe="yes" if overall["severe_miss_no_worse"] else "no",
                result="GO" if overall["comparison_go"] else "NO-GO",
            )
        )
    lines.extend(
        [
            "",
            "## Per-seed component-macro contrasts",
            "",
            "| Baseline | Seed | Utility cost | Baseline cost | Baseline - utility | Utility lower |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    per_seed_rows = comparisons[comparisons["row_type"] == "per_seed"]
    for row in per_seed_rows.to_dict(orient="records"):
        lines.append(
            "| {baseline} | {seed} | {utility:.6f} | {baseline_cost:.6f} | "
            "{difference:.6f} | {lower} |".format(
                baseline=row["baseline"],
                seed=row["seed"],
                utility=row["utility_component_macro_operational_cost"],
                baseline_cost=row["baseline_component_macro_operational_cost"],
                difference=row["cost_difference_baseline_minus_utility"],
                lower="yes" if row["utility_lower_cost"] else "no",
            )
        )
    bootstrap = analysis["paired_bootstrap"]
    lines.extend(
        [
            "",
            "## Decision rule and interpretation",
            "",
            (
                f"The locked percentile bootstrap uses {bootstrap['resamples']:,} "
                f"dependency-component resamples with random seed "
                f"{bootstrap['random_seed']}. Each sampled component retains all "
                "five model-seed and paired-policy values."
            ),
            "",
            (
                "GO requires both baselines to pass all three requirements: a "
                "strictly positive component-macro contrast in every seed, a "
                "strictly positive 95% CI lower bound, and a nonnegative "
                "baseline-minus-utility severe-miss contrast."
            ),
            "",
            (
                "For random_mc32 descriptive metrics, trajectories are averaged "
                "within sample before dependency-component aggregation."
            ),
            "",
            (
                "Adaptive STOP/ACQUIRE/DEFER results are secondary heuristics. "
                "No formal STOP calibration was performed because the current "
                "protocol has no independent policy-trajectory calibration role; "
                "the fail-closed variant disables STOP and the threshold-1 variant "
                "is risk-aware, not risk-controlled."
            ),
            "",
            (
                "This is a within-CVIAN development confirmation with historical "
                "base exposure, not external confirmation. It evaluates the "
                "label-cost Phase-1 baseline and does not complete the separate "
                "attestation-coverage claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _evaluation_fingerprint(
    args: argparse.Namespace,
    *,
    config_path: Path,
    protocol_dir: Path,
    protocol: Mapping[str, Any],
    cache_root: Path,
    base_root: Path,
    output_root: Path,
    visibility_path: Path,
    visibility_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    config = _load_json(config_path)
    execution = config["execution_lock"]
    registry_root = _resolve(execution["evaluation_registry_root"])
    test_manifest = protocol_dir / "prospective_test.csv"
    test_hash = _sha256(test_manifest)
    if test_hash != protocol["registered_test_manifest_sha256"]:
        raise ValueError("Prospective-test manifest no longer matches its commitment")
    seed_inputs: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        input_provenance = _validate_base_and_cache_metadata(
            seed=seed,
            base_root=base_root,
            cache_root=cache_root,
            protocol=protocol,
            roles=FIT_ROLES,
        )
        fit_expected = _fit_fingerprint(
            args,
            seed=seed,
            config_path=config_path,
            protocol=protocol,
            input_provenance=input_provenance,
        )
        seed_dir = output_root / f"seed{seed}"
        fit_completion_path = seed_dir / "fit_complete.json"
        fit_completion = _load_json(fit_completion_path)
        _validate_fit_completion(seed_dir, fit_completion, fit_expected)
        test_metadata = _validate_base_and_cache_metadata(
            seed=seed,
            base_root=base_root,
            cache_root=cache_root,
            protocol=protocol,
            roles=(TEST_ROLE,),
        )
        seed_inputs[seed] = {
            "fit_completion_sha256": _sha256(fit_completion_path),
            "fit_artifact_sha256": fit_completion["fit_artifact_sha256"],
            "prospective_test_cache_sha256": test_metadata["cache_sha256"][TEST_ROLE],
            "cache_metadata_sha256": test_metadata["cache_metadata_sha256"],
        }
    payload: dict[str, Any] = {
        "schema_version": COMMITMENT_SCHEMA,
        "config_sha256": _sha256(config_path),
        "protocol_summary_sha256": protocol["summary_sha256"],
        "test_commitment_sha256": protocol["commitment_sha256"],
        "prospective_test_manifest_sha256": test_hash,
        "source_sha256": _source_hashes(),
        "git_head": _git_head(),
        "runtime_versions": _runtime_versions(),
        "fit_hyperparameters": _fit_hyperparameters(args),
        "seed_inputs": {str(seed): values for seed, values in seed_inputs.items()},
        "visibility_sha256": _sha256(visibility_path) if visibility_path.is_file() else None,
        "visibility_provenance": dict(visibility_provenance),
        "locked_output_root": str(output_root),
        "global_registry_root": str(registry_root),
        "fixed_budget": 3,
        "random_mc_trajectories": 32,
        "primary_aggregate": {
            "schema_version": AGGREGATE_SCHEMA,
            "utility_policy": PRIMARY_UTILITY_POLICY,
            "baselines": list(PRIMARY_BASELINES),
            "paired_bootstrap": copy.deepcopy(
                config["primary_go_criterion"]["paired_bootstrap"]
            ),
            "report_path": str(PRIMARY_REPORT_PATH.relative_to(REPO_ROOT)).replace(
                "\\", "/"
            ),
        },
        "adaptive_results_secondary": True,
        "scoring_status_before_run": "not_loaded",
    }
    payload["evaluation_fingerprint_sha256"] = _canonical_hash(payload)
    return payload, seed_inputs


def _validate_evaluation_completion(
    seed_dir: Path,
    completion: Mapping[str, Any],
    *,
    evaluation_fingerprint: str,
    commitment_sha256: str,
    started_sha256: str,
    seed_input: Mapping[str, Any],
) -> None:
    if completion.get("schema_version") != EVALUATION_SCHEMA:
        raise ValueError(f"Unsupported evaluation completion: {seed_dir}")
    if completion.get("evaluation_fingerprint_sha256") != evaluation_fingerprint:
        raise ValueError(f"Evaluation fingerprint mismatch: {seed_dir}")
    if completion.get("evaluation_commitment_sha256") != commitment_sha256:
        raise ValueError(f"Evaluation commitment mismatch: {seed_dir}")
    if completion.get("evaluation_started_sha256") != started_sha256:
        raise ValueError(f"Evaluation-start attestation mismatch: {seed_dir}")
    if completion.get("fit_completion_sha256") != seed_input["fit_completion_sha256"]:
        raise ValueError(f"Fit completion changed after evaluation: {seed_dir}")
    for filename, field in (
        ("metrics.csv", "metrics_sha256"),
        ("per_sample_decisions.csv", "decisions_sha256"),
    ):
        path = seed_dir / filename
        if not path.is_file() or _sha256(path) != completion.get(field):
            raise ValueError(f"Evaluation artifact hash mismatch: {path}")


def _validate_aggregate_completion(
    output_root: Path,
    completion: Mapping[str, Any],
    *,
    evaluation_fingerprint: str,
    commitment_sha256: str,
    started_sha256: str,
    seed_completion_sha256: Mapping[str, str],
) -> None:
    """Verify aggregate/report hashes without reading prospective-test rows."""
    if completion.get("schema_version") != AGGREGATE_COMPLETION_SCHEMA:
        raise ValueError("Unsupported primary aggregate completion")
    if completion.get("evaluation_fingerprint_sha256") != evaluation_fingerprint:
        raise ValueError("Primary aggregate evaluation fingerprint mismatch")
    if completion.get("evaluation_commitment_sha256") != commitment_sha256:
        raise ValueError("Primary aggregate commitment mismatch")
    if completion.get("evaluation_started_sha256") != started_sha256:
        raise ValueError("Primary aggregate start attestation mismatch")
    if completion.get("seed_evaluation_completion_sha256") != dict(
        seed_completion_sha256
    ):
        raise ValueError("Primary aggregate seed-completion inputs changed")
    artifacts = (
        (output_root / AGGREGATE_JSON_NAME, "aggregate_json_sha256"),
        (output_root / AGGREGATE_CSV_NAME, "aggregate_csv_sha256"),
        (PRIMARY_REPORT_PATH, "report_sha256"),
    )
    for path, field in artifacts:
        if not path.is_file() or _sha256(path) != completion.get(field):
            raise ValueError(f"Primary aggregate artifact hash mismatch: {path}")
    analysis = _load_json(output_root / AGGREGATE_JSON_NAME)
    if analysis.get("schema_version") != AGGREGATE_SCHEMA:
        raise ValueError("Primary aggregate analysis schema changed")
    if analysis.get("decision") != completion.get("primary_decision"):
        raise ValueError("Primary aggregate decision does not match its completion")


def _validate_registry_completion(
    path: Path,
    *,
    evaluation_fingerprint: str,
    commitment_sha256: str,
    started_sha256: str,
    output_root: Path,
    aggregate_completion_sha256: str,
) -> None:
    completion = _load_json(path)
    expected = {
        "schema_version": REGISTRY_COMPLETION_SCHEMA,
        "evaluation_fingerprint_sha256": evaluation_fingerprint,
        "evaluation_commitment_sha256": commitment_sha256,
        "evaluation_started_sha256": started_sha256,
        "locked_output_root": str(output_root),
        "aggregate_completion_sha256": aggregate_completion_sha256,
        "status": "prospective_test_evaluation_complete",
    }
    for field, value in expected.items():
        if completion.get(field) != value:
            raise ValueError(f"Global evaluation registry mismatch: {field}")


def evaluate_once(args: argparse.Namespace) -> None:
    config_path = _resolve(args.config)
    protocol_dir = _resolve(args.protocol_dir)
    cache_root = _resolve(args.cache_root)
    base_root = _resolve(args.base_root)
    visibility_dir = _resolve(args.visibility_dir)
    visibility_path = visibility_dir / "prospective_test.npz"
    config = _load_json(config_path)
    _validate_config(config)
    output_root, registry_root = _validate_args_against_config(args, config)
    protocol = _validate_protocol_without_test_load(protocol_dir)
    if not visibility_path.is_file():
        raise ValueError(
            "The registered max_building_privileged baseline requires the locked "
            "prospective-test visibility cache before scoring can start"
        )
    visibility_provenance = validate_visibility_provenance(
        repo_root=REPO_ROOT,
        protocol_dir=protocol_dir,
        visibility_dir=visibility_dir,
    )

    # Everything below through commitment creation is hash/metadata validation;
    # the prospective-test NPZ has not been opened yet.
    fingerprint, seed_inputs = _evaluation_fingerprint(
        args,
        config_path=config_path,
        protocol_dir=protocol_dir,
        protocol=protocol,
        cache_root=cache_root,
        base_root=base_root,
        output_root=output_root,
        visibility_path=visibility_path,
        visibility_provenance=visibility_provenance,
    )
    completion_paths = [
        output_root / f"seed{seed}" / "evaluation_complete.json"
        for seed in EXPECTED_SEEDS
    ]
    completion_exists = [path.is_file() for path in completion_paths]
    # This registry is keyed by the signed test commitment, not by a caller-
    # selected output directory.  Changing --output-root therefore cannot make
    # the one-shot holdout look unused.
    registry_dir = registry_root / protocol["commitment_sha256"]
    commitment_path = registry_dir / "evaluation_commitment.json"
    started_path = registry_dir / "evaluation_started.json"
    registry_complete_path = registry_dir / "evaluation_complete.json"
    if all(completion_exists):
        if not commitment_path.is_file() or _sha256(commitment_path) != _canonical_commitment_file_hash(
            fingerprint
        ):
            raise ValueError("Evaluation commitment is missing or changed")
        if not started_path.is_file():
            raise ValueError("Evaluation-start attestation is missing")
        commitment_sha = _sha256(commitment_path)
        started_sha = _sha256(started_path)
        for seed, path in zip(EXPECTED_SEEDS, completion_paths):
            _validate_evaluation_completion(
                path.parent,
                _load_json(path),
                evaluation_fingerprint=fingerprint["evaluation_fingerprint_sha256"],
                commitment_sha256=commitment_sha,
                started_sha256=started_sha,
                seed_input=seed_inputs[seed],
            )
        seed_completion_sha = {
            str(seed): _sha256(path)
            for seed, path in zip(EXPECTED_SEEDS, completion_paths)
        }
        aggregate_completion_path = output_root / AGGREGATE_COMPLETION_NAME
        if not aggregate_completion_path.is_file():
            raise ValueError("Primary aggregate completion is missing")
        _validate_aggregate_completion(
            output_root,
            _load_json(aggregate_completion_path),
            evaluation_fingerprint=fingerprint["evaluation_fingerprint_sha256"],
            commitment_sha256=commitment_sha,
            started_sha256=started_sha,
            seed_completion_sha256=seed_completion_sha,
        )
        _validate_registry_completion(
            registry_complete_path,
            evaluation_fingerprint=fingerprint["evaluation_fingerprint_sha256"],
            commitment_sha256=commitment_sha,
            started_sha256=started_sha,
            output_root=output_root,
            aggregate_completion_sha256=_sha256(aggregate_completion_path),
        )
        print("Verified existing prospective-test evaluation; no holdout was loaded.")
        return
    if any(completion_exists):
        raise ValueError(
            "Partial prospective-test completion exists; refusing a second scoring pass"
        )
    if started_path.is_file():
        raise ValueError(
            "A prospective-test scoring pass was already started without a complete "
            "attestation; fail-closed reuse policy forbids loading it again"
        )
    if registry_complete_path.exists():
        raise ValueError("Global registry says the prospective test was already scored")
    for seed in EXPECTED_SEEDS:
        seed_dir = output_root / f"seed{seed}"
        if any((seed_dir / name).exists() for name in ("metrics.csv", "per_sample_decisions.csv")):
            raise ValueError(f"Unattested evaluation artifacts exist in {seed_dir}")
    aggregate_paths = (
        output_root / AGGREGATE_JSON_NAME,
        output_root / AGGREGATE_CSV_NAME,
        output_root / AGGREGATE_COMPLETION_NAME,
        PRIMARY_REPORT_PATH,
    )
    existing_aggregate = [path for path in aggregate_paths if path.exists()]
    if existing_aggregate:
        raise ValueError(
            "Unattested primary aggregate artifacts already exist: "
            + ", ".join(str(path) for path in existing_aggregate)
        )

    output_root.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)
    commitment_text = json.dumps(fingerprint, indent=2) + "\n"
    expected_commitment_hash = hashlib.sha256(commitment_text.encode("utf-8")).hexdigest()
    if commitment_path.is_file():
        if _sha256(commitment_path) != expected_commitment_hash:
            raise ValueError("Existing evaluation commitment does not match locked inputs")
    else:
        commitment_path.write_text(commitment_text, encoding="utf-8")
    commitment_sha = _sha256(commitment_path)

    # This sentinel is written immediately before the first data load.  If the
    # process fails afterwards, a rerun refuses to inspect the holdout again.
    started = {
        "schema_version": COMMITMENT_SCHEMA,
        "evaluation_fingerprint_sha256": fingerprint[
            "evaluation_fingerprint_sha256"
        ],
        "evaluation_commitment_sha256": commitment_sha,
        "prospective_test_manifest_sha256": fingerprint[
            "prospective_test_manifest_sha256"
        ],
        "locked_output_root": str(output_root),
        "status": "prospective_test_scoring_started",
        "rerun_if_incomplete_permitted": False,
    }
    # Exclusive creation is atomic on the local filesystem and prevents two
    # concurrent evaluators from both passing the unused-test check.
    with started_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(started, indent=2) + "\n")
    started_sha = _sha256(started_path)

    # First and only prospective-test data load begins here, after commitment.
    manifest = pd.read_csv(
        protocol_dir / "prospective_test.csv",
        dtype={"sample_id": str, "sequence_id": str, "spatial_block_id": str},
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
    staged: dict[int, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    for seed in EXPECTED_SEEDS:
        print(f"score prospective test once: seed={seed}", flush=True)
        role = _load_role_data(cache_root / f"seed{seed}" / "prospective_test.npz", manifest)
        classifier, regressor, temperature, fail_closed_threshold = _load_models(
            output_root / f"seed{seed}" / "fit_artifact.pt", device=args.device
        )
        visibility = _load_visibility_if_present(visibility_path, role)
        decision_rows = _evaluate_fixed_policies(
            seed=seed,
            role=role,
            classifier=classifier,
            regressor=regressor,
            temperature=temperature,
            config=config,
            device=args.device,
            visibility=visibility,
        )
        decision_rows.extend(
            _evaluate_adaptive_policies(
                seed=seed,
                role=role,
                classifier=classifier,
                regressor=regressor,
                temperature=temperature,
                fail_closed_threshold=fail_closed_threshold,
                config=config,
                device=args.device,
            )
        )
        decisions = pd.DataFrame(decision_rows)
        metrics = _metrics(decisions)
        metadata = {
            "seed": seed,
            "prospective_test_rows": role.cache.sample_count,
            "prospective_test_sequences": int(len(np.unique(role.cache.sequence_id))),
            "fail_closed_stop_threshold": fail_closed_threshold,
            "formal_stop_guarantee": False,
            "adaptive_claim_status": "secondary_risk_aware_not_risk_controlled",
            "building_policy_available": visibility is not None,
            "test_status": "selector_selection_holdout_with_historical_base_exposure",
            "claim_scope": config["claim_scope"],
        }
        staged[seed] = (metrics, decisions, metadata)
        del classifier, regressor, role
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Apply the pre-registered decision rule while all five seed outputs are
    # staged in memory.  A missing baseline or pairing defect fails before any
    # evaluation artifact or result report is written.
    aggregate_analysis, aggregate_comparisons = _aggregate_primary_analysis(
        pd.concat(
            [staged[seed][1] for seed in EXPECTED_SEEDS], ignore_index=True
        ),
        config,
    )
    report_text = _render_primary_report(
        aggregate_analysis, aggregate_comparisons
    )
    aggregate_json_text = (
        json.dumps(aggregate_analysis, indent=2, sort_keys=True) + "\n"
    )

    # Write only after every seed has scored successfully, avoiding partial completion.
    for seed in EXPECTED_SEEDS:
        metrics, decisions, metadata = staged[seed]
        seed_dir = output_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(seed_dir / "metrics.csv", index=False)
        decisions.to_csv(seed_dir / "per_sample_decisions.csv", index=False)
        completion = {
            "schema_version": EVALUATION_SCHEMA,
            **metadata,
            "evaluation_fingerprint_sha256": fingerprint[
                "evaluation_fingerprint_sha256"
            ],
            "evaluation_commitment_sha256": commitment_sha,
            "evaluation_started_sha256": started_sha,
            "fit_completion_sha256": seed_inputs[seed]["fit_completion_sha256"],
            "prospective_test_cache_sha256": seed_inputs[seed][
                "prospective_test_cache_sha256"
            ],
            "metrics_sha256": _sha256(seed_dir / "metrics.csv"),
            "decisions_sha256": _sha256(seed_dir / "per_sample_decisions.csv"),
        }
        (seed_dir / "evaluation_complete.json").write_text(
            json.dumps(completion, indent=2) + "\n", encoding="utf-8"
        )

    # The tracked report is intentionally last: it can only appear after every
    # seed evaluation artifact and completion has been materialized.
    aggregate_json_path = output_root / AGGREGATE_JSON_NAME
    aggregate_csv_path = output_root / AGGREGATE_CSV_NAME
    aggregate_json_path.write_text(aggregate_json_text, encoding="utf-8")
    aggregate_comparisons.to_csv(aggregate_csv_path, index=False)
    PRIMARY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRIMARY_REPORT_PATH.write_text(report_text, encoding="utf-8")
    seed_completion_sha = {
        str(seed): _sha256(output_root / f"seed{seed}" / "evaluation_complete.json")
        for seed in EXPECTED_SEEDS
    }
    aggregate_completion = {
        "schema_version": AGGREGATE_COMPLETION_SCHEMA,
        "evaluation_fingerprint_sha256": fingerprint[
            "evaluation_fingerprint_sha256"
        ],
        "evaluation_commitment_sha256": commitment_sha,
        "evaluation_started_sha256": started_sha,
        "seed_evaluation_completion_sha256": seed_completion_sha,
        "aggregate_json_sha256": _sha256(aggregate_json_path),
        "aggregate_csv_sha256": _sha256(aggregate_csv_path),
        "report_sha256": _sha256(PRIMARY_REPORT_PATH),
        "report_path": str(PRIMARY_REPORT_PATH.relative_to(REPO_ROOT)).replace(
            "\\", "/"
        ),
        "primary_decision": aggregate_analysis["decision"],
    }
    aggregate_completion_path = output_root / AGGREGATE_COMPLETION_NAME
    aggregate_completion_path.write_text(
        json.dumps(aggregate_completion, indent=2) + "\n", encoding="utf-8"
    )
    registry_completion = {
        "schema_version": REGISTRY_COMPLETION_SCHEMA,
        "evaluation_fingerprint_sha256": fingerprint[
            "evaluation_fingerprint_sha256"
        ],
        "evaluation_commitment_sha256": commitment_sha,
        "evaluation_started_sha256": started_sha,
        "locked_output_root": str(output_root),
        "aggregate_completion_sha256": _sha256(aggregate_completion_path),
        "primary_decision": aggregate_analysis["decision"],
        "status": "prospective_test_evaluation_complete",
    }
    # The protocol-level registry completion is written last.  A started entry
    # without this file permanently records an interrupted/consumed scoring pass.
    registry_complete_path.write_text(
        json.dumps(registry_completion, indent=2) + "\n", encoding="utf-8"
    )


def _canonical_commitment_file_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), indent=2) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    if (
        args.classifier_epochs < 1
        or args.utility_epochs < 1
        or args.patience < 0
        or args.batch_size < 1
        or args.learning_rate <= 0
        or args.num_workers < 0
    ):
        raise ValueError("Training hyperparameters must be positive")
    if args.phase == "fit":
        fit_all_seeds(args)
    else:
        evaluate_once(args)


if __name__ == "__main__":
    main()
