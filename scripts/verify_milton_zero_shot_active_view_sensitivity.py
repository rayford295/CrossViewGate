from __future__ import annotations

"""Independently verify the completed Milton zero-shot sensitivity ledger.

This verifier deliberately does not import the scoring entrypoint or any model
runtime.  Model artifacts and forward-cache ``.npz`` files are treated as
opaque byte strings: their SHA-256 digests are checked, but they are never
deserialized.  All semantic recomputation starts from the registered transfer
manifest and the emitted decision CSV files.
"""

import argparse
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_SCHEMA = "milton-zero-shot-active-view-sensitivity-v1"
COMMITMENT_SCHEMA = "milton-zero-shot-active-view-commitment-v1"
STARTED_SCHEMA = "milton-zero-shot-active-view-started-v1"
COMPLETION_SCHEMA = "milton-zero-shot-active-view-completion-v1"
AGGREGATE_SCHEMA = "milton-zero-shot-active-view-aggregate-v1"

EXPECTED_SEEDS = (42, 123, 456, 789, 1011)
EXPECTED_ORIGINS = tuple(range(8))
EXPECTED_SAMPLES = 1_707
EXPECTED_DEPENDENCY_GROUPS = 259
EXPECTED_SPATIAL_BLOCKS = 57
EXPECTED_JOINT_COMPONENTS = 11
EXPECTED_PER_SEED_RAW_ROWS = 518_928
EXPECTED_PER_SEED_TRAJECTORY_ROWS = 95_592
EXPECTED_PER_SEED_METRIC_ROWS = 126
EXPECTED_TOTAL_RAW_ROWS = 2_594_640
EXPECTED_TOTAL_TRAJECTORY_ROWS = 477_960
EXPECTED_POST_ORIGIN_ROWS = 59_745
EXPECTED_POST_SEED_ROWS = 11_949
EXPECTED_CONTRAST_ROWS = 12

FROZEN_CONFIG_SHA256 = (
    "c9457127d156aae00770d36196bdc1526934ca30bf5269978f75cadf088df463"
)
# Updated only if the final pre-score audit legitimately freezes a new scorer.
FROZEN_SCORER_SHA256 = (
    "142e0dc167482e33df7f34868bdbe265ce578b20f5cad7aeab9b59fa73e886b8"
)

SCORER_RELATIVE_PATH = Path("scripts/run_milton_zero_shot_active_view_sensitivity.py")
DEFAULT_CONFIG_RELATIVE_PATH = Path(
    "configs/milton_zero_shot_active_view_sensitivity_v1.json"
)
REPORT_RELATIVE_PATH = Path("docs/results/active_view_milton_zero_shot_v1.md")
GLOBAL_STARTED_NAME = "milton_zero_shot_active_view_sensitivity_v1_started.json"
GLOBAL_COMPLETE_NAME = "milton_zero_shot_active_view_sensitivity_v1_complete.json"

DECISION_COLUMNS = (
    "seed",
    "physical_origin",
    "policy",
    "policy_scope",
    "sample_id",
    "dependency_group_id",
    "spatial_block_id",
    "joint_component_id",
    "trajectory",
    "target",
    "prediction",
    "local_revealed_order",
    "physical_revealed_order",
    "view_count",
    "additional_view_count",
    "p0",
    "p1",
    "p2",
    "model_bayes_risk",
    "classification_cost",
    "view_cost",
    "operational_cost",
    "severe_miss",
    "correct",
)

METRIC_COLUMNS = (
    "seed",
    "scope",
    "physical_origin",
    "policy",
    "aggregation_unit",
    "groups",
    "sample_rows",
    "equal_unit_macro_operational_cost",
    "equal_unit_macro_severe_miss_rate",
    "equal_unit_macro_accuracy",
)

CONTRAST_COLUMNS = (
    "comparator",
    "comparator_scope",
    "main_policy",
    "aggregation_unit",
    "units",
    "cost_difference_comparator_minus_main",
    "primary_joint_component_bootstrap_95_ci_low",
    "primary_joint_component_bootstrap_95_ci_high",
    "secondary_direct_unit_bootstrap_95_ci_low",
    "secondary_direct_unit_bootstrap_95_ci_high",
    "severe_difference_comparator_minus_main",
    "rotation_contrast_median",
    "rotation_contrast_min",
    "rotation_contrast_max",
    "rotation_contrast_range",
    "rotation_contrast_worst_for_main",
    "descriptive_only",
    "go_no_go_test",
)


class VerificationError(RuntimeError):
    """Raised when a byte-level or semantic ledger edge does not verify."""


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise VerificationError(f"Required file is missing: {path}")
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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VerificationError(f"Expected a JSON object: {path}")
    return payload


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _expect_fields(payload: Mapping[str, Any], expected: Mapping[str, Any], context: str) -> None:
    for field, value in expected.items():
        if payload.get(field) != value:
            raise VerificationError(
                f"{context} field {field!r} changed: {payload.get(field)!r} != {value!r}"
            )


def _path_entry(
    entry: Mapping[str, Any], *, repo_root: Path, context: str
) -> tuple[Path, str]:
    path_value, declared = entry.get("path"), entry.get("sha256")
    if not isinstance(path_value, str) or not isinstance(declared, str):
        raise VerificationError(f"{context} is not a path/SHA-256 inventory entry")
    path = _resolve(repo_root, path_value)
    actual = _sha256(path)
    if actual != declared:
        raise VerificationError(
            f"{context} SHA-256 mismatch: declared {declared}, observed {actual}"
        )
    return path, actual


def _verify_inventory_paths(
    value: Any, *, repo_root: Path, context: str = "input_inventory"
) -> dict[str, str]:
    """Recursively hash every declared inventory path without opening its format."""

    verified: dict[str, str] = {}
    if isinstance(value, Mapping):
        if "path" in value or "sha256" in value:
            path, digest = _path_entry(value, repo_root=repo_root, context=context)
            verified[str(path)] = digest
            return verified
        for key, child in value.items():
            verified.update(
                _verify_inventory_paths(
                    child, repo_root=repo_root, context=f"{context}.{key}"
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            verified.update(
                _verify_inventory_paths(
                    child, repo_root=repo_root, context=f"{context}[{index}]"
                )
            )
    return verified


def _policy_inventory(config: Mapping[str, Any]) -> tuple[str, ...]:
    evaluation = config["evaluation"]
    policies = tuple(
        [evaluation["main_policy"]]
        + list(evaluation["comparators"])
        + list(evaluation["ood_diagnostics"])
        + list(evaluation["privileged_diagnostics"])
    )
    _expect(len(policies) == 7 and len(set(policies)) == 7, "Policy inventory changed")
    return policies


def joint_dependency_components(
    dependency_group_id: Sequence[object], spatial_block_id: Sequence[object]
) -> np.ndarray:
    """Rebuild deterministic components of the group--block bipartite graph."""

    dependencies = np.asarray(dependency_group_id).astype(str)
    blocks = np.asarray(spatial_block_id).astype(str)
    _expect(
        dependencies.ndim == 1 and dependencies.shape == blocks.shape,
        "Dependency-group and spatial-block arrays do not align",
    )
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

    for dependency, block in zip(dependencies, blocks):
        union(f"dependency:{dependency}", f"block:{block}")
    roots = sorted({find(f"dependency:{value}") for value in dependencies})
    names = {root: f"joint_component_{index:02d}" for index, root in enumerate(roots)}
    return np.asarray(
        [names[find(f"dependency:{value}")] for value in dependencies], dtype="U32"
    )


def _grid_index(value: object) -> int:
    numeric = Decimal(str(value))
    if not numeric.is_finite():
        raise VerificationError(f"Manifest coordinate is non-finite: {value!r}")
    return int((numeric / Decimal("0.001")).to_integral_value(rounding=ROUND_FLOOR))


def _spatial_block(latitude: object, longitude: object) -> str:
    return (
        f"milton_grid_0p001deg:{_grid_index(latitude):+07d}:"
        f"{_grid_index(longitude):+08d}"
    )


def _verify_manifest_chain(
    *, repo_root: Path, config: Mapping[str, Any], inventory: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = config["source_contract"]
    milton_inventory = inventory.get("milton_source", {})
    contract_path, contract_sha = _path_entry(
        milton_inventory.get("manifest_contract", {}),
        repo_root=repo_root,
        context="input_inventory.milton_source.manifest_contract",
    )
    _expect(contract_sha == source["manifest_contract_sha256"], "Manifest contract lock changed")
    contract = _load_json(contract_path)
    _expect_fields(
        contract,
        {
            "schema_version": "milton-active-view-transfer-contract-v1",
            "rows": EXPECTED_SAMPLES,
            "expected_rows": EXPECTED_SAMPLES,
            "confirmatory_eligible": False,
            "fitting_permitted": False,
            "calibration_permitted": False,
            "model_or_policy_selection_permitted": False,
        },
        "Milton transfer contract",
    )
    manifest_path, manifest_sha = _path_entry(
        milton_inventory.get("transfer_manifest", {}),
        repo_root=repo_root,
        context="input_inventory.milton_source.transfer_manifest",
    )
    _expect(manifest_sha == source["transfer_manifest_sha256"], "Transfer manifest lock changed")
    _expect(
        manifest_path == (contract_path.parent / contract["manifest"]["filename"]).resolve(),
        "Transfer manifest path differs from its contract",
    )
    canonical_path, canonical_sha = _path_entry(
        milton_inventory.get("canonical_source_cohort", {}),
        repo_root=repo_root,
        context="input_inventory.milton_source.canonical_source_cohort",
    )
    _expect(
        canonical_sha == source["canonical_source_cohort_sha256"]
        == contract["source"]["sha256"],
        "Canonical Milton cohort hash lock changed",
    )
    provenance_path, provenance_sha = _path_entry(
        milton_inventory.get("canonical_provenance_audit", {}),
        repo_root=repo_root,
        context="input_inventory.milton_source.canonical_provenance_audit",
    )
    _expect(
        provenance_sha == contract["source"]["provenance_audit_sha256"],
        "Canonical provenance audit hash lock changed",
    )
    _expect(
        canonical_path == Path(contract["source"]["path"]).resolve()
        and provenance_path == Path(contract["source"]["provenance_audit_path"]).resolve(),
        "Canonical source paths differ from the transfer contract",
    )

    string_fields = {
        "sample_id": str,
        "dependency_group_id": str,
        "spatial_block_id": str,
        "remote_content_group_id": str,
    }
    manifest = pd.read_csv(manifest_path, dtype=string_fields).reset_index(drop=True)
    canonical = pd.read_csv(
        canonical_path,
        dtype={"sample_id": str, "remote_content_group_id": str},
    ).reset_index(drop=True)
    required = {
        "sample_id",
        "label",
        "dependency_group_id",
        "remote_content_group_id",
        "spatial_block_id",
        "latitude",
        "longitude",
        "sequence_id",
        "sequence_id_is_observed",
        "compass_available",
    }
    _expect(len(manifest) == EXPECTED_SAMPLES and required.issubset(manifest.columns), "Manifest schema/count changed")
    _expect(len(canonical) == EXPECTED_SAMPLES, "Canonical source row count changed")
    _expect(not manifest["sample_id"].duplicated().any(), "Manifest sample IDs are not unique")
    for field in ("sample_id", "label", "latitude", "longitude", "remote_content_group_id"):
        _expect(field in canonical.columns, f"Canonical source lost field {field}")
        if field in {"latitude", "longitude"}:
            equal = np.allclose(
                manifest[field].to_numpy(dtype=np.float64),
                canonical[field].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            )
        else:
            equal = np.array_equal(
                manifest[field].astype(str).to_numpy(),
                canonical[field].astype(str).to_numpy(),
            )
        _expect(equal, f"Manifest/canonical source mismatch for {field}")
    coordinates = manifest[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    _expect(np.isfinite(coordinates).all(), "Manifest coordinates are not finite")
    _expect(
        ((coordinates[:, 0] >= -90.0) & (coordinates[:, 0] <= 90.0)).all()
        and ((coordinates[:, 1] >= -180.0) & (coordinates[:, 1] <= 180.0)).all(),
        "Manifest coordinates fall outside geographic bounds",
    )
    expected_blocks = np.asarray(
        [_spatial_block(lat, lon) for lat, lon in coordinates], dtype=str
    )
    _expect(
        np.array_equal(expected_blocks, manifest["spatial_block_id"].astype(str)),
        "Manifest spatial blocks do not reproduce from coordinates",
    )
    _expect(
        np.array_equal(
            manifest["dependency_group_id"].astype(str),
            manifest["remote_content_group_id"].astype(str),
        ),
        "Dependency groups are not exact remote-content groups",
    )
    _expect(
        manifest["sample_id"].map(lambda value: f"sequence_unavailable_for_sample:{value}").equals(
            manifest["sequence_id"].astype(str)
        ),
        "Missing-sequence sentinels changed",
    )
    for field in ("sequence_id_is_observed", "compass_available"):
        normalized = manifest[field].astype(str).str.strip().str.lower()
        _expect(normalized.isin({"false", "0"}).all(), f"Manifest unexpectedly enables {field}")
    labels = manifest["label"].to_numpy(dtype=np.int64)
    _expect(
        np.array_equal(np.bincount(labels, minlength=3), np.asarray([413, 817, 477])),
        "Manifest label distribution changed",
    )
    components = joint_dependency_components(
        manifest["dependency_group_id"], manifest["spatial_block_id"]
    )
    _expect(manifest["dependency_group_id"].nunique() == EXPECTED_DEPENDENCY_GROUPS, "Dependency-group count changed")
    _expect(manifest["spatial_block_id"].nunique() == EXPECTED_SPATIAL_BLOCKS, "Spatial-block count changed")
    _expect(len(np.unique(components)) == EXPECTED_JOINT_COMPONENTS, "Joint graph component count changed")
    manifest = manifest.copy()
    manifest["joint_component_id"] = components
    manifest["sorted_sample_ordinal"] = np.empty(len(manifest), dtype=np.int64)
    order = np.argsort(manifest["sample_id"].astype(str).to_numpy(), kind="mergesort")
    manifest.loc[order, "sorted_sample_ordinal"] = np.arange(len(manifest), dtype=np.int64)
    return manifest, {
        "manifest_sha256": manifest_sha,
        "canonical_source_sha256": canonical_sha,
        "provenance_audit_sha256": provenance_sha,
        "coordinate_rows_verified": int(len(manifest)),
        "joint_components": int(len(np.unique(components))),
    }


def _verify_registered_input_chain(
    *, repo_root: Path, config: Mapping[str, Any], inventory: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the config/source/cache/fit chain while treating binaries as opaque."""

    all_hashes = _verify_inventory_paths(inventory, repo_root=repo_root)
    config_path, config_sha = _path_entry(
        inventory.get("config", {}), repo_root=repo_root, context="input_inventory.config"
    )
    _expect(config_path == (repo_root / DEFAULT_CONFIG_RELATIVE_PATH).resolve(), "Registered config path changed")
    _expect(config_sha == FROZEN_CONFIG_SHA256, "Frozen Milton config SHA-256 changed")
    scorer_path, scorer_sha = _path_entry(
        inventory.get("source", {}).get("scorer", {}),
        repo_root=repo_root,
        context="input_inventory.source.scorer",
    )
    _expect(scorer_path == (repo_root / SCORER_RELATIVE_PATH).resolve(), "Registered scorer path changed")
    _expect(scorer_sha == FROZEN_SCORER_SHA256, "Frozen Milton scorer SHA-256 changed")

    source_contract = config["source_contract"]
    for inventory_key, config_key in (
        ("cache_entrypoint", "cache_entrypoint_sha256"),
        ("manifest_builder", "manifest_builder_sha256"),
    ):
        _, digest = _path_entry(
            inventory["milton_source"][inventory_key],
            repo_root=repo_root,
            context=f"input_inventory.milton_source.{inventory_key}",
        )
        _expect(digest == source_contract[config_key], f"Frozen {inventory_key} hash changed")
    _, cache_summary_sha = _path_entry(
        inventory["cache_summary"], repo_root=repo_root, context="input_inventory.cache_summary"
    )
    _expect(cache_summary_sha == source_contract["cache_summary_sha256"], "Cache summary lock changed")
    cache_summary_path = _resolve(repo_root, inventory["cache_summary"]["path"])
    cache_summary = _load_json(cache_summary_path)
    _expect_fields(
        cache_summary,
        {
            "schema_version": "milton-active-view-forward-cache-summary-v1",
            "forward_only": True,
            "confirmatory_eligible": False,
        },
        "Milton cache summary",
    )
    runs = cache_summary.get("runs", [])
    _expect(
        [int(run.get("seed", -1)) for run in runs] == list(EXPECTED_SEEDS),
        "Cache summary seed inventory changed",
    )
    runs_by_seed = {str(int(run["seed"])): run for run in runs}

    implementation = config["implementation_contract"]
    source_inventory = inventory["source"]
    for inventory_key, config_key in (
        ("feature_interface", "feature_interface_sha256"),
        ("active_view", "frozen_active_view_sha256"),
        ("active_view_utility", "frozen_active_view_utility_sha256"),
        ("frozen_cvian_runner", "frozen_cvian_runner_sha256"),
    ):
        _, digest = _path_entry(
            source_inventory[inventory_key],
            repo_root=repo_root,
            context=f"input_inventory.source.{inventory_key}",
        )
        _expect(digest == implementation[config_key], f"Frozen source {inventory_key} changed")
    fit_entrypoint = repo_root / "scripts" / "fit_cvian_relative_geometry_utility.py"
    _expect(
        _sha256(fit_entrypoint) == implementation["fit_entrypoint_sha256"],
        "Frozen relative-policy fit entrypoint changed",
    )

    relative_summary_path, _ = _path_entry(
        inventory["relative_fit_summary"],
        repo_root=repo_root,
        context="input_inventory.relative_fit_summary",
    )
    relative_summary = _load_json(relative_summary_path)
    _expect_fields(
        relative_summary,
        {
            "schema_version": "cvian-relative-geometry-utility-summary-v1",
            "config_sha256": FROZEN_CONFIG_SHA256,
            "fit_entrypoint_sha256": implementation["fit_entrypoint_sha256"],
            "feature_interface_sha256": implementation["feature_interface_sha256"],
            "seeds": list(EXPECTED_SEEDS),
            "prospective_test_loaded": False,
            "milton_data_loaded": False,
            "milton_fit_calibration_or_selection_performed": False,
        },
        "Relative-policy fit summary",
    )

    seed_inventory = inventory.get("seed_inputs", {})
    _expect(set(seed_inventory) == set(map(str, EXPECTED_SEEDS)), "Seed input inventory changed")
    for seed in EXPECTED_SEEDS:
        seed_key = str(seed)
        entries = seed_inventory[seed_key]
        expected_keys = {
            "cache",
            "main_completion",
            "main_artifact",
            "relative_completion",
            "relative_artifact",
        }
        _expect(set(entries) == expected_keys, f"Seed {seed} input keys changed")
        paths = {
            name: _path_entry(
                entry,
                repo_root=repo_root,
                context=f"input_inventory.seed_inputs.{seed_key}.{name}",
            )[0]
            for name, entry in entries.items()
        }
        run = runs_by_seed[seed_key]
        _expect_fields(
            run,
            {
                "schema_version": "milton-active-view-forward-cache-v1",
                "seed": seed,
                "forward_only": True,
                "parameters_updated": False,
                "calibration_performed": False,
                "selector_or_policy_applied": False,
                "cvian_prospective_test_read": False,
            },
            f"Milton cache run seed {seed}",
        )
        _expect(
            _sha256(paths["cache"]) == run["cache"]["sha256"],
            f"Seed {seed} cache summary edge changed",
        )
        for field, expected_hash in (
            ("manifest_contract_sha256", source_contract["manifest_contract_sha256"]),
            ("manifest_sha256", source_contract["transfer_manifest_sha256"]),
            ("source_cohort_sha256", source_contract["canonical_source_cohort_sha256"]),
        ):
            _expect(
                run["manifest_hashes"].get(field) == expected_hash,
                f"Seed {seed} cache manifest chain changed for {field}",
            )
        main_completion = _load_json(paths["main_completion"])
        _expect_fields(
            main_completion,
            {
                "schema_version": "cvian-sequence-utility-fit-v1",
                "seed": seed,
                "prospective_test_loaded": False,
            },
            f"Main CVIAN fit completion seed {seed}",
        )
        _expect(
            main_completion.get("fit_artifact_sha256") == _sha256(paths["main_artifact"]),
            f"Seed {seed} main fit artifact edge changed",
        )
        _expect(
            main_completion.get("source_sha256", {}).get("runner")
            == implementation["frozen_cvian_runner_sha256"],
            f"Seed {seed} frozen CVIAN runner edge changed",
        )
        relative_completion = _load_json(paths["relative_completion"])
        _expect_fields(
            relative_completion,
            {
                "schema_version": "cvian-relative-geometry-utility-fit-v1",
                "seed": seed,
                "config_sha256": FROZEN_CONFIG_SHA256,
                "feature_schema": "relative-geometry-utility-features-v1",
                "feature_width": 29,
                "classifier_parameters_updated": False,
                "base_encoder_parameters_updated": False,
                "temperature_updated": False,
                "milton_data_loaded": False,
                "prospective_test_loaded": False,
                "milton_fit_calibration_or_selection_performed": False,
            },
            f"Relative-policy fit completion seed {seed}",
        )
        _expect(
            relative_completion.get("artifact_sha256") == _sha256(paths["relative_artifact"]),
            f"Seed {seed} relative artifact edge changed",
        )
        _expect(
            relative_summary.get("seed_completion_sha256", {}).get(seed_key)
            == _sha256(paths["relative_completion"])
            and relative_summary.get("seed_artifact_sha256", {}).get(seed_key)
            == _sha256(paths["relative_artifact"]),
            f"Seed {seed} relative summary edge changed",
        )
        frozen_classifier = relative_completion.get("source_provenance", {}).get(
            "frozen_classifier", {}
        )
        _expect(
            frozen_classifier.get("fit_completion_sha256")
            == _sha256(paths["main_completion"])
            and frozen_classifier.get("fit_artifact_sha256")
            == _sha256(paths["main_artifact"]),
            f"Seed {seed} relative/main classifier chain changed",
        )
    return {
        "opaque_inventory_files_verified": int(len(all_hashes)),
        "config_sha256": config_sha,
        "scorer_sha256": scorer_sha,
        "cache_summary_sha256": cache_summary_sha,
        "seed_input_chains_verified": len(EXPECTED_SEEDS),
    }


def _parse_revealed_order(series: pd.Series, *, field: str) -> np.ndarray:
    parts = series.astype(str).str.split(",", expand=True)
    _expect(parts.shape[1] == 3, f"{field} does not contain exactly three sectors")
    try:
        values = parts.to_numpy(dtype=np.int64)
    except ValueError as exc:
        raise VerificationError(f"{field} contains a non-integer sector") from exc
    _expect(((values >= 0) & (values < 8)).all(), f"{field} contains an invalid sector")
    _expect(
        np.asarray([len(np.unique(row)) == 3 for row in values]).all(),
        f"{field} contains a repeated sector",
    )
    normalized = pd.Series([",".join(map(str, row)) for row in values], index=series.index)
    _expect(normalized.equals(series.astype(str)), f"{field} is not canonically encoded")
    return values


def _replay_random_orders(
    frame: pd.DataFrame,
    *,
    seed: int,
    ordinal_by_sample: Mapping[str, int],
) -> None:
    """Replay every registered SeedSequence, without using scorer code."""

    random_rows = frame.loc[
        frame["policy"].eq("random_mc32"),
        ["physical_origin", "trajectory", "sample_id", "local_revealed_order"],
    ]
    observed = _parse_revealed_order(
        random_rows["local_revealed_order"], field="random local_revealed_order"
    )
    expected = np.empty_like(observed)
    expected[:, 0] = 0
    for index, row in enumerate(random_rows.itertuples(index=False)):
        ordinal = ordinal_by_sample.get(str(row.sample_id))
        if ordinal is None:
            raise VerificationError(f"Random row uses unknown sample {row.sample_id}")
        generator = np.random.default_rng(
            np.random.SeedSequence(
                [seed, int(row.physical_origin), int(row.trajectory), int(ordinal), 20260710]
            )
        )
        available = list(range(1, 8))
        for step in (1, 2):
            offset = int(generator.integers(0, len(available)))
            expected[index, step] = available.pop(offset)
    _expect(np.array_equal(observed, expected), "Random SeedSequence trajectory replay failed")


def _validate_decision_semantics(
    frame: pd.DataFrame,
    *,
    seed: int,
    manifest: pd.DataFrame,
    config: Mapping[str, Any],
    expected_raw_rows: int = EXPECTED_PER_SEED_RAW_ROWS,
    replay_random: bool = True,
) -> dict[str, Any]:
    policies = _policy_inventory(config)
    _expect(tuple(frame.columns) == DECISION_COLUMNS, f"Seed {seed} decision columns changed")
    _expect(len(frame) == expected_raw_rows, f"Seed {seed} raw decision row count changed")
    numeric_integer = (
        "seed",
        "physical_origin",
        "trajectory",
        "target",
        "prediction",
        "view_count",
        "additional_view_count",
    )
    for column in numeric_integer:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        _expect(np.isfinite(values).all() and np.equal(values, np.floor(values)).all(), f"Seed {seed} {column} is not integral")
        frame[column] = values.astype(np.int64)
    _expect(frame["seed"].nunique() == 1 and int(frame["seed"].iloc[0]) == seed, f"Seed {seed} file contains another model seed")
    _expect(set(frame["physical_origin"]) == set(EXPECTED_ORIGINS), f"Seed {seed} origin inventory changed")
    _expect(set(frame["policy"].astype(str)) == set(policies), f"Seed {seed} policy inventory changed")

    keys = ["seed", "physical_origin", "policy", "sample_id", "trajectory"]
    _expect(not frame.duplicated(keys).any(), f"Seed {seed} decision lattice contains duplicates")
    multiplicity = frame.groupby(
        ["physical_origin", "policy", "sample_id"], sort=False, observed=True
    )["trajectory"].agg(["size", "nunique", "min", "max", "sum"])
    expected_groups = len(manifest) * len(EXPECTED_ORIGINS) * len(policies)
    _expect(len(multiplicity) == expected_groups, f"Seed {seed} sample/origin/policy lattice is incomplete")
    is_random = multiplicity.index.get_level_values("policy") == "random_mc32"
    expected_multiplicity = np.where(is_random, 32, 1)
    _expect(
        np.array_equal(multiplicity["size"].to_numpy(), expected_multiplicity)
        and np.array_equal(multiplicity["nunique"].to_numpy(), expected_multiplicity),
        f"Seed {seed} trajectory multiplicity changed",
    )
    _expect(
        (multiplicity.loc[is_random, "min"] == 0).all()
        and (multiplicity.loc[is_random, "max"] == 31).all()
        and (multiplicity.loc[is_random, "sum"] == 496).all()
        and (multiplicity.loc[~is_random, "min"] == 0).all()
        and (multiplicity.loc[~is_random, "max"] == 0).all(),
        f"Seed {seed} trajectory ID lattice changed",
    )

    manifest_index = pd.Index(manifest["sample_id"].astype(str))
    sample_values = frame["sample_id"].astype(str).to_numpy()
    positions = manifest_index.get_indexer(sample_values)
    _expect((positions >= 0).all(), f"Seed {seed} decisions contain unknown manifest IDs")
    for decision_field, manifest_field, numeric in (
        ("target", "label", True),
        ("dependency_group_id", "dependency_group_id", False),
        ("spatial_block_id", "spatial_block_id", False),
        ("joint_component_id", "joint_component_id", False),
    ):
        expected = manifest[manifest_field].to_numpy()[positions]
        observed = frame[decision_field].to_numpy()
        if numeric:
            expected = expected.astype(np.int64)
            observed = observed.astype(np.int64)
        else:
            expected = expected.astype(str)
            observed = observed.astype(str)
        _expect(np.array_equal(observed, expected), f"Seed {seed} decisions/manifest mismatch for {decision_field}")

    local = _parse_revealed_order(frame["local_revealed_order"], field="local_revealed_order")
    physical = _parse_revealed_order(frame["physical_revealed_order"], field="physical_revealed_order")
    origins = frame["physical_origin"].to_numpy(dtype=np.int64)
    _expect((local[:, 0] == 0).all(), f"Seed {seed} local trajectories do not start at zero")
    _expect(
        np.array_equal(physical, (local + origins[:, None]) % 8),
        f"Seed {seed} local-to-physical action mapping changed",
    )
    _expect(
        (frame["view_count"] == 3).all()
        and (frame["additional_view_count"] == 2).all(),
        f"Seed {seed} fixed view budget changed",
    )
    for policy, expected_order in (
        ("clockwise", np.asarray([0, 1, 2])),
        ("farthest", np.asarray([0, 4, 2])),
    ):
        selected = local[frame["policy"].eq(policy).to_numpy()]
        _expect(
            np.array_equal(selected, np.broadcast_to(expected_order, selected.shape)),
            f"Seed {seed} fixed {policy} trajectory changed",
        )
    if replay_random:
        ordinal_by_sample = dict(
            zip(
                manifest["sample_id"].astype(str),
                manifest["sorted_sample_ordinal"].astype(int),
            )
        )
        _replay_random_orders(frame, seed=seed, ordinal_by_sample=ordinal_by_sample)

    expected_scopes = {
        "relative_geometry_utility": "rolled_local_main_or_comparator",
        "farthest": "rolled_local_main_or_comparator",
        "clockwise": "rolled_local_main_or_comparator",
        "random_mc32": "rolled_local_main_or_comparator",
        "absolute_aware_utility_ood_diagnostic": "unrolled_classifier_off_support_ood_diagnostic",
        "max_confidence_privileged": "rolled_local_privileged_diagnostic",
        "greedy_label_oracle_privileged": "rolled_local_privileged_diagnostic",
    }
    observed_scope = frame["policy"].map(expected_scopes)
    _expect(observed_scope.notna().all() and observed_scope.equals(frame["policy_scope"].astype(str)), f"Seed {seed} policy scopes changed")

    probability_columns = ["p0", "p1", "p2"]
    probabilities = frame[probability_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    _expect(
        np.isfinite(probabilities).all()
        and (probabilities >= 0.0).all()
        and np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6),
        f"Seed {seed} probabilities are invalid",
    )
    cost_matrix = np.asarray(config["cost_matrix"], dtype=np.float64)
    action_risk = probabilities @ cost_matrix
    expected_prediction = action_risk.argmin(axis=1).astype(np.int64)
    expected_bayes_risk = action_risk[np.arange(len(frame)), expected_prediction]
    observed_prediction = frame["prediction"].to_numpy(dtype=np.int64)
    _expect(np.array_equal(observed_prediction, expected_prediction), f"Seed {seed} Bayes actions do not reproduce")
    numeric_float_columns = [
        "model_bayes_risk",
        "classification_cost",
        "view_cost",
        "operational_cost",
        "severe_miss",
        "correct",
    ]
    for column in numeric_float_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float64)
        _expect(np.isfinite(frame[column]).all(), f"Seed {seed} {column} is non-finite")
    _expect(
        np.allclose(frame["model_bayes_risk"], expected_bayes_risk, rtol=1e-12, atol=1e-12),
        f"Seed {seed} model Bayes risks do not reproduce",
    )
    targets = frame["target"].to_numpy(dtype=np.int64)
    expected_classification = cost_matrix[targets, expected_prediction]
    expected_view_cost = np.full(len(frame), 2.0 * float(config["view_cost"]))
    expected_operational = expected_classification + expected_view_cost
    expected_severe = ((targets == 2) & (expected_prediction != 2)).astype(np.float64)
    expected_correct = (targets == expected_prediction).astype(np.float64)
    for column, expected in (
        ("classification_cost", expected_classification),
        ("view_cost", expected_view_cost),
        ("operational_cost", expected_operational),
        ("severe_miss", expected_severe),
        ("correct", expected_correct),
    ):
        _expect(
            np.allclose(frame[column], expected, rtol=0.0, atol=1e-12),
            f"Seed {seed} {column} does not reproduce",
        )
    return {
        "raw_rows": int(len(frame)),
        "samples": int(frame["sample_id"].nunique()),
        "policies": int(frame["policy"].nunique()),
        "origins": int(frame["physical_origin"].nunique()),
        "random_seed_sequences_replayed": int(
            frame["policy"].eq("random_mc32").sum() if replay_random else 0
        ),
    }


def trajectory_average(decisions: pd.DataFrame) -> pd.DataFrame:
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
    return (
        decisions.groupby(keys, sort=True, observed=True)
        .agg(
            operational_cost=("operational_cost", "mean"),
            severe_miss=("severe_miss", "mean"),
            correct=("correct", "mean"),
            trajectory_count=("trajectory", "nunique"),
        )
        .reset_index()
    )


def recompute_seed_metrics(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectory = trajectory_average(decisions)
    expected_count = np.where(trajectory["policy"].eq("random_mc32"), 32, 1)
    _expect(
        np.array_equal(trajectory["trajectory_count"].to_numpy(dtype=np.int64), expected_count),
        "Trajectory averaging multiplicity changed",
    )
    rows: list[dict[str, Any]] = []
    units = ("dependency_group_id", "spatial_block_id")
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
    origin_average = (
        trajectory.groupby(origin_keys, sort=True, observed=True)
        .agg(
            operational_cost=("operational_cost", "mean"),
            severe_miss=("severe_miss", "mean"),
            correct=("correct", "mean"),
            origins=("physical_origin", "nunique"),
        )
        .reset_index()
    )
    _expect((origin_average["origins"] == 8).all(), "Origin averaging lost a rotation")
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
    return pd.DataFrame(rows, columns=METRIC_COLUMNS), trajectory


def direct_unit_bootstrap_interval(
    unit_values: np.ndarray, *, resamples: int, seed: int, confidence: float
) -> tuple[float, float]:
    values = np.asarray(unit_values, dtype=np.float64).reshape(-1)
    _expect(len(values) >= 2 and np.isfinite(values).all(), "Direct-unit bootstrap input is invalid")
    generator = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 256):
        stop = min(resamples, start + 256)
        index = generator.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[index].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha], method="linear")
    return float(low), float(high)


def joint_cluster_bootstrap_interval(
    unit_frame: pd.DataFrame,
    *,
    value_column: str,
    cluster_column: str,
    resamples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    _expect(not unit_frame[cluster_column].isna().any(), "A bootstrap unit has no joint component")
    cluster = unit_frame.groupby(cluster_column, sort=True)[value_column].agg(["sum", "count"])
    _expect(len(cluster) >= 2 and np.isfinite(cluster.to_numpy(dtype=float)).all(), "Joint-cluster bootstrap input is invalid")
    sums = cluster["sum"].to_numpy(dtype=np.float64)
    counts = cluster["count"].to_numpy(dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 256):
        stop = min(resamples, start + 256)
        index = generator.integers(0, len(cluster), size=(stop - start, len(cluster)))
        estimates[start:stop] = sums[index].sum(axis=1) / counts[index].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha], method="linear")
    return float(low), float(high)


def recompute_aggregate(
    trajectory_rows: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    raw_decision_rows: int,
    enforce_registered_counts: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
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
    origin_stage = (
        trajectory_rows.groupby(["seed"] + observation_keys, sort=True, observed=True)
        .agg(
            operational_cost=("operational_cost", "mean"),
            severe_miss=("severe_miss", "mean"),
            correct=("correct", "mean"),
            origins=("physical_origin", "nunique"),
            repeated_rows=("physical_origin", "size"),
        )
        .reset_index()
    )
    _expect((origin_stage["origins"] == 8).all() and (origin_stage["repeated_rows"] == 8).all(), "Origin-stage repeated measures changed")
    collapsed = (
        origin_stage.groupby(observation_keys, sort=True, observed=True)
        .agg(
            operational_cost=("operational_cost", "mean"),
            severe_miss=("severe_miss", "mean"),
            correct=("correct", "mean"),
            seeds=("seed", "nunique"),
            repeated_rows=("seed", "size"),
        )
        .reset_index()
    )
    _expect((collapsed["seeds"] == 5).all() and (collapsed["repeated_rows"] == 5).all(), "Seed-stage repeated measures changed")
    if enforce_registered_counts:
        _expect(
            raw_decision_rows == EXPECTED_TOTAL_RAW_ROWS
            and len(trajectory_rows) == EXPECTED_TOTAL_TRAJECTORY_ROWS
            and len(origin_stage) == EXPECTED_POST_ORIGIN_ROWS
            and len(collapsed) == EXPECTED_POST_SEED_ROWS,
            "Registered trajectory/origin/seed stage counts changed",
        )
    units = ("dependency_group_id", "spatial_block_id")
    policy_summary: dict[str, Any] = {}
    for policy in policies:
        subset = collapsed[collapsed["policy"] == policy]
        per_rotation = trajectory_rows[trajectory_rows["policy"] == policy]
        policy_summary[policy] = {}
        for unit in units:
            grouped = subset.groupby(unit, sort=True).agg(
                operational_cost=("operational_cost", "mean"),
                severe_miss=("severe_miss", "mean"),
                correct=("correct", "mean"),
            )
            rotation_cost = (
                per_rotation.groupby(["physical_origin", unit], sort=True)["operational_cost"]
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

    key_columns = ["sample_id", "dependency_group_id", "spatial_block_id", "joint_component_id"]
    metric_columns = ["operational_cost", "severe_miss", "correct"]
    main_rows = collapsed[collapsed["policy"] == main][key_columns + metric_columns]
    contrast_rows: list[dict[str, Any]] = []
    contrast_json: dict[str, Any] = {}
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
        _expect(len(paired) == len(main_rows) == len(other), f"Paired support changed for {comparator}")
        paired["cost_difference_comparator_minus_main"] = paired["operational_cost_comparator"] - paired["operational_cost_main"]
        paired["severe_difference_comparator_minus_main"] = paired["severe_miss_comparator"] - paired["severe_miss_main"]
        per_rotation_main = trajectory_rows[trajectory_rows["policy"] == main]
        per_rotation_other = trajectory_rows[trajectory_rows["policy"] == comparator]
        rotation_pair = per_rotation_main.merge(
            per_rotation_other,
            on=["seed", "physical_origin"] + key_columns + ["target"],
            validate="one_to_one",
            suffixes=("_main", "_comparator"),
        )
        rotation_pair["cost_difference"] = rotation_pair["operational_cost_comparator"] - rotation_pair["operational_cost_main"]
        contrast_json[comparator] = {}
        for unit in units:
            grouped = paired.groupby(unit, sort=True).agg(
                cost_difference=("cost_difference_comparator_minus_main", "mean"),
                severe_difference=("severe_difference_comparator_minus_main", "mean"),
                cluster_id=("joint_component_id", "first"),
                joint_component_memberships=("joint_component_id", "nunique"),
            )
            _expect((grouped["joint_component_memberships"] == 1).all(), f"{unit} crosses joint components")
            secondary_low, secondary_high = direct_unit_bootstrap_interval(
                grouped["cost_difference"].to_numpy(),
                resamples=int(bootstrap["resamples"]),
                seed=int(bootstrap["random_seed"]),
                confidence=float(bootstrap["confidence_level"]),
            )
            primary_low, primary_high = joint_cluster_bootstrap_interval(
                grouped.reset_index(),
                value_column="cost_difference",
                cluster_column="cluster_id",
                resamples=int(bootstrap["resamples"]),
                seed=int(bootstrap["random_seed"]),
                confidence=float(bootstrap["confidence_level"]),
            )
            rotation_values = (
                rotation_pair.groupby(["physical_origin", unit], sort=True)["cost_difference"]
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
                "cost_difference_comparator_minus_main": float(grouped["cost_difference"].mean()),
                "primary_joint_component_bootstrap_95_ci_low": primary_low,
                "primary_joint_component_bootstrap_95_ci_high": primary_high,
                "secondary_direct_unit_bootstrap_95_ci_low": secondary_low,
                "secondary_direct_unit_bootstrap_95_ci_high": secondary_high,
                "severe_difference_comparator_minus_main": float(grouped["severe_difference"].mean()),
                "rotation_contrast_median": float(rotation_values.median()),
                "rotation_contrast_min": float(rotation_values.min()),
                "rotation_contrast_max": float(rotation_values.max()),
                "rotation_contrast_range": float(rotation_values.max() - rotation_values.min()),
                "rotation_contrast_worst_for_main": float(rotation_values.min()),
                "descriptive_only": True,
                "go_no_go_test": False,
            }
            contrast_rows.append(record)
            contrast_json[comparator][unit] = record

    membership = (
        collapsed[["sample_id", "dependency_group_id", "spatial_block_id", "joint_component_id"]]
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
            "raw_decision_rows": int(raw_decision_rows),
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
            "secondary_descriptive_resampling_units": bootstrap["secondary_descriptive_resampling_units"],
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
        "physical_origins": list(EXPECTED_ORIGINS),
        "policy_summary": policy_summary,
        "contrasts": contrast_json,
    }
    return aggregate, pd.DataFrame(contrast_rows, columns=CONTRAST_COLUMNS)


def _compare_frames(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    sort_by: Sequence[str],
    context: str,
    atol: float = 2e-12,
) -> None:
    _expect(tuple(observed.columns) == tuple(expected.columns), f"{context} columns changed")
    left = observed.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    right = expected.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    _expect(len(left) == len(right), f"{context} row count changed")
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(right[column]):
            left_numeric = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=np.float64)
            right_numeric = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=np.float64)
            _expect(
                np.allclose(left_numeric, right_numeric, rtol=1e-12, atol=atol, equal_nan=True),
                f"{context} numeric column {column} differs from recomputation",
            )
        else:
            _expect(
                np.array_equal(left[column].astype(str), right[column].astype(str)),
                f"{context} column {column} differs from recomputation",
            )


def _compare_json(observed: Any, expected: Any, *, path: str = "aggregate") -> None:
    if isinstance(expected, Mapping):
        _expect(isinstance(observed, Mapping), f"{path} type changed")
        _expect(set(observed) == set(expected), f"{path} keys changed")
        for key in expected:
            _compare_json(observed[key], expected[key], path=f"{path}.{key}")
    elif isinstance(expected, list):
        _expect(isinstance(observed, list) and len(observed) == len(expected), f"{path} list changed")
        for index, value in enumerate(expected):
            _compare_json(observed[index], value, path=f"{path}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        _expect(
            isinstance(observed, (int, float))
            and np.isclose(float(observed), float(expected), rtol=1e-12, atol=2e-12),
            f"{path} differs from recomputation",
        )
    else:
        _expect(observed == expected, f"{path} differs from recomputation")


def _verify_commitment_and_started(
    *, repo_root: Path, output_root: Path, registry_root: Path
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
    commitment_path = output_root / "pre_score_commitment.json"
    local_started_path = output_root / "evaluation_started.json"
    global_started_path = registry_root / GLOBAL_STARTED_NAME
    commitment = _load_json(commitment_path)
    local_started = _load_json(local_started_path)
    global_started = _load_json(global_started_path)
    _expect_fields(
        commitment,
        {
            "schema_version": COMMITMENT_SCHEMA,
            "confirmatory_eligible": False,
            "go_no_go_criterion": None,
            "milton_parameter_fitting": False,
            "milton_calibration": False,
            "stop_or_adaptive_policy_enabled": False,
        },
        "Pre-score commitment",
    )
    fingerprint = _canonical_hash(commitment.get("input_inventory", {}))
    _expect(
        commitment.get("evaluation_fingerprint_sha256") == fingerprint,
        "Commitment input-inventory fingerprint changed",
    )
    expected_started = {
        "schema_version": STARTED_SCHEMA,
        "evaluation_fingerprint_sha256": fingerprint,
        "commitment_sha256": _sha256(commitment_path),
        "status": "started_before_first_semantic_milton_cache_load",
        "rerun_permitted": False,
    }
    _expect(local_started == expected_started, "Local started sentinel changed")
    _expect(global_started == expected_started, "Global started sentinel changed")
    _expect(
        _sha256(local_started_path) == _sha256(global_started_path),
        "Local/global started sentinel byte hashes differ",
    )
    return commitment, local_started, commitment_path, local_started_path, global_started_path


def _verify_seed_completion(
    *,
    seed: int,
    seed_dir: Path,
    decision_path: Path,
    metric_path: Path,
    commitment: Mapping[str, Any],
    commitment_path: Path,
    started_path: Path,
) -> Path:
    completion_path = seed_dir / "evaluation_complete.json"
    completion = _load_json(completion_path)
    _expect_fields(
        completion,
        {
            "schema_version": COMPLETION_SCHEMA,
            "scope": "per_seed",
            "seed": seed,
            "evaluation_fingerprint_sha256": commitment["evaluation_fingerprint_sha256"],
            "commitment_sha256": _sha256(commitment_path),
            "started_sha256": _sha256(started_path),
            "seed_input_inventory": commitment["input_inventory"]["seed_inputs"][str(seed)],
            "decisions_sha256": _sha256(decision_path),
            "metrics_sha256": _sha256(metric_path),
            "parameters_updated": False,
            "calibration_performed": False,
            "status": "seed_sensitivity_complete",
        },
        f"Seed {seed} completion",
    )
    _expect_fields(
        completion.get("row_counts", {}),
        {
            "raw_decision_rows": EXPECTED_PER_SEED_RAW_ROWS,
            "post_trajectory_rows": EXPECTED_PER_SEED_TRAJECTORY_ROWS,
            "origins": 8,
            "policies": 7,
            "samples": EXPECTED_SAMPLES,
            "random_trajectories": 32,
        },
        f"Seed {seed} completion row counts",
    )
    return completion_path


def verify_completion(
    *,
    repo_root: str | Path = REPO_ROOT,
    config_path: str | Path = DEFAULT_CONFIG_RELATIVE_PATH,
    replay_random: bool = True,
) -> dict[str, Any]:
    """Verify one completed sensitivity run without running scorer or inference."""

    root = Path(repo_root).resolve()
    config_file = _resolve(root, config_path)
    _expect(_sha256(config_file) == FROZEN_CONFIG_SHA256, "Frozen Milton config SHA-256 changed")
    config = _load_json(config_file)
    _expect_fields(
        config,
        {
            "schema_version": CONFIG_SCHEMA,
            "confirmatory_eligible": False,
            "seeds": list(EXPECTED_SEEDS),
            "sector_count": 8,
            "cyclic_origin_rotations": list(EXPECTED_ORIGINS),
            "fixed_budget_views": 3,
            "view_cost": 0.5,
        },
        "Milton sensitivity config",
    )
    evaluation = config["evaluation"]
    _expect_fields(
        evaluation,
        {
            "milton_parameter_fitting": False,
            "milton_calibration": False,
            "milton_model_selection": False,
            "milton_policy_selection": False,
            "stop_or_adaptive_policy_enabled": False,
            "go_no_go_criterion": None,
            "random_trajectories_per_seed_rotation": 32,
            "overwrite_or_rerun_permitted": False,
        },
        "Milton evaluation contract",
    )
    bootstrap = evaluation["bootstrap"]
    _expect_fields(
        bootstrap,
        {
            "primary_resampling_unit": "joint_dependency_spatial_component",
            "expected_joint_components": EXPECTED_JOINT_COMPONENTS,
            "resamples": 10_000,
            "random_seed": 42,
            "confidence_level": 0.95,
            "interval": "percentile",
        },
        "Milton bootstrap contract",
    )
    output_root = _resolve(root, evaluation["output_root"])
    registry_root = output_root.parent / "evaluation_registry"
    completion_path = output_root / evaluation["completion"]
    _expect(completion_path.is_file(), "Real Milton sensitivity completion does not exist; verifier will not run early")

    commitment, _, commitment_path, local_started_path, global_started_path = (
        _verify_commitment_and_started(
            repo_root=root, output_root=output_root, registry_root=registry_root
        )
    )
    inventory = commitment["input_inventory"]
    input_summary = _verify_registered_input_chain(
        repo_root=root, config=config, inventory=inventory
    )
    manifest, manifest_summary = _verify_manifest_chain(
        repo_root=root, config=config, inventory=inventory
    )

    trajectory_frames: list[pd.DataFrame] = []
    seed_completion_paths: dict[str, Path] = {}
    seed_summaries: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        decision_path = output_root / evaluation["per_seed_decisions"].format(seed=seed)
        metric_path = output_root / evaluation["per_seed_metrics"].format(seed=seed)
        seed_dir = decision_path.parent
        final_completion = _load_json(completion_path)
        declared_seed = final_completion.get("outputs", {}).get(str(seed), {})
        _path_entry(declared_seed.get("decisions", {}), repo_root=root, context=f"completion.outputs.{seed}.decisions")
        _path_entry(declared_seed.get("metrics", {}), repo_root=root, context=f"completion.outputs.{seed}.metrics")
        _expect(
            _resolve(root, declared_seed["decisions"]["path"]) == decision_path.resolve()
            and _resolve(root, declared_seed["metrics"]["path"]) == metric_path.resolve(),
            f"Seed {seed} final output paths changed",
        )
        decisions = pd.read_csv(
            decision_path,
            dtype={
                "sample_id": str,
                "dependency_group_id": str,
                "spatial_block_id": str,
                "joint_component_id": str,
                "policy": str,
                "policy_scope": str,
                "local_revealed_order": str,
                "physical_revealed_order": str,
            },
        )
        seed_summaries[str(seed)] = _validate_decision_semantics(
            decisions,
            seed=seed,
            manifest=manifest,
            config=config,
            replay_random=replay_random,
        )
        recomputed_metrics, trajectory = recompute_seed_metrics(decisions)
        _expect(len(trajectory) == EXPECTED_PER_SEED_TRAJECTORY_ROWS, f"Seed {seed} post-trajectory row count changed")
        observed_metrics = pd.read_csv(metric_path)
        _expect(tuple(observed_metrics.columns) == METRIC_COLUMNS, f"Seed {seed} metric columns changed")
        _expect(len(observed_metrics) == EXPECTED_PER_SEED_METRIC_ROWS, f"Seed {seed} metric row count changed")
        _compare_frames(
            observed_metrics,
            recomputed_metrics,
            sort_by=["seed", "scope", "physical_origin", "policy", "aggregation_unit"],
            context=f"Seed {seed} metrics",
        )
        seed_completion_path = _verify_seed_completion(
            seed=seed,
            seed_dir=seed_dir,
            decision_path=decision_path,
            metric_path=metric_path,
            commitment=commitment,
            commitment_path=commitment_path,
            started_path=local_started_path,
        )
        declared_completion_path, declared_completion_sha = _path_entry(
            declared_seed.get("completion", {}),
            repo_root=root,
            context=f"completion.outputs.{seed}.completion",
        )
        _expect(
            declared_completion_path == seed_completion_path.resolve()
            and declared_completion_sha == _sha256(seed_completion_path),
            f"Seed {seed} completion output edge changed",
        )
        seed_completion_paths[str(seed)] = seed_completion_path
        trajectory_frames.append(trajectory)
        del decisions, observed_metrics, recomputed_metrics

    all_trajectory = pd.concat(trajectory_frames, ignore_index=True)
    recomputed_aggregate, recomputed_contrasts = recompute_aggregate(
        all_trajectory,
        config,
        raw_decision_rows=EXPECTED_TOTAL_RAW_ROWS,
        enforce_registered_counts=True,
    )
    aggregate_path = output_root / evaluation["aggregate_json"]
    contrast_path = output_root / evaluation["aggregate_csv"]
    report_path = (root / REPORT_RELATIVE_PATH).resolve()
    observed_aggregate = _load_json(aggregate_path)
    observed_contrasts = pd.read_csv(contrast_path)
    _expect(tuple(observed_contrasts.columns) == CONTRAST_COLUMNS, "Aggregate contrast columns changed")
    _expect(len(observed_contrasts) == EXPECTED_CONTRAST_ROWS, "Aggregate contrast row count changed")
    _compare_json(observed_aggregate, recomputed_aggregate)
    _compare_frames(
        observed_contrasts,
        recomputed_contrasts,
        sort_by=["comparator", "aggregation_unit"],
        context="Aggregate contrasts",
    )

    completion = _load_json(completion_path)
    _expect_fields(
        completion,
        {
            "schema_version": COMPLETION_SCHEMA,
            "claim_scope": config["claim_scope"],
            "confirmatory_eligible": False,
            "go_no_go_decision_performed": False,
            "evaluation_fingerprint_sha256": commitment["evaluation_fingerprint_sha256"],
            "commitment_sha256": _sha256(commitment_path),
            "started_sha256": _sha256(local_started_path),
            "global_started_sha256": _sha256(global_started_path),
            "parameters_updated": False,
            "calibration_performed": False,
            "stop_or_adaptive_policy_enabled": False,
            "status": "sensitivity_complete_no_confirmation",
        },
        "Global sensitivity completion",
    )
    _expect_fields(
        completion.get("row_counts", {}),
        {
            "raw_decision_rows": EXPECTED_TOTAL_RAW_ROWS,
            "post_trajectory_rows": EXPECTED_TOTAL_TRAJECTORY_ROWS,
            "post_origin_rows": EXPECTED_POST_ORIGIN_ROWS,
            "post_seed_rows": EXPECTED_POST_SEED_ROWS,
            "aggregate_contrast_rows": EXPECTED_CONTRAST_ROWS,
        },
        "Global completion row counts",
    )
    aggregate_outputs = completion.get("outputs", {}).get("aggregate", {})
    for key, path in (("json", aggregate_path), ("csv", contrast_path), ("report", report_path)):
        declared_path, _ = _path_entry(
            aggregate_outputs.get(key, {}),
            repo_root=root,
            context=f"completion.outputs.aggregate.{key}",
        )
        _expect(declared_path == path, f"Aggregate {key} output path changed")

    registry_complete_path = registry_root / GLOBAL_COMPLETE_NAME
    registry_complete = _load_json(registry_complete_path)
    _expect_fields(
        registry_complete,
        {
            "schema_version": COMPLETION_SCHEMA,
            "scope": "global_registry",
            "evaluation_fingerprint_sha256": commitment["evaluation_fingerprint_sha256"],
            "commitment_sha256": _sha256(commitment_path),
            "started_sha256": _sha256(global_started_path),
            "locked_output_root": str(output_root),
            "completion_sha256": _sha256(completion_path),
            "status": "sensitivity_complete_no_confirmation",
        },
        "Global registry completion",
    )
    return {
        "schema_version": "milton-zero-shot-active-view-independent-verification-v1",
        "verified": True,
        "claim_status": "sensitivity_only_no_confirmation",
        "confirmatory_eligible": False,
        "go_no_go_decision_performed": False,
        "scoring_reexecuted": False,
        "model_inference_performed": False,
        "model_artifacts_semantically_loaded": False,
        "milton_cache_npz_semantically_loaded": False,
        "random_seedsequence_replay_performed": bool(replay_random),
        "input_chain": input_summary,
        "manifest_chain": manifest_summary,
        "seed_decision_verification": seed_summaries,
        "averaging_stage_rows": recomputed_aggregate["averaging_stage_rows"],
        "bootstrap": recomputed_aggregate["bootstrap"],
        "aggregate_semantically_recomputed": True,
        "aggregate_json_semantically_identical": True,
        "aggregate_csv_semantically_identical": True,
        "commitment_sha256": _sha256(commitment_path),
        "completion_sha256": _sha256(completion_path),
        "global_registry_completion_sha256": _sha256(registry_complete_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_RELATIVE_PATH))
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path under the registered output root; written exclusively.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.repo_root).resolve()
    result = verify_completion(repo_root=root, config_path=args.config)
    config = _load_json(_resolve(root, args.config))
    output_root = _resolve(root, config["evaluation"]["output_root"])
    output_path = (
        _resolve(root, args.json_output)
        if args.json_output
        else output_root / "independent_completion_verification.json"
    )
    try:
        output_path.relative_to(output_root)
    except ValueError as exc:
        raise VerificationError("Verification JSON must be written under the registered output root") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
