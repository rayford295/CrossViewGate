from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts import verify_cvian_sequence_utility_completion as verifier


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(
    path: Path, payload: dict[str, Any], *, crlf: bool = False, sort_keys: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n"
    if crlf:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _decision_frame(seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    samples = (
        ("sample-a", "component-a", 0),
        ("sample-b", "component-b", 1),
        ("sample-c", "component-b", 2),
    )
    costs = {
        "utility_regression": (1.0, 1.0, 1.0),
        "farthest": (2.0, 1.5, 1.5),
        "max_building_privileged": (1.5, 1.25, 1.25),
    }
    for policy, policy_costs in costs.items():
        for index, ((sample_id, component, target), cost) in enumerate(
            zip(samples, policy_costs)
        ):
            rows.append(
                {
                    "seed": seed,
                    "policy": policy,
                    "scope": "fixed_k3_primary",
                    "sample_id": sample_id,
                    "sequence_id": f"sequence-{index}",
                    "spatial_block_id": f"block-{index}",
                    "dependency_component": component,
                    "trajectory": 0,
                    "target": target,
                    "prediction": target,
                    "terminal_action": "stop",
                    "view_count": 3,
                    "operational_cost": cost,
                    "severe_miss": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _build_synthetic_completion(repo_root: Path, *, crlf_commitment: bool) -> dict[str, Path]:
    config_path = repo_root / "configs/cvian_sequence_utility_v1.json"
    config_path.parent.mkdir(parents=True)
    # Preserve the registered semantics while relocating only by virtue of the
    # verifier's injected repository root; all execution-lock paths are relative.
    config = json.loads(
        (verifier.REPO_ROOT / "configs/cvian_sequence_utility_v1.json").read_text(
            encoding="utf-8"
        )
    )
    _write_json(config_path, config)

    for index, relative in enumerate(verifier.SOURCE_FILES.values()):
        _write_bytes(repo_root / relative, f"frozen source {index}\n".encode("ascii"))
    source_hashes = {
        name: _sha(repo_root / relative)
        for name, relative in verifier.SOURCE_FILES.items()
    }

    protocol_dir = repo_root / "data/splits/ian_hurricane_sequence_four_role_v1"
    protocol_dir.mkdir(parents=True)
    role_rows = {role: index + 1 for index, role in enumerate(verifier.ROLE_NAMES)}
    role_hashes: dict[str, str] = {}
    for role in verifier.ROLE_NAMES:
        path = protocol_dir / f"{role}.csv"
        lines = ["sample_id,label"] + [
            f"{role}-{index},{index % 3}" for index in range(role_rows[role])
        ]
        _write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))
        role_hashes[role] = _sha(path)
    summary = {
        "schema_version": "cvian-sequence-four-role-summary-v1",
        "role_rows": role_rows,
        "role_manifest_sha256": {
            f"{role}.csv": role_hashes[role] for role in verifier.ROLE_NAMES
        },
        "record_overlap_zero": True,
        "sequence_overlap_zero": True,
        "spatial_block_overlap_zero": True,
        "spatial_buffer_clear": True,
        "test_status": {
            "status": "selector_selection_holdout_with_historical_base_exposure",
            "new_protocol_test_scored": False,
            "old_base_or_checkpoint_reuse_permitted": False,
        },
    }
    protocol_test_commitment = {
        "schema_version": "cvian-sequence-test-commitment-v1",
        "role": "prospective_test",
        "status": "selector_selection_holdout_with_historical_base_exposure",
        "historical_checkpoint_reuse_permitted": False,
        "row_count": role_rows["prospective_test"],
        "manifest_sha256": role_hashes["prospective_test"],
    }
    summary_path = protocol_dir / "protocol_summary.json"
    protocol_commitment_path = protocol_dir / "test_commitment.json"
    _write_json(summary_path, summary)
    _write_json(protocol_commitment_path, protocol_test_commitment)
    _, protocol_registration_sha = verifier._protocol_registration(
        protocol_dir, summary, protocol_test_commitment
    )

    output_root = repo_root / "outputs/cvian_sequence_active_v2/utility_experiment"
    cache_root = repo_root / "outputs/cvian_sequence_active_v2/cache"
    visibility_dir = repo_root / "outputs/cvian_sequence_active_v2/visibility"
    registry_root = repo_root / "outputs/cvian_sequence_active_v2/evaluation_registry"
    report_path = repo_root / "docs/results/cvian_sequence_utility_v1.md"

    visibility_hashes: dict[str, str] = {}
    visibility_splits: dict[str, Any] = {}
    for role in verifier.ROLE_NAMES:
        path = visibility_dir / f"{role}.npz"
        _write_bytes(path, f"opaque visibility {role}".encode("ascii"))
        visibility_hashes[role] = _sha(path)
        visibility_splits[role] = {"sha256": visibility_hashes[role]}
    visibility_metadata = {
        "schema_version": "cvian-sector-visibility-v3",
        "model_revision_resolved": "synthetic-revision",
        "splits": visibility_splits,
    }
    visibility_metadata_path = visibility_dir / "visibility_metadata.json"
    _write_json(visibility_metadata_path, visibility_metadata)
    visibility_compact = {
        "protocol_registration_sha256": protocol_registration_sha,
        "visibility_metadata_sha256": _sha(visibility_metadata_path),
        "visibility_artifact_sha256": visibility_hashes,
        "segformer_revision": "synthetic-revision",
    }
    visibility_provenance = {
        "protocol_registration_sha256": protocol_registration_sha,
        "visibility_metadata_sha256": _sha(visibility_metadata_path),
        "visibility_artifacts_sha256": verifier._canonical_hash(visibility_hashes),
        "prospective_test_visibility_sha256": visibility_hashes[
            "prospective_test"
        ],
        "visibility_provenance_sha256": verifier._canonical_hash(
            visibility_compact
        ),
    }

    seed_inputs: dict[str, Any] = {}
    decision_frames: list[pd.DataFrame] = []
    for seed in verifier.EXPECTED_SEEDS:
        cache_dir = cache_root / f"seed{seed}"
        cache_hashes: dict[str, str] = {}
        cache_splits: dict[str, Any] = {}
        for role in verifier.ROLE_NAMES:
            path = cache_dir / f"{role}.npz"
            _write_bytes(path, f"opaque embedding {seed} {role}".encode("ascii"))
            cache_hashes[role] = _sha(path)
            cache_splits[role] = {"sha256": cache_hashes[role]}
        cache_metadata = {
            "schema_version": "cvian-active-view-cache-v3",
            "seed": seed,
            "source_manifest_sha256": role_hashes,
            "splits": cache_splits,
        }
        cache_metadata_path = cache_dir / "cache_metadata.json"
        _write_json(cache_metadata_path, cache_metadata)

        seed_dir = output_root / f"seed{seed}"
        for field, filename in verifier.FIT_ARTIFACT_FIELDS.items():
            _write_bytes(seed_dir / filename, f"{field} seed {seed}".encode("ascii"))
        fit_payload: dict[str, Any] = {
            "schema_version": verifier.FIT_SCHEMA,
            "seed": seed,
            "config_sha256": _sha(config_path),
            "protocol_summary_sha256": _sha(summary_path),
            "test_commitment_sha256": _sha(protocol_commitment_path),
            "fit_manifest_sha256": {
                role: role_hashes[role] for role in verifier.FIT_ROLES
            },
            "input_provenance": {
                "base_completion_sha256": f"base-{seed}",
                "checkpoint_sha256": f"checkpoint-{seed}",
                "cache_metadata_sha256": _sha(cache_metadata_path),
                "cache_sha256": {
                    role: cache_hashes[role] for role in verifier.FIT_ROLES
                },
                "strict_provenance": {"synthetic": True},
            },
            "source_sha256": source_hashes,
            "git_head": "synthetic-head",
            "runtime_versions": {"python": "synthetic"},
            "base_runtime_lock_provenance": {"synthetic": True},
            "hyperparameters": config["training_hyperparameters"],
            "prospective_test_loaded": False,
        }
        fit_payload["fit_fingerprint_sha256"] = verifier._canonical_hash(
            {
                field: fit_payload[field]
                for field in verifier.FIT_FINGERPRINT_FIELDS
            }
        )
        for field, filename in verifier.FIT_ARTIFACT_FIELDS.items():
            fit_payload[field] = _sha(seed_dir / filename)
        fit_path = seed_dir / "fit_complete.json"
        _write_json(fit_path, fit_payload)

        decisions = _decision_frame(seed)
        decision_path = seed_dir / "per_sample_decisions.csv"
        metrics_path = seed_dir / "metrics.csv"
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decisions.to_csv(decision_path, index=False, lineterminator="\n")
        pd.DataFrame([{"seed": seed, "metric": 1.0}]).to_csv(
            metrics_path, index=False, lineterminator="\n"
        )
        decision_frames.append(decisions)
        seed_inputs[str(seed)] = {
            "fit_completion_sha256": _sha(fit_path),
            "fit_artifact_sha256": _sha(seed_dir / "fit_artifact.pt"),
            "prospective_test_cache_sha256": cache_hashes["prospective_test"],
            "cache_metadata_sha256": _sha(cache_metadata_path),
        }

    commitment: dict[str, Any] = {
        "schema_version": verifier.COMMITMENT_SCHEMA,
        "config_sha256": _sha(config_path),
        "protocol_summary_sha256": _sha(summary_path),
        "test_commitment_sha256": _sha(protocol_commitment_path),
        "prospective_test_manifest_sha256": role_hashes["prospective_test"],
        "source_sha256": source_hashes,
        "git_head": "synthetic-head",
        "runtime_versions": {"python": "synthetic"},
        "fit_hyperparameters": config["training_hyperparameters"],
        "seed_inputs": seed_inputs,
        "visibility_sha256": visibility_hashes["prospective_test"],
        "visibility_provenance": visibility_provenance,
        "locked_output_root": str(output_root.resolve()),
        "global_registry_root": str(registry_root.resolve()),
        "fixed_budget": 3,
        "random_mc_trajectories": 32,
        "primary_aggregate": {
            "schema_version": verifier.AGGREGATE_SCHEMA,
            "utility_policy": verifier.PRIMARY_UTILITY_POLICY,
            "baselines": list(verifier.PRIMARY_BASELINES),
            "paired_bootstrap": config["primary_go_criterion"][
                "paired_bootstrap"
            ],
            "report_path": "docs/results/cvian_sequence_utility_v1.md",
        },
        "adaptive_results_secondary": True,
        "scoring_status_before_run": "not_loaded",
    }
    commitment["evaluation_fingerprint_sha256"] = verifier._canonical_hash(
        commitment
    )
    registry_dir = registry_root / _sha(protocol_commitment_path)
    commitment_path = registry_dir / "evaluation_commitment.json"
    _write_json(commitment_path, commitment, crlf=crlf_commitment)
    commitment_sha = _sha(commitment_path)
    started = {
        "schema_version": verifier.COMMITMENT_SCHEMA,
        "evaluation_fingerprint_sha256": commitment[
            "evaluation_fingerprint_sha256"
        ],
        "evaluation_commitment_sha256": commitment_sha,
        "prospective_test_manifest_sha256": role_hashes["prospective_test"],
        "locked_output_root": str(output_root.resolve()),
        "status": "prospective_test_scoring_started",
        "rerun_if_incomplete_permitted": False,
    }
    started_path = registry_dir / "evaluation_started.json"
    _write_json(started_path, started)
    started_sha = _sha(started_path)

    seed_completion_hashes: dict[str, str] = {}
    for seed in verifier.EXPECTED_SEEDS:
        seed_dir = output_root / f"seed{seed}"
        completion = {
            "schema_version": verifier.EVALUATION_SCHEMA,
            "seed": seed,
            "evaluation_fingerprint_sha256": commitment[
                "evaluation_fingerprint_sha256"
            ],
            "evaluation_commitment_sha256": commitment_sha,
            "evaluation_started_sha256": started_sha,
            "fit_completion_sha256": seed_inputs[str(seed)][
                "fit_completion_sha256"
            ],
            "prospective_test_cache_sha256": seed_inputs[str(seed)][
                "prospective_test_cache_sha256"
            ],
            "metrics_sha256": _sha(seed_dir / "metrics.csv"),
            "decisions_sha256": _sha(seed_dir / "per_sample_decisions.csv"),
        }
        completion_path = seed_dir / "evaluation_complete.json"
        _write_json(completion_path, completion)
        seed_completion_hashes[str(seed)] = _sha(completion_path)

    analysis, comparisons = verifier.recompute_primary_analysis(
        pd.concat(decision_frames, ignore_index=True), config
    )
    aggregate_json_path = output_root / "aggregate_primary_analysis.json"
    aggregate_csv_path = output_root / "aggregate_primary_comparisons.csv"
    _write_json(aggregate_json_path, analysis, sort_keys=True)
    comparisons.to_csv(aggregate_csv_path, index=False, lineterminator="\n")
    _write_bytes(
        report_path,
        verifier.render_primary_report(analysis, comparisons).encode("utf-8"),
    )
    aggregate_completion = {
        "schema_version": verifier.AGGREGATE_COMPLETION_SCHEMA,
        "evaluation_fingerprint_sha256": commitment[
            "evaluation_fingerprint_sha256"
        ],
        "evaluation_commitment_sha256": commitment_sha,
        "evaluation_started_sha256": started_sha,
        "seed_evaluation_completion_sha256": seed_completion_hashes,
        "aggregate_json_sha256": _sha(aggregate_json_path),
        "aggregate_csv_sha256": _sha(aggregate_csv_path),
        "report_sha256": _sha(report_path),
        "report_path": "docs/results/cvian_sequence_utility_v1.md",
        "primary_decision": analysis["decision"],
    }
    aggregate_completion_path = output_root / "aggregate_complete.json"
    _write_json(aggregate_completion_path, aggregate_completion)
    registry_completion = {
        "schema_version": verifier.REGISTRY_COMPLETION_SCHEMA,
        "evaluation_fingerprint_sha256": commitment[
            "evaluation_fingerprint_sha256"
        ],
        "evaluation_commitment_sha256": commitment_sha,
        "evaluation_started_sha256": started_sha,
        "locked_output_root": str(output_root.resolve()),
        "aggregate_completion_sha256": _sha(aggregate_completion_path),
        "primary_decision": analysis["decision"],
        "status": "prospective_test_evaluation_complete",
    }
    registry_completion_path = registry_dir / "evaluation_complete.json"
    _write_json(registry_completion_path, registry_completion)
    return {
        "repo_root": repo_root,
        "commitment": commitment_path,
        "output_root": output_root,
        "registry_completion": registry_completion_path,
        "aggregate_json": aggregate_json_path,
        "aggregate_completion": aggregate_completion_path,
    }


def test_complete_synthetic_dag_tolerates_only_crlf_commitment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _build_synthetic_completion(tmp_path / "repo", crlf_commitment=True)
    monkeypatch.setattr(
        np,
        "load",
        lambda *args, **kwargs: pytest.fail("verifier attempted NPZ deserialization"),
    )
    result = verifier.verify_completion(repo_root=paths["repo_root"])
    assert result["verified"] is True
    assert result["verification_defect"] == "windows_newline_hash_mismatch"
    assert result["claim_status"] == "consumed_development_exploratory"
    assert result["scoring_reexecuted"] is False
    assert result["model_inference_performed"] is False
    assert result["prospective_npz_semantically_loaded"] is False
    assert result["primary_recomputation"][
        "aggregate_json_semantically_identical"
    ] is True


def test_canonical_lf_commitment_has_no_serialization_defect(tmp_path: Path) -> None:
    paths = _build_synthetic_completion(tmp_path / "repo", crlf_commitment=False)
    result = verifier.verify_completion(repo_root=paths["repo_root"])
    assert result["verified"] is True
    assert result["verification_defect"] is None


def test_decision_hash_tampering_breaks_declared_output_edge(tmp_path: Path) -> None:
    paths = _build_synthetic_completion(tmp_path / "repo", crlf_commitment=True)
    decision_path = paths["output_root"] / "seed42/per_sample_decisions.csv"
    decision_path.write_bytes(decision_path.read_bytes() + b"tamper")
    with pytest.raises(verifier.VerificationError, match="decisions SHA-256"):
        verifier.verify_completion(repo_root=paths["repo_root"])


def test_semantic_commitment_tampering_breaks_internal_fingerprint(
    tmp_path: Path,
) -> None:
    paths = _build_synthetic_completion(tmp_path / "repo", crlf_commitment=True)
    commitment = json.loads(paths["commitment"].read_text(encoding="utf-8"))
    commitment["fixed_budget"] = 4
    _write_json(paths["commitment"], commitment, crlf=True)
    with pytest.raises(verifier.VerificationError, match="canonical commitment"):
        verifier.verify_completion(repo_root=paths["repo_root"])


def test_current_evaluator_source_drift_is_rejected(tmp_path: Path) -> None:
    paths = _build_synthetic_completion(tmp_path / "repo", crlf_commitment=True)
    runner_path = paths["repo_root"] / verifier.SOURCE_FILES["runner"]
    runner_path.write_bytes(runner_path.read_bytes() + b"drift")
    with pytest.raises(verifier.VerificationError, match="source runner"):
        verifier.verify_completion(repo_root=paths["repo_root"])


def test_rechained_false_aggregate_is_caught_by_independent_recomputation(
    tmp_path: Path,
) -> None:
    paths = _build_synthetic_completion(tmp_path / "repo", crlf_commitment=True)
    analysis = json.loads(paths["aggregate_json"].read_text(encoding="utf-8"))
    analysis["comparisons"]["farthest"]["overall"][
        "cost_difference_baseline_minus_utility"
    ] = 999.0
    _write_json(paths["aggregate_json"], analysis, sort_keys=True)

    aggregate_completion = json.loads(
        paths["aggregate_completion"].read_text(encoding="utf-8")
    )
    aggregate_completion["aggregate_json_sha256"] = _sha(paths["aggregate_json"])
    _write_json(paths["aggregate_completion"], aggregate_completion)
    registry = json.loads(
        paths["registry_completion"].read_text(encoding="utf-8")
    )
    registry["aggregate_completion_sha256"] = _sha(paths["aggregate_completion"])
    _write_json(paths["registry_completion"], registry)

    with pytest.raises(verifier.VerificationError, match="differs from recomputation"):
        verifier.verify_completion(repo_root=paths["repo_root"])
