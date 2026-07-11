from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts import verify_milton_zero_shot_active_view_sensitivity as verifier


def _config() -> dict[str, Any]:
    return verifier._load_json(
        verifier.REPO_ROOT / "configs/milton_zero_shot_active_view_sensitivity_v1.json"
    )


def _scope(policy: str) -> str:
    if policy == "absolute_aware_utility_ood_diagnostic":
        return "unrolled_classifier_off_support_ood_diagnostic"
    if policy in {"max_confidence_privileged", "greedy_label_oracle_privileged"}:
        return "rolled_local_privileged_diagnostic"
    return "rolled_local_main_or_comparator"


def _random_order(seed: int, origin: int, trajectory: int, ordinal: int) -> list[int]:
    generator = np.random.default_rng(
        np.random.SeedSequence([seed, origin, trajectory, ordinal, 20260710])
    )
    available = list(range(1, 8))
    selected = [0]
    for _ in range(2):
        selected.append(available.pop(int(generator.integers(0, len(available)))))
    return selected


def _small_manifest() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": ["sample-b", "sample-a"],
            "label": [1, 0],
            "dependency_group_id": ["dependency-b", "dependency-a"],
            "spatial_block_id": ["block-b", "block-a"],
            "joint_component_id": ["joint_component_01", "joint_component_00"],
        }
    )
    order = np.argsort(frame["sample_id"].to_numpy(), kind="mergesort")
    ordinal = np.empty(len(frame), dtype=np.int64)
    ordinal[order] = np.arange(len(frame))
    frame["sorted_sample_ordinal"] = ordinal
    return frame


def _small_decisions(seed: int = 42) -> pd.DataFrame:
    config = _config()
    manifest = _small_manifest()
    policies = verifier._policy_inventory(config)
    rows: list[dict[str, Any]] = []
    for origin in verifier.EXPECTED_ORIGINS:
        for policy in policies:
            trajectory_ids = range(32) if policy == "random_mc32" else range(1)
            for trajectory in trajectory_ids:
                for sample in manifest.itertuples(index=False):
                    if policy == "random_mc32":
                        local = _random_order(
                            seed,
                            origin,
                            trajectory,
                            int(sample.sorted_sample_ordinal),
                        )
                    elif policy == "farthest":
                        local = [0, 4, 2]
                    else:
                        local = [0, 1, 2]
                    physical = [(value + origin) % 8 for value in local]
                    prediction = 0
                    classification_cost = float(
                        config["cost_matrix"][int(sample.label)][prediction]
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "physical_origin": origin,
                            "policy": policy,
                            "policy_scope": _scope(policy),
                            "sample_id": sample.sample_id,
                            "dependency_group_id": sample.dependency_group_id,
                            "spatial_block_id": sample.spatial_block_id,
                            "joint_component_id": sample.joint_component_id,
                            "trajectory": trajectory,
                            "target": int(sample.label),
                            "prediction": prediction,
                            "local_revealed_order": ",".join(map(str, local)),
                            "physical_revealed_order": ",".join(map(str, physical)),
                            "view_count": 3,
                            "additional_view_count": 2,
                            "p0": 1.0,
                            "p1": 0.0,
                            "p2": 0.0,
                            "model_bayes_risk": 0.0,
                            "classification_cost": classification_cost,
                            "view_cost": 1.0,
                            "operational_cost": classification_cost + 1.0,
                            "severe_miss": 0.0,
                            "correct": float(sample.label == prediction),
                        }
                    )
    return pd.DataFrame(rows, columns=verifier.DECISION_COLUMNS)


def test_verifier_has_no_scorer_or_model_runtime_import_and_no_npz_load() -> None:
    source_path = verifier.REPO_ROOT / "scripts/verify_milton_zero_shot_active_view_sensitivity.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "torch" not in imported
    assert not any(name.endswith("run_milton_zero_shot_active_view_sensitivity") for name in imported)
    assert "np.load(" not in source


def test_decision_lattice_actions_rng_probabilities_and_costs_reproduce() -> None:
    decisions = _small_decisions()
    result = verifier._validate_decision_semantics(
        decisions,
        seed=42,
        manifest=_small_manifest(),
        config=_config(),
        expected_raw_rows=len(decisions),
        replay_random=True,
    )
    assert result["random_seed_sequences_replayed"] == 2 * 8 * 32
    assert result["policies"] == 7


def test_random_seedsequence_mutation_is_rejected() -> None:
    decisions = _small_decisions()
    index = decisions.index[decisions["policy"].eq("random_mc32")][0]
    original = [int(value) for value in decisions.loc[index, "local_revealed_order"].split(",")]
    mutated = [original[0], original[2], original[1]]
    decisions.loc[index, "local_revealed_order"] = ",".join(map(str, mutated))
    origin = int(decisions.loc[index, "physical_origin"])
    decisions.loc[index, "physical_revealed_order"] = ",".join(
        map(str, [(value + origin) % 8 for value in mutated])
    )
    with pytest.raises(verifier.VerificationError, match="Random SeedSequence"):
        verifier._validate_decision_semantics(
            decisions,
            seed=42,
            manifest=_small_manifest(),
            config=_config(),
            expected_raw_rows=len(decisions),
            replay_random=True,
        )


def test_bayes_cost_mutation_is_rejected() -> None:
    decisions = _small_decisions()
    decisions.loc[0, "operational_cost"] += 0.25
    with pytest.raises(verifier.VerificationError, match="operational_cost"):
        verifier._validate_decision_semantics(
            decisions,
            seed=42,
            manifest=_small_manifest(),
            config=_config(),
            expected_raw_rows=len(decisions),
            replay_random=False,
        )


def test_joint_graph_components_join_crossed_group_and_block_edges() -> None:
    dependency = np.asarray(["g1", "g1", "g2", "g3", "g4"])
    block = np.asarray(["b1", "b2", "b2", "b3", "b4"])
    components = verifier.joint_dependency_components(dependency, block)
    assert components.tolist() == [
        "joint_component_00",
        "joint_component_00",
        "joint_component_00",
        "joint_component_01",
        "joint_component_02",
    ]


def test_bootstrap_uses_exact_default_rng_cluster_sums_and_counts() -> None:
    frame = pd.DataFrame(
        {
            "cluster": ["c0", "c0", "c1", "c2", "c2", "c2"],
            "value": [1.0, 3.0, -2.0, 2.0, 4.0, 8.0],
        }
    )
    observed = verifier.joint_cluster_bootstrap_interval(
        frame,
        value_column="value",
        cluster_column="cluster",
        resamples=1000,
        seed=42,
        confidence=0.95,
    )
    grouped = frame.groupby("cluster", sort=True)["value"].agg(["sum", "count"])
    generator = np.random.default_rng(42)
    index = generator.integers(0, 3, size=(1000, 3))
    estimates = (
        grouped["sum"].to_numpy()[index].sum(axis=1)
        / grouped["count"].to_numpy()[index].sum(axis=1)
    )
    expected = tuple(np.quantile(estimates, [0.025, 0.975], method="linear"))
    assert observed == pytest.approx(expected, rel=0.0, abs=0.0)


def _small_trajectory_rows() -> pd.DataFrame:
    config = _config()
    policies = verifier._policy_inventory(config)
    samples = (
        ("s0", "g0", "b0", "joint_component_00", 0),
        ("s1", "g1", "b1", "joint_component_00", 1),
        ("s2", "g2", "b2", "joint_component_01", 2),
        ("s3", "g3", "b3", "joint_component_01", 0),
    )
    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(verifier.EXPECTED_SEEDS):
        for origin in verifier.EXPECTED_ORIGINS:
            for policy_index, policy in enumerate(policies):
                for sample_id, group, block, component, target in samples:
                    cost = 1.0 + policy_index * 0.1 + seed_index * 0.01 + origin * 0.001
                    rows.append(
                        {
                            "seed": seed,
                            "physical_origin": origin,
                            "policy": policy,
                            "policy_scope": _scope(policy),
                            "sample_id": sample_id,
                            "dependency_group_id": group,
                            "spatial_block_id": block,
                            "joint_component_id": component,
                            "target": target,
                            "operational_cost": cost,
                            "severe_miss": float(target == 2 and policy_index % 2),
                            "correct": float((target + policy_index) % 3 == 0),
                            "trajectory_count": 32 if policy == "random_mc32" else 1,
                        }
                    )
    return pd.DataFrame(rows)


def test_aggregate_preserves_trajectory_origin_seed_order_and_both_estimands() -> None:
    config = _config()
    config = {**config, "evaluation": {**config["evaluation"]}}
    config["evaluation"]["bootstrap"] = {
        **config["evaluation"]["bootstrap"],
        "resamples": 200,
    }
    trajectory = _small_trajectory_rows()
    aggregate, contrasts = verifier.recompute_aggregate(
        trajectory,
        config,
        raw_decision_rows=1234,
        enforce_registered_counts=False,
    )
    assert aggregate["averaging_stage_rows"] == {
        "raw_decision_rows": 1234,
        "post_trajectory_rows": 1120,
        "post_origin_rows": 140,
        "post_seed_rows": 28,
    }
    assert len(contrasts) == 12
    assert set(contrasts["aggregation_unit"]) == {
        "dependency_group_id",
        "spatial_block_id",
    }
    # Every comparator was constructed exactly 0.1 * policy-index costlier.
    farthest = contrasts[contrasts["comparator"].eq("farthest")]
    assert np.allclose(
        farthest["cost_difference_comparator_minus_main"], 0.1, rtol=0.0, atol=1e-12
    )
    assert aggregate["bootstrap"]["seeds_and_origins_resampled"] is False


def test_opaque_hash_inventory_detects_tampering_without_deserializing(tmp_path: Path) -> None:
    artifact = tmp_path / "opaque-cache.npz"
    artifact.write_bytes(b"opaque bytes only")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    inventory = {"cache": {"path": str(artifact), "sha256": digest}}
    verified = verifier._verify_inventory_paths(inventory, repo_root=tmp_path)
    assert verified[str(artifact.resolve())] == digest
    artifact.write_bytes(b"tampered bytes")
    with pytest.raises(verifier.VerificationError, match="SHA-256 mismatch"):
        verifier._verify_inventory_paths(inventory, repo_root=tmp_path)
