from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from crossview_conflict.decision.active_view import ActiveViewCache
from scripts import run_cvian_sequence_utility_experiment as runner


def _cache(sample_count: int = 2) -> ActiveViewCache:
    rng = np.random.default_rng(17)
    sectors = 8
    embedding_dim = 4
    return ActiveViewCache(
        sample_id=np.asarray([f"sample-{index}" for index in range(sample_count)]),
        spatial_block_id=np.asarray([f"block-{index}" for index in range(sample_count)]),
        sequence_id=np.asarray([f"sequence-{index}" for index in range(sample_count)]),
        target=np.arange(sample_count, dtype=np.int64) % 3,
        latitude=np.linspace(26.0, 27.0, sample_count),
        longitude=np.linspace(-82.0, -81.0, sample_count),
        sector_id=np.arange(sectors),
        relative_azimuth_deg=np.arange(sectors, dtype=float) * 45.0,
        street_embedding=rng.normal(
            size=(sample_count, sectors, embedding_dim)
        ).astype(np.float32),
        overhead_embedding=rng.normal(size=(sample_count, embedding_dim)).astype(
            np.float32
        ),
        sector_logits=rng.normal(size=(sample_count, sectors, 3)).astype(np.float32),
        panorama_logits=rng.normal(size=(sample_count, 3)).astype(np.float32),
    )


def test_all_unique_initial_sector_subsets_through_k4() -> None:
    states = runner.enumerate_subset_states(
        _cache(), maximum_revealed=4, maximum_adaptive_views=4
    )
    # C(7,0)+C(7,1)+C(7,2)+C(7,3) = 64 states per sample.
    assert len(states.sample_indices) == 2 * 64
    assert states.revealed_mask[:, 0].all()
    assert np.array_equal(states.revealed_mask.sum(axis=1), states.revealed_count)
    assert len({tuple(row) for row in states.revealed_mask[:64]}) == 64
    assert np.array_equal(
        states.remaining_acquisitions,
        np.maximum(4 - states.revealed_count, 0),
    )


def test_candidate_weights_equalize_each_sample_and_k_cell() -> None:
    sample = np.asarray([0, 0, 0, 0, 0, 1, 1, 1])
    revealed = np.asarray([1, 1, 2, 2, 2, 1, 2, 2])
    weights = runner._balanced_candidate_weights(sample, revealed)
    frame = pd.DataFrame(
        {"sample": sample, "revealed": revealed, "weight": weights}
    )
    totals = frame.groupby(["sample", "revealed"])["weight"].sum().to_numpy()
    np.testing.assert_allclose(totals, np.full(len(totals), totals[0]))
    np.testing.assert_allclose(weights.mean(), 1.0)


def test_sequence_macro_regret_uses_top_prediction_and_canonical_ties() -> None:
    table = runner.UtilityTable(
        features=np.zeros((6, 1), dtype=np.float32),
        targets=np.asarray([1.0, 0.0, 0.2, 0.7, 0.1, 0.4], dtype=np.float32),
        weights=np.ones(6, dtype=np.float32),
        state_id=np.asarray([0, 0, 1, 1, 2, 2]),
        sample_index=np.asarray([0, 0, 1, 1, 2, 2]),
        sequence_id=np.asarray(["a", "a", "a", "a", "b", "b"]),
        revealed_count=np.asarray([1, 1, 2, 2, 1, 1]),
        candidate_sector=np.asarray([2, 1, 1, 3, 4, 2]),
    )
    # State 0 is a prediction tie, so canonical sector 1 is selected (regret 1).
    # State 1 selects its true best (regret 0); sequence a mean is 0.5.
    # State 2 selects sector 4 (regret 0.3); sequence b mean is 0.3.
    score, diagnostics = runner._sequence_macro_top1_regret(
        np.asarray([0.5, 0.5, 0.1, 0.9, 0.9, 0.1]), table
    )
    assert score == pytest.approx(np.mean([0.5, 0.3]))
    assert diagnostics.loc[diagnostics["state_id"] == 0, "selected_sector"].item() == 1


def test_fit_role_loader_has_no_prospective_test_path(monkeypatch) -> None:
    opened: list[str] = []

    def fake_load(path: Path, manifest: pd.DataFrame) -> object:
        opened.append(path.name)
        return object()

    monkeypatch.setattr(runner, "_load_role_data", fake_load)
    manifests = {role: pd.DataFrame() for role in runner.FIT_ROLES}
    loaded = runner._load_fit_roles(Path("cache"), 42, manifests)
    assert set(loaded) == set(runner.FIT_ROLES)
    assert opened == ["base_fit.npz", "selector_fit.npz", "validation.npz"]
    assert "prospective_test.npz" not in opened


def _registered_config() -> dict[str, object]:
    return json.loads(
        (runner.REPO_ROOT / "configs" / "cvian_sequence_utility_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _primary_decisions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    samples = (
        ("sample-a", "component-a", 0),
        ("sample-b", "component-b", 1),
        ("sample-c", "component-b", 2),
    )
    policy_cost = {
        "utility_regression": (1.0, 1.0, 1.0),
        "farthest": (2.0, 1.5, 1.5),
        "max_building_privileged": (1.5, 1.25, 1.25),
    }
    for seed in runner.EXPECTED_SEEDS:
        for policy, costs in policy_cost.items():
            for index, ((sample_id, component, target), cost) in enumerate(
                zip(samples, costs)
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


def test_primary_aggregate_applies_locked_all_seed_go_rule() -> None:
    analysis, comparison_rows = runner._aggregate_primary_analysis(
        _primary_decisions(), _registered_config()
    )
    assert analysis["decision"] == "GO"
    assert analysis["go"] is True
    assert analysis["dependency_components"] == 2
    assert len(comparison_rows[comparison_rows["row_type"] == "per_seed"]) == 10
    for baseline in runner.PRIMARY_BASELINES:
        overall = analysis["comparisons"][baseline]["overall"]
        assert overall["all_five_seeds_lower_cost"] is True
        assert overall["bootstrap_95_percentile_ci_lower"] > 0.0
        assert overall["severe_miss_no_worse"] is True
        assert overall["comparison_go"] is True
    report = runner._render_primary_report(analysis, comparison_rows)
    assert "**Primary decision: GO.**" in report
    assert "10,000 dependency-component resamples" in report


def test_primary_aggregate_is_no_go_if_one_seed_loses() -> None:
    decisions = _primary_decisions()
    losing = (
        (decisions["seed"] == runner.EXPECTED_SEEDS[-1])
        & (decisions["policy"] == "max_building_privileged")
    )
    decisions.loc[losing, "operational_cost"] = 0.0
    analysis, _ = runner._aggregate_primary_analysis(
        decisions, _registered_config()
    )
    building = analysis["comparisons"]["max_building_privileged"]["overall"]
    assert building["all_five_seeds_lower_cost"] is False
    assert building["comparison_go"] is False
    assert analysis["decision"] == "NO-GO"


def test_primary_aggregate_is_no_go_if_severe_miss_is_worse() -> None:
    decisions = _primary_decisions()
    severe = (
        (decisions["policy"] == "utility_regression")
        & (decisions["target"] == 2)
    )
    decisions.loc[severe, "severe_miss"] = 1.0
    analysis, _ = runner._aggregate_primary_analysis(
        decisions, _registered_config()
    )
    assert all(
        not analysis["comparisons"][baseline]["overall"]["severe_miss_no_worse"]
        for baseline in runner.PRIMARY_BASELINES
    )
    assert analysis["decision"] == "NO-GO"


def test_random_metrics_average_trajectories_within_sample_first() -> None:
    decisions = pd.DataFrame(
        [
            {
                "seed": 42,
                "policy": "random_mc32",
                "scope": "fixed_k3_primary",
                "sample_id": "a",
                "sequence_id": "sa",
                "spatial_block_id": "ba",
                "dependency_component": "component-a",
                "trajectory": 0,
                "target": 0,
                "prediction": 0,
                "terminal_action": "stop",
                "operational_cost": 0.0,
                "severe_miss": 0.0,
                "view_count": 3,
            },
            {
                "seed": 42,
                "policy": "random_mc32",
                "scope": "fixed_k3_primary",
                "sample_id": "a",
                "sequence_id": "sa",
                "spatial_block_id": "ba",
                "dependency_component": "component-a",
                "trajectory": 1,
                "target": 0,
                "prediction": 1,
                "terminal_action": "stop",
                "operational_cost": 2.0,
                "severe_miss": 0.0,
                "view_count": 3,
            },
            {
                "seed": 42,
                "policy": "random_mc32",
                "scope": "fixed_k3_primary",
                "sample_id": "b",
                "sequence_id": "sb",
                "spatial_block_id": "bb",
                "dependency_component": "component-a",
                "trajectory": 0,
                "target": 1,
                "prediction": 0,
                "terminal_action": "stop",
                "operational_cost": 4.0,
                "severe_miss": 0.0,
                "view_count": 3,
            },
        ]
    )
    metric = runner._metrics(decisions).iloc[0]
    # Sample a averages to 1.0, then it and sample b receive equal weight.
    assert metric["dependency_component_macro_operational_cost"] == pytest.approx(2.5)
    assert metric["mean_operational_cost"] == pytest.approx(2.5)
    assert metric["estimand_sample_rows"] == 2
    assert metric["trajectories_per_sample_min"] == 1
    assert metric["trajectories_per_sample_max"] == 2


def test_config_rejects_bootstrap_mutation() -> None:
    config = _registered_config()
    config["primary_go_criterion"]["paired_bootstrap"]["resamples"] = 9999
    with pytest.raises(ValueError, match="bootstrap contract"):
        runner._validate_config(config)


def _locked_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "classifier_epochs": 40,
        "utility_epochs": 60,
        "patience": 7,
        "batch_size": 512,
        "learning_rate": 0.001,
        "num_workers": 0,
        "device": "cuda",
        "output_root": "outputs/cvian_sequence_active_v2/utility_experiment",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_execution_lock_rejects_alternate_output_root_and_training_args() -> None:
    config = _registered_config()
    locked_output, registry_root = runner._validate_args_against_config(
        _locked_args(), config
    )
    assert locked_output == (
        runner.REPO_ROOT / "outputs/cvian_sequence_active_v2/utility_experiment"
    ).resolve()
    assert registry_root == (
        runner.REPO_ROOT / "outputs/cvian_sequence_active_v2/evaluation_registry"
    ).resolve()
    with pytest.raises(ValueError, match="one-shot execution root"):
        runner._validate_args_against_config(
            _locked_args(output_root="outputs/another-run"), config
        )
    with pytest.raises(ValueError, match="training hyperparameters"):
        runner._validate_args_against_config(
            _locked_args(utility_epochs=61), config
        )


def test_adaptive_contract_is_explicitly_not_certified() -> None:
    config = _registered_config()
    runner._validate_config(config)
    adaptive = config["adaptive_policy"]
    assert adaptive["formal_stop_calibration_performed"] is False
    assert adaptive["formal_guarantee"] is False
    assert adaptive["fail_closed_stop_threshold"] is None
    assert "not risk-controlled" in adaptive["claim_status"]
    assert not any("certified" in key for key in adaptive)


def test_completed_evaluation_verifies_aggregate_hashes_without_decisions(
    tmp_path: Path, monkeypatch
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    report_path = tmp_path / "docs" / "result.md"
    report_path.parent.mkdir()
    aggregate_json = output_root / runner.AGGREGATE_JSON_NAME
    aggregate_csv = output_root / runner.AGGREGATE_CSV_NAME
    aggregate_json.write_text(
        json.dumps(
            {"schema_version": runner.AGGREGATE_SCHEMA, "decision": "GO"}
        )
        + "\n",
        encoding="utf-8",
    )
    aggregate_csv.write_text("baseline,result\nfarthest,GO\n", encoding="utf-8")
    report_path.write_text("# Locked result\n", encoding="utf-8")
    monkeypatch.setattr(runner, "PRIMARY_REPORT_PATH", report_path)
    seed_hashes = {str(seed): f"hash-{seed}" for seed in runner.EXPECTED_SEEDS}
    completion = {
        "schema_version": runner.AGGREGATE_COMPLETION_SCHEMA,
        "evaluation_fingerprint_sha256": "fingerprint",
        "evaluation_commitment_sha256": "commitment",
        "evaluation_started_sha256": "started",
        "seed_evaluation_completion_sha256": seed_hashes,
        "aggregate_json_sha256": runner._sha256(aggregate_json),
        "aggregate_csv_sha256": runner._sha256(aggregate_csv),
        "report_sha256": runner._sha256(report_path),
        "primary_decision": "GO",
    }
    runner._validate_aggregate_completion(
        output_root,
        completion,
        evaluation_fingerprint="fingerprint",
        commitment_sha256="commitment",
        started_sha256="started",
        seed_completion_sha256=seed_hashes,
    )
    report_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        runner._validate_aggregate_completion(
            output_root,
            completion,
            evaluation_fingerprint="fingerprint",
            commitment_sha256="commitment",
            started_sha256="started",
            seed_completion_sha256=seed_hashes,
        )


def test_global_registry_completion_is_bound_to_locked_output(tmp_path: Path) -> None:
    output_root = tmp_path / "locked-output"
    aggregate_completion = output_root / runner.AGGREGATE_COMPLETION_NAME
    output_root.mkdir()
    aggregate_completion.write_text("{}\n", encoding="utf-8")
    registry_path = tmp_path / "registry-complete.json"
    payload = {
        "schema_version": runner.REGISTRY_COMPLETION_SCHEMA,
        "evaluation_fingerprint_sha256": "fingerprint",
        "evaluation_commitment_sha256": "commitment",
        "evaluation_started_sha256": "started",
        "locked_output_root": str(output_root),
        "aggregate_completion_sha256": runner._sha256(aggregate_completion),
        "status": "prospective_test_evaluation_complete",
    }
    registry_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    runner._validate_registry_completion(
        registry_path,
        evaluation_fingerprint="fingerprint",
        commitment_sha256="commitment",
        started_sha256="started",
        output_root=output_root,
        aggregate_completion_sha256=runner._sha256(aggregate_completion),
    )
    with pytest.raises(ValueError, match="locked_output_root"):
        runner._validate_registry_completion(
            registry_path,
            evaluation_fingerprint="fingerprint",
            commitment_sha256="commitment",
            started_sha256="started",
            output_root=tmp_path / "alternate-output",
            aggregate_completion_sha256=runner._sha256(aggregate_completion),
        )
