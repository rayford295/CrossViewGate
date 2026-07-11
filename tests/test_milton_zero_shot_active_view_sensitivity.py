from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from crossview_conflict.decision.active_view import ActiveViewCache, Standardizer
from scripts import run_milton_zero_shot_active_view_sensitivity as scorer


class _Linear(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, offset: int) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(input_dim, output_dim)
        with torch.no_grad():
            values = torch.arange(output_dim * input_dim, dtype=torch.float32)
            self.layer.weight.copy_(
                ((values.reshape(output_dim, input_dim) + offset) % 17 - 8) / 100.0
            )
            self.layer.bias.copy_(torch.linspace(-0.1, 0.1, output_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layer(values)


def _cache(sample_count: int = 4) -> ActiveViewCache:
    rng = np.random.default_rng(91)
    sectors, embedding_dim = 8, 2
    return ActiveViewCache(
        sample_id=np.asarray([f"sample-{index:03d}" for index in range(sample_count)]),
        spatial_block_id=np.asarray([f"block-{index % 2}" for index in range(sample_count)]),
        sequence_id=np.asarray([f"missing-sequence-{index}" for index in range(sample_count)]),
        target=np.arange(sample_count, dtype=np.int64) % 3,
        latitude=np.linspace(27.0, 27.1, sample_count),
        longitude=np.linspace(-82.0, -81.9, sample_count),
        sector_id=np.arange(sectors, dtype=np.int64),
        relative_azimuth_deg=scorer.EXPECTED_AZIMUTH.copy(),
        street_embedding=rng.normal(size=(sample_count, sectors, embedding_dim)).astype(
            np.float32
        ),
        overhead_embedding=rng.normal(size=(sample_count, embedding_dim)).astype(
            np.float32
        ),
        sector_logits=rng.normal(size=(sample_count, sectors, 3)).astype(np.float32),
        panorama_logits=rng.normal(size=(sample_count, 3)).astype(np.float32),
    )


def _data(sample_count: int = 4) -> scorer.MiltonData:
    cache = _cache(sample_count)
    dependency = np.asarray([f"dependency-{index % 2}" for index in range(sample_count)])
    component = scorer._joint_dependency_components(dependency, cache.spatial_block_id)
    return scorer.MiltonData(
        cache=cache,
        dependency_group_id=dependency,
        joint_component_id=component,
        sorted_sample_ordinal=scorer._sorted_ordinals(cache.sample_id),
    )


def _models(cache: ActiveViewCache) -> scorer.FrozenModels:
    state_dim = cache.embedding_dim * 3 + 15
    absolute_dim = state_dim + 6 + 14
    relative_dim = 29
    return scorer.FrozenModels(
        classifier=_Linear(state_dim, 3, 1),
        classifier_standardizer=Standardizer(
            np.zeros(state_dim, dtype=np.float32), np.ones(state_dim, dtype=np.float32)
        ),
        absolute_regressor=_Linear(absolute_dim, 1, 3),
        absolute_standardizer=Standardizer(
            np.zeros(absolute_dim, dtype=np.float32),
            np.ones(absolute_dim, dtype=np.float32),
        ),
        relative_regressor=_Linear(relative_dim, 1, 5),
        relative_standardizer=Standardizer(
            np.zeros(relative_dim, dtype=np.float32),
            np.ones(relative_dim, dtype=np.float32),
        ),
        temperature=1.3,
    )


def _config() -> dict[str, object]:
    return json.loads(
        (scorer.REPO_ROOT / "configs" / "milton_zero_shot_active_view_sensitivity_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _slice_data(data: scorer.MiltonData, row: int) -> scorer.MiltonData:
    cache = data.cache
    one = replace(
        cache,
        sample_id=cache.sample_id[row : row + 1],
        spatial_block_id=cache.spatial_block_id[row : row + 1],
        sequence_id=cache.sequence_id[row : row + 1],
        target=cache.target[row : row + 1],
        latitude=cache.latitude[row : row + 1],
        longitude=cache.longitude[row : row + 1],
        street_embedding=cache.street_embedding[row : row + 1],
        overhead_embedding=cache.overhead_embedding[row : row + 1],
        sector_logits=cache.sector_logits[row : row + 1],
        panorama_logits=cache.panorama_logits[row : row + 1],
    )
    return scorer.MiltonData(
        cache=one,
        dependency_group_id=data.dependency_group_id[row : row + 1],
        joint_component_id=data.joint_component_id[row : row + 1],
        # Preserve the ordinal derived from the full sorted cohort.
        sorted_sample_ordinal=data.sorted_sample_ordinal[row : row + 1],
    )


def test_rotation_rolls_physical_origin_to_local_zero_and_maps_actions_back() -> None:
    cache = _cache(2)
    marker = np.broadcast_to(
        np.arange(8, dtype=np.float32)[None, :, None], cache.street_embedding.shape
    ).copy()
    marked = replace(cache, street_embedding=marker)
    rolled = scorer.roll_cache_to_local_origin(marked, 3)
    np.testing.assert_array_equal(rolled.street_embedding[:, 0], marker[:, 3])
    np.testing.assert_array_equal(rolled.street_embedding[:, 7], marker[:, 2])
    selected = np.asarray([[0, 1, 7], [0, 4, 2]])
    np.testing.assert_array_equal(
        scorer.local_to_physical(selected, 3), np.asarray([[3, 4, 2], [3, 7, 5]])
    )
    scorer._validate_selected_orders(selected, physical_origin=3, expected_budget=3)


def test_scores_use_rotation_relative_tie_break_for_unrolled_ood() -> None:
    mask = np.zeros((2, 8), dtype=bool)
    mask[:, 3] = True
    scores = np.ones((2, 8), dtype=float)
    actions = scorer._scores_to_actions(scores, mask, tie_origin=3)
    # local sector zero (physical 3) is revealed, so local sector one (physical 4) wins.
    np.testing.assert_array_equal(actions, np.asarray([4, 4]))


def test_hidden_candidate_mutation_does_not_change_learned_first_action() -> None:
    data = _data()
    environments = (
        ("relative_geometry_utility", scorer.roll_cache_to_local_origin(data.cache, 5), 0),
        ("absolute_aware_utility_ood_diagnostic", data.cache, 3),
    )
    for policy, environment, origin in environments:
        models = _models(environment)
        mask = np.zeros((environment.sample_count, 8), dtype=bool)
        mask[:, origin] = True
        before = scorer._learned_actions(
            policy,
            environment,
            mask,
            models,
            device="cpu",
            batch_size=32,
            tie_origin=origin,
        )
        street = environment.street_embedding.copy()
        logits = environment.sector_logits.copy()
        hidden = np.arange(8) != origin
        street[:, hidden] = 10000.0
        logits[:, hidden] = -10000.0
        mutated = replace(environment, street_embedding=street, sector_logits=logits)
        after = scorer._learned_actions(
            policy,
            mutated,
            mask,
            models,
            device="cpu",
            batch_size=32,
            tie_origin=origin,
        )
        np.testing.assert_array_equal(before, after)


def test_label_permutation_leaves_every_non_oracle_trajectory_unchanged() -> None:
    data = _data()
    models = _models(data.cache)
    permuted = replace(data, cache=replace(data.cache, target=data.cache.target[::-1].copy()))
    policies = [
        "relative_geometry_utility",
        "absolute_aware_utility_ood_diagnostic",
        "farthest",
        "clockwise",
        "random_mc32",
        "max_confidence_privileged",
    ]
    for policy in policies:
        original = scorer._evaluate_policy(
            policy=policy,
            seed=42,
            physical_origin=3,
            data=data,
            models=models,
            config=_config(),
            device="cpu",
            batch_size=64,
        )
        changed = scorer._evaluate_policy(
            policy=policy,
            seed=42,
            physical_origin=3,
            data=permuted,
            models=models,
            config=_config(),
            device="cpu",
            batch_size=64,
        )
        columns = [
            "sample_id",
            "trajectory",
            "local_revealed_order",
            "physical_revealed_order",
            "prediction",
        ]
        pd.testing.assert_frame_equal(original[columns], changed[columns])


def test_vectorized_trajectories_equal_per_sample_reference() -> None:
    data = _data(3)
    models = _models(data.cache)
    config = _config()
    for policy in scorer._policy_inventory(config):
        vectorized = scorer._evaluate_policy(
            policy=policy,
            seed=123,
            physical_origin=6,
            data=data,
            models=models,
            config=config,
            device="cpu",
            batch_size=64,
        ).sort_values(["sample_id", "trajectory"], kind="mergesort")
        scalar = pd.concat(
            [
                scorer._evaluate_policy(
                    policy=policy,
                    seed=123,
                    physical_origin=6,
                    data=_slice_data(data, row),
                    models=models,
                    config=config,
                    device="cpu",
                    batch_size=64,
                )
                for row in range(data.cache.sample_count)
            ],
            ignore_index=True,
        ).sort_values(["sample_id", "trajectory"], kind="mergesort")
        columns = [
            "sample_id",
            "trajectory",
            "local_revealed_order",
            "physical_revealed_order",
            "prediction",
            "operational_cost",
        ]
        pd.testing.assert_frame_equal(
            vectorized[columns].reset_index(drop=True),
            scalar[columns].reset_index(drop=True),
            check_exact=False,
            atol=1e-7,
            rtol=0.0,
        )


def test_random_rng_uses_full_sample_ordinal_and_is_reproducible() -> None:
    ordinal = np.asarray([9, 1, 17], dtype=np.int64)
    first = scorer._random_trajectory(
        ordinal, model_seed=42, physical_origin=7, trajectory=4, budget=3
    )
    second = scorer._random_trajectory(
        ordinal, model_seed=42, physical_origin=7, trajectory=4, budget=3
    )
    np.testing.assert_array_equal(first, second)
    assert (first[:, 0] == 0).all()
    assert all(len(np.unique(row)) == 3 for row in first)


def test_evaluator_has_no_fit_calibration_or_parameter_update_call_path() -> None:
    source = Path(scorer.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {"fit", "fit_transform", "backward", "step", "minimize", "minimize_scalar"}
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not (called & forbidden_calls)
    assert "torch.optim" not in source
    assert "--phase" not in source


def test_exclusive_sentinel_refuses_second_write(tmp_path: Path) -> None:
    path = tmp_path / "started.json"
    scorer._write_json_exclusive(path, {"status": "started"})
    with pytest.raises(FileExistsError):
        scorer._write_json_exclusive(path, {"status": "started-again"})


def test_cluster_bootstrap_keeps_unequal_unit_counts_inside_component() -> None:
    frame = pd.DataFrame(
        {
            "unit": ["a", "b", "c", "d"],
            "cluster": ["large", "large", "large", "small"],
            "difference": [1.0, 1.0, 1.0, -1.0],
        }
    )
    low, high = scorer._cluster_bootstrap_interval(
        frame,
        value_column="difference",
        cluster_column="cluster",
        resamples=1000,
        seed=42,
        confidence=0.95,
    )
    assert low <= -1.0
    assert high >= 1.0


def _synthetic_complete_decisions(seeds: tuple[int, ...]) -> pd.DataFrame:
    policies = scorer._policy_inventory(_config())
    policy_cost = {
        policy: 1.0 + 0.2 * index for index, policy in enumerate(policies)
    }
    policy_cost["relative_geometry_utility"] = 1.0
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for origin in range(8):
            for policy in policies:
                trajectories = 32 if policy == "random_mc32" else 1
                for sample in range(2):
                    for trajectory in range(trajectories):
                        rows.append(
                            {
                                "seed": seed,
                                "physical_origin": origin,
                                "policy": policy,
                                "policy_scope": "synthetic",
                                "sample_id": f"sample-{sample}",
                                "dependency_group_id": f"dependency-{sample}",
                                "spatial_block_id": f"block-{sample}",
                                "joint_component_id": f"component-{sample}",
                                "trajectory": trajectory,
                                "target": sample,
                                "prediction": sample,
                                "view_count": 3,
                                "additional_view_count": 2,
                                "operational_cost": policy_cost[policy]
                                + trajectory * 0.001,
                                "severe_miss": 0.0,
                                "correct": 1.0,
                            }
                        )
    return pd.DataFrame(rows)


def test_decision_lattice_fails_closed_on_missing_or_duplicate_key() -> None:
    frame = _synthetic_complete_decisions((42,))
    policies = scorer._policy_inventory(_config())
    scorer._validate_decision_lattice(
        frame,
        policies=policies,
        expected_samples=2,
        expected_origins=8,
        expected_random_trajectories=32,
        expected_raw_rows=608,
        expected_dependency_groups=2,
        expected_spatial_blocks=2,
        expected_joint_components=2,
    )
    corrupted = pd.concat([frame.iloc[:-1], frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate|incomplete"):
        scorer._validate_decision_lattice(
            corrupted,
            policies=policies,
            expected_samples=2,
            expected_origins=8,
            expected_random_trajectories=32,
            expected_raw_rows=608,
            expected_dependency_groups=2,
            expected_spatial_blocks=2,
            expected_joint_components=2,
        )


def test_end_to_end_estimand_order_contrasts_bootstrap_and_report() -> None:
    raw = _synthetic_complete_decisions(scorer.EXPECTED_SEEDS)
    one_seed = raw[raw["seed"] == scorer.EXPECTED_SEEDS[0]].copy()
    metrics, _ = scorer._metrics_for_seed(
        one_seed, expected_trajectory_rows=2 * 8 * 7
    )
    assert len(metrics) == 126
    trajectory = scorer._trajectory_average(raw)
    aggregate, contrasts = scorer._aggregate_sensitivity(
        trajectory,
        _config(),
        raw_decision_rows=len(raw),
        expected_raw_rows=2 * 5 * 8 * 38,
        expected_trajectory_rows=2 * 5 * 8 * 7,
        expected_origin_rows=2 * 5 * 7,
        expected_seed_rows=2 * 7,
    )
    assert len(contrasts) == 12
    assert aggregate["averaging_stage_rows"] == {
        "raw_decision_rows": 2 * 5 * 8 * 38,
        "post_trajectory_rows": 2 * 5 * 8 * 7,
        "post_origin_rows": 2 * 5 * 7,
        "post_seed_rows": 2 * 7,
    }
    farthest = contrasts[
        (contrasts["comparator"] == "farthest")
        & (contrasts["aggregation_unit"] == "dependency_group_id")
    ].iloc[0]
    assert farthest["cost_difference_comparator_minus_main"] > 0.0
    assert np.isfinite(farthest["primary_joint_component_bootstrap_95_ci_low"])
    assert np.isfinite(farthest["primary_joint_component_bootstrap_95_ci_high"])
    report = scorer._render_report(aggregate, contrasts)
    assert "sensitivity only" in report
    assert "no confirmatory claim" in report
    assert "no GO/NO-GO decision" in report
