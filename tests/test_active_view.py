from __future__ import annotations

import numpy as np
import pytest

from crossview_conflict.decision.active_view import (
    ActiveViewCache,
    DEFAULT_OPERATIONAL_COST,
    build_state_features,
    expected_operational_cost,
    farthest_available_sector,
    realized_operational_cost,
    reveal_sector,
    validate_active_view_cache,
)
from crossview_conflict.decision.active_view_utility import (
    bayes_operational_action_and_risk,
    build_candidate_geometry_features,
    build_candidate_utility_features,
    select_adaptive_actions,
)


def _cache() -> ActiveViewCache:
    return ActiveViewCache(
        sample_id=np.asarray(["a", "b"]),
        spatial_block_id=np.asarray(["x", "y"]),
        sequence_id=np.asarray(["s1", "s2"]),
        target=np.asarray([0, 2]),
        latitude=np.asarray([1.0, 2.0]),
        longitude=np.asarray([3.0, 4.0]),
        sector_id=np.arange(4),
        relative_azimuth_deg=np.asarray([0.0, 90.0, -180.0, -90.0]),
        street_embedding=np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3),
        overhead_embedding=np.ones((2, 3), dtype=np.float32),
        sector_logits=np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3),
        panorama_logits=np.ones((2, 3), dtype=np.float32),
    )


def test_state_features_do_not_change_when_hidden_sector_changes() -> None:
    cache = _cache()
    validate_active_view_cache(cache)
    mask = np.asarray([[True, False, False, False], [True, False, False, False]])
    before = build_state_features(cache, mask)
    changed = _cache()
    changed.street_embedding[:, 3] += 10_000.0
    changed.sector_logits[:, 3] -= 10_000.0
    after = build_state_features(changed, mask)
    np.testing.assert_allclose(before, after)


def test_set_aggregator_features_are_order_invariant() -> None:
    cache = _cache()
    mask = np.asarray([[True, True, False, False], [True, True, False, False]])
    first = build_state_features(cache, mask, last_sector=np.asarray([0, 0]))
    second = build_state_features(cache, mask, last_sector=np.asarray([1, 1]))
    np.testing.assert_allclose(first, second)


def test_reveal_fails_closed_on_duplicate_or_invalid_action() -> None:
    mask = np.zeros((1, 4), dtype=bool)
    reveal_sector(mask, 0, 0)
    with pytest.raises(ValueError, match="already"):
        reveal_sector(mask, 0, 0)
    with pytest.raises(ValueError, match="Invalid"):
        reveal_sector(mask, 0, 4)


def test_operational_cost_prioritizes_severe_misses() -> None:
    predictions = np.asarray([2, 0, 1])
    targets = np.asarray([0, 2, 2])
    np.testing.assert_array_equal(
        realized_operational_cost(predictions, targets),
        np.asarray([4.0, 8.0, 8.0]),
    )
    probabilities = np.eye(3)[predictions]
    np.testing.assert_array_equal(
        expected_operational_cost(probabilities, targets),
        realized_operational_cost(predictions, targets),
    )
    assert DEFAULT_OPERATIONAL_COST.max() == 8.0


def test_farthest_policy_uses_only_geometry_and_available_actions() -> None:
    assert farthest_available_sector([0], range(1, 8), sector_count=8) == 4
    assert farthest_available_sector([0, 4], [1, 2, 3, 5, 6, 7], sector_count=8) == 2


def test_candidate_utility_features_use_only_state_mask_and_geometry() -> None:
    cache = _cache()
    mask = np.asarray([[True, False, False, False], [True, False, False, False]])
    state = build_state_features(cache, mask, remaining_budget=3)
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    candidate = np.asarray([2, 3])
    features = build_candidate_utility_features(
        state,
        probabilities,
        mask,
        candidate,
        cache.relative_azimuth_deg,
        compass_angle_deg=np.asarray([350.0, 10.0]),
        compass_available=np.asarray([True, True]),
    )
    geometry = build_candidate_geometry_features(
        mask,
        candidate,
        cache.relative_azimuth_deg,
        compass_angle_deg=np.asarray([350.0, 10.0]),
        compass_available=np.asarray([True, True]),
    )
    assert features.shape[0] == 2
    assert geometry.shape == (2, 10)

    changed = _cache()
    changed.street_embedding[:, 1:] += 100_000.0
    changed.sector_logits[:, 1:] -= 100_000.0
    changed_state = build_state_features(changed, mask, remaining_budget=3)
    changed_features = build_candidate_utility_features(
        changed_state,
        probabilities,
        mask,
        candidate,
        changed.relative_azimuth_deg,
        compass_angle_deg=np.asarray([350.0, 10.0]),
        compass_available=np.asarray([True, True]),
    )
    np.testing.assert_allclose(features, changed_features)


def test_candidate_geometry_rejects_revealed_sector() -> None:
    mask = np.asarray([[True, False, False, False]])
    with pytest.raises(ValueError, match="unrevealed"):
        build_candidate_geometry_features(
            mask,
            np.asarray([0]),
            np.asarray([0.0, 90.0, -180.0, -90.0]),
            compass_angle_deg=np.asarray([0.0]),
            compass_available=np.asarray([True]),
        )


def test_candidate_distance_uses_angles_not_sector_indices() -> None:
    geometry = build_candidate_geometry_features(
        np.asarray([[True, False, False, False]]),
        np.asarray([1]),
        np.asarray([0.0, 10.0, 170.0, -90.0]),
        compass_angle_deg=np.asarray([350.0]),
        compass_available=np.asarray([True]),
    )
    assert geometry[0, -1] == pytest.approx(10.0 / 180.0)


def test_candidate_geometry_marks_missing_compass_without_fabricating_azimuth() -> None:
    geometry = build_candidate_geometry_features(
        np.asarray([[True, False, False, False]]),
        np.asarray([1]),
        np.asarray([0.0, 90.0, -180.0, -90.0]),
        compass_angle_deg=np.asarray([np.nan]),
        compass_available=np.asarray([False]),
    )
    np.testing.assert_array_equal(geometry[0, 6:9], np.zeros(3))


def test_adaptive_action_acquires_then_stops_or_defers() -> None:
    mask = np.asarray(
        [
            [True, False, False, False],
            [True, False, False, False],
            [True, False, False, False],
        ]
    )
    utilities = np.asarray(
        [
            [-99.0, -0.2, 0.3, 0.1],
            [-99.0, -0.2, -0.1, -0.3],
            [-99.0, -0.2, -0.1, -0.3],
        ]
    )
    probabilities = np.asarray(
        [
            [0.34, 0.33, 0.33],
            [0.95, 0.04, 0.01],
            [0.34, 0.33, 0.33],
        ]
    )
    decision = select_adaptive_actions(
        utilities,
        mask,
        probabilities,
        defer_cost=1.5,
        stop_risk_threshold=1.0,
        view_cost=0.5,
    )
    assert decision.action.tolist() == ["acquire", "stop", "defer"]
    assert decision.sector_id.tolist() == [2, -1, -1]
    assert decision.bayes_risk[2] > 1.5


def test_bayes_operational_risk_validates_probabilities() -> None:
    action, risk = bayes_operational_action_and_risk(
        np.asarray([[0.95, 0.04, 0.01]])
    )
    assert action.tolist() == [0]
    assert risk[0] == pytest.approx(0.12)
    with pytest.raises(ValueError, match="sum to one"):
        bayes_operational_action_and_risk(np.asarray([[0.5, 0.4, 0.0]]))


def test_bayes_action_can_differ_from_argmax_and_uncertified_stop_is_disabled() -> None:
    probabilities = np.asarray([[0.5, 0.1, 0.4]])
    action, _ = bayes_operational_action_and_risk(probabilities)
    assert probabilities.argmax(axis=1).tolist() == [0]
    assert action.tolist() == [2]

    decision = select_adaptive_actions(
        np.zeros((1, 4)),
        np.ones((1, 4), dtype=bool),
        np.asarray([[0.95, 0.04, 0.01]]),
        defer_cost=1.5,
        stop_risk_threshold=None,
        view_cost=0.5,
    )
    assert decision.action.tolist() == ["defer"]


def test_adaptive_controls_reject_nan() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        select_adaptive_actions(
            np.zeros((1, 2)),
            np.asarray([[True, False]]),
            np.asarray([[0.9, 0.05, 0.05]]),
            defer_cost=1.5,
            stop_risk_threshold=float("nan"),
            view_cost=0.5,
        )
