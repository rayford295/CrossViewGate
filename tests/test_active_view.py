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
