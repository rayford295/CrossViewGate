from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from crossview_conflict.decision.active_view import (
    ActiveViewCache,
    ActiveViewMLP,
    Standardizer,
    build_selector_features,
    build_state_features,
    softmax,
)

from scripts.run_cvian_active_view_experiment import (
    EVALUATION_SCHEMA,
    NetworkResult,
    _predict_network,
    _select_online_action,
    _sha256,
    _validate_completed_artifact,
)
from scripts.summarize_cvian_active_view import (
    _sequence_connected_components,
    _validate_seed_tables,
)


def test_sequence_spatial_dependency_components_merge_linked_blocks() -> None:
    frame = pd.DataFrame(
        [
            {"spatial_block_id": "b0", "sequence_id": "link_01"},
            {"spatial_block_id": "b1", "sequence_id": "link_01"},
            {"spatial_block_id": "b1", "sequence_id": "link_12"},
            {"spatial_block_id": "b2", "sequence_id": "link_12"},
            {"spatial_block_id": "b3", "sequence_id": "only_3"},
            {"spatial_block_id": "b4", "sequence_id": "only_4"},
            {"spatial_block_id": "b5", "sequence_id": "only_5"},
        ]
    )
    components = _sequence_connected_components(frame)
    assert components["b0"] == components["b1"] == components["b2"]
    assert len(set(components.values())) == 4


def test_summary_rejects_duplicate_policy_budget_metrics() -> None:
    metrics = pd.DataFrame(
        [
            {"policy": "learned", "view_count": 3},
            {"policy": "learned", "view_count": 3},
        ]
    )
    with pytest.raises(ValueError, match="Duplicate metric key"):
        _validate_seed_tables(
            metrics, pd.DataFrame(), seed=42, view_cost=0.5
        )


def test_completion_reuse_fails_closed_on_artifact_mutation(tmp_path) -> None:
    files = {
        "metrics.csv": "metrics_sha256",
        "per_sample_decisions.csv": "decisions_sha256",
        "policy_artifact.pt": "artifact_sha256",
        "training_history.json": "training_history_sha256",
    }
    completion: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA,
        "run_fingerprint": "fingerprint",
    }
    for filename, field in files.items():
        path = tmp_path / filename
        path.write_bytes(filename.encode("utf-8"))
        completion[field] = _sha256(path)
    expected = {"run_fingerprint": "fingerprint"}
    _validate_completed_artifact(tmp_path, completion, expected)

    (tmp_path / "metrics.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_completed_artifact(tmp_path, completion, expected)


def test_completion_reuse_rejects_config_fingerprint_change(tmp_path) -> None:
    completion = {
        "schema_version": EVALUATION_SCHEMA,
        "run_fingerprint": "old",
    }
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _validate_completed_artifact(
            tmp_path, completion, {"run_fingerprint": "new"}
        )


def _network(input_dim: int, output_dim: int) -> NetworkResult:
    model = ActiveViewMLP(input_dim, output_dim, hidden_dim=8, dropout=0.0)
    model.eval()
    return NetworkResult(
        model=model,
        standardizer=Standardizer(
            mean=np.zeros(input_dim, dtype=np.float32),
            scale=np.ones(input_dim, dtype=np.float32),
        ),
        best_epoch=0,
        best_val_macro_f1=0.0,
        history=[],
    )


def _online_cache() -> ActiveViewCache:
    rng = np.random.default_rng(4)
    return ActiveViewCache(
        sample_id=np.asarray(["a", "b"]),
        spatial_block_id=np.asarray(["x", "y"]),
        sequence_id=np.asarray(["s1", "s2"]),
        target=np.asarray([0, 2]),
        latitude=np.asarray([1.0, 2.0]),
        longitude=np.asarray([3.0, 4.0]),
        sector_id=np.arange(4),
        relative_azimuth_deg=np.asarray([0.0, 90.0, -180.0, -90.0]),
        street_embedding=rng.normal(size=(2, 4, 3)).astype(np.float32),
        overhead_embedding=rng.normal(size=(2, 3)).astype(np.float32),
        sector_logits=rng.normal(size=(2, 4, 3)).astype(np.float32),
        panorama_logits=rng.normal(size=(2, 3)).astype(np.float32),
    )


def test_learned_online_action_is_fixed_before_hidden_oracle_diagnostics() -> None:
    torch.manual_seed(7)
    cache = _online_cache()
    mask = np.asarray([[True, True, False, False], [True, True, False, False]])
    last = np.asarray([1, 1])
    state = build_state_features(cache, mask, last_sector=last, remaining_budget=2)
    classifier = _network(state.shape[1], 3)
    logits = _predict_network(
        classifier.model, classifier.standardizer, state, device="cpu"
    )
    selector_features = build_selector_features(state, softmax(logits))
    selector = _network(selector_features.shape[1], cache.sector_count)
    before = _select_online_action(
        "learned",
        cache,
        mask,
        last,
        classifier,
        selector,
        temperature=1.0,
        seed=42,
        budget=2,
        device="cpu",
    )

    changed = _online_cache()
    changed.street_embedding[:, 2:] += 1_000_000.0
    changed.sector_logits[:, 2:] -= 1_000_000.0
    changed.target[:] = changed.target[::-1]
    after = _select_online_action(
        "learned",
        changed,
        mask,
        last,
        classifier,
        selector,
        temperature=1.0,
        seed=42,
        budget=2,
        device="cpu",
    )
    np.testing.assert_array_equal(before, after)
