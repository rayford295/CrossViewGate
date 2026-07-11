from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from crossview_conflict.decision.active_view import ActiveViewCache
from crossview_conflict.decision.relative_geometry_utility import (
    RELATIVE_GEOMETRY_FEATURE_NAMES,
    REGISTERED_RELATIVE_AZIMUTH_DEG,
    build_relative_geometry_utility_features,
)
from scripts import fit_cvian_relative_geometry_utility as fit


def _cache(sample_count: int = 2) -> ActiveViewCache:
    sectors = 8
    embedding_dim = 3
    street = np.empty((sample_count, sectors, embedding_dim), dtype=np.float32)
    logits = np.empty((sample_count, sectors, 3), dtype=np.float32)
    for sample in range(sample_count):
        for sector in range(sectors):
            street[sample, sector] = 100 * sample + sector
            logits[sample, sector] = [sector, sample, sector + sample]
    return ActiveViewCache(
        sample_id=np.asarray([f"sample-{index}" for index in range(sample_count)]),
        spatial_block_id=np.asarray([f"block-{index}" for index in range(sample_count)]),
        sequence_id=np.asarray([f"sequence-{index}" for index in range(sample_count)]),
        target=np.arange(sample_count, dtype=np.int64) % 3,
        latitude=np.linspace(26.0, 27.0, sample_count),
        longitude=np.linspace(-82.0, -81.0, sample_count),
        sector_id=np.arange(sectors, dtype=np.int64),
        relative_azimuth_deg=REGISTERED_RELATIVE_AZIMUTH_DEG.copy(),
        street_embedding=street,
        overhead_embedding=np.ones((sample_count, embedding_dim), dtype=np.float32),
        sector_logits=logits,
        panorama_logits=np.zeros((sample_count, 3), dtype=np.float32),
    )


def test_roll_maps_physical_origin_to_local_zero_and_keeps_local_zero_masks() -> None:
    cache = _cache()
    for origin in range(8):
        rolled = fit.roll_cache_to_local_origin(cache, origin)
        np.testing.assert_array_equal(
            rolled.street_embedding[:, 0], cache.street_embedding[:, origin]
        )
        np.testing.assert_array_equal(
            rolled.sector_logits[:, 0], cache.sector_logits[:, origin]
        )
        for local in range(8):
            physical = (local + origin) % 8
            np.testing.assert_array_equal(
                rolled.street_embedding[:, local],
                cache.street_embedding[:, physical],
            )
        states = fit.frozen_runner.enumerate_subset_states(
            rolled,
            maximum_revealed=3,
            maximum_adaptive_views=4,
            initial_sector_id=0,
        )
        assert states.revealed_mask[:, 0].all()


def test_relative_feature_interface_is_29d_and_rotation_local() -> None:
    probabilities = np.asarray([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    masks = np.asarray(
        [
            [True, False, True, False, False, False, False, False],
            [True, True, False, True, False, False, False, False],
        ]
    )
    candidates = np.asarray([3, 6])
    remaining = np.asarray([2.0, 1.0])
    features = build_relative_geometry_utility_features(
        probabilities,
        masks,
        candidates,
        REGISTERED_RELATIVE_AZIMUTH_DEG,
        origin_sector=0,
        remaining_acquisitions=remaining,
    )
    assert features.shape == (2, 29)
    assert features.shape[1] == len(RELATIVE_GEOMETRY_FEATURE_NAMES)
    # The registered normalized remaining-acquisition slot is (4-k)/8.
    np.testing.assert_allclose(features[:, 17], remaining / 8.0)


@pytest.mark.parametrize(
    ("azimuth", "origin", "remaining"),
    [
        (REGISTERED_RELATIVE_AZIMUTH_DEG + 1e-5, 0, np.asarray([3.0])),
        (REGISTERED_RELATIVE_AZIMUTH_DEG, 1, np.asarray([3.0])),
        (REGISTERED_RELATIVE_AZIMUTH_DEG, 0, np.asarray([2.0])),
    ],
)
def test_relative_feature_interface_fails_closed(
    azimuth: np.ndarray, origin: int, remaining: np.ndarray
) -> None:
    mask = np.asarray([[True, False, False, False, False, False, False, False]])
    if origin == 1:
        mask[0, 1] = True
    with pytest.raises(ValueError):
        build_relative_geometry_utility_features(
            np.asarray([[0.2, 0.3, 0.5]]),
            mask,
            np.asarray([2]),
            azimuth,
            origin_sector=origin,
            remaining_acquisitions=remaining,
        )


def test_fit_role_loader_opens_only_selector_and_validation(monkeypatch) -> None:
    opened: list[str] = []
    cache = _cache(1)

    def fake_load(path: Path, manifest: pd.DataFrame) -> object:
        opened.append(path.name)
        return SimpleNamespace(cache=cache)

    monkeypatch.setattr(fit.frozen_runner, "_load_role_data", fake_load)
    manifests = {role: pd.DataFrame() for role in fit.FIT_ROLES}
    roles = fit._load_fit_roles(Path("cache"), 42, manifests)
    assert list(roles) == ["selector_fit", "validation"]
    assert opened == ["selector_fit.npz", "validation.npz"]
    assert all(np.isnan(role.compass_angle_deg).all() for role in roles.values())
    assert all(not role.compass_available.any() for role in roles.values())


def test_manifest_reader_preserves_zero_padded_sample_ids(tmp_path: Path) -> None:
    header = (
        "sample_id,sequence_id,spatial_block_id,label,latitude,longitude,"
        "compass_angle_deg,ignored\n"
    )
    (tmp_path / "selector_fit.csv").write_text(
        header + "000003,seq-1,block-1,0,26.0,-82.0,45.0,x\n",
        encoding="utf-8",
    )
    (tmp_path / "validation.csv").write_text(
        header + "000008,seq-2,block-2,1,27.0,-81.0,90.0,y\n",
        encoding="utf-8",
    )
    manifests = fit._read_fit_manifests(tmp_path)
    assert manifests["selector_fit"]["sample_id"].item() == "000003"
    assert manifests["validation"]["sample_id"].item() == "000008"
    assert "ignored" not in manifests["selector_fit"].columns


def test_table_builder_never_sends_nonzero_initial_mask_to_classifier(
    monkeypatch,
) -> None:
    cache = _cache(1)
    role = fit.frozen_runner.RoleData(
        cache=cache,
        compass_angle_deg=np.asarray([np.nan]),
        compass_available=np.asarray([False]),
    )
    observed_masks: list[np.ndarray] = []

    def fake_state_features(
        local_cache: ActiveViewCache, states: fit.frozen_runner.StateTable
    ) -> np.ndarray:
        observed_masks.append(states.revealed_mask.copy())
        return np.zeros((len(states.sample_indices), 4), dtype=np.float32)

    def fake_predict(*args, **kwargs) -> np.ndarray:
        features = args[2]
        return np.tile(np.asarray([[0.1, 0.2, 0.3]]), (len(features), 1))

    monkeypatch.setattr(fit.frozen_runner, "_state_features", fake_state_features)
    monkeypatch.setattr(fit.frozen_runner, "_predict", fake_predict)
    data = fit._build_relative_utility_table(
        role,
        SimpleNamespace(model=None, standardizer=None),
        temperature=1.0,
        view_cost=0.5,
        cost_matrix=np.asarray([[0, 1, 4], [1, 0, 1], [8, 8, 0]], dtype=float),
        device="cpu",
    )
    assert observed_masks
    assert all(mask[:, 0].all() for mask in observed_masks)
    assert len(data.row_origin) == len(data.table.features)
    assert len(data.state_origin_by_state) == int(data.table.state_id.max()) + 1
    np.testing.assert_array_equal(
        data.row_origin,
        data.state_origin_by_state[data.table.state_id],
    )
    assert set(data.row_origin.tolist()) == set(range(8))
    assert set(data.state_origin_by_state.tolist()) == set(range(8))


def test_registered_config_is_descriptive_and_roll_local() -> None:
    config = json.loads(
        (fit.REPO_ROOT / "configs" / "milton_zero_shot_active_view_sensitivity_v1.json").read_text(
            encoding="utf-8"
        )
    )
    fit._validate_config(config)
    registered_fit = config["cvian_relative_policy_fit"]
    assert "physical origin r" in registered_fit["origin_augmentation"]
    assert "local sector 0" in registered_fit["origin_augmentation"]
    evaluation = config["evaluation"]
    assert evaluation["go_no_go_criterion"] is None
    assert evaluation["overwrite_or_rerun_permitted"] is False
    assert evaluation["bootstrap"]["expected_joint_components"] == 11
    assert evaluation["bootstrap"]["purpose"].startswith("descriptive")


def test_existing_completion_verifies_full_fingerprint_and_artifact_metadata(
    tmp_path: Path,
) -> None:
    fingerprint = {
        "schema_version": fit.FIT_SCHEMA,
        "seed": 42,
        "fit_fingerprint_sha256": "a" * 64,
        "source_provenance": {"roles_loaded": ["selector_fit", "validation"]},
    }
    model = fit.frozen_runner.ActiveViewMLP(
        len(RELATIVE_GEOMETRY_FEATURE_NAMES), 1
    )
    artifact_path = tmp_path / "relative_utility_artifact.pt"
    torch.save(
        {
            "schema_version": fit.ARTIFACT_SCHEMA,
            "utility_input_dim": len(RELATIVE_GEOMETRY_FEATURE_NAMES),
            "utility_state_dict": model.state_dict(),
            "utility_mean": np.zeros(len(RELATIVE_GEOMETRY_FEATURE_NAMES)),
            "utility_scale": np.ones(len(RELATIVE_GEOMETRY_FEATURE_NAMES)),
            "metadata": dict(fingerprint),
        },
        artifact_path,
    )
    history_path = tmp_path / "training_history.json"
    history_path.write_text("{}\n", encoding="utf-8")
    regret_path = tmp_path / "validation_top1_regret.csv"
    regret_path.write_text("state_id,regret\n", encoding="utf-8")
    completion = {
        **fingerprint,
        "artifact_sha256": fit._sha256(artifact_path),
        "history_sha256": fit._sha256(history_path),
        "validation_regret_sha256": fit._sha256(regret_path),
    }
    fit._validate_existing_completion(tmp_path, completion, fingerprint)
    drifted = {**completion, "seed": 123}
    with pytest.raises(ValueError, match="field 'seed' drifted"):
        fit._validate_existing_completion(tmp_path, drifted, fingerprint)
