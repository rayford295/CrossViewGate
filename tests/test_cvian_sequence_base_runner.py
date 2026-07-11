from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_cvian_sequence_base_multiseed import ROLES, _validate_protocol


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(tmp_path: Path) -> Path:
    hashes = {}
    for role_index, role in enumerate(ROLES):
        frame = pd.DataFrame(
            [
                {
                    "sample_id": f"{role_index}{label}",
                    "sequence_id": f"sequence-{role_index}-{label}",
                    "spatial_block_id": f"block-{role_index}-{label}",
                    "label": label,
                }
                for label in range(3)
            ]
        )
        path = tmp_path / f"{role}.csv"
        frame.to_csv(path, index=False)
        hashes[path.name] = _sha256(path)
    summary = {
        "schema_version": "cvian-sequence-four-role-summary-v1",
        "record_overlap_zero": True,
        "sequence_overlap_zero": True,
        "spatial_block_overlap_zero": True,
        "spatial_buffer_clear": True,
        "test_status": {
            "status": "selector_selection_holdout_with_historical_base_exposure",
            "old_base_or_checkpoint_reuse_permitted": False,
            "new_protocol_test_scored": False,
        },
        "role_manifest_sha256": hashes,
    }
    (tmp_path / "protocol_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (tmp_path / "test_commitment.json").write_text(
        json.dumps(
            {
                "schema_version": "cvian-sequence-test-commitment-v1",
                "historical_checkpoint_reuse_permitted": False,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_base_runner_accepts_clean_four_role_protocol(tmp_path: Path) -> None:
    result = _validate_protocol(_protocol(tmp_path))
    assert set(result["role_sha256"]) == set(ROLES)


def test_base_runner_rejects_cross_role_sequence_overlap(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    selector_path = protocol / "selector_fit.csv"
    selector = pd.read_csv(selector_path, dtype=str)
    selector.loc[0, "sequence_id"] = "sequence-0-0"
    selector.to_csv(selector_path, index=False)
    summary_path = protocol / "protocol_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["role_manifest_sha256"]["selector_fit.csv"] = _sha256(selector_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="sequence_id overlap"):
        _validate_protocol(protocol)
