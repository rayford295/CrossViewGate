from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
from PIL import Image
import pytest

from scripts.build_milton_canonical_sensitivity_manifest import (
    ExpectedCounts,
    build_canonical_sensitivity_manifest,
)


def _save_panorama(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 4), color).save(path, format="PNG")


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    canonical_root = tmp_path / "Bi-temporal_hurricane"
    legacy_root = tmp_path / "hurrican-milton-GenDisasterSVI"
    rows = [
        ("folder_0", "100_vs_900(0)", "mild", (10, 20, 30), (90, 80, 70)),
        ("folder_1", "101_vs_900(1)", "Moderate", (11, 21, 31), (90, 80, 70)),
        ("folder_2", "102_vs_901(0)", "Severe", (12, 22, 32), (60, 50, 40)),
        ("folder_0", "103_vs_902(0)", "mild", (13, 23, 33), (30, 40, 50)),
    ]
    location_rows: list[dict[str, object]] = []
    canonical_paths: dict[str, tuple[Path, Path]] = {}
    for index, (folder, sample_id, label, pre_color, post_color) in enumerate(rows):
        pair_dir = canonical_root / folder / sample_id
        pre_id, rest = sample_id.split("_vs_")
        post_id = rest.split("(", maxsplit=1)[0]
        pre_path = pair_dir / f"{pre_id}_2023.png"
        post_path = pair_dir / f"{post_id}_2024.png"
        _save_panorama(pre_path, pre_color)
        _save_panorama(post_path, post_color)
        canonical_paths[sample_id] = (pre_path, post_path)
        location_rows.append(
            {
                "root": f"/upstream/{folder}/{sample_id}",
                "lat": 29.0 + index / 1000,
                "lon": -83.0 - index / 1000,
                "human_damage_perception": label,
            }
        )
    location_csv = canonical_root / "Location.csv"
    pd.DataFrame(location_rows).to_csv(location_csv, index=False)

    # Map three of the four canonical pairs. The two rows sharing post id 900
    # intentionally have conflicting canonical labels, while the legacy label
    # remains mild for both.
    legacy_specs = [
        (0, "100_vs_900(0)", "mild_damage", "train"),
        (1, "101_vs_900(1)", "mild_damage", "test"),
        (2, "102_vs_901(0)", "severe_damage", "val"),
    ]
    source_rows: list[dict[str, object]] = []
    for pair_id, sample_id, damage_level, split_name in legacy_specs:
        pre_source = legacy_root / damage_level / "pre" / f"{pair_id}.png"
        post_source = legacy_root / damage_level / "post" / f"{pair_id}.png"
        overhead = legacy_root / "post_sat" / f"{pair_id}.png"
        pre_source.parent.mkdir(parents=True, exist_ok=True)
        post_source.parent.mkdir(parents=True, exist_ok=True)
        overhead.parent.mkdir(parents=True, exist_ok=True)
        canonical_pre, canonical_post = canonical_paths[sample_id]
        shutil.copyfile(canonical_pre, pre_source)
        shutil.copyfile(canonical_post, post_source)
        # Rows sharing the same post panorama must also resolve to the same
        # packaged overhead content, matching the real Milton snapshot.
        overhead_color = (int(sample_id.split("_vs_")[1].split("(")[0]) % 255, 2, 3)
        Image.new("RGB", (4, 4), overhead_color).save(overhead)
        location = next(
            row for row in location_rows if Path(str(row["root"])).name == sample_id
        )
        source_rows.append(
            {
                "pair_id": pair_id,
                "damage_level": damage_level,
                "prompt": damage_level.replace("_", " "),
                "pre_disaster_image_path": f"/source/{pair_id}.png",
                "post_disaster_image_path": f"/source/{pair_id}.png",
                "post_sat_image_path": f"/source/{pair_id}.png",
                "set": split_name,
                "lat": location["lat"],
                "lon": location["lon"],
            }
        )
    source_csv = legacy_root / "dataset_with_post_sat.csv"
    pd.DataFrame(source_rows).to_csv(source_csv, index=False)
    return canonical_root, legacy_root, location_csv, source_csv


def _run_builder(tmp_path: Path, output_dir: Path) -> dict[str, object]:
    canonical_root, legacy_root, location_csv, source_csv = _build_fixture(tmp_path)
    return build_canonical_sensitivity_manifest(
        gendisaster_root=legacy_root,
        canonical_root=canonical_root,
        location_csv=location_csv,
        source_csv=source_csv,
        output_dir=output_dir,
        expected_counts=ExpectedCounts(
            canonical_rows=4,
            mapped_rows=3,
            missing_rows=1,
            label_mismatches=1,
            conflict_post_groups=1,
            main_rows=1,
        ),
    )


def test_canonical_builder_writes_deduplicated_sensitivity_and_complete_audits(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "milton_canonical_sensitivity_v1"
    _run_builder(tmp_path, output_dir)

    all_rows = pd.read_csv(output_dir / "canonical_all_mapped.csv")
    main = pd.read_csv(output_dir / "main_sensitivity_cohort.csv")
    conflicts = pd.read_csv(output_dir / "conflicting_post_label_audit.csv")
    duplicates = pd.read_csv(output_dir / "unanimous_post_duplicate_audit.csv")
    missing = pd.read_csv(output_dir / "missing_crossview_rows.csv")
    audit = json.loads((output_dir / "provenance_audit.json").read_text())

    assert len(all_rows) == 3
    assert len(main) == 1
    assert main.iloc[0]["sample_id"] == "102_vs_901(0)"
    assert main.iloc[0]["cohort_status"] == "main_unanimous_post_representative"
    assert main.iloc[0]["claim_scope"] == "sensitivity_only_not_confirmatory"
    assert bool(main.iloc[0]["sequence_metadata_available"]) is False
    assert bool(main.iloc[0]["compass_metadata_available"]) is False
    assert pd.isna(main.iloc[0]["sequence_id"])
    assert pd.isna(main.iloc[0]["compass_angle_deg"])

    assert len(conflicts) == 2
    assert conflicts["post_panorama_sha256"].nunique() == 1
    assert set(conflicts["canonical_post_label_set"]) == {
        "mild_damage|moderate_damage"
    }
    assert conflicts["post_label_conflict"].all()
    assert duplicates.empty
    assert missing["sample_id"].tolist() == ["103_vs_902(0)"]
    assert missing.iloc[0]["missing_reason"] == (
        "no_legacy_crossview_row_or_post_sat_mapping"
    )

    assert audit["schema_version"] == "milton-canonical-sensitivity-v1"
    assert audit["confirmatory_eligible"] is False
    assert audit["claim_scope"] == "sensitivity_only_not_confirmatory"
    assert audit["metadata_availability"]["sequence_id"] is False
    assert audit["metadata_availability"]["compass_angle_deg"] is False
    assert audit["counts"]["missing_rows"] == 1
    assert audit["counts"]["conflict_post_groups"] == 1
    assert audit["counts"]["main_rows"] == 1
    assert audit["label_disagreement"]["rows"] == 1
    assert audit["cohort_policy"]["outcome_informed_curation"] is True
    assert set(audit["output_sha256"]) == {
        "canonical_all_mapped.csv",
        "main_sensitivity_cohort.csv",
        "conflicting_post_label_audit.csv",
        "unanimous_post_duplicate_audit.csv",
        "missing_crossview_rows.csv",
    }


def test_canonical_builder_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "milton_canonical_sensitivity_v1"
    _run_builder(tmp_path, output_dir)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_canonical_sensitivity_manifest(
            gendisaster_root=tmp_path / "hurrican-milton-GenDisasterSVI",
            canonical_root=tmp_path / "Bi-temporal_hurricane",
            location_csv=tmp_path / "Bi-temporal_hurricane" / "Location.csv",
            source_csv=(
                tmp_path
                / "hurrican-milton-GenDisasterSVI"
                / "dataset_with_post_sat.csv"
            ),
            output_dir=output_dir,
            expected_counts=ExpectedCounts(4, 3, 1, 1, 1, 1),
        )


def test_canonical_builder_fails_closed_on_nonidentical_packaged_post(
    tmp_path: Path,
) -> None:
    canonical_root, legacy_root, location_csv, source_csv = _build_fixture(tmp_path)
    _save_panorama(legacy_root / "mild_damage" / "post" / "0.png", (1, 1, 1))
    output_dir = tmp_path / "milton_canonical_sensitivity_v1"
    with pytest.raises(ValueError, match="not byte-identical"):
        build_canonical_sensitivity_manifest(
            gendisaster_root=legacy_root,
            canonical_root=canonical_root,
            location_csv=location_csv,
            source_csv=source_csv,
            output_dir=output_dir,
            expected_counts=ExpectedCounts(4, 3, 1, 1, 1, 1),
        )
    assert not output_dir.exists()
