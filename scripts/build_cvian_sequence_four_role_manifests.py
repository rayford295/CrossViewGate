"""Build a prospective four-role CVIAN active-view protocol.

The first CVIAN active-view study used a spatial split whose training role was
subdivided differently for each model seed.  Its validation role selected
epochs and its development test was consumed while correcting the protocol.
Simply making another random split would therefore not create a defensible new
policy test.

This builder reconstructs those historical selector-selection roles and fixes
the new ``prospective_test`` to every complete Mapillary sequence for which no
sample appeared in any of them.  Historical base/model fitting is audited
separately: the resulting test samples did enter the old base-fit path, so old
checkpoints, embedding caches, and fitted heads are explicitly ineligible for
reuse.  The remaining sequences are assigned to ``base_fit``,
``selector_fit``, and ``validation`` without sequence overlap.  A 25 m buffer
then removes lower-priority boundary samples while preserving the test first.

This is a selector-selection holdout with historical base exposure.  It does
not mean that the images have never been processed by any historical model,
nor does it replace an external-event confirmation set.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.data.cvian import purge_spatial_buffer


PROTOCOL_VERSION = "cvian-sequence-four-role-v1"
ROLE_NAMES = ("base_fit", "selector_fit", "validation", "prospective_test")
FIT_ROLE_NAMES = ROLE_NAMES[:3]
DEFAULT_HISTORICAL_SEEDS = (42, 123, 456, 789, 1011)
EARTH_RADIUS_M = 6_371_008.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sequence-held-out four-role CVIAN protocol whose test "
            "sequences did not enter the historical selector-selection roles."
        )
    )
    parser.add_argument(
        "--source-split-dir",
        default="data/splits/ian_hurricane_original_sequence",
        help="Three-way manifest directory whose union contains all CVIAN rows.",
    )
    parser.add_argument(
        "--historical-spatial-split-dir",
        default="data/splits/ian_hurricane_original",
        help="Spatial-v1 manifests used by the consumed active-view study.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits/ian_hurricane_sequence_four_role_v1",
    )
    parser.add_argument("--base-fit-fraction", type=float, default=0.70)
    parser.add_argument("--selector-fit-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--spatial-buffer-m", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument(
        "--historical-selector-seeds",
        default=",".join(str(value) for value in DEFAULT_HISTORICAL_SEEDS),
    )
    parser.add_argument("--historical-selector-block-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-test-rows", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(values: Iterable[object]) -> str:
    canonical = "\n".join(sorted(str(value) for value in values)) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_count_map(values: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in values.value_counts().sort_index().items()
    }


def _read_manifest(path: Path) -> pd.DataFrame:
    string_columns = {
        "sample_id": str,
        "mapillary_id": str,
        "objectid": str,
        "group_id": str,
        "sequence_id": str,
        "spatial_block_id": str,
        "tile_id": str,
    }
    return pd.read_csv(path, dtype=string_columns)


def load_complete_source(split_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load a complete three-way manifest union without trusting old roles."""
    frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    reference_columns: list[str] | None = None
    for role in ("train", "val", "test"):
        path = split_dir / f"{role}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source manifest: {path}")
        frame = _read_manifest(path)
        if reference_columns is None:
            reference_columns = frame.columns.tolist()
        elif frame.columns.tolist() != reference_columns:
            raise ValueError("Source manifests do not have identical ordered columns")
        frame["source_manifest_role"] = role
        frames.append(frame)
        source_hashes[role] = _sha256_file(path)

    full = pd.concat(frames, ignore_index=True)
    required = {
        "sample_id",
        "sequence_id",
        "spatial_block_id",
        "label",
        "latitude",
        "longitude",
    }
    missing = sorted(required - set(full.columns))
    if missing:
        raise ValueError(f"Source manifests are missing required columns: {missing}")
    for column in ("sample_id", "sequence_id", "spatial_block_id", "label"):
        if full[column].isna().any() or (full[column].astype(str).str.strip() == "").any():
            raise ValueError(f"Source column {column!r} must be complete")
    if full["sample_id"].duplicated().any():
        examples = full.loc[full["sample_id"].duplicated(), "sample_id"].head().tolist()
        raise ValueError(f"Source manifests contain duplicate sample ids: {examples}")
    coordinates = full[["latitude", "longitude"]].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Source coordinates must be finite")
    if ((coordinates[:, 0] < -90) | (coordinates[:, 0] > 90)).any() or (
        (coordinates[:, 1] < -180) | (coordinates[:, 1] > 180)
    ).any():
        raise ValueError("Source coordinates are outside WGS84 bounds")
    return full.sort_values("sample_id").reset_index(drop=True), source_hashes


def reconstruct_selector_fit(
    train: pd.DataFrame,
    *,
    seed: int,
    block_fraction: float,
    block_column: str = "spatial_block_id",
    label_column: str = "label",
) -> tuple[pd.DataFrame, set[str], int]:
    """Reproduce the spatial-v1 runner's seeded selector-block assignment."""
    if not 0.05 <= block_fraction <= 0.5:
        raise ValueError("historical selector block fraction must be between 0.05 and 0.5")
    for column in ("sample_id", block_column, label_column):
        if column not in train:
            raise ValueError(f"Historical train manifest is missing {column!r}")
    groups = np.unique(train[block_column].astype(str).to_numpy())
    target_count = max(1, int(round(len(groups) * block_fraction)))
    for attempt in range(1000):
        rng = np.random.default_rng(seed + 104729 * attempt)
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        selector_groups = set(shuffled[:target_count].tolist())
        selector_mask = train[block_column].astype(str).isin(selector_groups)
        model_fit = train.loc[~selector_mask]
        selector_fit = train.loc[selector_mask]
        if (
            not model_fit.empty
            and not selector_fit.empty
            and model_fit[label_column].nunique() == train[label_column].nunique()
            and selector_fit[label_column].nunique() == train[label_column].nunique()
        ):
            return selector_fit.copy().reset_index(drop=True), selector_groups, attempt
    raise ValueError("Could not reconstruct a class-complete historical selector split")


def historical_selection_exposure(
    split_dir: Path,
    *,
    seeds: Sequence[int],
    selector_block_fraction: float,
) -> tuple[set[str], set[str], pd.DataFrame, dict[str, object]]:
    """Return direct selection exposure and old base-fit ids.

    Direct selector-selection exposure includes every seed's selector-fit rows,
    the validation labels used for classifier/selector early stopping, and the
    consumed development-test rows.  The old spatial train is returned
    separately because it was base/model-fit exposure and therefore forbids
    checkpoint reuse even when a row was not direct selector-selection data.
    """
    paths = {role: split_dir / f"{role}.csv" for role in ("train", "val", "test")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing historical manifest: {path}")
    frames = {role: _read_manifest(path) for role, path in paths.items()}
    selector_union: set[str] = set()
    seed_rows: list[dict[str, object]] = []
    for seed in seeds:
        selector_fit, groups, attempt = reconstruct_selector_fit(
            frames["train"],
            seed=int(seed),
            block_fraction=selector_block_fraction,
        )
        ids = set(selector_fit["sample_id"].astype(str))
        selector_union |= ids
        seed_rows.append(
            {
                "seed": int(seed),
                "attempt": int(attempt),
                "selector_fit_rows": int(len(selector_fit)),
                "selector_fit_spatial_blocks": int(len(groups)),
                "selector_fit_sample_id_sha256": _canonical_digest(ids),
            }
        )

    validation_ids = set(frames["val"]["sample_id"].astype(str))
    consumed_test_ids = set(frames["test"]["sample_id"].astype(str))
    direct_exposure = selector_union | validation_ids | consumed_test_ids
    base_fit_ids = set(frames["train"]["sample_id"].astype(str))
    metadata: dict[str, object] = {
        "historical_split_dir": str(split_dir.resolve()),
        "historical_manifest_sha256": {
            role: _sha256_file(path) for role, path in paths.items()
        },
        "selector_seeds": [int(seed) for seed in seeds],
        "selector_block_fraction": float(selector_block_fraction),
        "selector_fit_union_rows": int(len(selector_union)),
        "selector_validation_rows": int(len(validation_ids)),
        "consumed_development_test_rows": int(len(consumed_test_ids)),
        "direct_selection_exposure_union_rows": int(len(direct_exposure)),
        "historical_base_fit_rows": int(len(base_fit_ids)),
    }
    return direct_exposure, base_fit_ids, pd.DataFrame(seed_rows), metadata


def derive_sequence_clean_holdout(
    full: pd.DataFrame,
    selection_exposed_sample_ids: set[str],
    *,
    sequence_column: str = "sequence_id",
    label_column: str = "label",
    minimum_rows: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fix the test to all sequences with zero direct selection exposure."""
    source_ids = set(full["sample_id"].astype(str))
    unknown = selection_exposed_sample_ids - source_ids
    if unknown:
        raise ValueError(
            "Historical selection exposure contains ids absent from the complete source: "
            f"{sorted(unknown)[:5]}"
        )
    rows: list[dict[str, object]] = []
    eligible_sequences: list[str] = []
    for sequence_id, group in full.groupby(sequence_column, sort=True):
        ids = set(group["sample_id"].astype(str))
        exposed = ids & selection_exposed_sample_ids
        eligible = not exposed
        if eligible:
            eligible_sequences.append(str(sequence_id))
        record: dict[str, object] = {
            "sequence_id": str(sequence_id),
            "row_count": int(len(group)),
            "selection_exposed_rows": int(len(exposed)),
            "eligible_for_prospective_test": bool(eligible),
        }
        for label, count in group[label_column].value_counts().sort_index().items():
            record[f"label_{label}_rows"] = int(count)
        rows.append(record)
    sequence_audit = pd.DataFrame(rows).fillna(0)
    holdout = full.loc[full[sequence_column].astype(str).isin(eligible_sequences)].copy()
    remainder = full.loc[~full[sequence_column].astype(str).isin(eligible_sequences)].copy()
    holdout = holdout.sort_values("sample_id").reset_index(drop=True)
    remainder = remainder.sort_values("sample_id").reset_index(drop=True)
    if len(holdout) < minimum_rows:
        raise ValueError(
            f"Only {len(holdout)} sequence-clean holdout rows; minimum is {minimum_rows}"
        )
    expected_labels = set(full[label_column].astype(str))
    actual_labels = set(holdout[label_column].astype(str))
    if actual_labels != expected_labels:
        raise ValueError(
            "Sequence-clean holdout is not class complete: "
            f"missing {sorted(expected_labels - actual_labels)}"
        )
    if set(holdout["sample_id"].astype(str)) & selection_exposed_sample_ids:
        raise RuntimeError("Internal error: holdout contains a directly exposed sample")
    return holdout, remainder, sequence_audit


def assign_grouped_roles(
    frame: pd.DataFrame,
    *,
    role_fractions: Mapping[str, float],
    group_column: str,
    label_column: str,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Greedily assign indivisible groups to an arbitrary number of roles."""
    roles = list(role_fractions)
    if not roles:
        raise ValueError("At least one role is required")
    fractions = np.asarray([role_fractions[role] for role in roles], dtype=float)
    if np.any(~np.isfinite(fractions)) or np.any(fractions <= 0):
        raise ValueError("Role fractions must be finite and positive")
    if not math.isclose(float(fractions.sum()), 1.0, abs_tol=1e-9):
        raise ValueError("Role fractions must sum to one")
    for column in (group_column, label_column):
        if column not in frame:
            raise ValueError(f"Assignment frame is missing {column!r}")
        if frame[column].isna().any():
            raise ValueError(f"Assignment column {column!r} cannot contain nulls")

    labels = sorted(frame[label_column].astype(str).unique().tolist())
    label_index = {label: index for index, label in enumerate(labels)}
    rng = np.random.default_rng(seed)
    group_rows: list[tuple[str, int, np.ndarray, float]] = []
    for group_name, group in frame.groupby(group_column, sort=True):
        counts = np.zeros(len(labels), dtype=float)
        for label, count in group[label_column].astype(str).value_counts().items():
            counts[label_index[label]] = float(count)
        group_rows.append((str(group_name), len(group), counts, float(rng.random())))
    if len(group_rows) < len(roles):
        raise ValueError(f"Need at least {len(roles)} groups for {len(roles)} roles")

    group_rows.sort(key=lambda item: (-item[1], -float(item[2].max()), item[3], item[0]))
    target_total = fractions * len(frame)
    global_labels = np.asarray(
        [(frame[label_column].astype(str) == label).sum() for label in labels],
        dtype=float,
    )
    target_labels = fractions[:, None] * global_labels[None, :]
    assigned_total = np.zeros(len(roles), dtype=float)
    assigned_labels = np.zeros((len(roles), len(labels)), dtype=float)
    assignments: dict[str, int] = {}

    for position, (group_name, group_size, group_labels, _) in enumerate(group_rows):
        empty_roles = [index for index in range(len(roles)) if index not in assignments.values()]
        groups_remaining = len(group_rows) - position
        candidates = empty_roles if groups_remaining == len(empty_roles) else list(range(len(roles)))
        best: tuple[float, int] | None = None
        for role_index in candidates:
            totals = assigned_total.copy()
            label_counts = assigned_labels.copy()
            totals[role_index] += group_size
            label_counts[role_index] += group_labels
            total_error = np.square(
                (totals - target_total) / np.maximum(target_total, 1.0)
            ).mean()
            label_error = np.square(
                (label_counts - target_labels) / np.maximum(target_labels, 1.0)
            ).mean()
            overflow = np.square(
                np.maximum(totals - target_total, 0.0) / np.maximum(target_total, 1.0)
            ).mean()
            candidate = (float(total_error + label_error + 2.0 * overflow), role_index)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        selected = best[1]
        assignments[group_name] = selected
        assigned_total[selected] += group_size
        assigned_labels[selected] += group_labels

    assignment_series = frame[group_column].astype(str).map(assignments)
    if assignment_series.isna().any():
        raise RuntimeError("Internal error: not every group was assigned")
    result = {
        role: frame.loc[assignment_series == index].copy().reset_index(drop=True)
        for index, role in enumerate(roles)
    }
    expected_labels = set(frame[label_column].astype(str))
    for role, role_frame in result.items():
        if role_frame.empty:
            raise ValueError(f"Grouped assignment produced an empty {role} role")
        actual_labels = set(role_frame[label_column].astype(str))
        if actual_labels != expected_labels:
            raise ValueError(
                f"Grouped assignment produced a class-incomplete {role} role; "
                f"missing {sorted(expected_labels - actual_labels)}"
            )

    assignment_rows: list[dict[str, object]] = []
    for group_name, group in frame.groupby(group_column, sort=True):
        role = roles[assignments[str(group_name)]]
        record: dict[str, object] = {
            group_column: str(group_name),
            "protocol_role": role,
            "raw_rows": int(len(group)),
        }
        for label, count in group[label_column].value_counts().sort_index().items():
            record[f"raw_label_{label}_rows"] = int(count)
        assignment_rows.append(record)
    return result, pd.DataFrame(assignment_rows).fillna(0)


def _assert_role_isolation(
    roles: Mapping[str, pd.DataFrame],
    *,
    label_column: str = "label",
) -> None:
    expected_labels = set().union(
        *(set(frame[label_column].astype(str)) for frame in roles.values())
    )
    for role, frame in roles.items():
        if frame.empty:
            raise ValueError(f"Protocol role {role!r} is empty")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"Protocol role {role!r} contains duplicate sample ids")
        actual_labels = set(frame[label_column].astype(str))
        if actual_labels != expected_labels:
            raise ValueError(f"Protocol role {role!r} is not class complete")
    for left, right in combinations(roles, 2):
        sample_overlap = set(roles[left]["sample_id"].astype(str)) & set(
            roles[right]["sample_id"].astype(str)
        )
        sequence_overlap = set(roles[left]["sequence_id"].astype(str)) & set(
            roles[right]["sequence_id"].astype(str)
        )
        spatial_block_overlap = set(roles[left]["spatial_block_id"].astype(str)) & set(
            roles[right]["spatial_block_id"].astype(str)
        )
        if sample_overlap:
            raise ValueError(f"sample_id leakage between {left} and {right}")
        if sequence_overlap:
            raise ValueError(f"sequence_id leakage between {left} and {right}")
        if spatial_block_overlap:
            raise ValueError(f"spatial_block_id leakage between {left} and {right}")


def purge_shared_spatial_blocks(
    roles: Mapping[str, pd.DataFrame],
    *,
    priority: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Remove lower-priority rows whose grid block is already protected."""
    if set(priority) != set(roles) or len(priority) != len(roles):
        raise ValueError("Spatial-block priority must name every role exactly once")
    protected_blocks: dict[str, str] = {}
    retained: dict[str, pd.DataFrame] = {}
    exclusions: list[pd.DataFrame] = []
    for role in priority:
        frame = roles[role].copy()
        block_keys = frame["spatial_block_id"].astype(str)
        excluded = block_keys.isin(protected_blocks)
        if excluded.any():
            removed = frame.loc[
                excluded,
                ["sample_id", "sequence_id", "spatial_block_id", "label"],
            ].copy()
            removed["excluded_from"] = role
            removed["near_split"] = block_keys.loc[excluded].map(protected_blocks).to_numpy()
            removed["minimum_distance_m"] = np.nan
            removed["exclusion_stage"] = "shared_spatial_block"
            removed["exclusion_reason"] = "spatial_block_owned_by_higher_priority_role"
            exclusions.append(removed)
        retained[role] = frame.loc[~excluded].reset_index(drop=True)
        for block in retained[role]["spatial_block_id"].astype(str).unique():
            protected_blocks[block] = role
    ordered = {role: retained[role] for role in roles}
    exclusion_frame = (
        pd.concat(exclusions, ignore_index=True)
        if exclusions
        else pd.DataFrame(
            columns=[
                "sample_id",
                "sequence_id",
                "spatial_block_id",
                "label",
                "excluded_from",
                "near_split",
                "minimum_distance_m",
                "exclusion_stage",
                "exclusion_reason",
            ]
        )
    )
    return ordered, exclusion_frame


def build_four_role_protocol(
    full: pd.DataFrame,
    *,
    selection_exposed_sample_ids: set[str],
    fit_role_fractions: Mapping[str, float],
    spatial_buffer_m: float,
    seed: int,
    minimum_test_rows: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct, buffer, and validate the four protocol roles."""
    if tuple(fit_role_fractions) != FIT_ROLE_NAMES:
        raise ValueError(f"Fit role order must be exactly {FIT_ROLE_NAMES}")
    prospective_test, remainder, eligibility = derive_sequence_clean_holdout(
        full,
        selection_exposed_sample_ids,
        minimum_rows=minimum_test_rows,
    )
    roles, assignment = assign_grouped_roles(
        remainder,
        role_fractions=fit_role_fractions,
        group_column="sequence_id",
        label_column="label",
        seed=seed,
    )
    roles["prospective_test"] = prospective_test
    raw_roles = {role: frame.copy() for role, frame in roles.items()}
    priority = ("prospective_test", "validation", "selector_fit", "base_fit")
    roles, block_exclusions = purge_shared_spatial_blocks(roles, priority=priority)
    roles, distance_exclusions = purge_spatial_buffer(
        roles,
        spatial_buffer_m,
        priority=priority,
    )
    if set(roles["prospective_test"]["sample_id"].astype(str)) != set(
        prospective_test["sample_id"].astype(str)
    ):
        raise RuntimeError("The spatial buffer changed the protected prospective test")
    _assert_role_isolation(roles)

    raw_lookup = pd.concat(
        [frame.assign(raw_protocol_role=role) for role, frame in raw_roles.items()],
        ignore_index=True,
    )[
        ["sample_id", "sequence_id", "spatial_block_id", "label", "raw_protocol_role"]
    ]
    if not distance_exclusions.empty:
        distance_exclusions = distance_exclusions.merge(
            raw_lookup, on="sample_id", how="left", validate="one_to_one"
        )
        distance_exclusions["exclusion_stage"] = "distance_buffer"
        distance_exclusions["exclusion_reason"] = "cross_role_distance_at_or_below_buffer"
        distance_exclusions = distance_exclusions[
            [
                "sample_id",
                "sequence_id",
                "spatial_block_id",
                "label",
                "excluded_from",
                "near_split",
                "minimum_distance_m",
                "exclusion_stage",
                "exclusion_reason",
            ]
        ]
    exclusions = pd.concat(
        [block_exclusions, distance_exclusions], ignore_index=True, sort=False
    )

    retained_by_sequence = pd.concat(
        [
            frame.groupby("sequence_id").size().rename("retained_rows").reset_index().assign(
                protocol_role=role
            )
            for role, frame in roles.items()
            if role in FIT_ROLE_NAMES
        ],
        ignore_index=True,
    )
    assignment = assignment.merge(
        retained_by_sequence,
        on=["sequence_id", "protocol_role"],
        how="left",
        validate="one_to_one",
    )
    assignment["retained_rows"] = assignment["retained_rows"].fillna(0).astype(int)

    for role, frame in roles.items():
        frame = frame.sort_values("sample_id").reset_index(drop=True)
        frame["protocol_role"] = role
        frame["protocol_version"] = PROTOCOL_VERSION
        frame["event_id"] = "hurricane_ian_cvian"
        frame["test_scope"] = (
            "selector_selection_holdout_with_historical_base_exposure"
            if role == "prospective_test"
            else "fit_or_selection_role"
        )
        roles[role] = frame
    return roles, exclusions, assignment, eligibility


def _minimum_cross_distance_m(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left_coordinates = left[["latitude", "longitude"]].to_numpy(dtype=float)
    right_coordinates = right[["latitude", "longitude"]].to_numpy(dtype=float)
    right_radians = np.radians(right_coordinates)
    minimum = math.inf
    for start in range(0, len(left_coordinates), 512):
        left_radians = np.radians(left_coordinates[start : start + 512])
        latitude_left = left_radians[:, 0, None]
        longitude_left = left_radians[:, 1, None]
        latitude_right = right_radians[None, :, 0]
        longitude_right = right_radians[None, :, 1]
        delta_latitude = latitude_right - latitude_left
        delta_longitude = longitude_right - longitude_left
        haversine = np.sin(delta_latitude / 2.0) ** 2
        haversine += (
            np.cos(latitude_left)
            * np.cos(latitude_right)
            * np.sin(delta_longitude / 2.0) ** 2
        )
        distances = 2.0 * EARTH_RADIUS_M * np.arcsin(
            np.sqrt(np.clip(haversine, 0.0, 1.0))
        )
        minimum = min(minimum, float(distances.min()))
    return minimum


def pairwise_role_audit(
    roles: Mapping[str, pd.DataFrame], spatial_buffer_m: float
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left, right in combinations(ROLE_NAMES, 2):
        left_frame = roles[left]
        right_frame = roles[right]
        minimum_distance = _minimum_cross_distance_m(left_frame, right_frame)
        record = {
            "left_role": left,
            "right_role": right,
            "left_rows": int(len(left_frame)),
            "right_rows": int(len(right_frame)),
            "sample_id_overlap": int(
                len(set(left_frame["sample_id"].astype(str)) & set(right_frame["sample_id"].astype(str)))
            ),
            "sequence_id_overlap": int(
                len(set(left_frame["sequence_id"].astype(str)) & set(right_frame["sequence_id"].astype(str)))
            ),
            "spatial_block_id_overlap": int(
                len(
                    set(left_frame["spatial_block_id"].astype(str))
                    & set(right_frame["spatial_block_id"].astype(str))
                )
            ),
            "minimum_distance_m": minimum_distance,
            "spatial_buffer_m": float(spatial_buffer_m),
            "within_or_on_buffer": bool(minimum_distance <= spatial_buffer_m),
        }
        rows.append(record)
    audit = pd.DataFrame(rows)
    if audit["sample_id_overlap"].any() or audit["sequence_id_overlap"].any():
        raise RuntimeError("Pairwise audit found record or sequence leakage")
    if audit["spatial_block_id_overlap"].any():
        raise RuntimeError("Pairwise audit found spatial-block leakage")
    if audit["within_or_on_buffer"].any():
        raise RuntimeError("Pairwise audit found a cross-role pair inside the spatial buffer")
    return audit


def strict_block_sensitivity(
    raw_roles: Mapping[str, pd.DataFrame], spatial_buffer_m: float
) -> pd.DataFrame:
    """Quantify the cost of additionally requiring zero shared grid blocks."""
    used_blocks: set[str] = set()
    block_purged: dict[str, pd.DataFrame] = {}
    block_exclusions: dict[str, int] = {}
    for role in ("prospective_test", "validation", "selector_fit", "base_fit"):
        frame = raw_roles[role]
        excluded = frame["spatial_block_id"].astype(str).isin(used_blocks)
        block_exclusions[role] = int(excluded.sum())
        block_purged[role] = frame.loc[~excluded].copy().reset_index(drop=True)
        used_blocks |= set(block_purged[role]["spatial_block_id"].astype(str))
    buffered, distance_exclusions = purge_spatial_buffer(
        block_purged,
        spatial_buffer_m,
        priority=("prospective_test", "validation", "selector_fit", "base_fit"),
    )
    distance_counts = _json_count_map(
        distance_exclusions.get("excluded_from", pd.Series(dtype=str))
    )
    return pd.DataFrame(
        [
            {
                "protocol_role": role,
                "raw_rows": int(len(raw_roles[role])),
                "shared_block_exclusions": block_exclusions[role],
                "distance_buffer_exclusions_after_block_purge": int(
                    distance_counts.get(role, 0)
                ),
                "retained_rows": int(len(buffered[role])),
                "retained_fraction_of_raw": float(len(buffered[role]) / len(raw_roles[role])),
            }
            for role in ROLE_NAMES
        ]
    )


def dependency_components(full: pd.DataFrame) -> pd.DataFrame:
    """Summarise sequence/block connected components without dropping rows."""
    sequences = sorted(full["sequence_id"].astype(str).unique().tolist())
    parent = {sequence: sequence for sequence in sequences}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        parent[high] = low

    for _, group in full.groupby("spatial_block_id", sort=True):
        members = sorted(group["sequence_id"].astype(str).unique().tolist())
        for member in members[1:]:
            union(members[0], member)
    roots = sorted({find(sequence) for sequence in sequences})
    root_to_id = {root: f"component_{index:03d}" for index, root in enumerate(roots)}
    component_ids = full["sequence_id"].astype(str).map(lambda value: root_to_id[find(value)])
    records: list[dict[str, object]] = []
    for component_id, group in full.assign(dependency_component=component_ids).groupby(
        "dependency_component", sort=True
    ):
        record: dict[str, object] = {
            "dependency_component": component_id,
            "row_count": int(len(group)),
            "sequence_count": int(group["sequence_id"].nunique()),
            "spatial_block_count": int(group["spatial_block_id"].nunique()),
            "sequence_id_set_sha256": _canonical_digest(group["sequence_id"].astype(str).unique()),
        }
        for label, count in group["label"].value_counts().sort_index().items():
            record[f"label_{label}_rows"] = int(count)
        records.append(record)
    return pd.DataFrame(records).fillna(0).sort_values(
        ["row_count", "dependency_component"], ascending=[False, True]
    ).reset_index(drop=True)


def _role_summary(roles: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for role in ROLE_NAMES:
        frame = roles[role]
        record: dict[str, object] = {
            "protocol_role": role,
            "rows": int(len(frame)),
            "sequences": int(frame["sequence_id"].nunique()),
            "spatial_blocks": int(frame["spatial_block_id"].nunique()),
        }
        for label, count in frame["label"].value_counts().sort_index().items():
            record[f"label_{label}_rows"] = int(count)
        rows.append(record)
    return pd.DataFrame(rows).fillna(0)


def write_protocol(
    *,
    output_dir: Path,
    roles: Mapping[str, pd.DataFrame],
    exclusions: pd.DataFrame,
    assignment: pd.DataFrame,
    eligibility: pd.DataFrame,
    selector_seed_audit: pd.DataFrame,
    pairwise_audit: pd.DataFrame,
    strict_sensitivity: pd.DataFrame,
    components: pd.DataFrame,
    source_split_dir: Path,
    source_hashes: Mapping[str, str],
    historical_metadata: Mapping[str, object],
    historical_base_fit_ids: set[str],
    fit_role_fractions: Mapping[str, float],
    spatial_buffer_m: float,
    seed: int,
    overwrite: bool,
) -> dict[str, object]:
    filenames = [f"{role}.csv" for role in ROLE_NAMES] + [
        "buffer_exclusions.csv",
        "sequence_assignment.csv",
        "historical_sequence_eligibility.csv",
        "historical_selector_seed_audit.csv",
        "role_audit.csv",
        "pairwise_audit.csv",
        "strict_spatial_block_sensitivity.csv",
        "dependency_components.csv",
        "leakage_audit.json",
        "test_commitment.json",
        "protocol_summary.json",
    ]
    existing = [output_dir / filename for filename in filenames if (output_dir / filename).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite protocol artifact {existing[0]}; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for role in ROLE_NAMES:
        roles[role].to_csv(output_dir / f"{role}.csv", index=False)
    exclusions.to_csv(output_dir / "buffer_exclusions.csv", index=False)
    assignment.to_csv(output_dir / "sequence_assignment.csv", index=False)
    eligibility.to_csv(output_dir / "historical_sequence_eligibility.csv", index=False)
    selector_seed_audit.to_csv(output_dir / "historical_selector_seed_audit.csv", index=False)
    role_audit = _role_summary(roles)
    role_audit.to_csv(output_dir / "role_audit.csv", index=False)
    pairwise_audit.to_csv(output_dir / "pairwise_audit.csv", index=False)
    strict_sensitivity.to_csv(output_dir / "strict_spatial_block_sensitivity.csv", index=False)
    components.to_csv(output_dir / "dependency_components.csv", index=False)

    test_frame = roles["prospective_test"]
    test_ids = set(test_frame["sample_id"].astype(str))
    test_sequences = set(test_frame["sequence_id"].astype(str))
    historical_base_overlap = test_ids & historical_base_fit_ids
    commitment = {
        "schema_version": "cvian-sequence-test-commitment-v1",
        "protocol_version": PROTOCOL_VERSION,
        "role": "prospective_test",
        "status": "selector_selection_holdout_with_historical_base_exposure",
        "holdout_definition": (
            "No sample from any retained test sequence appeared in the historical "
            "five-seed selector-fit union, selector validation, or consumed development test."
        ),
        "historically_never_seen": False,
        "historical_base_fit_overlap_rows": int(len(historical_base_overlap)),
        "historical_checkpoint_reuse_permitted": False,
        "row_count": int(len(test_frame)),
        "sequence_count": int(len(test_sequences)),
        "label_counts": _json_count_map(test_frame["label"]),
        "sample_id_set_sha256": _canonical_digest(test_ids),
        "sequence_id_set_sha256": _canonical_digest(test_sequences),
        "manifest_sha256": _sha256_file(output_dir / "prospective_test.csv"),
        "use_policy": (
            "Do not score until encoder, classifier, utility/selector, stopping rule, "
            "hyperparameters, and validation decisions are frozen and hashed."
        ),
        "claim_limit": (
            "One development confirmation of the within-CVIAN selector/policy after "
            "full retraining; not an external-event or historically never-seen "
            "confirmatory test."
        ),
    }
    (output_dir / "test_commitment.json").write_text(
        json.dumps(commitment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    leakage_audit = {
        "schema_version": "cvian-sequence-four-role-leakage-audit-v1",
        "protocol_version": PROTOCOL_VERSION,
        "sample_id_overlap_zero": bool((pairwise_audit["sample_id_overlap"] == 0).all()),
        "sequence_id_overlap_zero": bool((pairwise_audit["sequence_id_overlap"] == 0).all()),
        "spatial_buffer_clear": bool((~pairwise_audit["within_or_on_buffer"]).all()),
        "spatial_block_id_overlap_zero": bool(
            (pairwise_audit["spatial_block_id_overlap"] == 0).all()
        ),
        "pairwise": pairwise_audit.to_dict(orient="records"),
        "historical_selection": dict(historical_metadata),
        "test_commitment": commitment,
    }
    (output_dir / "leakage_audit.json").write_text(
        json.dumps(leakage_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    role_hashes = {
        f"{role}.csv": _sha256_file(output_dir / f"{role}.csv") for role in ROLE_NAMES
    }
    largest_component = components.iloc[0].to_dict()
    summary: dict[str, object] = {
        "schema_version": "cvian-sequence-four-role-summary-v1",
        "protocol_version": PROTOCOL_VERSION,
        "event_id": "hurricane_ian_cvian",
        "seed": int(seed),
        "source_split_dir": str(source_split_dir.resolve()),
        "source_manifest_sha256": dict(source_hashes),
        "fit_role_fractions_before_buffer": {
            role: float(value) for role, value in fit_role_fractions.items()
        },
        "spatial_buffer_m": float(spatial_buffer_m),
        "buffer_priority": ["prospective_test", "validation", "selector_fit", "base_fit"],
        "role_rows": {
            role: int(len(roles[role])) for role in ROLE_NAMES
        },
        "role_sequences": {
            role: int(roles[role]["sequence_id"].nunique()) for role in ROLE_NAMES
        },
        "assigned_sequence_count": int(
            len(assignment) + roles["prospective_test"]["sequence_id"].nunique()
        ),
        "retained_sequence_count": int(
            sum(roles[role]["sequence_id"].nunique() for role in ROLE_NAMES)
        ),
        "fully_excluded_fit_sequence_count": int(
            (assignment["retained_rows"] == 0).sum()
        ),
        "role_spatial_blocks": {
            role: int(roles[role]["spatial_block_id"].nunique()) for role in ROLE_NAMES
        },
        "role_label_counts": {
            role: _json_count_map(roles[role]["label"]) for role in ROLE_NAMES
        },
        "exclusion_rows": int(len(exclusions)),
        "exclusion_counts_by_role": _json_count_map(
            exclusions.get("excluded_from", pd.Series(dtype=str))
        ),
        "exclusion_counts_by_stage": _json_count_map(
            exclusions.get("exclusion_stage", pd.Series(dtype=str))
        ),
        "record_overlap_zero": True,
        "sequence_overlap_zero": True,
        "spatial_block_overlap_zero": True,
        "spatial_buffer_clear": True,
        "spatial_block_overlap_groups_by_pair": {
            f"{row.left_role}__{row.right_role}": int(row.spatial_block_id_overlap)
            for row in pairwise_audit.itertuples(index=False)
        },
        "minimum_distance_m_by_pair": {
            f"{row.left_role}__{row.right_role}": float(row.minimum_distance_m)
            for row in pairwise_audit.itertuples(index=False)
        },
        "dependency_component_count": int(len(components)),
        "largest_sequence_spatial_component": {
            str(key): (
                int(value)
                if isinstance(value, (np.integer, int))
                else float(value)
                if isinstance(value, (np.floating, float))
                else value
            )
            for key, value in largest_component.items()
        },
        "strict_zero_block_sensitivity_retained_rows": {
            str(row.protocol_role): int(row.retained_rows)
            for row in strict_sensitivity.itertuples(index=False)
        },
        "historical_selection": dict(historical_metadata),
        "test_status": {
            "status": "selector_selection_holdout_with_historical_base_exposure",
            "selector_selection_clean": True,
            "historically_never_seen": False,
            "old_base_or_checkpoint_reuse_permitted": False,
            "old_base_fit_overlap_rows": int(len(historical_base_overlap)),
            "new_protocol_test_scored": False,
            "external_event_confirmation": False,
        },
        "role_manifest_sha256": role_hashes,
        "audit_artifact_sha256": {
            filename: _sha256_file(output_dir / filename)
            for filename in (
                "buffer_exclusions.csv",
                "sequence_assignment.csv",
                "historical_sequence_eligibility.csv",
                "historical_selector_seed_audit.csv",
                "role_audit.csv",
                "pairwise_audit.csv",
                "strict_spatial_block_sensitivity.csv",
                "dependency_components.csv",
                "leakage_audit.json",
                "test_commitment.json",
            )
        },
    }
    (output_dir / "protocol_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    source_split_dir = Path(args.source_split_dir)
    historical_split_dir = Path(args.historical_spatial_split_dir)
    output_dir = Path(args.output_dir)
    seeds = tuple(
        int(value.strip())
        for value in str(args.historical_selector_seeds).split(",")
        if value.strip()
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Historical selector seeds must be a non-empty unique list")
    fit_role_fractions = {
        "base_fit": args.base_fit_fraction,
        "selector_fit": args.selector_fit_fraction,
        "validation": args.validation_fraction,
    }
    if not math.isclose(sum(fit_role_fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Base-fit, selector-fit, and validation fractions must sum to one")

    full, source_hashes = load_complete_source(source_split_dir)
    direct_exposure, historical_base_fit_ids, seed_audit, historical_metadata = (
        historical_selection_exposure(
            historical_split_dir,
            seeds=seeds,
            selector_block_fraction=args.historical_selector_block_fraction,
        )
    )
    roles, exclusions, assignment, eligibility = build_four_role_protocol(
        full,
        selection_exposed_sample_ids=direct_exposure,
        fit_role_fractions=fit_role_fractions,
        spatial_buffer_m=args.spatial_buffer_m,
        seed=args.seed,
        minimum_test_rows=args.minimum_test_rows,
    )
    test_ids = set(roles["prospective_test"]["sample_id"].astype(str))
    test_sequences = set(roles["prospective_test"]["sequence_id"].astype(str))
    historical_metadata = dict(historical_metadata)
    historical_metadata.update(
        {
            "complete_source_rows": int(len(full)),
            "sample_rows_not_directly_selection_exposed": int(
                len(set(full["sample_id"].astype(str)) - direct_exposure)
            ),
            "sequence_clean_test_rows": int(len(test_ids)),
            "sequence_clean_test_sequences": int(len(test_sequences)),
            "test_direct_selection_overlap_rows": int(len(test_ids & direct_exposure)),
            "test_historical_base_fit_overlap_rows": int(
                len(test_ids & historical_base_fit_ids)
            ),
            "historical_checkpoint_reuse_permitted": False,
        }
    )
    # Reconstruct pre-buffer rows from the retained sequence assignment.  A
    # sequence is assigned to exactly one role, so this is unambiguous even if
    # all of its samples were removed by the distance buffer.
    assignment_role = dict(zip(assignment["sequence_id"].astype(str), assignment["protocol_role"]))
    for sequence_id in test_sequences:
        assignment_role[sequence_id] = "prospective_test"
    raw_role_lookup = {
        role: full.loc[
            full["sequence_id"].astype(str).map(assignment_role).eq(role)
        ].copy().reset_index(drop=True)
        for role in ROLE_NAMES
    }
    pairwise = pairwise_role_audit(roles, args.spatial_buffer_m)
    strict_sensitivity = strict_block_sensitivity(raw_role_lookup, args.spatial_buffer_m)
    components = dependency_components(full)
    summary = write_protocol(
        output_dir=output_dir,
        roles=roles,
        exclusions=exclusions,
        assignment=assignment,
        eligibility=eligibility,
        selector_seed_audit=seed_audit,
        pairwise_audit=pairwise,
        strict_sensitivity=strict_sensitivity,
        components=components,
        source_split_dir=source_split_dir,
        source_hashes=source_hashes,
        historical_metadata=historical_metadata,
        historical_base_fit_ids=historical_base_fit_ids,
        fit_role_fractions=fit_role_fractions,
        spatial_buffer_m=args.spatial_buffer_m,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {PROTOCOL_VERSION} to {output_dir}")


if __name__ == "__main__":
    main()
