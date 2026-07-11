"""Create disjoint gate-fit, risk-calibration, and final-test manifests.

The image models are fitted on the repository's training split.  This command
subdivides the spatial validation split by whole groups so the reliability gate
and the finite-sample risk threshold do not reuse labels.  The existing test
split is copied as the locked final-test role.  A spatial boundary buffer is
then applied with final test and risk calibration protected in that order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.data.cvian import (
    assert_group_isolation,
    purge_spatial_buffer,
    split_grouped_frame,
)


ROLE_NAMES = ("gate_fit", "risk_calibration", "final_test")


def _read_source_manifest(path: Path) -> pd.DataFrame:
    """Preserve stable string identifiers, including CVIAN leading zeros."""
    return pd.read_csv(path, dtype={"sample_id": str})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split an existing validation manifest into group-disjoint gate-fit "
            "and risk-calibration roles while preserving test as final-test."
        )
    )
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-col", default="spatial_block_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--gate-fit-fraction", type=float, default=0.5)
    parser.add_argument("--spatial-buffer-m", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument(
        "--event-id",
        help=(
            "Stable event identifier written to every role. Required when the source "
            "manifests do not already contain one unique event_id/event/dataset value."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_input_frame(frame: pd.DataFrame, name: str, required: set[str]) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} must not be empty")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"{name}.sample_id must be non-null and unique")


def build_protocol_roles(
    validation: pd.DataFrame,
    final_test: pd.DataFrame,
    *,
    group_col: str,
    label_col: str,
    gate_fit_fraction: float,
    spatial_buffer_m: float,
    seed: int,
    event_id: str | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return role manifests and auditable spatial-boundary exclusions."""
    if not 0.0 < gate_fit_fraction < 1.0:
        raise ValueError("gate_fit_fraction must lie strictly between zero and one")
    required = {
        "sample_id",
        group_col,
        label_col,
        "latitude",
        "longitude",
    }
    _validate_input_frame(validation, "validation", required)
    _validate_input_frame(final_test, "final_test", required)

    validation_ids = set(validation["sample_id"].astype(str))
    final_ids = set(final_test["sample_id"].astype(str))
    overlap = validation_ids & final_ids
    if overlap:
        raise ValueError(f"validation/final-test sample overlap: {sorted(overlap)[:5]}")

    remainder = 1.0 - gate_fit_fraction
    temporary = split_grouped_frame(
        validation,
        group_col=group_col,
        label_col=label_col,
        train_fraction=gate_fit_fraction,
        val_fraction=remainder / 2.0,
        seed=seed,
    )
    roles = {
        "gate_fit": temporary["train"].copy(),
        "risk_calibration": pd.concat(
            [temporary["val"], temporary["test"]], ignore_index=True
        ),
        "final_test": final_test.copy().reset_index(drop=True),
    }
    for role_name, frame in roles.items():
        frame["protocol_role"] = role_name
        if event_id is not None:
            frame["event_id"] = event_id

    assert_group_isolation(roles, group_col)
    roles, exclusions = purge_spatial_buffer(
        roles,
        spatial_buffer_m,
        priority=("final_test", "risk_calibration", "gate_fit"),
    )
    assert_group_isolation(roles, group_col)
    for role_name, frame in roles.items():
        if frame.empty:
            raise ValueError(f"Spatial buffer removed every {role_name} sample")
        frame["protocol_role"] = role_name
        if event_id is not None:
            frame["event_id"] = event_id

    retained_ids = {
        role_name: set(frame["sample_id"].astype(str))
        for role_name, frame in roles.items()
    }
    for left_index, left_name in enumerate(ROLE_NAMES):
        for right_name in ROLE_NAMES[left_index + 1 :]:
            if retained_ids[left_name] & retained_ids[right_name]:
                raise RuntimeError(f"sample_id leakage between {left_name} and {right_name}")
    return roles, exclusions


def write_protocol(
    roles: dict[str, pd.DataFrame],
    exclusions: pd.DataFrame,
    output_dir: Path,
    *,
    source_split_dir: Path,
    group_col: str,
    label_col: str,
    gate_fit_fraction: float,
    spatial_buffer_m: float,
    seed: int,
    event_id: str,
    overwrite: bool,
) -> dict[str, object]:
    output_paths = [output_dir / f"{role}.csv" for role in ROLE_NAMES]
    output_paths += [output_dir / "buffer_exclusions.csv", output_dir / "protocol_summary.json"]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing protocol artifact: {existing[0]} "
            "(pass --overwrite to replace generated files)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for role_name in ROLE_NAMES:
        roles[role_name].to_csv(output_dir / f"{role_name}.csv", index=False)
    exclusions.to_csv(output_dir / "buffer_exclusions.csv", index=False)

    summary: dict[str, object] = {
        "schema_version": "p0.7-role-manifests-v1",
        "source_split_dir": str(source_split_dir.resolve()),
        "group_column": group_col,
        "label_column": label_col,
        "gate_fit_fraction_requested": gate_fit_fraction,
        "spatial_buffer_m": spatial_buffer_m,
        "seed": seed,
        "event_id": event_id,
        "role_rows": {role: len(roles[role]) for role in ROLE_NAMES},
        "role_groups": {
            role: int(roles[role][group_col].nunique()) for role in ROLE_NAMES
        },
        "role_label_counts": {
            role: {
                str(label): int(count)
                for label, count in roles[role][label_col].value_counts().sort_index().items()
            }
            for role in ROLE_NAMES
        },
        "buffer_exclusion_rows": len(exclusions),
        "buffer_exclusion_counts": {
            str(role): int(count)
            for role, count in exclusions.get(
                "excluded_from", pd.Series(dtype=object)
            ).value_counts().items()
        },
    }
    (output_dir / "protocol_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    validation = _read_source_manifest(split_dir / "val.csv")
    final_test = _read_source_manifest(split_dir / "test.csv")
    event_id = args.event_id
    if not event_id:
        for candidate in ("event_id", "event", "dataset"):
            if candidate in validation and candidate in final_test:
                values = set(validation[candidate].dropna().astype(str)) | set(
                    final_test[candidate].dropna().astype(str)
                )
                if len(values) == 1:
                    event_id = values.pop()
                    break
    if not event_id or not str(event_id).strip():
        raise ValueError(
            "A stable --event-id is required because the source manifests do not "
            "contain one unique event identifier"
        )
    event_id = str(event_id).strip()
    roles, exclusions = build_protocol_roles(
        validation,
        final_test,
        group_col=args.group_col,
        label_col=args.label_col,
        gate_fit_fraction=args.gate_fit_fraction,
        spatial_buffer_m=args.spatial_buffer_m,
        seed=args.seed,
        event_id=event_id,
    )
    summary = write_protocol(
        roles,
        exclusions,
        Path(args.output_dir),
        source_split_dir=split_dir,
        group_col=args.group_col,
        label_col=args.label_col,
        gate_fit_fraction=args.gate_fit_fraction,
        spatial_buffer_m=args.spatial_buffer_m,
        seed=args.seed,
        event_id=event_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
