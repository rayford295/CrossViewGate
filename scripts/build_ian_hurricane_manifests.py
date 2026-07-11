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
    CVIAN_CHECKSUM_RELATIVE_PATH,
    CVIAN_POSITION_RELATIVE_PATH,
    CVIAN_POSITION_SHA512,
    add_spatial_block_ids,
    assert_group_isolation,
    build_cvi_an_image_id_map,
    georeference_cvi_an_pairs,
    load_cvi_an_positions,
    purge_spatial_buffer,
    sha512_file,
    split_grouped_frame,
)
from crossview_conflict.labels import IAN_HURRICANE_CATEGORIES, build_class_mapping


DEFAULT_SEVERITY_TO_BINARY = {
    "0_MinorDamage": 0,
    "2_SevereDamage": 1,
}

DEFAULT_SEVERITY_TO_NAME = {
    "0_MinorDamage": "minor",
    "2_SevereDamage": "severe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build georeferenced CVIAN manifests. The main protocol uses spatial "
            "blocks; source preserves the legacy split only for an explicit audit."
        )
    )
    parser.add_argument("--dataset-root", required=True, help="Path to the IAN_hurricane dataset root.")
    parser.add_argument("--output-dir", required=True, help="Directory to write train/val/test manifest CSVs.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Legacy source strategy only.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Grouped strategies only.")
    parser.add_argument("--grouped-val-fraction", type=float, default=0.1, help="Grouped strategies only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task",
        choices=["original", "binary"],
        default="original",
        help="Use original 3-class severity labels by default; binary preserves the legacy collapsed task.",
    )
    parser.add_argument(
        "--positive-scheme",
        choices=["severe_only", "moderate_and_severe"],
        default="severe_only",
        help="Binary mapping for the positive class when --task binary.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["spatial-block", "sequence", "source"],
        default="spatial-block",
        help=(
            "spatial-block is the leakage-resistant main protocol; sequence is a "
            "sensitivity protocol; source reproduces the legacy leaked split."
        ),
    )
    parser.add_argument("--spatial-grid-degrees", type=float, default=0.005)
    parser.add_argument(
        "--spatial-buffer-m",
        type=float,
        default=25.0,
        help=(
            "For spatial-block splits, remove lower-priority boundary samples "
            "within this distance of a held-out split (test > val > train)."
        ),
    )
    parser.add_argument(
        "--georeferenced-pairs",
        help="Optional precomputed pairs_georeferenced.csv from georeference_ian_hurricane.py.",
    )
    parser.add_argument("--position-geojson", help="Official CVIAN position GeoJSON.")
    parser.add_argument("--checksums", help="Official checksums.sha512 used to recover renamed image ids.")
    parser.add_argument("--image-id-map", help="Optional precomputed image_id_map.csv.")
    parser.add_argument(
        "--skip-position-checksum",
        action="store_true",
        help="Allow a position file other than the currently recorded official version.",
    )
    return parser.parse_args()


def _normalize_paths(df: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    frame = df.copy()
    frame["street_view_path"] = frame["svi_path"].map(lambda value: str(image_root / Path(str(value)).name))
    frame["remote_sensing_path"] = frame["sat_path"].map(lambda value: str(image_root / Path(str(value)).name))
    return frame


def _binary_mapping(positive_scheme: str) -> tuple[dict[str, int], dict[str, str]]:
    if positive_scheme == "moderate_and_severe":
        return (
            {
                "0_MinorDamage": 0,
                "1_ModerateDamage": 1,
                "2_SevereDamage": 1,
            },
            {
                "0_MinorDamage": "minor",
                "1_ModerateDamage": "moderate_or_severe",
                "2_SevereDamage": "moderate_or_severe",
            },
        )
    return DEFAULT_SEVERITY_TO_BINARY, DEFAULT_SEVERITY_TO_NAME


def _add_original_labels(df: pd.DataFrame) -> pd.DataFrame:
    severity_to_label = build_class_mapping(IAN_HURRICANE_CATEGORIES)
    frame = df[df["severity"].isin(severity_to_label)].copy()
    frame["category"] = frame["severity"]
    frame["label"] = frame["severity"].map(severity_to_label).astype(int)
    frame["label_name"] = frame["severity"]
    return frame.reset_index(drop=True)


def _add_binary_labels(df: pd.DataFrame, positive_scheme: str) -> pd.DataFrame:
    severity_to_binary, severity_to_name = _binary_mapping(positive_scheme)
    frame = df[df["severity"].isin(severity_to_binary)].copy()
    frame["binary_label"] = frame["severity"].map(severity_to_binary).astype(int)
    frame["binary_name"] = frame["severity"].map(severity_to_name)
    frame["category"] = frame["severity"]
    return frame.reset_index(drop=True)


def _manifest_columns(df: pd.DataFrame, task: str, group_col: str) -> pd.DataFrame:
    frame = df.copy().reset_index(drop=True)
    frame["sample_id"] = frame["sample_id"].astype(str).str.zfill(6)
    frame["mapillary_id"] = frame["mapillary_id"].astype(str)
    frame["objectid"] = frame["mapillary_id"]
    frame["group_id"] = frame[group_col].astype(str)
    frame["remote_tile_filename"] = ""
    frame["position_source"] = "CVIAN DOI 10.14459/2024mp1749324"
    ordered = [
        "sample_id",
        "mapillary_id",
        "objectid",
        "group_id",
        "sequence_id",
        "spatial_block_id",
        "tile_id",
        "category",
    ]
    if task == "original":
        ordered += ["label", "label_name"]
    else:
        ordered += ["binary_label", "binary_name"]
    ordered += [
        "latitude",
        "longitude",
        "captured_at_ms",
        "captured_at_utc",
        "compass_angle_deg",
        "creator_id",
        "is_pano",
        "remote_tile_filename",
        "street_view_path",
        "remote_sensing_path",
        "severity",
        "position_source",
    ]
    optional = ["original_split", "sat_sha512", "svi_sha512"]
    ordered += [column for column in optional if column in frame]
    missing = [column for column in ordered if column not in frame]
    if missing:
        raise KeyError(f"Manifest columns missing after georeference: {missing}")
    return frame[ordered]


def _stratified_val_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for _, group in df.groupby(label_col, sort=True):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_count = max(1, int(round(len(group) * val_fraction)))
        val_parts.append(group.iloc[:val_count].copy())
        train_parts.append(group.iloc[val_count:].copy())
    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def _local_sample_ids(frame: pd.DataFrame) -> pd.Series:
    sat = frame["sat_path"].map(lambda value: Path(str(value)).stem.removesuffix("_sat"))
    svi = frame["svi_path"].map(lambda value: Path(str(value)).stem.removesuffix("_svi"))
    if not sat.equals(svi):
        raise ValueError("sat_path and svi_path sample ids disagree")
    return sat.astype(str).str.zfill(6)


def _read_csv_ids(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["sample_id"] = _local_sample_ids(frame)
    return frame


def _load_georeferenced_pairs(args: argparse.Namespace, dataset_root: Path) -> tuple[pd.DataFrame, str]:
    default_sidecar = dataset_root / "georeferenced" / "pairs_georeferenced.csv"
    sidecar = Path(args.georeferenced_pairs).resolve() if args.georeferenced_pairs else default_sidecar
    if sidecar.is_file():
        frame = pd.read_csv(
            sidecar,
            dtype={"sample_id": str, "mapillary_id": str, "creator_id": str},
        )
        frame["sample_id"] = frame["sample_id"].str.zfill(6)
        # A georeference sidecar may already contain block ids generated with
        # a different grid.  Always recompute from coordinates so the CLI
        # parameter, manifests, and summary provenance cannot disagree.
        frame = add_spatial_block_ids(frame, args.spatial_grid_degrees)
        return frame, str(sidecar)

    pairs = pd.read_csv(dataset_root / "pairs.csv")
    position_path = (
        Path(args.position_geojson).resolve()
        if args.position_geojson
        else dataset_root / CVIAN_POSITION_RELATIVE_PATH
    )
    checksums_path = (
        Path(args.checksums).resolve()
        if args.checksums
        else dataset_root / CVIAN_CHECKSUM_RELATIVE_PATH
    )
    if not position_path.is_file():
        raise FileNotFoundError(
            f"Missing official position file: {position_path}. Run scripts/georeference_ian_hurricane.py first."
        )
    if not args.skip_position_checksum:
        actual = sha512_file(position_path)
        if actual != CVIAN_POSITION_SHA512:
            raise ValueError(f"Unexpected CVIAN position SHA-512: {actual}")
    positions = load_cvi_an_positions(position_path)
    if args.image_id_map:
        image_id_map = pd.read_csv(
            args.image_id_map,
            dtype={"sample_id": str, "mapillary_id": str},
        )
        image_id_map["sample_id"] = image_id_map["sample_id"].str.zfill(6)
    else:
        if not checksums_path.is_file():
            raise FileNotFoundError(
                f"Missing official checksums: {checksums_path}. They are required for an auditable id join."
            )
        image_id_map = build_cvi_an_image_id_map(dataset_root, pairs, checksums_path)
    frame = georeference_cvi_an_pairs(
        pairs,
        image_id_map,
        positions,
        grid_degrees=args.spatial_grid_degrees,
    )
    return frame, "in-memory exact SHA-512 join"


def _attach_original_split(frame: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    split_rows: list[pd.DataFrame] = []
    for name, filename in (("train", "train_split.csv"), ("test", "test_split.csv")):
        split = _read_csv_ids(dataset_root / filename)
        split["original_split"] = name
        split_rows.append(split[["sample_id", "original_split"]])
    split_map = pd.concat(split_rows, ignore_index=True)
    if split_map["sample_id"].duplicated().any():
        raise ValueError("Original train/test CSVs contain duplicate sample ids")
    existing = frame.drop(columns=["original_split"], errors="ignore")
    result = existing.merge(split_map, on="sample_id", how="left", validate="one_to_one")
    if result["original_split"].isna().any():
        raise ValueError("Not every pair belongs to the original train/test split")
    return result


def _overlap_count(splits: dict[str, pd.DataFrame], column: str) -> int:
    names = list(splits)
    overlap: set[str] = set()
    for left_index, left_name in enumerate(names):
        left = set(splits[left_name][column].dropna().astype(str))
        for right_name in names[left_index + 1 :]:
            overlap |= left & set(splits[right_name][column].dropna().astype(str))
    return len(overlap)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    image_root = dataset_root / "images"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    georeferenced, georeference_source = _load_georeferenced_pairs(args, dataset_root)
    georeferenced["sample_id"] = georeferenced["sample_id"].astype(str).str.zfill(6)
    georeferenced = _attach_original_split(georeferenced, dataset_root)
    georeferenced = _normalize_paths(georeferenced, image_root)
    if args.task == "original":
        label_col = "label"
        labeled = _add_original_labels(georeferenced)
    else:
        label_col = "binary_label"
        labeled = _add_binary_labels(georeferenced, args.positive_scheme)

    buffer_exclusions = pd.DataFrame()
    if args.split_strategy == "source":
        source_train = labeled[labeled["original_split"] == "train"].copy()
        source_test = labeled[labeled["original_split"] == "test"].copy()
        source_train = _manifest_columns(source_train, args.task, "sequence_id")
        source_test = _manifest_columns(source_test, args.task, "sequence_id")
        train_df, val_df = _stratified_val_split(
            source_train,
            args.val_fraction,
            args.seed,
            label_col,
        )
        splits = {"train": train_df, "val": val_df, "test": source_test.reset_index(drop=True)}
    else:
        group_col = "spatial_block_id" if args.split_strategy == "spatial-block" else "sequence_id"
        manifest = _manifest_columns(labeled, args.task, group_col)
        splits = split_grouped_frame(
            manifest,
            group_col="group_id",
            label_col=label_col,
            train_fraction=args.train_fraction,
            val_fraction=args.grouped_val_fraction,
            seed=args.seed,
        )
        assert_group_isolation(splits, "group_id")
        if args.split_strategy == "spatial-block" and args.spatial_buffer_m > 0:
            splits, buffer_exclusions = purge_spatial_buffer(
                splits,
                args.spatial_buffer_m,
            )
            assert_group_isolation(splits, "group_id")

    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
        print(name, len(frame), frame[label_col].value_counts().sort_index().to_dict())
    buffer_exclusions.to_csv(output_dir / "buffer_exclusions.csv", index=False)
    if not buffer_exclusions.empty:
        print("buffer exclusions", len(buffer_exclusions))

    summary = {
        "dataset": "CVIAN",
        "task": args.task,
        "split_strategy": args.split_strategy,
        "seed": args.seed,
        "spatial_grid_degrees": args.spatial_grid_degrees,
        "spatial_buffer_m": args.spatial_buffer_m if args.split_strategy == "spatial-block" else 0.0,
        "buffer_exclusion_rows": int(len(buffer_exclusions)),
        "buffer_exclusion_counts": {
            str(key): int(value)
            for key, value in buffer_exclusions.get(
                "excluded_from", pd.Series(dtype=str)
            ).value_counts().items()
        },
        "georeference_source": georeference_source,
        "total_rows": int(sum(len(frame) for frame in splits.values())),
        "split_rows": {name: int(len(frame)) for name, frame in splits.items()},
        "split_label_counts": {
            name: {str(key): int(value) for key, value in frame[label_col].value_counts().items()}
            for name, frame in splits.items()
        },
        "unique_mapillary_ids": int(labeled["mapillary_id"].nunique()),
        "unique_sequences": int(labeled["sequence_id"].nunique()),
        "unique_spatial_blocks": int(labeled["spatial_block_id"].nunique()),
        "sequence_overlap_groups": _overlap_count(splits, "sequence_id"),
        "spatial_block_overlap_groups": _overlap_count(splits, "spatial_block_id"),
        "active_group_overlap_groups": _overlap_count(splits, "group_id"),
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote manifests to {output_dir}")


if __name__ == "__main__":
    main()
