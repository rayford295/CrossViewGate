from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.labels import MILTON_HURRICANE_CATEGORIES, build_class_mapping
from crossview_conflict.data.cvian import (
    add_spatial_block_ids,
    assert_group_isolation,
    purge_spatial_buffer,
    split_grouped_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build original 3-class manifests from the Milton GenDisasterSVI dataset."
    )
    parser.add_argument("--dataset-root", required=True, help="Path to hurrican-milton-GenDisasterSVI.")
    parser.add_argument(
        "--source-csv",
        default="dataset_with_post_sat.csv",
        help="CSV inside the dataset root. The default includes post-disaster satellite paths.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write train/val/test manifest CSVs.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation fraction from the provided train split.")
    parser.add_argument(
        "--split-strategy",
        choices=["spatial-block", "source"],
        default="spatial-block",
        help="Use a repaired spatial holdout by default; source preserves the released split for audit.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--grouped-val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--spatial-grid-degrees",
        type=float,
        default=0.001,
        help="Milton covers a compact town; 0.001 degrees is roughly a 100 m block.",
    )
    parser.add_argument("--spatial-buffer-m", type=float, default=25.0)
    parser.add_argument(
        "--rebuild-val-from-train",
        action="store_true",
        help=(
            "Legacy audit only: ignore the released source val rows and rebuild "
            "validation from source train. The default preserves all three source splits."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-verify-images", action="store_true", help="Skip image existence/readability checks.")
    return parser.parse_args()


def _is_readable_image(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _local_image_path(dataset_root: Path, damage_level: str, raw_path: object, view: str) -> Path:
    filename = Path(str(raw_path).replace("\\", "/")).name
    if view == "street_post":
        return dataset_root / damage_level / "post" / filename
    if view == "street_pre":
        return dataset_root / damage_level / "pre" / filename
    if view == "post_sat":
        return dataset_root / "post_sat" / filename
    raise ValueError(f"Unknown view: {view}")


def _add_manifest_columns(df: pd.DataFrame, dataset_root: Path, verify_images: bool) -> pd.DataFrame:
    damage_to_label = build_class_mapping(MILTON_HURRICANE_CATEGORIES)
    records: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        damage_level = str(row["damage_level"])
        if damage_level not in damage_to_label:
            continue
        street_path = _local_image_path(dataset_root, damage_level, row["post_disaster_image_path"], "street_post")
        remote_path = _local_image_path(dataset_root, damage_level, row["post_sat_image_path"], "post_sat")
        pre_street_path = _local_image_path(dataset_root, damage_level, row["pre_disaster_image_path"], "street_pre")
        post_street_available = _is_readable_image(street_path) if verify_images else street_path.is_file()
        post_overhead_available = _is_readable_image(remote_path) if verify_images else remote_path.is_file()
        pre_street_available = _is_readable_image(pre_street_path) if verify_images else pre_street_path.is_file()
        if verify_images and not (post_street_available and post_overhead_available):
            missing = []
            if not post_street_available:
                missing.append(str(street_path))
            if not post_overhead_available:
                missing.append(str(remote_path))
            raise FileNotFoundError(
                f"Milton pair {row.get('pair_id')} is missing required readable post-event views: {missing}"
            )
        pair_id = int(row["pair_id"])
        records.append(
            {
                "sample_id": f"milton_{pair_id}",
                "objectid": pair_id,
                "group_id": str(pair_id),
                "category": damage_level,
                "label": int(damage_to_label[damage_level]),
                "label_name": damage_level,
                "latitude": float(row["lat"]) if not pd.isna(row.get("lat")) else pd.NA,
                "longitude": float(row["lon"]) if not pd.isna(row.get("lon")) else pd.NA,
                "remote_tile_filename": Path(remote_path).name,
                "street_view_path": str(street_path),
                "remote_sensing_path": str(remote_path),
                # Optional missing views use both an empty path and an
                # explicit availability flag.  Generic manifest consumers
                # must not interpret a known-missing filename as loadable.
                "pre_street_view_path": str(pre_street_path) if pre_street_available else "",
                "pre_street_view_available": pre_street_available,
                "post_street_view_available": post_street_available,
                "post_overhead_view_available": post_overhead_available,
                "split_source": str(row.get("set", "")),
                "prompt": str(row.get("prompt", "")),
                "dataset": "milton_hurricane",
            }
        )
    return pd.DataFrame.from_records(records)


def _stratified_val_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("label", sort=True):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_count = max(1, int(round(len(group) * val_fraction)))
        val_parts.append(group.iloc[:val_count].copy())
        train_parts.append(group.iloc[val_count:].copy())
    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def main() -> None:
    args = parse_args()
    if args.rebuild_val_from_train and args.split_strategy != "source":
        raise ValueError("--rebuild-val-from-train is only valid with --split-strategy source")
    dataset_root = Path(args.dataset_root)
    source_csv = dataset_root / args.source_csv
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_csv)
    manifest = _add_manifest_columns(df, dataset_root, verify_images=not args.no_verify_images)
    if manifest.empty:
        raise RuntimeError("No usable Milton rows were found. Check the dataset root and source CSV.")

    manifest["split_source"] = manifest["split_source"].astype(str).str.strip().str.lower()
    manifest = add_spatial_block_ids(manifest, args.spatial_grid_degrees)
    unexpected_splits = sorted(set(manifest["split_source"]) - {"train", "val", "test"})
    if unexpected_splits:
        raise ValueError(f"Unexpected Milton source split values: {unexpected_splits}")
    train_source = manifest[manifest["split_source"] == "train"].copy()
    source_val = manifest[manifest["split_source"] == "val"].copy()
    test_df = manifest[manifest["split_source"] == "test"].copy()
    if train_source.empty or source_val.empty or test_df.empty:
        raise RuntimeError("Milton source CSV must contain non-empty train, val, and test rows in 'set'.")

    buffer_exclusions = pd.DataFrame()
    if args.split_strategy == "spatial-block":
        manifest["group_id"] = manifest["spatial_block_id"]
        splits = split_grouped_frame(
            manifest,
            group_col="group_id",
            label_col="label",
            train_fraction=args.train_fraction,
            val_fraction=args.grouped_val_fraction,
            seed=args.seed,
        )
        splits, buffer_exclusions = purge_spatial_buffer(
            splits,
            args.spatial_buffer_m,
        )
        assert_group_isolation(splits, "spatial_block_id")
        train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
        protocol = "spatial_block_with_boundary_buffer"
    elif args.rebuild_val_from_train:
        train_df, val_df = _stratified_val_split(train_source, args.val_fraction, args.seed)
        protocol = "legacy_rebuilt_val_source_val_excluded"
    else:
        train_df = train_source.reset_index(drop=True)
        val_df = source_val.reset_index(drop=True)
        protocol = "released_source_train_val_test"
        if len(train_df) + len(val_df) + len(test_df) != len(manifest):
            raise RuntimeError("Milton source split reconstruction did not retain every manifest row")
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    buffer_exclusions.to_csv(output_dir / "buffer_exclusions.csv", index=False)

    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(name, len(frame), frame["category"].value_counts().to_dict())
    summary = {
        "dataset": "milton_hurricane",
        "source_csv": args.source_csv,
        "protocol": protocol,
        "split_strategy": args.split_strategy,
        "spatial_grid_degrees": args.spatial_grid_degrees,
        "spatial_buffer_m": args.spatial_buffer_m if args.split_strategy == "spatial-block" else 0.0,
        "buffer_exclusion_rows": int(len(buffer_exclusions)),
        "buffer_exclusion_counts": {
            str(key): int(value)
            for key, value in buffer_exclusions.get(
                "excluded_from", pd.Series(dtype=str)
            ).value_counts().items()
        },
        "source_rows": int(len(df)),
        "manifest_rows": int(len(manifest)),
        "source_split_counts": {
            str(key): int(value) for key, value in manifest["split_source"].value_counts().items()
        },
        "output_split_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "unique_pair_ids": int(manifest["objectid"].nunique()),
        "pre_street_available": int(manifest["pre_street_view_available"].sum()),
        "post_street_available": int(manifest["post_street_view_available"].sum()),
        "post_overhead_available": int(manifest["post_overhead_view_available"].sum()),
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote manifests to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
