from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.labels import MILTON_HURRICANE_CATEGORIES, build_class_mapping


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
        if verify_images and not (_is_readable_image(street_path) and _is_readable_image(remote_path)):
            continue
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
                "pre_street_view_path": str(pre_street_path),
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
    dataset_root = Path(args.dataset_root)
    source_csv = dataset_root / args.source_csv
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_csv)
    manifest = _add_manifest_columns(df, dataset_root, verify_images=not args.no_verify_images)
    if manifest.empty:
        raise RuntimeError("No usable Milton rows were found. Check the dataset root and source CSV.")

    train_source = manifest[manifest["split_source"].astype(str).str.lower() == "train"].copy()
    test_df = manifest[manifest["split_source"].astype(str).str.lower() == "test"].copy()
    if train_source.empty or test_df.empty:
        raise RuntimeError("Milton source CSV must contain train and test rows in the 'set' column.")

    train_df, val_df = _stratified_val_split(train_source, args.val_fraction, args.seed)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(name, len(frame), frame["category"].value_counts().to_dict())
    print(f"Wrote manifests to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
