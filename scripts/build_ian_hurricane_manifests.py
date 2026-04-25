from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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
        description="Build CrossViewConflict-style binary manifests from the IAN hurricane cross-view dataset."
    )
    parser.add_argument("--dataset-root", required=True, help="Path to the IAN_hurricane dataset root.")
    parser.add_argument("--output-dir", required=True, help="Directory to write train/val/test manifest CSVs.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation fraction from the provided train split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--positive-scheme",
        choices=["severe_only", "moderate_and_severe"],
        default="severe_only",
        help="Binary mapping for the positive class.",
    )
    return parser.parse_args()


def _normalize_paths(df: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    frame = df.copy()
    frame["street_view_path"] = frame["svi_path"].map(lambda value: str(image_root / Path(value).name))
    frame["remote_sensing_path"] = frame["sat_path"].map(lambda value: str(image_root / Path(value).name))
    return frame


def _binary_mapping(positive_scheme: str) -> tuple[dict[str, int], dict[str, str]]:
    if positive_scheme == "moderate_and_severe":
        severity_to_binary = {
            "0_MinorDamage": 0,
            "1_ModerateDamage": 1,
            "2_SevereDamage": 1,
        }
        severity_to_name = {
            "0_MinorDamage": "minor",
            "1_ModerateDamage": "moderate_or_severe",
            "2_SevereDamage": "moderate_or_severe",
        }
        return severity_to_binary, severity_to_name
    return DEFAULT_SEVERITY_TO_BINARY, DEFAULT_SEVERITY_TO_NAME


def _filter_binary(df: pd.DataFrame, positive_scheme: str) -> pd.DataFrame:
    severity_to_binary, severity_to_name = _binary_mapping(positive_scheme)
    frame = df[df["severity"].isin(severity_to_binary)].copy()
    frame["binary_label"] = frame["severity"].map(severity_to_binary).astype(int)
    frame["binary_name"] = frame["severity"].map(severity_to_name)
    frame["category"] = frame["severity"]
    return frame.reset_index(drop=True)


def _add_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy().reset_index(drop=True)
    frame["sample_id"] = frame["street_view_path"].map(lambda value: Path(value).stem.replace("_svi", ""))
    frame["objectid"] = range(len(frame))
    frame["latitude"] = pd.NA
    frame["longitude"] = pd.NA
    frame["remote_tile_filename"] = ""
    ordered = [
        "sample_id",
        "objectid",
        "category",
        "binary_label",
        "binary_name",
        "latitude",
        "longitude",
        "remote_tile_filename",
        "street_view_path",
        "remote_sensing_path",
        "severity",
    ]
    return frame[ordered]


def _stratified_val_split(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("binary_label", sort=True):
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
    image_root = dataset_root / "images"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_split = pd.read_csv(dataset_root / "train_split.csv")
    test_split = pd.read_csv(dataset_root / "test_split.csv")

    train_split = _add_manifest_columns(_filter_binary(_normalize_paths(train_split, image_root), args.positive_scheme))
    test_split = _add_manifest_columns(_filter_binary(_normalize_paths(test_split, image_root), args.positive_scheme))
    train_df, val_df = _stratified_val_split(train_split, args.val_fraction, args.seed)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_split.to_csv(test_path, index=False)

    for name, frame in [("train", train_df), ("val", val_df), ("test", test_split)]:
        print(name, len(frame), frame["category"].value_counts().to_dict())
    print(f"Wrote manifests to {output_dir}")


if __name__ == "__main__":
    main()
