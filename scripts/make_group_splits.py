from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.labels import to_multiclass_label, to_multiclass_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create grouped train/val/test splits from a manifest CSV.")
    parser.add_argument("--manifest-csv", required=True, help="Input manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory to write train/val/test CSV files.")
    parser.add_argument("--group-col", default="objectid", help="Grouping column used to avoid leakage.")
    parser.add_argument(
        "--label-col",
        default="auto",
        help="Label column used for approximate stratification. Defaults to 'label' when present, else 'binary_label'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--path-rewrite-from", help="Optional path prefix to replace before writing splits.")
    parser.add_argument("--path-rewrite-to", help="Replacement path prefix used with --path-rewrite-from.")
    parser.add_argument("--require-existing-images", action="store_true", help="Drop rows whose street or remote image is missing.")
    return parser.parse_args()


def _split_groups(group_df: pd.DataFrame, train_frac: float, val_frac: float) -> tuple[list[object], list[object], list[object]]:
    n = len(group_df)
    train_n = max(1, int(round(n * train_frac)))
    val_n = max(1, int(round(n * val_frac)))
    if train_n + val_n >= n:
        if n >= 3:
            train_n = max(1, n - 2)
            val_n = 1
        elif n == 2:
            train_n, val_n = 1, 0
        else:
            train_n, val_n = 1, 0
    test_n = max(0, n - train_n - val_n)
    if test_n == 0 and n >= 3:
        train_n = max(1, train_n - 1)
        test_n = 1
    groups = group_df["group_id"].tolist()
    train = groups[:train_n]
    val = groups[train_n : train_n + val_n]
    test = groups[train_n + val_n :]
    return train, val, test


def _rewrite_paths(df: pd.DataFrame, source_prefix: str | None, target_prefix: str | None) -> pd.DataFrame:
    if not source_prefix or not target_prefix:
        return df
    frame = df.copy()
    path_columns = [
        "street_view_path",
        "remote_sensing_path",
        "generated_street_path",
        "pre_street_view_path",
    ]
    source_prefix = source_prefix.rstrip("\\/")
    target_prefix = target_prefix.rstrip("\\/")
    for column in path_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.replace(source_prefix, target_prefix, regex=False)
    return frame


def _filter_existing_images(df: pd.DataFrame) -> pd.DataFrame:
    required = ["street_view_path", "remote_sensing_path"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Cannot require images because columns are missing: {missing}")
    mask = df["street_view_path"].map(lambda value: Path(str(value)).exists())
    mask &= df["remote_sensing_path"].map(lambda value: Path(str(value)).exists())
    return df[mask].copy()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    df = _rewrite_paths(df, args.path_rewrite_from, args.path_rewrite_to)
    if args.require_existing_images:
        before_count = len(df)
        df = _filter_existing_images(df)
        print("filtered_missing_images", before_count - len(df))
    label_col = args.label_col
    if label_col == "auto":
        if "label" in df.columns:
            label_col = "label"
        elif "category" in df.columns:
            try:
                df["label"] = df["category"].map(to_multiclass_label).astype(int)
                df["label_name"] = df["category"].map(to_multiclass_name)
                label_col = "label"
            except (KeyError, ValueError):
                if "binary_label" not in df.columns:
                    raise
                label_col = "binary_label"
        elif "binary_label" in df.columns:
            label_col = "binary_label"
        else:
            raise KeyError("Manifest must include either 'label' or 'binary_label'.")
    usable = df[df[label_col].notna()].copy()
    usable["group_id"] = usable[args.group_col].fillna(usable.get("sample_id")).astype(str)

    group_rows = []
    for group_id, group in usable.groupby("group_id", sort=False):
        group_rows.append(
            {
                "group_id": group_id,
                "label": int(group[label_col].mode(dropna=True).iloc[0]),
            }
        )
    groups = pd.DataFrame(group_rows)

    train_groups: list[object] = []
    val_groups: list[object] = []
    test_groups: list[object] = []
    for _, label_groups in groups.groupby("label", sort=True):
        label_groups = label_groups.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        train, val, test = _split_groups(label_groups, args.train_frac, args.val_frac)
        train_groups.extend(train)
        val_groups.extend(val)
        test_groups.extend(test)

    split_map = {
        "train": set(train_groups),
        "val": set(val_groups),
        "test": set(test_groups),
    }
    for split_name, group_ids in split_map.items():
        split_df = usable[usable["group_id"].isin(group_ids)].copy()
        split_df.to_csv(output_dir / f"{split_name}.csv", index=False)
        print(split_name, len(split_df), split_df[label_col].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
