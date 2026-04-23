from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create grouped train/val/test splits from a manifest CSV.")
    parser.add_argument("--manifest-csv", required=True, help="Input manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory to write train/val/test CSV files.")
    parser.add_argument("--group-col", default="objectid", help="Grouping column used to avoid leakage.")
    parser.add_argument("--label-col", default="binary_label", help="Label column used for approximate stratification.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
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


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    usable = df[df[args.label_col].notna()].copy()
    usable["group_id"] = usable[args.group_col].fillna(usable.get("sample_id")).astype(str)

    group_rows = []
    for group_id, group in usable.groupby("group_id", sort=False):
        group_rows.append(
            {
                "group_id": group_id,
                "label": int(group[args.label_col].mode(dropna=True).iloc[0]),
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
        print(split_name, len(split_df), split_df[args.label_col].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
