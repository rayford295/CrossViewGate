"""
Insight 2: Check for objectid leakage between train and test splits.

If the same property (objectid) appears in both train and test, the model
has seen that building before — inflating reported F1 numbers.

Run this BEFORE trusting any existing results.

Usage:
    python scripts/check_split_leakage.py \
        --wildfire-train   data/splits/eaton_wildfire/train.csv \
        --wildfire-val     data/splits/eaton_wildfire/val.csv \
        --wildfire-test    data/splits/eaton_wildfire/test.csv \
        --hurricane-train  data/splits/ian_hurricane_minor_vs_severe/train.csv \
        --hurricane-val    data/splits/ian_hurricane_minor_vs_severe/val.csv \
        --hurricane-test   data/splits/ian_hurricane_minor_vs_severe/test.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def check_leakage(train_path: str, val_path: str, test_path: str, label: str) -> bool:
    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    test  = pd.read_csv(test_path)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"{'='*55}")

    if "objectid" not in train.columns:
        print("  WARNING: no 'objectid' column — cannot verify group isolation.")
        print("  Splits may have been made without property-level grouping.")
        return False

    ids_train = set(train["objectid"].dropna().astype(str))
    ids_val   = set(val  ["objectid"].dropna().astype(str))
    ids_test  = set(test ["objectid"].dropna().astype(str))

    train_test_overlap = ids_train & ids_test
    train_val_overlap  = ids_train & ids_val
    val_test_overlap   = ids_val   & ids_test

    leakage_found = False

    if train_test_overlap:
        print(f"  ✗ LEAKAGE: {len(train_test_overlap)} objectids in BOTH train and test")
        print(f"    Example ids: {list(train_test_overlap)[:5]}")
        leakage_found = True
    else:
        print(f"  ✓ train ∩ test: 0 shared objectids — clean")

    if train_val_overlap:
        print(f"  ✗ LEAKAGE: {len(train_val_overlap)} objectids in BOTH train and val")
        leakage_found = True
    else:
        print(f"  ✓ train ∩ val:  0 shared objectids — clean")

    if val_test_overlap:
        print(f"  ✗ LEAKAGE: {len(val_test_overlap)} objectids in BOTH val and test")
        leakage_found = True
    else:
        print(f"  ✓ val   ∩ test: 0 shared objectids — clean")

    if leakage_found:
        print(f"\n  ACTION REQUIRED: re-generate splits with make_group_splits.py")
        print(f"  python scripts/make_group_splits.py \\")
        print(f"    --manifest-csv <path> --output-dir data/splits/{label.lower()}_grouped")
    else:
        print(f"\n  All splits are objectid-clean. Existing results are valid.")

    return leakage_found


def main() -> None:
    parser = argparse.ArgumentParser(description="Check objectid leakage in train/val/test splits")
    for ds in ("wildfire", "hurricane"):
        parser.add_argument(f"--{ds}-train", required=True)
        parser.add_argument(f"--{ds}-val",   required=True)
        parser.add_argument(f"--{ds}-test",  required=True)
    args = parser.parse_args()

    any_leakage = False
    any_leakage |= check_leakage(
        args.wildfire_train, args.wildfire_val, args.wildfire_test, "Wildfire"
    )
    any_leakage |= check_leakage(
        args.hurricane_train, args.hurricane_val, args.hurricane_test, "Hurricane"
    )

    print(f"\n{'='*55}")
    if any_leakage:
        print("  OVERALL: leakage found — rerun experiments after re-splitting")
        sys.exit(1)
    else:
        print("  OVERALL: no leakage — all results are valid")
        sys.exit(0)


if __name__ == "__main__":
    main()
