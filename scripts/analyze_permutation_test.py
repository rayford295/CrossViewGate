from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run permutation tests on conflict-subset crossview performance."
    )
    parser.add_argument("--wildfire-csv", required=True)
    parser.add_argument("--hurricane-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _canonical_sample_id(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text


def _permutation_test_against_label_independence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    observed = float((y_true == y_pred).mean())
    null = np.empty(permutations, dtype=np.float64)
    for idx in range(permutations):
        shuffled = rng.permutation(y_true)
        null[idx] = (shuffled == y_pred).mean()
    p_value = float((null >= observed).mean())
    return {
        "observed_accuracy": observed,
        "null_mean": float(null.mean()),
        "p_value": p_value,
    }


def _paired_sign_flip_test(a_correct: np.ndarray, b_correct: np.ndarray, permutations: int, seed: int) -> dict[str, float]:
    diff = a_correct.astype(np.float64) - b_correct.astype(np.float64)
    observed = float(diff.mean())
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return {"observed_delta": observed, "p_value": 1.0}
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for idx in range(permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(nonzero), replace=True)
        flipped = nonzero * signs
        null[idx] = flipped.mean()
    p_value = float((np.abs(null) >= abs(observed)).mean())
    return {
        "observed_delta": observed,
        "p_value_two_sided": p_value,
    }


def _analyze_dataset(name: str, csv_path: str | Path, permutations: int, seed: int) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    df["sample_id"] = df["sample_id"].map(_canonical_sample_id)
    target = df["binary_label"].to_numpy()
    street_pred = df["street_prediction"].to_numpy()
    remote_pred = df["remote_prediction"].to_numpy()
    cross_pred = df["crossview_prediction"].to_numpy()
    street_correct = (street_pred == target).astype(np.int32)
    remote_correct = (remote_pred == target).astype(np.int32)
    cross_correct = (cross_pred == target).astype(np.int32)
    best_single = np.maximum(street_correct, remote_correct)

    return {
        "dataset": name,
        "num_examples": int(len(df)),
        "street_accuracy": float(street_correct.mean()),
        "remote_accuracy": float(remote_correct.mean()),
        "crossview_accuracy": float(cross_correct.mean()),
        "best_single_accuracy": float(best_single.mean()),
        "crossview_vs_label_independence": _permutation_test_against_label_independence(
            target,
            cross_pred,
            permutations,
            seed,
        ),
        "crossview_vs_street": _paired_sign_flip_test(cross_correct, street_correct, permutations, seed),
        "crossview_vs_remote": _paired_sign_flip_test(cross_correct, remote_correct, permutations, seed),
        "crossview_vs_best_single": _paired_sign_flip_test(cross_correct, best_single, permutations, seed),
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    summary = {
        "wildfire_conflict": _analyze_dataset("wildfire_conflict", args.wildfire_csv, args.permutations, args.seed),
        "hurricane_conflict": _analyze_dataset("hurricane_conflict", args.hurricane_csv, args.permutations, args.seed),
    }
    save_json(summary, output_dir / "permutation_test_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
