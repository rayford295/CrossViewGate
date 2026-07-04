from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap and paired tests for conflict-subset gains.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--conflict-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _canonical_sample_id(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text


def _target_column(df: pd.DataFrame) -> str:
    for candidate in ("label", "binary_label", "street_target", "target"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No target column found.")


def _bootstrap_delta_ci(
    a_correct: np.ndarray,
    b_correct: np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    diff = a_correct.astype(np.float64) - b_correct.astype(np.float64)
    observed = float(diff.mean()) if len(diff) else float("nan")
    if len(diff) <= 1:
        return {"delta": observed, "ci_low": observed, "ci_high": observed}
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    indices = np.arange(len(diff))
    for idx in range(resamples):
        sample_indices = rng.choice(indices, size=len(indices), replace=True)
        draws[idx] = diff[sample_indices].mean()
    alpha = 1.0 - confidence_level
    low, high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"delta": observed, "ci_low": float(low), "ci_high": float(high)}


def _paired_sign_flip(a_correct: np.ndarray, b_correct: np.ndarray, permutations: int, seed: int) -> dict[str, float]:
    diff = a_correct.astype(np.float64) - b_correct.astype(np.float64)
    observed = float(diff.mean()) if len(diff) else float("nan")
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return {"delta": observed, "p_value_two_sided": 1.0}
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for idx in range(permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(nonzero), replace=True)
        null[idx] = float((nonzero * signs).sum() / len(diff))
    p_value = float((np.abs(null) >= abs(observed)).mean())
    return {"delta": observed, "p_value_two_sided": p_value}


def _mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict[str, float | int]:
    b = int(((a_correct == 1) & (b_correct == 0)).sum())
    c = int(((a_correct == 0) & (b_correct == 1)).sum())
    n = b + c
    if n == 0:
        return {"a_only_correct": b, "b_only_correct": c, "p_value_two_sided": 1.0}
    try:
        from scipy.stats import binomtest

        p_value = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        z = (abs(b - c) - 1.0) / math.sqrt(max(n, 1))
        p_value = float(math.erfc(max(z, 0.0) / math.sqrt(2.0)))
    return {"a_only_correct": b, "b_only_correct": c, "p_value_two_sided": p_value}


def _comparison(
    name: str,
    cross_correct: np.ndarray,
    other_correct: np.ndarray,
    *,
    resamples: int,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    return {
        "comparison": f"crossview_vs_{name}",
        "crossview_accuracy": float(cross_correct.mean()) if len(cross_correct) else None,
        f"{name}_accuracy": float(other_correct.mean()) if len(other_correct) else None,
        "bootstrap_delta_ci": _bootstrap_delta_ci(cross_correct, other_correct, resamples=resamples, seed=seed),
        "paired_sign_flip": _paired_sign_flip(cross_correct, other_correct, permutations=permutations, seed=seed),
        "mcnemar": _mcnemar(cross_correct.astype(int), other_correct.astype(int)),
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.conflict_csv)
    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].map(_canonical_sample_id)
    target_col = _target_column(df)
    target = df[target_col].to_numpy(dtype=np.int64)
    street_correct = (df["street_prediction"].to_numpy(dtype=np.int64) == target).astype(np.int32)
    remote_correct = (df["remote_prediction"].to_numpy(dtype=np.int64) == target).astype(np.int32)
    cross_correct = (df["crossview_prediction"].to_numpy(dtype=np.int64) == target).astype(np.int32)
    concat_correct = None
    if "concat_prediction" in df.columns:
        concat_correct = (df["concat_prediction"].to_numpy(dtype=np.int64) == target).astype(np.int32)

    best_single_name = "street" if street_correct.mean() >= remote_correct.mean() else "remote"
    best_named_correct = street_correct if best_single_name == "street" else remote_correct
    oracle_single_correct = np.maximum(street_correct, remote_correct)

    comparisons = [
        _comparison(
            "street",
            cross_correct,
            street_correct,
            resamples=args.bootstrap_resamples,
            permutations=args.permutations,
            seed=args.seed,
        ),
        _comparison(
            "remote",
            cross_correct,
            remote_correct,
            resamples=args.bootstrap_resamples,
            permutations=args.permutations,
            seed=args.seed + 1,
        ),
        _comparison(
            f"best_named_single_{best_single_name}",
            cross_correct,
            best_named_correct,
            resamples=args.bootstrap_resamples,
            permutations=args.permutations,
            seed=args.seed + 2,
        ),
        _comparison(
            "oracle_single_upper_bound",
            cross_correct,
            oracle_single_correct,
            resamples=args.bootstrap_resamples,
            permutations=args.permutations,
            seed=args.seed + 3,
        ),
    ]
    if concat_correct is not None:
        comparisons.append(
            _comparison(
                "concat",
                cross_correct,
                concat_correct,
                resamples=args.bootstrap_resamples,
                permutations=args.permutations,
                seed=args.seed + 4,
            )
        )

    summary = {
        "dataset": args.dataset_name,
        "target_column": target_col,
        "num_conflict_examples": int(len(df)),
        "street_accuracy": float(street_correct.mean()) if len(df) else None,
        "remote_accuracy": float(remote_correct.mean()) if len(df) else None,
        "crossview_accuracy": float(cross_correct.mean()) if len(df) else None,
        "concat_accuracy": float(concat_correct.mean()) if concat_correct is not None and len(df) else None,
        "best_named_single": best_single_name,
        "best_named_single_accuracy": float(best_named_correct.mean()) if len(df) else None,
        "oracle_single_accuracy": float(oracle_single_correct.mean()) if len(df) else None,
        "comparisons": comparisons,
    }
    save_json(summary, args.output_json)
    if args.output_csv:
        rows = []
        for comparison in comparisons:
            row = {
                "dataset": args.dataset_name,
                "comparison": comparison["comparison"],
                "crossview_accuracy": comparison["crossview_accuracy"],
                "bootstrap_delta": comparison["bootstrap_delta_ci"]["delta"],
                "bootstrap_ci_low": comparison["bootstrap_delta_ci"]["ci_low"],
                "bootstrap_ci_high": comparison["bootstrap_delta_ci"]["ci_high"],
                "sign_flip_p": comparison["paired_sign_flip"]["p_value_two_sided"],
                "mcnemar_p": comparison["mcnemar"]["p_value_two_sided"],
            }
            rows.append(row)
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(summary)


if __name__ == "__main__":
    main()
