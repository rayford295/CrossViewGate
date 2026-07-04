from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import ensure_dir, save_json


def _canonical_sample_id(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CrossViewConflict triage conflict subset from per-view predictions.")
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--street-preds-csv", required=True)
    parser.add_argument("--remote-preds-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--crossview-preds-csv", help="Optional crossview predictions for conflict-resolution accuracy.")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_predictions(path: str | Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sample_id", "target", "prediction"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    if "probability" not in df.columns and "confidence" in df.columns:
        df["probability"] = df["confidence"]
    df["sample_id"] = df["sample_id"].map(_canonical_sample_id)
    rename_columns = {
        "target": f"{prefix}_target",
        "target_name": f"{prefix}_target_name",
        "probability": f"{prefix}_probability",
        "confidence": f"{prefix}_confidence",
        "prediction": f"{prefix}_prediction",
        "prediction_name": f"{prefix}_prediction_name",
        "logit": f"{prefix}_logit",
    }
    renamed = df.rename(columns={column: renamed for column, renamed in rename_columns.items() if column in df.columns})
    return renamed


def _accuracy(frame: pd.DataFrame, column: str, target_column: str) -> float | None:
    if frame.empty or column not in frame or target_column not in frame:
        return None
    return float((frame[column] == frame[target_column]).mean())


def _bootstrap_accuracy_ci(
    frame: pd.DataFrame,
    column: str,
    target_column: str,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float] | None:
    if frame.empty or column not in frame or target_column not in frame:
        return None

    correct = (frame[column].to_numpy() == frame[target_column].to_numpy()).astype(float)
    mean = float(correct.mean())
    if len(correct) == 0:
        return None
    if len(correct) == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean}

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for idx in range(n_resamples):
        sample = rng.choice(correct, size=len(correct), replace=True)
        draws[idx] = sample.mean()

    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean": mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _target_distribution(frame: pd.DataFrame, target_column: str) -> dict[str, int]:
    if frame.empty or target_column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[target_column].value_counts(dropna=False).to_dict().items()}


def _conflict_type(row: pd.Series) -> str:
    street_name = row.get("street_prediction_name")
    remote_name = row.get("remote_prediction_name")
    if isinstance(street_name, str) and isinstance(remote_name, str):
        return f"{street_name}_vs_{remote_name}"
    street_prediction = row["street_prediction"]
    remote_prediction = row["remote_prediction"]
    if street_prediction > remote_prediction:
        return "street_higher_remote_lower"
    if street_prediction < remote_prediction:
        return "street_lower_remote_higher"
    return "agreement"


def main() -> None:
    args = parse_args()
    split_df = pd.read_csv(args.split_csv)
    split_df["sample_id"] = split_df["sample_id"].map(_canonical_sample_id)
    metadata_columns = [
        "sample_id",
        "objectid",
        "category",
        "label",
        "label_name",
        "binary_label",
        "binary_name",
        "latitude",
        "longitude",
        "remote_tile_filename",
        "street_view_path",
        "remote_sensing_path",
    ]
    metadata = split_df[[column for column in metadata_columns if column in split_df.columns]].copy()

    street = _load_predictions(args.street_preds_csv, "street")
    remote = _load_predictions(args.remote_preds_csv, "remote")

    merged = metadata.merge(street, on="sample_id", how="inner").merge(remote, on="sample_id", how="inner")
    merged["conflict"] = merged["street_prediction"] != merged["remote_prediction"]
    merged["conflict_type"] = merged.apply(_conflict_type, axis=1)

    if args.crossview_preds_csv:
        crossview = _load_predictions(args.crossview_preds_csv, "crossview")
        merged = merged.merge(crossview, on="sample_id", how="left")

    conflict_df = merged[merged["conflict"]].copy()
    output_csv = Path(args.output_csv)
    ensure_dir(output_csv.parent)
    conflict_df.to_csv(output_csv, index=False)

    target_column = "label" if "label" in conflict_df.columns else ("binary_label" if "binary_label" in conflict_df.columns else "street_target")
    summary = {
        "total_examples": int(len(merged)),
        "conflict_examples": int(len(conflict_df)),
        "conflict_rate": float(len(conflict_df) / max(len(merged), 1)),
        "street_accuracy_on_conflicts": _accuracy(conflict_df, "street_prediction", target_column),
        "remote_accuracy_on_conflicts": _accuracy(conflict_df, "remote_prediction", target_column),
        "crossview_accuracy_on_conflicts": _accuracy(conflict_df, "crossview_prediction", target_column),
        "street_accuracy_on_conflicts_ci": _bootstrap_accuracy_ci(
            conflict_df,
            "street_prediction",
            target_column,
            n_resamples=args.bootstrap_resamples,
            confidence_level=args.confidence_level,
            seed=args.seed,
        ),
        "remote_accuracy_on_conflicts_ci": _bootstrap_accuracy_ci(
            conflict_df,
            "remote_prediction",
            target_column,
            n_resamples=args.bootstrap_resamples,
            confidence_level=args.confidence_level,
            seed=args.seed,
        ),
        "crossview_accuracy_on_conflicts_ci": _bootstrap_accuracy_ci(
            conflict_df,
            "crossview_prediction",
            target_column,
            n_resamples=args.bootstrap_resamples,
            confidence_level=args.confidence_level,
            seed=args.seed,
        ),
        "target_distribution_on_conflicts": _target_distribution(conflict_df, target_column),
        "conflict_type_counts": conflict_df["conflict_type"].value_counts().to_dict(),
    }
    unique_targets = set(conflict_df[target_column].dropna().astype(int).tolist()) if len(conflict_df) else set()
    if unique_targets and unique_targets.issubset({0, 1}):
        summary["positive_rate_on_conflicts"] = float(conflict_df[target_column].mean())
    save_json(summary, args.summary_json)
    print(summary)


if __name__ == "__main__":
    main()
