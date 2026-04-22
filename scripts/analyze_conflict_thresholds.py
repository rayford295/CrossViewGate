from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    parser = argparse.ArgumentParser(
        description="Analyze sensitivity to conflict-definition thresholds."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--street-preds-csv", required=True)
    parser.add_argument("--remote-preds-csv", required=True)
    parser.add_argument("--crossview-preds-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5])
    return parser.parse_args()


def _load_predictions(path: str | Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sample_id"] = df["sample_id"].map(_canonical_sample_id)
    return df.rename(
        columns={
            "target": f"{prefix}_target",
            "probability": f"{prefix}_probability",
            "prediction": f"{prefix}_prediction",
            "logit": f"{prefix}_logit",
        }
    )


def _accuracy(frame: pd.DataFrame, prediction_column: str, target_column: str) -> float | None:
    if len(frame) == 0:
        return None
    return float((frame[prediction_column] == frame[target_column]).mean())


def _save_plot(results: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(results["threshold"], results["conflict_rate"], marker="o")
    axes[0].set_title("Conflict rate vs threshold")
    axes[0].set_xlabel("threshold")
    axes[0].set_ylabel("conflict rate")

    axes[1].plot(results["threshold"], results["street_accuracy"], marker="o", label="street")
    axes[1].plot(results["threshold"], results["remote_accuracy"], marker="o", label="remote")
    axes[1].plot(results["threshold"], results["crossview_accuracy"], marker="o", label="crossview")
    axes[1].set_title("Accuracy on soft-conflict subset")
    axes[1].set_xlabel("threshold")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    split_df = pd.read_csv(args.split_csv)[["sample_id", "binary_label"]].copy()
    split_df["sample_id"] = split_df["sample_id"].map(_canonical_sample_id)
    street = _load_predictions(args.street_preds_csv, "street")
    remote = _load_predictions(args.remote_preds_csv, "remote")
    cross = _load_predictions(args.crossview_preds_csv, "crossview")
    merged = split_df.merge(street, on="sample_id").merge(remote, on="sample_id").merge(cross, on="sample_id")
    results: list[dict[str, float | int | None]] = []
    prob_gap = np.abs(merged["street_probability"] - merged["remote_probability"])
    for threshold in args.thresholds:
        subset = merged[prob_gap > threshold].copy()
        results.append(
            {
                "threshold": float(threshold),
                "num_examples": int(len(subset)),
                "conflict_rate": float(len(subset) / max(len(merged), 1)),
                "street_accuracy": _accuracy(subset, "street_prediction", "binary_label"),
                "remote_accuracy": _accuracy(subset, "remote_prediction", "binary_label"),
                "crossview_accuracy": _accuracy(subset, "crossview_prediction", "binary_label"),
            }
        )
    results_df = pd.DataFrame(results)
    csv_path = output_dir / f"{args.dataset_name}_threshold_sensitivity.csv"
    plot_path = output_dir / f"{args.dataset_name}_threshold_sensitivity.png"
    results_df.to_csv(csv_path, index=False)
    _save_plot(results_df, plot_path)
    summary = {
        "dataset": args.dataset_name,
        "csv": str(csv_path),
        "plot": str(plot_path),
        "rows": results,
    }
    save_json(summary, output_dir / f"{args.dataset_name}_threshold_sensitivity.json")
    print(summary)


if __name__ == "__main__":
    main()
