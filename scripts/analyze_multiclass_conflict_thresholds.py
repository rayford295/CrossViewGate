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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multiclass soft-conflict threshold sensitivity.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--street-preds-csv", required=True)
    parser.add_argument("--remote-preds-csv", required=True)
    parser.add_argument("--crossview-preds-csv", required=True)
    parser.add_argument("--concat-preds-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4])
    return parser.parse_args()


def _canonical_sample_id(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text


def _numbered_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    pairs: list[tuple[int, str]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for column in df.columns:
        match = pattern.match(column)
        if match:
            pairs.append((int(match.group(1)), column))
    return [column for _, column in sorted(pairs)]


def _load_predictions(path: str | Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sample_id"] = df["sample_id"].map(_canonical_sample_id)
    rename = {
        "target": f"{prefix}_target",
        "target_name": f"{prefix}_target_name",
        "prediction": f"{prefix}_prediction",
        "prediction_name": f"{prefix}_prediction_name",
        "probability": f"{prefix}_probability",
        "confidence": f"{prefix}_confidence",
        "logit": f"{prefix}_logit",
    }
    for column in _numbered_columns(df, "prob"):
        rename[column] = f"{prefix}_{column}"
    for column in _numbered_columns(df, "logit"):
        rename[column] = f"{prefix}_{column}"
    return df.rename(columns={key: value for key, value in rename.items() if key in df.columns})


def _target_column(df: pd.DataFrame) -> str:
    for candidate in ("label", "binary_label", "street_target"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No target column found.")


def _prob_matrix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    prob_cols = _numbered_columns(df, f"{prefix}_prob")
    if prob_cols:
        return df[prob_cols].to_numpy(dtype=np.float64)
    probability = df[f"{prefix}_probability"].to_numpy(dtype=np.float64)
    return np.stack([1.0 - probability, probability], axis=1)


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    return 0.5 * (kl_pm + kl_qm) / np.log(2.0)


def _accuracy(frame: pd.DataFrame, prediction_col: str, target_col: str) -> float | None:
    if len(frame) == 0 or prediction_col not in frame.columns:
        return None
    return float((frame[prediction_col].to_numpy() == frame[target_col].to_numpy()).mean())


def _save_plot(results: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(results["threshold"], results["conflict_rate"], marker="o")
    axes[0].set_xlabel("JS threshold")
    axes[0].set_ylabel("selected rate")
    axes[0].set_title("Soft-conflict rate")

    for column, label in [
        ("street_accuracy", "street"),
        ("remote_accuracy", "remote"),
        ("concat_accuracy", "concat"),
        ("crossview_accuracy", "crossview"),
    ]:
        if column in results.columns and results[column].notna().any():
            axes[1].plot(results["threshold"], results[column], marker="o", label=label)
    axes[1].set_xlabel("JS threshold")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy on soft-conflict subset")
    axes[1].legend(frameon=False)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    split = pd.read_csv(args.split_csv)
    split["sample_id"] = split["sample_id"].map(_canonical_sample_id)
    keep_cols = [column for column in ["sample_id", "label", "binary_label"] if column in split.columns]
    merged = split[keep_cols].merge(_load_predictions(args.street_preds_csv, "street"), on="sample_id")
    merged = merged.merge(_load_predictions(args.remote_preds_csv, "remote"), on="sample_id")
    merged = merged.merge(_load_predictions(args.crossview_preds_csv, "crossview"), on="sample_id")
    if args.concat_preds_csv:
        merged = merged.merge(_load_predictions(args.concat_preds_csv, "concat"), on="sample_id", how="left")

    target_col = _target_column(merged)
    street_probs = _prob_matrix(merged, "street")
    remote_probs = _prob_matrix(merged, "remote")
    js_score = _jensen_shannon(street_probs, remote_probs)
    hard_conflict = merged["street_prediction"] != merged["remote_prediction"]
    rows: list[dict[str, object]] = []

    for threshold in args.thresholds:
        subset = merged[js_score > threshold].copy()
        best_single_accuracy = None
        best_single_view = None
        street_accuracy = _accuracy(subset, "street_prediction", target_col)
        remote_accuracy = _accuracy(subset, "remote_prediction", target_col)
        if street_accuracy is not None and remote_accuracy is not None:
            if street_accuracy >= remote_accuracy:
                best_single_accuracy = street_accuracy
                best_single_view = "street"
            else:
                best_single_accuracy = remote_accuracy
                best_single_view = "remote"
        crossview_accuracy = _accuracy(subset, "crossview_prediction", target_col)
        rows.append(
            {
                "dataset": args.dataset_name,
                "threshold": float(threshold),
                "num_examples": int(len(subset)),
                "conflict_rate": float(len(subset) / max(len(merged), 1)),
                "hard_conflict_fraction": float(hard_conflict[js_score > threshold].mean()) if len(subset) else None,
                "street_accuracy": street_accuracy,
                "remote_accuracy": remote_accuracy,
                "concat_accuracy": _accuracy(subset, "concat_prediction", target_col),
                "crossview_accuracy": crossview_accuracy,
                "best_single_view": best_single_view,
                "best_single_accuracy": best_single_accuracy,
                "crossview_minus_best_single": (
                    float(crossview_accuracy - best_single_accuracy)
                    if crossview_accuracy is not None and best_single_accuracy is not None
                    else None
                ),
            }
        )

    results = pd.DataFrame(rows)
    csv_path = output_dir / f"{args.dataset_name}_threshold_sensitivity.csv"
    json_path = output_dir / f"{args.dataset_name}_threshold_sensitivity.json"
    plot_path = output_dir / f"{args.dataset_name}_threshold_sensitivity.png"
    results.to_csv(csv_path, index=False)
    _save_plot(results, plot_path)
    save_json(
        {
            "dataset": args.dataset_name,
            "score": "normalized_jensen_shannon_divergence_between_single_view_probability_vectors",
            "target_column": target_col,
            "csv": str(csv_path),
            "plot": str(plot_path),
            "rows": rows,
        },
        json_path,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
