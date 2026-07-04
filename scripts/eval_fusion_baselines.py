from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Evaluate non-trained fusion baselines from per-view predictions.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--street-preds-csv", required=True)
    parser.add_argument("--remote-preds-csv", required=True)
    parser.add_argument("--crossview-preds-csv")
    parser.add_argument("--concat-preds-csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--predictions-csv")
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
    required = {"sample_id", "target", "prediction"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    df["sample_id"] = df["sample_id"].map(_canonical_sample_id)

    rename: dict[str, str] = {
        "target": f"{prefix}_target",
        "target_name": f"{prefix}_target_name",
        "prediction": f"{prefix}_prediction",
        "prediction_name": f"{prefix}_prediction_name",
        "confidence": f"{prefix}_confidence",
        "probability": f"{prefix}_probability",
        "logit": f"{prefix}_logit",
    }
    for column in _numbered_columns(df, "prob"):
        rename[column] = f"{prefix}_{column}"
    for column in _numbered_columns(df, "logit"):
        rename[column] = f"{prefix}_{column}"
    return df.rename(columns={key: value for key, value in rename.items() if key in df.columns})


def _prob_matrix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    prob_cols = _numbered_columns(df, f"{prefix}_prob")
    if prob_cols:
        return df[prob_cols].to_numpy(dtype=np.float64)
    probability_col = f"{prefix}_probability"
    if probability_col not in df.columns:
        raise KeyError(f"No probability columns found for {prefix}.")
    p1 = df[probability_col].to_numpy(dtype=np.float64)
    return np.stack([1.0 - p1, p1], axis=1)


def _logit_matrix(df: pd.DataFrame, prefix: str, probs: np.ndarray) -> tuple[np.ndarray, str]:
    logit_cols = _numbered_columns(df, f"{prefix}_logit")
    if logit_cols:
        return df[logit_cols].to_numpy(dtype=np.float64), "logits"
    logit_col = f"{prefix}_logit"
    if logit_col in df.columns:
        logit = df[logit_col].to_numpy(dtype=np.float64)
        return np.stack([np.zeros_like(logit), logit], axis=1), "binary_logit"
    return np.log(np.clip(probs, 1e-12, 1.0)), "log_probability_fallback"


def _metric_row(
    dataset: str,
    method: str,
    target: np.ndarray,
    prediction: np.ndarray,
    conflict_mask: np.ndarray,
    *,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    num_classes = int(max(target.max(initial=0), prediction.max(initial=0)) + 1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(target, prediction):
        if 0 <= int(y_true) < num_classes and 0 <= int(y_pred) < num_classes:
            confusion[int(y_true), int(y_pred)] += 1
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / max(total, 1))
    f1_values: list[float] = []
    weighted_f1 = 0.0
    for class_index in range(num_classes):
        tp = float(confusion[class_index, class_index])
        fp = float(confusion[:, class_index].sum() - tp)
        fn = float(confusion[class_index, :].sum() - tp)
        support = float(confusion[class_index, :].sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if support > 0:
            f1_values.append(f1)
            weighted_f1 += f1 * support
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    weighted_f1 = float(weighted_f1 / max(total, 1))
    conflict_accuracy = None
    if conflict_mask.any():
        conflict_accuracy = float((prediction[conflict_mask] == target[conflict_mask]).mean())
    row: dict[str, object] = {
        "dataset": dataset,
        "method": method,
        "num_examples": int(len(target)),
        "conflict_examples": int(conflict_mask.sum()),
        "conflict_rate": float(conflict_mask.mean()) if len(conflict_mask) else 0.0,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy_on_conflicts": conflict_accuracy,
    }
    if details:
        row["details"] = json.dumps(details, sort_keys=True)
    return row


def _append_reference(
    rows: list[dict[str, object]],
    df: pd.DataFrame,
    dataset: str,
    method: str,
    prefix: str,
    target: np.ndarray,
    conflict_mask: np.ndarray,
) -> None:
    prediction_col = f"{prefix}_prediction"
    if prediction_col not in df.columns:
        return
    rows.append(_metric_row(dataset, method, target, df[prediction_col].to_numpy(dtype=np.int64), conflict_mask))


def main() -> None:
    args = parse_args()
    street = _load_predictions(args.street_preds_csv, "street")
    remote = _load_predictions(args.remote_preds_csv, "remote")
    merged = street.merge(remote, on="sample_id", how="inner")
    if args.crossview_preds_csv:
        merged = merged.merge(_load_predictions(args.crossview_preds_csv, "crossview"), on="sample_id", how="left")
    if args.concat_preds_csv:
        merged = merged.merge(_load_predictions(args.concat_preds_csv, "concat"), on="sample_id", how="left")

    target = merged["street_target"].to_numpy(dtype=np.int64)
    street_probs = _prob_matrix(merged, "street")
    remote_probs = _prob_matrix(merged, "remote")
    street_logits, street_logit_source = _logit_matrix(merged, "street", street_probs)
    remote_logits, remote_logit_source = _logit_matrix(merged, "remote", remote_probs)

    conflict_mask = merged["street_prediction"].to_numpy() != merged["remote_prediction"].to_numpy()
    rows: list[dict[str, object]] = []
    _append_reference(rows, merged, args.dataset_name, "street_only", "street", target, conflict_mask)
    _append_reference(rows, merged, args.dataset_name, "remote_only", "remote", target, conflict_mask)
    _append_reference(rows, merged, args.dataset_name, "concat_reference", "concat", target, conflict_mask)
    _append_reference(rows, merged, args.dataset_name, "crossview_reference", "crossview", target, conflict_mask)

    prob_avg = (street_probs + remote_probs) / 2.0
    prob_avg_pred = prob_avg.argmax(axis=1)
    rows.append(_metric_row(args.dataset_name, "late_fusion_probability_average", target, prob_avg_pred, conflict_mask))

    logit_avg = (street_logits + remote_logits) / 2.0
    logit_avg_pred = logit_avg.argmax(axis=1)
    rows.append(
        _metric_row(
            args.dataset_name,
            "late_fusion_logit_average",
            target,
            logit_avg_pred,
            conflict_mask,
            details={"street_source": street_logit_source, "remote_source": remote_logit_source},
        )
    )

    street_pred = merged["street_prediction"].to_numpy(dtype=np.int64)
    remote_pred = merged["remote_prediction"].to_numpy(dtype=np.int64)
    street_conf = np.max(street_probs, axis=1)
    remote_conf = np.max(remote_probs, axis=1)
    voting_pred = np.where(street_pred == remote_pred, street_pred, np.where(street_conf >= remote_conf, street_pred, remote_pred))
    rows.append(_metric_row(args.dataset_name, "confidence_voting", target, voting_pred, conflict_mask))

    summary_df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    ensure_dir(output_csv.parent)
    summary_df.to_csv(output_csv, index=False)
    save_json(
        {
            "dataset": args.dataset_name,
            "num_examples": int(len(merged)),
            "conflict_examples": int(conflict_mask.sum()),
            "conflict_rate": float(conflict_mask.mean()) if len(conflict_mask) else 0.0,
            "rows": rows,
        },
        args.output_json,
    )

    if args.predictions_csv:
        pred_df = pd.DataFrame(
            {
                "sample_id": merged["sample_id"],
                "target": target,
                "late_fusion_probability_average_prediction": prob_avg_pred,
                "late_fusion_probability_average_confidence": prob_avg.max(axis=1),
                "late_fusion_logit_average_prediction": logit_avg_pred,
                "late_fusion_logit_average_confidence": logit_avg.max(axis=1),
                "confidence_voting_prediction": voting_pred,
            }
        )
        pred_path = Path(args.predictions_csv)
        ensure_dir(pred_path.parent)
        pred_df.to_csv(pred_path, index=False)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
