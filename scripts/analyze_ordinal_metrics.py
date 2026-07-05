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

from crossview_conflict.utils.io import ensure_dir

MODES = ["street_only", "remote_only", "concat", "crossview"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ordinal metrics for the 3-class damage tasks: quadratic weighted "
            "kappa (QWK) and mean absolute error (MAE) treat adjacent-class "
            "confusions as less severe than no_damage<->destroyed confusions, "
            "matching triage costs."
        )
    )
    parser.add_argument("--multiseed-root", default="outputs/multiseed_main")
    parser.add_argument("--datasets", default="altadena_3class,ian_original,milton_original")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--output-dir", default="outputs/analysis/ordinal_metrics")
    parser.add_argument("--doc-path", default="docs/ordinal_metrics.md")
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


def quadratic_weighted_kappa(target: np.ndarray, prediction: np.ndarray, num_classes: int) -> float:
    observed = np.zeros((num_classes, num_classes), dtype=np.float64)
    for y_true, y_pred in zip(target, prediction):
        observed[int(y_true), int(y_pred)] += 1.0
    n = observed.sum()
    if n == 0:
        return float("nan")
    weights = np.zeros_like(observed)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col) / n
    denominator = (weights * expected).sum()
    if denominator <= 0:
        return float("nan")
    return float(1.0 - (weights * observed).sum() / denominator)


def main() -> None:
    args = parse_args()
    root = Path(args.multiseed_root)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            predictions: dict[str, pd.DataFrame] = {}
            for mode in MODES:
                df = pd.read_csv(root / dataset / f"{mode}_seed{seed}" / "test_predictions.csv")
                df["sample_id"] = df["sample_id"].map(_canonical_sample_id)
                predictions[mode] = df.set_index("sample_id").sort_index()
            base = predictions["street_only"]
            target = base["target"].to_numpy(dtype=np.int64)
            num_classes = len(_numbered_columns(base, "prob"))
            method_preds: dict[str, np.ndarray] = {
                mode: predictions[mode]["prediction"].to_numpy(dtype=np.int64) for mode in MODES
            }
            street_probs = base[_numbered_columns(base, "prob")].to_numpy(dtype=np.float64)
            remote_probs = predictions["remote_only"][
                _numbered_columns(predictions["remote_only"], "prob")
            ].to_numpy(dtype=np.float64)
            method_preds["late_fusion_probability_average"] = (
                (street_probs + remote_probs) / 2.0
            ).argmax(axis=1)
            for method, prediction in method_preds.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "qwk": quadratic_weighted_kappa(target, prediction, num_classes),
                        "mae": float(np.abs(prediction - target).mean()),
                        "extreme_error_rate": float((np.abs(prediction - target) >= 2).mean()),
                    }
                )
        print(f"computed ordinal metrics for {dataset}")

    results = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    results.to_csv(output_dir / "ordinal_metrics_raw.csv", index=False)

    lines = [
        "# Ordinal Metrics (QWK / MAE)",
        "",
        "The 3-class damage tasks are ordinal: confusing `no_or_trace_damage` with",
        "`destroyed` is operationally worse than confusing adjacent classes.",
        "Quadratic weighted kappa (QWK) and MAE reflect this; `extreme_error_rate`",
        "is the fraction of two-step errors (class 0 predicted as 2 or vice versa).",
        "Mean +/- std over seeds, main 3-class protocol.",
        "",
        "| dataset | method | QWK | MAE | extreme_error_rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for (dataset, method), group in results.groupby(["dataset", "method"]):
        def fmt(column: str) -> str:
            return f"{group[column].mean():.4f} +/- {group[column].std():.4f}"
        lines.append(
            f"| {dataset} | {method} | {fmt('qwk')} | {fmt('mae')} | {fmt('extreme_error_rate')} |"
        )
    doc_path = Path(args.doc_path)
    ensure_dir(doc_path.parent)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {doc_path}")
    summary = results.groupby(["dataset", "method"])[["qwk", "mae", "extreme_error_rate"]].mean().round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
