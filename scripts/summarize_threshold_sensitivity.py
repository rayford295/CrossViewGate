from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize conflict-threshold sensitivity across seeds.")
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--raw-output-csv")
    parser.add_argument("--summary-output-csv", required=True)
    parser.add_argument("--summary-output-md")
    return parser.parse_args()


def _rows(outputs_root: Path, datasets: list[str], seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            path = outputs_root / dataset / f"analysis_seed{seed}" / f"{dataset}_threshold_sensitivity.csv"
            if not path.exists():
                rows.append({"dataset": dataset, "seed": seed, "completed": False})
                continue
            df = pd.read_csv(path)
            for row in df.to_dict(orient="records"):
                row["dataset"] = dataset
                row["seed"] = seed
                row["completed"] = True
                rows.append(row)
    return rows


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    completed = raw[raw["completed"]].copy()
    if completed.empty:
        return pd.DataFrame()
    completed["crossview_positive"] = completed["crossview_minus_best_single"] > 0.0
    metrics = [
        "num_examples",
        "conflict_rate",
        "hard_conflict_fraction",
        "street_accuracy",
        "remote_accuracy",
        "concat_accuracy",
        "crossview_accuracy",
        "best_single_accuracy",
        "crossview_minus_best_single",
    ]
    grouped = completed.groupby(["dataset", "threshold"], dropna=False)
    parts = []
    for metric in metrics:
        if metric not in completed.columns:
            continue
        stats = grouped[metric].agg(["count", "mean", "std"]).reset_index()
        stats = stats.rename(
            columns={
                "count": f"{metric}_n",
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
            }
        )
        parts.append(stats)
    summary = parts[0]
    for part in parts[1:]:
        summary = summary.merge(part, on=["dataset", "threshold"], how="outer")
    positive = grouped["crossview_positive"].sum().reset_index().rename(columns={"crossview_positive": "crossview_positive_count"})
    summary = summary.merge(positive, on=["dataset", "threshold"], how="left")
    return summary.sort_values(["dataset", "threshold"]).reset_index(drop=True)


def _format_mean_std(mean: object, std: object) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def _write_markdown(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "threshold", "n", "hard_conflict", "crossview_minus_best", "positive seeds"]
    lines = [
        "# Threshold Sensitivity",
        "",
        "Mean +/- std over completed seeds.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in summary.to_dict(orient="records"):
        seeds = int(row.get("crossview_minus_best_single_n", 0))
        values = [
            str(row["dataset"]),
            f"{float(row['threshold']):.3f}",
            _format_mean_std(row.get("num_examples_mean"), row.get("num_examples_std")),
            _format_mean_std(row.get("hard_conflict_fraction_mean"), row.get("hard_conflict_fraction_std")),
            _format_mean_std(row.get("crossview_minus_best_single_mean"), row.get("crossview_minus_best_single_std")),
            f"{int(row.get('crossview_positive_count', 0))}/{seeds}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    raw = pd.DataFrame(_rows(outputs_root, args.datasets, args.seeds))
    summary = _aggregate(raw)
    summary_csv = Path(args.summary_output_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    if args.raw_output_csv:
        raw_csv = Path(args.raw_output_csv)
        raw_csv.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_csv, index=False)
    if args.summary_output_md:
        _write_markdown(summary, Path(args.summary_output_md))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
