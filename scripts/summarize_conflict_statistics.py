from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize conflict-subset statistical tests across seeds.")
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--raw-output-csv")
    parser.add_argument("--summary-output-csv", required=True)
    parser.add_argument("--summary-output-md")
    return parser.parse_args()


def _rows(outputs_root: Path, datasets: list[str], seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            path = outputs_root / dataset / f"analysis_seed{seed}" / "conflict_statistics.csv"
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


def _aggregate(raw: pd.DataFrame, alpha: float) -> pd.DataFrame:
    completed = raw[raw["completed"]].copy()
    if completed.empty:
        return pd.DataFrame()
    completed["comparison_group"] = completed["comparison"].astype(str)
    completed.loc[
        completed["comparison_group"].str.startswith("crossview_vs_best_named_single"),
        "comparison_group",
    ] = "crossview_vs_best_named_single"
    completed["ci_excludes_zero"] = (completed["bootstrap_ci_low"] > 0.0) | (completed["bootstrap_ci_high"] < 0.0)
    completed["sign_flip_significant"] = completed["sign_flip_p"] < alpha
    completed["mcnemar_significant"] = completed["mcnemar_p"] < alpha
    grouped = completed.groupby(["dataset", "comparison_group"], dropna=False)
    summary = grouped.agg(
        comparison_details=("comparison", lambda values: ",".join(sorted(set(str(value) for value in values)))),
        seeds=("seed", "count"),
        crossview_accuracy_mean=("crossview_accuracy", "mean"),
        crossview_accuracy_std=("crossview_accuracy", "std"),
        bootstrap_delta_mean=("bootstrap_delta", "mean"),
        bootstrap_delta_std=("bootstrap_delta", "std"),
        bootstrap_ci_low_min=("bootstrap_ci_low", "min"),
        bootstrap_ci_high_max=("bootstrap_ci_high", "max"),
        ci_excludes_zero_count=("ci_excludes_zero", "sum"),
        sign_flip_p_median=("sign_flip_p", "median"),
        sign_flip_significant_count=("sign_flip_significant", "sum"),
        mcnemar_p_median=("mcnemar_p", "median"),
        mcnemar_significant_count=("mcnemar_significant", "sum"),
    ).reset_index()
    summary = summary.rename(columns={"comparison_group": "comparison"})
    return summary.sort_values(["dataset", "comparison"]).reset_index(drop=True)


def _format_mean_std(mean: object, std: object) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def _write_markdown(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "comparison",
        "delta",
        "crossview_acc",
        "CI excludes 0",
        "perm sig",
        "McNemar sig",
    ]
    lines = [
        "# Conflict Statistical Tests",
        "",
        "Mean +/- std over completed seeds. Significance counts use alpha from the command line.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in summary.to_dict(orient="records"):
        seeds = int(row["seeds"])
        values = [
            str(row["dataset"]),
            str(row["comparison"]),
            _format_mean_std(row.get("bootstrap_delta_mean"), row.get("bootstrap_delta_std")),
            _format_mean_std(row.get("crossview_accuracy_mean"), row.get("crossview_accuracy_std")),
            f"{int(row['ci_excludes_zero_count'])}/{seeds}",
            f"{int(row['sign_flip_significant_count'])}/{seeds}",
            f"{int(row['mcnemar_significant_count'])}/{seeds}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    raw = pd.DataFrame(_rows(outputs_root, args.datasets, args.seeds))
    summary = _aggregate(raw, args.alpha)
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
