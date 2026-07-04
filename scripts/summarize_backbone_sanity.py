from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize backbone sanity runs.")
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--backbones", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", default=["street_only", "remote_only", "crossview"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--raw-output-csv")
    parser.add_argument("--summary-output-csv", required=True)
    parser.add_argument("--summary-output-md")
    return parser.parse_args()


def _safe_read(path: Path) -> dict:
    if not path.exists():
        return {}
    return read_json(path)


def _conflict_prefix(mode: str) -> str:
    return {"street_only": "street", "remote_only": "remote", "crossview": "crossview", "concat": "concat"}.get(mode, mode)


def _rows(outputs_root: Path, datasets: list[str], backbones: list[str], modes: list[str], seeds: list[int]) -> list[dict]:
    rows: list[dict] = []
    for dataset in datasets:
        for backbone in backbones:
            for seed in seeds:
                conflict = _safe_read(outputs_root / dataset / backbone / f"analysis_seed{seed}" / "test_conflict_summary.json")
                for mode in modes:
                    metrics = _safe_read(outputs_root / dataset / backbone / f"{mode}_seed{seed}" / "test_metrics.json")
                    rows.append(
                        {
                            "dataset": dataset,
                            "backbone": backbone,
                            "mode": mode,
                            "seed": seed,
                            "completed": bool(metrics),
                            "accuracy": metrics.get("accuracy"),
                            "macro_f1": metrics.get("macro_f1"),
                            "weighted_f1": metrics.get("weighted_f1"),
                            "conflict_rate": conflict.get("conflict_rate"),
                            "accuracy_on_conflicts": conflict.get(f"{_conflict_prefix(mode)}_accuracy_on_conflicts"),
                        }
                    )
    return rows


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    completed = raw[raw["completed"]].copy()
    if completed.empty:
        return pd.DataFrame()
    metrics = ["accuracy", "macro_f1", "weighted_f1", "conflict_rate", "accuracy_on_conflicts"]
    grouped = completed.groupby(["dataset", "backbone", "mode"], dropna=False)
    parts = []
    for metric in metrics:
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
        summary = summary.merge(part, on=["dataset", "backbone", "mode"], how="outer")
    return summary.sort_values(["dataset", "backbone", "mode"]).reset_index(drop=True)


def _format_mean_std(mean: object, std: object) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def _write_markdown(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "backbone", "mode", "accuracy", "macro_f1", "weighted_f1", "conflict_acc"]
    lines = [
        "# Backbone Sanity Results",
        "",
        "Mean +/- std over completed seeds.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in summary.to_dict(orient="records"):
        values = [
            str(row["dataset"]),
            str(row["backbone"]),
            str(row["mode"]),
            _format_mean_std(row.get("accuracy_mean"), row.get("accuracy_std")),
            _format_mean_std(row.get("macro_f1_mean"), row.get("macro_f1_std")),
            _format_mean_std(row.get("weighted_f1_mean"), row.get("weighted_f1_std")),
            _format_mean_std(row.get("accuracy_on_conflicts_mean"), row.get("accuracy_on_conflicts_std")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    raw = pd.DataFrame(_rows(outputs_root, args.datasets, args.backbones, args.modes, args.seeds))
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
