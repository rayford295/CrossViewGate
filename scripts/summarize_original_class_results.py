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
    parser = argparse.ArgumentParser(description="Summarize original-class CrossViewConflict experiments.")
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--output-md")
    parser.add_argument("--datasets", nargs="*", default=["altadena_original", "ian_original", "milton_original"])
    parser.add_argument("--modes", nargs="*", default=["street_only", "remote_only", "crossview"])
    return parser.parse_args()


def _read_metric(path: Path, key: str) -> object:
    if not path.exists():
        return None
    metrics = read_json(path)
    return metrics.get(key)


def _read_conflict_metric(path: Path, key: str) -> object:
    if not path.exists():
        return None
    summary = read_json(path)
    return summary.get(key)


def _rows(outputs_root: Path, datasets: list[str], modes: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    conflict_prefix = {
        "street_only": "street",
        "remote_only": "remote",
        "concat": "concat",
        "crossview": "crossview",
    }
    for dataset in datasets:
        analysis_path = outputs_root / dataset / "analysis" / "test_conflict_summary.json"
        for mode in modes:
            metric_path = outputs_root / dataset / mode / "test_metrics.json"
            prefix = conflict_prefix.get(mode, mode)
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "accuracy": _read_metric(metric_path, "accuracy"),
                    "macro_f1": _read_metric(metric_path, "macro_f1"),
                    "weighted_f1": _read_metric(metric_path, "weighted_f1"),
                    "conflict_rate": _read_conflict_metric(analysis_path, "conflict_rate"),
                    "accuracy_on_conflicts": _read_conflict_metric(
                        analysis_path, f"{prefix}_accuracy_on_conflicts"
                    ),
                }
            )
    return rows


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(df.columns)
    table_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.to_dict(orient="records"):
        table_lines.append("| " + " | ".join("" if pd.isna(row[column]) else str(row[column]) for column in columns) + " |")
    lines = [
        "# Original-Class Results",
        "",
        "These results use each dataset's native class space rather than a binary collapse.",
        "",
        "\n".join(table_lines),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    df = pd.DataFrame(_rows(outputs_root, args.datasets, args.modes))
    print(df.to_string(index=False))
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    if args.output_md:
        _write_markdown(df, Path(args.output_md))


if __name__ == "__main__":
    main()
