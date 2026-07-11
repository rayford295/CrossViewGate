"""Compare legacy and repaired CVIAN split results across matched seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


RUN_PATTERN = re.compile(r"^(?P<mode>.+)_seed(?P<seed>\d+)$")
DEFAULT_MODES = ("street_only", "remote_only", "concat", "crossview")
DEFAULT_SEEDS = (42, 123, 456, 789, 1011)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare five-seed Ian metrics on the legacy and repaired spatial splits."
    )
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--repaired-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    return parser.parse_args()


def _severe_recall(metrics: dict[str, object]) -> float:
    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict) or not per_class:
        return float("nan")
    severe_keys = [
        key
        for key in per_class
        if re.search(r"severe|destroy", str(key), flags=re.IGNORECASE)
    ]
    if not severe_keys:
        severe_keys = [list(per_class)[-1]]
    recalls = [float(per_class[key]["recall"]) for key in severe_keys]
    return float(np.mean(recalls))


def load_protocol_runs(
    root: Path,
    *,
    protocol: str,
    modes: Iterable[str],
    seeds: Iterable[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode in modes:
        for seed in seeds:
            metrics_path = root / f"{mode}_seed{seed}" / "test_metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Missing {protocol} metrics: {metrics_path}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "protocol": protocol,
                    "mode": mode,
                    "seed": int(seed),
                    "accuracy": float(metrics["accuracy"]),
                    "macro_f1": float(
                        metrics["macro_f1"] if "macro_f1" in metrics else metrics["f1"]
                    ),
                    "severe_recall": _severe_recall(metrics),
                    "test_count": int(
                        sum(
                            int(values["support"])
                            for values in metrics.get("per_class", {}).values()
                        )
                    ),
                    "checkpoint_epoch": metrics.get("checkpoint_epoch"),
                    "metrics_path": str(metrics_path.resolve()),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_mean_interval(
    values: np.ndarray, *, replicates: int, seed: int
) -> tuple[float, float]:
    if replicates < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def summarize_comparison(
    runs: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ("accuracy", "macro_f1", "severe_recall")
    summary = (
        runs.groupby(["protocol", "mode"], sort=True)[list(metrics)]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]

    legacy = runs[runs["protocol"] == "legacy"].set_index(["mode", "seed"])
    repaired = runs[runs["protocol"] == "repaired_spatial"].set_index(["mode", "seed"])
    if set(legacy.index) != set(repaired.index):
        raise ValueError("Legacy and repaired runs do not have identical mode/seed keys")
    difference_rows: list[dict[str, object]] = []
    for mode in sorted(runs["mode"].unique()):
        mode_index = [index for index in legacy.index if index[0] == mode]
        for metric_index, metric in enumerate(metrics):
            differences = np.asarray(
                [repaired.loc[index, metric] - legacy.loc[index, metric] for index in mode_index],
                dtype=np.float64,
            )
            low, high = _bootstrap_mean_interval(
                differences,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + metric_index,
            )
            difference_rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "seed_count": len(differences),
                    "repaired_minus_legacy_mean": float(differences.mean()),
                    "paired_seed_difference_std": float(differences.std(ddof=1)),
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                }
            )
    return summary, pd.DataFrame(difference_rows)


def _format(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def write_report(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "per_run.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    differences.to_csv(output_dir / "paired_seed_differences.csv", index=False)
    lines = [
        "# CVIAN legacy vs repaired spatial split",
        "",
        "The protocols use different held-out samples (legacy n=300; repaired n=415).",
        "Seed-matched differences therefore quantify training-seed stability of the",
        "protocol change; they are not paired per-sample test estimates.",
        "",
        "| mode | protocol | accuracy | macro-F1 | severe recall |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in summary.sort_values(["mode", "protocol"]).iterrows():
        lines.append(
            f"| {row['mode']} | {row['protocol']} | "
            f"{_format(row['accuracy_mean'], row['accuracy_std'])} | "
            f"{_format(row['macro_f1_mean'], row['macro_f1_std'])} | "
            f"{_format(row['severe_recall_mean'], row['severe_recall_std'])} |"
        )
    lines += [
        "",
        "## Repaired minus legacy",
        "",
        "| mode | metric | mean difference | seed-bootstrap 95% interval |",
        "| --- | --- | ---: | ---: |",
    ]
    for _, row in differences.iterrows():
        lines.append(
            f"| {row['mode']} | {row['metric']} | "
            f"{row['repaired_minus_legacy_mean']:.4f} | "
            f"[{row['bootstrap_95_low']:.4f}, {row['bootstrap_95_high']:.4f}] |"
        )
    (output_dir / "ian_split_performance_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    runs = pd.concat(
        [
            load_protocol_runs(
                Path(args.legacy_root), protocol="legacy", modes=modes, seeds=seeds
            ),
            load_protocol_runs(
                Path(args.repaired_root),
                protocol="repaired_spatial",
                modes=modes,
                seeds=seeds,
            ),
        ],
        ignore_index=True,
    )
    summary, differences = summarize_comparison(
        runs,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_report(runs, summary, differences, Path(args.output_dir))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
