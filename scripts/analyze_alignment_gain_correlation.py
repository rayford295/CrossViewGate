from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate building-alignment proxies with crossview conflict resolution."
    )
    parser.add_argument("--wildfire-conflict-csv", required=True)
    parser.add_argument("--hurricane-conflict-csv", required=True)
    parser.add_argument("--wildfire-alignment-csv", required=True)
    parser.add_argument("--hurricane-alignment-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _spearman_with_permutation(x: np.ndarray, y: np.ndarray, permutations: int, seed: int) -> dict[str, float]:
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    observed = float(np.corrcoef(x_rank, y_rank)[0, 1]) if len(x) > 1 else 0.0
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for idx in range(permutations):
        shuffled = rng.permutation(y_rank)
        null[idx] = np.corrcoef(x_rank, shuffled)[0, 1]
    p_value = float((np.abs(null) >= abs(observed)).mean())
    return {"spearman_r": observed, "p_value_two_sided": p_value}


def _merge(conflict_csv: str | Path, alignment_csv: str | Path, dataset: str) -> pd.DataFrame:
    conflict = pd.read_csv(conflict_csv)
    alignment = pd.read_csv(alignment_csv)
    merged = conflict.merge(
        alignment[
            [
                "sample_id",
                "building_ratio",
                "center_building_ratio",
                "center_minus_global",
                "centroid_distance_norm",
            ]
        ],
        on="sample_id",
        how="inner",
    )
    target = merged["binary_label"].to_numpy()
    merged["street_correct"] = (merged["street_prediction"].to_numpy() == target).astype(int)
    merged["remote_correct"] = (merged["remote_prediction"].to_numpy() == target).astype(int)
    merged["crossview_correct"] = (merged["crossview_prediction"].to_numpy() == target).astype(int)
    merged["crossview_advantage_over_remote"] = merged["crossview_correct"] - merged["remote_correct"]
    merged["crossview_advantage_over_street"] = merged["crossview_correct"] - merged["street_correct"]
    merged["dataset"] = dataset
    return merged


def _summarize(df: pd.DataFrame, feature: str, permutations: int, seed: int) -> dict[str, float]:
    x = df[feature].to_numpy(dtype=np.float64)
    y = df["crossview_correct"].to_numpy(dtype=np.float64)
    result = _spearman_with_permutation(x, y, permutations, seed)
    result["feature_mean"] = float(x.mean())
    result["crossview_correct_mean"] = float(y.mean())
    return result


def _save_plot(combined: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    rng = np.random.default_rng(42)
    for dataset, color in [("wildfire", "#d95f02"), ("hurricane", "#1b9e77")]:
        subset = combined[combined["dataset"] == dataset].copy()
        jitter = rng.uniform(-0.03, 0.03, size=len(subset))
        ax.scatter(
            subset["building_ratio"],
            subset["crossview_correct"] + jitter,
            alpha=0.75,
            label=dataset,
            s=28,
            color=color,
        )
        bins = pd.qcut(subset["building_ratio"], q=min(4, len(subset)), duplicates="drop")
        grouped = subset.groupby(bins, observed=True).agg(
            building_ratio=("building_ratio", "mean"),
            crossview_correct=("crossview_correct", "mean"),
        )
        ax.plot(grouped["building_ratio"], grouped["crossview_correct"], color=color, linewidth=2)
    ax.set_xlabel("Building ratio")
    ax.set_ylabel("Crossview correct on conflict subset")
    ax.set_title("Alignment proxy vs crossview conflict resolution")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(False)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    wildfire = _merge(args.wildfire_conflict_csv, args.wildfire_alignment_csv, "wildfire")
    hurricane = _merge(args.hurricane_conflict_csv, args.hurricane_alignment_csv, "hurricane")
    combined = pd.concat([wildfire, hurricane], ignore_index=True)
    combined.to_csv(output_dir / "alignment_gain_merged.csv", index=False)

    summary = {
        "wildfire": _summarize(wildfire, "building_ratio", args.permutations, args.seed),
        "hurricane": _summarize(hurricane, "building_ratio", args.permutations, args.seed),
        "combined": _summarize(combined, "building_ratio", args.permutations, args.seed),
        "combined_center_ratio": _summarize(combined, "center_building_ratio", args.permutations, args.seed),
        "combined_centroid_distance": _summarize(combined, "centroid_distance_norm", args.permutations, args.seed),
    }
    plot_path = output_dir / "alignment_gain_correlation.png"
    _save_plot(combined, plot_path)
    summary["plot"] = str(plot_path)
    save_json(summary, output_dir / "alignment_gain_correlation_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
