"""
B3: Scatter plot of per-sample building_ratio vs crossview correctness on
conflict subset, with a logistic regression curve overlaid.

This script merges:
  - per-sample SegFormer building alignment output (from analyze_building_alignment.py
    run with --save-per-sample)
  - per-sample conflict predictions (from build_conflict_subset.py)

and produces the key mechanism figure for the paper.

Usage:
    python scripts/plot_building_ratio_vs_gain.py \
        --wildfire-alignment outputs/eaton_wildfire/analysis/alignment_per_sample.csv \
        --hurricane-alignment outputs/ian_hurricane/analysis/alignment_per_sample.csv \
        --wildfire-conflict  outputs/eaton_wildfire/analysis/test_conflicts.csv \
        --hurricane-conflict outputs/ian_hurricane/analysis/test_conflicts.csv \
        --output             outputs/figures/building_ratio_vs_crossview_gain.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    raise SystemExit("pip install matplotlib before running this script.")


COLORS = {
    "wildfire": "#D95F42",
    "hurricane": "#4A90C4",
}


def load_and_merge(alignment_path: str, conflict_path: str, label: str) -> pd.DataFrame:
    align = pd.read_csv(alignment_path)   # expects: sample_id, building_ratio
    conflict = pd.read_csv(conflict_path)  # expects: sample_id, crossview_pred, label

    if "building_ratio" not in align.columns:
        raise ValueError(
            f"{alignment_path} must have a 'building_ratio' column. "
            "Re-run analyze_building_alignment.py with --save-per-sample."
        )

    conflict["crossview_correct"] = (
        conflict["crossview_pred"] == conflict["label"]
    ).astype(int)

    merged = conflict.merge(align[["sample_id", "building_ratio"]], on="sample_id")
    merged["dataset"] = label
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="B3: building_ratio vs crossview gain")
    parser.add_argument("--wildfire-alignment", required=True)
    parser.add_argument("--hurricane-alignment", required=True)
    parser.add_argument("--wildfire-conflict", required=True)
    parser.add_argument("--hurricane-conflict", required=True)
    parser.add_argument(
        "--output",
        default="outputs/figures/building_ratio_vs_crossview_gain.pdf",
    )
    args = parser.parse_args()

    wf = load_and_merge(args.wildfire_alignment, args.wildfire_conflict, "wildfire")
    ian = load_and_merge(args.hurricane_alignment, args.hurricane_conflict, "hurricane")
    combined = pd.concat([wf, ian], ignore_index=True)

    X = combined["building_ratio"].values.reshape(-1, 1)
    y = combined["crossview_correct"].values
    lr = LogisticRegression(random_state=42, max_iter=500)
    lr.fit(X, y)

    r, p = spearmanr(combined["building_ratio"], combined["crossview_correct"])
    print(f"Spearman r = {r:.3f}   p = {p:.4f}   n = {len(combined)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.default_rng(42)

    for ds, color in COLORS.items():
        sub = combined[combined["dataset"] == ds]
        jitter = rng.uniform(-0.015, 0.015, len(sub))
        ax.scatter(
            sub["building_ratio"],
            sub["crossview_correct"] + jitter,
            alpha=0.45,
            s=18,
            color=color,
            edgecolors="none",
            label=ds,
        )

    x_range = np.linspace(0, combined["building_ratio"].max() * 1.05, 300).reshape(-1, 1)
    ax.plot(
        x_range,
        lr.predict_proba(x_range)[:, 1],
        color="black",
        linewidth=2,
        label=f"logistic fit  (r = {r:.2f}, p = {p:.3f})",
    )

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Building ratio (SegFormer)", fontsize=11)
    ax.set_ylabel("P(crossview correct | conflict)", fontsize=11)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlim(-0.01, combined["building_ratio"].max() * 1.05)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_title(
        "Target alignment predicts cross-view conflict resolution",
        fontsize=11,
        pad=8,
    )

    # Dataset mean annotations
    for ds, color in COLORS.items():
        sub = combined[combined["dataset"] == ds]
        mx = sub["building_ratio"].mean()
        my = lr.predict_proba([[mx]])[0, 1]
        ax.annotate(
            ds,
            xy=(mx, my),
            xytext=(mx + 0.01, my - 0.08),
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
