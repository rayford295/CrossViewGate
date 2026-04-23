"""
Innovation I3: Tile-level conflict rate as an unsupervised damage localisation signal.

For each geographic tile, compute the fraction of samples where street_only
and remote_only disagree (conflict rate), then measure its Spearman correlation
with the tile's actual damage rate.

If r > 0.4 and p < 0.05, the conflict rate is a zero-annotation damage proxy.

Usage:
    python scripts/analyze_tile_conflict_rate.py \
        --street-preds  outputs/eaton_wildfire/triage_street_only_resnet18/test_predictions.csv \
        --remote-preds  outputs/eaton_wildfire/triage_remote_only_resnet18/test_predictions.csv \
        --split-csv     data/splits/eaton_wildfire/test.csv \
        --tile-col      remote_tile_filename \
        --label-col     label \
        --output-dir    outputs/analysis/tile_conflict_rate \
        --dataset       wildfire

Run once for wildfire, once for hurricane. Compare Spearman r across datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def main() -> None:
    parser = argparse.ArgumentParser(description="I3: tile conflict rate vs damage rate")
    parser.add_argument("--street-preds",  required=True)
    parser.add_argument("--remote-preds",  required=True)
    parser.add_argument("--split-csv",     required=True)
    parser.add_argument("--tile-col",      default="remote_tile_filename",
                        help="Column in split CSV that identifies the geographic tile")
    parser.add_argument("--label-col",     default="label")
    parser.add_argument("--min-tile-size", type=int, default=5,
                        help="Minimum number of samples per tile to include")
    parser.add_argument("--output-dir",    default="outputs/analysis/tile_conflict_rate")
    parser.add_argument("--dataset",       default="dataset")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    street = pd.read_csv(args.street_preds)[["sample_id", "pred_label"]].rename(
        columns={"pred_label": "pred_s"}
    )
    remote = pd.read_csv(args.remote_preds)[["sample_id", "pred_label"]].rename(
        columns={"pred_label": "pred_r"}
    )
    split = pd.read_csv(args.split_csv)

    required_cols = {"sample_id", args.label_col}
    if args.tile_col not in split.columns:
        raise ValueError(
            f"tile column '{args.tile_col}' not found in split CSV. "
            f"Available: {list(split.columns)}"
        )
    required_cols.add(args.tile_col)

    df = (
        split[list(required_cols)]
        .merge(street, on="sample_id")
        .merge(remote,  on="sample_id")
    )
    df["is_conflict"] = (df["pred_s"] != df["pred_r"]).astype(int)

    tile_stats = (
        df.groupby(args.tile_col)
        .agg(
            conflict_rate=(  "is_conflict",   "mean"),
            damage_rate=(    args.label_col,   "mean"),
            n=(              args.label_col,   "count"),
        )
        .query(f"n >= {args.min_tile_size}")
        .reset_index()
    )

    r, p = spearmanr(tile_stats["conflict_rate"], tile_stats["damage_rate"])
    n_tiles = len(tile_stats)

    print(f"\n{'='*55}")
    print(f"  Dataset: {args.dataset}   tiles (n>={args.min_tile_size}): {n_tiles}")
    print(f"{'='*55}")
    print(f"  Tile-level Spearman r = {r:.3f}   p = {p:.4f}")
    print(f"  Conflict rate  mean={tile_stats['conflict_rate'].mean():.3f}  "
          f"std={tile_stats['conflict_rate'].std():.3f}")
    print(f"  Damage rate    mean={tile_stats['damage_rate'].mean():.3f}  "
          f"std={tile_stats['damage_rate'].std():.3f}")

    if r > 0.4 and p < 0.05:
        print("\n  ✓ STRONG: conflict rate is a significant tile-level damage proxy.")
        print("    I3 hypothesis SUPPORTED — report in paper as unsupervised signal.")
    elif r > 0.2 and p < 0.10:
        print("\n  ~ WEAK: trend present but not conclusive.")
        print("    Soften claim: 'directional association, not a hard proxy.'")
    else:
        print("\n  ✗ NOT SUPPORTED: conflict rate does not predict tile damage rate.")
        print("    I3 hypothesis rejected for this dataset.")
    print(f"{'='*55}\n")

    tile_stats.to_csv(out / f"tile_stats_{args.dataset}.csv", index=False)

    summary = {
        "dataset": args.dataset,
        "n_tiles": n_tiles,
        "min_tile_size": args.min_tile_size,
        "spearman_r": float(r),
        "spearman_p": float(p),
        "supported": bool(r > 0.4 and p < 0.05),
    }
    with open(out / f"summary_{args.dataset}.json", "w") as f:
        json.dump(summary, f, indent=2)

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(
            tile_stats["conflict_rate"],
            tile_stats["damage_rate"],
            s=tile_stats["n"] * 2,
            alpha=0.55,
            color="#D95F42",
            edgecolors="none",
        )
        # Trend line
        z = np.polyfit(tile_stats["conflict_rate"], tile_stats["damage_rate"], 1)
        xr = np.linspace(tile_stats["conflict_rate"].min(),
                         tile_stats["conflict_rate"].max(), 100)
        ax.plot(xr, np.polyval(z, xr), color="black", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Tile conflict rate (street ≠ remote, no labels)", fontsize=11)
        ax.set_ylabel("Tile damage rate (ground truth)", fontsize=11)
        ax.set_title(
            f"{args.dataset}: conflict rate vs damage rate\n"
            f"Spearman r={r:.3f},  p={p:.4f},  n={n_tiles} tiles",
            fontsize=10,
        )
        plt.tight_layout()
        fig_path = out / f"tile_conflict_vs_damage_{args.dataset}.pdf"
        plt.savefig(fig_path, dpi=200)
        print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
