from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_pred(path: str, pred_name: str, prob_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    pred_col = "prediction" if "prediction" in df.columns else "pred_label"
    prob_col = "probability" if "probability" in df.columns else "prob_damaged"
    out = df[["sample_id", pred_col, prob_col]].rename(
        columns={pred_col: pred_name, prob_col: prob_name}
    )
    out["sample_id"] = (
        out["sample_id"].astype(str).str.replace(r"^tensor\((.+)\)$", r"\1", regex=True)
    )
    return out


def aggregate_tile_metrics(
    split_csv: str,
    street_csv: str,
    remote_csv: str,
    cross_csv: str,
    tile_csv: str,
) -> pd.DataFrame:
    split = pd.read_csv(split_csv)
    split["sample_id"] = split["sample_id"].astype(str)

    label_col = "binary_label" if "binary_label" in split.columns else "label"
    keep_cols = ["sample_id", "latitude", "longitude", "remote_tile_filename", label_col]
    split = split[keep_cols].rename(columns={label_col: "gt"})

    street = load_pred(street_csv, "pred_s", "prob_s")
    remote = load_pred(remote_csv, "pred_r", "prob_r")
    cross = load_pred(cross_csv, "pred_c", "prob_c")

    df = split.merge(street, on="sample_id").merge(remote, on="sample_id").merge(cross, on="sample_id")
    df = df[df["remote_tile_filename"].notna()].copy()

    df["correct_s"] = (df["pred_s"] == df["gt"]).astype(float)
    df["correct_r"] = (df["pred_r"] == df["gt"]).astype(float)
    df["correct_c"] = (df["pred_c"] == df["gt"]).astype(float)
    df["best_single_correct"] = df[["correct_s", "correct_r"]].max(axis=1)

    tile_stats = pd.read_csv(tile_csv)
    tile_geo = (
        df.groupby("remote_tile_filename")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            n=("gt", "count"),
            crossview_acc=("correct_c", "mean"),
            best_single_acc=("best_single_correct", "mean"),
            street_acc=("correct_s", "mean"),
            remote_acc=("correct_r", "mean"),
        )
        .reset_index()
    )
    tile_geo["crossview_gain"] = tile_geo["crossview_acc"] - tile_geo["best_single_acc"]
    tile_geo["dominant_view_score"] = tile_geo["street_acc"] - tile_geo["remote_acc"]

    merged = tile_geo.merge(tile_stats, on="remote_tile_filename", how="left", suffixes=("_samples", "_tile"))
    if "n_samples" in merged.columns:
        merged["n"] = merged["n_samples"]
    elif "n" not in merged.columns and "n_tile" in merged.columns:
        merged["n"] = merged["n_tile"]
    return merged


def square_size(n: pd.Series) -> np.ndarray:
    return np.clip(70 + 7 * np.sqrt(n.to_numpy()), 70, 170)


def style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.15, linewidth=0.6)
    ax.set_facecolor("#fbfbf7")
    for spine in ax.spines.values():
        spine.set_color("#8c8c84")
        spine.set_linewidth(0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build wildfire GeoAI-style map figures.")
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--street-preds", required=True)
    parser.add_argument("--remote-preds", required=True)
    parser.add_argument("--cross-preds", required=True)
    parser.add_argument("--tile-stats-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/wildfire_maps")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tile_df = aggregate_tile_metrics(
        args.split_csv,
        args.street_preds,
        args.remote_preds,
        args.cross_preds,
        args.tile_stats_csv,
    )
    tile_df.to_csv(outdir / "wildfire_tile_map_metrics.csv", index=False)

    sizes = square_size(tile_df["n"])
    lon = tile_df["longitude"]
    lat = tile_df["latitude"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), constrained_layout=True)
    fig.patch.set_facecolor("#f4f2ea")

    panels = [
        ("damage_rate", "YlOrRd", (0.0, 1.0), "Observed Damage Rate"),
        ("conflict_rate", "PuBuGn", (0.0, max(0.2, float(tile_df["conflict_rate"].max()))), "Conflict Density"),
        ("crossview_gain", "RdYlGn", (float(min(-0.05, tile_df["crossview_gain"].min())), float(max(0.05, tile_df["crossview_gain"].max()))), "Cross-View Gain"),
    ]

    for ax, (col, cmap, vlim, title) in zip(axes, panels):
        sc = ax.scatter(
            lon,
            lat,
            c=tile_df[col],
            cmap=cmap,
            vmin=vlim[0],
            vmax=vlim[1],
            s=sizes,
            marker="s",
            edgecolors="#3a3934",
            linewidths=0.55,
            alpha=0.95,
        )
        style_axes(ax, title)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        if col == "crossview_gain":
            cbar.set_label("crossview acc - best single-view acc", fontsize=9)
        elif col == "conflict_rate":
            cbar.set_label("mean disagreement rate by tile", fontsize=9)
        else:
            cbar.set_label("ground-truth positive rate by tile", fontsize=9)

    fig.suptitle(
        "Wildfire Spatial Pattern Maps: Damage, Conflict Density, and Cross-View Gain",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(outdir / "wildfire_spatial_maps.png", dpi=240, bbox_inches="tight")
    fig.savefig(outdir / "wildfire_spatial_maps.pdf", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    fig2.patch.set_facecolor("#f4f2ea")
    sc2 = ax.scatter(
        lon,
        lat,
        c=tile_df["dominant_view_score"],
        cmap="coolwarm",
        vmin=-0.4,
        vmax=0.4,
        s=sizes,
        marker="s",
        edgecolors="#3a3934",
        linewidths=0.55,
        alpha=0.95,
    )
    style_axes(ax, "Street-vs-Remote Dominance by Tile")
    cbar2 = fig2.colorbar(sc2, ax=ax, shrink=0.86, pad=0.02)
    cbar2.set_label("street acc - remote acc", fontsize=9)
    fig2.savefig(outdir / "wildfire_view_dominance_map.png", dpi=240, bbox_inches="tight")
    fig2.savefig(outdir / "wildfire_view_dominance_map.pdf", dpi=240, bbox_inches="tight")
    plt.close(fig2)

    print(f"Saved tile metrics to {outdir / 'wildfire_tile_map_metrics.csv'}")
    print(f"Saved main map panel to {outdir / 'wildfire_spatial_maps.png'}")
    print(f"Saved dominance map to {outdir / 'wildfire_view_dominance_map.png'}")


if __name__ == "__main__":
    main()
