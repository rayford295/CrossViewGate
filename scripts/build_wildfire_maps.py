from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle


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


def set_theme() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def infer_cell_size(values: np.ndarray, fallback: float) -> float:
    uniq = np.unique(np.round(values, 6))
    if uniq.size < 2:
        return fallback
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 1e-6]
    if diffs.size == 0:
        return fallback
    return float(np.median(diffs))


def format_axes(ax: plt.Axes, title: str, bbox: tuple[float, float, float, float]) -> None:
    ax.set_title(title, loc="left", fontweight="bold", color="#222222")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.8)


def draw_tile_panel(
    ax: plt.Axes,
    tile_df: pd.DataFrame,
    value_col: str,
    title: str,
    cmap,
    norm,
    tile_w: float,
    tile_h: float,
    bbox: tuple[float, float, float, float],
) -> None:
    ax.add_patch(
        Rectangle(
            (bbox[0], bbox[2]),
            bbox[1] - bbox[0],
            bbox[3] - bbox[2],
            facecolor="#ffffff",
            edgecolor="#dddddd",
            linewidth=0.8,
            zorder=0,
        )
    )
    square_size = np.clip(155 + 2.3 * np.sqrt(tile_df["n"].fillna(1.0)), 155, 255)
    ax.scatter(
        tile_df["longitude"],
        tile_df["latitude"],
        c=tile_df[value_col],
        cmap=cmap,
        norm=norm,
        s=square_size,
        marker="s",
        edgecolors="#ffffff",
        linewidths=0.95,
        alpha=0.97,
        zorder=2,
    )
    ax.scatter(
        tile_df["longitude"],
        tile_df["latitude"],
        s=np.clip(6 + 0.8 * np.sqrt(tile_df["n"].fillna(1.0)), 6, 14),
        color="#4a3d34",
        alpha=0.18,
        linewidths=0.0,
        zorder=3,
    )
    format_axes(ax, title, bbox)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build wildfire GeoAI-style map figures.")
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--street-preds", required=True)
    parser.add_argument("--remote-preds", required=True)
    parser.add_argument("--cross-preds", required=True)
    parser.add_argument("--tile-stats-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/wildfire_maps")
    args = parser.parse_args()

    set_theme()
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

    lon = tile_df["longitude"].to_numpy()
    lat = tile_df["latitude"].to_numpy()
    tile_w = infer_cell_size(lon, 0.012) * 0.92
    tile_h = infer_cell_size(lat, 0.008) * 0.92
    bbox = (
        float(lon.min() - tile_w * 0.9),
        float(lon.max() + tile_w * 0.9),
        float(lat.min() - tile_h * 0.9),
        float(lat.max() + tile_h * 0.9),
    )

    damage_cmap = LinearSegmentedColormap.from_list(
        "wildfire_damage",
        ["#fff7df", "#f5cb73", "#ef8a3a", "#cf5330", "#8c1d2c"],
    )
    conflict_cmap = LinearSegmentedColormap.from_list(
        "wildfire_conflict",
        ["#ffffff", "#d7dde3", "#98b7c8", "#4c8ea3", "#0d5b57"],
    )
    gain_cmap = LinearSegmentedColormap.from_list(
        "wildfire_gain",
        ["#b53f2e", "#e9c46a", "#ffffff", "#93c47d", "#2f7d43"],
    )

    damage_norm = Normalize(vmin=0.0, vmax=1.0)
    conflict_max = float(max(0.20, np.nanmax(tile_df["conflict_rate"].to_numpy())))
    conflict_norm = Normalize(vmin=0.0, vmax=conflict_max)
    gain_min = float(min(-0.08, np.nanmin(tile_df["crossview_gain"].to_numpy())))
    gain_max = float(max(0.08, np.nanmax(tile_df["crossview_gain"].to_numpy())))
    gain_norm = TwoSlopeNorm(vmin=gain_min, vcenter=0.0, vmax=gain_max)

    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.7), constrained_layout=True)
    fig.patch.set_facecolor("#ffffff")

    panels = [
        ("damage_rate", "a. Damage Rate", damage_cmap, damage_norm, "Observed positive-label rate"),
        ("conflict_rate", "b. Conflict Density", conflict_cmap, conflict_norm, "Mean disagreement rate"),
        ("crossview_gain", "c. Cross-View Gain", gain_cmap, gain_norm, "Cross-view acc minus best single-view acc"),
    ]

    for ax, (value_col, title, cmap, norm, cbar_label) in zip(axes, panels):
        draw_tile_panel(ax, tile_df, value_col, title, cmap, norm, tile_w, tile_h, bbox)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.055, pad=0.08)
        cbar.set_label(cbar_label, fontsize=8.5)
        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_edgecolor("#666666")

    fig.savefig(outdir / "wildfire_spatial_maps.png", dpi=260, bbox_inches="tight", facecolor="white")
    fig.savefig(outdir / "wildfire_spatial_maps.pdf", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    fig2.patch.set_facecolor("#ffffff")
    dominance_norm = TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    dominance_cmap = LinearSegmentedColormap.from_list(
        "dominance",
        ["#31688e", "#b8c8d6", "#ffffff", "#d3b28b", "#8c4c2e"],
    )
    draw_tile_panel(
        ax,
        tile_df,
        "dominant_view_score",
        "Street vs. Remote Dominance",
        dominance_cmap,
        dominance_norm,
        tile_w,
        tile_h,
        bbox,
    )
    sm2 = mpl.cm.ScalarMappable(norm=dominance_norm, cmap=dominance_cmap)
    sm2.set_array([])
    cbar2 = fig2.colorbar(sm2, ax=ax, orientation="horizontal", fraction=0.065, pad=0.08)
    cbar2.set_label("Street accuracy minus remote accuracy", fontsize=8.5)
    cbar2.ax.tick_params(labelsize=8)
    cbar2.outline.set_edgecolor("#666666")
    fig2.savefig(outdir / "wildfire_view_dominance_map.png", dpi=260, bbox_inches="tight", facecolor="white")
    fig2.savefig(outdir / "wildfire_view_dominance_map.pdf", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    print(f"Saved tile metrics to {outdir / 'wildfire_tile_map_metrics.csv'}")
    print(f"Saved main map panel to {outdir / 'wildfire_spatial_maps.png'}")
    print(f"Saved dominance map to {outdir / 'wildfire_view_dominance_map.png'}")


if __name__ == "__main__":
    main()
