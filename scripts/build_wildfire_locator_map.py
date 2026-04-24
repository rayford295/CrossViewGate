from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import ConnectionPatch, Polygon, Rectangle


ALTADENA_LAT = 34.1897
ALTADENA_LON = -118.1312


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build wildfire dataset locator map.")
    parser.add_argument("--tile-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/dataset_maps")
    return parser.parse_args()


def california_polygon() -> np.ndarray:
    return np.array(
        [
            [-124.35, 42.00],
            [-123.00, 42.00],
            [-122.00, 41.20],
            [-121.40, 40.40],
            [-120.90, 39.40],
            [-120.00, 38.20],
            [-119.40, 37.20],
            [-118.80, 36.10],
            [-118.10, 35.00],
            [-117.50, 34.30],
            [-117.10, 33.60],
            [-116.30, 33.00],
            [-114.65, 32.72],
            [-114.55, 34.90],
            [-114.62, 37.00],
            [-114.63, 39.00],
            [-114.63, 41.10],
            [-120.00, 42.00],
            [-124.35, 42.00],
        ]
    )


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


def format_geo_axes(ax: plt.Axes, facecolor: str) -> None:
    ax.set_facecolor(facecolor)
    ax.grid(True, color="#c8c1b5", linestyle=":", linewidth=0.5, alpha=0.55)
    for spine in ax.spines.values():
        spine.set_color("#8e8578")
        spine.set_linewidth(0.8)


def draw_north_arrow(ax: plt.Axes, x: float, y: float, length: float) -> None:
    ax.annotate(
        "",
        xy=(x, y + length),
        xytext=(x, y),
        arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "#4c4a45"},
        annotation_clip=False,
    )
    ax.text(x, y + length + 0.002, "N", ha="center", va="bottom", fontsize=9, color="#4c4a45")


def draw_scalebar(ax: plt.Axes, start_lon: float, start_lat: float, km: float) -> None:
    deg_lon = km / (111.32 * np.cos(np.deg2rad(start_lat)))
    ax.plot(
        [start_lon, start_lon + deg_lon],
        [start_lat, start_lat],
        color="#4c4a45",
        linewidth=2.0,
        solid_capstyle="butt",
        zorder=6,
    )
    ax.plot([start_lon, start_lon], [start_lat - 0.0007, start_lat + 0.0007], color="#4c4a45", linewidth=1.2)
    ax.plot(
        [start_lon + deg_lon, start_lon + deg_lon],
        [start_lat - 0.0007, start_lat + 0.0007],
        color="#4c4a45",
        linewidth=1.2,
    )
    ax.text(start_lon + deg_lon / 2.0, start_lat + 0.0014, f"{int(km)} km", ha="center", va="bottom", fontsize=8.5)


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_theme()

    df = pd.read_csv(args.tile_csv)
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    if "damage_rate" not in df.columns:
        df["damage_rate"] = 0.0
    if "n" not in df.columns:
        df["n"] = df.get("n_samples", 1)

    lon = df["longitude"].to_numpy()
    lat = df["latitude"].to_numpy()
    damage = df["damage_rate"].fillna(0.0).to_numpy()
    sample_n = df["n"].fillna(1).to_numpy()

    tile_w = infer_cell_size(lon, 0.012) * 0.90
    tile_h = infer_cell_size(lat, 0.008) * 0.90
    tile_size = np.clip(160 + 4.0 * np.sqrt(sample_n), 160, 320)

    damage_cmap = LinearSegmentedColormap.from_list(
        "wildfire_damage",
        ["#fff7df", "#f7c971", "#f1823a", "#cf4f2e", "#8f1d2c"],
    )
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(13.8, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.65], wspace=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    fig.patch.set_facecolor("#f5f1e8")

    ax0.add_patch(
        Rectangle(
            (-126.0, 31.5),
            14.0,
            12.5,
            facecolor="#edf2ef",
            edgecolor="none",
            zorder=0,
        )
    )
    ca_poly = california_polygon()
    ax0.add_patch(
        Polygon(
            ca_poly,
            closed=True,
            facecolor="#dfd6c4",
            edgecolor="#766b5d",
            linewidth=1.1,
            zorder=2,
        )
    )
    ax0.scatter(
        [ALTADENA_LON],
        [ALTADENA_LAT],
        marker="*",
        s=160,
        color="#c15a36",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )
    ax0.text(
        ALTADENA_LON + 0.35,
        ALTADENA_LAT + 0.08,
        "Altadena / Eaton Fire",
        fontsize=9.5,
        fontweight="bold",
        color="#5a4336",
        zorder=6,
    )
    focus_box = Rectangle(
        (-119.15, 33.45),
        1.6,
        1.35,
        facecolor="none",
        edgecolor="#b85a3c",
        linestyle="--",
        linewidth=1.0,
        zorder=4,
    )
    ax0.add_patch(focus_box)
    format_geo_axes(ax0, "#f8f5ee")
    ax0.set_xlim(-125.1, -113.7)
    ax0.set_ylim(32.2, 42.3)
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.set_title("a. Regional Setting", loc="left", fontweight="bold", color="#433c34")

    local_bbox = (
        float(lon.min() - tile_w * 0.85),
        float(lat.min() - tile_h * 0.85),
        float((lon.max() - lon.min()) + tile_w * 1.7),
        float((lat.max() - lat.min()) + tile_h * 1.7),
    )
    ax1.add_patch(
        Rectangle(
            (local_bbox[0], local_bbox[1]),
            local_bbox[2],
            local_bbox[3],
            facecolor="#f8f4ea",
            edgecolor="#d7ccbc",
            linewidth=0.8,
            zorder=0,
        )
    )
    ax1.scatter(
        lon,
        lat,
        s=tile_size,
        c=damage,
        cmap=damage_cmap,
        norm=norm,
        marker="s",
        edgecolors="#f3ede2",
        linewidths=1.0,
        alpha=0.96,
        zorder=2,
    )
    ax1.scatter(
        lon,
        lat,
        s=np.clip(8 + 1.2 * np.sqrt(sample_n), 8, 20),
        color="#5b4636",
        alpha=0.36,
        edgecolors="none",
        zorder=4,
    )
    ax1.text(
        0.03,
        0.97,
        "Tile color shows damage rate; center dots scale with sample count",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#5f564a",
        bbox={"facecolor": "#fbf8f1", "edgecolor": "none", "pad": 0.4, "alpha": 0.9},
    )
    draw_north_arrow(ax1, float(lon.max() + tile_w * 0.38), float(lat.min() + tile_h * 0.35), tile_h * 3.2)
    draw_scalebar(ax1, float(lon.max() - tile_w * 4.3), float(lat.min() + tile_h * 0.15), 2.0)
    format_geo_axes(ax1, "#fbf8f1")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(local_bbox[0], local_bbox[0] + local_bbox[2])
    ax1.set_ylim(local_bbox[1], local_bbox[1] + local_bbox[3])
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title("b. Local Wildfire Tile Footprint", loc="left", fontweight="bold", color="#433c34")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=damage_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, orientation="horizontal", fraction=0.07, pad=0.08)
    cbar.set_label("Observed damage rate", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_edgecolor("#8e8578")

    con = ConnectionPatch(
        xyA=(-117.55, 34.80),
        coordsA=ax0.transData,
        xyB=(local_bbox[0] + tile_w * 0.25, local_bbox[1] + local_bbox[3] - tile_h * 0.25),
        coordsB=ax1.transData,
        color="#b85a3c",
        linewidth=0.9,
        linestyle="--",
        alpha=0.8,
    )
    fig.add_artist(con)

    png_path = outdir / "wildfire_locator_map.png"
    pdf_path = outdir / "wildfire_locator_map.pdf"
    fig.savefig(png_path, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
