from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle, ConnectionPatch


ALTADENA_LAT = 34.1897
ALTADENA_LON = -118.1312


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build wildfire dataset locator map.")
    parser.add_argument("--tile-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/dataset_maps")
    return parser.parse_args()


def california_polygon() -> np.ndarray:
    # A lightweight schematic outline for figure-level geographic context.
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


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.tile_csv)
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()

    lon = df["longitude"].to_numpy()
    lat = df["latitude"].to_numpy()
    n = df["n"].fillna(df["n_samples"]).fillna(1).to_numpy()
    sizes = np.clip(35 + 5 * np.sqrt(n), 35, 120)

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.7),
        gridspec_kw={"width_ratios": [1.0, 1.4]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#f6f2ea")

    ca_poly = california_polygon()
    ax0.add_patch(
        Polygon(
            ca_poly,
            closed=True,
            facecolor="#e8dfcf",
            edgecolor="#746a5d",
            linewidth=1.2,
            zorder=1,
        )
    )
    ax0.scatter(
        [ALTADENA_LON],
        [ALTADENA_LAT],
        marker="*",
        s=220,
        color="#c44e2b",
        edgecolors="white",
        linewidths=0.9,
        zorder=3,
    )
    ax0.text(
        ALTADENA_LON + 0.45,
        ALTADENA_LAT + 0.12,
        "Altadena / Eaton Fire",
        fontsize=10,
        fontweight="bold",
        color="#4b3f35",
    )

    la_box = Rectangle(
        (-119.1, 33.45),
        1.5,
        1.35,
        fill=False,
        linestyle="--",
        linewidth=1.1,
        edgecolor="#8f3d2e",
        zorder=2,
    )
    ax0.add_patch(la_box)
    ax0.set_xlim(-125.0, -113.8)
    ax0.set_ylim(32.2, 42.4)
    ax0.set_title("California Context", fontsize=13, fontweight="bold", pad=10)
    ax0.set_xlabel("Longitude", fontsize=10)
    ax0.set_ylabel("Latitude", fontsize=10)
    ax0.grid(True, alpha=0.10)
    ax0.set_facecolor("#fbf9f4")
    for spine in ax0.spines.values():
        spine.set_color("#8f867a")
        spine.set_linewidth(0.8)

    sc = ax1.scatter(
        lon,
        lat,
        s=sizes,
        c=df["damage_rate"].fillna(0.0),
        cmap="YlOrRd",
        marker="s",
        edgecolors="#4a433a",
        linewidths=0.5,
        alpha=0.95,
        zorder=3,
    )
    ax1.set_title("Wildfire Tile Footprint in Altadena", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Longitude", fontsize=10)
    ax1.set_ylabel("Latitude", fontsize=10)
    ax1.grid(True, alpha=0.12)
    ax1.set_facecolor("#fbf9f4")
    for spine in ax1.spines.values():
        spine.set_color("#8f867a")
        spine.set_linewidth(0.8)

    lon_pad = 0.01
    lat_pad = 0.008
    ax1.set_xlim(lon.min() - lon_pad, lon.max() + lon_pad)
    ax1.set_ylim(lat.min() - lat_pad, lat.max() + lat_pad)
    ax1.text(
        lon.min() + 0.002,
        lat.max() + 0.002,
        "Tile centers sized by sample count",
        fontsize=9.5,
        color="#5d5448",
    )
    cbar = fig.colorbar(sc, ax=ax1, shrink=0.88, pad=0.02)
    cbar.set_label("Observed damage rate", fontsize=9)
    cbar.ax.tick_params(labelsize=8.5)

    con = ConnectionPatch(
        xyA=(-117.6, 34.8),
        coordsA=ax0.transData,
        xyB=(lon.min(), lat.max()),
        coordsB=ax1.transData,
        color="#8f3d2e",
        linewidth=1.0,
        linestyle="--",
    )
    fig.add_artist(con)

    fig.suptitle(
        "Wildfire Dataset Location and Local Tile Coverage",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    png_path = outdir / "wildfire_locator_map.png"
    pdf_path = outdir / "wildfire_locator_map.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
