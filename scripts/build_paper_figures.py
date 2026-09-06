"""Publication figures for the CrossViewGate ISPRS manuscript (Figs 2-5).

Palette: validated reference categorical slots, fixed identity per method
across all figures. Yellow slot carries direct labels (contrast relief rule).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "street_only": "#1baf7a",
    "remote_only": "#4a3aa7",
    "crossview": "#2a78d6",
    "calib_avg": "#eda100",
    "gate": "#e34948",
    "oracle": "#52514e",
    "ink": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e5e4e0",
}
LABEL = {
    "street_only": "Street only",
    "remote_only": "Remote only",
    "concat_reference": "Concat",
    "crossview_reference": "Cross-view",
    "calibrated_probability_average": "Calibrated avg.",
    "gate3_linear": "Reliability gate",
}
DS_LABEL = {"altadena_3class": "Eaton wildfire (Altadena)", "ian_original": "Hurricane Ian", "milton_original": "Hurricane Milton"}

mpl.rcParams.update(
    {
        "font.size": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": COL["muted"],
        "axes.labelcolor": COL["ink"],
        "xtick.color": COL["muted"],
        "ytick.color": COL["muted"],
        "xtick.labelcolor": COL["ink"],
        "ytick.labelcolor": COL["ink"],
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote {name}")


def load_methods() -> pd.DataFrame:
    fusion = pd.read_csv(REPO / "outputs/analysis/calibration_fusion_v2/fusion_methods_raw.csv")
    gate = pd.read_csv(REPO / "outputs/analysis/reliability_gate_v2/gate_results_raw.csv")
    gate = gate[gate.variant == "gate3_linear_in_domain"].copy()
    gate["method"] = "gate3_linear"
    gate = gate.rename(columns={"conflict_accuracy": "conflict_accuracy"})
    keep = ["dataset", "seed", "method", "accuracy", "conflict_accuracy"]
    fusion_keep = fusion[fusion.method.isin(
        ["street_only", "remote_only", "concat_reference", "crossview_reference", "calibrated_probability_average"]
    )][keep + ["best_single_conflict_accuracy", "oracle_conflict_accuracy"]]
    gate_keep = gate[keep + ["best_single_conflict_accuracy", "oracle_conflict_accuracy"]]
    return pd.concat([fusion_keep, gate_keep], ignore_index=True)


# ---------------------------------------------------------------- Figure 2
def fig_oracle_gap(df: pd.DataFrame) -> None:
    datasets = ["altadena_3class", "ian_original", "milton_original"]
    methods = [
        ("concat_reference", COL["muted"]),
        ("crossview_reference", COL["crossview"]),
        ("calibrated_probability_average", COL["calib_avg"]),
        ("gate3_linear", COL["gate"]),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for row, ds in enumerate(datasets):
        y = (len(datasets) - 1 - row) * 1.0
        sub = df[df.dataset == ds]
        base = sub["best_single_conflict_accuracy"].mean()
        oracle = sub["oracle_conflict_accuracy"].mean()
        ax.hlines(y, base, oracle, color=COL["grid"], lw=5, zorder=1)
        for x_val, name in ((base, "best single"), (oracle, "oracle")):
            ax.plot([x_val], [y], marker="|", ms=13, color=COL["ink"], mew=1.4, zorder=3)
            ax.annotate(f"{name}\n{x_val:.2f}", (x_val, y), textcoords="offset points",
                        xytext=(0, -13), ha="center", va="top", fontsize=6, color=COL["muted"])
        for method, color in methods:
            value = sub[sub.method == method]["conflict_accuracy"].mean()
            ax.plot([value], [y], "o", ms=7, color=color, mec="white", mew=1.2, zorder=4)
        ax.annotate(DS_LABEL[ds], (base, y), textcoords="offset points", xytext=(-4, 12),
                    ha="left", va="bottom", fontsize=7.5, fontweight="bold", color=COL["ink"])
    handles = [plt.Line2D([], [], marker="o", ls="", ms=6, color=c, mec="white", mew=1.0) for _, c in methods]
    ax.legend(handles, [LABEL[m] for m, _ in methods], frameon=False, fontsize=6.5, ncol=4,
              loc="lower right", bbox_to_anchor=(1.0, -0.02), handletextpad=0.2, columnspacing=1.0)
    ax.set_xlim(0.47, 1.01)
    ax.set_ylim(-0.75, len(datasets) - 0.35)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Accuracy on conflict cases")
    ax.xaxis.grid(True, color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig2_oracle_gap")


# ---------------------------------------------------------------- Figure 3
def fig_gate(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.32})

    ax = axes[0]
    datasets = ["altadena_3class", "ian_original", "milton_original"]
    methods = [
        ("street_only", COL["street_only"]),
        ("remote_only", COL["remote_only"]),
        ("crossview_reference", COL["crossview"]),
        ("calibrated_probability_average", COL["calib_avg"]),
        ("gate3_linear", COL["gate"]),
    ]
    width = 0.15
    xs = np.arange(len(datasets))
    for index, (method, color) in enumerate(methods):
        means, stds = [], []
        for ds in datasets:
            values = df[(df.dataset == ds) & (df.method == method)]["conflict_accuracy"]
            means.append(values.mean())
            stds.append(values.std())
        pos = xs + (index - 2) * (width + 0.012)
        ax.bar(pos, means, width=width, color=color, yerr=stds, error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": COL["muted"]}, zorder=3)
    for ds_index, ds in enumerate(datasets):
        oracle = df[df.dataset == ds]["oracle_conflict_accuracy"].mean()
        ax.hlines(oracle, ds_index - 0.42, ds_index + 0.42, color=COL["oracle"], lw=0.9, ls=(0, (4, 2)), zorder=2)
        if ds_index == 0:
            ax.annotate("oracle", (ds_index - 0.42, oracle), ha="left", va="bottom", fontsize=6, color=COL["oracle"])
    ax.set_xticks(xs)
    ax.set_xticklabels(["Eaton\nwildfire", "Hurricane\nIan", "Hurricane\nMilton"])
    ax.set_ylabel("Conflict-case accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.yaxis.grid(True, color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("(a) Conflict-case accuracy by method", loc="left", pad=22)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in methods]
    ax.legend(handles, [LABEL[m] for m, _ in methods], frameon=False, fontsize=6, ncol=5,
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), handlelength=1.0, columnspacing=0.7, handletextpad=0.4)

    ax = axes[1]
    coef = pd.read_csv(REPO / "outputs/analysis/reliability_gate_v2/gate_linear_coefficients.csv")
    coef = coef[coef.dataset == "altadena_3class"]
    stats = coef.groupby("feature")["coefficient"].agg(["mean", "std"]).sort_values("mean")
    nice = {
        "building_ratio": "Building pixel ratio",
        "center_building_ratio": "Centered building ratio",
        "center_minus_global": "Building centering",
        "centroid_distance_norm": "Centroid distance",
        "street_confidence": "Street confidence",
        "remote_confidence": "Remote confidence",
        "street_entropy": "Street entropy",
        "remote_entropy": "Remote entropy",
        "confidence_gap": "Confidence gap",
        "js_divergence": "JS divergence",
        "views_disagree": "Views disagree",
    }
    ys = np.arange(len(stats))
    colors = [COL["street_only"] if value > 0 else COL["remote_only"] for value in stats["mean"]]
    ax.barh(ys, stats["mean"], xerr=stats["std"], height=0.62, color=colors,
            error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": COL["muted"]}, zorder=3)
    ax.axvline(0, color=COL["muted"], lw=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([nice.get(f, f) for f in stats.index], fontsize=6.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_xlabel("Linear gate coefficient")
    ax.set_title("(b) Gate coefficients (wildfire)", loc="left", pad=22)
    ax.xaxis.grid(True, color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True)
    ax.annotate("trust street view →", (1.0, 1.045), xycoords="axes fraction", ha="right", fontsize=6.5, color=COL["street_only"], fontweight="bold")
    ax.annotate("← trust remote view", (0.0, 1.045), xycoords="axes fraction", ha="left", fontsize=6.5, color=COL["remote_only"], fontweight="bold")
    save(fig, "fig3_gate")


# ---------------------------------------------------------------- Figure 4
def fig_fov() -> None:
    fig = plt.figure(figsize=(7.0, 3.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], width_ratios=[2.1, 1.0, 1.0], hspace=0.42, wspace=0.18)

    pano_path = Path(r"C:\Users\yyang295\Desktop\disaster-dataset-Yifan-all\IAN_hurricane\images\003128_svi.png")
    log = pd.read_csv(REPO / "data/fov_intervention/ian_original/crop_log.csv")
    row = log[log.sample_id == 3128].iloc[0]

    ax = fig.add_subplot(grid[0, :])
    pano = np.asarray(Image.open(pano_path))
    ax.imshow(pano)
    h, w = pano.shape[:2]
    y0 = (h - 256) // 2
    placements = (
        (row.building_center_x, COL["gate"], "building-centered", -60),
        (row.random_center_x, "#3d3c3a", "random", 60),
    )
    for center, color, name, dx in placements:
        x0 = center - 128
        for shift in (0, -w, w):
            ax.add_patch(Rectangle((x0 + shift, y0), 256, 256, fill=False, ec=color, lw=1.6))
        ax.annotate(name, (min(max(center + dx, 130), w - 90), y0 - 12), ha="center", va="bottom",
                    fontsize=7, color=color, fontweight="bold")
    ax.set_title("(a) One panorama, two geometrically identical 90° crops", loc="left")
    ax.axis("off")

    for col, (variant, color) in enumerate((("building", COL["gate"]), ("random", COL["muted"]))):
        ax = fig.add_subplot(grid[1, col + 1])
        crop = np.asarray(Image.open(REPO / f"data/fov_intervention/ian_original/{variant}/3128.png"))
        ax.imshow(crop)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.8)
            spine.set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{variant} crop", fontsize=7, color=color, fontweight="bold")

    ax = fig.add_subplot(grid[1, 0])
    raw = pd.read_csv(REPO / "outputs/analysis/fov_intervention/fov_intervention_raw.csv")
    variants = ["original_panorama", "fov_building", "fov_random"]
    names = ["original\npanorama", "building\ncentered", "random\ncontrol"]
    colors = [COL["crossview"], COL["gate"], COL["muted"]]
    xs = np.arange(2)
    width = 0.24
    for index, (variant, color) in enumerate(zip(variants, colors)):
        means, dots = [], []
        for ds in ["ian_original", "milton_original"]:
            values = raw[(raw.dataset == ds) & (raw.variant == variant)]["conflict_gain"]
            means.append(values.mean())
            dots.append(values.to_numpy())
        pos = xs + (index - 1) * (width + 0.02)
        ax.bar(pos, means, width=width, color=color, zorder=3,
               label=names[index].replace("\n", " "))
        for p, seed_values in zip(pos, dots):
            ax.plot(np.full(len(seed_values), p), seed_values, "o", ms=2.4, color=COL["ink"], alpha=0.55, zorder=4)
    ax.axhline(0, color=COL["muted"], lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Hurricane Ian", "Hurricane Milton"])
    ax.set_ylabel("Conflict gain\n(cross-view − best single)")
    ax.set_title("(b) Fusion benefit by crop variant", loc="left", pad=14)
    ax.legend(frameon=False, fontsize=5.8, ncol=3, loc="lower left", bbox_to_anchor=(-0.02, 0.99),
              handlelength=1.0, columnspacing=0.7, handletextpad=0.4)
    ax.yaxis.grid(True, color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig4_fov_intervention")


# ---------------------------------------------------------------- Figure 5
def fig_density() -> None:
    tiles = pd.read_csv(REPO / "outputs/analysis/conflict_density_v2/alt_grid01/tiles_altadena_3class.csv")
    summary = pd.read_csv(REPO / "outputs/analysis/conflict_density_v2/alt_grid01/conflict_density_summary.csv").iloc[0]
    grid = 0.01
    fig = plt.figure(figsize=(7.0, 2.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.2, 1.0, 1.0], wspace=0.42, left=0.02, right=0.99, top=0.86, bottom=0.16)

    def tile_map(ax, column, cmap, title):
        values = tiles[column]
        norm = mpl.colors.Normalize(values.min(), values.max())
        for _, tile in tiles.iterrows():
            ax.add_patch(Rectangle((tile.tile_lon, tile.tile_lat), grid, grid,
                                   facecolor=cmap(norm(tile[column])), edgecolor="white", lw=0.6))
        ax.set_xlim(tiles.tile_lon.min() - 0.003, tiles.tile_lon.max() + grid + 0.003)
        ax.set_ylim(tiles.tile_lat.min() - 0.010, tiles.tile_lat.max() + grid + 0.003)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(34.18)), adjustable="datalim")
        ax.set_title(title, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            orientation="horizontal", fraction=0.05, pad=0.04, shrink=0.75)
        cbar.ax.tick_params(labelsize=5.5, length=2)
        cbar.outline.set_visible(False)

    blues = mpl.colors.LinearSegmentedColormap.from_list("seq_blue", ["#f2f6fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    aquas = mpl.colors.LinearSegmentedColormap.from_list("seq_aqua", ["#eefaf5", "#c9efe1", "#84dcbd", "#1baf7a", "#0e7d56", "#075e40"])

    tile_map(fig.add_subplot(gs[0, 0]), "damage", aquas, "(a) Damage (labels)")
    tile_map(fig.add_subplot(gs[0, 1]), "conflict_hard", blues, "(b) Conflict density")

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(tiles.conflict_hard, tiles.damage, s=np.sqrt(tiles.n) * 3.2, color=COL["crossview"], alpha=0.75, edgecolor="white", lw=0.5, zorder=3)
    slope, intercept = np.polyfit(tiles.conflict_hard, tiles.damage, 1)
    xr = np.linspace(tiles.conflict_hard.min(), tiles.conflict_hard.max(), 50)
    ax.plot(xr, slope * xr + intercept, color=COL["ink"], lw=0.9, ls=(0, (4, 2)))
    ax.set_xlabel("Tile conflict density")
    ax.set_ylabel("Tile mean damage")
    ax.set_title("(c) Density vs. damage", loc="left")
    p_text = "p < 0.001" if summary.conflict_hard_spearman_p < 0.001 else f"p = {summary.conflict_hard_spearman_p:.3f}"
    ax.annotate(f"Spearman r = {summary.conflict_hard_spearman_r:.2f}\n{p_text}",
                (0.04, 0.96), xycoords="axes fraction", va="top", fontsize=6.5, color=COL["ink"])
    ax.yaxis.grid(True, color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[0, 3])
    ax.scatter(tiles.uncertainty, tiles.damage, s=np.sqrt(tiles.n) * 3.2, color=COL["muted"], alpha=0.7, edgecolor="white", lw=0.5, zorder=3)
    slope, intercept = np.polyfit(tiles.uncertainty, tiles.damage, 1)
    xr = np.linspace(tiles.uncertainty.min(), tiles.uncertainty.max(), 50)
    ax.plot(xr, slope * xr + intercept, color=COL["ink"], lw=0.9, ls=(0, (4, 2)))
    ax.set_xlabel("Tile mean uncertainty")
    ax.set_title("(d) Uncertainty control", loc="left")
    ax.annotate(f"Spearman r = {summary.uncertainty_spearman_r:.2f}\np = {summary.uncertainty_spearman_p:.3f}",
                (0.04, 0.96), xycoords="axes fraction", va="top", fontsize=6.5, color=COL["ink"])
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color=COL["grid"], lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig5_conflict_density")


# ---------------------------------------------------------------- Figure 1
def fig_overview() -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    base = Path(r"C:\Users\yyang295\Desktop\disaster-dataset-Yifan-all\Altadena_Images\Eaton_Fire_attachments_index_output\dataset\sample_00005")
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f5f4f2", ec=COL["muted"], fontsize=6.5, weight="normal", tc=COL["ink"]):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                color=tc, fontweight=weight, linespacing=1.4)

    def arrow(x0, y0, x1, y1, color=COL["muted"], lw=1.0, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9,
                                     color=color, lw=lw, shrinkA=2, shrinkB=2))

    # input images
    for img_name, y, label in (("street_view.jpg", 24.5, "street view"), ("remote_sensing.jpg", 4.5, "overhead view")):
        image = Image.open(base / img_name)
        image = image.resize((256, 256)) if img_name.endswith("remote_sensing.jpg") else image.resize((256, 192))
        extent = (2, 15, y, y + (13 * image.size[1] / image.size[0]) * (44 / 100) * (7.0 / 3.1))
        ax.imshow(np.asarray(image), extent=(2, 15, y, y + 13.5), aspect="auto", zorder=2)
        ax.add_patch(Rectangle((2, y), 13, 13.5, fill=False, ec=COL["muted"], lw=0.8, zorder=3))
        ax.text(8.5, y - 1.6, label, ha="center", va="top", fontsize=6.5, color=COL["ink"], fontweight="bold")

    # single-view models
    box(21, 27.5, 13, 7, "street model\n$f_s$", fc="#eafaf3", ec=COL["street_only"])
    box(21, 7.5, 13, 7, "remote model\n$f_r$", fc="#efedf9", ec=COL["remote_only"])
    arrow(15.3, 31, 20.6, 31)
    arrow(15.3, 11, 20.6, 11)

    # conflict split
    box(41, 17.5, 15, 8, "views disagree?\nconflict cases:\n10–33% of samples", fc="#fdf3e3", ec=COL["calib_avg"], weight="bold")
    arrow(34.3, 31, 41.2, 23.5)
    arrow(34.3, 11, 41.2, 19.5)

    # oracle gap annotation under conflict box
    ax.text(48.5, 13.2, "oracle single-view gap:\n0.37–0.41 unclaimed accuracy", ha="center", va="top",
            fontsize=6, color=COL["gate"], fontweight="bold")

    # gate
    box(63.5, 15.5, 17, 13,
        "reliability gate (linear)\nbuilding visibility\ncalibrated confidence\ncross-view disagreement",
        fc="#fdecec", ec=COL["gate"], weight="bold")
    arrow(56.3, 21.5, 63.1, 21.5)

    # outputs
    box(87, 25.5, 11, 8, "gated triage\nprediction", fc="#e9f1fb", ec=COL["crossview"], weight="bold")
    box(87, 8.5, 11, 8, "conflict-density\ndamage map\n(no labels)", fc="#e9f1fb", ec=COL["crossview"])
    arrow(80.9, 23.5, 86.6, 28.5)
    arrow(48.5, 17.1, 86.6, 12.5)

    # causal intervention note
    ax.text(63.5 + 8.5, 13.4, "validated causally by the\nfield-of-view intervention",
            ha="center", va="top", fontsize=5.8, color=COL["muted"], style="italic")

    ax.text(1, 43, "Which view should be trusted, where, and why?", fontsize=8.5, fontweight="bold", color=COL["ink"], va="top")
    save(fig, "fig1_overview")


if __name__ == "__main__":
    methods = load_methods()
    fig_overview()
    fig_oracle_gap(methods)
    fig_gate(methods)
    fig_fov()
    fig_density()
