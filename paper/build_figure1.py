from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
OUT_PNG = FIG_DIR / "figure1_overview.png"
OUT_PDF = FIG_DIR / "figure1_overview.pdf"


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.8),
    )


def draw_conflict_protocol(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    street_color = "#D96C3F"
    remote_color = "#4477AA"
    cross_color = "#1B4332"

    def box(x, y, w, h, text, fc, ec="black", text_color="black", size=11, weight="normal"):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.2,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=text_color, fontweight=weight)

    box(0.07, 0.72, 0.32, 0.14, "Street-only\npredictor", "#F6DDCF", ec=street_color, weight="bold")
    box(0.61, 0.72, 0.32, 0.14, "Remote-only\npredictor", "#DCE8F5", ec=remote_color, weight="bold")
    box(0.35, 0.44, 0.30, 0.15, "|p_street - p_remote|\n> tau = 0.1 ?", "#F3F4F6", weight="bold")
    box(0.05, 0.12, 0.36, 0.14, "Low disagreement:\naverage single-view scores", "#EFEFEF")
    box(0.59, 0.12, 0.36, 0.14, "High disagreement:\ncall crossview fusion", "#DCEFE6", ec=cross_color, weight="bold")

    arrow_kw = dict(arrowstyle="->", lw=1.8, color="black")
    ax.annotate("", xy=(0.46, 0.59), xytext=(0.28, 0.72), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.54, 0.59), xytext=(0.77, 0.72), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.25, 0.26), xytext=(0.42, 0.44), arrowprops=arrow_kw)
    ax.annotate("", xy=(0.77, 0.26), xytext=(0.58, 0.44), arrowprops=arrow_kw)

    ax.text(
        0.5,
        0.94,
        "Conflict-aware framing",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.03,
        "Average F1 hides the cases where fusion matters most.",
        ha="center",
        va="bottom",
        fontsize=11,
        style="italic",
    )


def draw_view_switch(ax) -> None:
    street_color = "#D96C3F"
    remote_color = "#4477AA"
    cross_color = "#1B4332"

    datasets = ["Wildfire\nsensitive", "Hurricane\nmoderate+severe"]
    street = np.array([0.6230, 0.7324])
    remote = np.array([0.5237, 0.7742])
    cross = np.array([0.6618, 0.7867])

    x = np.arange(len(datasets))
    w = 0.24
    ax.bar(x - w, street, width=w, label="street_only", color=street_color)
    ax.bar(x, remote, width=w, label="remote_only", color=remote_color)
    ax.bar(x + w, cross, width=w, label="crossview", color=cross_color)

    ax.set_ylim(0.45, 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Conflict-F1 (tau=0.1)", fontsize=11)
    ax.set_title("View Dominance Switching", fontsize=14, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    ax.text(x[0], 0.84, "street > remote", ha="center", va="top", fontsize=11, fontweight="bold", color=street_color)
    ax.text(x[1], 0.84, "remote > street", ha="center", va="top", fontsize=11, fontweight="bold", color=remote_color)

    for xpos, vals in zip(x, zip(street, remote, cross)):
        for dx, val in zip([-w, 0, w], vals):
            ax.text(xpos + dx, val + 0.008, f"{val:.3f}", ha="center", va="bottom", fontsize=9, rotation=90)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    alignment = mpimg.imread(FIG_DIR / "building_alignment_comparison.png")
    spatial = mpimg.imread(FIG_DIR / "wildfire_spatial_maps.png")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
        }
    )

    fig = plt.figure(figsize=(15.5, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], width_ratios=[1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    ax_a.imshow(alignment)
    ax_a.axis("off")
    ax_a.set_title("Spatial Observation Regimes", fontsize=14, fontweight="bold", pad=8)
    ax_a.text(
        0.5,
        -0.06,
        "Wildfire images are building-centric; hurricane images are panoramic and environment-dominant.",
        transform=ax_a.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
    )
    add_panel_label(ax_a, "(a)")

    draw_conflict_protocol(ax_b)
    add_panel_label(ax_b, "(b)")

    draw_view_switch(ax_c)
    add_panel_label(ax_c, "(c)")

    ax_d.imshow(spatial)
    ax_d.axis("off")
    ax_d.set_title("Spatial Consequence", fontsize=14, fontweight="bold", pad=8)
    ax_d.text(
        0.5,
        -0.06,
        "Wildfire conflict density tracks tile-level damage rate (Spearman r = 0.512, p = 0.0063).",
        transform=ax_d.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
    )
    add_panel_label(ax_d, "(d)")

    fig.suptitle(
        "Figure 1. Cross-view disaster triage is conflict-aware and regime-aware.",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
