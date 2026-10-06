"""Single-column rendering of the linear gate coefficients (camera-ready Fig. 2).

The GeoSearch '26 camera-ready has no room for the two-panel ``fig3_gate``
figure at a legible size, and panel (a) duplicates Table 1. This script
re-draws panel (b) alone at single-column width, using the same palette and
typography as ``build_paper_figures.py``.

The result CSVs (``outputs/analysis/reliability_gate_v2/``) are not tracked in
the repository, so the values below were digitized from the vector geometry
of ``figures/fig3_gate.pdf`` (bar extents and error-bar ends measured in PDF
points against the axis ticks; precision about 0.01). They match the
tracked coefficient table in ``docs/results/reliability_gate_results_v2.md``
exactly. Note: these are coefficients of the two-expert (street vs. remote)
gate, which has a single logit; the three-expert headline gate has no
single "trust street" coefficient. Re-run
``build_paper_figures.py`` on the experiment machine to regenerate from data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures"

STREET = "#1baf7a"
REMOTE = "#4a3aa7"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e5e4e0"

# (feature, mean coefficient, std over five seeds); order = figure order, top to bottom
COEF = [
    ("Building centering", 0.755, 0.330),
    ("Views disagree", 0.385, 0.631),
    ("Street confidence", 0.372, 0.276),
    ("Confidence gap", 0.339, 0.159),
    ("Centroid distance", 0.233, 0.186),
    ("Remote entropy", 0.230, 0.088),
    ("Centered building ratio", -0.132, 0.320),
    ("Remote confidence", -0.173, 0.251),
    ("JS divergence", -0.236, 0.360),
    ("Building pixel ratio", -0.446, 0.305),
    ("Street entropy", -0.469, 0.566),
]

mpl.rcParams.update(
    {
        "font.size": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)


def main() -> None:
    names = [c[0] for c in COEF]
    means = np.array([c[1] for c in COEF])
    stds = np.array([c[2] for c in COEF])
    y = np.arange(len(COEF))[::-1]

    fig, ax = plt.subplots(figsize=(3.33, 2.35))
    colors = [STREET if m > 0 else REMOTE for m in means]
    ax.barh(y, means, color=colors, height=0.62, zorder=3)
    ax.errorbar(means, y, xerr=stds, fmt="none", ecolor=MUTED, elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=4)
    ax.axvline(0, color=MUTED, lw=0.7, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_xlabel("Linear gate coefficient")
    ax.tick_params(axis="y", length=0)
    ax.text(-0.04, 1.02, "← trust remote view", transform=ax.transAxes, ha="right", va="bottom", color=REMOTE, fontsize=7.5, fontweight="bold")
    ax.text(0.04, 1.02, "trust street view →", transform=ax.transAxes, ha="left", va="bottom", color=STREET, fontsize=7.5, fontweight="bold")
    # keep the direction labels centred on the zero line
    x0 = (0 - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
    for t, dx, ha in ((ax.texts[0], -0.02, "right"), (ax.texts[1], 0.02, "left")):
        t.set_x(x0 + dx)
        t.set_ha(ha)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig3_gate_coef.pdf")
    fig.savefig(OUT / "fig3_gate_coef.png")


if __name__ == "__main__":
    main()
