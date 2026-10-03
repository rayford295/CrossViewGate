# Figures

Publication figure set (PNG 300 dpi for viewing, PDF vector for
submission). Regenerate with `python scripts/build_paper_figures.py` — all
panels are driven by the result CSVs under `outputs/analysis/` (converged
5-seed v2 protocol).

| File | Content |
| --- | --- |
| `fig0_pipeline` | Pipeline diagram (drawn by hand): street/remote models, conflict cases, linear trust gate, gated prediction and conflict-density map |
| `fig2_oracle_gap` | The oracle single-view gap per dataset, with each method placed inside the best-single→oracle span |
| `fig3_gate` | (a) Conflict-case accuracy by method (5 seeds, oracle reference); (b) linear gate coefficients on wildfire |
| `fig3_gate_coef` | Panel (b) of `fig3_gate` alone at single-column size, used as Fig. 2 of the GeoSearch camera-ready; built by `scripts/build_gate_coef_panel.py` (values digitized from the vector `fig3_gate.pdf`) |
| `fig4_fov_intervention` | Causal FOV intervention: crop windows on a real panorama, the two crops, and conflict gain by variant |
| `fig5_conflict_density` | Damage map vs label-free conflict-density map, rank correlation, and the uncertainty control |
| `fig6_qualitative_conflicts` | Qualitative conflict cases with per-model predictions and segmentation overlays |

Method colors are fixed across all figures (validated categorical palette):
street `#1baf7a`, remote `#4a3aa7`, cross-view `#2a78d6`, calibrated
averaging `#eda100`, reliability gate `#e34948`.
