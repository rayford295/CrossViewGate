# Figure 1 Storyboard

## Goal

Create a single opening figure that makes the paper's core claim obvious in
less than 10 seconds:

**Cross-view fusion helps most under conflict, and the dominant corrective view
depends on the spatial observation regime.**

## Layout

Use a four-panel horizontal layout for the conference version, or a 2x2 layout
for the journal version.

### Panel (a): Spatial Observation Regimes

Show one representative wildfire pair and one representative hurricane pair.

- left: wildfire ground image + overhead patch
- right: hurricane ground image + overhead patch

Annotations:

- wildfire: `building-centric`, `property-aligned`
- hurricane: `panoramic`, `environment-dominant`

Purpose:

- establish that the paper is not only comparing two disasters
- visually define the two regimes before any numbers appear

### Panel (b): Conflict-Aware Framing

Show a compact schematic of the evaluation logic.

- `street_only` predicts one probability
- `remote_only` predicts another probability
- `|p_street - p_remote| > tau` defines the conflict subset
- `crossview` is evaluated especially on those cases

Text callout:

`Average F1 hides the cases where fusion matters most.`

Purpose:

- explain why the paper does not use only aggregate evaluation

### Panel (c): View Dominance Switching

Use grouped bars for conflict accuracy (`tau=0.1`).

Recommended values:

- wildfire sensitive:
  - street `0.6230`
  - remote `0.5237`
  - crossview `0.6618`
- hurricane moderate+severe:
  - street `0.7324`
  - remote `0.7742`
  - crossview `0.7867`

Overlay arrows or labels:

- wildfire: `street > remote`
- hurricane: `remote > street`

Purpose:

- make the rank-order flip visible immediately
- this is the paper's main mechanism finding

### Panel (d): Spatial Consequence

Use the wildfire map outputs already generated:

- observed damage rate
- conflict density
- cross-view gain

Preferred composition:

- one compact triptych inset
- or a single conflict-density map with a small caption:
  `Conflict density correlates with tile damage rate (Spearman r = 0.512, p = 0.0063).`

Purpose:

- show that disagreement is not just a classifier diagnostic
- it becomes a GIS-native spatial signal

## Caption Draft

**Figure 1:** Cross-view disaster triage should be understood as
conflict-aware, regime-aware spatial reasoning. (a) The wildfire and hurricane
benchmarks differ not only by disaster type but also by spatial observation
regime: the wildfire ground view is property-centric, whereas the hurricane
ground view is panoramic and environment-dominant. (b) Our Conflict-Aware
Evaluation Protocol isolates the disagreement subset where paired-view fusion is
most useful. (c) On that subset, the dominant single-view arbitrator changes
across regimes: wildfire favors the street view, whereas broader hurricane
settings favor the overhead view. (d) In wildfire, cross-view conflict density
also forms a significant unsupervised spatial damage signal.

## Design Notes

- Avoid default Matplotlib styling; use a clean journal palette with one color
  per modality:
  - street: warm orange
  - remote: cool blue
  - crossview: deep charcoal or dark green
- Keep backgrounds white.
- Use the same typography as the final paper.
- Avoid decorative icons unless they clarify modality.
- Label panels `(a)–(d)` in the top-left corner of each panel.
- Make panel (c) visually dominant because it contains the central claim.

## What This Figure Must Achieve

If a reviewer looks only at Figure 1, they should understand:

1. why the paper is not just another multimodal classifier paper
2. what "conflict-aware" means
3. what "regime-aware" means
4. what the main discovery is
5. why the work belongs in GeoAI / GIS rather than only in generic vision
