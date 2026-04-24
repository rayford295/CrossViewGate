# Writing Review: Suggestions for Top-Tier Venue Submission

This document records structured writing suggestions for the SIGSPATIAL 2026 paper
"When Cross-View Helps Most." Suggestions are ordered by impact and grouped by section.

---

## Applied Changes (Already in Draft)

The following changes have been applied directly to `paper/sigspatial2026_draft.md`:

- **Formatting**: Removed all code-style backtick notation from model names, metric
  names, and numerical values. Model variants now use *italics* (*street-only*,
  *remote-only*, *cross-view*); Greek letters τ, γ used for thresholds/hyperparameters.
- **Equations**: Added formal display-math definitions for (1) the CAE conflict subset
  $\mathcal{C}_\tau$ and the Δ_view metric; (2) the ConflictFocalLoss objective
  $\mathcal{L}_\text{CFL}$; (3) the conflict density map CD(*t*).
- **Prose quality**: Removed author self-evaluation phrases ("This is the sharpest
  mechanism finding," "This is a strong GeoAI result," etc.) that undermine academic
  tone. Results sections now let data speak directly.
- **Factual language**: Tightened several sentences (Section 3.4, 7.1, 7.4, 7.6,
  7.9, 7.10, 7.12) to be more precise and less informal.

---

## High-Priority Remaining Suggestions

### 1. Related Work: Add Specific Citations (Critical)

**Problem**: Section 2 contains no named prior works. Reviewers at competitive venues
will immediately notice this and question whether the authors have surveyed the field.

**Required additions**:

- **xBD/xView2** (Gupta et al., 2019): the benchmark standard for overhead-only
  satellite damage assessment. Must be cited to establish what "remote-only" baselines
  represent.
- **CVDisaster / Hao et al. (2025)**: the closest prior cross-view hurricane work.
  The paper currently says "the closest prior direction uses Hurricane Ian ground
  imagery"—name the paper explicitly and contrast more precisely. Specifically,
  explain what CAE adds over their evaluation framework.
- **Selective prediction / abstaining classifiers**: the adaptive cascade (Section 5.4)
  is related to the selective prediction literature (Geifman & El-Yaniv, 2017; Huang
  et al., 2023). Acknowledge this connection or a reviewer may raise it as an oversight.
- **Multimodal fusion for geospatial tasks**: cite at least one representative work on
  ground-satellite fusion outside disaster domains (e.g., cross-view geo-localization)
  to justify the claim that symmetric fusion is the standard assumption.
- **Focal Loss** (Lin et al., 2017): ConflictFocalLoss is structurally a reweighted
  focal-style loss. Cite the original to clarify the conceptual lineage.

**Suggested rewrite of the final Related Work paragraph**:
> "This paper differs from prior cross-view disaster work in three ways. First, it
> defines the conflict subset $\mathcal{C}_\tau$ as the primary evaluation target
> rather than average accuracy. Second, it uses a cross-disaster design to compare
> two observation regimes rather than evaluating a single dataset. Third, it translates
> disagreement analysis into spatial and algorithmic outputs—a conflict-density map, a
> conflict-aware training objective, and an adaptive cascade—operationalizing the
> conflict-centric view across the modeling pipeline."

---

### 2. Architecture: Add Missing Specifics (High Impact)

**Problem**: Section 5.1 does not specify encoder output dimension, fusion MLP
architecture, or training hyperparameters. Reviewers may request these for reproducibility.

**Add to Section 5.1** (or to an Appendix):
- Encoder backbone: ResNet18 pretrained on ImageNet-1k; output embedding dimension 512.
- Fusion head: four-vector concatenation → linear layer → binary classifier.
- Optimizer, learning rate schedule, batch size, number of epochs.
- Whether encoders are frozen or jointly fine-tuned during cross-view training.

---

### 3. Discussion: Deepen Mechanism Analysis (High Impact)

**Problem**: Section 8 is approximately 350 words—thin for the richness of the
experimental findings. Top-tier reviewers expect the discussion to explain *why*
the phenomena occur, not just *that* they occur.

**Suggested additions**:

1. **Why does the ground view dominate in wildfire but not hurricane?**
   The current explanation references "target alignment" via the segmentation proxy.
   Strengthen this: the wildfire inspection image is typically taken within 1–2 meters
   of the facade at a fixed angle; the hurricane panoramic image is taken from the
   street at varying distances with adjacent buildings and environmental clutter. This
   is not just an observation style difference—it reflects the data collection protocol
   of each dataset. Acknowledge that future datasets designed with controlled viewpoints
   could test whether the regime effect is learnable or purely a function of data
   collection protocol.

2. **When does conflict density fail as a spatial signal?**
   The hurricane dataset lacks tile-level metadata, preventing the spatial map analysis.
   But even if it had metadata, the spatial map relies on geographic clustering of
   similar damage patterns. In low-damage or spatially heterogeneous events, conflict
   density may decorrelate from observed damage. State this as a scope limitation.

3. **Relationship to selective prediction**:
   The adaptive cascade (Section 5.4) is functionally a selective prediction system
   that abstains from full cross-view inference when single-view agreement is high.
   Framing this in terms of selective prediction literature would strengthen the
   methodological contribution.

---

### 4. Limitations: Add Two Missing Items (Medium Impact)

**Current limitations section** addresses: (1) hurricane spatial map, (2) conservative
hurricane F1, (3) ConflictFocalLoss validated on one setting only.

**Missing**:
- **Binary triage only**: The paper studies two-class damage classification throughout.
  Most operational damage assessment uses 5-class scales (e.g., xBD: No Damage,
  Minor, Major, Destroyed, Unclassified). The conflict-aware framework may not extend
  trivially to multi-class settings where "disagreement" is less well-defined.
- **Dataset scope**: Both disasters are from the continental United States. The regime
  distinction (property-centric vs. panoramic) may not hold for datasets from urban
  environments in other countries where inspection protocols differ.

---

### 5. Abstract: Sharpen the Framing Sentence (Medium Impact)

**Current**: "Cross-view disaster assessment combines ground-level and overhead imagery
to infer building damage, but most prior work evaluates it only by average test accuracy."

**Suggested**: "Cross-view disaster assessment combines ground-level and overhead imagery
to infer building damage. Prior evaluation has relied almost entirely on aggregate
accuracy metrics, which mask the cases where fusion actually matters."

The current framing reads as a mild critique. The revision makes the gap sharper and
motivates the CAE contribution more directly.

---

### 6. Introduction: Remove Redundancy with Abstract (Low-Medium Impact)

The Introduction currently repeats phrases that also appear verbatim in the Abstract
(e.g., "conflict-dependent," "regime-dependent," "view dominance switching"). This is
acceptable for conference papers but will be noticed by careful reviewers. Either:

- Move the "conflict-dependent and regime-dependent" terminology first to the
  Introduction and use only a brief callback in the Abstract; or
- Accept the repetition and ensure the phrasing differs enough across both sections.

---

### 7. Figures: Required for Final Submission

The draft explicitly lists four figures needed. For a competitive submission, figure
quality matters as much as prose. Specific recommendations:

1. **View Dominance Switching figure**: Use grouped bars with error bars (bootstrap CI).
   Show wildfire and hurricane side-by-side with the single-view rank order annotated
   with arrows. Color the bars to distinguish regimes, not just datasets.

2. **CAE summary table**: A table (not a figure) is more appropriate here. Include all
   settings (wildfire endpoint/sensitive, hurricane endpoint/sensitivity/clean-split)
   with Overall-F1, Conflict-F1, and Δ_view. This becomes the paper's central exhibit.

3. **Wildfire spatial maps**: Use a consistent colormap (e.g., Reds for damage rate,
   Blues→Reds diverging for conflict density). Include a colorbar and geographic scale.
   The three panels (observed damage, conflict density, cross-view gain) should share
   the same tile grid so readers can compare spatially.

4. **ConflictFocalLoss sweep**: Show a two-panel plot: left panel Overall-F1 vs. γ,
   right panel Conflict-F1 vs. γ. Add a dashed horizontal line for the baseline
   (*cross-view*, γ = 0). This makes the non-monotone behavior (improvement then
   collapse at γ = 1.0) visually clear.

---

## Pre-Submission Checklist

- [ ] All citations in `references.bib` correspond to in-text `\cite{}` commands
- [ ] xBD/xView2, CVDisaster/Hao et al., Focal Loss, and selective prediction cited
- [ ] Encoder/training hyperparameters documented in paper or appendix
- [ ] All four figures finalized with consistent colormaps and axis labels
- [ ] CAE summary table included as main results exhibit
- [ ] Abstract ≤ 250 words (ACM SIGSPATIAL requirement)
- [ ] Page limit checked (10 pages + references for full papers)
- [ ] LaTeX compiled without warnings; no overfull hboxes in main text
- [ ] Figures embedded as vector PDF or high-DPI PNG (≥ 300 dpi)
- [ ] Acknowledgments include dataset and funding sources

---

## Summary Assessment

**Current strengths (ready for submission):**
- Experimental breadth: 12 result subsections with backbone, seed, label, and
  threshold sensitivity analysis—strong reviewability.
- Honest treatment of the clean hurricane split: shows scientific integrity.
- ConflictFocalLoss + adaptive cascade: two concrete method contributions beyond analysis.
- Tile-level spatial map with Spearman *r* = 0.512, *p* = 0.0063: strong empirical signal.

**Remaining gaps (must fix before submission):**
- Related Work needs named citations—this is a dealbreaker at SIGSPATIAL.
- Discussion needs ~200 more words of mechanism analysis.
- Architecture section needs training hyperparameters for reproducibility.
- Four figures must be finalized.

With these additions, the paper is competitive for SIGSPATIAL 2026 acceptance.
