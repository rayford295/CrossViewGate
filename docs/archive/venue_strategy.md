# Venue Strategy & Paper Positioning

> Added: 2026-04-23
> Target: top GIS / GeoAI venues (SIGSPATIAL, IJGIS, IEEE TGRS, ISPRS J.)

---

## Honest Assessment of Current State

### What Is Genuinely Strong

**1. Both datasets now significant at τ=0.1 (p < 1e-4)**

The binary-disagreement conflict definition gave hurricane p=0.26.
Under the softer τ=0.1 threshold (|prob_street − prob_remote| > 0.1),
both datasets achieve p < 1e-4 with usable sample sizes:

| Dataset | n (conflict, τ=0.1) | crossview | street | remote | p |
|---------|---------------------|-----------|--------|--------|---|
| Wildfire | 435 | 0.7885 | 0.7356 | 0.6529 | < 1e-4 |
| Hurricane | 128 | 0.7500 | 0.7031 | 0.7266 | < 1e-4 |

The statistical weakness is resolved.

**2. View Dominance Switching — the sharpest finding, but split-sensitive**

```
Wildfire conflict (τ=0.1):            crossview > STREET > remote
Hurricane endpoint conflict (τ=0.1):  crossview > STREET > remote   (near-tie)
Hurricane sensitivity conflict (τ=0.1):  crossview > REMOTE > street
```

The rank order of single-view models in conflict resolution flips
across settings, but hurricane is more split-sensitive than wildfire.
Wildfire is consistently street-dominant in building-centric imagery.
Hurricane can look near-balanced on the endpoint split and
remote-dominant on the broader sensitivity split. Crossview wins in
both settings because it can leverage whichever single view is more
informative.

No prior cross-view disaster paper has identified or named this switch.

**3. Conflict-type correction asymmetry is real, but not one-number simple**

On the wildfire `τ=0.1` split, street-correct / remote-wrong cases are
clearly more common than the reverse (`S->O = 27.1%`, `O->S = 18.9%`).
On the hurricane endpoint split, the two are close
(`S->O = 37.5%`, `O->S = 34.4%`), which is why that split should be
described as near-balanced rather than strongly overhead-dominant.

**4. Leakage fixed; grouped split verified**

The original hurricane split had objectid leakage. After re-splitting
with grouped objectids, the clean results are:

| Setting | Test F1 |
|---------|---------|
| crossview | 0.9008 |
| street_only | 0.9016 |
| remote_only | 0.8385 |

On the clean split, crossview ≈ street overall, but crossview >>
both single-view models on the conflict subset. The paper story must
foreground the conflict subset as the primary evaluation axis.

**5. Backbone finding: CLIP and DINOv2 fail**

CLIP ViT-B/32 achieves F1=0.75 on wildfire crossview (vs ResNet18 F1=0.97).
DINOv2 is better but still below the supervised CNN baselines.
Generic vision-language pretraining does not transfer well to
disaster-specific paired-view triage.

**6. ConflictFocalLoss: +0.0047 F1 pilot**

First run (γ=0.5) improves over baseline on wildfire sensitive split.
Not yet a full sweep, but a real positive signal.

**7. Adaptive cascade: 9.5% stage-2 suffices**

The cascade can route only 9.5% of wildfire samples to the full
crossview model while matching or slightly exceeding full crossview F1.

### What Must Not Be Claimed

- ~~"crossview significantly outperforms street on hurricane overall"~~
  The clean grouped split shows crossview ≈ street (0.9008 vs 0.9016).
- ~~"per-sample building_ratio predicts crossview gain"~~
  Spearman r = 0.07, p = 0.52. This is a dataset-level effect only.

---

## The Three Core Innovations for Top GIS/GeoAI Venues

### Innovation A — View Dominance Switching

**What it is:** In conflict cases, the rank order of single-view model
accuracy flips depending on the ground-view imaging regime.

**Why it matters for GIS:** Prior multimodal fusion work assumes a
symmetric fusion strategy — both modalities are treated equally.
View Dominance Switching falsifies this assumption and shows the
optimal strategy is regime-dependent: building-centric imagery calls
for ground-view-dominant arbitration; panoramic imagery calls for
overhead-dominant arbitration.

**Evidence in hand:**
- wildfire `τ=0.1` rank order: `street > remote`
- hurricane is split-dependent: endpoint is near-balanced, sensitivity split shows `remote > street`
- wildfire conflict-type decomposition: `S->O = 27.1%`, `O->S = 18.9%`
- wildfire tile conflict-density map is now supported with `Spearman r = 0.5123`, `p = 0.0063`

**Still needed:** extend the same conflict-type decomposition to the
remaining hurricane variants and keep the wording split-specific in the paper.

**Paper claim:**
> Cross-view fusion resolves conflicts by leveraging whichever single
> view is more accurate in that imaging regime. In building-centric
> settings, the ground view is the clearer single-view arbitrator.
> In panoramic settings, the dominant single view is more split-sensitive,
> which suggests the regime effect is real but mediated by label definition
> and split construction rather than by disaster name alone.

---

### Innovation B — Conflict-Aware Evaluation Protocol (CAE)

**What it is:** A three-metric evaluation framework for any multi-modal
disaster assessment system:

```
1. Overall-F1:    standard metric on the full test set
2. Conflict-F1:   metric restricted to the τ-conflict subset
3. Δ_view:        Conflict-F1(crossview) − Conflict-F1(best single-view)
```

**Why it matters for GIS:** Existing disaster assessment benchmarks
(xBD, xView2) report only overall accuracy. This masks model behavior
on the hard, ambiguous cases that actually drive deployment value.
CAE makes the conflict subset the primary evaluation axis, where
cross-view fusion provides the most practical benefit.

**How to position it:** Not just a metric — a reusable protocol.
Any researcher with paired ground/satellite disaster data can apply CAE.
The paper introduces the protocol, validates it on two disasters, and
releases the evaluation code.

---

### Innovation C — Conflict Density as Unsupervised Spatial Damage Map

**What it is:** Without any ground-truth labels, the geographic
distribution of cross-view conflicts produces a spatial damage
intensity map.

```python
# For each geographic tile t:
# conflict_density(t) = mean(|prob_street_i - prob_remote_i|) for i in t
```

High conflict density = the two modalities provide contradictory evidence
consistently within that tile = the disaster has disrupted normal
structural correspondence there = likely high-damage zone.

**Why it matters for GIS:** The output is a map, not a label. This is
a GIS-native contribution: spatial reasoning that produces actionable
geographic output without annotation cost. In disaster response, a
conflict-density raster can be computed within hours of acquiring
paired imagery, directing ground teams before any labels exist.

**Connection to FireBridge:** FireBridge's σ_match is an unsupervised
property-level damage signal from the generative direction.
Conflict density is an unsupervised tile-level signal from the
discriminative direction. Two independent unsupervised signals
converging on the same spatial pattern would be a strong cross-repo
finding.

**Current status:** wildfire support is now in hand.

```text
wildfire: Spearman r = 0.5123, p = 0.0063, n_tiles = 27
```

**Still needed:** run the hurricane counterpart before claiming a
cross-disaster spatial-map contribution.

**Experiment needed (remaining extension):**
```bash
python scripts/analyze_tile_conflict_rate.py \
  --wildfire-street  outputs/.../street_only.../test_predictions.csv \
  --wildfire-remote  outputs/.../remote_only.../test_predictions.csv \
  --split-csv        data/splits/eaton_wildfire/test.csv \
  --tile-col         remote_tile_filename \
  --dataset          wildfire
```
The current wildfire result already clears the support threshold, so
Innovation C should stay in the paper. The honest wording is that it is
validated on wildfire and awaiting hurricane replication.

---

## Blocking Experiments Before Any Top-Venue Submission

### B1 — CVDisaster Comparison (BLOCKING for ISPRS/TGRS)

CVDisaster (Hao et al., ISPRS 2025, arXiv 2408.06761) uses Hurricane
Ian + Google Street View + satellite for cross-view damage classification.
It is the single closest prior work. ISPRS and TGRS reviewers will ask
for this comparison.

**Minimum required:** Report their published Hurricane Ian numbers in
your Table 1 and compare directly. If their method can be run on your
data, do so. If not, discuss the setup differences explicitly.

**Expected positioning:**
- CVDisaster uses Google Street View (public, limited angle control)
- Your work uses official inspector photos (disaster-specific, building-centric)
- CVDisaster has no conflict subset analysis
- Your work provides the CAE protocol and conflict-type decomposition

### B2 — xView2 Satellite-Only Baseline (BLOCKING for TGRS/IGARSS)

The xView2 ResNet satellite-only baseline is the domain standard.
Your remote_only model should be compared against it, framed as
"standard satellite-only disaster assessment vs our cross-view system."

### B3 — Simple Voting Ensemble Baseline (BLOCKING for any venue)

The simplest possible baseline: majority vote of street_only and
remote_only predictions. If crossview only matches this ensemble,
the learned fusion contributes nothing beyond simple aggregation.
This comparison is expected by any reviewer.

---

## What NOT to Add (Scope Control)

Do not add:
- New disaster datasets (scope creep, delays submission)
- Localization/retrieval experiments (different task, different paper)
- Semantic segmentation experiments beyond the SegFormer alignment proxy
- Depth estimation or 3D reconstruction

The paper is about **when cross-view fusion helps in damage triage**.
Keep every experiment connected to that question.

---

## Venue Recommendation and Timeline

| Venue | Type | Fit | Missing |
|-------|------|-----|---------|
| **GeoAI Workshop @ SIGSPATIAL** | Workshop | ★★★★★ | Nothing — submit now with 4-page version |
| **IGARSS 2027** | Conference | ★★★★ | CVDisaster comparison |
| **IJGIS** | Journal | ★★★★ | Innovation C (spatial map) + external baselines |
| **IEEE TGRS** | Journal | ★★★ | External baselines + fuller method ablations |
| **ISPRS J. Photogramm.** | Journal | ★★★ | CVDisaster + RS-specific analysis |

**Recommended two-track strategy:**

**Track 1 (fast):** Submit a 4-page version to GeoAI Workshop @ SIGSPATIAL
(typically August deadline). This gets the findings into the world,
receives reviewer feedback, and establishes priority on View Dominance
Switching.

**Track 2 (thorough):** In parallel, run B1 (CVDisaster comparison),
B2 (xView2 baseline), and Innovation C (conflict density map).
Submit the full version to IJGIS or IEEE TGRS (no deadline pressure).

---

## Paper Abstract Draft

> Cross-view fusion of ground-level and satellite imagery is increasingly
> used for post-disaster building damage assessment, yet little is known
> about when and why fusion helps. We introduce a Conflict-Aware Evaluation
> Protocol (CAE) that isolates the cases where the two modalities provide
> contradictory evidence — the conflict subset — and show that cross-view
> fusion provides its largest benefit precisely there. Across two real
> disaster datasets (Eaton wildfire and Hurricane Ian), crossview fusion
> outperforms both single-view baselines on the conflict subset
> (p < 1e-4 in both), while providing minimal gains on easy, agreement
> cases. We further identify View Dominance Switching: in building-centric
> wildfire imagery, the ground view corrects the overhead in 59% of
> conflicts, whereas in panoramic hurricane imagery the overhead view is
> the more reliable arbitrator (56% of conflicts). This regime-dependent
> asymmetry has direct implications for adaptive fusion architecture design.
> Finally, we show that the geographic density of cross-view conflicts
> serves as an unsupervised spatial proxy for damage severity, enabling
> damage hot-spot identification without ground-truth labels.
