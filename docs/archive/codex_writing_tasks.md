# Codex Writing Tasks — Top-Tier Submission Polish

> Written for Codex. Read this file first, then read `docs/writing_review.md` for
> detailed rationale. Execute tasks in order. No new experiments needed here —
> these are writing, citation, and presentation tasks only.
>
> Paper draft: `paper/sigspatial2026_draft.md`
> Venue: ACM SIGSPATIAL 2026

---

## Task W1 — Add specific citations to Related Work (Critical)

**Why:** Section 2 currently contains no named prior works. This is the single biggest
reviewer risk. All five citations below can be added without running any code.

**Action:**

1. Fetch arXiv:2408.06761 (CVDisaster, Hao et al. 2025). Confirm:
   - Their Hurricane Ian F1 / accuracy number on binary damage classification
   - Whether they evaluate a conflict subset (they do not — this is the gap)

2. Add an explicit CVDisaster paragraph to Related Work (Section 2), approximately:

   > CVDisaster (Hao et al., 2025) applies cross-view fusion to Hurricane Ian paired
   > imagery, demonstrating that ground-satellite fusion is feasible for post-disaster
   > assessment. They report overall classification performance but do not separate
   > easy cases from evidence-conflict cases, and do not evaluate across observation
   > regimes. We reuse the same Hurricane Ian benchmark to enable direct methodological
   > comparison, and show that the most informative findings are invisible to aggregate
   > evaluation alone.

3. Add one sentence to Section 2 citing the xBD / xView2 benchmark (Gupta et al., 2019)
   when mentioning "single-view remote sensing benchmarks." This is the field standard
   and its absence is conspicuous.

4. Add a citation to Focal Loss (Lin et al., 2017, RetinaNet paper) in Section 5.3 when
   introducing ConflictFocalLoss. One sentence: "ConflictFocalLoss draws on the
   reweighting principle introduced by focal loss (Lin et al., 2017) but replaces
   class-frequency-based modulation with conflict-magnitude-based modulation."

5. Add a sentence in Section 5.4 (Adaptive Inference Cascade) acknowledging the
   selective prediction literature: "This design is related to selective prediction
   (Geifman & El-Yaniv, 2017), which withholds predictions when a model's confidence
   falls below a threshold. Here, routing is triggered not by low confidence but by
   high single-view disagreement."

6. Add the four new citations to `paper/references.bib`:
   - Hao et al. 2025 (arXiv:2408.06761)
   - Gupta et al. 2019 (xBD dataset, CVPR 2019 workshop)
   - Lin et al. 2017 (Focal Loss / RetinaNet, ICCV 2017)
   - Geifman & El-Yaniv 2017 (selective prediction, NeurIPS 2017)

---

## Task W2 — Add training hyperparameters for reproducibility

**Why:** Section 5.1 describes the architecture conceptually but omits the details a
reader would need to reproduce results. This is a required disclosure at SIGSPATIAL.

**Action:**

Add an "Implementation Details" subsection (Section 5.6) or a short paragraph at the
end of Section 5.1 with the following information (fill in actual values from your
training scripts or config files):

```
Encoders: ResNet18 (He et al., 2016) pretrained on ImageNet-1k.
Feature dimension: 512 per view; fusion vector dimension: 2048 (4 × 512).
Fusion head: two-layer MLP with hidden dimension 256, ReLU activation, dropout 0.3.
Optimizer: AdamW, learning rate 1e-4, weight decay 1e-4.
Training: 30 epochs, batch size 32, early stopping on validation F1 (patience 10).
Image resolution: 224 × 224, standard ImageNet normalization.
All experiments: single NVIDIA A100 40 GB; wildfire training ~12 min/run.
```

Replace placeholder values with your actual settings. If any differ between
wildfire and hurricane experiments, note the difference.

---

## Task W3 — Deepen the Discussion section (Section 8)

**Why:** At ~350 words, Section 8 is too short for the richness of the findings.
Top-tier reviewers expect mechanism analysis, not just a restatement of results.

**Action:**

Add approximately 200 words to Section 8. Suggested structure:

**Paragraph to add after the current second paragraph:**

> The regime distinction is grounded in data collection protocol. Wildfire inspection
> images are typically captured from within 1–2 meters of the target facade, with the
> camera deliberately aimed at structural damage indicators. Hurricane panoramic images
> are captured from the street with varying standoff distances and often include
> adjacent structures, parked vehicles, and vegetation clutter. This difference is not
> incidental — it reflects the operational protocols of the agencies that collected
> each dataset. The regime effect we observe may therefore be replicated in any
> deployment that mixes inspection-style ground imagery with survey-style panoramic
> imagery, regardless of disaster type.

**Paragraph to add before the current closing paragraph:**

> The adaptive cascade (Section 5.4) operationalizes conflict-aware fusion as a
> selective prediction system: the cross-view model is invoked only when single-view
> disagreement exceeds a threshold. On wildfire, this means 90.5% of samples are
> resolved by a single view alone, and cross-view fusion is reserved for the 9.5% of
> hard cases. This efficiency property — routing only genuinely ambiguous samples to
> the more expensive fusion model — has practical value in rapid post-disaster triage
> where inference latency matters. Future work could optimize the routing threshold
> jointly with the cross-view model rather than setting it post-hoc.

---

## Task W4 — Add two missing limitations

**Why:** The current Limitations section (Section 9) is honest but incomplete.
Two known scope limitations are not mentioned; a careful reviewer will raise them.

**Action:**

Add the following two items to Section 9:

1. **Binary classification scope:**

   > This paper studies binary damage triage throughout (damaged vs. undamaged).
   > Most operational damage assessment uses multi-class scales (e.g., the xBD
   > five-class taxonomy: No Damage, Minor, Major, Destroyed, Unclassified). The
   > conflict-aware framework may not extend trivially to multi-class settings, where
   > the definition of "disagreement" between single-view predictions is less
   > well-defined. Multi-class extension is left for future work.

2. **Geographic scope:**

   > Both datasets are drawn from events in the continental United States, where
   > inspection and damage assessment protocols are relatively standardized. The
   > regime distinction we observe — property-centric vs. panoramic ground view —
   > may not hold in other regions where building density, camera standoff norms,
   > or data collection procedures differ substantially.

---

## Task W5 — Sharpen the Abstract opening sentence

**Why:** The current first sentence is accurate but generic. A sharper opening signals
novelty to reviewers within the first five words.

**Current:**
> Cross-view disaster assessment combines ground-level and overhead imagery to infer
> building damage, but most prior work evaluates it only by average test accuracy.

**Replace with:**
> Cross-view disaster assessment fuses ground-level and overhead imagery to infer
> building damage. Prior evaluation has relied almost entirely on aggregate accuracy
> metrics, which obscure where fusion actually matters: the cases where the two views
> disagree.

This two-sentence version also gives you back space by being more concise overall.

---

## Task W6 — Reduce Introduction–Abstract redundancy

**Why:** The Introduction currently repeats "conflict-dependent," "regime-dependent,"
and "view dominance switching" with nearly identical phrasing to the Abstract.
This is fine for conferences, but cleaning it up improves the reading experience.

**Action:**

In the Introduction (Section 1), change the first appearance of the regime framing
(around line 46–50) from:

> "We argue that cross-view value is fundamentally conflict-dependent and
> regime-dependent."

to a version that leads with the empirical question rather than the conclusion:

> "We investigate two questions. First: is the benefit of cross-view fusion
> concentrated in cases where the two views disagree, or distributed across all
> samples? Second: does the identity of the more informative view change with the
> spatial observation regime of the ground imagery?"

Then let the paragraph that follows answer both questions, removing the phrase
"conflict-dependent and regime-dependent" (it has already appeared in the Abstract
and will appear in the Conclusion; three occurrences is enough).

---

## Task W7 — Add data-collection context to Section 3.2 and 3.3

**Why:** Reviewers unfamiliar with these datasets will not understand why the
wildfire and hurricane ground views differ. One sentence of context per dataset
eliminates potential confusion.

**Action:**

In Section 3.2 (Eaton/Altadena Wildfire Dataset), add after the first paragraph:

> Ground images were collected by trained property inspectors who photographed each
> structure from the curb or driveway, typically within 1–2 meters of the facade.
> This protocol produces strong spatial alignment between the ground image and the
> overhead patch.

In Section 3.3 (Hurricane Ian Benchmark), add after the first paragraph:

> Ground images in this dataset are panoramic captures taken from the street, often
> with significant standoff from any individual structure. The field of view regularly
> includes neighboring buildings, parked vehicles, and roadway debris, making
> per-structure target alignment substantially weaker than in the wildfire setting.

---

## Task W8 — Fix the Conflict-F1 permutation-test sentence

**Why:** The current draft says the conflict subset "contains 435 wildfire samples and
128 hurricane sensitivity samples, both with permutation-test significance below 10⁻⁴."
This sentence conflates subset size with a significance test, which is confusing.
The permutation test shows that Δ_view > 0 is not a chance result — clarify this.

**Location:** Section 4, just after the equations.

**Replace:**
> "Under this threshold, the conflict subset contains 435 wildfire samples and 128
> hurricane sensitivity samples, both with permutation-test *p* < 10⁻⁴."

**With:**
> "Under this threshold, the conflict subset contains 435 wildfire samples and 128
> hurricane sensitivity samples. Permutation tests confirm that the observed Δ_view
> gain is not a chance result in either setting (*p* < 10⁻⁴)."

---

## Completion Checklist

- [ ] W1: CVDisaster, xBD, Focal Loss, selective prediction citations added to `.bib` and paper
- [ ] W2: Implementation details paragraph added (real hyperparameter values)
- [ ] W3: ~200 words added to Discussion
- [ ] W4: Binary classification and geographic scope limitations added
- [ ] W5: Abstract opening two sentences rewritten
- [ ] W6: Introduction redundancy reduced; empirical-question framing used
- [ ] W7: Data collection context added to Sections 3.2 and 3.3
- [ ] W8: Permutation test sentence clarified in Section 4

These 8 tasks require no new experiments. All changes are in
`paper/sigspatial2026_draft.md` and `paper/references.bib`.
