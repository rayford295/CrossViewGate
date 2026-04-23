# Next Experiments — Supplementary Roadmap

> Added: 2026-04-22
> Complements `improvement_notes.md` (P1–P6). Focuses on experiments that
> close the remaining gaps for a venue-quality submission (AAAI / ECCV
> Workshop / IEEE TGRS level).

---

## E1 — Permutation Test on Conflict Subset (Implemented)

**Why bootstrap alone is not enough:**
Bootstrap CIs characterize sampling uncertainty around a point estimate.
They do not test H₀: "crossview gain on conflict = 0."
A permutation test directly tests that null hypothesis and produces a
p-value reviewers can cite.

**Implementation:**

```python
import numpy as np

def permutation_test_conflict_gain(y_true, crossview_pred, n_permutations=10_000, seed=42):
    """
    H0: crossview accuracy on conflict subset = chance (50%).
    Permute labels, recompute accuracy, build null distribution.
    """
    rng = np.random.default_rng(seed)
    observed = np.mean(np.array(y_true) == np.array(crossview_pred))
    null_dist = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(y_true)
        null_dist.append(np.mean(shuffled == np.array(crossview_pred)))
    p_value = np.mean(np.array(null_dist) >= observed)
    return {"observed_accuracy": observed, "p_value": p_value}
```

Status now:

- wildfire: implemented, significant against label-independence null
- hurricane: implemented, positive trend but not statistically strong yet

Original reporting target:

```
Wildfire conflict crossview accuracy: 0.761  p < 0.001
Hurricane conflict crossview accuracy: 0.619  p = 0.031
```

If hurricane p > 0.05, soften the claim accordingly.

---

## E2 — Per-Sample Spearman Correlation: building_ratio → crossview gain (Implemented, currently weak)

**Why:** `improvement_notes.md` P4 describes computing building_ratio and
shows dataset-level means. That gives two points on a scatter plot — not
enough for a correlation claim. We need per-sample data.

**Design:**

```python
from scipy.stats import spearmanr

# For every sample in the conflict subset (both datasets combined):
# x_i = building_ratio from SegFormer analysis (already computed)
# y_i = 1 if crossview correct AND (street wrong OR remote wrong), else 0

# Fit logistic regression: P(crossview_correct | building_ratio)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(building_ratios.reshape(-1,1), crossview_gains)

r, p = spearmanr(building_ratios, crossview_gains)
# Target: r > 0.3, p < 0.05
```

Current outcome:

- wildfire Spearman is weak and not significant
- hurricane Spearman is positive but still not below 0.05
- combined correlation is weak

This means the current evidence supports an alignment effect mainly at the
dataset / view-regime level rather than as a strong per-sample ranking signal.

Expected figure:
x = building_ratio, y = crossview_correct (jittered), logistic curve overlaid.
This turns the mechanism from "we observe" into "we quantify."

---

## E3 — Backbone Ablation

**Why:** All results use ResNet18. Reviewers will ask whether the pattern
holds for stronger encoders. If yes, the claim is architecture-agnostic.

**Recommended grid (train + eval on both datasets):**

| Backbone | Pretrain | Params |
|----------|----------|--------|
| ResNet18 | ImageNet-1k | 11M |
| ResNet50 | ImageNet-1k | 25M |
| CLIP ViT-B/32 | LAION-2B | 88M |

CLIP is especially valuable: it was not trained on damage imagery, so it tests
whether the view-regime effect is about *semantics* rather than low-level
texture.

**Minimum acceptable:** ResNet18 vs ResNet50. CLIP is a bonus.

Current status:

- wildfire `ResNet50`: completed, strong and consistent with the main claim
- wildfire `CLIP ViT-B/32`: completed, clearly underperforms the CNN baselines
- wildfire `DINOv2 ViT-S/14`: completed, much stronger than CLIP but still below the CNN baselines
- hurricane `ResNet50`: completed, strong but below the earlier `ResNet18` hurricane baseline
- hurricane `CLIP ViT-B/32`: completed, unstable and clearly below the CNN baselines
- hurricane `DINOv2 ViT-S/14`: now running

Updated interpretation:

- the paper's main result is no longer tied to `ResNet18`
- `ResNet50` already shows that the conflict-aware crossview benefit survives a stronger supervised backbone
- backbone choice is itself informative: off-the-shelf generic pretraining is not automatically the best fit for disaster triage

**What to report:** For each backbone, report street_only / remote_only /
crossview F1 on both datasets AND conflict-subset accuracy. If the rank order
is preserved across backbones, the claim is robust.

---

## E4 — Fusion Strategy Ablation

**Why:** "crossview" is not a single architecture — it depends on how the two
views are fused. The current implementation must be documented and ablated.

**Three strategies to compare:**

```python
# Strategy A: Late fusion (average logits)
logits = 0.5 * street_logits + 0.5 * remote_logits

# Strategy B: Concat features → MLP head
features = torch.cat([street_feat, remote_feat], dim=1)
logits = mlp_head(features)

# Strategy C: Cross-attention (street query, remote key/value)
# street_feat attends to remote_feat
attn_out = cross_attention(query=street_feat, key=remote_feat, value=remote_feat)
logits = classifier(attn_out)
```

**Expected finding:** Cross-attention should show the largest conflict-subset
gain because it explicitly conditions the street prediction on the remote
signal. If it does, that is a mechanistic result, not just an architecture choice.

---

## E5 — Multi-Seed Variance Reporting (In progress)

**Why:** Single-run results are not reproducible claims. Three seeds give
mean ± std and catch lucky/unlucky initialization.

**Implementation:** Run all three baselines (street_only, remote_only,
crossview) with seeds 42, 123, 456.

**Report format:**

```
crossview F1 (wildfire): 0.971 ± 0.003  (n=3 seeds)
conflict accuracy (wildfire): 0.761 ± 0.018
```

This adds less than 3× compute and is expected by reviewers at any
major venue.

Current status:

- wildfire `crossview`: completed for seeds `42 / 123 / 456`
- wildfire `street_only`: completed for seeds `42 / 123 / 456`
- wildfire `remote_only`: completed for seeds `42 / 123 / 456`
- hurricane multi-seed: intentionally postponed for now while backbone ablation is prioritized

---

## E6 — Zero-Shot Cross-Disaster Transfer

**Design:** Train crossview model on wildfire only → evaluate on hurricane
test set (no fine-tuning). Compare against model trained on hurricane.

**What this tests:** Whether the conflict-awareness mechanism learned from
property-centric disaster imagery transfers to panoramic disaster imagery.

**Expected result:** Performance drops, but the rank order
(crossview > remote_only > street_only) likely holds even in zero-shot.
If it does, it supports the claim that the mechanism is not dataset-specific.

**Table to add:**

| Train → Test | street_only F1 | crossview F1 | Δ |
|---|---|---|---|
| Wildfire → Hurricane (zero-shot) | ? | ? | ? |
| Hurricane → Hurricane (in-domain) | 0.891 | 0.921 | +0.030 |

---

## E7 — Qualitative Figure (Implemented)

Every paper on visual damage assessment needs a qualitative results figure.

**Design (4-panel per example, 3–4 examples):**

```
Row = one conflict example
Col 1: ground-view image (street / inspection photo)
Col 2: overhead patch
Col 3: prediction table — street_only / remote_only / crossview / GT
Col 4: SegFormer building segmentation overlay
```

Select examples deliberately:
- 2 wildfire conflict cases where crossview is correct, others wrong
- 2 hurricane conflict cases where crossview is correct, others wrong
- 1 failure case where all three are wrong (shows model limits)

This figure appears in Section 4 (Experiments) and is cited in the abstract.
Without it, the paper reads as a pure numbers paper and is harder to accept.

---

## E8 — Comparison Against Published Baselines

**Why:** Without external baselines, reviewers cannot assess whether the
crossview improvement is large or small relative to the field.

**Suggested comparisons:**

| Method | Source | Notes |
|--------|--------|-------|
| xView2 challenge winner | Gupta et al. 2019 | Single-view satellite baseline |
| Simple ensemble (voting) | ablation | 2-model majority vote |
| MAD (multimodal attention damage) | if applicable | if architecture matches |
| Your crossview (best) | this paper | — |

At minimum, compare against a simple voting ensemble and xView2 satellite-only
approach. This contextualizes whether F1=0.97 on wildfire is SOTA or expected.

---

## E9 — Sensitivity to Conflict Definition Threshold (Implemented)

**Current definition:** conflict = street_only and remote_only predict
different classes (binary disagreement).

**Softer definition:** conflict = |P(damaged|street) − P(damaged|remote)| > τ
for τ ∈ {0.1, 0.2, 0.3, 0.5}.

**Why this matters:** At τ=0 (current), conflict rate = 3.4% wildfire.
Lowering τ to 0.2 might give 2–3× more samples, making statistical tests
more powerful.

**Figure to make:** plot conflict_rate and crossview_gain vs τ for both
datasets. The gap between curves (wildfire vs hurricane) should be stable
across τ if the view-regime effect is real.

---

## Summary Priority Table

| ID | Experiment | Effort | Impact | Already in improvement_notes? |
|----|-----------|--------|--------|-------------------------------|
| E1 | Permutation test | 2 hrs | Required for H₀ | No |
| E2 | Per-sample Spearman correlation | 3 hrs | Mechanism proof | Partial (only means) |
| E3 | Backbone ablation | 2 days | Architecture robustness | No |
| E4 | Fusion strategy ablation | 1 day | Mechanistic insight | No |
| E5 | Multi-seed variance | 1 day | Reproducibility | No |
| E6 | Zero-shot transfer | 4 hrs | Generalization | No |
| E7 | Qualitative figure | 4 hrs | Required for submission | No |
| E8 | External baseline comparison | 1 day | Contextualization | No |
| E9 | Conflict threshold sensitivity | 3 hrs | Robustness | No |

---

## Minimum Viable Submission Checklist

To be acceptable at a tier-2 venue (ECCV Workshop, IEEE TGRS):

- [x] E1 — permutation test on conflict subset
- [x] E2 — per-sample building_ratio × crossview_gain correlation
- [ ] E5 — multi-seed mean ± std
- [x] E7 — qualitative figure (3–4 examples)
- [ ] `improvement_notes.md` P2 — bootstrap CIs
- [ ] `improvement_notes.md` P3 — justify hurricane label drop
- [ ] `improvement_notes.md` P4 — building coverage quantification

To be competitive at a tier-1 venue (AAAI, ECCV main):

- All of the above, plus E3, E4, E8.
