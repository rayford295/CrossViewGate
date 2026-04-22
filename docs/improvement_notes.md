# CrossViewConflict — Improvement Notes

> Added: 2026-04-22 (Claude Opus 4.7 review)

These notes summarize the highest-value improvements identified after a full
code and results review. They are organized by priority.

---

## Current Results (Baseline)

| Dataset | Ground-view regime | street_only F1 | remote_only F1 | crossview F1 |
|---------|-------------------|---------------|---------------|-------------|
| Eaton wildfire | building / property-centric | 0.9604 | 0.9653 | 0.9713 |
| IAN hurricane | 360 / panoramic | 0.8912 | 0.9082 | 0.9208 |

**Conflict subset:**

| Dataset | street_only | remote_only | crossview |
|---------|------------|------------|----------|
| Eaton wildfire | 0.4179 | 0.5821 | 0.7612 |
| IAN hurricane | 0.4286 | 0.5714 | 0.6190 |

The pattern is clear. The paper story holds. The improvements below
strengthen its credibility and depth.

---

## Priority 1 — Fix Conflict Rate Reporting (Required)

**Problem:** `docs/results.md` shows the hurricane conflict rate as `0.1050`
but notes that the wildfire conflict rate is "not yet normalized in this repo
summary." Two numbers from different calculation methods are not comparable.

**Fix:** Report conflict rate for both datasets the same way:

```
conflict_rate = n_conflict / n_test_total
```

Update `docs/results.md` and the README table with both normalized values.
This is a one-line fix but a reviewer-facing issue — inconsistent reporting
undermines the comparison.

---

## Priority 2 — Bootstrap Confidence Intervals on Conflict Subset (Required)

**Problem:** The conflict subset sizes are small (wildfire n≈57, hurricane
n≈67). The difference 0.761 vs 0.619 is the paper's central quantitative
claim, but without confidence intervals it cannot be asserted as statistically
meaningful.

**Fix:** Add bootstrap confidence intervals to `scripts/build_conflict_subset.py`
output:

```python
from scipy.stats import bootstrap
import numpy as np

def bootstrap_accuracy(y_true, y_pred, n_resamples=2000, confidence=0.95):
    correct = (np.array(y_true) == np.array(y_pred)).astype(float)
    result = bootstrap(
        (correct,),
        statistic=np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence,
        random_state=42,
    )
    return {
        "mean": float(np.mean(correct)),
        "ci_low": float(result.confidence_interval.low),
        "ci_high": float(result.confidence_interval.high),
    }
```

Report format for the paper table:

```
Eaton wildfire  crossview conflict accuracy: 0.761 [0.63, 0.88]
IAN hurricane   crossview conflict accuracy: 0.619 [0.49, 0.74]
```

If the confidence intervals overlap, the claim needs to be softened.
If they do not, the claim is strengthened significantly.

---

## Priority 3 — Hurricane Label Definition Needs Justification

**Problem:** The IAN hurricane dataset drops `ModerateDamage` and uses only
the two endpoint classes (Minor vs Severe). This artificially widens the
inter-class gap and may inflate classifier performance. The paper must explain
this choice clearly.

**Options:**

1. **Keep the current choice but justify it explicitly** — e.g., "We use
   endpoint-to-endpoint comparison to maximize label clarity and isolate the
   effect of view regime; moderate damage is excluded because it is
   semantically ambiguous for both single-view models."

2. **Add a sensitivity experiment** — train and evaluate on the full
   three-class setting (Minor / Moderate / Severe), then map to binary
   post-hoc with `Moderate → 1`. Report whether conclusions hold.

Option 2 is stronger for reviewers who will notice the dropped class.

---

## Priority 4 — Quantify WHY Hurricane Gain Is Smaller (Key Analysis)

**Problem:** The paper currently explains the smaller hurricane gain with a
qualitative hypothesis: "panoramic views have weaker structural alignment with
the target building." This is plausible but unverified.

**The analysis that turns "we observe" into "we explain":**

Measure the approximate fraction of each ground-view image occupied by the
target building. This can be estimated cheaply:

```python
# Option A: Use a pretrained segmentation model (SegFormer, SAM) to
# segment "building" pixels in each image and compute coverage ratio.

# Option B (cheaper): Use image crops — if the inspection photo is
# property-centric, the building should be near-center and large.
# Compute: ratio of central 50% crop vs full image SSIM to overhead.

# Option C (quickest): Manual annotation of ~20 conflict examples from
# each dataset — label each ground image as "building-dominant" or
# "environment-dominant." Compute the fraction per dataset.
```

If wildfire images show higher building coverage than hurricane images,
this directly validates the alignment hypothesis and gives the paper a
mechanism-level figure.

**Expected figure:** scatter plot of building coverage % vs conflict-subset
crossview gain, one point per dataset.

---

## Priority 5 — Augmentation Must Be Consistent Across Experiments

**Problem:** `CrossViewTriageDataset` in `crossview_conflict/data/datasets.py`
builds transforms with `augment=False` and no option to enable it. Meanwhile,
`CrossViewRetrievalDataset` has `street_augment` and `overhead_augment`
parameters. If training scripts use different augmentation policies for the
two datasets, the comparison is confounded.

**Fix:** Add `street_augment` and `overhead_augment` parameters to
`CrossViewTriageDataset.__init__` (same pattern as `CrossViewRetrievalDataset`),
and ensure `scripts/train_triage.py` applies the same augmentation policy to
both datasets.

```python
class CrossViewTriageDataset(_BaseManifestDataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        street_size: int = 224,
        overhead_size: int = 224,
        normalize: bool = True,
        include_generated: bool = False,
        street_augment: bool = False,   # add this
        overhead_augment: bool = False, # add this
    ) -> None:
        ...
        self.street_transform = build_transform(
            street_size, normalize=normalize, augment=street_augment
        )
        self.overhead_transform = build_transform(
            overhead_size, normalize=normalize, augment=overhead_augment
        )
```

---

## Priority 6 — Integrate Generative Uncertainty Signal

**Background:** In the companion `FireBridge` repo, generative uncertainty
from a ControlNet conditioned on overhead patches has been confirmed as a
damage signal:

- Wildfire balanced subset AUC: **0.636**
- Wildfire conflict-only AUC: **0.799**

The model is `use_generated=True` path in `CrossViewTriageNet`, and the
infrastructure already exists.

**The cross-disaster experiment:** If generative uncertainty correlates more
strongly with damage in wildfire (property-centric) than in hurricane
(panoramic), this independently validates the view-alignment hypothesis using
a completely different signal.

**Experiment design:**

```
For each dataset (wildfire + hurricane):
  1. Sample T=8 bridge hypotheses per test example using the FireBridge
     ControlNet checkpoint
  2. Compute sigma_match = std(cosine(encode(ground_view), encode(hypothesis_t)))
  3. Compute AUC of sigma_match as a binary damage classifier
  4. Compare AUC across datasets

Hypothesis:
  wildfire AUC > hurricane AUC
  (because property-centric views constrain generation better)
```

If confirmed, this gives the paper a third column in the conflict analysis:

| Dataset | crossview gain on conflict | uncertainty AUC |
|---------|--------------------------|-----------------|
| Wildfire | **+34pp** | **0.80** |
| Hurricane | +19pp | ? |

This links CrossViewConflict directly to FireBridge's core finding.

---

## Summary Table

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 🔴 P1 | Normalize conflict rate for both datasets | 30 min | Credibility |
| 🔴 P2 | Bootstrap CIs on conflict subset accuracy | 2 hrs | Required for claims |
| 🟡 P3 | Justify / sensitivity-test hurricane label drop | 1 day | Reviewer defense |
| 🟡 P4 | Quantify building coverage ratio analysis | 1–2 days | "We explain" vs "we observe" |
| 🟡 P5 | Consistent augmentation across datasets | 2 hrs | Fair comparison |
| 🟢 P6 | Integrate generative uncertainty cross-disaster | 2–3 days | Novel cross-repo contribution |

---

## One-Sentence Summary

The paper story is solid and the results are real. The missing pieces are:
(1) normalized conflict rates, (2) confidence intervals, (3) a visual analysis
that explains mechanistically why the hurricane gain is smaller than the
wildfire gain.
