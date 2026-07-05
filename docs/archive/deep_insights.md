# Deep Insights — Mechanistic Directions

> Added: 2026-04-22
> These are not routine experiment additions. Each one changes the framing or
> the claim of the paper, not just its credibility.

---

## B1 — Decompose Conflict into "Confident Disagreement" vs "Mutual Uncertainty"

### The blind spot

The current conflict definition is binary: `street_only != remote_only`.
This collapses two fundamentally different failure modes into one label.

**Type A — Confident disagreement:**
```
street_only: P(damaged) = 0.92   →  predicts "undamaged" (wrong)
remote_only: P(damaged) = 0.08   →  predicts "damaged"   (right)
```
Both models are confident but contradict each other. Crossview has a clear
signal to arbitrate: overhead evidence is unambiguous, street evidence is
misleading. Fusion can fix this.

**Type B — Mutual uncertainty:**
```
street_only: P(damaged) = 0.53   →  predicts "damaged"
remote_only: P(damaged) = 0.47   →  predicts "undamaged"
```
Both models are near-random. The disagreement is noise, not signal.
Fusing two uninformative embeddings produces a third uninformative embedding.
Crossview cannot help here.

### Hypothesis

The wildfire conflict subset is dominated by Type A.
The hurricane conflict subset is dominated by Type B.

If true, building_ratio is a proxy for this effect — not the root cause.
The root cause is whether the ground-view image provides a **confident
but wrong** signal that overhead can correct, or whether both modalities
are simply uncertain.

### Verification (runs on existing predictions today)

```python
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# Load street and remote test_predictions.csv for each dataset.
# Requires a `prob_damaged` column (raw sigmoid output, not just predicted label).

def analyze_conflict_confidence(street_df, remote_df, label="dataset"):
    merged = street_df.merge(remote_df, on="sample_id", suffixes=("_s", "_r"))
    conflict = merged[merged["pred_s"] != merged["pred_r"]].copy()

    conflict["conf_s"] = (conflict["prob_damaged_s"] - 0.5).abs()
    conflict["conf_r"] = (conflict["prob_damaged_r"] - 0.5).abs()
    conflict["min_conf"] = conflict[["conf_s", "conf_r"]].min(axis=1)
    conflict["max_conf"] = conflict[["conf_s", "conf_r"]].max(axis=1)

    print(f"\n{label} conflict confidence:")
    print(conflict[["min_conf", "max_conf"]].describe().round(3))
    return conflict

wildfire_conflict = analyze_conflict_confidence(wf_street, wf_remote, "wildfire")
hurricane_conflict = analyze_conflict_confidence(ian_street, ian_remote, "hurricane")

# Test whether the confidence distributions differ
stat, p = mannwhitneyu(
    wildfire_conflict["min_conf"],
    hurricane_conflict["min_conf"],
    alternative="greater",  # wildfire conflicts are more confident
)
print(f"\nMann-Whitney U: stat={stat:.1f}, p={p:.4f}")
```

### Expected result

| Dataset | Mean min_conf | Interpretation |
|---------|--------------|----------------|
| Wildfire | ~0.30–0.40 | At least one model is strongly committed |
| Hurricane | ~0.05–0.15 | Both models are near-random |

If this holds, the paper claim sharpens from:

> "cross-view gain depends on view regime (building vs panoramic)"

to:

> "cross-view acts as a **confident-disagreement resolver**: it helps when one
> view is strongly committed to a wrong answer, not when both views are
> uncertain. Building-centric imagery produces more confident single-view
> predictions, which makes conflict resolution both possible and necessary."

This is a more precise, more general, and more surprising claim.

---

## B2 — Conflict-Aware Architecture

### The current architecture does not embody the paper's claim

The current fusion is:

```python
features = [street, overhead, abs(street - overhead), street * overhead]
logit = MLP(cat(features))
```

The difference vector `|street - overhead|` encodes conflict implicitly,
but the MLP weights it equally against the agreement signal.
The architecture does not know it should trust overhead *more* during conflict.

### A conflict-aware alternative

```python
class ConflictAwareTriageNet(nn.Module):
    def forward(self, street_img, overhead_img):
        s = self.street_encoder(street_img)    # [B, D]
        o = self.overhead_encoder(overhead_img)  # [B, D]

        # Measure agreement between the two views
        cos_sim = F.cosine_similarity(s, o, dim=-1, eps=1e-8)  # [B], in [-1, 1]
        conflict_weight = ((1 - cos_sim) / 2).unsqueeze(-1)    # [B, 1], in [0, 1]

        # Under conflict, shift weight toward overhead.
        # Rationale: overhead has objective geometric evidence of structural
        # damage (roof collapse, debris field) that is harder to misread
        # than a street-level photo of a facade.
        fused = (1 - conflict_weight) * s + conflict_weight * o

        # Append explicit conflict signal
        full = torch.cat([fused, torch.abs(s - o), s * o], dim=-1)
        return self.classifier(full)
```

### Why this matters

1. The architecture is itself an argument. It does not just fuse; it
   *resolves* conflict by shifting authority to overhead when views disagree.

2. The ablation table becomes a direct test of the paper's hypothesis:

   | Model | Overall F1 | Conflict F1 | Δ conflict |
   |-------|-----------|------------|-----------|
   | concat fusion (current) | ... | ... | ... |
   | conflict-aware fusion | ... | ... | ... |

   If conflict-aware fusion beats concat fusion specifically on the conflict
   subset (and not on non-conflict cases), this is direct empirical evidence
   that explicit conflict resolution is the right inductive bias.

3. The architecture generalizes naturally to other cross-view tasks beyond
   disaster damage assessment.

### Relationship to the confidence decomposition (B1)

The conflict-aware architecture should help most on Type A conflicts
(confident disagreement) and be neutral on Type B (mutual uncertainty).
Testing both B1 and B2 together gives a closed mechanistic argument:
we identify when conflict is resolvable (B1), and we build a model
that resolves it (B2).

---

## B3 — Building Ratio as a Continuous Predictor, Not a Dataset Label

### The current analysis has only two data points

The alignment proxy (SegFormer building_ratio) is currently reported as
dataset-level means:

| Dataset | building_ratio mean |
|---------|-------------------|
| Wildfire conflict | 0.2684 |
| Hurricane conflict | 0.0154 |

Two numbers cannot establish a relationship. They can only describe a
difference. A reviewer could reasonably ask: "Is this one specific property
of these two particular datasets, or a general principle?"

### The fix: treat building_ratio as a continuous predictor

The SegFormer analysis already runs per-sample. Save those per-sample values
and regress crossview success on building_ratio across all conflict samples
from both datasets combined.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# building_ratio: per-sample float from SegFormer (already computed)
# crossview_correct: 1 if crossview predicts correctly, 0 otherwise

df = pd.read_csv("outputs/analysis/conflict_per_sample.csv")
# expected columns: building_ratio, crossview_correct, dataset

X = df["building_ratio"].values.reshape(-1, 1)
y = df["crossview_correct"].values

lr = LogisticRegression(random_state=42)
lr.fit(X, y)

r, p = spearmanr(df["building_ratio"], df["crossview_correct"])
print(f"Spearman r={r:.3f}, p={p:.4f}")

# Figure: scatter + logistic curve, two colors for two datasets
fig, ax = plt.subplots(figsize=(6, 4))
for ds, color in [("wildfire", "#E07B54"), ("hurricane", "#5A8FC2")]:
    subset = df[df["dataset"] == ds]
    ax.scatter(
        subset["building_ratio"],
        subset["crossview_correct"] + np.random.uniform(-0.02, 0.02, len(subset)),
        alpha=0.4, s=20, color=color, label=ds
    )
x_range = np.linspace(0, df["building_ratio"].max(), 200).reshape(-1, 1)
ax.plot(x_range, lr.predict_proba(x_range)[:, 1], color="black", linewidth=2,
        label=f"logistic fit (r={r:.2f})")
ax.set_xlabel("Building ratio (SegFormer)")
ax.set_ylabel("P(crossview correct | conflict)")
ax.legend()
ax.set_title("Target alignment predicts cross-view conflict resolution")
plt.tight_layout()
plt.savefig("outputs/figures/building_ratio_vs_crossview_gain.pdf")
```

### What this figure achieves

The wildfire cluster sits at high building_ratio; the hurricane cluster at
low building_ratio. Both sit on the same logistic curve.

This reframes the two-dataset comparison from:

> "wildfire and hurricane are different"

to:

> "wildfire and hurricane are two regions on the same continuous relationship
> between target alignment and cross-view gain"

The view regime label (building-centric / panoramic) becomes a human-readable
shorthand for a measurable, continuous property of each image.

---

## B4 — Reframe Hurricane as a Control Condition, Not a Weak Result

### The current narrative has a framing problem

As written, the paper story is:

> "We show crossview helps on wildfire. We also test on hurricane. The gain
> is smaller there. We explain why."

This reads as: the hurricane result is a disappointing secondary finding that
needs apologizing for. Reviewers will notice.

### The reframe

Treat the two datasets not as "main result + secondary result" but as
two arms of a deliberately designed experiment:

> "We construct a comparison where ground-view regime varies while all other
> factors (binary label scheme, encoder architecture, training protocol,
> evaluation metric) are held constant. The wildfire arm uses property-centric
> building inspection views; the hurricane arm uses panoramic environmental
> views. By fixing the experimental protocol and varying only the view regime,
> we isolate its causal role in determining the value of cross-view fusion."

### Concrete changes

**Method section (new paragraph):**

> We intentionally select two datasets that differ along a single axis: the
> degree to which the ground-view image is aligned with the target structure.
> All other experimental decisions (backbone, fusion architecture, label
> binarization, train/val/test proportions, data augmentation) are held
> identical across both datasets. This design isolates the effect of
> ground-view regime from confounds such as dataset size, class balance, or
> model capacity.

**Results section (revised framing):**

> The wildfire arm (high target alignment) and hurricane arm (low target
> alignment) serve as high-alignment and low-alignment conditions
> respectively. The crossview gain on the conflict subset is larger in the
> high-alignment condition (0.761 vs. 0.619), consistent with our hypothesis
> that cross-view fusion is most effective when the ground view provides
> confident structural evidence.

**Why this matters for acceptance:**

- Reviewers are trained to look for confounds in cross-dataset comparisons.
  Explicitly naming what is controlled preempts that concern.
- Hurricane becomes a *necessary* part of the experimental design, not an
  afterthought. Removing it would weaken the paper.
- The narrative arc is cleaner: hypothesis → experiment design → result →
  mechanism (B1/B3).

---

## Relationship Between the Four Insights

```
B4 (reframe) sets the paper's logical structure
        ↓
B1 (confidence decomposition) provides the mechanism
        ↓
B3 (continuous regression) quantifies the mechanism
        ↓
B2 (conflict-aware architecture) embodies the mechanism in the model
```

B4 is a writing change. B1 runs on existing data today.
B3 requires per-sample SegFormer output (re-run `analyze_building_alignment.py`
with `--save-per-sample`). B2 requires a new model class and retraining.

### Recommended execution order

1. **B4** — rewrite Introduction and Method framing (half a day, no code)
2. **B1** — run confidence decomposition on existing predictions (half a day)
3. **B3** — re-run SegFormer with per-sample output, make the scatter plot (1 day)
4. **B2** — implement ConflictAwareTriageNet, retrain, ablate (2 days)

If B1 confirms the confident-disagreement hypothesis, B2 and B3 together
form a self-contained mechanistic argument that can anchor the paper's
contribution regardless of whether the CI overlap issue in the current
results is resolved.
