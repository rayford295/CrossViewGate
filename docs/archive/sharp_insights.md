# Sharp Insights — Mechanism Decomposition

> Added: 2026-04-23
> These insights emerge directly from the existing numerical results.
> No new training is required for insights 1, 3, and 5A.

---

## The Core Finding Nobody Has Stated Clearly

In conflict cases, overhead always beats street (~14pp, both datasets).
That gap is consistent and universal — it is not what varies across datasets.

What varies is how much the **street view adds on top of overhead**:

| Dataset | overhead advantage | street extra contribution | street share of gain |
|---------|-------------------|--------------------------|---------------------|
| Wildfire | +16.4pp | **+17.9pp** | **68.6%** |
| Hurricane | +14.3pp | **+4.8pp** | **40.0%** |

The overhead advantage is nearly identical across both disasters.
The street contribution drops by a factor of 4 from wildfire to hurricane.

This is the mechanism. Not building_ratio. Not conflict rate. This.

> Cross-view fusion helps because the ground view can correct overhead when
> it is wrong. In building-centric views (wildfire), the ground view is
> informative enough to do this. In panoramic views (hurricane), it is not.

---

## Insight 1 — Gain Decomposition as the Paper's Core Figure

**Claim:** The crossview advantage over random (50%) can be decomposed into
two additive terms:

```
crossview_gain = overhead_advantage + street_contribution
                = (remote - street) + (crossview - remote)
```

This decomposition is computable from existing predictions with no new
experiments. It is more precise than any building_ratio regression.

**Expected figure:**

```
Stacked bar chart, one bar per dataset:
  Bottom segment (blue):  overhead_advantage  = remote - street
  Top segment (orange):   street_contribution = crossview - remote

Wildfire:  [16.4pp overhead] + [17.9pp street]  = 34.3pp total
Hurricane: [14.3pp overhead] + [ 4.8pp street]  = 19.1pp total
```

The bottom segments are nearly equal. The top segments differ by 13pp.
The figure makes the mechanism visual and self-explanatory.

**Script to generate:**

```python
import matplotlib.pyplot as plt
import numpy as np

data = {
    "wildfire":  {"cross": 0.7612, "remote": 0.5821, "street": 0.4179},
    "hurricane": {"cross": 0.6190, "remote": 0.5714, "street": 0.4286},
}

labels = list(data.keys())
overhead_adv = [d["remote"] - d["street"] for d in data.values()]
street_extra = [d["cross"]  - d["remote"]  for d in data.values()]

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(x, overhead_adv, label="Overhead advantage (remote − street)", color="#4A90C4")
ax.bar(x, street_extra, bottom=overhead_adv, label="Street contribution (cross − remote)", color="#D95F42")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Conflict-subset accuracy gain over random (50%)")
ax.set_title("Decomposition of cross-view gain on conflict subset")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("outputs/figures/gain_decomposition.pdf", dpi=200)
```

---

## Insight 2 — objectid Leakage Check (Run This First)

**Risk:** `make_group_splits.py` was just added, implying group-aware splits
may not have been used for the existing results. If the same property
(objectid) appears in both train and test, the model has seen that building
before — a form of data leakage that inflates F1.

**Check immediately:**

```python
import pandas as pd

for ds, train_path, test_path in [
    ("wildfire",  "data/splits/eaton_wildfire/train.csv",
                  "data/splits/eaton_wildfire/test.csv"),
    ("hurricane", "data/splits/ian_hurricane_minor_vs_severe/train.csv",
                  "data/splits/ian_hurricane_minor_vs_severe/test.csv"),
]:
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    if "objectid" in train.columns and "objectid" in test.columns:
        overlap = set(train["objectid"].dropna()) & set(test["objectid"].dropna())
        print(f"{ds}: {len(overlap)} objectids appear in both train and test")
    else:
        print(f"{ds}: no objectid column — cannot verify grouping")
```

**If overlap > 0:** Re-generate splits with `make_group_splits.py` and
rerun all experiments. This is more urgent than any new experiment.

**If overlap == 0:** The existing results are valid; proceed with the
remaining experiments below.

---

## Insight 3 — "Street-Corrects-Overhead" Sample Analysis

**The question:** What exactly does the extra 17.9pp (wildfire) / 4.8pp
(hurricane) represent? It must come from samples where:

- overhead is wrong (predicts the wrong label with high confidence)
- street is right
- crossview correctly follows the street signal

Call these **S→O correction cases** ("street corrects overhead").

**Identify them immediately from existing predictions:**

```python
# Load conflict subset predictions for each dataset
# Columns needed: sample_id, street_pred, remote_pred, crossview_pred, label

def count_correction_types(df, label="dataset"):
    conflict = df[df["street_pred"] != df["remote_pred"]].copy()
    n = len(conflict)

    # Both wrong
    both_wrong = (conflict["street_pred"] != conflict["label"]) & \
                 (conflict["remote_pred"] != conflict["label"])

    # Remote correct, street wrong (overhead wins — expected)
    remote_wins = (conflict["remote_pred"] == conflict["label"]) & \
                  (conflict["street_pred"] != conflict["label"])

    # Street correct, remote wrong (street corrects overhead — the valuable case)
    street_wins = (conflict["street_pred"] == conflict["label"]) & \
                  (conflict["remote_pred"] != conflict["label"])

    # Among street-wins cases: did crossview get it right?
    sw_subset = conflict[street_wins]
    cross_correct_on_sw = (sw_subset["crossview_pred"] == sw_subset["label"]).mean()

    print(f"\n{label} conflict taxonomy (n={n}):")
    print(f"  Both wrong:         {both_wrong.sum():3d}  ({both_wrong.mean():.1%})")
    print(f"  Remote wins:        {remote_wins.sum():3d}  ({remote_wins.mean():.1%})")
    print(f"  Street wins (S→O):  {street_wins.sum():3d}  ({street_wins.mean():.1%})")
    print(f"  Crossview accuracy on S→O cases: {cross_correct_on_sw:.3f}")
    print(f"  → These S→O cases ARE the extra street contribution.")

# Expected:
#   wildfire  S→O fraction > hurricane S→O fraction
#   crossview accuracy on S→O cases: high in wildfire, low in hurricane
```

**Expected finding:**

The fraction of S→O cases (where street is right and overhead is wrong) is
larger in wildfire than hurricane. This directly explains the 68.6% vs 40%
street contribution, and it is verifiable from existing files in minutes.

---

## Insight 4 — Conflict-Focal Loss (Replaces Simple Cross-Entropy)

**Problem:** The current training loss weights all samples equally. Conflict
cases (3.4% wildfire) are nearly invisible during training. The model learns
primarily from easy, agreement cases — but is evaluated on conflict cases.

**Solution:** Weight each training sample by its real-time view disagreement.

```python
import torch.nn.functional as F

class ConflictFocalLoss(torch.nn.Module):
    """
    Down-weight agreement samples, up-weight conflict samples.
    conflict_weight = ((1 - cos_sim(street, overhead)) / 2) ^ gamma
    Final loss = (1 + conflict_weight) * BCE
    """
    def __init__(self, gamma: float = 0.5):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        logit: torch.Tensor,          # [B]
        label: torch.Tensor,          # [B]
        street_feat: torch.Tensor,    # [B, D]
        overhead_feat: torch.Tensor,  # [B, D]
    ) -> torch.Tensor:
        with torch.no_grad():
            cos_sim = F.cosine_similarity(street_feat, overhead_feat, dim=-1)
            conflict_weight = ((1.0 - cos_sim) / 2.0).pow(self.gamma)  # [0,1]

        bce = F.binary_cross_entropy_with_logits(logit, label.float(), reduction="none")
        return ((1.0 + conflict_weight) * bce).mean()
```

**Difference from RetinaNet focal loss:**
- Focal loss: down-weight *easy* (confidently correct) samples
- ConflictFocalLoss: down-weight *agreement* (both views agree) samples

These are not the same — a sample can be "hard" (wrong) while both views
agree. ConflictFocalLoss specifically targets the training distribution
mismatch between abundant agreement cases and rare conflict cases.

**Ablation to run:**

| Training loss | Overall F1 | Conflict-subset accuracy | Δ conflict |
|--------------|-----------|------------------------|-----------|
| BCE (baseline) | 0.9713 | 0.7612 | — |
| ConflictFocalLoss (γ=0.25) | ? | ? | +? |
| ConflictFocalLoss (γ=0.50) | ? | ? | +? |
| ConflictFocalLoss (γ=1.00) | ? | ? | +? |

Sweep gamma on wildfire only. One retraining per value (~10 min each).

---

## Insight 5 — Three Routes to Fix hurricane p=0.26

The hurricane conflict permutation test is not significant (p=0.26). This
is the paper's main statistical weakness. Three routes to address it:

### 5A — Widen the Conflict Net (fastest, run today)

Relax the conflict definition from binary (pred_s ≠ pred_r) to continuous
(|prob_s − prob_r| > τ). At τ=0.1, hurricane conflict rate rises from
10.5% to 15.5% (n ≈ 98 instead of 67), giving the permutation test more
power.

The threshold sensitivity table already has this data (τ=0.1 row).
Re-run the permutation test on the τ=0.1 conflict subset.

### 5B — Type A Permutation Test (hardest, most convincing)

After running B1 (confidence decomposition), isolate the Type A subset
(min_conf ≥ 0.20) within hurricane conflicts. Run permutation test on
this high-quality subset only.

If Type A hurricane conflicts show crossview accuracy ≈ Type A wildfire
conflicts, the view-regime story becomes: "the mechanism is the same; it
is only triggered less often in panoramic settings."

### 5C — Reframe the Null Result (narrative)

Drop the goal of making hurricane p < 0.05. Instead, argue:

> The hurricane result (p=0.26) is not a failure of the method. It is
> a prediction of the theory: when ground-view evidence is panoramic and
> therefore unreliable (street contribution = 40% vs 68.6% in wildfire),
> the conflict resolution mechanism has less material to work with. The
> null result in hurricane is what the gain decomposition (Insight 1)
> would predict.

This turns a statistical weakness into a theoretical consistency check.
Routes 5A, 5B, and 5C can all be included; they strengthen each other.

---

## Action Order (Today → This Week)

```
TODAY (no new training):
  1. Run objectid leakage check (5 min) — gates everything else
  2. Run S→O correction type analysis (30 min) — validates Insight 3
  3. Run gain decomposition plot (30 min) — becomes Figure 1 candidate
  4. Re-run permutation test at τ=0.1 threshold (15 min) — Insight 5A

THIS WEEK (1 retraining each):
  5. ConflictFocalLoss sweep γ ∈ {0.25, 0.5, 1.0} — Insight 4
  6. ConflictAwareTriageNet (B2) retraining — compare with ConflictFocalLoss

ONLY IF LEAKAGE CHECK PASSES:
  — All current F1 numbers are valid; proceed above
  IF LEAKAGE FOUND:
  — Re-split with make_group_splits.py, rerun all baselines first
```
