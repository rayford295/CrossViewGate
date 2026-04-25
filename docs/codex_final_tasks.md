# Codex Final Tasks — Paper Completion

> Written for Codex. Read this first. Execute tasks in order.
> Priority: unblock submission, not add new experiments.
> Last updated: 2026-04-24

---

## Context: What This Paper Is

**Title:** When Cross-View Helps Most: Conflict-Aware Disaster Triage
Across Spatial Observation Regimes

**Venue target:** ACM SIGSPATIAL GeoAI Workshop 2026 (fast track),
then IJGIS or IEEE TGRS for the full version.

**Core paper draft:** `paper/sigspatial2026_draft.md`

**Dataset structure:**
- Eaton/Altadena wildfire — novel dataset, contributed by this paper,
  building-centric inspector photos + satellite. No prior work uses it.
- IAN hurricane — same dataset used by CVDisaster (Hao et al., ISPRS 2025,
  arXiv:2408.06761). This shared dataset enables direct methodological
  comparison without rerunning their code.

**Key insight about the comparison:**
CVDisaster reports only overall F1 on IAN hurricane. This paper applies
the Conflict-Aware Evaluation Protocol (CAE) to the same data and
discovers View Dominance Switching — a finding invisible to aggregate
evaluation. The comparison is therefore about analytical depth, not
raw accuracy.

---

## Task 1 — Look up CVDisaster numbers and update the paper draft

**Why:** CVDisaster (Hao et al., ISPRS 2025) uses the same IAN hurricane
dataset. Adding their published numbers to Table 1 resolves the
"no external baseline" reviewer concern without running any new code.

**Action:**
1. Fetch arXiv:2408.06761 and find their Hurricane Ian classification
   results (overall F1 or accuracy on the same binary damage task).
2. Note any methodological differences:
   - do they use Google Street View or the same panoramic IAN imagery?
   - what label collapse do they use (Minor vs Severe? binary collapse?)
   - what backbone?
3. Add a row to the main results table in `paper/sigspatial2026_draft.md`
   under Section 7 or in a new comparison table:

   | Method | Dataset | Overall-F1 | Conflict-F1 | Notes |
   |--------|---------|-----------|------------|-------|
   | CVDisaster (Hao et al. 2025) | IAN hurricane | [their F1] | N/A — not reported | arXiv:2408.06761 |
   | street_only (ours) | IAN hurricane | 0.8912 | ... | this work |
   | crossview (ours) | IAN hurricane | 0.9208 | ... | this work |

4. Add one paragraph to Related Work (Section 2) that explicitly names
   CVDisaster and says the IAN hurricane dataset is shared:

   ```
   CVDisaster (Hao et al., 2025) applies cross-view fusion to Hurricane Ian
   paired imagery and reports [F1=X] on binary damage classification.
   We reuse the same Hurricane Ian benchmark to enable direct methodological
   comparison. Rather than seeking higher overall accuracy, we apply our
   Conflict-Aware Evaluation Protocol to the same data and reveal that
   cross-view gain is concentrated in the conflict subset (p < 1e-4),
   and that the dominant evidence modality is overhead — a finding invisible
   to aggregate evaluation alone. The Eaton/Altadena wildfire dataset is
   contributed by this paper and has no prior cross-view counterpart.
   ```

---

## Task 2 — Add Conflict-F1 to the ConflictFocalLoss table

**Why:** The paper argues that CAE (Conflict-F1) is the right evaluation
axis, but the ConflictFocalLoss results only show Overall-F1. This is
an internal inconsistency reviewers will catch.

**Action:**
Evaluate each ConflictFocalLoss checkpoint on the τ=0.1 conflict subset
of the wildfire sensitive test set. Report both Overall-F1 and Conflict-F1.

The ConflictFocalLoss checkpoints are presumably saved in:
`outputs/eaton_wildfire/triage_conflictfocal_gamma*/`

Use `scripts/eval_triage.py` to get predictions, then run the τ=0.1
conflict subset analysis on each.

Expected table after this task:

| Setting | Overall-F1 | Conflict-F1 (τ=0.1) |
|---------|-----------|---------------------|
| Baseline crossview | 0.9473 | ? |
| gamma=0.1 | 0.9514 | ? |
| gamma=0.25 | 0.9507 | ? |
| gamma=0.5 | 0.9520 | ? |
| gamma=1.0 | 0.9470 | ? |

Add this table to `docs/results.md` under "ConflictFocalLoss sweep"
and to `paper/sigspatial2026_draft.md` Section 7.10.

**What to look for:** Does gamma=0.5 (best Overall-F1) also give the
best Conflict-F1? If not, the optimal gamma may differ depending on
which metric is prioritized. Report both optimal points.

---

## Task 3 — Add voting ensemble baseline

**Why:** The simplest possible cross-view fusion baseline (majority vote
of street_only + remote_only) is expected by all reviewers. No retraining
needed — just combine existing prediction CSVs.

**Action:**
For wildfire and hurricane, load the street_only and remote_only
test_predictions.csv files and compute:
- vote = 1 if (pred_s + pred_r) >= 1 (either or both predicts damage)
  or use average probability: pred = (prob_s + prob_r) / 2 > 0.5
- compute F1 of this ensemble

Also compute Conflict-F1 for the ensemble using the τ=0.1 definition.

```python
import pandas as pd
from sklearn.metrics import f1_score

street = pd.read_csv("outputs/.../street_only.../test_predictions.csv")
remote = pd.read_csv("outputs/.../remote_only.../test_predictions.csv")
split  = pd.read_csv("data/splits/.../test.csv")

df = split.merge(
    street[["sample_id","prob_damaged"]].rename(columns={"prob_damaged":"p_s"}),
    on="sample_id"
).merge(
    remote[["sample_id","prob_damaged"]].rename(columns={"prob_damaged":"p_r"}),
    on="sample_id"
)

df["ensemble_pred"] = ((df["p_s"] + df["p_r"]) / 2 > 0.5).astype(int)
print("Voting ensemble F1:", f1_score(df["label"], df["ensemble_pred"]))
```

Add one row to the main results table for each dataset:
`Voting ensemble | wildfire | [F1] | [Conflict-F1] | — |`

---

## Task 4 — Compute S→O analysis on the main endpoint splits

**Why:** The current S→O analysis (59% wildfire, 44% hurricane) was
computed on the sensitivity splits, not the main endpoint splits.
Reviewers may ask whether the finding holds on the primary experiments.

**Action:**
Run `scripts/analyze_gain_decomposition.py` on:
- wildfire endpoint split (the one that gives F1=0.9713)
- hurricane grouped clean split (the one that gives F1=0.9008)

Report S→O fraction, O→S fraction, and crossview accuracy on each type.

Add results to `docs/results.md` under "View Dominance Switching"
and update `paper/sigspatial2026_draft.md` Section 7.3 with these numbers.

---

## Task 5 — Tile robustness check for the spatial map

**Why:** The tile-level Spearman r=0.512, p=0.0063 is based on n=27 tiles.
Reviewers may worry about leverage points (a few extreme tiles driving the r).

**Action:**
Load `outputs/analysis/tile_conflict_rate/tile_stats_wildfire.csv`.
Compute Spearman r with the top-2 and bottom-2 tiles by conflict_rate removed.
Report:
- Full set: r=0.512, p=0.006, n=27
- Trimmed: r=?, p=?, n=23

If trimmed r is still > 0.35 and p < 0.05, the result is robust.
Add this robustness note to `docs/results.md` and to
`paper/sigspatial2026_draft.md` Section 7.12.

---

## Task 6 — Consolidate the hurricane story in the paper draft

**Why:** Hurricane results appear in four separate sections of the draft
(7.1, 7.3, 7.8, 7.9), which makes the narrative hard to follow.

**Action:**
In `paper/sigspatial2026_draft.md`, rewrite the Discussion section
(Section 8) to explicitly thread together the hurricane findings:

Key narrative to convey:
1. On the clean grouped hurricane split, crossview ≈ street overall
   (crossview=0.9008, street=0.9016) — this is honest and expected.
2. But on the conflict subset (τ=0.1), crossview remains significantly
   better than both single-view models (p < 1e-4).
3. The dominant view in hurricane conflicts is overhead (not street),
   which is the opposite of wildfire — this is View Dominance Switching.
4. Therefore: the hurricane result is not a failure of the method.
   It shows exactly what the theory predicts: in a panoramic regime
   where ground-view evidence is weaker, the conflict resolution benefit
   is smaller on average but still significant at the conflict level.

This thread should appear in ≤ 3 sentences in the Discussion.
It should NOT require the reader to cross-reference four subsections.

---

## Task 7 — Rewrite the abstract to lead with View Dominance Switching

**Why:** The current abstract leads with ConflictFocalLoss (+0.0047 F1),
which is the weakest contribution. The strongest finding is View Dominance
Switching. The abstract should reflect the paper's actual hierarchy.

**Current abstract opening (weak):**
> ... We further show that tile-level cross-view conflict density forms an
> unsupervised spatial damage signal in the wildfire dataset, and that a
> conflict-aware training objective improves wildfire cross-view F1 from
> 0.9473 to 0.9520.

**Rewrite the abstract in `paper/sigspatial2026_draft.md` to:**
1. Open with the conflict-centered framing (1 sentence)
2. Name View Dominance Switching as the central finding (2 sentences,
   include the rank-order flip: wildfire street>remote, hurricane remote>street)
3. Name the tile-level spatial map contribution (1 sentence, include r=0.512)
4. Name ConflictFocalLoss as a method extension (1 sentence)
5. Close with the broader implication (1 sentence)

---

## Task 8 — Build a single unified Table 1

**Why:** Results are currently in 12 subsections. A single comprehensive
table lets a reviewer understand all main results in 30 seconds.

**Action:**
Add a Table 1 to `paper/sigspatial2026_draft.md` in Section 7 (before
the subsections), structured as:

| Method | Dataset | Split | Overall-F1 | Conflict-F1 (τ=0.1) | Δ_view | Seeds |
|--------|---------|-------|-----------|---------------------|--------|-------|
| CVDisaster (Hao et al. 2025) | Hurricane | original | [their #] | N/A | — | — |
| Voting ensemble | Wildfire | endpoint | ? | ? | — | 1 |
| street_only | Wildfire | endpoint | 0.9604 | ? | — | 3 |
| remote_only | Wildfire | endpoint | 0.9653 | ? | — | 3 |
| crossview | Wildfire | endpoint | 0.9713 | ? | +? | 3 |
| street_only | Hurricane | grouped clean | 0.9016 | ? | — | 1 |
| remote_only | Hurricane | grouped clean | 0.8385 | ? | — | 1 |
| crossview | Hurricane | grouped clean | 0.9008 | ? | -0.015 | 1 |
| crossview + ConflictFocalLoss (γ=0.5) | Wildfire | sensitive | 0.9520 | ? | — | 1 |

Δ_view = Conflict-F1(crossview) − Conflict-F1(best single-view).
Fill in all "?" cells from existing results or from Task 2 outputs.

---

## Do NOT add these (scope control)

- No new datasets
- No new backbone experiments
- No EDL / evidential fusion (different paper)
- No localization / retrieval experiments
- No new disaster types

The paper is complete scientifically. These tasks are about
presentation clarity and the two missing numbers (CVDisaster + voting
ensemble). Both can be done without any GPU.
