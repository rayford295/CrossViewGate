# Codex Priority — Paper Contribution Targets

> This file is written for Codex. Read this first before doing anything else.
> Last updated: 2026-04-23

---

## The Five Contributions to Complete

These are the paper's five contributions in priority order.
Each one has a clear status and a concrete next action.

---

### Contribution 1 — Dataset (DONE)

Two real-disaster paired datasets:
- Eaton/Altadena wildfire: inspector ground photos + satellite overhead patches
- IAN hurricane: panoramic ground photos + satellite overhead patches

Status: complete. No action needed.

---

### Contribution 2 — Conflict-Aware Evaluation Protocol (DONE)

CAE = three metrics evaluated on every model:
1. Overall-F1: standard metric on full test set
2. Conflict-F1: metric restricted to the τ=0.1 conflict subset
3. Δ_view: Conflict-F1(crossview) − Conflict-F1(best single-view)

Status: the protocol is defined and all numbers exist.
The paper needs a dedicated Table that presents all three metrics
side by side for every model configuration.

**Codex action:** Add a summary table to `docs/results.md` that
shows Overall-F1, Conflict-F1 (τ=0.1), and Δ_view for every
completed experiment (wildfire and hurricane, all three modes).

---

### Contribution 3 — View Dominance Switching (NEEDS ONE SCRIPT RUN)

Finding: in conflict cases (τ=0.1), the rank order of single-view
models flips across datasets:

```
Wildfire:  crossview > STREET(0.736) > remote(0.653)   street dominates
Hurricane: crossview > REMOTE(0.727) > street(0.703)   overhead dominates
```

This is supported by the S→O correction rate:
- Wildfire:  59.0% of conflicts are S→O (street corrects overhead)
- Hurricane: 44.1% of conflicts are O→S (overhead corrects street)

Both permutation tests: p < 1e-4.

Status: numbers are confirmed from existing runs.
The formal per-type accuracy breakdown (crossview accuracy on S→O cases
vs O→S cases) has not yet been computed on the main endpoint splits.

**Codex action:** Run `scripts/analyze_view_dominance_switch.py` on
the wildfire and hurricane test prediction CSVs with `--tau 0.1`.
Save the output JSON and figure to `outputs/analysis/view_dominance_switch/`.
Paste the key numbers (so_frac, os_frac, cross_acc_on_so, cross_acc_on_os)
into `docs/results.md` under a new section "View Dominance Switching".

---

### Contribution 4 — ConflictFocalLoss (NEEDS FULL SWEEP)

ConflictFocalLoss up-weights training samples whose street and overhead
embeddings disagree, so the model focuses more on conflict-like cases.

Pilot result (γ=0.5, wildfire sensitive split):
- baseline crossview F1: 0.9473
- ConflictFocalLoss F1:  0.9520  (+0.0047)

This is promising but a single data point. A full gamma sweep is needed
before this can be claimed as a contribution.

**Codex action:** Retrain crossview on the wildfire sensitive split
using ConflictFocalLoss with γ ∈ {0.1, 0.25, 0.5, 1.0}.
For each γ, also run the conflict subset evaluation (τ=0.1).
Report both Overall-F1 and Conflict-F1 for each γ.
Save results and add a "ConflictFocalLoss ablation" table to `docs/results.md`.

Loss implementation: `crossview_conflict/training/conflict_focal_loss.py`
Training script: `scripts/train_triage.py` (add `--loss conflict_focal`
flag and `--focal-gamma` argument).

---

### Contribution 5 — Conflict Density Spatial Map (NEEDS EXPERIMENT)

Hypothesis: the geographic density of cross-view conflicts (measured as
mean |prob_street − prob_remote| per tile) correlates with tile-level
actual damage rate — without using any ground-truth labels.

If tile-level Spearman r > 0.4 (p < 0.05), this becomes a zero-annotation
unsupervised spatial damage localisation method — a GIS-native contribution.

Status: not yet run. Script is ready.

**Codex action:** Run `scripts/analyze_tile_conflict_rate.py` on both
wildfire and hurricane test sets. The `--tile-col` argument should point
to the column in the split CSV that identifies the geographic tile
(e.g., `remote_tile_filename` for wildfire).

```powershell
python scripts\analyze_tile_conflict_rate.py `
  --street-preds  outputs\eaton_wildfire\triage_street_only_resnet18\test_predictions.csv `
  --remote-preds  outputs\eaton_wildfire\triage_remote_only_resnet18\test_predictions.csv `
  --split-csv     data\splits\eaton_wildfire\test.csv `
  --tile-col      remote_tile_filename `
  --label-col     label `
  --output-dir    outputs\analysis\tile_conflict_rate `
  --dataset       wildfire
```

Save the Spearman r, p-value, and figure.
Add results to `docs/results.md` under "Conflict Density Spatial Map".

If r > 0.4: this contribution is confirmed — add it to the abstract.
If r < 0.2: drop this contribution from the paper.

---

## Blocking Experiments (External Baselines)

These are required before submitting to any top venue.
They do not change the findings — they provide the comparison context.

### B1 — CVDisaster comparison

CVDisaster (Hao et al., ISPRS 2025, arXiv:2408.06761) is the closest
prior work. It uses Hurricane Ian + Google Street View + satellite.

Action: look up their published Hurricane Ian F1 numbers from the paper.
Add a row to the main results table in `docs/results.md`:
  "CVDisaster (Hao et al. 2025) | hurricane | [their F1] | — | —"

If their code is available and runnable on your data, run it.
If not, cite their numbers and note the setup differences in a footnote.

### B2 — Simple voting ensemble

Majority vote of street_only and remote_only predictions.
This is the simplest possible fusion baseline.

Action: compute voting ensemble F1 from existing prediction CSVs.
No retraining needed — just load the two pred_label columns and take
majority vote (0 if street+remote both 0, 1 if both 1, tie → 0 or
use prob average). Add to main results table.

### B3 — xView2 satellite-only baseline

The xView2 challenge published a standard ResNet satellite-only baseline.
Look up their published accuracy on xBD test set and cite it as context
for your remote_only model.

---

## One-Line Summary for Codex

Run view_dominance_switch → run tile_conflict_rate → run ConflictFocalLoss
gamma sweep → look up CVDisaster numbers → compute voting ensemble.
All scripts already exist. Just run them and paste results into docs/results.md.
