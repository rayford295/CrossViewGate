# CrossViewConflict

Conflict-aware cross-view disaster triage across two ground-view regimes:

- wildfire `building/property-centric` views
- hurricane `360/panoramic environment-centric` views

This repository is a focused paper repo built around one specific claim:

> Cross-view fusion helps most when street-side and overhead evidence conflict,
> and the size of that benefit depends on the ground-view regime.

## Why this repo exists

This repo strips the problem down to one clean research question:

1. Train `street_only`, `remote_only`, and `crossview` triage models.
2. Build the `conflict subset`, where the two single-view models disagree.
3. Compare how much `crossview` helps on:
   - wildfire building-view data
   - hurricane panoramic-view data

The resulting comparison is useful for a paper because it is:

- cross-disaster
- cross-view-regime
- mechanism-oriented rather than only accuracy-oriented

## Local datasets

The full image datasets are intentionally **not** tracked in this repository.
Keep them local and build manifests from your machine.

Expected local datasets used in our experiments:

- `Eaton/Altadena wildfire paired dataset`
- `IAN_hurricane`

Example local paths used on our machine:

- `C:/Users/yyang295/Desktop/Altadena_Images`
- `C:/Users/yyang295/Desktop/IAN_hurricane`

## Repository structure

```text
CrossViewConflict/
|- crossview_conflict/
|  |- data/
|  |- models/
|  |- training/
|  `- utils/
|- scripts/
|- docs/
|- data/
`- outputs/
```

## Included code

This repo contains everything needed for the paper-focused triage pipeline:

- manifest builders
- triage training
- triage evaluation
- conflict subset construction
- cross-dataset comparison support

Core scripts:

- `scripts/build_eaton_manifest.py`
- `scripts/build_ian_hurricane_manifests.py`
- `scripts/train_triage.py`
- `scripts/eval_triage.py`
- `scripts/build_conflict_subset.py`

## Binary task definition

### Wildfire

Use the Eaton/Altadena wildfire binary scheme:

- `0 = No Damage + Affected`
- `1 = Minor + Major + Destroyed`
- `Inaccessible` excluded

### Hurricane

The original `IAN_hurricane` dataset has three classes:

- `0_MinorDamage`
- `1_ModerateDamage`
- `2_SevereDamage`

For this paper repo, we keep only the two endpoint classes:

- `0 = MinorDamage`
- `1 = SevereDamage`

and drop `ModerateDamage` for a cleaner endpoint-to-endpoint comparison.

This choice is intentional. We use endpoint-to-endpoint comparison to maximize
label clarity and isolate the effect of view regime; `ModerateDamage` is
excluded because it is semantically ambiguous for both single-view models.

## How to run

### 1. Install

```powershell
cd "C:\Users\yyang295\Documents\New project\CrossViewConflict"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Build the IAN hurricane manifests

```powershell
python scripts\build_ian_hurricane_manifests.py ^
  --dataset-root "C:\Users\yyang295\Desktop\IAN_hurricane" ^
  --output-dir "data\splits\ian_hurricane_minor_vs_severe"
```

### 3. Train the three baselines

```powershell
python scripts\train_triage.py --train-csv "data\splits\ian_hurricane_minor_vs_severe\train.csv" --val-csv "data\splits\ian_hurricane_minor_vs_severe\val.csv" --output-dir "outputs\ian_hurricane\triage_crossview_resnet18" --mode crossview --device cuda --batch-size 32 --epochs 10
python scripts\train_triage.py --train-csv "data\splits\ian_hurricane_minor_vs_severe\train.csv" --val-csv "data\splits\ian_hurricane_minor_vs_severe\val.csv" --output-dir "outputs\ian_hurricane\triage_street_only_resnet18" --mode street_only --device cuda --batch-size 32 --epochs 10
python scripts\train_triage.py --train-csv "data\splits\ian_hurricane_minor_vs_severe\train.csv" --val-csv "data\splits\ian_hurricane_minor_vs_severe\val.csv" --output-dir "outputs\ian_hurricane\triage_remote_only_resnet18" --mode remote_only --device cuda --batch-size 32 --epochs 10
```

### 4. Evaluate on the test split

```powershell
python scripts\eval_triage.py --checkpoint "outputs\ian_hurricane\triage_crossview_resnet18\triage_best.pt" --split-csv "data\splits\ian_hurricane_minor_vs_severe\test.csv" --output-json "outputs\ian_hurricane\triage_crossview_resnet18\test_metrics.json" --predictions-csv "outputs\ian_hurricane\triage_crossview_resnet18\test_predictions.csv" --device cuda --batch-size 32
```

Repeat the same command for `street_only` and `remote_only`.

### 5. Build the conflict subset

```powershell
python scripts\build_conflict_subset.py ^
  --split-csv "data\splits\ian_hurricane_minor_vs_severe\test.csv" ^
  --street-preds-csv "outputs\ian_hurricane\triage_street_only_resnet18\test_predictions.csv" ^
  --remote-preds-csv "outputs\ian_hurricane\triage_remote_only_resnet18\test_predictions.csv" ^
  --crossview-preds-csv "outputs\ian_hurricane\triage_crossview_resnet18\test_predictions.csv" ^
  --output-csv "outputs\ian_hurricane\analysis\test_conflicts.csv" ^
  --summary-json "outputs\ian_hurricane\analysis\test_conflict_summary.json"
```

## Current findings

### Eaton wildfire building-view benchmark

Previously established in our wildfire experiments:

- `crossview` test `F1 = 0.9713`
- `street_only` test `F1 = 0.9604`
- `remote_only` test `F1 = 0.9653`

Conflict subset:

- `street_only = 0.4179`
- `remote_only = 0.5821`
- `crossview = 0.7612`
- `conflict rate = 0.0338`
- `crossview 95% bootstrap CI = [0.6567, 0.8507]`

### IAN hurricane panoramic-view benchmark

Current comparison in this repo:

- `crossview` test `F1 = 0.9208`
- `street_only` test `F1 = 0.8912`
- `remote_only` test `F1 = 0.9082`

Conflict subset:

- `street_only = 0.4286`
- `remote_only = 0.5714`
- `crossview = 0.6190`
- `conflict rate = 0.1050`
- `crossview 95% bootstrap CI = [0.4286, 0.8095]`

## Main conclusion

The core pattern transfers across disasters:

- `crossview` is the best overall configuration in both datasets
- its strongest value still appears on the `conflict subset`

But the size of the gain differs:

- on wildfire `building-centric` views, the improvement is stronger
- on hurricane `360/panoramic` views, the improvement is still real but smaller

Right now this should be framed as a supported trend rather than a hard
significance claim, because the conflict-subset bootstrap intervals still
overlap across the two datasets.

This supports a paper story centered on **when cross-view helps most**, rather
than simply whether it helps at all.

## Paper-ready interpretation

Our current interpretation is:

> Cross-view fusion is most valuable under evidence conflict, and its benefit is
> amplified when the ground-view image is tightly aligned with the target
> structure rather than dominated by broad environmental context.

## Related docs

- `README_CN.md`
- `docs/datasets.md`
- `docs/method.md`
- `docs/conclusions.md`
- `docs/paper_story.md`
- `docs/results.md`
