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

### Wildfire backbone ablation snapshot

We also started a backbone robustness sweep on the wildfire `crossview`
benchmark.

- `ResNet18`: best val `F1 = 0.9676`
- `ResNet50`: best val `F1 = 0.9693`
- `DINOv2 ViT-S/14`: best val `F1 = 0.9378`
- `CLIP ViT-B/32`: best val `F1 = 0.7502`

This already supports one useful paper claim:

- the main `crossview` result is **not** a `ResNet18` accident
- a stronger supervised CNN (`ResNet50`) preserves the finding
- generic large-scale pretraining (`CLIP`) is not automatically better for
  this disaster-specific cross-view triage task

### Wildfire multi-seed snapshot

The wildfire benchmark now has complete three-seed validation runs for all
three modes:

- `crossview`: `0.9689 ± 0.0011`
- `street_only`: `0.9657 ± 0.0001`
- `remote_only`: `0.9654 ± 0.0015`

So the main wildfire conclusion is now supported not only by single best runs,
but also by low-variance multi-seed evidence.

### Hurricane backbone ablation snapshot

The hurricane backbone-ablation sweep is now complete:

- `ResNet18`: best val `F1 = 0.9694`
- `ResNet50`: best val `F1 = 0.9470`
- `CLIP ViT-B/32`: best val `F1 = 0.6437`
- `DINOv2 ViT-S/14`: best val `F1 = 0.7975`

This now makes the hurricane backbone story quite clear:

- the earlier `ResNet18` baseline remains strongest
- `ResNet50` is still competitive, but lower
- `DINOv2` is much better than `CLIP`, but still below the CNN baselines
- `CLIP` performs much worse here, which reinforces the idea that larger or
  more generic pretraining is not automatically better in this task

### Hurricane multi-seed snapshot

The hurricane benchmark now also has complete three-seed validation runs for
all three modes:

- `crossview`: `0.9455 ± 0.0063`
- `street_only`: `0.9433 ± 0.0012`
- `remote_only`: `0.8959 ± 0.0065`

This gives the paper a second stability result:

- `crossview` remains the strongest hurricane mode on average
- `street_only` is close, but consistently lower
- `remote_only` is clearly weaker and more variable

### Hurricane label-sensitivity snapshot

We also started a stricter label-sensitivity check where the hurricane task is
redefined as:

- `0 = MinorDamage`
- `1 = ModerateDamage + SevereDamage`

On this harder split, the first completed test runs are:

- `crossview`: test `F1 = 0.8915`
- `street_only`: test `F1 = 0.8738`
- `remote_only`: test `F1 = 0.8873`

So the task clearly becomes harder once `ModerateDamage` is folded into the
positive class, but the same rank order still appears so far:

- `crossview > remote_only > street_only`

### Wildfire label-sensitivity snapshot

We also completed a stricter wildfire binary mapping where the task is
redefined as:

- `0 = No Damage`
- `1 = Affected + Minor + Major + Destroyed`

On this broader positive-class split, all three test runs are now complete:

- `crossview`: test `F1 = 0.9473`
- `street_only`: test `F1 = 0.9392`
- `remote_only`: test `F1 = 0.9238`

So the wildfire conclusion also survives this robustness check:

- `crossview > street_only > remote_only`
- the stronger property-centric regime still benefits most from cross-view
  fusion even under a more sensitive damage definition

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

## Lightweight alignment analysis

We also ran a lightweight semantic-segmentation analysis on the `conflict subset`
to estimate how much of each ground-view image is occupied by `building`
pixels and how centrally those building pixels are located.

Using `nvidia/segformer-b5-finetuned-ade-640-640` as a frozen analyzer:

- wildfire conflict images:
  - mean `building_ratio = 0.2684`
  - mean `center_building_ratio = 0.4101`
  - mean normalized centroid distance `= 0.2958`
- hurricane conflict images:
  - mean `building_ratio = 0.0154`
  - mean `center_building_ratio = 0.0271`
  - mean normalized centroid distance `= 0.3965`

This is exactly the mechanism we wanted to test:

- wildfire conflict images contain much more visible building area
- that building evidence is also more centered
- hurricane conflict images are much more environment-dominant

So the smaller hurricane crossview gain is no longer just an observation. It is
now supported by a measurable target-alignment proxy.

## Additional statistical checks

We also completed four reviewer-oriented supplementary analyses:

1. `Permutation test on conflict subset`
   - wildfire: crossview accuracy is clearly above the label-independence null
   - hurricane: the same trend exists, but the current conflict subset is too small for a strong significance claim
2. `Threshold sensitivity`
   - crossview stays best across soft conflict thresholds in both datasets
3. `Per-sample alignment correlation`
   - the current building-ratio proxy does **not** show a strong monotonic per-sample correlation with crossview correctness
   - this suggests the alignment effect is stronger at the `dataset / view-regime` level than as a within-conflict ranking signal
4. `Qualitative figure`
   - a conflict-case figure with street image, overhead patch, prediction table, and building overlay is now available

## Paper-ready interpretation

Our current interpretation is:

> Cross-view fusion is most valuable under evidence conflict, and its benefit is
> amplified when the ground-view image is tightly aligned with the target
> structure rather than dominated by broad environmental context.

## Related docs

- `README_CN.md`
- `docs/alignment_analysis.md`
- `docs/statistical_checks.md`
- `docs/datasets.md`
- `docs/method.md`
- `docs/conclusions.md`
- `docs/paper_story.md`
- `docs/results.md`
