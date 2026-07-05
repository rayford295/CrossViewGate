# CrossViewConflict

Conflict-aware cross-view disaster triage across two ground-view regimes:

Project website: https://rayford295.github.io/CrossViewConflict/

- wildfire `building/property-centric` views
- hurricane `360/panoramic environment-centric` views

This repository is a focused paper repo built around one specific claim:

> Cross-view fusion helps most when street-side and overhead evidence conflict,
> and the size of that benefit depends on the ground-view regime.

## Current status (journal track, 2026-07)

The journal-track experiments are complete. Headline evidence, all under the
converged 5-seed protocol (`docs/multiseed_v2_results.md`) with pooled
sign-flip statistics (`docs/pooled_seed_tests_v2.md`):

1. **Motivation — the oracle gap.** On conflict cases, an oracle that picks
   the correct single view is 0.24–0.31 above every existing method on all
   three datasets (`docs/calibration_decomposition_v2.md`).
2. **Method — visibility-conditioned reliability gate.** A *linear* gate over
   building-visibility, calibrated-confidence, and disagreement features
   mixes street/remote/crossview probabilities. It is the only method that
   significantly beats calibrated probability averaging (Altadena conflicts
   +0.051, p = 0.0001) and beats crossview on conflicts (+0.072, p < 1e-4)
   and the full test set (+0.018, p < 1e-4), with parity elsewhere
   (`docs/reliability_gate_results_v2.md`).
3. **Mechanism — causal FOV intervention.** Cropping hurricane panoramas to
   building-centered 90° views moves the fusion benefit toward the wildfire
   regime (Milton oracle-gap closure 0.177 → 0.365); random crops with
   identical geometry do not. Direction consistent in 6/6 dataset-seed pairs
   (`docs/fov_intervention_results.md`).
4. **Application — conflict-density damage mapping.** Tile-level cross-view
   conflict density predicts damage without labels (wildfire r = 0.615,
   p = 0.001), while single-view uncertainty density does not
   (`docs/conflict_density_maps.md`).
5. **External reference.** Our IAN data is the released CVIAN pairing; the
   CVDisaster ranking replicates under our stricter grouped split
   (`docs/cvdisaster_comparison.md`). Ordinal metrics in
   `docs/ordinal_metrics_v2.md`.

The sections below describe the original pipeline; historical binary-era
numbers are retained further down for provenance.

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
- `hurrican-milton-GenDisasterSVI`

Example local paths used on our machine:

- `C:/Users/yyang295/Desktop/disaster-dataset-Yifan-all/Altadena_Images`
- `C:/Users/yyang295/Desktop/disaster-dataset-Yifan-all/IAN_hurricane`
- `C:/Users/yyang295/Desktop/disaster-dataset-Yifan-all/hurrican-milton-GenDisasterSVI`

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
- `scripts/build_milton_hurricane_manifests.py`
- `scripts/train_triage.py`
- `scripts/eval_triage.py`
- `scripts/build_conflict_subset.py`
- `scripts/run_original_class_experiments.ps1`
- `scripts/run_main_multiseed.ps1`
- `scripts/run_label_sensitivity_multiseed.ps1`
- `scripts/run_backbone_sanity.ps1`
- `scripts/analyze_building_alignment_multi.py`

## Non-binary task definition

### Wildfire

The Eaton/Altadena wildfire main experiment uses an ordinal 3-class task:

- `0 = no_or_trace_damage`: `No Damage + Affected (1-9%)`
- `1 = damaged_repairable`: `Minor (10-25%) + Major (26-50%)`
- `2 = destroyed`: `Destroyed (>50%)`
- `Inaccessible` excluded

The raw 6-class label space is still supported as a native-label audit.

### Hurricane

The original `IAN_hurricane` dataset is kept as a 3-class task:

- `0_MinorDamage`
- `1_ModerateDamage`
- `2_SevereDamage`

The Milton GenDisasterSVI dataset is also kept as a 3-class task:

- `mild_damage`
- `moderate_damage`
- `severe_damage`

Legacy binary scripts remain supported through `binary_label`, but new
non-binary experiments use `label` / `label_name`, multiclass CE, softmax
predictions, macro-F1, weighted-F1, per-class metrics, and multiclass conflict
subsets.

## Experimental protocol

The paper-facing protocol keeps three label settings:

- main: wildfire 3-class + IAN 3-class + Milton 3-class
- audit: wildfire raw 6-class labels
- sensitivity: legacy binary collapse where historical baselines exist

The main trained modes are:

- `street_only`
- `remote_only`
- `concat`: simple feature concatenation
- `crossview`: concatenation plus absolute-difference and product interactions

Non-trained fusion baselines are evaluated from the single-view predictions:

- late probability averaging
- late logit averaging
- confidence voting

## How to run

### 1. Install

```powershell
cd "C:\Users\yyang295\Documents\New project\CrossViewConflict"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Run non-binary experiments

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_original_class_experiments.ps1 `
  -DatasetRoot "C:\Users\yyang295\Desktop\disaster-dataset-Yifan-all" `
  -DatasetNames "altadena_3class,ian_original,milton_original" `
  -Epochs 10 `
  -BatchSize 32 `
  -NumWorkers 4 `
  -Device cuda
```

The script builds local manifests/splits, trains `street_only`, `remote_only`,
`concat`, and `crossview`, evaluates test metrics, builds conflict subsets, and
writes `docs/original_class_results.md`.

### 3. Run multiseed main experiments

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_main_multiseed.ps1 `
  -DatasetRoot "C:\Users\yyang295\Desktop\disaster-dataset-Yifan-all" `
  -Seeds "42,123,456" `
  -Epochs 3 `
  -BatchSize 64 `
  -NumWorkers 8 `
  -Device cuda
```

This writes per-seed runs under `outputs/multiseed_main`, plus:

- `outputs/multiseed_main/multiseed_summary.csv`
- `outputs/multiseed_main/fusion_baselines_summary.csv`
- `outputs/multiseed_main/conflict_statistics_summary.csv`
- `outputs/multiseed_main/threshold_sensitivity_summary.csv`
- `docs/multiseed_main_results.md`
- `docs/fusion_baselines_multiseed.md`
- `docs/conflict_statistics_multiseed.md`
- `docs/threshold_sensitivity_multiseed.md`
- per-seed fusion baselines, conflict statistics, and threshold sweeps

### 4. Run label-sensitivity experiments

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_label_sensitivity_multiseed.ps1 `
  -Seeds "42,123,456" `
  -Epochs 3 `
  -BatchSize 64 `
  -NumWorkers 8 `
  -Device cuda
```

This keeps the wildfire 3-class main setting, wildfire 6-class audit, and
legacy binary sensitivity in parallel. The default modes are `street_only`,
`remote_only`, `concat`, and `crossview`, and the script writes both model and
fusion summaries:

- `outputs/multiseed_label_sensitivity/multiseed_summary.csv`
- `outputs/multiseed_label_sensitivity/fusion_baselines_summary.csv`
- `outputs/multiseed_label_sensitivity/conflict_statistics_summary.csv`
- `docs/label_sensitivity_multiseed_results.md`
- `docs/label_sensitivity_fusion_baselines.md`
- `docs/label_sensitivity_conflict_statistics.md`

### 5. Run stronger-backbone sanity checks

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_backbone_sanity.ps1 `
  -Backbones "resnet50" `
  -Modes "street_only,remote_only,crossview" `
  -Epochs 3 `
  -BatchSize 32 `
  -Device cuda
```

This trains the selected backbone for the main datasets, builds conflict
subsets when all three modes are present, and writes:

- `outputs/backbone_sanity/backbone_sanity_summary.csv`
- `docs/backbone_sanity_results.md`

### 6. Run building-alignment mechanism analysis

```powershell
python scripts\analyze_building_alignment_multi.py `
  --dataset altadena_3class=outputs\multiseed_main\altadena_3class\analysis_seed42\test_conflicts.csv `
  --dataset ian_original=outputs\multiseed_main\ian_original\analysis_seed42\test_conflicts.csv `
  --dataset milton_original=outputs\multiseed_main\milton_original\analysis_seed42\test_conflicts.csv `
  --output-dir outputs\analysis\building_alignment_main_seed42 `
  --model-id nvidia/segformer-b0-finetuned-ade-512-512 `
  --batch-size 4 `
  --device cuda
```

The paper-facing mechanism summary is in
`docs/building_alignment_main_seed42.md`. A concise interpretation of the full
experimental package is in `docs/paper_experiment_interpretation.md`.

## Historical binary findings

The numbers below are previous binary/collapsed-label results. For the current
non-binary results, see `docs/original_class_results.md`.

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

### Clean grouped hurricane rerun

We also rebuilt the endpoint hurricane split with `objectid` grouping to remove
the leakage found in the earlier convenience split, then reran the core
baselines.

- `crossview`: test `F1 = 0.9008`
- `street_only`: test `F1 = 0.9016`
- `remote_only`: test `F1 = 0.8385`

This is an important correction for the paper:

- the hurricane benchmark remains learnable on a clean grouped split
- `remote_only` drops substantially
- `crossview` and `street_only` become nearly tied, so the clean-split story
  should be framed more carefully than the original leaked result

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

We also added a wider-threshold permutation check using `tau = 0.1`.

- wildfire: `n = 435`, `crossview accuracy = 0.7885`, label-independence
  `p < 1e-4`
- hurricane: `n = 128`, `crossview accuracy = 0.7500`, label-independence
  `p < 1e-4`

So the conflict-aware story still holds under a softer disagreement definition,
not just under the original hard conflict subset.

## Conflict-aware training sweep

We completed the first `ConflictFocalLoss` sweep on the wildfire sensitive
split.

- standard `crossview` baseline: test `F1 = 0.9473`
- `ConflictFocalLoss (gamma = 0.1)`: test `F1 = 0.9514`
- `ConflictFocalLoss (gamma = 0.25)`: test `F1 = 0.9507`
- `ConflictFocalLoss (gamma = 0.5)`: test `F1 = 0.9520`
- `ConflictFocalLoss (gamma = 1.0)`: test `F1 = 0.9470`

The best result in the current sweep is `gamma = 0.5`, and all moderate
settings (`0.1`, `0.25`, `0.5`) beat the standard baseline. This is a stronger
signal than the original single-point pilot because it shows conflict-aware
training is useful across a small range of hyperparameters rather than only at
one cherry-picked value.

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
