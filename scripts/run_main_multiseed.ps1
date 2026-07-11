param(
  [string]$RepoRoot = "",
  [string]$DatasetRoot = "C:\Users\yyang295\Desktop\disaster-dataset-Yifan-all",
  [string]$AltadenaManifest = "data\manifests\altadena_sensitive_manifest.csv",
  [string]$OutputRoot = "outputs\multiseed_main",
  [int]$Epochs = 3,
  [int]$BatchSize = 64,
  [int]$ImageSize = 224,
  [int]$NumWorkers = 8,
  [string]$Device = "cuda",
  [string]$Seeds = "42,123,456",
  [string]$DatasetNames = "altadena_3class,ian_original,milton_original",
  [string]$Modes = "street_only,remote_only,concat,crossview",
  [switch]$SkipManifestBuild,
  [switch]$Force
)

$ErrorActionPreference = "Continue"
if (-not $RepoRoot) {
  $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
Set-Location $RepoRoot

function Split-List([string]$Value) {
  $items = @()
  foreach ($part in $Value.Split(",")) {
    $trimmed = $part.Trim()
    if ($trimmed) { $items += $trimmed }
  }
  return $items
}

$selectedDatasetNames = Split-List $DatasetNames
$selectedModes = Split-List $Modes
$selectedSeeds = @()
foreach ($seedText in (Split-List $Seeds)) {
  $selectedSeeds += [int]$seedText
}

if (-not $SkipManifestBuild) {
  python scripts\make_group_splits.py `
    --manifest-csv $AltadenaManifest `
    --output-dir data\splits\altadena_3class_objectid `
    --group-col objectid `
    --label-col auto `
    --category-scheme wildfire_3class `
    --path-rewrite-from "C:\Users\yyang295\Desktop\Altadena_Images" `
    --path-rewrite-to (Join-Path $DatasetRoot "Altadena_Images") `
    --require-existing-images
  if ($LASTEXITCODE -ne 0) { throw "Failed to build Altadena 3-class splits." }

  python scripts\build_ian_hurricane_manifests.py `
    --dataset-root (Join-Path $DatasetRoot "IAN_hurricane") `
    --output-dir data\splits\ian_hurricane_original `
    --task original `
    --split-strategy spatial-block `
    --spatial-buffer-m 25
  if ($LASTEXITCODE -ne 0) { throw "Failed to build IAN original splits." }

  python scripts\build_milton_hurricane_manifests.py `
    --dataset-root (Join-Path $DatasetRoot "hurrican-milton-GenDisasterSVI") `
    --output-dir data\splits\milton_hurricane_original `
    --split-strategy spatial-block `
    --spatial-buffer-m 25
  if ($LASTEXITCODE -ne 0) { throw "Failed to build Milton original splits." }
}

$datasets = @(
  @{
    Name = "altadena_3class"
    Train = "data\splits\altadena_3class_objectid\train.csv"
    Val = "data\splits\altadena_3class_objectid\val.csv"
    Test = "data\splits\altadena_3class_objectid\test.csv"
    LabelCol = "label"
  },
  @{
    Name = "ian_original"
    Train = "data\splits\ian_hurricane_original\train.csv"
    Val = "data\splits\ian_hurricane_original\val.csv"
    Test = "data\splits\ian_hurricane_original\test.csv"
    LabelCol = "label"
  },
  @{
    Name = "milton_original"
    Train = "data\splits\milton_hurricane_original\train.csv"
    Val = "data\splits\milton_hurricane_original\val.csv"
    Test = "data\splits\milton_hurricane_original\test.csv"
    LabelCol = "label"
  }
)
$datasets = $datasets | Where-Object { $selectedDatasetNames -contains $_.Name }

foreach ($dataset in $datasets) {
  foreach ($seed in $selectedSeeds) {
    foreach ($mode in $selectedModes) {
      $runDir = Join-Path $OutputRoot (Join-Path $dataset.Name "$($mode)_seed$seed")
      New-Item -ItemType Directory -Force -Path $runDir | Out-Null
      $checkpoint = Join-Path $runDir "triage_best.pt"
      $metrics = Join-Path $runDir "test_metrics.json"
      $predictions = Join-Path $runDir "test_predictions.csv"

      if ($Force -or -not (Test-Path $checkpoint)) {
        Write-Host "Training $($dataset.Name) $mode seed=$seed"
        python scripts\train_triage.py `
          --train-csv $dataset.Train `
          --val-csv $dataset.Val `
          --output-dir $runDir `
          --mode $mode `
          --label-col $dataset.LabelCol `
          --class-weighting balanced `
          --street-augment `
          --overhead-augment `
          --batch-size $BatchSize `
          --epochs $Epochs `
          --image-size $ImageSize `
          --num-workers $NumWorkers `
          --seed $seed `
          --device $Device `
          1> (Join-Path $runDir "train_stdout.log") `
          2> (Join-Path $runDir "train_stderr.log")
        if ($LASTEXITCODE -ne 0) { throw "Training failed for $($dataset.Name) $mode seed=$seed." }
      } else {
        Write-Host "Skipping existing checkpoint $($dataset.Name) $mode seed=$seed"
      }

      if ($Force -or -not ((Test-Path $metrics) -and (Test-Path $predictions))) {
        Write-Host "Evaluating $($dataset.Name) $mode seed=$seed"
        python scripts\eval_triage.py `
          --checkpoint $checkpoint `
          --split-csv $dataset.Test `
          --output-json $metrics `
          --predictions-csv $predictions `
          --batch-size $BatchSize `
          --image-size $ImageSize `
          --num-workers $NumWorkers `
          --device $Device `
          1> (Join-Path $runDir "eval_stdout.log") `
          2> (Join-Path $runDir "eval_stderr.log")
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $($dataset.Name) $mode seed=$seed." }
      }
    }

    $analysisDir = Join-Path $OutputRoot (Join-Path $dataset.Name "analysis_seed$seed")
    New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null
    $streetPreds = Join-Path $OutputRoot (Join-Path $dataset.Name "street_only_seed$seed\test_predictions.csv")
    $remotePreds = Join-Path $OutputRoot (Join-Path $dataset.Name "remote_only_seed$seed\test_predictions.csv")
    $concatPreds = Join-Path $OutputRoot (Join-Path $dataset.Name "concat_seed$seed\test_predictions.csv")
    $crossviewPreds = Join-Path $OutputRoot (Join-Path $dataset.Name "crossview_seed$seed\test_predictions.csv")

    $conflictArgs = @(
      "scripts\build_conflict_subset.py",
      "--split-csv", $dataset.Test,
      "--street-preds-csv", $streetPreds,
      "--remote-preds-csv", $remotePreds,
      "--crossview-preds-csv", $crossviewPreds,
      "--output-csv", (Join-Path $analysisDir "test_conflicts.csv"),
      "--summary-json", (Join-Path $analysisDir "test_conflict_summary.json"),
      "--seed", $seed
    )
    if (Test-Path $concatPreds) { $conflictArgs += @("--concat-preds-csv", $concatPreds) }
    python @conflictArgs
    if ($LASTEXITCODE -ne 0) { throw "Conflict subset build failed for $($dataset.Name) seed=$seed." }

    python scripts\eval_fusion_baselines.py `
      --dataset-name $dataset.Name `
      --street-preds-csv $streetPreds `
      --remote-preds-csv $remotePreds `
      --crossview-preds-csv $crossviewPreds `
      --concat-preds-csv $concatPreds `
      --output-csv (Join-Path $analysisDir "fusion_baselines.csv") `
      --output-json (Join-Path $analysisDir "fusion_baselines.json") `
      --predictions-csv (Join-Path $analysisDir "fusion_baseline_predictions.csv")
    if ($LASTEXITCODE -ne 0) { throw "Fusion baseline evaluation failed for $($dataset.Name) seed=$seed." }

    python scripts\analyze_conflict_statistics.py `
      --dataset-name $dataset.Name `
      --conflict-csv (Join-Path $analysisDir "test_conflicts.csv") `
      --output-json (Join-Path $analysisDir "conflict_statistics.json") `
      --output-csv (Join-Path $analysisDir "conflict_statistics.csv") `
      --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Conflict statistics failed for $($dataset.Name) seed=$seed." }

    python scripts\analyze_multiclass_conflict_thresholds.py `
      --dataset-name $dataset.Name `
      --split-csv $dataset.Test `
      --street-preds-csv $streetPreds `
      --remote-preds-csv $remotePreds `
      --crossview-preds-csv $crossviewPreds `
      --concat-preds-csv $concatPreds `
      --output-dir $analysisDir
    if ($LASTEXITCODE -ne 0) { throw "Threshold analysis failed for $($dataset.Name) seed=$seed." }
  }
}

python scripts\summarize_multiseed_results.py `
  --outputs-root $OutputRoot `
  --datasets $selectedDatasetNames `
  --modes $selectedModes `
  --seeds $selectedSeeds `
  --raw-output-csv (Join-Path $OutputRoot "multiseed_raw.csv") `
  --summary-output-csv (Join-Path $OutputRoot "multiseed_summary.csv") `
  --summary-output-md docs\multiseed_main_results.md
if ($LASTEXITCODE -ne 0) { throw "Failed to summarize multiseed results." }

python scripts\summarize_fusion_baselines.py `
  --outputs-root $OutputRoot `
  --datasets $selectedDatasetNames `
  --seeds $selectedSeeds `
  --raw-output-csv (Join-Path $OutputRoot "fusion_baselines_raw.csv") `
  --summary-output-csv (Join-Path $OutputRoot "fusion_baselines_summary.csv") `
  --summary-output-md docs\fusion_baselines_multiseed.md
if ($LASTEXITCODE -ne 0) { throw "Failed to summarize fusion baseline results." }

python scripts\summarize_conflict_statistics.py `
  --outputs-root $OutputRoot `
  --datasets $selectedDatasetNames `
  --seeds $selectedSeeds `
  --raw-output-csv (Join-Path $OutputRoot "conflict_statistics_raw.csv") `
  --summary-output-csv (Join-Path $OutputRoot "conflict_statistics_summary.csv") `
  --summary-output-md docs\conflict_statistics_multiseed.md
if ($LASTEXITCODE -ne 0) { throw "Failed to summarize conflict statistics." }

python scripts\summarize_threshold_sensitivity.py `
  --outputs-root $OutputRoot `
  --datasets $selectedDatasetNames `
  --seeds $selectedSeeds `
  --raw-output-csv (Join-Path $OutputRoot "threshold_sensitivity_raw.csv") `
  --summary-output-csv (Join-Path $OutputRoot "threshold_sensitivity_summary.csv") `
  --summary-output-md docs\threshold_sensitivity_multiseed.md
if ($LASTEXITCODE -ne 0) { throw "Failed to summarize threshold sensitivity." }
