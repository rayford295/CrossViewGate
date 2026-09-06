param(
  [string]$RepoRoot = "",
  [string]$DatasetRoot = $env:CROSSVIEW_DATASET_ROOT,                     # local root holding Altadena_Images/, IAN_hurricane/, ...
  [string]$AltadenaPathRewriteFrom = $env:ALTADENA_MANIFEST_IMAGE_ROOT,  # image root recorded inside the Altadena manifest CSV
  [string]$AltadenaManifest = "data\manifests\altadena_sensitive_manifest.csv",
  [string]$OutputRoot = "outputs\original_class",
  [int]$Epochs = 10,
  [int]$BatchSize = 16,
  [int]$ImageSize = 224,
  [int]$NumWorkers = 4,
  [string]$Device = "cuda",
  [string]$DatasetNames = "altadena_3class,ian_original,milton_original",
  [string]$Modes = "street_only,remote_only,concat,crossview",
  [switch]$SkipTraining,
  [switch]$SkipManifestBuild
)

$ErrorActionPreference = "Continue"

if (-not $RepoRoot) {
  $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

Set-Location $RepoRoot
$selectedDatasetNames = @()
foreach ($name in $DatasetNames.Split(",")) {
  $trimmed = $name.Trim()
  if ($trimmed) { $selectedDatasetNames += $trimmed }
}
$selectedModes = @()
foreach ($name in $Modes.Split(",")) {
  $trimmed = $name.Trim()
  if ($trimmed) { $selectedModes += $trimmed }
}

if (-not $SkipManifestBuild) {
  python scripts\make_group_splits.py `
    --manifest-csv $AltadenaManifest `
    --output-dir data\splits\altadena_original_objectid `
    --group-col objectid `
    --label-col auto `
    --category-scheme wildfire_original `
    --path-rewrite-from $AltadenaPathRewriteFrom `
    --path-rewrite-to (Join-Path $DatasetRoot "Altadena_Images") `
    --require-existing-images
  if ($LASTEXITCODE -ne 0) { throw "Failed to build Altadena original splits." }

  python scripts\make_group_splits.py `
    --manifest-csv $AltadenaManifest `
    --output-dir data\splits\altadena_3class_objectid `
    --group-col objectid `
    --label-col auto `
    --category-scheme wildfire_3class `
    --path-rewrite-from $AltadenaPathRewriteFrom `
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
    Name = "altadena_original"
    Train = "data\splits\altadena_original_objectid\train.csv"
    Val = "data\splits\altadena_original_objectid\val.csv"
    Test = "data\splits\altadena_original_objectid\test.csv"
  },
  @{
    Name = "altadena_3class"
    Train = "data\splits\altadena_3class_objectid\train.csv"
    Val = "data\splits\altadena_3class_objectid\val.csv"
    Test = "data\splits\altadena_3class_objectid\test.csv"
  },
  @{
    Name = "ian_original"
    Train = "data\splits\ian_hurricane_original\train.csv"
    Val = "data\splits\ian_hurricane_original\val.csv"
    Test = "data\splits\ian_hurricane_original\test.csv"
  },
  @{
    Name = "milton_original"
    Train = "data\splits\milton_hurricane_original\train.csv"
    Val = "data\splits\milton_hurricane_original\val.csv"
    Test = "data\splits\milton_hurricane_original\test.csv"
  }
)

$datasets = $datasets | Where-Object { $selectedDatasetNames -contains $_.Name }

foreach ($dataset in $datasets) {
  foreach ($mode in $selectedModes) {
    $runDir = Join-Path $OutputRoot (Join-Path $dataset.Name $mode)
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    if (-not $SkipTraining) {
      python scripts\train_triage.py `
        --train-csv $dataset.Train `
        --val-csv $dataset.Val `
        --output-dir $runDir `
        --mode $mode `
        --label-col label `
        --class-weighting balanced `
        --street-augment `
        --overhead-augment `
        --batch-size $BatchSize `
        --epochs $Epochs `
        --image-size $ImageSize `
        --num-workers $NumWorkers `
        --device $Device `
        1> (Join-Path $runDir "train_stdout.log") `
        2> (Join-Path $runDir "train_stderr.log")
      if ($LASTEXITCODE -ne 0) { throw "Training failed for $($dataset.Name) $mode." }
    }

    $checkpoint = Join-Path $runDir "triage_best.pt"
    if (Test-Path $checkpoint) {
      python scripts\eval_triage.py `
        --checkpoint $checkpoint `
        --split-csv $dataset.Test `
        --output-json (Join-Path $runDir "test_metrics.json") `
        --predictions-csv (Join-Path $runDir "test_predictions.csv") `
        --batch-size $BatchSize `
        --image-size $ImageSize `
        --num-workers $NumWorkers `
        --device $Device
      if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $($dataset.Name) $mode." }
    } else {
      Write-Warning "Skipping evaluation for $($dataset.Name) $mode because no checkpoint exists."
    }
  }

  $analysisDir = Join-Path $OutputRoot (Join-Path $dataset.Name "analysis")
  New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null
  $streetPreds = Join-Path $OutputRoot (Join-Path $dataset.Name "street_only\test_predictions.csv")
  $remotePreds = Join-Path $OutputRoot (Join-Path $dataset.Name "remote_only\test_predictions.csv")
  $concatPreds = Join-Path $OutputRoot (Join-Path $dataset.Name "concat\test_predictions.csv")
  $crossviewPreds = Join-Path $OutputRoot (Join-Path $dataset.Name "crossview\test_predictions.csv")
  if ((Test-Path $streetPreds) -and (Test-Path $remotePreds) -and (Test-Path $crossviewPreds)) {
    $conflictArgs = @(
      "scripts\build_conflict_subset.py",
      "--split-csv", $dataset.Test,
      "--street-preds-csv", $streetPreds,
      "--remote-preds-csv", $remotePreds,
      "--crossview-preds-csv", $crossviewPreds,
      "--output-csv", (Join-Path $analysisDir "test_conflicts.csv"),
      "--summary-json", (Join-Path $analysisDir "test_conflict_summary.json")
    )
    if (Test-Path $concatPreds) {
      $conflictArgs += @("--concat-preds-csv", $concatPreds)
    }
    python @conflictArgs
    if ($LASTEXITCODE -ne 0) { throw "Conflict subset build failed for $($dataset.Name)." }
  } else {
    Write-Warning "Skipping conflict subset for $($dataset.Name) because prediction CSVs are missing."
  }
}

python scripts\summarize_original_class_results.py `
  --outputs-root $OutputRoot `
  --datasets $selectedDatasetNames `
  --output-csv (Join-Path $OutputRoot "summary.csv") `
  --output-md docs\original_class_results.md
if ($LASTEXITCODE -ne 0) { throw "Failed to summarize original-class results." }
