param(
  [string]$RepoRoot = "C:\Users\yyang295\Documents\New project\CrossViewConflict",
  [string]$OutputRoot = "outputs\backbone_sanity",
  [int]$Epochs = 3,
  [int]$BatchSize = 32,
  [int]$ImageSize = 224,
  [int]$NumWorkers = 8,
  [string]$Device = "cuda",
  [string]$Seed = "42",
  [string]$Backbones = "resnet50",
  [string]$DatasetNames = "altadena_3class,ian_original,milton_original",
  [string]$Modes = "street_only,remote_only,crossview",
  [switch]$Force
)

$ErrorActionPreference = "Continue"
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
$selectedBackbones = Split-List $Backbones
$seedInt = [int]$Seed

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
  foreach ($backbone in $selectedBackbones) {
    foreach ($mode in $selectedModes) {
      $runDir = Join-Path $OutputRoot (Join-Path $dataset.Name (Join-Path $backbone "$($mode)_seed$seedInt"))
      New-Item -ItemType Directory -Force -Path $runDir | Out-Null
      $checkpoint = Join-Path $runDir "triage_best.pt"
      $metrics = Join-Path $runDir "test_metrics.json"
      if ($Force -or -not (Test-Path $checkpoint)) {
        Write-Host "Training backbone sanity $($dataset.Name) $backbone $mode seed=$seedInt"
        python scripts\train_triage.py `
          --train-csv $dataset.Train `
          --val-csv $dataset.Val `
          --output-dir $runDir `
          --mode $mode `
          --label-col $dataset.LabelCol `
          --street-backbone $backbone `
          --overhead-backbone $backbone `
          --class-weighting balanced `
          --street-augment `
          --overhead-augment `
          --batch-size $BatchSize `
          --epochs $Epochs `
          --image-size $ImageSize `
          --num-workers $NumWorkers `
          --seed $seedInt `
          --device $Device `
          1> (Join-Path $runDir "train_stdout.log") `
          2> (Join-Path $runDir "train_stderr.log")
        if ($LASTEXITCODE -ne 0) { throw "Training failed for $($dataset.Name) $backbone $mode." }
      }
      if ($Force -or -not (Test-Path $metrics)) {
        python scripts\eval_triage.py `
          --checkpoint $checkpoint `
          --split-csv $dataset.Test `
          --output-json $metrics `
          --predictions-csv (Join-Path $runDir "test_predictions.csv") `
          --batch-size $BatchSize `
          --image-size $ImageSize `
          --num-workers $NumWorkers `
          --device $Device `
          1> (Join-Path $runDir "eval_stdout.log") `
          2> (Join-Path $runDir "eval_stderr.log")
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $($dataset.Name) $backbone $mode." }
      }
    }

    $streetPreds = Join-Path $OutputRoot (Join-Path $dataset.Name (Join-Path $backbone "street_only_seed$seedInt\test_predictions.csv"))
    $remotePreds = Join-Path $OutputRoot (Join-Path $dataset.Name (Join-Path $backbone "remote_only_seed$seedInt\test_predictions.csv"))
    $crossviewPreds = Join-Path $OutputRoot (Join-Path $dataset.Name (Join-Path $backbone "crossview_seed$seedInt\test_predictions.csv"))
    if ((Test-Path $streetPreds) -and (Test-Path $remotePreds) -and (Test-Path $crossviewPreds)) {
      $analysisDir = Join-Path $OutputRoot (Join-Path $dataset.Name (Join-Path $backbone "analysis_seed$seedInt"))
      New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null
      python scripts\build_conflict_subset.py `
        --split-csv $dataset.Test `
        --street-preds-csv $streetPreds `
        --remote-preds-csv $remotePreds `
        --crossview-preds-csv $crossviewPreds `
        --output-csv (Join-Path $analysisDir "test_conflicts.csv") `
        --summary-json (Join-Path $analysisDir "test_conflict_summary.json") `
        --seed $seedInt
      if ($LASTEXITCODE -ne 0) { throw "Backbone conflict subset failed for $($dataset.Name) $backbone." }

      python scripts\analyze_conflict_statistics.py `
        --dataset-name "$($dataset.Name)_$backbone" `
        --conflict-csv (Join-Path $analysisDir "test_conflicts.csv") `
        --output-json (Join-Path $analysisDir "conflict_statistics.json") `
        --output-csv (Join-Path $analysisDir "conflict_statistics.csv") `
        --seed $seedInt
      if ($LASTEXITCODE -ne 0) { throw "Backbone conflict statistics failed for $($dataset.Name) $backbone." }
    }
  }
}

python scripts\summarize_backbone_sanity.py `
  --outputs-root $OutputRoot `
  --datasets $selectedDatasetNames `
  --backbones $selectedBackbones `
  --modes $selectedModes `
  --seeds $seedInt `
  --raw-output-csv (Join-Path $OutputRoot "backbone_sanity_raw.csv") `
  --summary-output-csv (Join-Path $OutputRoot "backbone_sanity_summary.csv") `
  --summary-output-md docs\backbone_sanity_results.md
if ($LASTEXITCODE -ne 0) { throw "Failed to summarize backbone sanity results." }
