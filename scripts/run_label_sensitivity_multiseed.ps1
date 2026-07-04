param(
  [string]$RepoRoot = "C:\Users\yyang295\Documents\New project\CrossViewConflict",
  [string]$OutputRoot = "outputs\multiseed_label_sensitivity",
  [int]$Epochs = 3,
  [int]$BatchSize = 64,
  [int]$ImageSize = 224,
  [int]$NumWorkers = 8,
  [string]$Device = "cuda",
  [string]$Seeds = "42,123,456",
  [string]$DatasetNames = "altadena_3class,altadena_original,altadena_sensitive,ian_binary",
  [string]$Modes = "street_only,remote_only,concat,crossview",
  [string]$AltadenaSensitiveManifest = "data\manifests\altadena_sensitive_manifest.csv",
  [string]$AltadenaPathRewriteFrom = "C:\Users\yyang295\Desktop\Altadena_Images",
  [string]$AltadenaPathRewriteTo = "C:\Users\yyang295\Desktop\disaster-dataset-Yifan-all\Altadena_Images",
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
$selectedSeeds = @()
foreach ($seedText in (Split-List $Seeds)) { $selectedSeeds += [int]$seedText }

if ($selectedDatasetNames -contains "altadena_sensitive") {
  $splitDir = "data\splits\altadena_sensitive_objectid"
  python scripts\make_group_splits.py `
    --manifest-csv $AltadenaSensitiveManifest `
    --output-dir $splitDir `
    --group-col objectid `
    --label-col binary_label `
    --seed 42 `
    --require-existing-images `
    --path-rewrite-from $AltadenaPathRewriteFrom `
    --path-rewrite-to $AltadenaPathRewriteTo
  if ($LASTEXITCODE -ne 0) { throw "Failed to prepare filtered altadena_sensitive splits." }
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
    Name = "altadena_original"
    Train = "data\splits\altadena_original_objectid\train.csv"
    Val = "data\splits\altadena_original_objectid\val.csv"
    Test = "data\splits\altadena_original_objectid\test.csv"
    LabelCol = "label"
  },
  @{
    Name = "altadena_sensitive"
    Train = "data\splits\altadena_sensitive_objectid\train.csv"
    Val = "data\splits\altadena_sensitive_objectid\val.csv"
    Test = "data\splits\altadena_sensitive_objectid\test.csv"
    LabelCol = "binary_label"
  },
  @{
    Name = "ian_binary"
    Train = "data\splits\ian_hurricane_minor_vs_moderate_severe_grouped\train.csv"
    Val = "data\splits\ian_hurricane_minor_vs_moderate_severe_grouped\val.csv"
    Test = "data\splits\ian_hurricane_minor_vs_moderate_severe_grouped\test.csv"
    LabelCol = "binary_label"
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
        Write-Host "Training label sensitivity $($dataset.Name) $mode seed=$seed"
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
      }
      if ($Force -or -not ((Test-Path $metrics) -and (Test-Path $predictions))) {
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
    if ($LASTEXITCODE -ne 0) { throw "Conflict subset failed for $($dataset.Name) seed=$seed." }

    $fusionArgs = @(
      "scripts\eval_fusion_baselines.py",
      "--dataset-name", $dataset.Name,
      "--street-preds-csv", $streetPreds,
      "--remote-preds-csv", $remotePreds,
      "--crossview-preds-csv", $crossviewPreds,
      "--output-csv", (Join-Path $analysisDir "fusion_baselines.csv"),
      "--output-json", (Join-Path $analysisDir "fusion_baselines.json"),
      "--predictions-csv", (Join-Path $analysisDir "fusion_baseline_predictions.csv")
    )
    if (Test-Path $concatPreds) { $fusionArgs += @("--concat-preds-csv", $concatPreds) }
    python @fusionArgs
    if ($LASTEXITCODE -ne 0) { throw "Fusion baseline evaluation failed for $($dataset.Name) seed=$seed." }

    python scripts\analyze_conflict_statistics.py `
      --dataset-name $dataset.Name `
      --conflict-csv (Join-Path $analysisDir "test_conflicts.csv") `
      --output-json (Join-Path $analysisDir "conflict_statistics.json") `
      --output-csv (Join-Path $analysisDir "conflict_statistics.csv") `
      --seed $seed
    if ($LASTEXITCODE -ne 0) { throw "Conflict statistics failed for $($dataset.Name) seed=$seed." }
  }
}

python scripts\summarize_multiseed_results.py `
  --outputs-root $OutputRoot `
  --datasets $selectedDatasetNames `
  --modes $selectedModes `
  --seeds $selectedSeeds `
  --raw-output-csv (Join-Path $OutputRoot "multiseed_raw.csv") `
  --summary-output-csv (Join-Path $OutputRoot "multiseed_summary.csv") `
  --summary-output-md docs\label_sensitivity_multiseed_results.md
if ($LASTEXITCODE -ne 0) { throw "Failed to summarize label sensitivity results." }

python scripts\summarize_fusion_baselines.py `
  --outputs-root $OutputRoot `
  --datasets $selectedDatasetNames `
  --seeds $selectedSeeds `
  --raw-output-csv (Join-Path $OutputRoot "fusion_baselines_raw.csv") `
  --summary-output-csv (Join-Path $OutputRoot "fusion_baselines_summary.csv") `
  --summary-output-md docs\label_sensitivity_fusion_baselines.md
if ($LASTEXITCODE -ne 0) { throw "Failed to summarize label sensitivity fusion baselines." }

python scripts\summarize_conflict_statistics.py `
  --outputs-root $OutputRoot `
  --datasets $selectedDatasetNames `
  --seeds $selectedSeeds `
  --raw-output-csv (Join-Path $OutputRoot "conflict_statistics_raw.csv") `
  --summary-output-csv (Join-Path $OutputRoot "conflict_statistics_summary.csv") `
  --summary-output-md docs\label_sensitivity_conflict_statistics.md
if ($LASTEXITCODE -ne 0) { throw "Failed to summarize label sensitivity conflict statistics." }
