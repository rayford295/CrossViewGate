$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$outDir = "outputs\grouped_rerun\hurricane_minor_vs_severe\crossview_seed42_full"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

python scripts\train_triage.py `
  --train-csv data\splits\ian_hurricane_minor_vs_severe_grouped\train.csv `
  --val-csv data\splits\ian_hurricane_minor_vs_severe_grouped\val.csv `
  --output-dir $outDir `
  --mode crossview `
  --street-augment `
  --overhead-augment `
  --batch-size 16 `
  --epochs 10 `
  --seed 42 `
  1>> "$outDir\train_stdout.log" `
  2>> "$outDir\train_stderr.log"
