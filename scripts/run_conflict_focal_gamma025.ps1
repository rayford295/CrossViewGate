$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = "C:\python310\python.exe"
$outdir = Join-Path $repo "outputs\conflict_focal\altadena_sensitive_gamma025_seed42"

New-Item -ItemType Directory -Force -Path $outdir | Out-Null

Set-Location $repo

& $python scripts/train_triage.py `
  --train-csv data/splits/altadena_sensitive_objectid/train.csv `
  --val-csv data/splits/altadena_sensitive_objectid/val.csv `
  --output-dir outputs/conflict_focal/altadena_sensitive_gamma025_seed42 `
  --mode crossview `
  --street-augment `
  --overhead-augment `
  --batch-size 16 `
  --epochs 10 `
  --seed 42 `
  --conflict-gamma 0.25
