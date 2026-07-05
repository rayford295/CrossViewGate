#!/bin/bash
# Converged 5-seed main suite (v2): epochs=15 with patience=4 early stopping,
# otherwise matched to run_main_multiseed.ps1 (batch 64, image 224, balanced
# class weighting, augmentation). Produces test AND val predictions per run so
# the calibration/gate analyses can be rerun under the converged protocol.
cd "/c/Users/yyang295/Documents/New project/CrossViewConflict"

declare -A SPLITS=(
  [altadena_3class]="data/splits/altadena_3class_objectid"
  [ian_original]="data/splits/ian_hurricane_original"
  [milton_original]="data/splits/milton_hurricane_original"
)

for ds in altadena_3class ian_original milton_original; do
  split_dir="${SPLITS[$ds]}"
  for seed in 42 123 456 789 1011; do
    for mode in street_only remote_only concat crossview; do
      run_dir="outputs/multiseed_v2/$ds/${mode}_seed${seed}"
      mkdir -p "$run_dir"
      if [ ! -f "$run_dir/triage_best.pt" ]; then
        echo "TRAIN $ds $mode seed$seed"
        python scripts/train_triage.py \
          --train-csv "$split_dir/train.csv" \
          --val-csv "$split_dir/val.csv" \
          --output-dir "$run_dir" \
          --mode "$mode" \
          --label-col label \
          --class-weighting balanced \
          --street-augment \
          --overhead-augment \
          --batch-size 64 \
          --epochs 15 \
          --patience 4 \
          --image-size 224 \
          --num-workers 8 \
          --seed "$seed" \
          --device cuda \
          > "$run_dir/train_stdout.log" 2> "$run_dir/train_stderr.log" \
          || { echo "TRAIN_FAILED $ds $mode seed$seed"; continue; }
      fi
      for split in test val; do
        pred="$run_dir/${split}_predictions.csv"
        if [ ! -f "$pred" ]; then
          echo "EVAL $ds $mode seed$seed $split"
          python scripts/eval_triage.py \
            --checkpoint "$run_dir/triage_best.pt" \
            --split-csv "$split_dir/${split}.csv" \
            --output-json "$run_dir/${split}_metrics.json" \
            --predictions-csv "$pred" \
            --batch-size 64 --image-size 224 --num-workers 4 --device cuda \
            > "$run_dir/eval_${split}_stdout.log" 2> "$run_dir/eval_${split}_stderr.log" \
            || echo "EVAL_FAILED $ds $mode seed$seed $split"
        fi
      done
    done
  done
done
echo ALL_DONE
