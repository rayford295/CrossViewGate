#!/bin/bash
# Phase 2 FOV intervention: retrain street_only + crossview on building-centered
# and random-centered panorama crops, matching the main multiseed protocol
# (epochs=3, batch 64, image 224, balanced weighting, augmentation).
# remote_only is reused from outputs/multiseed_main (overhead views unchanged).
cd "$(dirname "$0")/.."

EPOCHS=3
for ds in ian_original milton_original; do
  for variant in building random; do
    split_dir="data/splits/${ds}_fov_${variant}"
    for seed in 42 123 456; do
      for mode in street_only crossview; do
        run_dir="outputs/fov_intervention/${ds}_fov_${variant}/${mode}_seed${seed}"
        mkdir -p "$run_dir"
        if [ ! -f "$run_dir/triage_best.pt" ]; then
          echo "TRAIN $ds $variant $mode seed$seed"
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
            --epochs $EPOCHS \
            --image-size 224 \
            --num-workers 8 \
            --seed "$seed" \
            --device cuda \
            > "$run_dir/train_stdout.log" 2> "$run_dir/train_stderr.log" \
            || { echo "TRAIN_FAILED $ds $variant $mode seed$seed"; continue; }
        fi
        if [ ! -f "$run_dir/test_predictions.csv" ]; then
          echo "EVAL $ds $variant $mode seed$seed"
          python scripts/eval_triage.py \
            --checkpoint "$run_dir/triage_best.pt" \
            --split-csv "$split_dir/test.csv" \
            --output-json "$run_dir/test_metrics.json" \
            --predictions-csv "$run_dir/test_predictions.csv" \
            --batch-size 64 --image-size 224 --num-workers 4 --device cuda \
            > "$run_dir/eval_stdout.log" 2> "$run_dir/eval_stderr.log" \
            || echo "EVAL_FAILED $ds $variant $mode seed$seed"
        fi
      done
    done
  done
done
echo ALL_DONE
