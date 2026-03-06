#!/usr/bin/env bash
set -euo pipefail

ulimit -n 65536 || true

python "/Users/anastasiyakulikova/Desktop/SAE_project/src/python/filter_for_neg_dataset.py" \
  --download \
  --neuropep-fasta "/Volumes/T7 Shield/neuropep_training_SAE.fasta" \
  --outdir "/Users/anastasiyakulikova/Desktop/SAE_project/data/negatives_build" \
  --max-download 150000 \
  --page-size 500 \
  --reviewed \
  --identity 0.40 \
  --coverage 0.80 \
  --cov-mode 1 \
  --final-n 32493 \
  --threads 16
