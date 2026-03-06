#!/usr/bin/env bash
# count_clusters.sh
# Count AC entries (space‑separated words) in fold_0.txt … fold_9.txt
# and print the per‑fold counts plus a grand total.

# ── fixed input directory ──────────────────────────────────────────────────
DIR="/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/folds_40"

printf "File\tCount\n"
printf "----\t-----\n"

grand_total=0

# ── loop through files fold_0.txt … fold_9.txt ─────────────────────────────
for f in "$DIR"/fold_{0..9}.txt; do
  if [[ -f "$f" ]]; then
    count=$(wc -w < "$f")
    printf "%s\t%s\n" "$(basename "$f")" "$count"
    grand_total=$((grand_total + count))
  else
    printf "%s\t%s\n" "$(basename "$f")" "file not found"
  fi
done

printf "----\t-----\n"
printf "TOTAL\t%s\n" "$grand_total"
