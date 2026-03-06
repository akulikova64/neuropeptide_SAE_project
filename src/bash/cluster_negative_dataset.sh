#!/usr/bin/env bash
set -euo pipefail

# --- Make a no-space alias in your HOME (safe to rerun) ---
ALIAS="$HOME/T7_Shield"
if [[ ! -L "$ALIAS" ]]; then
  ln -s "/Volumes/T7 Shield" "$ALIAS"
fi

# --- Use the alias (no spaces!) ---
IN="$ALIAS/filtered_negative_dataset_from_saved_lists.fasta"
OUT_PREFIX="$ALIAS/negset_cluster40"
TMP_DIR="$ALIAS/negset_cluster40_tmp"
THREADS="${THREADS:-$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu || echo 8 )}"

# sanity checks
[[ -f "$IN" ]] || { echo "Input FASTA not found: $IN" >&2; exit 1; }
mkdir -p "$(dirname "$OUT_PREFIX")" "$TMP_DIR"

echo "Clustering at 40% id, cov>=0.7 (both), threads=$THREADS"
mmseqs easy-cluster "$IN" "$OUT_PREFIX" "$TMP_DIR" \
  --min-seq-id 0.4 \
  -c 0.7 \
  --cov-mode 2 \
  --threads "$THREADS" \
  --remove-tmp-files 0

# representatives written here on success:
REP_FASTA="${OUT_PREFIX}_rep_seq.fasta"

# count reps
COUNT=$(grep -c '^>' "$REP_FASTA")
echo "Final representative count: $COUNT"
echo "Saved representatives → $REP_FASTA"
