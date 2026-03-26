#!/usr/bin/env bash
set -euo pipefail

IN="../../data/fig_6_zebrafish_secretome/secretome_final_cdhit_95_no_A0AB32TF33.fasta"
OUT="../../data/fig_6_zebrafish_secretome/secretome_entry_ids.txt"

# Extract accession between the first two '|' on header lines; write one per line.
awk '
  /^>/ {
    n = split($0, a, /\|/);
    if (n >= 3) print a[2]
  }
' "$IN" > "$OUT"

echo "Wrote $(wc -l < "$OUT") Entry IDs to $OUT"

