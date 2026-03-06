#!/usr/bin/env bash
set -euo pipefail

FASTA="../../data/test_dataset_stats/human_neuro_names_clean.fasta"
CSV="../../data/test_dataset_stats/human_propeptides_clean.csv"

# write header
echo "Entry,propeptide" > "$CSV"

awk '
  BEGIN { OFS="," }
  # on header lines:
  /^>/ {
    # if we already have a sequence buffered, emit it:
    if (NR>1) print entry, seq
    # reset seq, extract the accession between pipes:
    seq = ""
    split($0, parts, "|")
    entry = parts[2]
    next
  }
  # on sequence lines, just append (no spaces):
  {
    gsub(/[ \r\n]/, "")
    seq = seq $0
  }
  END {
    # emit the last record
    if (entry) print entry, seq
  }
' "$FASTA" >> "$CSV"

echo "Wrote CSV to $CSV"
