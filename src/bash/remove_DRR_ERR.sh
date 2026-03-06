#!/usr/bin/env bash
set -euo pipefail

IN="../../data/amphibian_analysis/combined_Anura_secretomes_pep.fa"
OUT="../../data/amphibian_analysis/combined_Anura_secretomes_no_ERR_DRR.fasta"

# Filter FASTA records:
# - DROP all records whose header ID starts with "DRR"
# - DROP all records whose header ID starts with "ERR", EXCEPT keep ERR13306314..ERR13306324 (inclusive)
awk '
BEGIN { keep=1 }

# Header line: decide whether to keep this whole record
/^>/{
  keep=1
  id=$0
  sub(/^>/, "", id)          # remove leading ">"
  split(id, a, /[ \t]/)      # ID is first token before whitespace/tab
  id=a[1]

  # Always drop DRR*
  if (id ~ /^DRR/) { keep=0 }

  # Drop ERR* unless it is ERR13306314..ERR13306324
  else if (id ~ /^ERR/) {
    if (id ~ /^ERR133063(1[4-9]|2[0-4])(\.|$)/) keep=1
    else keep=0
  }
}

# Print header/sequence lines only if current record is kept
{
  if (keep) print
}
' "$IN" > "$OUT"

echo "Wrote filtered FASTA to: $OUT"