#!/usr/bin/env bash
set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────
INPUT_CSV="../../data/test_dataset_stats/human_neuro_names_clean.csv"
OUTPUT_FASTA="../../data/test_dataset_stats/human_neuro_names_clean.fasta"

# ─── Prep output ─────────────────────────────────────────────────────────
mkdir -p "$(dirname "$OUTPUT_FASTA")"
: > "$OUTPUT_FASTA"

# ─── Download loop ───────────────────────────────────────────────────────
tail -n +2 "$INPUT_CSV" | \
while IFS=, read -r raw_entry _; do
  # 1) clean the ID: remove quotes, CRs, trim whitespace
  entry="${raw_entry//\"/}"
  entry="${entry//$'\r'/}"
  entry="$(echo -n "$entry" | xargs)"   # trim leading/trailing spaces

  # skip empties
  [[ -z "$entry" ]] && continue

  echo "Fetching $entry …"
  if ! curl -s --fail "https://rest.uniprot.org/uniprotkb/${entry}.fasta" >> "$OUTPUT_FASTA"; then
    echo "⚠️  Failed to fetch $entry (invalid accession or network error)" >&2
  else
    echo >> "$OUTPUT_FASTA"
  fi
done

echo "✅ Done. Sequences written to $OUTPUT_FASTA"
