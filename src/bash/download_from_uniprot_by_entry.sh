#!/usr/bin/env bash
set -euo pipefail

#INPUT_CSV="../../data/test_dataset_stats/human_mature_pep.csv"
#OUTPUT_CSV="../../data/test_dataset_stats/human_mature_pep_annotations.csv"
INPUT_CSV="../../data/test_dataset_stats/human_mature_pep.csv"
OUTPUT_CSV="../../data/test_dataset_stats/human_mature_pep_annotations.csv"
TMP_TSV="${OUTPUT_CSV%.csv}.tsv"

FIELDS="accession,id,protein_name,organism_name,sequence,length,cc_ptm,ft_signal,ft_propep,ft_peptide,go_f,go_p,protein_families"

mkdir -p "$(dirname "$OUTPUT_CSV")"

ENTRY_COL_IDX=$(head -n1 "$INPUT_CSV" \
  | tr ',' '\n' \
  | nl -w1 -s',' \
  | awk -F',' 'tolower($2)=="uniprot" {print $1}')

if [[ -z "$ENTRY_COL_IDX" ]]; then
  echo "Error: could not find a 'Uniprot' column in $INPUT_CSV" >&2
  exit 1
fi

# write a single header
echo "$FIELDS" > "$TMP_TSV"

# Build a list of unique accessions, then read line-by-line
tail -n +2 "$INPUT_CSV" \
  | cut -d',' -f"$ENTRY_COL_IDX" \
  | tr -d '\r' \
  | grep -v '^$' \
  | sort -u \
  > /tmp/uni_ids.txt

while IFS= read -r raw; do
  # raw now has no CR, no splitting
  ACC="$raw"
  echo "Fetching $ACC…"
  URL="https://rest.uniprot.org/uniprotkb/${ACC}.tsv?fields=${FIELDS}"
  if ! curl -fSs "$URL" | tail -n +2 >> "$TMP_TSV"; then
    echo "Warning: failed to download $ACC" >&2
  fi
done < /tmp/uni_ids.txt

# convert TSV to CSV
tr '\t' ',' < "$TMP_TSV" > "$OUTPUT_CSV"
echo "✅ Annotations written to $OUTPUT_CSV"
