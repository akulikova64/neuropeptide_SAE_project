#!/usr/bin/env bash
set -euo pipefail

# Input: space-separated UniProt accessions
INPUT_IDS="../../data/fig_6_zebrafish_secretome/secretome_entry_ids.txt"

# Outputs
OUT_CSV="../../data/fig_6_zebrafish_secretome/secretome_neuropeptide_flags.csv"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# 1) Normalize your IDs to one-per-line
tr ' \t' '\n' < "$INPUT_IDS" | sed '/^$/d' | sort -u > "$TMP_DIR/secretome_ids.txt"

# 2) Fetch all HUMAN entries annotated with Neuropeptide or Hormone keywords
#    (We only need accession + keyword; we’ll intersect by accession)
UNI_URL="https://rest.uniprot.org/uniprotkb/stream"
UNI_QUERY="(taxonomy_id:7955) AND (keyword:Neuropeptide OR keyword:Hormone)"

echo "Fetching human Neuropeptide/Hormone accessions from UniProt…"
curl -fsSL \
  --get "$UNI_URL" \
  --data-urlencode "query=$UNI_QUERY" \
  --data-urlencode "format=tsv" \
  --data-urlencode "fields=accession,keyword" \
  > "$TMP_DIR/uniprot_kw.tsv"

# 3) Extract the matched accessions (skip header)
tail -n +2 "$TMP_DIR/uniprot_kw.tsv" | cut -f1 | sort -u > "$TMP_DIR/neuro_hormone_ids.txt"

# 4) Build the output CSV: Entry,neuropeptide (TRUE/FALSE)
echo "Entry,neuropeptide" > "$OUT_CSV"
# Use a hash set for quick membership checks
awk 'NR==FNR {a[$0]=1; next} {flag = ($0 in a) ? "TRUE" : "FALSE"; print $0 "," flag}' \
  "$TMP_DIR/neuro_hormone_ids.txt" \
  "$TMP_DIR/secretome_ids.txt" \
  >> "$OUT_CSV"

echo "✅ Wrote: $OUT_CSV"
