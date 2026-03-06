#!/usr/bin/env bash
set -euo pipefail

# 1) List of UniProt accessions
IDS=(P01185 P61278 P30990 P01178 P01308 P01275 P09683 P13508 P16043)

# 2) Fields you want in the TSV
FIELDS="accession,id,protein_name,organism_name,sequence,length,cc_ptm,ft_signal,ft_propep,ft_peptide,go_f,go_p,protein_families"

# 3) Build the query string for the TSV
QUERY=$(printf "accession:%s OR " "${IDS[@]}")
QUERY=${QUERY% OR }          # strip trailing “ OR ”
QUERY=${QUERY// /+}          # URL-encode spaces as “+”

# 4) UniProt REST endpoint for TSV
TSV_URL="https://rest.uniprot.org/uniprotkb/search?query=${QUERY}&fields=${FIELDS}&format=tsv"

# 5) Output path for the TSV
OUT_DIR="../../data/known_peptides_test"
mkdir -p "$OUT_DIR"
OUT_TSV="${OUT_DIR}/uniprot_neuropep_entries.tsv"

echo "Fetching TSV entries from UniProt…"
curl -L --fail "$TSV_URL" -o "$OUT_TSV"
echo "✔ TSV saved to $OUT_TSV"

# 6) Now fetch ALL accessions into a single FASTA
FASTA_OUT="${OUT_DIR}/uniprot_known_neuropep_sequences.fasta"
: > "$FASTA_OUT"      # truncate or create empty

echo "Fetching combined FASTA for all accessions…"
for ID in "${IDS[@]}"; do
  echo " • Adding $ID"
  curl -L --fail "https://rest.uniprot.org/uniprotkb/${ID}.fasta" >> "$FASTA_OUT"
done

echo "✔ Combined FASTA saved to $FASTA_OUT"
