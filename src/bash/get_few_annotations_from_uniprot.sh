#!/bin/bash

# Set fields you want to retrieve
FIELDS="accession,id,protein_name,organism_name,sequence,length,cc_ptm,ft_signal,ft_propep,ft_peptide,go_f,go_p,protein_families"

# Temporary output file
TMP_TSV="uniprot_data.tsv"

# Optional: Add header row (only once)
HEADER_URL="https://rest.uniprot.org/uniprotkb/A0A1B0GTK4.tsv?fields=${FIELDS}"
curl -s "$HEADER_URL" | head -n 1 > "$TMP_TSV"

# Loop through accessions listed in /tmp/uni_ids.txt
while IFS= read -r ACC; do
  [ -z "$ACC" ] && continue  # Skip empty lines
  echo "Fetching $ACC…"
  URL="https://rest.uniprot.org/uniprotkb/${ACC}.tsv?fields=${FIELDS}"
  if ! curl -fsS "$URL" | tail -n +2 >> "$TMP_TSV"; then
    echo "⚠️ Warning: failed to download $ACC" >&2
  fi
done < /tmp/uni_ids.txt

echo "✅ Download complete. Data saved to $TMP_TSV"
