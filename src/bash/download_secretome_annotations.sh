#!/usr/bin/env bash
set -euo pipefail

# --- Outputs ---
TSV_FILE="../../data/fig_6_zebrafish_secretome/secretome_annotations_zebrafish.tsv"
CSV_FILE="../../data/fig_6_zebrafish_secretome/secretome_annotations_zebrafish.csv"

# --- UniProt fields (order controls column order in output) ---
FIELDS="accession,\
id,\
protein_name,\
organism_name,\
sequence,length,\
cc_subcellular_location,\
ft_topo_dom,\
cc_tissue_specificity,\
cc_ptm,\
ft_signal,\
ft_propep,\
ft_disulfid,\
ft_carbohyd,\
ft_peptide,\
ft_act_site,\
gene_primary,\
cc_function,\
keyword,\
go_f,\
go_p,\
protein_families,\
cc_domain,\
ft_domain,\
ft_motif,\
annotation_score"

# Ensure output dir exists
mkdir -p "$(dirname "$TSV_FILE")"

# --- Build UniProt REST URL ---
# Use + for spaces and avoid parentheses to keep it URL-safe:
# (taxonomy_id:9606) AND (keyword:KW-0964)  ==>  taxonomy_id:9606+AND+keyword:KW-0964

# Human secretome
#QUERY="taxonomy_id:9606+AND+keyword:KW-0964"
# Mouse (Mus musculus) secretome:
#QUERY="taxonomy_id:10090+AND+keyword:KW-0964"
# C. elegans (Caenorhabditis elegans)
#QUERY="taxonomy_id:6239+AND+keyword:KW-0964"
# Drosophila (Drosophila melanogaster)
#QUERY="taxonomy_id:7227+AND+keyword:KW-0964"
# Drosophila (Drosophila melanogaster)
QUERY="taxonomy_id:7955+AND+keyword:KW-0964"



URL="https://rest.uniprot.org/uniprotkb/stream?query=${QUERY}&format=tsv&fields=${FIELDS}"

echo "Downloading UniProt annotations for query: (${QUERY//+/ AND })"
curl -sS -L -f "$URL" -o "$TSV_FILE"

# Verify and convert
if [[ -s "$TSV_FILE" ]]; then
  echo "Download complete: $TSV_FILE"
  echo "Converting TSV → CSV…"
  tr '\t' ',' < "$TSV_FILE" > "$CSV_FILE"
  echo "✅ Conversion complete: $CSV_FILE"
else
  echo "❌ Error: Download failed or empty file: $TSV_FILE" >&2
  exit 1
fi
