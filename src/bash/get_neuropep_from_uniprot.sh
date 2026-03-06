#!/bin/bash

# Define output filenames
TSV_FILE="../../data/raw_data/neuropeptides_KW-0527.tsv"
CSV_FILE="../../data/raw_data/neuropeptides_KW-0527.csv"

# Define the corrected UniProt API URL with valid fields
URL="https://rest.uniprot.org/uniprotkb/stream?query=keyword:KW-0527&format=tsv&fields=\
accession,\
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

# Download neuropeptide data from UniProt
echo "Downloading UniProt neuropeptide data..."
curl -o "$TSV_FILE" "$URL"

# Check if download was successful
if [[ -s "$TSV_FILE" ]]; then
    echo "Download complete: $TSV_FILE"

    # Convert TSV to CSV
    echo "Converting TSV to CSV..."
    cat "$TSV_FILE" | tr '\t' ',' > "$CSV_FILE"

    echo "✅ Conversion complete: $CSV_FILE"
else
    echo "❌ Error: Download failed or file is empty!"
    exit 1
fi

