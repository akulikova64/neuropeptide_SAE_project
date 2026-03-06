# Define output filenames
TSV_FILE="../../data/test_dataset_stats/toxins_KW-0800.tsv"
CSV_FILE="../../data/test_dataset_stats/toxins_KW-0800.csv"

# Define the corrected UniProt API URL with valid fields
URL="https://rest.uniprot.org/uniprotkb/stream?query=(keyword:KW-0800)%20AND%20(reviewed:true)%20AND%20(taxonomy_id:33208)&format=tsv&fields=\
accession,\
id,\
protein_name,\
organism_name,\
sequence,length,\
cc_ptm,\
ft_signal,\
ft_propep,\
ft_peptide,\
go_f,\
go_p,\
protein_families"

# Download data from UniProt
echo "Downloading UniProt data..."
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

