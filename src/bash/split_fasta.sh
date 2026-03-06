#!/bin/bash

# splitting was needed to run the sequences through the online Deeploc

# === Configuration ===
input_fasta="../../data/non_neuropeptides_sampled.fasta"     # <-- update this
output_folder="../../data/split_data_non-neuropeptides"                          # folder to store smaller files
sequences_per_file = 500                                # max sequences per split file

# === Create output folder if it doesn't exist ===
mkdir -p "$output_folder"

# === Split using seqkit ===
seqkit split -s "$sequences_per_file" -O "$output_folder" "$input_fasta"

echo "✅ FASTA file split complete! Files are in: $output_folder" 