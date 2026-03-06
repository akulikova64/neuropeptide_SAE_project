#!/bin/bash

# this script is used for concatenating the deeploc csv results

# === Configuration ===
input_folder="../../data/deeploc_non-neuropeptides"             # folder with your 22 CSV files
output_file="../../data/combined_deeploc_results_non-neuro.csv" # name of final combined file

# === Concatenate CSV files ===

# Get the first CSV file only
first_file=$(ls "$input_folder"/*.csv | head -n 1)

# Write its header to the output file
head -n 1 "$first_file" > "$output_file"

# Append all rows from all CSVs, skipping their headers
for file in "$input_folder"/*.csv; do
    tail -n +2 "$file" >> "$output_file"
done

echo "Combined CSV written to: $output_file"