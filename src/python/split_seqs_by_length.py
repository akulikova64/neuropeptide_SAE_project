from Bio import SeqIO
import os

# this script is run right before sample_by_length.py. The sequences already need to be binned
# into their corresponding length folders before being sampled from each folder.

#all_sequences_path = "../../data/non-neuropeptides_raw_unsampled.fasta"
all_sequences_path = "../../data/neuropeptide_hormone_activity_1.fasta" # now called: "group_1_positive.fasta"

# Define bins and output folders
bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250),
        (250, 300), (300, 350), (350, 400), (400, 450), (450, 500),
        (500, 550), (550, 600)]

#output_dir = "../../data/non-neuropeptides_binned"
output_dir = "../../data/group_1_binned"
os.makedirs(output_dir, exist_ok=True)

# Open output files for each bin
bin_files = {
    f"{low}-{high-1}": open(f"{output_dir}/{low}-{high-1}.fasta", "w")
    for (low, high) in bins
}

# Stream through the massive FASTA file
with open(all_sequences_path) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        length = len(record.seq)
        for (low, high) in bins:
            if low <= length < high:
                SeqIO.write(record, bin_files[f"{low}-{high-1}"], "fasta")
                break  # Once written, no need to check other bins

# Close all bin files
for f in bin_files.values():
    f.close()