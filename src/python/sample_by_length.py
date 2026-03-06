import pandas as pd
from random import sample
from Bio import SeqIO
import os

# Important: Run this script right after running "split_seqs_by_length.py"

# The non-neuropeptides need to be sampled by length to match the length 
# distribution of the neuropeptides dataset from Uniprot. 

binning_plan = pd.read_csv("../../data/sequence_counts_per_bin_group_1.csv")
input_path = "../../data/group_1_binned"

def sample_by_index(fasta_path, n):
    # First pass to count sequences
    count = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
    
    # Randomly sample indices
    indices_to_keep = set(sample(range(count), n))

    # Second pass to extract only those that were selected above
    sampled = []
    for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        if i in indices_to_keep:
            sampled.append(record)
        if len(sampled) == n:
            break
    return sampled

sampled_records = []

for _, row in binning_plan.iterrows():
    bin_label = row["bin"]
    n = row["sequence_count"]
    fasta_path = os.path.join(input_path, f"{bin_label}.fasta")

    if not os.path.exists(fasta_path):
        print(f"❌ Missing file: {fasta_path}")
        continue

    # Use memory-efficient sampling
    count = sum(1 for _ in SeqIO.parse(fasta_path, "fasta"))
    if count < n:
        print(f"⚠️ Not enough sequences in bin {bin_label}. Skipping or taking all.")
        selected = list(SeqIO.parse(fasta_path, "fasta"))
    #elif n < 100:
        #selected = sample_by_index(fasta_path, 100)
    else:
        selected = sample_by_index(fasta_path, n)
        
        sampled_records.extend(selected)

#SeqIO.write(sampled_records, "../../data/non_neuropeptides_sampled.fasta", "fasta")
SeqIO.write(sampled_records, "../../data/group_1_negative.fasta", "fasta")




