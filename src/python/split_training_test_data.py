#!/usr/bin/env python3
"""
This script performs the following steps:

1. Reads “cluster_sizes.tsv” and selects all entries whose cluster size is exactly 1.
2. Randomly samples 500 of those singleton‐cluster entries.
3. From “../../data/pph.fa”:
     a) Writes the 500 sampled entries into “../../data/SAE_training/test_50_homology.fasta”
     b) Writes all remaining entries into “../../data/SAE_training/all_training_data.fasta”
4. Reads “cluster_members.tsv” and writes a new file “cluster_members_training.tsv” that excludes the 500 sampled entries.

Usage:
    python3 sample_singletons.py
"""

#NOTE: Run this script before "split_training_val_data.py"!!!!!!!!!!!

import os
import random
import pandas as pd
from Bio import SeqIO

# ─── PARAMETERS & PATHS ────────────────────────────────────────────────────────

# Path to "cluster_sizes.tsv": two columns (entry, cluster_size), no header
cluster_sizes_tsv   = "../../data/SAE_training_data_mmseqs50/cluster_sizes.tsv"

# Path to the full FASTA from which to extract sequences
pph_fasta           = "../../data/pph.fa"

# Path to the TSV that maps entries to clusters/members
cluster_members_tsv = "../../data/SAE_training_data_mmseqs50/cluster_members.tsv"

# Where to write the sampled 500‐entry FASTA
test_fasta          = "../../data/SAE_training/test_50_homology.fasta"

# Where to write the “all training” FASTA (excluding the 500)
all_training_fasta  = "../../data/SAE_training/all_training_data.fasta"

# Where to write the new cluster_members_training.tsv
cluster_members_training_tsv = "../../data/SAE_training/cluster_members_training.tsv"

# Number of singleton entries to sample (singleton: "clusters" with only one sequence in the cluster)
N_SAMPLE = 500

# For reproducibility
random.seed(42)

# ─── STEP 1: Read “cluster_sizes.tsv” & select singleton‐cluster entries ─────

df_sizes = pd.read_csv(
    cluster_sizes_tsv,
    sep="\t",
    header=None,
    names=["entry", "cluster_size"],
    dtype={"entry": str, "cluster_size": int}
)

# Keep only rows where cluster_size == 1
df_singletons = df_sizes[df_sizes["cluster_size"] == 1].copy()
# Creates a boolean mask (a Series of True/False) that is True for every row where cluster_size equals 1, and False elsewhere.
# Uses that boolean mask to filter df_sizes, keeping only the rows for which the mask is True.
# Makes an explicit copy of the filtered subset (rather than a view), so that any subsequent modifications to df_singletons won’t affect the original df_sizes.

# handling the case where there are less singletons than the requested sample size:
if df_singletons.shape[0] < N_SAMPLE:
    raise RuntimeError(
        f"Only {df_singletons.shape[0]} singleton entries available, but requested {N_SAMPLE}"
    )

# keeping the entry names of the singletons (all of them- we haven't sampled yet):
all_single_entries = df_singletons["entry"].tolist()

# ─── STEP 2: Randomly sample 500 entries from the singletons ────────────────

sampled_entries = random.sample(all_single_entries, N_SAMPLE)
sampled_set = set(sampled_entries)  # for fast membership testing we convert sample to python set
# quick way to speed up “is this entry in the sampled list?” lookups

print(f"> Sampled {N_SAMPLE} singleton‐cluster entries.")

# ─── STEP 3: Extract/write FASTA records from “pph.fa” ──────────────────────

# Ensure output directory exists
os.makedirs(os.path.dirname(test_fasta), exist_ok=True)

with open(pph_fasta, "r") as infile, \
     open(test_fasta, "w") as out_test, \
     open(all_training_fasta, "w") as out_all:

    for record in SeqIO.parse(infile, "fasta"):
        rec_id = record.id
        if rec_id in sampled_set:
            SeqIO.write(record, out_test, "fasta")
        else:
            SeqIO.write(record, out_all, "fasta")

print(f"> Wrote {N_SAMPLE} sampled sequences to {test_fasta}")
print(f"> Wrote the remaining sequences to {all_training_fasta}")

# ─── STEP 4: Write a new cluster_members.tsv without the 500 sampled entries ──────

with open(cluster_members_tsv, "r") as file_in, \
     open(cluster_members_training_tsv, "w") as file_out:
    for line in file_in:
        # strip newline, then split on tabs
        members = line.rstrip("\n").split("\t")
        # if this row is a singleton AND that one entry is in sampled_set, skip it
        if len(members) == 1 and members[0] in sampled_set:
            continue
        # otherwise write the full original line (including its newline)
        file_out.write(line)

print(f"> Wrote filtered cluster members to {cluster_members_training_tsv}")
