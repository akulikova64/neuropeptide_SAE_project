#!/usr/bin/env python3
"""
split_training_val_data.py

1) From an 80%‐homology clustering, identify singleton clusters.
2) Randomly sample 10% of all peptides (3,199 entries) from those singletons to form a validation set.
3) Extract those 3,199 sequences from all_training_data.fasta → val_80_homology.fasta.
4) Save the remaining sequences to all_training_data_FINAL.fasta.
5) Rewrite cluster_members_training.tsv to exclude the 3,199 sampled entries:
   - For each cluster (one tab‐separated line), drop any sampled entry.
   - If a cluster becomes empty, omit that line entirely.
6) At the end, print:
   • Number of sequences in val_80_homology.fasta
   • Number of sequences in all_training_data_FINAL.fasta
   • Number of clusters remaining in cluster_members_training.tsv
   • Total number of entries in the new cluster_members_training.tsv

Usage:
    python split_training_val_data.py
"""

import os
import random
import pandas as pd
from Bio import SeqIO

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
# Paths (adjust if your working directory differs)
df_sizes = pd.read_csv(
    "../../data/SAE_training_data_mmseqs80/cluster_sizes.tsv",
    sep="\t",
    header=None,
    names=["entry", "cluster_size"],
    dtype={"entry": str, "cluster_size": int}
)
all_train_fasta   = "../../data/SAE_training/all_training_data.fasta"
cluster_members_tsv = "../../data/SAE_training/cluster_members_training.tsv"

# Output paths
val_fasta        = "../../data/SAE_training/val_80_homology.fasta"
train_final_fasta = "../../data/SAE_training/all_training_data_FINAL.fasta"
cluster_members_out = "../../data/SAE_training/cluster_members_training_FINAL.tsv"

# Seed for reproducibility
random.seed(42)

# ─── 1) Extract singletons ──────────────────────────
# Keep only rows where cluster_size == 1
singleton_df = df_sizes[df_sizes["cluster_size"] == 1].copy()

# Extract the entry names of those singletons
singleton_entries = singleton_df["entry"].tolist()

# ─── 2) Sample 10% of total peptides (i.e., 3,199) from the singletons ─────────
n_to_sample = 3199
if len(singleton_entries) < n_to_sample:
    raise RuntimeError(
        f"Not enough singletons ({len(singleton_entries)}) to sample {n_to_sample}."
    )

sampled_entries = random.sample(singleton_entries, n_to_sample)
sampled_set = set(sampled_entries)  # for fast membership testing

# ─── 3) Split all_training_data.fasta into val and training FINAL ─────────────
# Make sure output directory exists
os.makedirs(os.path.dirname(val_fasta), exist_ok=True)

# Counters
val_count = 0
train_final_count = 0

with open(val_fasta, "w") as val_handle, open(train_final_fasta, "w") as train_handle:
    for record in SeqIO.parse(all_train_fasta, "fasta"):
        seq_id = record.id
        if seq_id in sampled_set:
            SeqIO.write(record, val_handle, "fasta")
            val_count += 1
        else:
            SeqIO.write(record, train_handle, "fasta")
            train_final_count += 1

# ─── 4) Rewrite cluster_members_training.tsv excluding sampled entries ────────
# We’ll read each line, split on tabs, filter out sampled IDs, and write back any non‐empty cluster.
clusters_kept = 0
entries_kept = 0

with open(cluster_members_tsv, "r") as infile, open(cluster_members_out, "w") as outfile:
    for line in infile:
        line = line.rstrip("\n")
        if not line:
            continue
        members = line.split("\t")
        # Drop any sampled entries
        filtered = [m for m in members if m not in sampled_set]
        if filtered:
            outfile.write("\t".join(filtered) + "\n")
            clusters_kept += 1
            entries_kept += len(filtered)

# ─── 5) Print summary statistics ────────────────────────────────────────────────
print(f"> Sampled singleton entries for validation: {val_count}")
print(f"> Wrote {val_count} sequences to {val_fasta}")
print(f"> Wrote {train_final_count} sequences to {train_final_fasta}\n")

print(f"> Clusters remaining in filtered TSV: {clusters_kept}")
print(f"> Total entries across those clusters: {entries_kept}")
print(f"> Wrote filtered cluster members to {cluster_members_out}")
