#!/usr/bin/env python3
import os
import pandas as pd
from Bio import SeqIO

# ─── Configuration ─────────────────────────────────────────────────────────
homology_groups = [40, 50, 60, 70, 80, 90]

pos_path   = "../../data/linear_regression_data/group_2_positive.csv"
neg_path   = "../../data/linear_regression_data/group_2_negative.csv"
output_csv = "../../data/linear_regression_data/homology_subset_counts.csv"

# ─── Load full label set ────────────────────────────────────────────────────
pos_df = pd.read_csv(pos_path).assign(label=1)
neg_df = pd.read_csv(neg_path).assign(label=0)
labels = pd.concat([pos_df, neg_df], ignore_index=True)

records = []
for hom in homology_groups:
    # fasta of representatives for this homology cutoff
    repr_fasta = f"../../data/classifier_training_data_mmseqs{hom}/final_rep_train_val_nr{hom}.fasta"
    
    # collect all Entry IDs in that fasta
    allowed = {rec.id for rec in SeqIO.parse(repr_fasta, "fasta")}
    
    # subset labels to those allowed entries
    subset = labels[labels["Entry"].isin(allowed)]
    
    # counts
    n_proteins = len(allowed)
    n_total    = len(subset)
    n_positive = int(subset["label"].sum())
    n_negative = n_total - n_positive

    records.append({
        "homology_pct": hom,
        "n_proteins":   n_proteins,
        "n_positive":   n_positive,
        "n_negative":   n_negative,
        "n_total":      n_total
    })

# ─── Save to CSV ───────────────────────────────────────────────────────────
df_counts = pd.DataFrame(records)
df_counts.to_csv(output_csv, index=False)
print(f"Wrote homology subset counts (including protein counts) to {output_csv}")
