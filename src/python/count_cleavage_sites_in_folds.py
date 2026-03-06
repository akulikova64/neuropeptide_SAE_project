#!/usr/bin/env python3
import os
import pandas as pd

# ─── Paths ───────────────────────────────────────────────────────────────
pos_path   = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/group_2_positive_toxin_neuro.csv"
neg_path   = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/group_2_negative_toxin_neuro.csv"
folds_dir  = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/folds_40"

# ─── Load the positive/negative site tables ──────────────────────────────
pos_df = pd.read_csv(pos_path, usecols=["Entry", "residue_number"])
neg_df = pd.read_csv(neg_path, usecols=["Entry", "residue_number"])

# ─── Prepare to accumulate counts ────────────────────────────────────────
print("Fold\tPositive_Sites\tNegative_Sites")
grand_pos = 0
grand_neg = 0

# ─── Loop over fold_0.txt … fold_9.txt ──────────────────────────────────
for i in range(10):
    fold_file = os.path.join(folds_dir, f"fold_{i}.txt")
    label = "Test" if i == 9 else str(i)
    if not os.path.isfile(fold_file):
        print(f"{label}\tfile not found\tfile not found")
        continue

    # read this fold's entries (peptide ACs)
    with open(fold_file, encoding="utf-8", errors="ignore") as fh:
        entries = fh.read().strip().split()

    # count how many rows in each CSV whose Entry is in this fold
    pos_count = pos_df["Entry"].isin(entries).sum()
    neg_count = neg_df["Entry"].isin(entries).sum()

    print(f"{label}\t{pos_count}\t{neg_count}")

    # accumulate totals only for folds 0–8 (exclude test fold 9)
    if i < 9:
        grand_pos += pos_count
        grand_neg += neg_count

# ─── Print grand totals for folds 0–8 only ───────────────────────────────
print(f"TOTAL\t{grand_pos}\t{grand_neg}")
