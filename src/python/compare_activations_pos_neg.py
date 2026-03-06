
# This script: finds features that are active across more than 0% of positions in the positive dataset
# consistently across all sequences, 50% does not work; I tried :-(
# it then looks at the corresponding features in the negative dataset and calculates
# what percent of positions are non-zero on average across the negative sequences.
# the goal is to identify the features where the positive dataset consistently has
# non-zero activations while the negative sequences do not. What are those features?

import os
import torch
import pandas as pd
import numpy as np

# ─── Configuration ─────────────────────────────────────────────────────────────
positive_dir = "../../data/concept_groups/embeddings/group_1/training/positive"
negative_dir = "../../data/concept_groups/embeddings/group_1/training/negative"
output_csv   = "../../data/concept_groups/stats/group_1/consistent_positive_features.csv"

# ─── Gather file lists ──────────────────────────────────────────────────────────
pos_files = [f for f in os.listdir(positive_dir) if f.endswith(".pt")]
neg_files = [f for f in os.listdir(negative_dir) if f.endswith(".pt")]
num_pos   = len(pos_files)

# ─── Determine feature dimension from first positive file ──────────────────────
first_tensor = torch.load(os.path.join(positive_dir, pos_files[0]))
_, feature_dim = first_tensor.shape

# ─── 1) Identify features >0% non-zero in every positive sequence ─────────────
feature_meets_criteria = np.zeros(feature_dim, dtype=int)

for file_name in pos_files:
    data = torch.load(os.path.join(positive_dir, file_name))  # (seq_len, feature_dim)
    seq_len = data.size(0)
    if seq_len == 0:
        continue  # skip empty if any
    
    nonzero_counts   = torch.count_nonzero(data, dim=0).numpy()
    percent_nonzero  = nonzero_counts / seq_len
    # change to `>= 0.5` if you truly want the 50% threshold:
    feature_meets_criteria += (percent_nonzero > 0).astype(int)

# proper count of passing features
passing = np.where(feature_meets_criteria == num_pos)[0]
print(f"Features passing in all {num_pos} positives:", passing.size)

if passing.size == 0:
    raise RuntimeError("No features meet the >0% non‑zero criterion in all positive sequences")

valid_features = passing

# ─── 2) & 3) For each valid feature, per sequence, record stats ───────────────
rows = []

def process_folder(folder, files, dataset_label):
    for file_name in files:
        seq_id = file_name.split("_", 1)[0]
        data   = torch.load(os.path.join(folder, file_name))
        seq_len= data.size(0)
        if seq_len == 0:
            continue

        for feat in valid_features:
            nonzeros   = torch.count_nonzero(data[:, feat]).item()
            pct_nonzero= nonzeros / seq_len
            max_act    = float(data[:, feat].max().item())
            rows.append({
                "sequence_ID":     seq_id,
                "feature":         int(feat),
                "percent_non_zero":pct_nonzero,
                "max_activation":  max_act,
                "dataset":         dataset_label
            })

# Positive then Negative
process_folder(positive_dir, pos_files, "positive")
process_folder(negative_dir, neg_files, "negative")

# ─── 4) Write out CSV ───────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)
print(f"✅ Written {len(df)} rows to {output_csv}")