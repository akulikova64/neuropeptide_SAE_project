import os
import torch
import numpy as np
import pandas as pd

# ─── Configuration ─────────────────────────────────────────────────────────────
positive_dir    = "../../data/concept_groups/embeddings/group_1/training/positive"
negative_dir    = "../../data/concept_groups/embeddings/group_1/training/negative"
output_csv      = "../../data/concept_groups/stats/group_1/features_unique_to_positive_only.csv"
presence_thresh = 0  # minimum fraction in positive dataset

# ─── Gather file lists ──────────────────────────────────────────────────────────
pos_files = [f for f in os.listdir(positive_dir) if f.endswith(".pt")]
neg_files = [f for f in os.listdir(negative_dir) if f.endswith(".pt")]
num_pos   = len(pos_files)
num_neg   = len(neg_files)
if num_pos == 0:
    raise RuntimeError("No positive embeddings found")

# ─── Determine feature dimension ────────────────────────────────────────────────
first = torch.load(os.path.join(positive_dir, pos_files[0]))
_, num_features = first.shape

# ─── Initialize presence counters ───────────────────────────────────────────────
presence_pos = np.zeros(num_features, dtype=int)
presence_neg = np.zeros(num_features, dtype=int)

# ─── Count presence in positive dataset ────────────────────────────────────────
for fn in pos_files:
    data = torch.load(os.path.join(positive_dir, fn))  # shape (seq_len, features)
    # flag features with any non-zero
    presence_pos += (data.abs().sum(dim=0).numpy() > 0).astype(int)

# ─── Count presence in negative dataset ────────────────────────────────────────
for fn in neg_files:
    data = torch.load(os.path.join(negative_dir, fn))
    presence_neg += (data.abs().sum(dim=0).numpy() > 0).astype(int)

# ─── Compute fractions ─────────────────────────────────────────────────────────
frac_pos = presence_pos / num_pos
frac_neg = presence_neg / num_neg if num_neg else np.zeros_like(frac_pos)

# ─── Select features present in >threshold in positive and 0 in negative ────
mask = (frac_pos > presence_thresh) & (frac_neg == 0)
selected_feats = np.where(mask)[0]

# ─── Build output DataFrame ───────────────────────────────────────────────────
df = pd.DataFrame({
    "feature": selected_feats,
    "presence_frac_positive": frac_pos[selected_feats],
    "presence_frac_negative": frac_neg[selected_feats]
})
df.sort_values("presence_frac_positive", ascending=False, inplace=True)

# ─── Save to CSV ───────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)
df