import os
import torch
import numpy as np
import pandas as pd

# ─── Configuration ─────────────────────────────────────────────────────────────
positive_dir = "../../data/concept_groups/embeddings/group_1/training/positive"
negative_dir = "../../data/concept_groups/embeddings/group_1/training/negative"
OUTPUT_CSV   = "../../data/concept_groups/stats/group_1/top50_feature_stats.csv"
TOP_N        = 50

# ─── 1) Gather all positive and negative files ─────────────────────────────────
positive_files = sorted(f for f in os.listdir(positive_dir) if f.endswith(".pt"))
negative_files = sorted(f for f in os.listdir(negative_dir) if f.endswith(".pt"))

num_positive_sequences = len(positive_files)
num_negative_sequences = len(negative_files)

# ─── 2) Determine feature dimension from the first positive file ───────────────
sample_tensor = torch.load(os.path.join(positive_dir, positive_files[0]))
_, num_features = sample_tensor.shape

# ─── 3) Count in how many positive sequences each feature appears at least once ─
feature_presence_count_pos = np.zeros(num_features, dtype=int)
feature_max_activation_pos = np.full(num_features, -np.inf, dtype=float)

for filename in positive_files:
    path = os.path.join(positive_dir, filename)
    tensor = torch.load(path)  # shape = (sequence_length, num_features)

    # How many positions are non-zero for each feature?
    nonzero_position_counts = torch.count_nonzero(tensor, dim=0).numpy() # dim 0 is sequence_length
    # Mark “presence” if at least one non-zero activation exists
    feature_presence_count_pos += (nonzero_position_counts > 0).astype(int)
    # Track the global max activation per feature
    feature_max_activation_pos = np.maximum(
        feature_max_activation_pos,
        tensor.max(dim=0).values.numpy()
    )

# ─── 4) Compute the fraction of positive sequences in which each feature appears ─
feature_presence_fraction_pos = feature_presence_count_pos / num_positive_sequences

# ─── 5) Select the top N features by that fraction ──────────────────────────────
top_features = np.argsort(-feature_presence_fraction_pos)[:TOP_N]

# For quick lookup:
fraction_pos_map = {f: feature_presence_fraction_pos[f] for f in top_features}
max_act_pos_map  = {f: feature_max_activation_pos[f]      for f in top_features}

# ─── 6) Repeat presence counting for negatives (only for top features) ────────
feature_presence_count_neg = np.zeros(num_features, dtype=int)
feature_max_activation_neg = np.full(num_features, -np.inf, dtype=float)

for filename in negative_files:
    path = os.path.join(negative_dir, filename)
    tensor = torch.load(path)

    nonzero_position_counts = torch.count_nonzero(tensor, dim=0).numpy()
    feature_presence_count_neg += (nonzero_position_counts > 0).astype(int)
    feature_max_activation_neg = np.maximum(
        feature_max_activation_neg,
        tensor.max(dim=0).values.numpy()
    )

fraction_neg_map = {
    f: feature_presence_count_neg[f] / num_negative_sequences
    for f in top_features
}
max_act_neg_map = {f: feature_max_activation_neg[f] for f in top_features}

# ─── 7) Build final rows per sequence, per feature ─────────────────────────────
rows = []

def collect_sequence_stats(folder, file_list, dataset_label, fraction_map, max_map):
    """Append one row per (sequence, feature) for the given dataset."""
    for filename in file_list:
        sequence_id = filename.split("_", 1)[0]
        tensor      = torch.load(os.path.join(folder, filename))
        seq_len     = tensor.size(0)
        nonzero_counts = torch.count_nonzero(tensor, dim=0).numpy()

        for feature_idx in top_features:
            percent_non_zero = nonzero_counts[feature_idx] / seq_len
            rows.append({
                "sequence_ID":         sequence_id,
                "feature":             int(feature_idx),
                "percent_of_dataset":  fraction_map[feature_idx],
                "percent_non_zero":    percent_non_zero,
                "max_activation":      float(max_map[feature_idx]),
                "dataset":             dataset_label
            })

# Positive then negative
collect_sequence_stats(positive_dir, positive_files, "positive", fraction_pos_map, max_act_pos_map)
collect_sequence_stats(negative_dir, negative_files, "negative", fraction_neg_map, max_act_neg_map)

# ─── 8) Save as CSV ────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved {len(df)} rows to {OUTPUT_CSV}")