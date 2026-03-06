import os
import glob
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# this script calculates unmodified F1 scores (strict- per residue) for concept group 2- cleaveage sites


# Paths to data
embeddings_dir = "../../data/concept_groups/embeddings/group_2/training"
pos_path = "../../data/concept_groups/sequences/group_2/training/group_2_positive_single_flank.csv"
neg_path = "../../data/concept_groups/sequences/group_2/training/group_2_negative_single_flank.csv"
output_dir = "../../data/concept_groups/F1_scores/group_2/single_flank_F1/training"

# Read positive and negative labels
pos_df = pd.read_csv(pos_path)  # has columns: Entry, residue_number
pos_df["label"] = 1
neg_df = pd.read_csv(neg_path)
neg_df["label"] = 0

# Combine into a single DataFrame
labels_df = pd.concat([pos_df, neg_df], ignore_index=True)

features_list = []
labels_list = []
missing_entries = set()

for _, row in labels_df.iterrows():
    entry = row["Entry"]
    # Convert 1-based residue_number to 0-based index!!
    res_index = int(row["residue_number"]) - 1

    # Build path to "<Entry>_original_SAE.pt"
    emb_file = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")
    if not os.path.isfile(emb_file):
        missing_entries.add(entry)
        continue

    # Load the tensor and move to CPU
    tensor = torch.load(emb_file, map_location="cpu")  # shape: (seq_len, num_features)
    # Convert to NumPy array
    emb = tensor.numpy()

    # Append feature vector and label
    features_list.append(emb[res_index, :]) # emb[specific_row (position), take_all_cols (features)]
    labels_list.append(row["label"]) # this list will be parallel to the features_list

if missing_entries:
    print(f"Warning: missing embeddings for entries: {missing_entries}")

features_array = np.vstack(features_list)  # shape: (n_samples (positive and negative), num_features (~10200))
labels_array = np.array(labels_list)        # shape: (n_samples,)

num_features = features_array.shape[1]

# 3Compute modified F1 for each feature at each threshold
thresholds = [0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8]
os.makedirs(output_dir, exist_ok=True)

for thresh in thresholds:
    results = []
    for feat_index in range(num_features):
        print("Threshold:", thresh, "Feature:", feat_index)
        activations = features_array[:, feat_index]
        # Binarize activations
        preds = (activations > thresh).astype(int)
        
        # Compute precision, recall, F1
        prec = precision_score(labels_array, preds, zero_division=0)
        rec  = recall_score(labels_array, preds, zero_division=0)
        f1   = f1_score(labels_array, preds, zero_division=0)
        
        results.append({
            "feature": feat_index,
            "threshold": thresh,
            "precision": prec,
            "recall": rec,
            "F1_score": f1
        })
        print(f"  ✓ feature {feat_index} done, threshold {thresh}")
    
    # Save results for this threshold
    df_out = pd.DataFrame(results, columns=["feature", "threshold", "precision", "recall", "F1_score"])
    out_file = os.path.join(output_dir, f"threshold_{str(thresh).replace('.', '_')}.csv")
    df_out.to_csv(out_file, index=False)
    print(f"> Saved F1 scores at threshold={thresh} to {out_file}")
