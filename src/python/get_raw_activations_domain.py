import os
import pandas as pd
import torch

# ─── Paths ──────────────────────────────────────────────────────────────
embeddings_dir = "../../data/concept_groups/embeddings/group_2/training"
pos_path       = "../../data/concept_groups/sequences/group_2/training/group_2_positive.csv"
neg_path       = "../../data/concept_groups/sequences/group_2/training/group_2_negative.csv"
output_csv     = "../../data/concept_groups/stats/group_2/raw_activations.csv"

# ─── Read labels ─────────────────────────────────────────────────────────
pos_df = pd.read_csv(pos_path)
pos_df["dataset"] = "positive"
neg_df = pd.read_csv(neg_path)
neg_df["dataset"] = "negative"

# Combine into single DataFrame
labels_df = pd.concat([pos_df, neg_df], ignore_index=True)

# ─── Collect raw activations ─────────────────────────────────────────────
records = []
missing = set()

# collecting stats
num_zero_activations_pos = 0
total_activations_pos = 0
num_zero_activations_neg = 0
total_activations_neg = 0

for _, row in labels_df.iterrows():
    entry      = row["Entry"]
    position   = int(row["residue_number"]) - 1  # zero‐based index into embedding!!
    dataset    = row["dataset"]
    
    # Build .pt filename and path
    fname = f"{entry}_original_SAE.pt"
    fpath = os.path.join(embeddings_dir, fname)
    if not os.path.isfile(fpath):
        missing.add(entry)
        continue
    
    # Load tensor and move to CPU
    tensor = torch.load(fpath, map_location="cpu")  # shape: (seq_len, num_features)
    
    # For each feature at this position, record activation
    seq_len, num_features = tensor.shape
    for feat_idx in range(num_features):
        activation = tensor[position, feat_idx].item()
        # 1) Count datapoints:
        if dataset == "positive":
            total_activations_pos += 1
        else:
            total_activations_neg += 1

        # 2) Then count zero vs non-zero
        if activation == 0:
            if dataset == "positive":
                num_zero_activations_pos += 1
            else:
                num_zero_activations_neg += 1
        else:
            # Append your record only for non-zeros (if that’s your goal)
            records.append({
                "Entry":      entry,
                "position":   position + 1,
                "feature":    feat_idx,
                "activation": activation,
                "dataset":    dataset
            })
            
        
if missing:
    print(f"Warning: missing embeddings for entries: {missing}")

# ─── Save to CSV ─────────────────────────────────────────────────────────
df_out = pd.DataFrame(records,
    columns=["Entry", "position", "feature", "activation", "dataset"]
)
#os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_out.to_csv(output_csv, index=False)

print(f"✅ Saved raw activations to {output_csv}")
print("Num zero activations in pos datset:", num_zero_activations_pos, "total:", total_activations_pos)
print("percent non-zero", ((total_activations_pos - num_zero_activations_pos)/total_activations_pos)*100)
print("\n")

print("Num zero activations in neg datset:", num_zero_activations_neg, "total:", total_activations_neg)
print("percent non-zero", ((total_activations_neg - num_zero_activations_neg)/total_activations_neg)*100)

print("\n")
total_pos = len(pos_df) * num_features
total_neg = len(neg_df) * num_features
print("Totals:", total_pos, total_neg)
print("Positive labels:", len(pos_df))
print("Negative labels:", len(neg_df))