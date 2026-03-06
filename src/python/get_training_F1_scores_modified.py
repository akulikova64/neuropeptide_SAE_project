import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import precision_score

# this is the main script that I use to calculate F1 scores (hybrid)

# ─── Paths ──────────────────────────────────────────────────────────────
# paths for neuropeptides:
#embeddings_dir     = "../../data/concept_groups/embeddings/group_2/training"
#pos_path_single    = "../../data/concept_groups/sequences/group_2/training/group_2_positive.csv"
#pos_path_domain    = "../../data/concept_groups/sequences/group_2/training/group_2_positive_domain.csv"
#neg_path           = "../../data/concept_groups/sequences/group_2/training/group_2_negative.csv"
#output_dir         = "../../data/concept_groups/F1_scores/group_2/binned_F1/training"

#paths for toxin dataset:
pos_path_single    = "/Volumes/T7 Shield/toxin_dataset/input_data/group_2_positive_toxins.csv"
pos_path_domain    = "/Volumes/T7 Shield/toxin_dataset/input_data/group_2_positive_domain_toxin.csv"
neg_path           = "/Volumes/T7 Shield/toxin_dataset/input_data/group_2_negative_toxins.csv"
embeddings_dir     = "/Volumes/T7 Shield/toxin_dataset/toxin_SAE_embeddings"
output_dir         = "/Volumes/T7 Shield/toxin_dataset/F1_scores_hybrid"
os.makedirs(output_dir, exist_ok=True)

print("Read in all data paths")

# ─── Read in labels ──────────────────────────────────────────────────────
pos_df          = pd.read_csv(pos_path_single).assign(label=1)
neg_df          = pd.read_csv(neg_path).assign(label=0)
labels_df       = pd.concat([pos_df, neg_df], ignore_index=True)

# Domain table 
pos_domain_df   = (pd.read_csv(pos_path_domain))

print("Processed all pos and neg data and labels")
print("Collecting embeddings")
# ─── Collect embeddings ───────────────────────────────────────
features_list, labels_list = [], []
entries_list, residues_list = [], []
missing = set()

for _, row in labels_df.iterrows():
    entry = row["Entry"]
    res_index = int(row["residue_number"]) - 1  # zero-based
    
    embeddings_path = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")
    if not os.path.isfile(embeddings_path):
        missing.add(entry)
        continue
    
    tensor = torch.load(embeddings_path, map_location="cpu")
    emb = tensor.numpy()
    
    # appending all entry information into parallel lists
    features_list.append(emb[res_index, :])
    labels_list.append(row["label"])
    entries_list.append(entry)
    residues_list.append(int(row["residue_number"]))

if missing:
    print("Missing embeddings for entries:", missing)

# to arrays
X = np.vstack(features_list)
y = np.array(labels_list)
entries_arr = np.array(entries_list)
residues_arr = np.array(residues_list)
num_features = X.shape[1] # 0- number of residue samples, 1 - number of features

# Number of domains for recall
n_domains = pos_domain_df["domain_id"].nunique() # counts how many domains there are

# ─── Compute hybrid F1 (residue precision, domain recall) ───────────────
thresholds = [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8]
print("Beginning to parse thresholds")

for threshold in thresholds:
    print("Processing threshold:", threshold)
    records = []
    for feature in range(num_features):
        # residue-level predictions
        predictions = (X[:, feature] > threshold).astype(int)
        # X is my 2D array of shape (n_samples, num_features)
        # taking all n_samples for a specific feature
        # result: boolean array (predictions) activated/unactivated  
        # for all activations and for all residules sampled
        precision = precision_score(y, predictions, zero_division=0) # input: labels, predictions (tru/false)
        
        # domain-level recall: OR over two residues per domain
        # get predictions for true-positive residues only
        mask_pos = (y == 1)  # turns labels list into a boolean array (1- positive data (TRUE), 0- negative data (FALSE)) 
        df_pos = pd.DataFrame({  # mini pos dataframe made to later merge with the domain df. 
            # only keeps entries for the positive dataset (mask_pos == TRUE)
            "Entry": entries_arr[mask_pos],
            "residue_number": residues_arr[mask_pos],
            "prediction": predictions[mask_pos]
        })
        # each row in df_pos gets its corresponding domain_id attached:
        merged = df_pos.merge(pos_domain_df, on=["Entry", "residue_number"], how="left")
        # taking the max value (1 or 0 - activated or not) for each domain: 
        domain_pred = merged.groupby("domain_id")["prediction"].max().values
        # getting domain-level recall while avoiding division by zero:
        recall = domain_pred.sum() / n_domains if n_domains else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        
        records.append({
            "feature": feature,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "F1_score": f1
        })
    
    df_out = pd.DataFrame(records)
    out_file = os.path.join(output_dir, f"threshold_{str(threshold).replace('.', '_')}.csv")
    df_out.to_csv(out_file, index=False)
    print(f"Saved hybrid F1 at threshold={threshold} to {out_file}")
print("Done!")