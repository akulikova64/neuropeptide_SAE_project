import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import precision_score

# ─── Paths ──────────────────────────────────────────────────────────────
embeddings_dir     = "../../data/concept_groups/embeddings/group_2/training"
pos_path_single    = "../../data/concept_groups/sequences/group_2/training/group_2_positive.csv"
pos_path_domain    = "../../data/concept_groups/sequences/group_2/training/group_2_positive_domain.csv"
neg_path           = "../../data/concept_groups/sequences/group_2/training/group_2_negative.csv"
bins_path          = "../../data/concept_groups/sequences/group_2/training/alt_peptide_bins.csv"

base_output_dir    = "../../data/concept_groups/F1_scores/group_2/binned_F1/training"

# read the bin assignments
bins_df = pd.read_csv(bins_path)   # columns: Entry, bin

# read the labels
pos_df    = pd.read_csv(pos_path_single).assign(label=1)
neg_df    = pd.read_csv(neg_path).assign(label=0)
labels_all = pd.concat([pos_df, neg_df], ignore_index=True)

# read the domain table once
pos_domain_all = pd.read_csv(pos_path_domain)

# hybrid‐F1 parameters
thresholds = [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8]

for bin_name in bins_df['bin'].unique():
    # make per‐bin output directory
    out_dir = os.path.join(base_output_dir, bin_name.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    # filter to this bin's Entries
    entries_in_bin = bins_df.loc[bins_df['bin']==bin_name, 'Entry']
    labels_df      = labels_all[labels_all['Entry'].isin(entries_in_bin)].reset_index(drop=True)
    pos_domain_df  = pos_domain_all[pos_domain_all['Entry'].isin(entries_in_bin)]

    # count domains in this bin for recall
    n_domains = pos_domain_df["domain_id"].nunique()

    # collect embeddings & labels
    features_list, labels_list = [], []
    entries_list, residues_list = [], []
    missing = set()

    for _, row in labels_df.iterrows():
        entry        = row["Entry"]
        res_index    = int(row["residue_number"]) - 1  
        embeddings_p = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")

        if not os.path.isfile(embeddings_p):
            missing.add(entry)
            continue

        emb_tensor = torch.load(embeddings_p, map_location="cpu")
        emb        = emb_tensor.numpy()

        features_list.append(emb[res_index, :])
        labels_list.append(row["label"])
        entries_list.append(entry)
        residues_list.append(int(row["residue_number"]))

    if missing:
        print(f"[{bin_name}] Missing embeddings for:", missing)

    X            = np.vstack(features_list)
    y            = np.array(labels_list)
    entries_arr  = np.array(entries_list)
    residues_arr = np.array(residues_list)
    num_features = X.shape[1]

    # compute hybrid F1 per threshold & feature
    for threshold in thresholds:
        records = []
        for feature in range(num_features):
            preds    = (X[:, feature] > threshold).astype(int)
            precision = precision_score(y, preds, zero_division=0)

            # domain-level recall for positives
            mask_pos = (y == 1)
            df_pos   = pd.DataFrame({
                "Entry": entries_arr[mask_pos],
                "residue_number": residues_arr[mask_pos],
                "prediction": preds[mask_pos]
            })
            merged     = df_pos.merge(pos_domain_df, on=["Entry", "residue_number"], how="left")
            domain_pred= merged.groupby("domain_id")["prediction"].max().values
            recall     = domain_pred.sum() / n_domains if n_domains else 0.0

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            records.append({
                "feature": feature,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "F1_score": f1
            })

        df_out   = pd.DataFrame(records)
        fname    = f"threshold_{str(threshold).replace('.', '_')}.csv"
        out_file = os.path.join(out_dir, fname)
        df_out.to_csv(out_file, index=False)
        print(f"[{bin_name}] Saved hybrid F1 at threshold={threshold} → {out_file}")

print("All bins done!")
