#!/usr/bin/env python3
import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# ─── Paths ──────────────────────────────────────────────────────────────────────
embeddings_dir = "../../data/concept_groups/embeddings/group_2/training"
pos_dom_path   = "../../data/concept_groups/sequences/group_2/training/group_2_positive_domain.csv"
neg_dom_path   = "../../data/concept_groups/sequences/group_2/training/group_2_negative_domain.csv"
output_dir     = "../../data/concept_groups/F1_scores/group_2/domain_based_F1/training"
os.makedirs(output_dir, exist_ok=True)

# ─── Read domain-level labels ───────────────────────────────────────────────────
pos_dom = pd.read_csv(pos_dom_path).assign(label=1)
neg_dom = pd.read_csv(neg_dom_path).assign(label=0)

dom_df = pd.concat([pos_dom, neg_dom], ignore_index=True)
# Columns: Entry, residue_number, domain_id, label

# ─── Collect embeddings & assemble matrices ────────────────────────────────────
features_list, labels_res = [], []
entries_list, residues_list = [], []
missing = set()

for _, row in dom_df.iterrows():
    entry = row["Entry"]
    idx0  = int(row["residue_number"]) - 1
    fpt   = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")

    if not os.path.isfile(fpt):
        missing.add(entry)
        continue

    emb = torch.load(fpt, map_location="cpu").numpy()
    features_list.append(emb[idx0, :])
    labels_res.append(row["label"])
    entries_list.append(entry)
    residues_list.append(row["residue_number"])

if missing:
    print("Missing embeddings for entries:", ", ".join(sorted(missing)))

X            = np.vstack(features_list)
entries_arr  = np.array(entries_list)
residues_arr = np.array(residues_list)
num_features = X.shape[1]

# ─── Domain-level metrics ───────────────────────────────────────────────────────
thresholds = [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8]

for thr in thresholds:
    print("Working on threshold:", thr)
    records = []

    for feat in range(num_features):
        preds_res = (X[:, feat] > thr).astype(int)

        df_pred = pd.DataFrame(
            {
                "Entry":          entries_arr,
                "residue_number": residues_arr,
                "pred":           preds_res,
            }
        )

        merged = df_pred.merge(dom_df,
                               on=["Entry", "residue_number"],
                               how="right")

        # ★ NEW LINE ↓ — treat “no prediction” as 0 instead of NaN
        merged["pred"] = merged["pred"].fillna(0).astype(int)

        domain_pred = merged.groupby("domain_id")["pred"].max().values
        domain_true = merged.groupby("domain_id")["label"].max().values

        prec_dom = precision_score(domain_true, domain_pred, zero_division=0)
        rec_dom  = recall_score   (domain_true, domain_pred, zero_division=0)
        f1_dom   = f1_score       (domain_true, domain_pred, zero_division=0)

        records.append(
            {
                "feature":   feat,
                "threshold": thr,
                "precision": prec_dom,
                "recall":    rec_dom,
                "F1_score":  f1_dom,
            }
        )
        print(f"  ✓ feature {feat} done")

    out_df = pd.DataFrame(records)
    fn = f"threshold_{str(thr).replace('.','_')}.csv"
    out_df.to_csv(os.path.join(output_dir, fn), index=False)
    print(f"Saved domain-level F1 at threshold={thr} → {fn}")

print("All done.")
