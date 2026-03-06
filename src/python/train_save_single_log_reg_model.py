#!/usr/bin/env python3
import os
import pandas as pd
import torch
import numpy as np
from Bio import SeqIO
from sklearn.linear_model import LogisticRegression
import joblib

num_features = 600
homology = 90

# ─── Configuration ─────────────────────────────────────────────────────────
embeddings_dir    = "../../data/concept_groups/embeddings/group_2/training"
feature_rank_path = "../../data/linear_regression_data/feature_ranking_hybrid.csv"
pos_path          = "../../data/linear_regression_data/group_2_positive.csv"
neg_path          = "../../data/linear_regression_data/group_2_negative.csv"
repr_fasta        = f"../../data/classifier_training_data_mmseqs{homology}/final_rep_train_val_nr{homology}.fasta"

# where to save the final model
model_dir   = "../../data/linear_regression_data/logistic_models"
os.makedirs(model_dir, exist_ok=True)
model_path  = os.path.join(model_dir, f"logreg_hom{homology}_{num_features}.pkl")

# ─── Load feature ranking and select top 600 features ──────────────────────
feat_df = pd.read_csv(feature_rank_path).sort_values("rank")
feature_indices = feat_df["feature"].tolist()
top600 = feature_indices[:num_features]

# ─── Load labels ────────────────────────────────────────────────────────────
pos_df    = pd.read_csv(pos_path).assign(label=1)
neg_df    = pd.read_csv(neg_path).assign(label=0)
labels_df = pd.concat([pos_df, neg_df], ignore_index=True)

# ─── Filter to only entries in the 90%‐homology representative set ─────────
allowed = {rec.id for rec in SeqIO.parse(repr_fasta, "fasta")}
labels_df = labels_df[labels_df["Entry"].isin(allowed)].reset_index(drop=True)

# ─── Build feature matrix X_all and label vector y_all ─────────────────────
features_list = []
labels_list   = []

for _, row in labels_df.iterrows():
    entry     = row["Entry"]
    res_index = int(row["residue_number"]) - 1
    emb_file  = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")
    if not os.path.isfile(emb_file):
        continue
    emb_tensor = torch.load(emb_file, map_location="cpu")
    arr        = emb_tensor.numpy()
    features_list.append(arr[res_index, top600])
    labels_list.append(row["label"])

X_all = np.vstack(features_list)
y_all = np.array(labels_list)

# ─── Train logistic regression on all data ─────────────────────────────────
clf = LogisticRegression(max_iter=1000)
clf.fit(X_all, y_all)

# ─── Save trained model ────────────────────────────────────────────────────
joblib.dump(clf, model_path)
print(f"Saved {homology}%‐homology, {num_features}‐feature logistic model to:\n  {model_path}")
