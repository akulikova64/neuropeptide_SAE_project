#!/usr/bin/env python3
import os
import pandas as pd
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
import joblib

homology_training_list = [40, 50, 60, 70, 80, 90]

# ─── Paths for SVM outputs ───────────────────────────────────────────────────
models_root = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/classifier_results/SVM_models"
csv_root    = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/classifier_results/svm_csv_results"
os.makedirs(models_root, exist_ok=True)
os.makedirs(csv_root, exist_ok=True)

for homology_training in homology_training_list:
    print(f"\n=== HOMOLOGY: {homology_training}% ===")
    # ─── Paths ───────────────────────────────────────────────────────────────
    embeddings_dir    = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/toxin_neuro_SAE_embeddings"
    feature_rank_path = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"input_data/training_data/homology_group{homology_training}/"
        f"feature_ranking_hybrid_hom{homology_training}.csv"
    )
    pos_path = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"input_data/training_data/homology_group{homology_training}/"
        f"group_2_positive_toxin_neuro_train.csv"
    )
    neg_path = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"input_data/training_data/homology_group{homology_training}/"
        f"group_2_negative_toxin_neuro_train.csv"
    )
    output_csv = os.path.join(
        csv_root,
        f"svm_poly_results_{homology_training}_hom_dup.csv"
    )

    # ─── Make per-homology model directory ──────────────────────────────────
    models_dir = os.path.join(models_root, f"hom{homology_training}")
    os.makedirs(models_dir, exist_ok=True)

    print("Loading feature ranking…")
    feat_df = pd.read_csv(feature_rank_path).sort_values("rank")
    feature_indices = feat_df["feature"].tolist()

    print("Loading positive/negative labels…")
    pos_df    = pd.read_csv(pos_path).assign(label=1)
    neg_df    = pd.read_csv(neg_path).assign(label=0)
    labels_df = pd.concat([pos_df, neg_df], ignore_index=True)

    print("Collecting embeddings…")
    features_list, labels_list, groups_list = [], [], []
    for _, row in labels_df.iterrows():
        entry     = row["Entry"]
        res_index = int(row["residue_number"]) - 1
        emb_file  = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")
        if not os.path.isfile(emb_file):
            continue
        emb = torch.load(emb_file, map_location="cpu").numpy()
        features_list.append(emb[res_index, :])
        labels_list.append(row["label"])
        groups_list.append(entry)

    X_all  = np.vstack(features_list)
    y_all  = np.array(labels_list)
    groups = np.array(groups_list)

    # ─── Group‐aware Monte Carlo CV ────────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=200, test_size=0.1, random_state=42)

    print("Running Monte Carlo CV with polynomial‐kernel SVM…")
    records = []
    for num_feat in range(5, 605, 5):
        print(f" • num_features = {num_feat}")
        top_idxs = feature_indices[:num_feat]
        X        = X_all[:, top_idxs]

        accs = []
        for train_idx, test_idx in gss.split(X, y_all, groups=groups):
            # polynomial kernel of degree 3
            clf = SVC(kernel="poly", degree=3, gamma="scale")
            clf.fit(X[train_idx], y_all[train_idx])
            y_pred = clf.predict(X[test_idx])
            accs.append(accuracy_score(y_all[test_idx], y_pred))

        avg_acc = np.mean(accs) * 100
        records.append({
            "num_features": num_feat,
            "validation":   f"{avg_acc:.2f}"
        })

        # ─── train final poly‐SVM on full data and save
        final_clf = SVC(kernel="poly", degree=3, gamma="scale")
        final_clf.fit(X, y_all)
        model_path = os.path.join(
            models_dir,
            f"svm_poly_hom{homology_training}_{num_feat}.joblib"
        )
        joblib.dump(final_clf, model_path)

    # ─── Save CV results CSV ───────────────────────────────────────────────
    pd.DataFrame(records).to_csv(output_csv, index=False)
    print(f"✔ SVM (poly) CV results → {output_csv}")
    print(f"✔ Saved SVM (poly) models in {models_dir}")
