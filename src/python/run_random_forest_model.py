#!/usr/bin/env python3
import os
import pandas as pd
import torch
import numpy as np
import joblib
import gc

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score

homology_training_list = [40, 50, 60, 70, 80, 90]  # You can add more: 50, 60, 70, 80, 90

# Where to dump all final models
models_root = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/classifier_results/random_forest_models"
os.makedirs(models_root, exist_ok=True)

for homology_training in homology_training_list:
    print(f"\n=== HOMOLOGY: {homology_training}% ===")

    # ─── Paths ───────────────────────────────────────────────────────────────
    embeddings_dir = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/toxin_neuro_SAE_embeddings"
    feature_rank_path = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"input_data/training_data/homology_group{homology_training}/"
        f"feature_ranking_hybrid_hom{homology_training}.csv"
    )
    pos_path = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"input_data/training_data/homology_group{homology_training}/group_2_positive_toxin_neuro_train.csv"
    )
    neg_path = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"input_data/training_data/homology_group{homology_training}/group_2_negative_toxin_neuro_train.csv"
    )
    output_csv = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"classifier_results/random_forest_csv_results/"
        f"random_forest_results_{homology_training}_hom_dup.csv"
    )

    models_dir = os.path.join(models_root, f"hom{homology_training}")
    os.makedirs(models_dir, exist_ok=True)

    print("Loading feature ranking…")
    feat_df = pd.read_csv(feature_rank_path).sort_values("rank")
    feature_indices = feat_df["feature"].tolist()

    print("Loading positive/negative labels…")
    pos_df = pd.read_csv(pos_path).assign(label=1)
    neg_df = pd.read_csv(neg_path).assign(label=0)
    labels_df = pd.concat([pos_df, neg_df], ignore_index=True)

    print("Collecting embeddings…")
    features_list, labels_list, groups_list = [], [], []
    for _, row in labels_df.iterrows():
        entry = row["Entry"]
        res_index = int(row["residue_number"]) - 1  # zero-based
        emb_file = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")
        if not os.path.isfile(emb_file):
            continue
        tensor = torch.load(emb_file, map_location="cpu")
        arr = tensor.numpy()
        features_list.append(arr[res_index, :])
        labels_list.append(row["label"])
        groups_list.append(entry)

    X_all = np.vstack(features_list)
    y_all = np.array(labels_list)
    groups = np.array(groups_list)

    # One-split GroupShuffleSplit for efficiency
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

    print("Running CV across feature counts…")
    records = []
    for num_feat in range(5, 305, 10):
        print(f" • num_features = {num_feat}")
        top_idxs = feature_indices[:num_feat]
        X = X_all[:, top_idxs]

        # 1 GroupShuffleSplit split
        for train_idx, test_idx in gss.split(X, y_all, groups=groups):
            clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=None,
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X[train_idx], y_all[train_idx])
            y_pred = clf.predict(X[test_idx])
            acc = accuracy_score(y_all[test_idx], y_pred)
            bacc = balanced_accuracy_score(y_all[test_idx], y_pred)

        records.append({
            "num_features": num_feat,
            "validation": f"{acc * 100:.2f}",
            "balanced_accuracy": f"{bacc * 100:.2f}"
        })

        # Train final model on *all* data and use OOB estimate
        final_clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=None,
            oob_score=True,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        final_clf.fit(X, y_all)
        print(f"   ↪ OOB Accuracy: {final_clf.oob_score_ * 100:.2f}%")

        model_path = os.path.join(models_dir, f"randomforest_hom{homology_training}_{num_feat}.joblib")
        joblib.dump(final_clf, model_path)

        # Memory cleanup
        del clf, final_clf, X
        gc.collect()

    # Save results
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)
    print(f"✔ CV results → {output_csv}")
    print(f"✔ Saved models ✓ in {models_dir}")
