#!/usr/bin/env python3
import os
import pandas as pd
import torch
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# ─── Config ──────────────────────────────────────────────────────────────────
homology_training_list = [40, 50, 60, 70, 80, 90]

models_root = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/classifier_results/logistic_regression_models"
embeddings_dir = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/toxin_neuro_SAE_embeddings"
output_root = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/classifier_results/test_log_regression_csv_results"
os.makedirs(output_root, exist_ok=True)

# ─── Loop over homology groups ───────────────────────────────────────────────
for hom in homology_training_list:
    print(f"\n=== HOMOLOGY: {hom}% ===")
    # paths
    feature_rank_path = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"input_data/training_data/homology_group{hom}/"
        f"feature_ranking_hybrid_hom{hom}.csv"
    )
    pos_path = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"input_data/test_data/homology_group{hom}/"
        f"group_2_positive_toxin_neuro_test.csv"
    )
    neg_path = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"input_data/test_data/homology_group{hom}/"
        f"group_2_negative_toxin_neuro_test.csv"
    )
    output_csv = os.path.join(
        output_root,
        f"linear_regression_test_results_{hom}_hom_dup.csv"
    )
    model_dir = os.path.join(models_root, f"hom{hom}")

    # ─── Load feature ranking ────────────────────────────────────────────────
    feat_df = pd.read_csv(feature_rank_path).sort_values("rank")
    feature_indices = feat_df["feature"].tolist()

    # ─── Load test labels ──────────────────────────────────────────────────
    pos_df = pd.read_csv(pos_path).assign(label=1)
    neg_df = pd.read_csv(neg_path).assign(label=0)
    test_labels = pd.concat([pos_df, neg_df], ignore_index=True)

    # ─── Collect test embeddings ───────────────────────────────────────────
    X_test_list = []
    y_test_list = []
    entries = []
    for _, row in test_labels.iterrows():
        entry = row["Entry"]
        res_i = int(row["residue_number"]) - 1
        pt_file = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")
        if not os.path.isfile(pt_file):
            continue
        emb = torch.load(pt_file, map_location="cpu").numpy()
        X_test_list.append(emb[res_i, :])
        y_test_list.append(row["label"])
        entries.append(entry)
    X_test = np.vstack(X_test_list)
    y_test = np.array(y_test_list)

    # ─── Evaluate each saved model ─────────────────────────────────────────
    # Only evaluate the same feature counts we trained on: 5,10,…,600
    feature_steps = range(5, 605, 5)
    results = []
    for num_feat in feature_steps:
        model_path = os.path.join(model_dir, f"logreg_hom{hom}_{num_feat}.joblib")
        if not os.path.isfile(model_path):
            print(f"  ⚠️  Missing model for {num_feat} features: {model_path}")
            continue

        # load
        clf = joblib.load(model_path)
        # select top features
        top_idxs = feature_indices[:num_feat]
        X_sub = X_test[:, top_idxs]

        # predict & score
        y_pred = clf.predict(X_sub)
        acc    = accuracy_score(y_test, y_pred) * 100

        results.append({
            "num_features": num_feat,
            "test_accuracy": f"{acc:.2f}"
        })
        print(f"  • {num_feat} features → test acc {acc:.2f}%")

    # ─── Save CSV ─────────────────────────────────────────────────────────
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"✔ Wrote test results to {output_csv}")
