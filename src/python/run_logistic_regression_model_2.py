#!/usr/bin/env python3
import os
import re
import pandas as pd
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# ─── Configuration ─────────────────────────────────────────────────────────
test_folds       = list(range(10))
percent_identity = 40
run = 4

for test_fold in test_folds:
    # where to dump all of your final models
    models_root = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis/classifier_results/run_{run}/logistic_regression_models/"
        f"graphpart_{percent_identity}/"
        f"excluded_fold_{test_fold}/"
    )
    os.makedirs(models_root, exist_ok=True)

    # ─── Paths ───────────────────────────────────────────────────────────────
    #embeddings_dir    = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/toxin_neuro_SAE_embeddings"
    embeddings_dir    = f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/toxin_neuro_finetuned_SAE_embeddings/run_{run}/"
    feature_rank_path = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis/"
        f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}/run_{run}/"
        f"excluded_fold_{test_fold}/"
        f"feature_ranking_excluded_fold_{test_fold}.csv"
    )
    pos_path          = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/group_2_positive_toxin_neuro.csv"
    neg_path          = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/group_2_negative_toxin_neuro.csv"
    folds_path        = f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/folds_{percent_identity}"
    output_path       = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis/classifier_results/run_{run}/log_regression_csv_results/"
        f"graphpart_{percent_identity}/"
        f"excluded_fold_{test_fold}/"
    )
    output_csv = os.path.join(output_path, f"logistic_regression_results_excluded_fold_{test_fold}.csv")

    # directory for models for *this* homology
    models_dir = os.path.join(models_root)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    print("Loading feature ranking…")
    feat_df = pd.read_csv(feature_rank_path).sort_values("rank")
    feature_indices = feat_df["feature"].tolist()

    print("Loading positive/negative labels…")
    pos_df    = pd.read_csv(pos_path).assign(label=1)
    neg_df    = pd.read_csv(neg_path).assign(label=0)
    labels_df = pd.concat([pos_df, neg_df], ignore_index=True)

    print("Collecting embeddings…")
    features_list = []
    labels_list   = []
    entries_list  = []
    not_found_files = []

    for _, row in labels_df.iterrows():
        entry     = row["Entry"]
        res_index = int(row["residue_number"]) - 1  # zero‐based
        emb_file  = os.path.join(embeddings_dir, f"{entry}.pt") #old: f"{entry}_original_SAE.pt
        if not os.path.isfile(emb_file):
            if entry not in not_found_files:
                not_found_files.append(entry)
            continue
        tensor = torch.load(emb_file, map_location="cpu")
        arr    = tensor.numpy()
        features_list.append(arr[res_index, :])
        labels_list.append(row["label"])
        entries_list.append(entry)

    print(print("Could not find embeddings for the following entries:", not_found_files))

    X_all       = np.vstack(features_list)
    y_all       = np.array(labels_list)
    entries_arr = np.array(entries_list)

    # ─── Load excluded fold entries and build mask for “not test” rows ───────────────
    test_fold_path = os.path.join(folds_path, f"fold_{test_fold}.txt")
    with open(test_fold_path, encoding="utf-8", errors="ignore") as file:
        test_fold_entries = file.read().strip().split()
    # train_mask is True for rows NOT in test fold → these are your training/validation rows
    train_mask = ~np.isin(entries_arr, test_fold_entries)

    # ─── Gather only non-excluded folds (ignore test fold) ───
    all_folds = list(range(10))
    non_excluded_folds = [f for f in all_folds if f != test_fold]

    # build a regex like r"^fold_(?:0|1|2|3|4|5|6|7|8)\.txt$"
    nums = "|".join(str(f) for f in non_excluded_folds)
    pattern = re.compile(rf"^fold_(?:{nums})\.txt$")

    fold_files = sorted(
        file_name for file_name in os.listdir(folds_path)
        if pattern.match(file_name)
    )

    print(f"Using the following folds for training/validation: {fold_files}")

    print("Running logistic regression with fold‑based CV:")
    records = []
    for num_feat in range(10, 1000, 10):
        print(f"\n • num_features = {num_feat}")
        top_idxs = feature_indices[:num_feat]
        X        = X_all[:, top_idxs]

        accs = []
        for fold_file in fold_files:
            # load this fold's validation entries
            path = os.path.join(folds_path, fold_file)
            with open(path, encoding="utf-8", errors="ignore") as file:
                content     = file.read()
                val_entries = content.strip().split()
            # split X/y into train/test by checking Entry membership
            val_mask  = np.isin(entries_arr, val_entries)
            test_idx  = np.where(val_mask)[0]
            train_idx = np.where(~val_mask)[0]

            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(X[train_idx], y_all[train_idx])
            preds = classifier.predict(X[test_idx])
            val_accuracy = accuracy_score(y_all[test_idx], preds)
            accs.append(val_accuracy)
            print(f"fold: {fold_file} → validation accuracy: {val_accuracy:.3f}")

        avg_acc = np.mean(accs) * 100
        print(f"Average val accuracy: {avg_acc:.2f}%")
        records.append({
            "num_features": num_feat,
            "validation":   f"{avg_acc:.2f}"
        })

        # ─── train final model on *only folds 0–8* for this feature‐count and save it
        final_clf = LogisticRegression(max_iter=1000)
        X_train_final = X_all[train_mask][:, top_idxs]
        y_train_final = y_all[train_mask]
        final_clf.fit(X_train_final, y_train_final)

        model_path = os.path.join(models_dir, f"logreg_final_features_{num_feat}.joblib")
        joblib.dump(final_clf, model_path)

    # write out your CSV of CV results
    pd.DataFrame(records).to_csv(output_csv, index=False)
    print(f"✔ CV results → {output_csv}")
    print(f"✔ Saved models ✓ in {models_dir}")
