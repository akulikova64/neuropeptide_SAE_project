#!/usr/bin/env python3
import os
import re
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import joblib

# ─── Configuration ─────────────────────────────────────────────────────────
test_folds       = list(range(10))
percent_identity = 40
run = 4

# ─── Static paths ──────────────────────────────────────────────────────────
embeddings_dir = (
    "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
    f"toxin_neuro_finetuned_SAE_embeddings/run_{run}/"
)
pos_path = (
    "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
    "input_data/group_2_positive_toxin_neuro.csv"
)
neg_path = (
    "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
    "input_data/group_2_negative_toxin_neuro.csv"
)
folds_path = (
    "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
    f"input_data/folds_{percent_identity}"
)
feature_rank_path = (
    "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis/"
    f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}/run_{run}/"
    "excluded_fold_{fold}/feature_ranking_excluded_fold_{fold}.csv"
)
models_path = (
    "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis/"
    f"classifier_results/run_{run}/logistic_regression_models/"
    f"graphpart_{percent_identity}/excluded_fold_{{fold}}"
)
output_root_template = (
    "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis/"
    f"classifier_results/run_{run}/test_log_regression_csv_results/"
    f"graphpart_{percent_identity}/excluded_fold_{{fold}}"
)

# ─── Read all labels & embeddings (once) ────────────────────────────────────
pos_df = pd.read_csv(pos_path).assign(label=1)
neg_df = pd.read_csv(neg_path).assign(label=0)
labels_df = pd.concat([pos_df, neg_df], ignore_index=True)

features_list, labels_list, entries_list = [], [], []
for _, row in labels_df.iterrows():
    entry   = row["Entry"]
    idx0    = int(row["residue_number"]) - 1
    emb_fp  = os.path.join(embeddings_dir, f"{entry}.pt") # old: {entry}_original_SAE.pt
    if not os.path.isfile(emb_fp):
        continue
    arr = torch.load(emb_fp, map_location="cpu").numpy()
    features_list.append(arr[idx0])
    labels_list.append(row["label"])
    entries_list.append(entry)

X_all       = np.vstack(features_list)
y_all       = np.array(labels_list)
entries_arr = np.array(entries_list)

# ─── Loop over each excluded fold → evaluate on that fold ─────────────────
for test_fold in test_folds:
    print(f"\n=== Testing on test fold {test_fold} ===")

    # — build test index for this fold
    folds_dir = folds_path
    test_fp   = os.path.join(folds_dir, f"fold_{test_fold}.txt")
    with open(test_fp, encoding="utf-8", errors="ignore") as fh:
        test_entries = fh.read().split()
    test_mask = np.isin(entries_arr, test_entries)
    test_idx  = np.where(test_mask)[0]
    if test_idx.size == 0:
        print(f"⚠️  No entries for fold {test_fold}, skipping.")
        continue

    # — load feature ranking for this fold
    feat_rank_fp = feature_rank_path.format(fold=test_fold)
    feat_df      = pd.read_csv(feat_rank_fp).sort_values("rank")
    feature_indices = feat_df["feature"].tolist()

    # — find all the saved models for this fold
    models_dir = models_path.format(fold=test_fold)
    model_files = sorted(
        f for f in os.listdir(models_dir)
        if re.match(r"logreg_final_features_\d+\.joblib$", f)
    )

    # — prepare output directory & CSV path
    out_dir  = output_root_template.format(fold=test_fold)
    os.makedirs(out_dir, exist_ok=True)
    out_csv  = os.path.join(
        out_dir,
        f"linear_regression_test_results_test_fold_{test_fold}.csv"
    )

    results = []
    # — evaluate each model (varying num_features)
    for fn in model_files:
        m = re.match(r"logreg_final_features_(\d+)\.joblib$", fn)
        num_feat = int(m.group(1))
        clf_path = os.path.join(models_dir, fn)
        clf      = joblib.load(clf_path)

        top_idxs = feature_indices[:num_feat]
        X_test   = X_all[test_idx][:, top_idxs]
        y_test   = y_all[test_idx]

        preds = clf.predict(X_test)
        acc   = accuracy_score(y_test, preds) * 100

        print(f" Features={num_feat:3d} → Test Acc = {acc:5.2f}%")
        results.append({
            "num_features": num_feat,
            "test_accuracy": f"{acc:.2f}"
        })

    # — write per-fold test results
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"✔ Saved fold {test_fold} test results → {out_csv}")
