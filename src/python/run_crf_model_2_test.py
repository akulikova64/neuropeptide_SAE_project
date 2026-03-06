#!/usr/bin/env python3
import os
import re
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn_crfsuite import CRF
from sklearn.metrics     import accuracy_score

# ─── Configuration ─────────────────────────────────────────────────────────
test_folds           = list(range(10))
percent_identity     = 40

# ─── Static roots & templates ─────────────────────────────────────────────
DATA_ROOT              = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
EMB_ROOT               = os.path.join(DATA_ROOT, "toxin_neuro_SAE_embeddings")
POS_CSV                = os.path.join(DATA_ROOT, "input_data", "group_2_positive_toxin_neuro.csv")
FOLDS_ROOT             = os.path.join(DATA_ROOT, "input_data", f"folds_{percent_identity}")

feat_rank_template     = (
    f"{DATA_ROOT}/F1_scores_hybrid/F1_scores_graphpart_{percent_identity}/"
    "excluded_fold_{fold}/feature_ranking_excluded_fold_{fold}.csv"
)
models_root_template   = (
    f"{DATA_ROOT}/classifier_results/conditional_random_field_models/"
    f"graphpart_{percent_identity}/excluded_fold_{{fold}}"
)
output_root_template   = (
    f"{DATA_ROOT}/classifier_results/crf_csv_results/"
    f"graphpart_{percent_identity}/excluded_fold_{{fold}}"
)

# ─── Load positive‐site table once ─────────────────────────────────────────
pos_df = pd.read_csv(POS_CSV)
pos_df["residue_number"] = pos_df["residue_number"].astype(int)

for test_fold in test_folds:
    print(f"\n=== Testing CRF on excluded fold {test_fold} ===")

    # — build test‐entries set
    fold_fp = os.path.join(FOLDS_ROOT, f"fold_{test_fold}.txt")
    with open(fold_fp, encoding="utf-8", errors="ignore") as fh:
        test_entries = set(fh.read().split())
    if not test_entries:
        print(f"⚠️  No entries in fold {test_fold}, skipping.")
        continue

    # — load feature ranking for this fold
    feat_rank_fp = feat_rank_template.format(fold=test_fold)
    feat_df      = pd.read_csv(feat_rank_fp).sort_values("rank")
    feature_list = feat_df["feature"].tolist()

    # — prepare test data sequences
    test_seq_dict = {}
    for emb_fn in os.listdir(EMB_ROOT):
        entry = emb_fn.replace("_original_SAE.pt", "")
        if entry not in test_entries:
            continue
        arr = torch.load(os.path.join(EMB_ROOT, emb_fn), map_location="cpu").numpy()
        L   = arr.shape[0]
        pos_sites = set(pos_df.loc[pos_df["Entry"] == entry, "residue_number"])
        labels    = [str(1 if (i+1) in pos_sites else 0) for i in range(L)]
        test_seq_dict[entry] = {"raw": arr, "y": labels}

    entries_test = list(test_seq_dict.keys())
    if not entries_test:
        print(f"⚠️  No valid embeddings for fold {test_fold}, skipping.")
        continue
    print(f"  {len(entries_test)} entries in test set")

    # — discover saved CRF models for this fold
    models_dir = models_root_template.format(fold=test_fold)
    model_files = sorted(
        fn for fn in os.listdir(models_dir)
        if re.match(rf"crf_excl_fold{test_fold}_(\d+)\.joblib$", fn)
    )

    # — prepare output directory & CSV path
    out_dir = output_root_template.format(fold=test_fold)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"crf_test_results_fold_{test_fold}.csv")

    results = []
    # — evaluate each saved model
    for fn in model_files:
        m = re.match(rf"crf_excl_fold{test_fold}_(\d+)\.joblib$", fn)
        num_feat = int(m.group(1))
        clf      = joblib.load(os.path.join(models_dir, fn))

        # build X_test_seqs and Y_test_raw
        keys       = {f"f{i}" for i in feature_list[:num_feat]}
        X_test_seqs = [
            [{k: seq_dict["raw"][pos,i] for k in keys for i in [int(k[1:])]} 
             for pos in range(seq_dict["raw"].shape[0])]
            for seq_dict in test_seq_dict.values()
        ]
        # simpler comprehension:
        X_test_seqs = []
        Y_test_raw  = []
        for entry, seq_dict in test_seq_dict.items():
            arr = seq_dict["raw"]
            seq_feats = [
                {f"f{i}": float(arr[pos, i]) for i in feature_list[:num_feat]}
                for pos in range(arr.shape[0])
            ]
            X_test_seqs.append(seq_feats)
            Y_test_raw.append(seq_dict["y"])

        # predict & flatten
        y_pr_seqs = clf.predict(X_test_seqs)
        y_te_flat = sum(Y_test_raw, [])
        y_pr_flat = sum(y_pr_seqs, [])

        acc = accuracy_score(y_te_flat, y_pr_flat) * 100
        print(f" Features={num_feat:3d} → Test Acc = {acc:5.2f}%")
        results.append({
            "num_features": num_feat,
            "test_accuracy": f"{acc:.2f}"
        })

    # — write per-fold test results CSV
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"✔ Saved CRF test results → {out_csv}")
