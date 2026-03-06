#!/usr/bin/env python3
import os
import gc
import torch
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score

# ─── CONFIG ──────────────────────────────────────────────────────────────────
homology_training_list = [40, 50, 60, 70, 80, 90]

DATA_ROOT      = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
MODELS_ROOT    = os.path.join(DATA_ROOT, "classifier_results", "conditional_random_field_models")
EMB_ROOT       = os.path.join(DATA_ROOT, "toxin_neuro_SAE_embeddings")
INPUT_TEST_DIR = os.path.join(DATA_ROOT, "input_data", "test_data")
OUTPUT_ROOT    = os.path.join(DATA_ROOT, "classifier_results", "test_crf_results")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

for hom in homology_training_list:
    print(f"\n=== TESTING HOMOLOGY: {hom}% ===")

    # ─── File paths ───────────────────────────────────────────────────────────
    feat_csv = os.path.join(DATA_ROOT, "input_data", "training_data",
                            f"homology_group{hom}", f"feature_ranking_hybrid_hom{hom}.csv")
    pos_csv  = os.path.join(INPUT_TEST_DIR, f"homology_group{hom}", "group_2_positive_toxin_neuro_test.csv")
    txt_ent  = os.path.join(INPUT_TEST_DIR, f"homology_group{hom}", "test_entry_names.txt")
    model_dir = os.path.join(MODELS_ROOT, f"hom{hom}")
    out_csv   = os.path.join(OUTPUT_ROOT, f"crf_test_results_{hom}_hom.csv")

    # ─── Load feature rankings ───────────────────────────────────────────────
    feat_df  = pd.read_csv(feat_csv).sort_values("rank")
    feat_idx = feat_df["feature"].tolist()

    # ─── Load test positive sites ────────────────────────────────────────────
    pos_df = pd.read_csv(pos_csv)
    pos_df["residue_number"] = pos_df["residue_number"].astype(int)

    # ─── Load test entry names ───────────────────────────────────────────────
    with open(txt_ent) as fh:
        allowed = set(fh.read().split())

    # ─── Load embeddings and build sequences ─────────────────────────────────
    seq_dict = {}
    for entry in allowed:
        emb_file = os.path.join(EMB_ROOT, f"{entry}_original_SAE.pt")
        if not os.path.isfile(emb_file):
            print(f"Skipping missing embedding: {entry}")
            continue
        arr = torch.load(emb_file, map_location="cpu").numpy()  # shape = (L, num_features)
        L = arr.shape[0]
        pos_sites = set(pos_df.loc[pos_df["Entry"] == entry, "residue_number"].tolist())
        labels = [1 if (i+1) in pos_sites else 0 for i in range(L)]
        seq_dict[entry] = {"raw": [arr[i] for i in range(L)], "y": labels}

    entries = list(seq_dict.keys())
    if not entries:
        print(f"No valid entries found for homology {hom}. Skipping.")
        continue

    X_raw = [seq_dict[e]["raw"] for e in entries]
    Y_raw = [seq_dict[e]["y"] for e in entries]
    Y_raw = [[str(l) for l in seq] for seq in Y_raw]  # convert to str for CRF

    # ─── Precompute full feature dictionary for each sequence ────────────────
    full_dict_seqs = [
        [{f"f{i}": float(v[i]) for i in feat_idx} for v in seq]
        for seq in X_raw
    ]

    records = []

    for num_feat in range(5, 205, 10):
        print(f" • Testing with top {num_feat} features")

        model_file = os.path.join(model_dir, f"crf_hom{hom}_{num_feat}.joblib")
        if not os.path.isfile(model_file):
            print(f" → Model not found: {model_file}")
            continue

        # Select top-N features
        keys = {f"f{i}" for i in feat_idx[:num_feat]}
        X_test = [
            [{k: token[k] for k in keys} for token in seq_dicts]
            for seq_dicts in full_dict_seqs
        ]

        # Load model and predict
        crf_model = joblib.load(model_file)
        y_pred = crf_model.predict(X_test)

        # Flatten for scoring
        y_true_flat = sum(Y_raw, [])
        y_pred_flat = sum(y_pred, [])

        # Compute accuracy and balanced accuracy
        acc  = accuracy_score(y_true_flat, y_pred_flat) * 100
        bacc = balanced_accuracy_score([int(i) for i in y_true_flat],
                                       [int(i) for i in y_pred_flat]) * 100

        records.append({
            "num_features": num_feat,
            "test_accuracy": f"{acc:.2f}",
            "balanced_accuracy": f"{bacc:.2f}"
        })

        # Cleanup
        del crf_model, X_test, y_pred
        gc.collect()

    # ─── Save results ────────────────────────────────────────────────────────
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"✔ Saved test results → {out_csv}")
