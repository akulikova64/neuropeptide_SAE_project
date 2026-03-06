#!/usr/bin/env python3
import os
import gc
import torch
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics       import accuracy_score
from sklearn_crfsuite      import CRF

# ─── CONFIG ──────────────────────────────────────────────────────────────────
homology_training_list = [90]   # ← only 40% here!
n_splits   = 1    # folds for cross‐val
test_size  = 0.1
max_iter   = 20

# ─── ROOT FOLDERS ─────────────────────────────────────────────────────────────
DATA_ROOT   = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
INPUT_ROOT  = os.path.join(DATA_ROOT, "input_data",  "training_data")
EMB_ROOT    = os.path.join(DATA_ROOT, "toxin_neuro_SAE_embeddings")
MODELS_ROOT = os.path.join(DATA_ROOT, "classifier_results", "conditional_random_field_models")
CSV_ROOT    = os.path.join(DATA_ROOT, "classifier_results", "crf_csv_results")

os.makedirs(MODELS_ROOT, exist_ok=True)
os.makedirs(CSV_ROOT,    exist_ok=True)

for hom in homology_training_list:
    print(f"\n=== HOMOLOGY: {hom}% ===")
    group_dir    = os.path.join(INPUT_ROOT, f"homology_group{hom}")
    feature_rank = os.path.join(group_dir, f"feature_ranking_hybrid_hom{hom}.csv")
    pos_csv      = os.path.join(group_dir, "group_2_positive_toxin_neuro_train.csv")
    txt_entries  = os.path.join(group_dir, "training_entry_names.txt")

    models_dir = os.path.join(MODELS_ROOT, f"hom{hom}")
    os.makedirs(models_dir, exist_ok=True)

    out_csv = os.path.join(CSV_ROOT, f"crf_results_{hom}_hom_dup.csv")

    # ─── load your ranked features and your positive‐sites list ───────────────
    feat_df  = pd.read_csv(feature_rank).sort_values("rank")
    feat_idx = feat_df["feature"].tolist()

    pos_df = pd.read_csv(pos_csv)
    pos_df["residue_number"] = pos_df["residue_number"].astype(int)

    # ─── which entries to include in this homology group ──────────────────────
    with open(txt_entries) as fh:
        allowed = set(fh.read().split())

    # ─── now build full sequences & labels per‐Entry ──────────────────────────
    seq_dict = {}
    for entry in allowed:
        emb_file = os.path.join(EMB_ROOT, f"{entry}_original_SAE.pt")
        if not os.path.isfile(emb_file):
            continue
        arr = torch.load(emb_file, map_location="cpu").numpy()  # shape = (L, num_features)
        L   = arr.shape[0]
        pos_sites = set(pos_df.loc[pos_df["Entry"] == entry, "residue_number"].tolist())
        labels    = [1 if (i+1) in pos_sites else 0 for i in range(L)]
        seq_dict[entry] = {"raw": [arr[i] for i in range(L)], "y": labels}

    entries = list(seq_dict.keys())
    X_raw   = [seq_dict[e]["raw"] for e in entries]
    Y_raw   = [seq_dict[e]["y"]   for e in entries]

    # ─── **convert all labels to strings** for python-crfsuite ───────────────
    Y_raw = [[str(l) for l in seq] for seq in Y_raw]

    # ─── pre‐build dict form of every sequence once ───────────────────────────
    full_dict_seqs = [
        [{f"f{i}": float(v[i]) for i in feat_idx} for v in seq]
        for seq in X_raw
    ]

    # ─── split by Entry ──────────────────────────────────────────────────────
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

    records = []
    for num_feat in range(5, 205, 10):
        print(f" • num_features = {num_feat}")
        keys   = {f"f{i}" for i in feat_idx[:num_feat]}
        X_seqs = [
            [{k: token[k] for k in keys} for token in seq_dicts]
            for seq_dicts in full_dict_seqs
        ]

        accs = []
        for train_i, test_i in gss.split(entries, entries, entries):
            X_tr = [X_seqs[i] for i in train_i]
            y_tr = [Y_raw[i]  for i in train_i]
            X_te = [X_seqs[i] for i in test_i]
            y_te = [Y_raw[i]  for i in test_i]

            crf = CRF(algorithm="lbfgs",
                      max_iterations=max_iter,
                      all_possible_transitions=True)
            crf.fit(X_tr, y_tr)
            y_pr = crf.predict(X_te)

            accs.append(accuracy_score(sum(y_te, []), sum(y_pr, [])))

            # cleanup this fold
            del crf, X_tr, y_tr, X_te, y_te, y_pr
            gc.collect()

        avg = np.mean(accs) * 100
        records.append({"num_features": num_feat, "validation": f"{avg:.2f}"})

        # ─── train final on all sequences & save
        crf_final = CRF(algorithm="lbfgs",
                        max_iterations=max_iter,
                        all_possible_transitions=True)
        crf_final.fit(X_seqs, Y_raw)
        joblib.dump(crf_final,
                    os.path.join(models_dir, f"crf_hom{hom}_{num_feat}.joblib"))

        # cleanup
        del X_seqs, crf_final
        gc.collect()

    # ─── Save CV results ──────────────────────────────────────────────────────
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"✔ CRF CV results → {out_csv}")
