#!/usr/bin/env python3
import os
import re
import gc
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn_crfsuite import CRF
from sklearn.metrics import accuracy_score

# ─── Configuration ─────────────────────────────────────────────────────────
test_folds       = list(range(10))
percent_identity = 40
max_iter         = 20
feature_step     = 10
feature_max      = 255

# ─── Static roots ──────────────────────────────────────────────────────────
DATA_ROOT    = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
EMB_ROOT     = os.path.join(DATA_ROOT, "toxin_neuro_SAE_embeddings")
POS_CSV      = os.path.join(DATA_ROOT, "input_data", "group_2_positive_toxin_neuro.csv")
FOLDS_ROOT   = os.path.join(DATA_ROOT, "input_data", f"folds_{percent_identity}")

for test_fold in test_folds:
    print(f"\n=== EXCLUDING FOLD {test_fold} AS TEST SET ===")

    # ─── Paths for this fold ────────────────────────────────────────────────
    feature_rank_path = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}/"
        f"excluded_fold_{test_fold}/"
        f"feature_ranking_excluded_fold_{test_fold}.csv"
    )
    models_root = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"classifier_results/conditional_random_field_models_2/"
        f"graphpart_{percent_identity}/excluded_fold_{test_fold}"
    )
    csv_root = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
        f"classifier_results/crf_csv_results_2/"
        f"graphpart_{percent_identity}/excluded_fold_{test_fold}"
    )
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(csv_root,    exist_ok=True)

    # ─── Load feature ranking ────────────────────────────────────────────────
    feat_df       = pd.read_csv(feature_rank_path).sort_values("rank")
    feature_list  = feat_df["feature"].tolist()

    # ─── Load positive sites ─────────────────────────────────────────────────
    pos_df        = pd.read_csv(POS_CSV)
    pos_df["residue_number"] = pos_df["residue_number"].astype(int)

    # ─── Determine allowed entries (exclude test_fold) ───────────────────────
    test_fp = os.path.join(FOLDS_ROOT, f"fold_{test_fold}.txt")
    with open(test_fp, encoding="utf-8", errors="ignore") as fh:
        test_entries = set(fh.read().split())

    # Collect all entries that ever appear in your positives CSV
    pos_entries = set(pos_df["Entry"])

    # Build seq_dict for entries that are both in pos_entries AND not in the test fold
    seq_dict = {}
    for emb_fn in os.listdir(EMB_ROOT):
        entry = emb_fn.replace("_original_SAE.pt", "")
        # skip if it's in the test fold or not in the positives list
        if entry in test_entries or entry not in pos_entries:
            continue

        full_path = os.path.join(EMB_ROOT, emb_fn)
        arr       = torch.load(full_path, map_location="cpu").numpy()
        L         = arr.shape[0]

        # label = 1 if residue in pos_df, else 0
        pos_sites = set(pos_df.loc[pos_df["Entry"] == entry, "residue_number"])
        labels    = [str(1 if (i+1) in pos_sites else 0) for i in range(L)]
        seq_dict[entry] = {"raw": arr, "y": labels}

    entries = list(seq_dict.keys())
    print(f"  {len(entries)} entries after excluding fold {test_fold}")

    # ─── Build full_dict_seqs once ───────────────────────────────────────────
    # dict form per token: {"f0": val, "f1": val, ...}
    full_dict_seqs = []
    for entry in entries:
        arr = seq_dict[entry]["raw"]
        seq_feats = [
            {f"f{i}": float(arr[pos,i]) for i in feature_list}
            for pos in range(arr.shape[0])
        ]
        full_dict_seqs.append(seq_feats)

    Y_raw = [seq_dict[e]["y"] for e in entries]

    # ─── Gather inner-folds 0–9 minus test_fold ─────────────────────────────
    all_folds = list(range(10))
    non_ex_folds = [f for f in all_folds if f != test_fold]
    nums = "|".join(str(f) for f in non_ex_folds)
    pattern = re.compile(rf"^fold_(?:{nums})\.txt$")
    fold_files = sorted(
        fn for fn in os.listdir(FOLDS_ROOT)
        if pattern.match(fn)
    )
    print(f"  Inner CV folds: {fold_files}")

    # ─── Nested CV: vary num_features ───────────────────────────────────────
    records = []
    for num_feat in range(5, feature_max+1, feature_step):
        print(f"\n • num_features = {num_feat}")
        keys    = {f"f{i}" for i in feature_list[:num_feat]}
        # slice features
        X_seqs  = [
            [{k: token[k] for k in keys} for token in seq]
            for seq in full_dict_seqs
        ]

        accs = []
        pos_accs = []
        neg_accs = []
        for fold_file in fold_files:
            # validation entries for this inner fold
            path = os.path.join(FOLDS_ROOT, fold_file)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                val_entries = set(fh.read().split())
            # build train/test indices
            val_idx = [i for i,e in enumerate(entries) if e in val_entries]
            trn_idx = [i for i in range(len(entries)) if i not in val_idx]

            X_tr = [X_seqs[i] for i in trn_idx]
            y_tr = [Y_raw[i]   for i in trn_idx]
            X_te = [X_seqs[i] for i in val_idx]
            y_te = [Y_raw[i]   for i in val_idx]

            crf = CRF(
                algorithm="lbfgs",
                max_iterations=max_iter,
                all_possible_transitions=True
            )
            crf.fit(X_tr, y_tr)
            y_pr = crf.predict(X_te)

            # ── Compute overall, positive-class, and negative-class accuracy
            y_true_flat = np.array([int(v) for v in sum(y_te, [])], dtype=int)
            y_pred_flat = np.array([int(v) for v in sum(y_pr, [])], dtype=int)

            acc = (y_true_flat == y_pred_flat).mean()

            pos_mask = (y_true_flat == 1)
            neg_mask = (y_true_flat == 0)

            pos_acc = (y_pred_flat[pos_mask] == 1).mean() if pos_mask.any() else np.nan
            neg_acc = (y_pred_flat[neg_mask] == 0).mean() if neg_mask.any() else np.nan

            accs.append(acc)
            pos_accs.append(pos_acc)
            neg_accs.append(neg_acc)

            # cleanup this fold
            del crf, X_tr, y_tr, X_te, y_te, y_pr
            gc.collect()

        avg_acc = np.mean(accs) * 100.0
        avg_pos = float(np.nanmean(pos_accs)) * 100.0
        avg_neg = float(np.nanmean(neg_accs)) * 100.0

        print(f"   ↳ avg inner-fold acc = {avg_acc:.2f}% | pos_acc = {avg_pos:.2f}% | neg_acc = {avg_neg:.2f}%")

        records.append({
            "num_features": num_feat,
            "val_acc":      f"{avg_acc:.2f}",
            "pos_acc":      f"{avg_pos:.2f}",
            "neg_acc":      f"{avg_neg:.2f}"
        })

        # ─── Train final CRF on all non-test entries ────────────────────────
        crf_final = CRF(
            algorithm="lbfgs",
            max_iterations=max_iter,
            all_possible_transitions=True
        )
        crf_final.fit(X_seqs, Y_raw)
        joblib.dump(
            crf_final,
            os.path.join(models_root, f"crf_excl_fold{test_fold}_{num_feat}.joblib")
        )
        del crf_final
        gc.collect()

    # ─── Save nested‐CV results CSV ─────────────────────────────────────────
    out_csv = os.path.join(csv_root, f"crf_results_excluded_fold_{test_fold}.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"✔ Saved CRF CV results → {out_csv}")
