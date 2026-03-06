#!/usr/bin/env python3
import os
import re
import gc
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# this script is modified to run on the chpc server. Change paths if you want to run it locally. 

# ─── Configuration ─────────────────────────────────────────────────────────
test_folds       = list(range(10))
percent_identity = 40

# ─── Paths (aligned with your RF training script) ──────────────────────────
embeddings_dir = "../toxin_neuro_SAE_embeddings"

folds_path = f"./input_files/folds_{percent_identity}"

pos_path = "./input_files/group_2_positive_toxin_neuro.csv"
neg_path = "./input_files/group_2_negative_toxin_neuro.csv"

# Feature ranks used to pick top-N features per fold
feature_rank_path_template = (
    f"./input_files/F1_scores_graphpart_{percent_identity}/"
    f"excluded_fold_{{fold}}/feature_ranking_excluded_fold_{{fold}}.csv"
)

# Where trained RF models were saved by your training script
models_root_template = "./output_models/excluded_fold_{fold}/"

# Where to write per-fold test CSVs
output_root_template = "./output_csv_files/excluded_fold_{fold}/"


# ─── Helpers ───────────────────────────────────────────────────────────────
def load_rank_indices(path_csv: str) -> list[int]:
    """Return list of feature indices sorted by ascending rank."""
    feat_df = pd.read_csv(path_csv).sort_values("rank")
    return feat_df["feature"].astype(int).tolist()


def load_labels_and_embeddings():
    """Load labels and per-residue embeddings once (like in training)."""
    pos_df = pd.read_csv(pos_path).assign(label=1)
    neg_df = pd.read_csv(neg_path).assign(label=0)
    labels_df = pd.concat([pos_df, neg_df], ignore_index=True)

    X_list, y_list, entry_list = [], [], []
    missing = set()

    for _, row in labels_df.iterrows():
        entry = row["Entry"]
        idx0 = int(row["residue_number"]) - 1
        emb_fp = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")

        if not os.path.isfile(emb_fp):
            missing.add(entry)
            continue

        arr = torch.load(emb_fp, map_location="cpu").numpy()
        if idx0 < 0 or idx0 >= arr.shape[0]:
            continue

        X_list.append(arr[idx0, :])
        y_list.append(int(row["label"]))
        entry_list.append(entry)

    if missing:
        print("⚠️  Missing embeddings for entries:", len(missing))

    X_all = np.vstack(X_list)
    y_all = np.array(y_list, dtype=int)
    entries_all = np.array(entry_list)

    return X_all, y_all, entries_all


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    print("Loading labels and embeddings once…")
    X_all, y_all, entries_all = load_labels_and_embeddings()
    D = X_all.shape[1]
    print(f"✔ Data matrix: {X_all.shape[0]} residues × {D} features")

    for test_fold in test_folds:
        print(f"\n=== Testing on excluded fold {test_fold} ===")

        # — Build test indices for this fold
        test_fold_path = os.path.join(folds_path, f"fold_{test_fold}.txt")
        if not os.path.isfile(test_fold_path):
            print(f"  ⚠️  Missing fold file: {test_fold_path}; skipping.")
            continue

        with open(test_fold_path, encoding="utf-8", errors="ignore") as fh:
            test_entries = set(fh.read().split())

        test_mask = np.isin(entries_all, list(test_entries))
        test_idx = np.where(test_mask)[0]
        if test_idx.size == 0:
            print(f"  ⚠️  No entries found for test fold {test_fold}; skipping.")
            continue

        # — Load feature ranking for this fold
        rank_csv = feature_rank_path_template.format(fold=test_fold)
        if not os.path.isfile(rank_csv):
            print(f"  ⚠️  Rank file missing for fold {test_fold} → {rank_csv}")
            continue

        feature_indices = load_rank_indices(rank_csv)

        # — Locate saved RF models for this fold
        models_dir = models_root_template.format(fold=test_fold)
        if not os.path.isdir(models_dir):
            print(f"  ⚠️  Models dir missing: {models_dir}; skipping.")
            continue

        model_files = sorted(
            f for f in os.listdir(models_dir)
            if re.match(r"rf_final_features_(\d+)\.joblib$", f)
        )
        if not model_files:
            print(f"  ⚠️  No RF model files found in {models_dir}; skipping.")
            continue

        # — Prepare output
        out_dir = output_root_template.format(fold=test_fold)
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(
            out_dir,
            f"random_forest_test_results_test_fold_{test_fold}.csv"
        )

        results = []

        # — Evaluate each saved model (varying num_features)
        for fn in model_files:
            m = re.match(r"rf_final_features_(\d+)\.joblib$", fn)
            num_feat = int(m.group(1))
            clf_path = os.path.join(models_dir, fn)

            # Guard against inconsistent feature counts
            top_idx = feature_indices[:num_feat]
            if len(top_idx) < num_feat:
                print(f"  ⚠️  Fold {test_fold}: only {len(top_idx)} ranked features; "
                      f"skipping model expecting {num_feat}.")
                continue

            # Slice test set features
            X_test = X_all[test_idx][:, top_idx]
            y_test = y_all[test_idx]

            # Load and predict
            clf = joblib.load(clf_path)
            preds = clf.predict(X_test)

            acc = accuracy_score(y_test, preds) * 100.0
            bacc = balanced_accuracy_score(y_test, preds) * 100.0

            print(f"   Features={num_feat:3d} → "
                  f"Test Acc = {acc:5.2f}% | BalAcc = {bacc:5.2f}%")

            results.append({
                "num_features": num_feat,
                "test_accuracy": f"{acc:.2f}",
                "test_balanced_accuracy": f"{bacc:.2f}",
            })

            # cleanup
            del clf
            gc.collect()

        # — Write per-fold test results
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"✔ Saved fold {test_fold} test results → {out_csv}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
