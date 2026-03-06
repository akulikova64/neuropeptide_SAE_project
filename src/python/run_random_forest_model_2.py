#!/usr/bin/env python3
import os
import re
import gc
import torch
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# ─── Configuration ─────────────────────────────────────────────────────────
test_folds       = list(range(10))
percent_identity = 40

# ─── Paths (mirroring your LR script) ──────────────────────────────────────
embeddings_dir = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/toxin_neuro_SAE_embeddings"

folds_path = (
    f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
    f"input_data/folds_{percent_identity}"
)

pos_path = (
    "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
    "input_data/group_2_positive_toxin_neuro.csv"
)
neg_path = (
    "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
    "input_data/group_2_negative_toxin_neuro.csv"
)

models_root_template = (
    f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/classifier_results/random_forest_models/"
    f"graphpart_{percent_identity}/500_trees/excluded_fold_{{fold}}/"
)

output_root_template = (
    f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/classifier_results/random_forest_csv_results/"
    f"graphpart_{percent_identity}/500_trees/excluded_fold_{{fold}}/"
)

feature_rank_path_template = (
    "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
    f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}/"
    f"excluded_fold_{{fold}}/feature_ranking_excluded_fold_{{fold}}.csv"
)

# ─── Helpers ───────────────────────────────────────────────────────────────
def load_rank_indices(path_csv: str) -> list[int]:
    feat_df = pd.read_csv(path_csv).sort_values("rank")
    return feat_df["feature"].astype(int).tolist()

def load_labels_and_embeddings():
    # Read labels
    pos_df = pd.read_csv(pos_path).assign(label=1)
    neg_df = pd.read_csv(neg_path).assign(label=0)
    labels_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # Collect embeddings per residue
    X_list, y_list, entry_list = [], [], []
    missing = set()

    for _, row in labels_df.iterrows():
        entry   = row["Entry"]
        idx0    = int(row["residue_number"]) - 1
        emb_fp  = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")
        if not os.path.isfile(emb_fp):
            missing.add(entry)
            continue
        arr = torch.load(emb_fp, map_location="cpu").numpy()
        if idx0 < 0 or idx0 >= arr.shape[0]:
            continue
        X_list.append(arr[idx0, :])
        y_list.append(row["label"])
        entry_list.append(entry)

    if missing:
        print("⚠️  Missing embeddings for entries:", sorted(missing))

    X_all       = np.vstack(X_list)
    y_all       = np.array(y_list, dtype=int)
    entries_all = np.array(entry_list)

    return X_all, y_all, entries_all

# ─── Main loop over outer test folds ───────────────────────────────────────
def main():
    print("Loading labels and embeddings once…")
    X_all, y_all, entries_all = load_labels_and_embeddings()
    D = X_all.shape[1]
    print(f"✔ Data matrix: {X_all.shape[0]} residues × {D} features")

    for test_fold in test_folds:
        print(f"\n=== Outer test fold {test_fold} (excluded) ===")

        # Paths for this outer fold
        models_dir = models_root_template.format(fold=test_fold)
        out_dir    = output_root_template.format(fold=test_fold)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(out_dir,    exist_ok=True)

        out_csv = os.path.join(out_dir, f"random_forest_results_excluded_fold_{test_fold}.csv")

        rank_csv = feature_rank_path_template.format(fold=test_fold)
        if not os.path.isfile(rank_csv):
            print(f"⚠️  Skipping fold {test_fold}: rank file missing → {rank_csv}")
            continue
        feature_indices = load_rank_indices(rank_csv)

        # Build mask excluding the outer test fold
        test_fold_path = os.path.join(folds_path, f"fold_{test_fold}.txt")
        with open(test_fold_path, encoding="utf-8", errors="ignore") as fh:
            test_entries = set(fh.read().split())

        train_mask = ~np.isin(entries_all, list(test_entries))

        # Prepare inner-fold files (all except the test fold)
        all_folds = list(range(10))
        inner_folds = [f for f in all_folds if f != test_fold]
        nums = "|".join(str(f) for f in inner_folds)
        pattern = re.compile(rf"^fold_(?:{nums})\.txt$")
        inner_fold_files = sorted(fn for fn in os.listdir(folds_path) if pattern.match(fn))

        print(f"Inner validation folds: {inner_fold_files}")

        # Range of features to try (don’t exceed embedding dimensionality)
        # Match your LR sweep style; clamp to D to avoid index error.
        feature_counts = [n for n in range(10, 201, 10) if n <= len(feature_indices) and n <= D]
        if not feature_counts:
            feature_counts = [min(len(feature_indices), D)]

        records = []

        for num_feat in feature_counts:
            print(f"\n • num_features = {num_feat}")
            top_idx = feature_indices[:num_feat]
            X = X_all[:, top_idx]

            accs = []
            baccs = []

            # Inner CV across the non-excluded folds
            for fold_file in inner_fold_files:
                fold_fp = os.path.join(folds_path, fold_file)
                with open(fold_fp, encoding="utf-8", errors="ignore") as fh:
                    val_entries = set(fh.read().split())

                # Validation indices are those entries in this inner fold,
                # but also must be from the training mask (exclude the outer test fold)
                val_mask_all  = np.isin(entries_all, list(val_entries))
                val_mask      = val_mask_all & train_mask
                train_mask_in = (~val_mask_all) & train_mask

                val_idx   = np.where(val_mask)[0]
                train_idx = np.where(train_mask_in)[0]

                if val_idx.size == 0 or train_idx.size == 0:
                    # Skip if this inner fold has nothing after the outer fold exclusion
                    continue

                clf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                )
                clf.fit(X[train_idx], y_all[train_idx])
                preds = clf.predict(X[val_idx])

                acc  = accuracy_score(y_all[val_idx], preds)
                bacc = balanced_accuracy_score(y_all[val_idx], preds)
                accs.append(acc)
                baccs.append(bacc)

                # cleanup
                del clf
                gc.collect()

            if len(accs) == 0:
                print("  ⚠️  No inner folds had data after masking; recording NaNs.")
                avg_acc  = np.nan
                avg_bacc = np.nan
            else:
                avg_acc  = float(np.mean(accs)) * 100.0
                avg_bacc = float(np.mean(baccs)) * 100.0

            print(f"   ↳ avg inner-fold Acc = {avg_acc:.2f}% | BalAcc = {avg_bacc:.2f}%")

            records.append({
                "num_features": num_feat,
                "validation":   f"{avg_acc:.2f}",
                "balanced_accuracy": f"{avg_bacc:.2f}",
            })

            # ─── Train final model on all NON-test data (like LR script) ───
            X_train_final = X[train_mask]
            y_train_final = y_all[train_mask]

            final_clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
                oob_score=False,   # keep consistent with LR flow (no OOB in final)
            )
            final_clf.fit(X_train_final, y_train_final)

            model_path = os.path.join(models_dir, f"rf_final_features_{num_feat}.joblib")
            joblib.dump(final_clf, model_path)

            del final_clf, X_train_final, y_train_final
            gc.collect()

        # Save per-outer-fold CV summary
        pd.DataFrame(records).to_csv(out_csv, index=False)
        print(f"✔ Saved CV summary → {out_csv}")
        print(f"✔ Models written to {models_dir}")

if __name__ == "__main__":
    main()
