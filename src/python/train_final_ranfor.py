#!/usr/bin/env python3
import os
import gc
import re
import json
import torch
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier

# ─── Config ────────────────────────────────────────────────────────────────
percent_identity = 40
num_features     = 70
random_state     = 42
trees = 100

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

# Where to save the final model
models_root_template = "./output_models/"  # final model will go under this root


# ─── Helpers ───────────────────────────────────────────────────────────────
def load_union_entries(folds_dir: str) -> set[str]:
    allowed = set()
    for k in range(10):
        fp = os.path.join(folds_dir, f"fold_{k}.txt")
        if not os.path.isfile(fp):
            print(f"⚠️  Missing fold file: {fp} (skipping)")
            continue
        with open(fp, encoding="utf-8", errors="ignore") as fh:
            allowed.update(fh.read().split())
    if not allowed:
        raise RuntimeError(f"No entries found under {folds_dir}")
    return allowed


def load_labels_df() -> pd.DataFrame:
    pos_df = pd.read_csv(pos_path).assign(label=1)
    neg_df = pd.read_csv(neg_path).assign(label=0)
    both = pd.concat([pos_df, neg_df], ignore_index=True)
    both["residue_number"] = both["residue_number"].astype(int)
    return both


def aggregate_top_features_across_folds(k_top: int) -> list[int]:
    """
    Combine per-fold rankings into a single list by averaging ranks.
    Uses only folds whose CSV exists.
    """
    rank_sums = defaultdict(float)
    counts    = defaultdict(int)

    seen_any = False
    for fold in range(10):
        csv_path = feature_rank_path_template.format(fold=fold)
        if not os.path.isfile(csv_path):
            print(f"⚠️  Rank CSV missing for fold {fold}: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if "rank" not in df.columns or "feature" not in df.columns:
            print(f"⚠️  Bad columns in {csv_path}; needs ['rank','feature']")
            continue

        seen_any = True
        for _, row in df.iterrows():
            feat = int(row["feature"])
            rnk  = float(row["rank"])
            rank_sums[feat] += rnk
            counts[feat]    += 1

    if not seen_any:
        raise RuntimeError("No per-fold ranking CSVs found; cannot aggregate features.")

    # average rank: smaller is better
    avg = [(feat, rank_sums[feat] / counts[feat]) for feat in rank_sums.keys()]
    avg.sort(key=lambda x: x[1])
    top_feats = [f for f, _ in avg[:k_top]]

    if len(top_feats) < k_top:
        raise RuntimeError(f"Only {len(top_feats)} unique features aggregated; need {k_top}.")
    return top_feats


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    print("✔ Aggregating feature rankings across folds…")
    top_feats = aggregate_top_features_across_folds(num_features)
    print(f"   Using top {num_features} features (max index = {max(top_feats)})")

    print("✔ Collecting entries from all folds…")
    allowed_entries = load_union_entries(folds_path)
    print(f"   Combined {len(allowed_entries)} unique entries")

    print("✔ Loading labels…")
    labels_df = load_labels_df()
    labels_df = labels_df[labels_df["Entry"].isin(allowed_entries)].copy()
    if labels_df.empty:
        raise RuntimeError("No labeled residues after filtering by fold entries.")

    print("✔ Building training matrix from embeddings…")
    X_list, y_list, used_entries = [], [], []
    missing = set()
    D_checked = False
    D_dim = None

    for _, r in labels_df.iterrows():
        entry = r["Entry"]
        idx0  = int(r["residue_number"]) - 1
        emb_fp = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")
        if not os.path.isfile(emb_fp):
            missing.add(entry)
            continue

        arr = torch.load(emb_fp, map_location="cpu").numpy()  # (L, D)
        L, D = arr.shape
        if idx0 < 0 or idx0 >= L:
            continue

        if not D_checked:
            D_dim = D
            if max(top_feats) >= D_dim:
                raise RuntimeError(
                    f"Feature index {max(top_feats)} out of bounds for embedding dim {D_dim}."
                )
            D_checked = True

        X_list.append(arr[idx0, top_feats])
        y_list.append(int(r["label"]))
        used_entries.append(entry)

    if missing:
        print(f"⚠️  Missing embeddings for {len(missing)} entries (skipping those).")

    if not X_list:
        raise RuntimeError("No data assembled; check embeddings_dir and CSVs.")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    counts = Counter(y.tolist())
    print(f"✔ Data ready: X={X.shape}, positives={counts.get(1,0)}, negatives={counts.get(0,0)}")

    print("✔ Training final RandomForest on ALL data…")
    clf = RandomForestClassifier(
        n_estimators=trees,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X, y)
    print("✔ Training complete")

    # ── Save artifacts
    final_dir = os.path.join(models_root_template, "final_all_folds")
    os.makedirs(final_dir, exist_ok=True)

    model_path = os.path.join(final_dir, f"rf_final_allfolds_features_{num_features}.joblib")
    joblib.dump(clf, model_path)
    print(f"✔ Saved model → {model_path}")

    meta = {
        "model_type": "RandomForestClassifier",
        "percent_identity": percent_identity,
        "num_features": num_features,
        "feature_indices": top_feats,
        "embeddings_dir": embeddings_dir,
        "pos_path": pos_path,
        "neg_path": neg_path,
        "folds_path": folds_path,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_counts": dict(counts),
        "random_state": random_state,
        "sklearn_version": joblib.__version__,
    }
    meta_path = os.path.join(final_dir, f"rf_final_allfolds_features_{num_features}.meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✔ Saved metadata → {meta_path}")

    entries_path = os.path.join(final_dir, "rf_final_training_entries.txt")
    with open(entries_path, "w") as f:
        f.write("\n".join(sorted(set(used_entries))))
    print(f"✔ Saved training entries → {entries_path}")

    # hygiene
    del X, y, clf
    gc.collect()
    print("All done.")

if __name__ == "__main__":
    main()
