# here I will train the final crf on all folds (all 10) and use a specific optimum 
# number of feautures and generate a "final" crf model. 

#!/usr/bin/env python3
import os
import gc
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn_crfsuite import CRF

# ─── Config ────────────────────────────────────────────────────────────────
percent_identity = 40
num_features     = 175
max_iter         = 20

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT  = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
EMB_ROOT   = os.path.join(DATA_ROOT, "toxin_neuro_SAE_embeddings")
POS_CSV    = os.path.join(DATA_ROOT, "input_data", "group_2_positive_toxin_neuro.csv")
FOLDS_ROOT = os.path.join(DATA_ROOT, "input_data", f"folds_{percent_identity}")

ranking_csv = os.path.join(
    DATA_ROOT,
    f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}",
    "feature_ranking_all_folds.csv",
)

models_root = os.path.join(
    DATA_ROOT,
    "classifier_results",
    "conditional_random_field_models",
    f"graphpart_{percent_identity}",
)
os.makedirs(models_root, exist_ok=True)
model_out = os.path.join(models_root, "final_crf_model_175_features.joblib")

def main():
    # 1) Load ranking and select top 175 features
    rank_df = pd.read_csv(ranking_csv).sort_values("rank")
    feat_list = rank_df["feature"].astype(int).tolist()
    if len(feat_list) < num_features:
        raise RuntimeError(f"Ranking has only {len(feat_list)} features, need {num_features}.")
    top_feats = feat_list[:num_features]
    print(f"✔ Using top {num_features} features (max index = {max(top_feats)})")

    # 2) Combine all entries from fold_0..fold_9
    allowed = set()
    for k in range(10):
        fp = os.path.join(FOLDS_ROOT, f"fold_{k}.txt")
        if not os.path.isfile(fp):
            print(f"⚠️ Missing fold file: {fp} (skipping)")
            continue
        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
            allowed.update(fh.read().split())
    if not allowed:
        raise RuntimeError(f"No entries found in folds under {FOLDS_ROOT}")
    print(f"✔ Combined {len(allowed)} unique entries from folds")

    # 3) Load positives; convert residue_number to int
    pos_df = pd.read_csv(POS_CSV)
    pos_df["residue_number"] = pos_df["residue_number"].astype(int)

    # Only train on entries that appear in positives (so each sequence has some 1s)
    pos_entries = set(pos_df["Entry"])
    target_entries = sorted(allowed.intersection(pos_entries))
    if not target_entries:
        raise RuntimeError("No overlap between fold entries and positive entries.")
    print(f"✔ Training on {len(target_entries)} entries that have positives")

    # 4) Build sequences (features + labels) per entry
    seq_dict = {}
    dim_check_done = False
    for entry in target_entries:
        emb_fp = os.path.join(EMB_ROOT, f"{entry}_original_SAE.pt")
        if not os.path.isfile(emb_fp):
            continue
        arr = torch.load(emb_fp, map_location="cpu").numpy()  # (L, D)
        L, D = arr.shape

        if not dim_check_done:
            if max(top_feats) >= D:
                raise RuntimeError(
                    f"Feature index {max(top_feats)} out of bounds for embedding dim {D}."
                )
            dim_check_done = True

        pos_sites = set(pos_df.loc[pos_df["Entry"] == entry, "residue_number"])
        y_seq = [str(1 if (i + 1) in pos_sites else 0) for i in range(L)]

        # features as dict per position: {"f5": value, ...}
        seq_feats = [
            {f"f{i}": float(arr[pos, i]) for i in top_feats}
            for pos in range(L)
        ]

        seq_dict[entry] = {"X": seq_feats, "y": y_seq}

    if not seq_dict:
        raise RuntimeError("No sequences were built (check embeddings and positives overlap).")

    entries = list(seq_dict.keys())
    X_seqs  = [seq_dict[e]["X"] for e in entries]
    Y_seqs  = [seq_dict[e]["y"] for e in entries]
    n_tokens = sum(len(x) for x in X_seqs)

    print(f"✔ Data ready: {len(entries)} sequences, {n_tokens} tokens total")

    # 5) Train a single CRF on ALL sequences
    crf = CRF(
        algorithm="lbfgs",
        max_iterations=max_iter,
        all_possible_transitions=True,
        # Optional: set c1/c2 or random_state for reproducibility
        # c1=0.0, c2=0.0, random_state=42
    )
    crf.fit(X_seqs, Y_seqs)
    print("✔ CRF training complete")

    # 6) Save the final model
    joblib.dump(crf, model_out)
    print(f"✔ Saved model → {model_out}")

    # memory hygiene
    del X_seqs, Y_seqs, seq_dict
    gc.collect()

if __name__ == "__main__":
    main()
