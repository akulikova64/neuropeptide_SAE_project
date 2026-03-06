#!/usr/bin/env python3
#!/usr/bin/env python3
import os
import gc
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score

# ─── Define test folds correctly ─────────────────────────────────────────────
test_folds = list(range(10))              
percent_identity = 40
run = 2 # this is the finetuned model id

for test_fold in test_folds:
    # ─── CONFIGURATION & PATHS ────────────────────────────────────────────────
    embeddings_dir  = f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/toxin_neuro_finetuned_SAE_embeddings/run_{run}/"
    base_input_dir  = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/"
    base_output_dir = (
        f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/finetuned_SAE_analysis/"
        f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}/run_{run}/excluded_fold_{test_fold}"
    )
    folds_dir = f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/folds_{percent_identity}"
    os.makedirs(base_output_dir, exist_ok=True)

    thresholds  = [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8]

    # ─── Paths ──────────────────────────────────────────────────────────────
    pos_path_single = os.path.join(base_input_dir, "group_2_positive_toxin_neuro.csv")
    pos_domain_path = os.path.join(base_input_dir, "group_2_positive_toxin_neuro_domain.csv")
    neg_path        = os.path.join(base_input_dir, "group_2_negative_toxin_neuro.csv")
    output_dir      = base_output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Excluding fold {test_fold}: reading data …")

    # ─── Read in labels ──────────────────────────────────────────────────────
    pos_df        = pd.read_csv(pos_path_single).assign(label=1)
    neg_df        = pd.read_csv(neg_path).assign(label=0)
    labels_df     = pd.concat([pos_df, neg_df], ignore_index=True)
    pos_domain_df = pd.read_csv(pos_domain_path)

    # ─── Build list of folds to TRAIN on ────────────────────────────────────
    all_folds = list(range(10))                                   # same fix
    non_excluded_folds = [f for f in all_folds if f != test_fold]  # no .remove()

    # ─── FILTER TO ALLOWED ENTRIES ────────────────────────────────────────
    allowed = set()
    for fold in non_excluded_folds:
        fold_path = os.path.join(folds_dir, f"fold_{fold}.txt")
        with open(fold_path, "r") as fold_file:
            allowed.update(fold_file.read().split())

    # Sanity check: ensure no excluded‑fold entries slipped in
    excluded_fold_path = os.path.join(folds_dir, f"fold_{test_fold}.txt")
    with open(excluded_fold_path, "r") as excluded_fold_file:
        excluded_entries = set(excluded_fold_file.read().split())
    assert allowed.isdisjoint(excluded_entries), (
        f"❌ Leakage: fold {test_fold} entries found in allowed set!"
    )

    # Subset labels_df to only the allowed entries
    labels_df = labels_df[labels_df["Entry"].isin(allowed)].reset_index(drop=True)
    print(f"✔ {len(labels_df)} total positions after excluding fold {test_fold}")

    # ─── Collect embeddings ─────────────────────────────────────────────────
    features, labs, ents, resnums = [], [], [], []
    missing = set()
    for _, row in labels_df.iterrows():
        entry    = row["Entry"]
        idx0     = int(row["residue_number"]) - 1
        emb_file = os.path.join(embeddings_dir, f"{entry}.pt")
        if not os.path.isfile(emb_file):
            missing.add(entry)
            continue
        arr = torch.load(emb_file, map_location="cpu").numpy()
        features.append(arr[idx0, :])
        labs.append(row["label"])
        ents.append(entry)
        resnums.append(int(row["residue_number"]))
    if missing:
        print("⚠️ Missing embeddings for entries:", missing)

    # to arrays
    X            = np.vstack(features)
    y            = np.array(labs)
    entries_arr  = np.array(ents)
    residues_arr = np.array(resnums)
    num_features = X.shape[1]

    print("Beginning threshold loop …")

    # ─── Precompute domain mapping ───────────────────────────────────────────
    mask_pos = (y == 1)
    orig_idx = np.nonzero(mask_pos)[0]
    pos_df_sm = pd.DataFrame({
        "Entry":          entries_arr[mask_pos],
        "residue_number": residues_arr[mask_pos],
        "orig_idx":       orig_idx
    })
    merged_df = pos_df_sm.merge(
        pos_domain_df,
        on=["Entry", "residue_number"],
        how="left"
    )

    # ─── Loop over thresholds ───────────────────────────────────────────────
    for thr in thresholds:
        print(f"  • threshold = {thr}")
        B = (X > thr).astype(int)  # binary predictions

        # residue‑level precision
        precisions = [
            precision_score(y, B[:, f], zero_division=0)
            for f in range(num_features)
        ]

        records = []
        for feat_idx, prec in enumerate(precisions):
            merged_df["prediction"] = B[merged_df["orig_idx"], feat_idx]
            dom_pred = merged_df.groupby("domain_id")["prediction"].max().to_numpy()
            recall   = dom_pred.sum() / len(dom_pred) if dom_pred.size else 0.0
            f1       = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
            records.append({
                "feature":   feat_idx,
                "threshold": thr,
                "precision": prec,
                "recall":    recall,
                "F1_score":  f1
            })

        # save CSV
        df_out = pd.DataFrame(records)
        out_path = os.path.join(output_dir, f"threshold_{str(thr).replace('.', '_')}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"    ✅ saved {out_path}")

        # free memory
        del B, records, df_out
        gc.collect()

    print(f"✔ Done F1 scores for all folds excluding test_folds!")
