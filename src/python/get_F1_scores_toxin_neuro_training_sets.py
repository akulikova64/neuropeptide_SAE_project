#!/usr/bin/env python3
import os
import gc
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score

# ─── CONFIGURATION & PATHS ────────────────────────────────────────────────
embeddings_dir  = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/toxin_neuro_SAE_embeddings"
base_input_dir  = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/input_data/training_data"
base_output_dir = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/F1_scores_hybrid"

homology    = [80, 90]  # or [40,50,60,70,80,90]
thresholds  = [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8]

for hom in homology:
    print("Processing homology:", hom)
    # ─── Paths ──────────────────────────────────────────────────────────────
    pos_path_single = os.path.join(
        base_input_dir, f"homology_group{hom}",
        "group_2_positive_toxin_neuro_train.csv"
    )
    pos_domain_path = os.path.join(
        base_input_dir, f"homology_group{hom}",
        f"group_2_positive_neuro_toxin_domain_{hom}.csv"
    )
    neg_path = os.path.join(
        base_input_dir, f"homology_group{hom}",
        "group_2_negative_toxin_neuro_train.csv"
    )
    output_dir = os.path.join(base_output_dir, f"homology_group{hom}")
    os.makedirs(output_dir, exist_ok=True)
    print("Read in all data paths")

    # ─── Read in labels ──────────────────────────────────────────────────────
    pos_df       = pd.read_csv(pos_path_single).assign(label=1)
    neg_df       = pd.read_csv(neg_path).assign(label=0)
    labels_df    = pd.concat([pos_df, neg_df], ignore_index=True)
    pos_domain_df = pd.read_csv(pos_domain_path)
    print("Processed all pos and neg data and labels")
    print("Collecting embeddings")

    # ─── Collect embeddings ─────────────────────────────────────────────────
    features, labs, ents, resnums = [], [], [], []
    missing = set()
    for _, row in labels_df.iterrows():
        entry     = row["Entry"]
        idx0      = int(row["residue_number"]) - 1
        emb_file  = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")
        if not os.path.isfile(emb_file):
            missing.add(entry)
            continue
        arr = torch.load(emb_file, map_location="cpu").numpy()
        features.append(arr[idx0, :])
        labs.append(row["label"])
        ents.append(entry)
        resnums.append(int(row["residue_number"]))
    if missing:
        print("Missing embeddings for entries:", missing)

    # to arrays
    X            = np.vstack(features)
    y            = np.array(labs)
    entries_arr  = np.array(ents)
    residues_arr = np.array(resnums)
    num_features = X.shape[1]
    n_domains    = pos_domain_df["domain_id"].nunique()

    print("Beginning to parse thresholds")

    # ─── Precompute merged domain mappings with orig_idx ───────────────────
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
    # now merged_df has columns: Entry, residue_number, orig_idx, domain_id
    unique_domains = merged_df["domain_id"].unique()

    # ─── Loop over thresholds ───────────────────────────────────────────────
    for thr in thresholds:
        print("Processing threshold:", thr, "homology:", hom)
        # 1) binary prediction matrix
        B = (X > thr).astype(int)  # shape (n_samples, n_features)

        # 2) compute residue-level precision for each feature
        precisions = [
            precision_score(y, B[:, f], zero_division=0)
            for f in range(num_features)
        ]

        records = []
        # 3) compute domain-level recall & hybrid F1
        for feat_idx, prec in enumerate(precisions):
            # assign predictions back to merged rows via orig_idx
            merged_df["prediction"] = B[merged_df["orig_idx"], feat_idx]
            # max across overlapping residues per domain
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

        # 4) save to CSV
        df_out = pd.DataFrame(records)
        out_fp = os.path.join(
            output_dir,
            f"threshold_{str(thr).replace('.', '_')}.csv"
        )
        df_out.to_csv(out_fp, index=False)
        print(f"Saved hybrid F1 at threshold={thr} to {out_fp}")

        # 5) free memory
        del B, records, df_out
        gc.collect()

    print("Done homology", hom)
