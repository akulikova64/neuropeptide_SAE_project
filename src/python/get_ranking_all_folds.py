#!/usr/bin/env python3
import os
import gc
import re
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score

# ─── Configuration ─────────────────────────────────────────────────────────
percent_identity       = 40
thresholds             = [0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.8]
number_of_top_features = 2000
run = 6

# ─── Paths ─────────────────────────────────────────────────────────────────
DATA_ROOT      = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/"
#embeddings_dir = os.path.join(DATA_ROOT, "toxin_neuro_SAE_embeddings")
embeddings_dir = f"/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro/toxin_neuro_finetuned_SAE_embeddings/run_{run}/"
base_input_dir = os.path.join(DATA_ROOT, "input_data")
folds_dir      = os.path.join(base_input_dir, f"folds_{percent_identity}")

# labels
pos_path_single = os.path.join(base_input_dir, "group_2_positive_toxin_neuro.csv")
pos_domain_path = os.path.join(base_input_dir, "group_2_positive_toxin_neuro_domain.csv")
neg_path        = os.path.join(base_input_dir, "group_2_negative_toxin_neuro.csv")

# outputs
base_dir   = os.path.join(
    DATA_ROOT,
    "finetuned_SAE_analysis",
    f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}"
)
all_out_dir = os.path.join(base_dir, "all_folds") # making a folder specifically for folds.
os.makedirs(all_out_dir, exist_ok=True)

# final combined ranking file (exact path/name as requested)
ranking_csv = os.path.join(base_dir, f"feature_ranking_all_folds_run_{run}.csv")

def thr_to_name(t: float) -> str:
    """Convert a float threshold to a filename-friendly piece, e.g. 0.15 -> '0_15'."""
    s = f"{t:.2f}".rstrip('0').rstrip('.')  # 0.50→0.5, 0.60→0.6, 0.00→0
    return s.replace('.', '_')

def main():
    # ─── 1) Load all fold files and combine into a single list of entries ───
    fold_files = [f"fold_{i}.txt" for i in range(10)]
    allowed_entries = []
    missing_folds = []

    for fn in fold_files:
        path = os.path.join(folds_dir, fn)
        if not os.path.isfile(path):
            missing_folds.append(path)
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            allowed_entries.extend(fh.read().split())

    allowed = sorted(set(allowed_entries))
    if not allowed:
        raise RuntimeError(f"No entries found from folds under: {folds_dir}")

    # Save the combined entries file (space-separated)
    combined_entries_path = os.path.join(all_out_dir, "combined_entries.txt")
    with open(combined_entries_path, "w") as fh:
        fh.write(" ".join(allowed))
    print(f"✔ Combined {len(allowed)} unique entries → {combined_entries_path}")
    if missing_folds:
        print("⚠️ Missing fold files:")
        for m in missing_folds:
            print("   -", m)

    # ─── 2) Load labels and filter to allowed entries ───────────────────────
    pos_df        = pd.read_csv(pos_path_single).assign(label=1)
    neg_df        = pd.read_csv(neg_path).assign(label=0)
    labels_df     = pd.concat([pos_df, neg_df], ignore_index=True)
    pos_domain_df = pd.read_csv(pos_domain_path)

    labels_df = labels_df[labels_df["Entry"].isin(allowed)].reset_index(drop=True)
    print(f"✔ {len(labels_df)} total positions after filtering to all folds")

    # ─── 3) Load embeddings for those entries/positions ─────────────────────
    features, labs, ents, resnums = [], [], [], []
    missing_emb = set()

    for _, row in labels_df.iterrows():
        entry = row["Entry"]
        idx0  = int(row["residue_number"]) - 1  # 0-based
        emb_file = os.path.join(embeddings_dir, f"{entry}.pt") # {entry}_original_SAE.pt
        if not os.path.isfile(emb_file):
            missing_emb.add(entry)
            continue
        arr = torch.load(emb_file, map_location="cpu").numpy()  # (L, D)
        if idx0 < 0 or idx0 >= arr.shape[0]:
            continue  # out-of-range guard
        features.append(arr[idx0, :])
        labs.append(row["label"])
        ents.append(entry)
        resnums.append(int(row["residue_number"]))

    if missing_emb:
        print(f"⚠️ Missing embeddings for {len(missing_emb)} entries (skipped).")

    # to arrays
    X            = np.vstack(features)
    y            = np.array(labs)
    entries_arr  = np.array(ents)
    residues_arr = np.array(resnums)
    num_features = X.shape[1]
    print(f"✔ Built matrices: X={X.shape}, y={y.shape}, features={num_features}")

    # ─── Precompute domain mapping for positive residues ────────────────────
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
    if merged_df["domain_id"].isna().any():
        n_nan = merged_df["domain_id"].isna().sum()
        print(f"⚠️ {n_nan} positive residues missing domain_id; recall for them may be undercounted.")

    # ─── 4) Compute and SAVE F1 by threshold (per-feature) ──────────────────
    # Also accumulate a big list for the “winning” step.
    all_threshold_rows = []

    print("→ Computing hybrid F1 across thresholds …")
    for thr in thresholds:
        print(f"  • threshold = {thr}")
        thr_name = thr_to_name(thr)

        # binarize features by threshold
        B = (X > thr).astype(int)   # shape (N, num_features)

        # residue-level precision per feature
        precisions = [
            precision_score(y, B[:, f], zero_division=0)
            for f in range(num_features)
        ]

        # domain-level recall + F1
        rows = []
        for feat_idx, prec in enumerate(precisions):
            # For recall, only look at positive residues (merged_df rows)
            merged_df["prediction"] = B[merged_df["orig_idx"], feat_idx]
            # domain is positive if ANY residue in that domain predicted positive
            dom_pred = merged_df.groupby("domain_id")["prediction"].max().to_numpy()
            recall   = dom_pred.sum() / len(dom_pred) if dom_pred.size else 0.0
            f1       = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

            rows.append({
                "feature":   feat_idx,
                "threshold": thr,
                "precision": prec,
                "recall":    recall,
                "F1_score":  f1
            })

        # Save per-threshold CSV
        df_thr = pd.DataFrame(rows)
        out_path = os.path.join(all_out_dir, f"threshold_{thr_name}.csv")
        df_thr.to_csv(out_path, index=False)
        print(f"    ✓ saved {out_path}")

        # accumulate
        all_threshold_rows.extend(rows)

        # memory hygiene
        del B, rows, df_thr
        gc.collect()

    # ─── 5) Winning F1 per feature (best threshold), save CSV ───────────────
    all_df = pd.DataFrame(all_threshold_rows)
    if all_df.empty:
        raise RuntimeError("No threshold rows collected; nothing to rank.")

    # For each feature, pick the threshold with max F1_score
    best_idx = all_df.groupby("feature")["F1_score"].idxmax()
    best_df  = all_df.loc[best_idx, ["feature", "threshold", "F1_score"]].copy()
    best_df.rename(columns={
        "threshold": "winning_threshold",
        "F1_score":  "winning_F1_score"
    }, inplace=True)

    winning_csv = os.path.join(all_out_dir, "winning_thresholds_all_folds.csv")
    best_df.to_csv(winning_csv, index=False)
    print(f"✔ Wrote winning thresholds for {len(best_df)} features → {winning_csv}")

    # ─── 6) Build and SAVE the ranking file (top N) ─────────────────────────
    f1_sorted = best_df.sort_values("winning_F1_score", ascending=False).head(number_of_top_features)
    f1_sorted = f1_sorted.reset_index(drop=True)
    f1_sorted["rank"] = f1_sorted.index + 1

    out_df = f1_sorted[["feature", "rank"]]
    os.makedirs(base_dir, exist_ok=True)
    out_df.to_csv(ranking_csv, index=False)
    print(f"✔ Saved ranking ({out_df.shape[0]} features) → {ranking_csv}")

if __name__ == "__main__":
    main()
