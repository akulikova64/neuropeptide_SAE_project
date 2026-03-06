# the following script was used to find the probabilities for each residue in the secratome. 
# the probabilities are for each residue and are the prob that the residue is a cleavage site in 
# neuropeptides (signaling peptides)

#!/usr/bin/env python3
import os
import re
import glob
import csv
import torch
import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.linear_model import LogisticRegression  # needed so joblib can load

# ─── Config ────────────────────────────────────────────────────────────────
percent_identity = 40
num_features     = 70  # from your elbow result

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT   = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
#EMB_DIR     = "/Volumes/T7 Shield/secratome_analysis/secratome_SAE_embeddings"
#FASTA_PATH  = "../../data/uniprotkb_taxonomy_id_9606_AND_keyword_2025_08_20.fasta"
EMB_DIR     = "/Volumes/T7 Shield/layer_33_highest_scoring_novo_embeddings"
FASTA_PATH  = "../../data/novo_smORF_highest_scoring_data.fasta"

ranking_csv = os.path.join(
    DATA_ROOT,
    f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}",
    "feature_ranking_all_folds.csv"
)

# trained logistic regression (already fit on all 10 folds, 70 features)
model_path = os.path.join(
    DATA_ROOT, "classifier_results", "logistic_regression_models",
    f"graphpart_{percent_identity}",
    f"final_logreg_model_{num_features}_features.joblib"
)

# output
out_dir = "/Volumes/T7 Shield/novo_cleavage_site_predictions"
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, f"secratome_logreg_probs_{num_features}.csv")

# ─── Helpers ───────────────────────────────────────────────────────────────
def parse_fasta_to_map(fa_path: str) -> dict[str, str]:
    """Return {Entry accession → sequence} from a UniProt FASTA."""
    seqs = {}
    for rec in SeqIO.parse(fa_path, "fasta"):
        # header id like: "sp|P12345|NAME_HUMAN" or "tr|A0A0...|..."
        parts = rec.id.split("|")
        entry = parts[1] if len(parts) >= 3 else rec.id
        seqs[entry] = str(rec.seq)
    return seqs

def entry_from_embedding_path(path: str) -> str | None:
    """
    Extract the UniProt accession from an embedding filename.
    Handles both:
      - 'sp|P01178|NEU1_HUMAN_original_SAE.pt' (take parts[1])
      - 'A0A024RBI1_original_SAE.pt' (regex fallback)
    """
    base = os.path.basename(path)
    if "|" in base:
        parts = base.split("|")
        return parts[1] if len(parts) >= 2 else None
    m = re.match(r"([A-Za-z0-9]+)_original_SAE\.pt$", base)
    return m.group(1) if m else None

def main():
    # 1) Load ranking & select top-N features
    rank_df = pd.read_csv(ranking_csv).sort_values("rank")
    feat_list = rank_df["feature"].astype(int).tolist()
    if len(feat_list) < num_features:
        raise RuntimeError(f"Ranking has only {len(feat_list)} features; need {num_features}.")
    top_feats = feat_list[:num_features]
    print(f"✔ Using top {num_features} features (max index = {max(top_feats)})")

    # 2) Load model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    clf: LogisticRegression = joblib.load(model_path)
    print(f"✔ Loaded logistic regression model from {model_path}")

    # 3) Load FASTA sequences to map Entry→sequence
    entry_to_seq = parse_fasta_to_map(FASTA_PATH)
    print(f"✔ Loaded {len(entry_to_seq)} sequences from FASTA")

    # 4) Iterate over secratome embeddings and write rows incrementally
    emb_paths = sorted(glob.glob(os.path.join(EMB_DIR, "*.pt")))
    if not emb_paths:
        raise RuntimeError(f"No .pt files found under {EMB_DIR}")

    n_files = len(emb_paths)
    total_rows = 0
    missing_seq_entries = 0
    length_mismatches = 0

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Entry", "residue", "residue_number", "probability"])  # header

        for idx, emb_fp in enumerate(emb_paths, 1):
            entry = entry_from_embedding_path(emb_fp)
            if not entry:
                print(f"⚠️  Could not parse Entry from {emb_fp}; skipping.")
                continue

            arr = torch.load(emb_fp, map_location="cpu").numpy()  # (L, D)
            if arr.ndim != 2:
                print(f"⚠️  Bad shape for {entry}: {arr.shape}; skipping.")
                continue
            L, D = arr.shape
            if max(top_feats) >= D:
                print(f"⚠️  Feature index {max(top_feats)} out of bounds for {entry} (D={D}); skipping.")
                continue

            seq = entry_to_seq.get(entry)
            if seq is None:
                # Not expected if embeddings were generated from this FASTA, but handle gracefully
                missing_seq_entries += 1
                seq = "X" * L
            use_L = min(L, len(seq))
            if use_L != L or use_L != len(seq):
                length_mismatches += 1

            # Slice features and get probabilities P(y=1)
            X_seq = arr[:use_L, top_feats]  # (use_L, num_features)
            # predict_proba returns (N, 2): [:,1] is P(class=1)
            probs = clf.predict_proba(X_seq)[:, 1]

            # Write rows
            for i in range(use_L):
                writer.writerow([entry, seq[i], i + 1, float(probs[i])])
            total_rows += use_L

            # Progress ping every 500 files
            if idx % 500 == 0 or idx == n_files:
                print(f"Processed {idx}/{n_files} embeddings…")

    print(f"✔ Wrote {total_rows} rows → {out_csv}")
    if missing_seq_entries:
        print(f"⚠️  Missing FASTA sequence for {missing_seq_entries} entries (used 'X').")
    if length_mismatches:
        print(f"⚠️  Length mismatches encountered for {length_mismatches} entries (used min length).")

if __name__ == "__main__":
    main()
