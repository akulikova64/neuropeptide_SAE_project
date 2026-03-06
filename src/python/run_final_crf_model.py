#!/usr/bin/env python3
import os
import re
import glob
import torch
import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn_crfsuite import CRF  # ensure importable for joblib.load

# ─── Config ────────────────────────────────────────────────────────────────
percent_identity = 40
num_features     = 175
PRED_THRESH      = 0.5  # probability threshold for predicting class "1"

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT  = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
EMB_DIR    = "/Volumes/T7 Shield/known_peptides_SAE_embeddings"
FASTA_PATH = "/Volumes/T7 Shield/uniprot_known_neuropep_sequences.fasta"

# feature ranking (all folds)
ranking_csv = os.path.join(
    DATA_ROOT,
    f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}",
    "feature_ranking_all_folds.csv"
)

# positives CSV (for ground-truth labels)
POS_CSV = os.path.join("../../data/concept_groups/sequences/group_2/training/group_2_positive_known_peps.csv")

# trained CRF model
model_path = os.path.join(
    DATA_ROOT,
    "classifier_results",
    "conditional_random_field_models",
    f"graphpart_{percent_identity}",
    "final_crf_model_175_features.joblib"
)

# output
out_dir  = os.path.join(
    DATA_ROOT, "classifier_results", "crf_known_peptides", f"graphpart_{percent_identity}"
)
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, "known_peptides_crf_probs_predictions_175.csv")


# ─── Helpers ───────────────────────────────────────────────────────────────
def parse_fasta_to_map(fa_path: str) -> dict[str, str]:
    """Parse FASTA with Biopython and return {Entry: sequence}."""
    seqs = {}
    for rec in SeqIO.parse(fa_path, "fasta"):
        parts = rec.id.split("|")
        entry = parts[1] if len(parts) >= 3 else rec.id
        seqs[entry] = str(rec.seq)
    return seqs

def entry_from_embedding_path(path: str) -> str | None:
    """Extract UniProt accession between '|' in filenames like sp|P01178|NEU1_HUMAN_original_SAE.pt"""
    parts = os.path.basename(path).split("|")
    return parts[1] if len(parts) >= 3 else None

def build_feature_dict_sequence(arr: np.ndarray, top_feats: list[int]) -> list[dict]:
    """Convert (L, D) array to list of dicts [{f5: val, f23: val, ...}, ...] using provided feature indices."""
    L, D = arr.shape
    if max(top_feats) >= D:
        raise ValueError(f"Top feature index {max(top_feats)} exceeds embedding dim {D}")
    return [{f"f{i}": float(arr[pos, i]) for i in top_feats} for pos in range(L)]

def main():
    # 0) Load ground-truth positives
    pos_df = pd.read_csv(POS_CSV)
    pos_df["residue_number"] = pos_df["residue_number"].astype(int)
    pos_sites_map = (
        pos_df.groupby("Entry")["residue_number"]
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )

    # 1) Load model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"CRF model not found: {model_path}")
    crf: CRF = joblib.load(model_path)
    print(f"✔ Loaded CRF model from {model_path}")

    # 2) Load ranking and grab top 175 features
    rank_df = pd.read_csv(ranking_csv).sort_values("rank")
    feat_list = rank_df["feature"].astype(int).tolist()
    if len(feat_list) < num_features:
        raise RuntimeError(f"Ranking has only {len(feat_list)} features (need {num_features}).")
    top_feats = feat_list[:num_features]
    print(f"✔ Using top {num_features} features; max feature index = {max(top_feats)}")

    # 3) Load FASTA sequences → {Entry: seq}
    entry_to_seq = parse_fasta_to_map(FASTA_PATH)
    print(f"✔ Loaded {len(entry_to_seq)} sequences from FASTA")

    # 4) Iterate embeddings and score
    rows = []
    total_correct = 0
    total_count   = 0

    emb_paths = sorted(glob.glob(os.path.join(EMB_DIR, "*.pt")))
    if not emb_paths:
        raise RuntimeError(f"No .pt files found in {EMB_DIR}")

    for emb_fp in emb_paths:
        entry = entry_from_embedding_path(emb_fp)
        if not entry:
            print(f"⚠️  Could not parse Entry from filename: {emb_fp}; skipping.")
            continue
        if not os.path.isfile(emb_fp):
            continue

        arr = torch.load(emb_fp, map_location="cpu").numpy()  # shape (L, D)
        L, D = arr.shape

        X_seq = build_feature_dict_sequence(arr, top_feats)

        # Per-position probability for class "1"
        marginals = crf.predict_marginals_single(X_seq)

        # 👇 Viterbi (“raw CRF”) class predictions per position
        try:
            viterbi_labels = crf.predict_single(X_seq)          # list like ['0','0','1',...]
        except AttributeError:
            viterbi_labels = crf.predict([X_seq])[0]
        viterbi_int = [int(v) for v in viterbi_labels]          # convert to 0/1 ints

        seq = entry_to_seq.get(entry, None)
        if seq is None:
            print(f"⚠️  No FASTA sequence for {entry}; residues will be 'X'.")
            seq = "X" * L

        if len(seq) != L or len(marginals) != L or len(viterbi_int) != L:
            print(f"⚠️  Length mismatch for {entry}: FASTA={len(seq)} EMB={L} "
                  f"MARG={len(marginals)} VIT={len(viterbi_int)}. Using min length.")
        use_L = min(L, len(seq), len(marginals), len(viterbi_int))

        # Ground-truth positives for this entry (others are negatives)
        pos_set = pos_sites_map.get(entry, set())

        for i in range(use_L):
            prob_pos = float(marginals[i].get("1", 0.0))
            residue  = seq[i] if i < len(seq) else "X"
            pred_cls = viterbi_int[i]                           # 👈 raw CRF prediction (0/1)

            rows.append({
                "Entry": entry,
                "position": i + 1,          # 1-based
                "residue": residue,
                "probability": prob_pos,
                "predictions": pred_cls     # 👈 new column
            })

            # Accuracy accounting (still thresholding marginals at PRED_THRESH)
            y_true = 1 if (i + 1) in pos_set else 0
            y_pred = 1 if prob_pos >= PRED_THRESH else 0
            total_correct += int(y_true == y_pred)
            total_count   += 1

    # 5) Save CSV (now includes "predictions")
    out_df = pd.DataFrame(rows, columns=["Entry", "position", "residue", "probability", "predictions"])
    out_df.to_csv(out_csv, index=False)
    print(f"✔ Saved probabilities + predictions for {len(out_df)} positions → {out_csv}")

    # 6) Print overall accuracy (thresholded marginals)
    if total_count > 0:
        acc = total_correct / total_count * 100.0
        print(f"✔ Overall accuracy @ threshold={PRED_THRESH:.2f}: {acc:.2f}%  "
              f"({total_correct}/{total_count} correct)")
    else:
        print("⚠️  No positions evaluated; accuracy not computed.")

if __name__ == "__main__":
    main()
