#!/usr/bin/env python3
"""
Compute per-residue probabilities on the secratome using the final CRF model.

Outputs CSV with columns:
  Entry, residue, residue_number, probability
"""

import os
import re
import glob
import csv
import torch
import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
# Ensure CRF class is importable so joblib can unpickle the model:
from sklearn_crfsuite import CRF  # noqa: F401

# ─── Config ────────────────────────────────────────────────────────────────
percent_identity = 40
num_features     = 175  # final CRF uses 175 features

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT  = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
EMB_DIR     = "/Volumes/T7 Shield/layer_33_highest_scoring_novo_embeddings"
FASTA_PATH  = "../../data/novo_smORF_highest_scoring_data.fasta"

ranking_csv = os.path.join(
    DATA_ROOT,
    f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}",
    "feature_ranking_all_folds.csv",
)

model_path = os.path.join(
    DATA_ROOT, "classifier_results", "conditional_random_field_models",
    f"graphpart_{percent_identity}",
    f"final_crf_model_{num_features}_features.joblib",
)

# output
out_dir = "/Volumes/T7 Shield/novo_cleavage_site_predictions"
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, f"secratome_crf_probs_{num_features}.csv")


# ─── Helpers ───────────────────────────────────────────────────────────────
def parse_fasta_to_map(fa_path: str) -> dict[str, str]:
    """Return {Entry accession → sequence} from a UniProt FASTA."""
    seqs = {}
    for rec in SeqIO.parse(fa_path, "fasta"):
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


def pick_positive_label(crf_model) -> str:
    """
    Determine which label is the positive class in the CRF.
    Training used string labels '0' and '1'; prefer '1' if present.
    """
    labels = set(getattr(crf_model, "classes_", []) or [])
    for cand in ("1", 1, "yes", "Y", "POS", "True", True):
        if cand in labels:
            return str(cand)
    # fallback: if '1' not present but '0' is, still return '1' (marginals may still carry it)
    return "1"


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    # 1) Load ranking & select top-N features (to rebuild CRF feature dicts)
    rank_df = pd.read_csv(ranking_csv).sort_values("rank")
    feat_list = rank_df["feature"].astype(int).tolist()
    if len(feat_list) < num_features:
        raise RuntimeError(f"Ranking has only {len(feat_list)} features; need {num_features}.")
    top_feats = feat_list[:num_features]
    print(f"✔ Using top {num_features} features (max index = {max(top_feats)})")

    # 2) Load CRF model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"CRF model not found: {model_path}")
    crf: CRF = joblib.load(model_path)
    pos_label = pick_positive_label(crf)
    print(f"✔ Loaded CRF model from {model_path} (positive label = {pos_label!r})")

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
                missing_seq_entries += 1
                seq = "X" * L
            use_L = min(L, len(seq))
            if use_L != L or use_L != len(seq):
                length_mismatches += 1

            # Build CRF features: one dict per position {"f5": value, ...}
            X_feats = [
                {f"f{i}": float(arr[pos, i]) for i in top_feats}
                for pos in range(use_L)
            ]

            # Per-position marginals: list of dicts {label -> prob}
            marginals = crf.predict_marginals_single(X_feats)

            # Extract P(positive label) per residue
            for i in range(use_L):
                m = marginals[i]
                # prefer exact key; else try string/int conversions; else fallback to max prob
                p = (
                    m.get(pos_label)
                    or m.get(str(pos_label))
                    or m.get(int(pos_label)) if isinstance(pos_label, str) and pos_label.isdigit() else None
                )
                if p is None:
                    # Fallback: if labels are different, take probability of whichever label is not '0'
                    p = m.get("1")
                if p is None:
                    p = max(m.values())  # last-resort fallback to avoid crashes
                writer.writerow([entry, seq[i], i + 1, float(p)])
                total_rows += 1

            # Progress ping every 200 files
            if idx % 200 == 0 or idx == n_files:
                print(f"Processed {idx}/{n_files} embeddings…")

    print(f"✔ Wrote {total_rows} rows → {out_csv}")
    if missing_seq_entries:
        print(f"⚠️  Missing FASTA sequence for {missing_seq_entries} entries (used 'X').")
    if length_mismatches:
        print(f"⚠️  Length mismatches encountered for {length_mismatches} entries (used min length).")


if __name__ == "__main__":
    main()
