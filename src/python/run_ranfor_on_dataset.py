#!/usr/bin/env python3
"""
Compute per-residue probabilities on the secratome using the final Random Forest model.

Output CSV columns:
  Entry, residue, residue_number, probability
"""

import os
import re
import sys
import glob
import csv
import warnings
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from Bio import SeqIO

# Silence sklearn cross-version warning if versions differ slightly
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# ─── Config ────────────────────────────────────────────────────────────────
percent_identity = 40
num_features     = 70  # RF final features

# ─── Paths ────────────────────────────────────────────────────────────────
DATA_ROOT  = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
EMB_DIR     = "/Volumes/T7 Shield/layer_33_highest_scoring_novo_embeddings"
FASTA_PATH  = "../../data/novo_smORF_highest_scoring_data.fasta"

RANKING_ALL_FOLDS = os.path.join(
    DATA_ROOT,
    f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}",
    "feature_ranking_all_folds.csv",
)

MODEL_PATH = "/Volumes/T7 Shield/random_forest_chpc/rf_final_allfolds_features_70.joblib"

# Output
OUT_DIR = os.path.join(
    DATA_ROOT, "classifier_results", "ranfor_secratome", f"graphpart_{percent_identity}"
)
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, f"secratome_ranfor_probs_{num_features}.csv")


# ─── Runtime compatibility helper (optional auto-venv) ─────────────────────
def _ver_tuple(s: str):
    import re as _re
    parts = _re.findall(r"\d+", s)
    return tuple(int(x) for x in parts[:3]) or (0,)

def ensure_compatible_runtime():
    """
    The RF pickle was trained on newer libs. Ensure NumPy>=2.0 and scikit-learn==1.7.x.
    If RF_BOOTSTRAP=1, create ~/.cache/rf_infer_env and re-exec inside it.
    """
    import importlib.metadata as im
    np_ok = _ver_tuple(im.version("numpy")) >= (2, 0, 0)
    skv   = _ver_tuple(im.version("scikit-learn"))
    sk_ok = (1, 7, 0) <= skv < (1, 8, 0)

    if np_ok and sk_ok:
        return

    if os.environ.get("RF_BOOTSTRAP", "0") != "1":
        print("✖ Environment incompatible for loading this model.")
        print(f"  Detected: numpy={im.version('numpy')} scikit-learn={im.version('scikit-learn')}")
        print("  Needs:    numpy>=2.0  and  scikit-learn==1.7.x")
        print("\nTwo options:\n"
              "A) conda create -n rf_infer python=3.11 numpy=2.0.* scikit-learn=1.7.1 joblib pandas biopython pytorch -c conda-forge\n"
              "   conda activate rf_infer\n"
              "   python run_rf_secratome.py\n\n"
              "B) Auto-create a small venv & re-run:\n"
              "   RF_BOOTSTRAP=1 python run_rf_secratome.py\n")
        sys.exit(2)

    cache_dir = Path.home() / ".cache" / "rf_infer_env"
    pybin = cache_dir / "bin" / "python"
    if not pybin.exists():
        print("ℹ️  Creating local venv with compatible NumPy/Sklearn …")
        subprocess.check_call([sys.executable, "-m", "venv", str(cache_dir)])
        pip = str(cache_dir / "bin" / "pip")
        subprocess.check_call([pip, "install", "--upgrade", "pip", "wheel", "setuptools"])
        subprocess.check_call([pip, "install",
                               "numpy>=2.0,<3.0",
                               "scikit-learn==1.7.1",
                               "joblib",
                               "pandas",
                               "biopython",
                               "torch"])
    print("✔ Re-executing in the compatible venv …")
    os.execv(str(pybin), [str(pybin), *sys.argv])


# ─── Helpers ───────────────────────────────────────────────────────────────
def parse_fasta_to_map(fa_path: str) -> dict:
    """Return {Entry accession → sequence} from a UniProt FASTA."""
    seqs = {}
    for rec in SeqIO.parse(fa_path, "fasta"):
        parts = rec.id.split("|")
        entry = parts[1] if len(parts) >= 3 else rec.id
        seqs[entry] = str(rec.seq)
    return seqs

def entry_from_embedding_path(path: str) -> str | None:
    """
    Extract UniProt accession from:
      - 'sp|P01178|NEU1_HUMAN_original_SAE.pt'  → P01178
      - 'A0A024RBI1_original_SAE.pt'            → A0A024RBI1
    """
    base = os.path.basename(path)
    if "|" in base:
        parts = base.split("|")
        return parts[1] if len(parts) >= 2 else None
    m = re.match(r"([A-Za-z0-9]+)_original_SAE\.pt$", base)
    return m.group(1) if m else None


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    ensure_compatible_runtime()

    # 1) Load ranking & select top-N features
    if not os.path.isfile(RANKING_ALL_FOLDS):
        raise FileNotFoundError(f"Ranking file not found: {RANKING_ALL_FOLDS}")
    rank_df = pd.read_csv(RANKING_ALL_FOLDS).sort_values("rank")
    feat_list = rank_df["feature"].astype(int).tolist()
    if len(feat_list) < num_features:
        raise RuntimeError(f"Ranking has only {len(feat_list)} features; need {num_features}.")
    top_feats = feat_list[:num_features]
    print(f"✔ Using top {num_features} features (max index = {max(top_feats)})")

    # 2) Load RF model
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)
    print(f"✔ Loaded Random Forest model from {MODEL_PATH}")

    # 3) FASTA map
    if not os.path.isfile(FASTA_PATH):
        raise FileNotFoundError(f"FASTA not found: {FASTA_PATH}")
    entry_to_seq = parse_fasta_to_map(FASTA_PATH)
    print(f"✔ Loaded {len(entry_to_seq)} sequences from FASTA")

    # 4) Iterate embeddings and write CSV
    emb_paths = sorted(glob.glob(os.path.join(EMB_DIR, "*.pt")))
    if not emb_paths:
        raise RuntimeError(f"No .pt files found under {EMB_DIR}")

    n_files = len(emb_paths)
    total_rows = 0
    missing_seq_entries = 0
    length_mismatches = 0

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Entry", "residue", "residue_number", "probability"])

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

            X_seq = arr[:use_L, top_feats]  # (use_L, num_features)

            # RF supports predict_proba → take class-1 column
            probs = clf.predict_proba(X_seq)[:, 1]

            for i in range(use_L):
                writer.writerow([entry, seq[i], i + 1, float(probs[i])])
            total_rows += use_L

            if idx % 500 == 0 or idx == n_files:
                print(f"Processed {idx}/{n_files} embeddings…")

    print(f"✔ Wrote {total_rows} rows → {OUT_CSV}")
    if missing_seq_entries:
        print(f"⚠️  Missing FASTA sequence for {missing_seq_entries} entries (used 'X').")
    if length_mismatches:
        print(f"⚠️  Length mismatches for {length_mismatches} entries (used min length).")

if __name__ == "__main__":
    main()
