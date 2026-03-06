#!/usr/bin/env python3
import os
import re
import sys
import glob
import json
import gc
import warnings
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from Bio import SeqIO
from sklearn.metrics import balanced_accuracy_score

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# ─── Config ────────────────────────────────────────────────────────────────
percent_identity = 40
num_features     = 70
PRED_THRESH      = 0.5

# ─── Paths ────────────────────────────────────────────────────────────────
MODEL_PATH = "/Volumes/T7 Shield/random_forest_chpc/rf_final_allfolds_features_70.joblib"

DATA_ROOT  = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
EMB_DIR    = "/Volumes/T7 Shield/known_peptides_SAE_embeddings"
FASTA_PATH = "/Volumes/T7 Shield/uniprot_known_neuropep_sequences.fasta"

RANKING_ALL_FOLDS = os.path.join(
    DATA_ROOT, f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}",
    "feature_ranking_all_folds.csv"
)

POS_CSV_OPT = os.path.join(DATA_ROOT, "input_data", "group_2_positive_known_peps.csv")

OUT_DIR = os.path.join(
    DATA_ROOT, "classifier_results", "ranfor_known_peptides", f"graphpart_{percent_identity}"
)
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV  = os.path.join(OUT_DIR, f"known_peptides_rf_probs_predictions_{num_features}.csv")
OUT_SUM  = os.path.join(OUT_DIR, f"known_peptides_rf_summary_{num_features}.csv")
OUT_FEAT = os.path.join(OUT_DIR, f"known_peptides_rf_features_used_{num_features}.json")

# ─── Runtime compatibility helpers ─────────────────────────────────────────
def _version_tuple(v: str):
    parts = re.findall(r"\d+", v)
    return tuple(int(x) for x in parts[:3]) or (0,)

def ensure_compatible_runtime():
    """
    Ensure NumPy>=2.0 and scikit-learn==1.7.1 (or 1.7.x) for loading the RF pickle.
    If RF_BOOTSTRAP=1, create a small venv with compatible versions and re-exec.
    """
    import importlib.metadata as im
    np_ver = _version_tuple(im.version("numpy"))
    sk_ver = _version_tuple(im.version("scikit-learn"))

    np_ok = np_ver >= (2,0,0)
    sk_ok = sk_ver >= (1,7,0) and sk_ver < (1,8,0)

    if np_ok and sk_ok:
        return  # good to go

    if os.environ.get("RF_BOOTSTRAP", "0") != "1":
        print("✖ Environment incompatible for loading this model.")
        print(f"  Detected: numpy={im.version('numpy')} scikit-learn={im.version('scikit-learn')}")
        print("  Needs:    numpy>=2.0  and  scikit-learn==1.7.x")
        print("\nTwo options:\n"
              "A) Activate a compatible env and run again, e.g.\n"
              "   conda create -n rf_infer python=3.11 numpy=2.0.* scikit-learn=1.7.1 joblib pandas biopython pytorch -c conda-forge\n"
              "   conda activate rf_infer\n"
              "   python run_final_ranfor_model.py\n\n"
              "B) Let this script create a local venv once and re-run itself:\n"
              "   RF_BOOTSTRAP=1 python run_final_ranfor_model.py\n")
        sys.exit(2)

    # Bootstrap a small venv and re-exec this script in it
    cache_dir = Path.home() / ".cache" / "rf_infer_env"
    pybin = cache_dir / "bin" / "python"
    if not pybin.exists():
        print("ℹ️  Creating local venv with compatible NumPy/Sklearn …")
        subprocess.check_call([sys.executable, "-m", "venv", str(cache_dir)])
        pip = str(cache_dir / "bin" / "pip")
        # Upgrade pip/wheel, then install exact versions
        subprocess.check_call([pip, "install", "--upgrade", "pip", "wheel", "setuptools"])
        subprocess.check_call([pip, "install",
                               "numpy>=2.0,<3.0",
                               "scikit-learn==1.7.1",
                               "joblib",
                               "pandas",
                               "biopython",
                               "torch"])  # if torch already exists system-wide, pip will skip

    print("✔ Re-executing in the compatible venv …")
    os.execv(str(pybin), [str(pybin), *sys.argv])

# ─── Domain helpers ────────────────────────────────────────────────────────
def parse_fasta_to_map(fa_path: str) -> dict:
    seqs = {}
    for rec in SeqIO.parse(fa_path, "fasta"):
        parts = rec.id.split("|")
        entry = parts[1] if len(parts) >= 3 else rec.id
        seqs[entry] = str(rec.seq)
    return seqs

def entry_from_embedding_path(path: str) -> str | None:
    base = os.path.basename(path)
    if "|" in base:
        parts = base.split("|")
        return parts[1] if len(parts) >= 3 else None
    m = re.match(r"([A-Za-z0-9]+)(?:_.*)?\.pt$", base)
    return m.group(1) if m else None

def try_load_feature_indices_from_meta(model_path: str) -> list[int] | None:
    meta_path = re.sub(r"\.joblib$", ".meta.json", model_path)
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        feats = meta.get("feature_indices")
        if feats and len(feats) == num_features:
            print(f"✔ Loaded {len(feats)} feature indices from meta: {meta_path}")
            return [int(x) for x in feats]
        print(f"⚠️  Meta found but feature_indices missing or wrong length in {meta_path}")
    else:
        print("ℹ️  No meta JSON next to model; will try all-folds ranking.")
    return None

def load_features_from_ranking(ranking_csv: str, k: int) -> list[int]:
    if not os.path.isfile(ranking_csv):
        raise FileNotFoundError(f"Ranking file not found: {ranking_csv}")
    df = pd.read_csv(ranking_csv).sort_values("rank")
    feats = df["feature"].astype(int).tolist()
    if len(feats) < k:
        raise RuntimeError(f"Ranking has only {len(feats)} features; need {k}.")
    print(f"✔ Using top {k} features from ranking (max index = {max(feats[:k])})")
    return feats[:k]

def build_pos_map_if_available(pos_csv: str) -> dict[str, set[int]] | None:
    if not os.path.isfile(pos_csv):
        print("ℹ️  No ground-truth CSV found; will skip accuracy metrics.")
        return None
    df = pd.read_csv(pos_csv)
    df["residue_number"] = df["residue_number"].astype(int)
    return df.groupby("Entry")["residue_number"].apply(lambda s: set(s.tolist())).to_dict()

# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    # Ensure runtime is compatible or auto-bootstrap if asked
    ensure_compatible_runtime()

    # FASTA
    if not os.path.isfile(FASTA_PATH):
        raise FileNotFoundError(f"FASTA not found: {FASTA_PATH}")
    entry_to_seq = parse_fasta_to_map(FASTA_PATH)
    print(f"✔ Loaded {len(entry_to_seq)} sequences from FASTA")

    # Load model (now in a compatible runtime)
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)
    print(f"✔ Loaded RF model from {MODEL_PATH}")

    # Features
    feats = try_load_feature_indices_from_meta(MODEL_PATH)
    if feats is None:
        feats = load_features_from_ranking(RANKING_ALL_FOLDS, num_features)
    assert len(feats) == num_features

    pos_map = build_pos_map_if_available(POS_CSV_OPT)

    emb_paths = sorted(glob.glob(os.path.join(EMB_DIR, "*.pt")))
    if not emb_paths:
        raise RuntimeError(f"No .pt files found in {EMB_DIR}")
    print(f"✔ Found {len(emb_paths)} embedding files")

    rows = []
    total_correct = 0
    total_count   = 0

    for emb_fp in emb_paths:
        entry = entry_from_embedding_path(emb_fp)
        if not entry:
            print(f"⚠️  Could not parse Entry from filename: {emb_fp}; skipping.")
            continue

        arr = torch.load(emb_fp, map_location="cpu").numpy()  # (L, D)
        L, D = arr.shape
        if max(feats) >= D:
            print(f"⚠️  Skipping {entry}: feature index {max(feats)} >= embedding dim {D}")
            continue

        X_seq = arr[:, feats]
        probs = clf.predict_proba(X_seq)[:, 1] if hasattr(clf, "predict_proba") else None
        if probs is None:
            # Shouldn't happen for RF; but fallback if needed
            scores = clf.decision_function(X_seq)
            smin, smax = float(scores.min()), float(scores.max())
            probs = (scores - smin) / (smax - smin + 1e-12)
        preds = (probs >= PRED_THRESH).astype(int)

        seq = entry_to_seq.get(entry, "X" * L)
        use_L = min(L, len(seq))
        pos_set = pos_map.get(entry, set()) if pos_map else set()

        for i in range(use_L):
            y_true = (1 if (i + 1) in pos_set else 0) if pos_map else None
            rows.append({
                "Entry": entry,
                "position": i + 1,
                "residue": seq[i],
                "pred_prob": float(probs[i]),
                "pred_label": int(preds[i]),
                **({"true_label": y_true} if pos_map else {})
            })
            if pos_map:
                total_correct += int(y_true == int(preds[i]))
                total_count   += 1

    cols = ["Entry", "position", "residue", "pred_prob", "pred_label"]
    if pos_map:
        cols.append("true_label")
    out_df = pd.DataFrame(rows, columns=cols)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"✔ Saved probabilities + predictions for {len(out_df)} positions → {OUT_CSV}")

    if pos_map and total_count > 0:
        y_true_all = out_df["true_label"].to_numpy()
        y_pred_all = out_df["pred_label"].to_numpy()
        bacc = balanced_accuracy_score(y_true_all, y_pred_all) * 100.0
        acc  = (y_true_all == y_pred_all).mean() * 100.0
        pd.DataFrame([{
            "model_path": MODEL_PATH,
            "num_features": num_features,
            "threshold": PRED_THRESH,
            "accuracy": f"{acc:.2f}",
            "balanced_accuracy": f"{bacc:.2f}",
            "n_positions": int(total_count),
        }]).to_csv(OUT_SUM, index=False)
        print(f"✔ Wrote summary → {OUT_SUM}")
    else:
        print("ℹ️  No ground truth provided; skipped accuracy summary.")

    with open(OUT_FEAT, "w") as f:
        json.dump({"feature_indices": feats}, f, indent=2)
    print(f"✔ Wrote features used → {OUT_FEAT}")

    gc.collect()
    print("All done.")

if __name__ == "__main__":
    main()
