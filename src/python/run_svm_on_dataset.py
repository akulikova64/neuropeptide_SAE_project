#!/usr/bin/env python3
import os, re, sys, glob, csv, json, warnings, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from Bio import SeqIO

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# ── Config ─────────────────────────────────────────────────────────────────
percent_identity = 40
num_features     = 100   # default; overridden by model filename if it has _features_###.joblib

DATA_ROOT  = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
EMB_DIR     = "/Volumes/T7 Shield/layer_33_highest_scoring_novo_embeddings"
FASTA_PATH  = "../../data/novo_smORF_highest_scoring_data.fasta"

RANKING_ALL_FOLDS = os.path.join(
    DATA_ROOT, f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}",
    "feature_ranking_all_folds.csv"
)

SVM_MODEL_PATH = "/Volumes/T7 Shield/svm_rbf_final_features_100.joblib"

OUT_DIR = "/Volumes/T7 Shield/novo_cleavage_site_predictions"
OUT_CSV = os.path.join(OUT_DIR, "secratome_svm_probs_AUTO.csv")

# ── Version bootstrap (like you used for RF) ───────────────────────────────
def _v(s): 
    import re as _re
    t = _re.findall(r"\d+", s)
    return tuple(int(x) for x in t[:3]) or (0,)

def ensure_compatible_runtime():
    import importlib.metadata as im
    try:
        np_ver = im.version("numpy")
        sk_ver = im.version("scikit-learn")
    except Exception:
        np_ver, sk_ver = "0.0.0", "0.0.0"
    need_np = _v(np_ver) >= (2,0,0)
    need_sk = (1,7,0) <= _v(sk_ver) < (1,8,0)
    if need_np and need_sk:
        return
    if os.environ.get("SVM_BOOTSTRAP","0") != "1":
        print("✖ Environment incompatible for loading this SVM model.")
        print(f"  Detected: numpy={np_ver} scikit-learn={sk_ver}")
        print("  Needs:    numpy>=2.0 and scikit-learn==1.7.x\n")
        print("Two options:")
        print("A) conda create -n svm_infer python=3.11 numpy=2.0.* scikit-learn=1.7.1 joblib pandas biopython pytorch -c conda-forge")
        print("   conda activate svm_infer")
        print("   python run_svm_on_dataset.py\n")
        print("B) Auto-create a local venv & re-run:")
        print("   SVM_BOOTSTRAP=1 python run_svm_on_dataset.py\n")
        sys.exit(2)

    cache = Path.home()/".cache"/"svm_infer_env"
    pybin = cache/"bin"/"python"
    if not pybin.exists():
        print("ℹ️  Creating local venv with compatible NumPy/Sklearn …")
        subprocess.check_call([sys.executable, "-m", "venv", str(cache)])
        pip = str(cache/"bin"/"pip")
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

# ── Helpers ────────────────────────────────────────────────────────────────
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
        return parts[1] if len(parts) >= 2 else None
    m = re.match(r"([A-Za-z0-9]+)_original_SAE\.pt$", base)
    return m.group(1) if m else None

def try_load_feature_indices_from_meta(model_path: str, expect_n: int):
    meta_path = re.sub(r"\.joblib$", ".meta.json", model_path)
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        feats = meta.get("feature_indices")
        if feats and len(feats)==expect_n:
            print(f"✔ Loaded {len(feats)} feature indices from meta: {meta_path}")
            return [int(x) for x in feats]
        print(f"⚠️  Meta found but wrong/missing feature_indices in {meta_path}")
        return None
    except Exception as e:
        print(f"⚠️  Failed reading meta {meta_path}: {e}")
        return None

def load_features_from_ranking(ranking_csv: str, k: int):
    if not os.path.isfile(ranking_csv):
        raise FileNotFoundError(f"Ranking file not found: {ranking_csv}")
    df = pd.read_csv(ranking_csv).sort_values("rank")
    feats = df["feature"].astype(int).tolist()
    if len(feats) < k:
        raise RuntimeError(f"Ranking has only {len(feats)} features; need {k}.")
    print(f"✔ Using top {k} features from ranking (max index = {max(feats[:k])})")
    return feats[:k]

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0/(1.0+np.exp(-x))

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ensure_compatible_runtime()

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SVM_MODEL_PATH)
    ap.add_argument("--num-features", type=int, default=num_features)
    ap.add_argument("--emb-dir", default=EMB_DIR)
    ap.add_argument("--fasta",   default=FASTA_PATH)
    ap.add_argument("--ranking", default=RANKING_ALL_FOLDS)
    ap.add_argument("--out",     default=OUT_CSV)
    args = ap.parse_args()

    model_path = args.model
    nfeat      = args.num_features
    emb_dir    = args.emb_dir
    fasta_path = args.fasta
    ranking    = args.ranking
    out_csv    = args.out

    # FASTA
    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    entry_to_seq = parse_fasta_to_map(fasta_path)
    print(f"✔ Loaded {len(entry_to_seq)} sequences from FASTA")

    # Load model with numpy._core shim
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    try:
        clf = joblib.load(model_path)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            import numpy.core as _ncore
            sys.modules["numpy._core"] = _ncore
            clf = joblib.load(model_path)
        else:
            raise
    print(f"✔ Loaded SVM model from {model_path}")

    # Auto-infer features from filename
    m = re.search(r"_features_(\d+)\.joblib$", os.path.basename(model_path))
    if m:
        nf = int(m.group(1))
        if nf != nfeat:
            print(f"ℹ️  Overriding num_features from filename → {nf}")
            nfeat = nf

    # Features
    feats = try_load_feature_indices_from_meta(model_path, nfeat)
    if feats is None:
        feats = load_features_from_ranking(ranking, nfeat)

    emb_paths = sorted(glob.glob(os.path.join(emb_dir, "*.pt")))
    if not emb_paths:
        raise RuntimeError(f"No .pt files found under {emb_dir}")
    print(f"✔ Found {len(emb_paths)} embedding files")

    total_rows = 0
    missing_seq_entries = 0
    length_mismatches = 0

    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Entry","residue","residue_number","probability"])

        for idx, emb_fp in enumerate(emb_paths, 1):
            entry = entry_from_embedding_path(emb_fp)
            if not entry:
                print(f"⚠️  Could not parse Entry from {emb_fp}; skipping.")
                continue

            arr = torch.load(emb_fp, map_location="cpu").numpy()
            if arr.ndim != 2:
                print(f"⚠️  Bad shape for {entry}: {arr.shape}; skipping.")
                continue
            L, D = arr.shape
            if max(feats) >= D:
                print(f"⚠️  Feature index {max(feats)} out of bounds for {entry} (D={D}); skipping.")
                continue

            seq = entry_to_seq.get(entry)
            if seq is None:
                missing_seq_entries += 1
                seq = "X"*L

            use_L = min(L, len(seq))
            if use_L != L or use_L != len(seq):
                length_mismatches += 1

            X = arr[:use_L, feats]

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)[:,1]
            elif hasattr(clf, "decision_function"):
                scores = clf.decision_function(X)
                probs = sigmoid(scores.astype(np.float64))
            else:
                preds = clf.predict(X)
                probs = preds.astype(float)

            for i in range(use_L):
                w.writerow([entry, seq[i], i+1, float(probs[i])])
            total_rows += use_L

            if idx % 500 == 0 or idx == len(emb_paths):
                print(f"Processed {idx}/{len(emb_paths)} embeddings…")

    print(f"✔ Wrote {total_rows} rows → {out_csv}")
    if missing_seq_entries:
        print(f"⚠️  Missing FASTA sequence for {missing_seq_entries} entries (used 'X').")
    if length_mismatches:
        print(f"⚠️  Length mismatches for {length_mismatches} entries (used min length).")

if __name__ == "__main__":
    main()
