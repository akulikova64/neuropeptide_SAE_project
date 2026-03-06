#!/usr/bin/env python3
"""
Run 3 models (LogReg, CRF, SVM) on embeddings for a *restricted set* of entries
and write a single CSV with per-residue probabilities from each model plus
their mean and per-peptide normalized probabilities.

Current configuration:
  - Embeddings are read from:
      /Volumes/T7 Shield/layer_33_highest_scoring_novo_embeddings
    (this directory should contain .pt files for the 23 flagged neuropeptides)

  - Only entries present in:
      ../../data/novo_alldata_filter_out_enzymes/neuropeptide_filter/novo_flagged_neuropeptides.fasta
    are used. This FASTA should contain exactly the 23 flagged neuropeptides.

Output CSV columns:
  Entry, residue, residue_number,
  prob_logreg, prob_crf, prob_svm, prob_mean,
  prob_logreg_norm, prob_crf_norm, prob_svm_norm, prob_mean_norm
"""

import os
import re
import glob
import csv
import sys
import warnings
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
import joblib
from Bio import SeqIO

# For RF/SVM version warnings (even though RF is removed)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# ─── Config ────────────────────────────────────────────────────────────────
percent_identity = 40

# Feature counts per model (from your original scripts)
NUM_FEATS_LOGREG      = 70
NUM_FEATS_CRF         = 175
NUM_FEATS_SVM_DEFAULT = 100   # overridden by filename if needed

DATA_ROOT = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"

# Embeddings directory (now contains .pt files for the 23 flagged entries)
#EMB_DIR = "/Volumes/T7 Shield/layer_33_highest_scoring_novo_embeddings"
#EMB_DIR = "/Volumes/T7 Shield/layer_33_whited_embeddings"
#EMB_DIR = "/Volumes/T7 Shield/layer_33_de_Souza_embeddings"
EMB_DIR = "/Volumes/T7 Shield/layer_33_highest_scoring_secretome_embeddings"

# FASTA: ONLY the 23 flagged neuropeptides
FASTA_PATH = "../../data/top_secretome_hits.fasta"

RANKING_CSV = os.path.join(
    DATA_ROOT,
    f"F1_scores_hybrid/F1_scores_graphpart_{percent_identity}",
    "feature_ranking_all_folds.csv",
)

# Model paths
LOGREG_MODEL_PATH = os.path.join(
    DATA_ROOT,
    "classifier_results",
    "logistic_regression_models",
    f"graphpart_{percent_identity}",
    f"final_logreg_model_{NUM_FEATS_LOGREG}_features.joblib",
)

CRF_MODEL_PATH = os.path.join(
    DATA_ROOT,
    "classifier_results",
    "conditional_random_field_models",
    f"graphpart_{percent_identity}",
    f"final_crf_model_{NUM_FEATS_CRF}_features.joblib",
)

SVM_MODEL_PATH = "/Volumes/T7 Shield/svm_rbf_final_features_100.joblib"

# Output
OUT_DIR = "/Volumes/T7 Shield/novo_cleavage_site_predictions"
OUT_CSV = os.path.join(
    OUT_DIR,
    "top_2_secretome_cleavage_site_predictions.csv",
)
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Shared helpers ────────────────────────────────────────────────────────
def parse_fasta_to_map(fa_path: str) -> dict[str, str]:
    """Return {Entry accession → sequence} from FASTA."""
    seqs: dict[str, str] = {}
    for rec in SeqIO.parse(fa_path, "fasta"):
        parts = rec.id.split("|")
        entry = parts[1] if len(parts) >= 3 else rec.id
        seqs[entry] = str(rec.seq)
    return seqs

def entry_from_embedding_path(path: str) -> str | None:
    """
    Extract Entry from embedding filename, e.g.:
      - 'sp|P01178|NEU1_HUMAN_original_SAE.pt' → 'P01178'
      - 'A0A024RBI1_original_SAE.pt'          → 'A0A024RBI1'
      - 'STRG.4794.2+chr1:..._original_SAE.pt' → full ID before suffix
    """
    base = os.path.basename(path)
    if "|" in base:
        parts = base.split("|")
        return parts[1] if len(parts) >= 2 else None
    m = re.match(r"(.+)_original_SAE\.pt$", base)
    return m.group(1) if m else None


def load_top_features(k: int) -> list[int]:
    """Load top-k feature indices from ranking CSV (shared across models)."""
    if not os.path.isfile(RANKING_CSV):
        raise FileNotFoundError(f"Ranking file not found: {RANKING_CSV}")
    rank_df = pd.read_csv(RANKING_CSV).sort_values("rank")
    feat_list = rank_df["feature"].astype(int).tolist()
    if len(feat_list) < k:
        raise RuntimeError(f"Ranking has only {len(feat_list)} features; need {k}.")
    top_feats = feat_list[:k]
    print(f"✔ Using top {k} features (max index = {max(top_feats)})")
    return top_feats


# ─── CRF-specific helpers ──────────────────────────────────────────────────
def pick_positive_label(crf_model) -> str:
    """Determine which label is the positive class in the CRF."""
    labels = set(getattr(crf_model, "classes_", []) or [])
    for cand in ("1", 1, "yes", "Y", "POS", "True", True):
        if cand in labels:
            return str(cand)
    return "1"


# ─── SVM helpers ───────────────────────────────────────────────────────────
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def try_load_svm_feature_indices_from_meta(
    model_path: str, expect_n: int
) -> list[int] | None:
    meta_path = re.sub(r"\.joblib$", ".meta.json", model_path)
    if not os.path.isfile(meta_path):
        return None
    try:
        import json

        with open(meta_path, "r") as f:
            meta = json.load(f)
        feats = meta.get("feature_indices")
        if feats and len(feats) == expect_n:
            print(f"✔ Loaded {len(feats)} SVM feature indices from meta: {meta_path}")
            return [int(x) for x in feats]
        print(f"⚠️ Meta found but wrong/missing feature_indices in {meta_path}")
    except Exception as e:
        print(f"⚠️ Failed reading meta {meta_path}: {e}")
    return None


# ─── Runtime compatibility helper ──────────────────────────────────────────
def _ver_tuple(s: str):
    import re as _re

    parts = _re.findall(r"\d+", s)
    return tuple(int(x) for x in parts[:3]) or (0,)


def ensure_compatible_runtime():
    """
    Ensure NumPy>=2.0 and scikit-learn==1.7.x.

    If incompatible:
      - If RF_BOOTSTRAP is not set to 1: print instructions and exit(2).
      - If RF_BOOTSTRAP=1: create a small venv in ~/.cache/rf_infer_env
        with compatible versions and re-exec this script inside it.
    """
    import importlib.metadata as im

    try:
        np_ver = im.version("numpy")
    except Exception:
        np_ver = "0.0.0"
    try:
        sk_ver = im.version("scikit-learn")
    except Exception:
        sk_ver = "0.0.0"

    np_ok = _ver_tuple(np_ver) >= (2, 0, 0)
    sk_ok = (1, 7, 0) <= _ver_tuple(sk_ver) < (1, 8, 0)

    if np_ok and sk_ok:
        return  # current env is fine

    if os.environ.get("RF_BOOTSTRAP", "0") != "1":
        print("✖ Environment incompatible for loading SVM pickles.")
        print(f"  Detected: numpy={np_ver} scikit-learn={sk_ver}")
        print("  Needs:    numpy>=2.0  and  scikit-learn==1.7.x")
        print(
            "\nTwo options:\n"
            "A) conda create -n rf_infer python=3.11 numpy=2.0.* scikit-learn=1.7.1 "
            "joblib pandas biopython pytorch sklearn-crfsuite -c conda-forge\n"
            "   conda activate rf_infer\n"
            "   python run_cleavage_site_predictor.py\n\n"
            "B) Auto-create a small venv & re-run:\n"
            "   RF_BOOTSTRAP=1 python run_cleavage_site_predictor.py\n"
        )
        sys.exit(2)

    # Auto-bootstrap mode
    cache_dir = Path.home() / ".cache" / "rf_infer_env"
    pybin = cache_dir / "bin" / "python"

    if not pybin.exists():
        print("ℹ️  Creating local venv with compatible NumPy/Sklearn …")
        subprocess.check_call([sys.executable, "-m", "venv", str(cache_dir)])
        pip = str(cache_dir / "bin" / "pip")
        subprocess.check_call([pip, "install", "--upgrade", "pip", "wheel", "setuptools"])
        subprocess.check_call(
            [
                pip,
                "install",
                "numpy>=2.0,<3.0",
                "scikit-learn==1.7.1",
                "joblib",
                "pandas",
                "biopython",
                "torch",
                "sklearn-crfsuite",
            ]
        )

    print("✔ Re-executing in the compatible venv …")
    os.execv(str(pybin), [str(pybin), *sys.argv])


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize 1D array to [0, 1] (per peptide); all-equal → zeros."""
    amin = float(arr.min())
    amax = float(arr.max())
    if amax > amin:
        return (arr - amin) / (amax - amin)
    else:
        return np.zeros_like(arr, dtype=float)


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    # Ensure we have a NumPy / sklearn combo that can load SVM pickles
    ensure_compatible_runtime()

    # Now that we're in the final runtime, ensure sklearn-crfsuite is available
    try:
        from sklearn_crfsuite import CRF  # noqa: F401
    except ModuleNotFoundError:
        print("ℹ️ Installing sklearn-crfsuite into the current environment …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "sklearn-crfsuite"]
        )
        from sklearn_crfsuite import CRF  # noqa: F401

    # 0) Basic checks & FASTA (defines which entries to use)
    if not os.path.isfile(FASTA_PATH):
        raise FileNotFoundError(f"FASTA not found: {FASTA_PATH}")
    entry_to_seq = parse_fasta_to_map(FASTA_PATH)
    allowed_entries = set(entry_to_seq.keys())
    print(f"✔ Loaded {len(entry_to_seq)} sequences from FASTA (allowed entries)")

    emb_paths = sorted(glob.glob(os.path.join(EMB_DIR, "*.pt")))
    if not emb_paths:
        raise RuntimeError(f"No .pt files found under {EMB_DIR}")
    print(f"✔ Found {len(emb_paths)} embedding files (before filtering)")

    # Filter embeddings to only those entries present in the FASTA (23 entries)
    emb_paths_filtered: list[str] = []
    for p in emb_paths:
        entry = entry_from_embedding_path(p)
        if entry and entry in allowed_entries:
            emb_paths_filtered.append(p)

    if not emb_paths_filtered:
        raise RuntimeError("No embedding files matched entries in the FASTA.")
    print(f"✔ Using {len(emb_paths_filtered)} embedding files matching FASTA entries")

    # 1) Load feature rankings
    feats_logreg = load_top_features(NUM_FEATS_LOGREG)
    feats_crf = load_top_features(NUM_FEATS_CRF)

    # SVM: num features from filename if present
    num_feats_svm = NUM_FEATS_SVM_DEFAULT
    m = re.search(r"_features_(\d+)\.joblib$", os.path.basename(SVM_MODEL_PATH))
    if m:
        nf = int(m.group(1))
        if nf != num_feats_svm:
            print(f"ℹ️ Overriding SVM num_features from filename → {nf}")
            num_feats_svm = nf

    svm_feats = try_load_svm_feature_indices_from_meta(SVM_MODEL_PATH, num_feats_svm)
    if svm_feats is None:
        svm_feats = load_top_features(num_feats_svm)

    # 2) Load models
    if not os.path.isfile(LOGREG_MODEL_PATH):
        raise FileNotFoundError(f"LogReg model not found: {LOGREG_MODEL_PATH}")
    clf_logreg = joblib.load(LOGREG_MODEL_PATH)
    print(f"✔ Loaded Logistic Regression model from {LOGREG_MODEL_PATH}")

    if not os.path.isfile(CRF_MODEL_PATH):
        raise FileNotFoundError(f"CRF model not found: {CRF_MODEL_PATH}")
    crf_model = joblib.load(CRF_MODEL_PATH)
    crf_pos_label = pick_positive_label(crf_model)
    print(
        f"✔ Loaded CRF model from {CRF_MODEL_PATH} (positive label = {crf_pos_label!r})"
    )

    if not os.path.isfile(SVM_MODEL_PATH):
        raise FileNotFoundError(f"SVM model not found: {SVM_MODEL_PATH}")
    try:
        svm_model = joblib.load(SVM_MODEL_PATH)
    except ModuleNotFoundError as e:
        # handle numpy._core shim if needed
        if "numpy._core" in str(e):
            import numpy.core as _ncore

            sys.modules["numpy._core"] = _ncore
            svm_model = joblib.load(SVM_MODEL_PATH)
        else:
            raise
    print(f"✔ Loaded SVM model from {SVM_MODEL_PATH}")

    # 3) Iterate embeddings and compute probabilities
    total_rows = 0
    length_mismatches = 0

    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "Entry",
                "residue",
                "residue_number",
                "prob_logreg",
                "prob_crf",
                "prob_svm",
                "prob_mean",
                "prob_logreg_norm",
                "prob_crf_norm",
                "prob_svm_norm",
                "prob_mean_norm",
            ]
        )

        n_files = len(emb_paths_filtered)
        for idx, emb_fp in enumerate(emb_paths_filtered, 1):
            entry = entry_from_embedding_path(emb_fp)
            if not entry:
                print(f"⚠️  Could not parse Entry from {emb_fp}; skipping.")
                continue

            arr = torch.load(emb_fp, map_location="cpu").numpy()
            if arr.ndim != 2:
                print(f"⚠️  Bad shape for {entry}: {arr.shape}; skipping.")
                continue

            L, D = arr.shape
            if (
                max(feats_logreg) >= D
                or max(feats_crf) >= D
                or max(svm_feats) >= D
            ):
                print(
                    f"⚠️  Feature index out of bounds for {entry} (D={D}); skipping."
                )
                continue

            seq = entry_to_seq.get(entry)
            if seq is None:
                # Ideally should not happen because we filtered by allowed_entries
                seq = "X" * L

            use_L = min(L, len(seq))
            if use_L != L or use_L != len(seq):
                length_mismatches += 1

            # Slice feature matrices
            X_logreg = arr[:use_L, feats_logreg]
            X_svm = arr[:use_L, svm_feats]

            # --- Logistic Regression probs ---
            probs_logreg = clf_logreg.predict_proba(X_logreg)[:, 1]

            # --- SVM probs ---
            if hasattr(svm_model, "predict_proba"):
                probs_svm = svm_model.predict_proba(X_svm)[:, 1]
            elif hasattr(svm_model, "decision_function"):
                scores = svm_model.decision_function(X_svm)
                probs_svm = sigmoid(scores.astype(np.float64))
            else:
                preds = svm_model.predict(X_svm)
                probs_svm = preds.astype(float)

            # --- CRF probs ---
            X_crf_feats = [
                {f"f{i}": float(arr[pos, i]) for i in feats_crf}
                for pos in range(use_L)
            ]
            marginals = crf_model.predict_marginals_single(X_crf_feats)
            probs_crf = np.empty(use_L, dtype=float)
            for i in range(use_L):
                m = marginals[i]
                p = (m.get(crf_pos_label) or m.get(str(crf_pos_label)))
                if p is None:
                    p = m.get("1")
                if p is None:
                    p = max(m.values())
                probs_crf[i] = float(p)

            # --- Combine and normalize per peptide ---
            probs_logreg = probs_logreg.astype(float)
            probs_crf = probs_crf.astype(float)
            probs_svm = probs_svm.astype(float)

            probs_mean = (probs_logreg + probs_crf + probs_svm) / 3.0

            probs_logreg_norm = minmax_norm(probs_logreg)
            probs_crf_norm = minmax_norm(probs_crf)
            probs_svm_norm = minmax_norm(probs_svm)
            probs_mean_norm = (
                probs_logreg_norm + probs_crf_norm + probs_svm_norm
            ) / 3.0

            # --- Write combined rows ---
            for i in range(use_L):
                r = seq[i]
                pos = i + 1
                pl = float(probs_logreg[i])
                pc = float(probs_crf[i])
                ps = float(probs_svm[i])
                pm = float(probs_mean[i])
                pln = float(probs_logreg_norm[i])
                pcn = float(probs_crf_norm[i])
                psn = float(probs_svm_norm[i])
                pmn = float(probs_mean_norm[i])

                writer.writerow(
                    [
                        entry,
                        r,
                        pos,
                        pl,
                        pc,
                        ps,
                        pm,
                        pln,
                        pcn,
                        psn,
                        pmn,
                    ]
                )
            total_rows += use_L

            if idx % 50 == 0 or idx == n_files:
                print(f"Processed {idx}/{n_files} embeddings…")

    print(f"✔ Wrote {total_rows} rows → {OUT_CSV}")
    if length_mismatches:
        print(
            f"⚠️ Length mismatches for {length_mismatches} entries (used min length)."
        )


if __name__ == "__main__":
    main()
