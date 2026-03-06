#!/usr/bin/env python3
"""
Train a single Logistic Regression model on ALL data using top-N features
ranked by the global winners file (all folds), sorted by winning_F1_score (desc).

Input:
  --winners_csv : CSV with columns [feature, winning_F1_score, winning_threshold|winning_theshold]
                  e.g. "/Volumes/T7 Shield/layer_18_F1_scores/winning_thresholds_l18.csv"
  --num_features: how many top features to use (by winning_F1_score)
  --positives / --negatives: embedding folders (.pt with shape (L,F); we max-pool over L)
  [optional] --permitted_pos / --permitted_neg: FASTAs to restrict training set
  --out_model  : where to save the trained model (and a JSON metadata file)

Example:
  python train_log_reg_l18_final.py \
    --positives "/Volumes/T7 Shield/layer_18_embeddings" \
    --negatives "/Volumes/T7 Shield/layer_18_negative_embeddings" \
    --winners_csv "/Volumes/T7 Shield/layer_18_F1_scores/winning_thresholds_l18.csv" \
    --num_features 900 \
    --out_model "/Volumes/T7 Shield/layer_18_classifiers/final/logreg_allfolds_top900.joblib" \
    --permitted_pos "../../data/combined_l18_positive_final.fasta" \
    --permitted_neg "../../data/negative_l18_dataset_final.fasta" \
    --seed 9687254
"""

import os, re, json, time, argparse
from glob import glob
import numpy as np
import pandas as pd
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump as joblib_dump

# ── Logging ────────────────────────────────────────────────────────────
def log(msg): print(msg, flush=True)
def human_time(s):
    m, s = divmod(int(max(0, s)), 60); h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

# ── ID / IO helpers ────────────────────────────────────────────────────
ID_FROM_FILENAME = re.compile(r"^(?P<id>.+?)(?:_original_SAE)?\.pt$")

def filename_to_id(filename: str) -> str:
    m = ID_FROM_FILENAME.match(filename)
    if m: return m.group("id")
    return filename[:-3] if filename.endswith(".pt") else filename

def list_pt(d):
    return sorted(p for p in glob(os.path.join(d, "*.pt"))
                  if not os.path.basename(p).startswith("._"))

def safe_load_tensor(path):
    if os.path.getsize(path) < 64:
        raise EOFError("file too small")
    t = torch.load(path, map_location="cpu")
    if t.ndim != 2:
        raise ValueError(f"bad shape {tuple(t.shape)}")
    return t

def fasta_ids(path):
    ids = set()
    with open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                ids.add(line[1:].strip().split()[0])
    return ids

def build_embed_index(embed_dir: str):
    id_to_path = {}
    for p in list_pt(embed_dir):
        sid = filename_to_id(os.path.basename(p))
        id_to_path.setdefault(sid, p)
    return id_to_path

# ── Dataset build ──────────────────────────────────────────────────────
def build_dataset(pos_dir, neg_dir, top_feature_ids, permitted_pos=None, permitted_neg=None, log_every=2000):
    pos_idx = build_embed_index(pos_dir)
    neg_idx = build_embed_index(neg_dir)

    pos_ids = sorted(pos_idx.keys()) if permitted_pos is None else sorted(pos_idx.keys() & permitted_pos)
    neg_ids = sorted(neg_idx.keys()) if permitted_neg is None else sorted(neg_idx.keys() & permitted_neg)

    pos_missing = len(permitted_pos - set(pos_idx.keys())) if permitted_pos is not None else 0
    neg_missing = len(permitted_neg - set(neg_idx.keys())) if permitted_neg is not None else 0
    if pos_missing or neg_missing:
        log(f"[permit] Missing embeddings → POS={pos_missing:,}  NEG={neg_missing:,}")

    files = [(pos_idx[i], 1) for i in pos_ids] + [(neg_idx[i], 0) for i in neg_ids]
    if not files:
        raise RuntimeError("No training samples after applying permitted-ID filtering.")

    # Infer feature dim
    F = None
    for path, _ in files:
        try:
            t = safe_load_tensor(path)
            F = t.shape[1]
            break
        except Exception:
            continue
    if F is None:
        raise RuntimeError("Could not load any valid .pt tensors to infer feature dimension.")

    top_feature_ids = [int(f) for f in top_feature_ids]
    if any(f < 0 or f >= F for f in top_feature_ids):
        raise ValueError(f"Feature id outside [0,{F-1}] in winners selection.")

    n = len(files)
    d = len(top_feature_ids)
    X = np.empty((n, d), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)

    start = time.time()
    bad = 0
    for i, (path, label) in enumerate(files):
        try:
            v = safe_load_tensor(path).max(dim=0).values.numpy()
            X[i, :] = v[top_feature_ids]
            y[i] = label
        except Exception:
            bad += 1
            X[i, :] = 0.0
            y[i] = label

        if (i+1) % log_every == 0 or (i+1) == n:
            el = time.time() - start
            rate = (i+1)/el if el > 0 else 0.0
            eta = (n-(i+1))/rate if rate > 0 else 0.0
            log(f"  [build] {i+1:>6}/{n:<6} | elapsed {human_time(el)} | {rate:6.1f} samp/s | ETA {human_time(eta)}")

    if bad:
        log(f"[data] WARNING: {bad} files failed to load; rows zero-filled.")

    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    return X, y, n_pos, n_neg, pos_missing, neg_missing

# ── Main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Train a single LR model on all data using winners file ranking.")
    ap.add_argument("--positives", required=True, help="Folder with positive .pt tensors")
    ap.add_argument("--negatives", required=True, help="Folder with negative .pt tensors")
    ap.add_argument("--winners_csv", required=True, help="Global winners CSV (feature, winning_F1_score, …)")
    ap.add_argument("--num_features", type=int, required=True, help="Top-N features to use (ranked by winning_F1_score desc)")
    ap.add_argument("--out_model", required=True, help="Output model path (.joblib); JSON metadata written alongside")
    ap.add_argument("--permitted_pos", default=None, help="(Optional) FASTA of permitted POS IDs")
    ap.add_argument("--permitted_neg", default=None, help="(Optional) FASTA of permitted NEG IDs")
    ap.add_argument("--seed", type=int, default=9687254, help="Random seed (default 9687254)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)

    # Load global winners CSV and derive ranking (desc by winning_F1_score)
    winners = pd.read_csv(args.winners_csv)
    if "feature" not in winners.columns or "winning_F1_score" not in winners.columns:
        raise ValueError("winners_csv must contain columns: 'feature' and 'winning_F1_score'")
    # Be robust to threshold column typos
    if "winning_threshold" not in winners.columns and "winning_theshold" in winners.columns:
        winners = winners.rename(columns={"winning_theshold": "winning_threshold"})

    winners["feature"] = pd.to_numeric(winners["feature"], errors="coerce")
    winners["winning_F1_score"] = pd.to_numeric(winners["winning_F1_score"], errors="coerce")
    winners = winners.dropna(subset=["feature", "winning_F1_score"]).copy()

    winners.sort_values("winning_F1_score", ascending=False, inplace=True, ignore_index=True)
    if len(winners) < args.num_features:
        raise ValueError(f"winners_csv has only {len(winners)} rows; cannot select {args.num_features} features.")

    top_feats = winners["feature"].astype(int).tolist()[:int(args.num_features)]
    log(f"[rank] using top {len(top_feats)} features from winners file: {os.path.basename(args.winners_csv)}")

    # Optional permitted sets
    permitted_pos = fasta_ids(args.permitted_pos) if args.permitted_pos else None
    permitted_neg = fasta_ids(args.permitted_neg) if args.permitted_neg else None
    if permitted_pos is not None or permitted_neg is not None:
        log("[permit] restricting to permitted IDs (and reporting any missing embeddings)")

    # Build training matrix
    log("[data] building training matrix (max pooling over residues) …")
    X, y, n_pos, n_neg, pos_miss, neg_miss = build_dataset(
        args.positives, args.negatives, top_feats,
        permitted_pos=permitted_pos, permitted_neg=permitted_neg
    )
    log(f"[data] X={X.shape}  positives={n_pos:,}  negatives={n_neg:,}")
    if pos_miss or neg_miss:
        log(f"[data] permitted-but-missing embeddings → POS={pos_miss:,}  NEG={neg_miss:,}")

    # Train final model on ALL data
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", n_jobs=None, random_state=args.seed))
    ])
    t0 = time.time()
    pipe.fit(X, y)
    dt = human_time(time.time() - t0)
    train_acc = accuracy_score(y, pipe.predict(X)) * 100.0
    log(f"[train] finished in {dt}  |  training accuracy: {train_acc:.2f}%")

    # Save model + metadata
    joblib_dump(pipe, args.out_model)
    meta = {
        "created_at": time.ctime(),
        "model_path": os.path.abspath(args.out_model),
        "winners_csv": os.path.abspath(args.winners_csv),
        "num_features": int(args.num_features),
        "feature_ids": top_feats,
        "positives_dir": os.path.abspath(args.positives),
        "negatives_dir": os.path.abspath(args.negatives),
        "permitted_pos": os.path.abspath(args.permitted_pos) if args.permitted_pos else None,
        "permitted_neg": os.path.abspath(args.permitted_neg) if args.permitted_neg else None,
        "seed": args.seed,
        "train_shape": list(X.shape),
        "train_counts": {"positives": n_pos, "negatives": n_neg},
        "train_accuracy_percent": round(float(train_acc), 4)
    }
    with open(args.out_model + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"[✓] saved model → {args.out_model}")
    log(f"[✓] saved metadata → {args.out_model}.json")

if __name__ == "__main__":
    main()
