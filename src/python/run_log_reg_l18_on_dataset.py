##!/usr/bin/env python3
import os
import csv
import json
import time
import argparse
from glob import glob

import numpy as np
import pandas as pd
import torch
from joblib import load as joblib_load

# -------- helpers --------
def log(msg): print(msg, flush=True)

def list_pt(folder):
    return sorted(
        p for p in glob(os.path.join(folder, "*.pt"))
        if not os.path.basename(p).startswith("._")
    )

def entry_from_filename(path):
    base = os.path.basename(path)
    if base.endswith("_original_SAE.pt"):
        base = base[: -len("_original_SAE.pt")]
    else:
        base = os.path.splitext(base)[0]
    return base

def human_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def load_top_features_from_model_json(model_path):
    meta_path = model_path + ".json"
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        # Prefer explicit feature_ids if present
        if "feature_ids" in meta and isinstance(meta["feature_ids"], list) and len(meta["feature_ids"]) > 0:
            feats = [int(x) for x in meta["feature_ids"]]
            return feats
        # Else, if num_features present but no list, we can't reconstruct order → return None
        return None
    except Exception:
        return None

def load_top_features_from_ranking(ranking_csv, top_n):
    """
    Supports:
      - ranking CSV with columns: feature, rank  (ascending rank)
      - winners CSV with columns: feature, winning_F1_score (descending by score)
        (optionally with winning_threshold or winning_theshold)
    """
    df = pd.read_csv(ranking_csv)

    if {"feature", "rank"} <= set(df.columns):
        df["feature"] = pd.to_numeric(df["feature"], errors="coerce")
        df["rank"]    = pd.to_numeric(df["rank"], errors="coerce")
        df = df.dropna(subset=["feature", "rank"]).sort_values("rank", ascending=True)
        feats = df["feature"].astype(int).tolist()
        if len(feats) < top_n:
            raise ValueError(f"Ranking has only {len(feats)} rows; need top_n={top_n}.")
        return feats[:top_n]

    if {"feature", "winning_F1_score"} <= set(df.columns):
        # Be tolerant to threshold column typo; we don't need it here anyway.
        df["feature"] = pd.to_numeric(df["feature"], errors="coerce")
        df["winning_F1_score"] = pd.to_numeric(df["winning_F1_score"], errors="coerce")
        df = df.dropna(subset=["feature", "winning_F1_score"]).sort_values("winning_F1_score", ascending=False)
        feats = df["feature"].astype(int).tolist()
        if len(feats) < top_n:
            raise ValueError(f"Winners file has only {len(feats)} rows; need top_n={top_n}.")
        return feats[:top_n]

    raise ValueError(
        "Ranking file must contain either (feature,rank) or (feature,winning_F1_score)."
    )

def safe_load_tensor(path):
    if os.path.getsize(path) < 64:
        raise EOFError("file too small")
    t = torch.load(path, map_location="cpu")
    if t.ndim != 2:
        raise ValueError(f"bad shape {tuple(t.shape)}")
    return t

# -------- main --------
def main():
    ap = argparse.ArgumentParser(description="Apply trained LR model to an embeddings folder and write per-sequence probabilities.")
    ap.add_argument("--model",   required=True, help="Path to trained joblib model (e.g., logreg_allfolds_top900.joblib)")
    ap.add_argument("--ranking", required=True, help="Ranking file: either (feature,rank) or winners (feature,winning_F1_score)")
    ap.add_argument("--embeds",  required=True, help="Folder with .pt tensors (L,F) to score")
    ap.add_argument("--out",     required=True, help="Output CSV path")
    ap.add_argument("--top_n",   type=int, default=None,
                    help="Number of features to use. If omitted, will try to read feature_ids from model JSON.")
    ap.add_argument("--log_every", type=int, default=500, help="Progress print frequency (default 500)")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) model (Pipeline with scaler + LR)
    log(f"[load] model: {args.model}")
    pipe = joblib_load(args.model)

    # 2) feature set
    feats = load_top_features_from_model_json(args.model)
    if feats is not None:
        # Use exact features from training metadata
        log(f"[features] using {len(feats)} features from model JSON (authoritative)")
    else:
        if args.top_n is None:
            raise SystemExit(
                "Model JSON lacks 'feature_ids' and --top_n not provided. "
                "Please pass --top_n to match the model (e.g., 900)."
            )
        log(f"[load] ranking/winners: {args.ranking}")
        feats = load_top_features_from_ranking(args.ranking, args.top_n)
        log(f"[features] using top-{args.top_n} features from ranking/winners (first 5): {feats[:5]} …")

    # 3) tensors
    tensor_files = list_pt(args.embeds)
    log(f"[scan] tensors: {len(tensor_files):,} in {args.embeds}")

    written = 0
    skipped = 0
    start = time.time()

    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Entry", "probability"])

        for i, path in enumerate(tensor_files, 1):
            entry = entry_from_filename(path)
            try:
                t = safe_load_tensor(path)                   # (L, F)
                v = t.max(dim=0).values.numpy().astype(np.float32)  # (F,)
                x = v[feats].reshape(1, -1)
                proba = pipe.predict_proba(x)[0, 1]
                writer.writerow([entry, f"{float(proba):.6f}"])
                written += 1
            except Exception as e:
                skipped += 1
                log(f"  ⚠️  skip {os.path.basename(path)} ({e})")

            if (i % args.log_every) == 0 or i == len(tensor_files):
                el = time.time() - start
                rate = i / el if el > 0 else 0.0
                eta = (len(tensor_files) - i) / rate if rate > 0 else 0.0
                log(f"  [prog] {i:>6}/{len(tensor_files):<6} | wrote={written:,} | "
                    f"elapsed {human_time(el)} | ETA {human_time(eta)}")

    log(f"[done] CSV → {args.out}")
    log(f"[stats] wrote={written:,}, skipped={skipped:,}")
    log(f"[time] total {human_time(time.time() - t0)}")

if __name__ == "__main__":
    main()
