#!/usr/bin/env python3
import os
import csv
import json
import time
import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from joblib import load as joblib_load
from Bio import SeqIO

from interplm.sae.inference import load_sae_from_hf
from interplm.esm.embed import embed_single_sequence


# ---------- helpers ----------
def log(msg: str):
    print(msg, flush=True)

def human_time(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def load_top_features_from_model_json(model_path: str):
    meta_path = model_path + ".json"
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        if "feature_ids" in meta and isinstance(meta["feature_ids"], list) and len(meta["feature_ids"]) > 0:
            return [int(x) for x in meta["feature_ids"]]
        return None
    except Exception:
        return None

def load_top_features_from_ranking(ranking_csv: str, top_n: int):
    import pandas as pd
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
        df["feature"] = pd.to_numeric(df["feature"], errors="coerce")
        df["winning_F1_score"] = pd.to_numeric(df["winning_F1_score"], errors="coerce")
        df = df.dropna(subset=["feature", "winning_F1_score"]).sort_values("winning_F1_score", ascending=False)
        feats = df["feature"].astype(int).tolist()
        if len(feats) < top_n:
            raise ValueError(f"Winners file has only {len(feats)} rows; need top_n={top_n}.")
        return feats[:top_n]

    raise ValueError("Ranking file must contain either (feature,rank) or (feature,winning_F1_score).")

def max_pool_features(features_2d: torch.Tensor) -> np.ndarray:
    """
    features_2d: (L, F) SAE features
    returns: (F,) numpy float32
    """
    v = features_2d.max(dim=0).values
    return v.detach().cpu().numpy().astype(np.float32)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Stream FASTA → ESM embedding → SAE → LR prediction (no saving embeddings)."
    )
    ap.add_argument("--fasta", required=True, help="Input FASTA")
    ap.add_argument("--out", required=True, help="Output CSV (Entry, probability)")
    ap.add_argument("--model", required=True, help="Joblib model (Pipeline with scaler + LR)")
    ap.add_argument("--ranking", required=True, help="Ranking/winners CSV used to choose feature indices")
    ap.add_argument("--top_n", type=int, default=None,
                    help="If model JSON lacks feature_ids, use top_n features from ranking (e.g. 320).")
    ap.add_argument("--plm_layer", type=int, default=18, help="ESM/SAE layer (default 18)")
    ap.add_argument("--plm_model", default="esm2-650m", help="SAE HF model name (default esm2-650m)")
    ap.add_argument("--esm_model_name", default="esm2_t33_650M_UR50D",
                    help="ESM model string for embed_single_sequence")
    ap.add_argument("--log_every", type=int, default=200, help="Progress print frequency")
    args = ap.parse_args()

    fasta_path = Path(args.fasta)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not fasta_path.exists():
        raise SystemExit(f"❌ FASTA not found: {fasta_path}")
    if not os.path.exists(args.model):
        raise SystemExit(f"❌ Model not found: {args.model}")
    if not os.path.exists(args.ranking):
        raise SystemExit(f"❌ Ranking not found: {args.ranking}")

    t0 = time.time()

    # 1) Load model
    log(f"[load] model: {args.model}")
    pipe = joblib_load(args.model)

    # 2) Load feature indices
    feats = load_top_features_from_model_json(args.model)
    if feats is not None:
        log(f"[features] using {len(feats)} features from model JSON (authoritative)")
    else:
        if args.top_n is None:
            raise SystemExit(
                "Model JSON lacks 'feature_ids' and --top_n not provided. "
                "Pass --top_n to match the model (e.g., 320)."
            )
        log(f"[load] ranking/winners: {args.ranking}")
        feats = load_top_features_from_ranking(args.ranking, args.top_n)
        log(f"[features] using top-{args.top_n} from ranking (first 5): {feats[:5]} …")

    feats = np.array(feats, dtype=np.int64)

    # 3) Load SAE once
    log(f"[load] SAE: plm_model={args.plm_model}, plm_layer={args.plm_layer}")
    sae = load_sae_from_hf(plm_model=args.plm_model, plm_layer=args.plm_layer)

    written = 0
    skipped = 0
    start = time.time()

    # CSV output
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Entry", "probability"])

        # Stream through FASTA
        for i, record in enumerate(SeqIO.parse(str(fasta_path), "fasta"), start=1):
            entry = record.id
            seq = str(record.seq)

            try:
                with torch.no_grad():
                    # ESM embedding (returns something compatible with sae.encode)
                    embedding = embed_single_sequence(
                        sequence=seq,
                        model_name=args.esm_model_name,
                        layer=args.plm_layer
                    )

                    # SAE encode -> (L, F)
                    features = sae.encode(embedding)

                    # max pool -> (F,)
                    v = max_pool_features(features)

                x = v[feats].reshape(1, -1)
                proba = float(pipe.predict_proba(x)[0, 1])
                writer.writerow([entry, f"{proba:.6f}"])
                written += 1

            except Exception as e:
                skipped += 1
                log(f"  ⚠️ skip {entry} ({e})")

            # aggressively free memory each iteration
            try:
                del embedding
            except Exception:
                pass
            try:
                del features
            except Exception:
                pass
            try:
                del v
            except Exception:
                pass
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (i % args.log_every) == 0:
                el = time.time() - start
                rate = i / el if el > 0 else 0.0
                log(f"  [prog] {i:,} seqs | wrote={written:,} skipped={skipped:,} | elapsed {human_time(el)} | rate {rate:.2f}/s")

    log(f"[done] CSV → {out_path}")
    log(f"[stats] wrote={written:,}, skipped={skipped:,}")
    log(f"[time] total {human_time(time.time() - t0)}")


if __name__ == "__main__":
    main()